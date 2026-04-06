#!/usr/bin/env python3
"""
ar-harness: Autoresearch agent loop via OpenRouter.

Reads program.md from --cwd as the system prompt, then runs a tool-using
agent loop with Read/Write/Edit/Bash tools until max turns or interruption.

Required env var: OPENROUTER_API_KEY
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from openai import OpenAI

# ---------------------------------------------------------------------------
# Model fallback chain
# Free models are tried first (in order). When all free models are
# rate-limited, the harness switches to paid models and logs a warning.
# ---------------------------------------------------------------------------

FREE_MODELS = [
    "qwen/qwen3-coder:free",        # coding-specific, 262k ctx
    "qwen/qwen3.6-plus:free",       # 1M ctx, current default
    "openai/gpt-oss-120b:free",     # 120B OSS, 131k ctx
    "meta-llama/llama-3.3-70b-instruct:free",  # proven tool use, 65k ctx
]

PAID_MODELS = [
    "mistralai/devstral-small",         # $0.10/Mtok, 131k, Mistral coding model
    "deepseek/deepseek-chat-v3-0324",   # $0.20/Mtok, 163k, strong coder
    "qwen/qwen3-coder",                 # $0.22/Mtok, 262k, paid coding tier
    "google/gemini-2.5-flash",          # $0.30/Mtok, 1M ctx, strong safety net
]


def build_model_chain(primary: str) -> list[str]:
    """Return ordered fallback list starting with primary, then remaining
    free models, then paid models. Deduplicates if primary is already in a list."""
    seen = {primary}
    chain = [primary]
    for m in FREE_MODELS:
        if m not in seen:
            seen.add(m)
            chain.append(m)
    for m in PAID_MODELS:
        if m not in seen:
            seen.add(m)
            chain.append(m)
    return chain


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read the full contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute or relative (to cwd) path to read.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file, creating parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Destination path."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": (
                "Replace the first occurrence of old_string with new_string in a file. "
                "old_string must match exactly (including whitespace and indentation)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string", "description": "Exact text to find."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": (
                "Run a bash command in the experiment directory. "
                "Use for git, uv, grep, tail, etc. "
                "Stdout and stderr are both returned. "
                "Long-running commands (e.g. uv run train.py) may take several minutes — "
                "use a generous timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to run."},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 600).",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def append_jsonl(path: Path, obj: dict) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")
    except Exception:
        pass  # never let logging crash the harness


def summarize_text(text: str, max_len: int = 300) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"… [{len(text)} chars]"


def summarize_args(args: dict) -> dict:
    out = {}
    for k, v in args.items():
        if k == "content":
            out[k] = summarize_text(str(v), 200)
        elif k in ("old_string", "new_string"):
            out[k] = summarize_text(str(v), 100)
        else:
            s = str(v)
            out[k] = s if len(s) <= 200 else s[:200] + "…"
    return out


def log_event(jsonl_path: Path, audit_path: Path, event: dict) -> None:
    ts = iso_now()
    event = {"ts": ts, **event}
    append_jsonl(jsonl_path, event)

    # Plain-text audit line
    kind = event.get("event", "?")
    parts = [ts, kind]
    if "turn" in event:
        parts.append(f"turn={event['turn']}")
    if "model" in event:
        parts.append(f"model={event['model']}")
    if "finish_reason" in event:
        parts.append(f"finish_reason={event['finish_reason']}")
    if "duration_ms" in event:
        parts.append(f"duration={event['duration_ms']}ms")
    if "tool" in event:
        parts.append(f"tool={event['tool']}")
    if "error" in event:
        parts.append(f"error={summarize_text(str(event['error']), 120)}")
    if "msg" in event:
        parts.append(event["msg"])

    line = "  ".join(parts)
    try:
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(name: str, args: dict, cwd: Path) -> str:
    if name == "Read":
        path = Path(args["file_path"])
        if not path.is_absolute():
            path = cwd / path
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading {path}: {exc}"

    elif name == "Write":
        path = Path(args["file_path"])
        if not path.is_absolute():
            path = cwd / path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"], encoding="utf-8")
            return f"Written: {path}"
        except Exception as exc:
            return f"Error writing {path}: {exc}"

    elif name == "Edit":
        path = Path(args["file_path"])
        if not path.is_absolute():
            path = cwd / path
        try:
            text = path.read_text(encoding="utf-8")
            old = args["old_string"]
            if old not in text:
                return f"Error: old_string not found in {path}"
            path.write_text(text.replace(old, args["new_string"], 1), encoding="utf-8")
            return f"Edited: {path}"
        except Exception as exc:
            return f"Error editing {path}: {exc}"

    elif name == "Bash":
        timeout = int(args.get("timeout", 600))
        try:
            result = subprocess.run(
                args["command"],
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(cwd),
                timeout=timeout,
            )
            out = result.stdout
            if result.stderr:
                out += ("\n" if out else "") + "[stderr]\n" + result.stderr
            if result.returncode != 0:
                out += f"\n[exit code: {result.returncode}]"
            return out or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        except Exception as exc:
            return f"Error running command: {exc}"

    else:
        return f"Error: unknown tool '{name}'"


def _assistant_msg(msg) -> dict:
    """Convert an OpenAI ChatCompletionMessage to a plain dict for the messages list."""
    d: dict = {"role": "assistant"}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cwd: str, max_turns: int, model: str, event_log: str | None, request_timeout: int) -> None:
    cwd_path = Path(cwd).resolve()
    program_md = cwd_path / "program.md"

    if not cwd_path.is_dir():
        print(f"Error: --cwd '{cwd}' is not a directory", file=sys.stderr)
        sys.exit(1)
    if not program_md.exists():
        print(f"Error: program.md not found in '{cwd_path}'", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    jsonl_path = Path(event_log) if event_log else cwd_path / "harness.events.jsonl"
    audit_path = cwd_path / "harness.audit.log"

    system_prompt = program_md.read_text(encoding="utf-8").strip()
    model_chain = build_model_chain(model)
    print(f"[harness] cwd:             {cwd_path}", flush=True)
    print(f"[harness] model:           {model_chain[0]}", flush=True)
    print(f"[harness] fallback chain:  {' → '.join(model_chain[1:])}", flush=True)
    print(f"[harness] max_turns:       {max_turns}", flush=True)
    print(f"[harness] request_timeout: {request_timeout}s", flush=True)
    print(f"[harness] event_log:       {jsonl_path}", flush=True)
    print(f"[harness] system_prompt:   {len(system_prompt)} chars", flush=True)
    print("[harness] starting agent loop...\n", flush=True)

    model_idx = 0
    current_model = model_chain[0]

    log_event(jsonl_path, audit_path, {
        "event": "run_start",
        "model": current_model,
        "model_chain": model_chain,
        "cwd": str(cwd_path),
        "max_turns": max_turns,
        "request_timeout": request_timeout,
        "system_prompt_chars": len(system_prompt),
    })

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=request_timeout,
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Begin the experiment. Follow your system prompt exactly. "
                "Start by reading any files you need, then proceed with your first experiment iteration."
            ),
        },
    ]

    finish_reason = "unknown"
    turns_completed = 0

    try:
        turn = 1
        while turn <= max_turns:
            turns_completed = turn - 1

            log_event(jsonl_path, audit_path, {
                "event": "request_start",
                "turn": turn,
                "model": current_model,
                "message_count": len(messages),
            })

            t0 = time.monotonic()
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except openai.RateLimitError as exc:
                duration_ms = int((time.monotonic() - t0) * 1000)
                log_event(jsonl_path, audit_path, {
                    "event": "error",
                    "turn": turn,
                    "model": current_model,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                    "error_type": "RateLimitError",
                })
                model_idx += 1
                if model_idx >= len(model_chain):
                    print(f"[harness] all models exhausted — giving up", flush=True)
                    raise
                next_model = model_chain[model_idx]
                prev_was_free = current_model in FREE_MODELS or current_model.endswith(":free")
                next_is_paid = next_model in PAID_MODELS
                if prev_was_free and next_is_paid:
                    print(
                        f"[harness] WARNING: all free models rate-limited"
                        f" — switching to paid: {next_model}",
                        flush=True,
                    )
                else:
                    print(
                        f"[harness] {current_model} rate-limited"
                        f" — switching to: {next_model}",
                        flush=True,
                    )
                log_event(jsonl_path, audit_path, {
                    "event": "model_switch",
                    "turn": turn,
                    "from_model": current_model,
                    "to_model": next_model,
                    "to_paid": next_is_paid,
                })
                current_model = next_model
                continue  # retry this turn with new model
            except Exception as exc:
                duration_ms = int((time.monotonic() - t0) * 1000)
                log_event(jsonl_path, audit_path, {
                    "event": "error",
                    "turn": turn,
                    "model": current_model,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                })
                print(f"[harness] request error turn {turn}: {exc}", flush=True)
                raise

            duration_ms = int((time.monotonic() - t0) * 1000)
            choice = response.choices[0]
            msg = choice.message
            finish_reason = choice.finish_reason
            messages.append(_assistant_msg(msg))

            log_event(jsonl_path, audit_path, {
                "event": "request_end",
                "turn": turn,
                "model": current_model,
                "duration_ms": duration_ms,
                "finish_reason": finish_reason,
                "has_content": bool(msg.content),
                "tool_call_count": len(msg.tool_calls) if msg.tool_calls else 0,
            })

            if msg.content:
                print(msg.content, flush=True)

            if finish_reason == "tool_calls" and msg.tool_calls:
                tool_msgs = []
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    arg_summary = ", ".join(
                        f"{k}={repr(v)[:80]}" for k, v in args.items()
                    )
                    print(f"[tool:{turn}] {tc.function.name}({arg_summary})", flush=True)

                    log_event(jsonl_path, audit_path, {
                        "event": "tool_call",
                        "turn": turn,
                        "tool": tc.function.name,
                        "args": summarize_args(args),
                    })

                    result = execute_tool(tc.function.name, args, cwd_path)

                    log_event(jsonl_path, audit_path, {
                        "event": "tool_result",
                        "turn": turn,
                        "tool": tc.function.name,
                        "result_chars": len(result),
                        "result_preview": summarize_text(result, 200),
                        "is_error": result.startswith("Error"),
                    })

                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                messages.extend(tool_msgs)
                turn += 1
            else:
                turns_completed = turn
                finish_reason = finish_reason or "stop"
                print(f"\n[harness] done — finish_reason: {finish_reason}", flush=True)
                log_event(jsonl_path, audit_path, {
                    "event": "run_end",
                    "turns_completed": turns_completed,
                    "finish_reason": finish_reason,
                })
                return

        turns_completed = max_turns
        finish_reason = "max_turns"
        print(f"\n[harness] done — max_turns ({max_turns}) reached", flush=True)

    except KeyboardInterrupt:
        finish_reason = "interrupted"
        print("\n[harness] interrupted — exiting", flush=True)
    except Exception as exc:
        finish_reason = "error"
        print(f"\n[harness] fatal error: {exc}", flush=True)
        log_event(jsonl_path, audit_path, {
            "event": "error",
            "turn": turns_completed + 1,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "fatal": True,
        })
        raise
    finally:
        log_event(jsonl_path, audit_path, {
            "event": "run_end",
            "turns_completed": turns_completed,
            "finish_reason": finish_reason,
        })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the autoresearch agent loop in a given experiment directory."
    )
    parser.add_argument(
        "--cwd",
        required=True,
        help="Path to the autoresearch experiment directory (must contain program.md)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Maximum agent turns before stopping (default: 100)",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3.6-plus:free",
        help="OpenRouter model ID (default: qwen/qwen3.6-plus:free)",
    )
    parser.add_argument(
        "--event-log",
        default=None,
        help="Path for JSONL event log (default: <cwd>/harness.events.jsonl)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    def _sigint(sig, frame):
        print("\n[harness] interrupted — exiting", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)
    run(args.cwd, args.max_turns, args.model, args.event_log, args.request_timeout)


if __name__ == "__main__":
    main()
