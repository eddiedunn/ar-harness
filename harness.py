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
from pathlib import Path

from openai import OpenAI

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


def run(cwd: str, max_turns: int, model: str) -> None:
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

    system_prompt = program_md.read_text(encoding="utf-8").strip()
    print(f"[harness] cwd:        {cwd_path}", flush=True)
    print(f"[harness] model:      {model}", flush=True)
    print(f"[harness] max_turns:  {max_turns}", flush=True)
    print(f"[harness] system prompt: {len(system_prompt)} chars", flush=True)
    print("[harness] starting agent loop...\n", flush=True)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Begin the experiment. Follow the instructions in your system prompt.",
        },
    ]

    for turn in range(1, max_turns + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg = choice.message
        messages.append(_assistant_msg(msg))

        if msg.content:
            print(msg.content, flush=True)

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            tool_msgs = []
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                arg_summary = ", ".join(
                    f"{k}={repr(v)[:80]}" for k, v in args.items()
                )
                print(f"[tool:{turn}] {tc.function.name}({arg_summary})", flush=True)
                result = execute_tool(tc.function.name, args, cwd_path)
                tool_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
            messages.extend(tool_msgs)
        else:
            print(
                f"\n[harness] done — finish_reason: {choice.finish_reason}",
                flush=True,
            )
            return

    print(f"\n[harness] done — max_turns ({max_turns}) reached", flush=True)


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
    args = parser.parse_args()

    def _sigint(sig, frame):
        print("\n[harness] interrupted — exiting", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)
    run(args.cwd, args.max_turns, args.model)


if __name__ == "__main__":
    main()
