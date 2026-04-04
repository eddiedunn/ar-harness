#!/usr/bin/env python3
"""
ar-harness: Claude Agent SDK harness for Andrej Karpathy's autoresearch protocol.

Reads program.md from --cwd as the system prompt, then runs the agent loop
with Read/Write/Edit/Bash tools until done or --max-turns is reached.
"""

import argparse
import signal
import sys
from pathlib import Path

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    query,
)
from claude_agent_sdk._errors import ProcessError


def parse_args() -> argparse.Namespace:
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
        default="claude-sonnet-4-6",
        help="Model to use (default: claude-sonnet-4-6)",
    )
    return parser.parse_args()


async def run(cwd: str, max_turns: int, model: str) -> None:
    cwd_path = Path(cwd).resolve()
    program_md = cwd_path / "program.md"

    if not cwd_path.is_dir():
        print(f"Error: --cwd '{cwd}' is not a directory", file=sys.stderr)
        sys.exit(1)

    if not program_md.exists():
        print(f"Error: program.md not found in '{cwd_path}'", file=sys.stderr)
        sys.exit(1)

    system_prompt = program_md.read_text(encoding="utf-8").strip()
    print(f"[harness] cwd: {cwd_path}", flush=True)
    print(f"[harness] model: {model}  max_turns: {max_turns}", flush=True)
    print(f"[harness] system prompt loaded ({len(system_prompt)} chars)", flush=True)
    print("[harness] starting agent loop...\n", flush=True)

    options = ClaudeAgentOptions(
        cwd=str(cwd_path),
        allowed_tools=["Read", "Write", "Edit", "Bash"],
        permission_mode="bypassPermissions",
        system_prompt=system_prompt,
        max_turns=max_turns,
        model=model,
    )

    initial_prompt = (
        "Begin the experiment. Follow the instructions in your system prompt."
    )

    got_result = False
    try:
        async for message in query(prompt=initial_prompt, options=options):
            if isinstance(message, SystemMessage):
                if message.subtype == "init":
                    session_id = message.data.get("session_id", "unknown")
                    print(f"[harness] session: {session_id}\n", flush=True)

            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, flush=True)

            elif isinstance(message, ResultMessage):
                got_result = True
                print(f"\n[harness] done — stop_reason: {message.stop_reason}", flush=True)
                if message.result:
                    print(f"[harness] result:\n{message.result}", flush=True)

    except Exception as exc:
        # When Claude exits with code 1 after hitting --max-turns (error_max_turns),
        # a ResultMessage was already yielded — treat exit code 1 as a normal
        # termination in that case rather than a fatal harness error.
        # The SDK surfaces this as a plain Exception with the message
        # "Command failed with exit code 1" re-raised from receive_messages().
        msg = str(exc)
        is_max_turns_exit = "exit code: 1" in msg or (
            isinstance(exc, ProcessError) and exc.exit_code == 1
        )
        if got_result and is_max_turns_exit:
            print(
                "[harness] subprocess exited with code 1 after result (normal for max_turns)",
                flush=True,
            )
        else:
            raise


def main() -> None:
    args = parse_args()

    def _sigint_handler(sig, frame):
        print("\n[harness] interrupted — exiting", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    anyio.run(run, args.cwd, args.max_turns, args.model)


if __name__ == "__main__":
    main()
