#!/usr/bin/env python3
"""Token usage analysis for OpenCode session exports."""

import json
import sys
from pathlib import Path


def analyze(session_path: str) -> dict:
    data = Path(session_path).read_text(encoding="utf-8", errors="replace")

    total_input = 0
    total_output = 0
    total_cache = 0
    tool_calls = 0

    # Try parsing as JSON
    try:
        session = json.loads(data)
        if isinstance(session, dict):
            usage = session.get("usage", {})
            total_input = usage.get("input_tokens", 0)
            total_output = usage.get("output_tokens", 0)
            total_cache = usage.get("cache_read_input_tokens", 0)
        elif isinstance(session, list):
            for entry in session:
                if isinstance(entry, dict):
                    usage = entry.get("usage", {})
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)
                    total_cache += usage.get("cache_read_input_tokens", 0)
                    if entry.get("type") == "tool_use":
                        tool_calls += 1
    except json.JSONDecodeError:
        # Try line-by-line JSONL
        for line in data.strip().split("\n"):
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    usage = entry.get("usage", {})
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)
                    total_cache += usage.get("cache_read_input_tokens", 0)
                    if entry.get("type") == "tool_use":
                        tool_calls += 1
            except json.JSONDecodeError:
                continue

    # Count tool calls from grep if not found in structured data
    if tool_calls == 0:
        tool_calls = data.count('"type":"tool_use"') + data.count('"type": "tool_use"')

    return {
        "file": session_path,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cache_read_tokens": total_cache,
        "total_tokens": total_input + total_output,
        "tool_calls": tool_calls,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <session.json> [--json]", file=sys.stderr)
        return 1

    session_path = sys.argv[1]
    use_json = "--json" in sys.argv

    if not Path(session_path).exists():
        print(f"[ERROR] File not found: {session_path}", file=sys.stderr)
        return 1

    result = analyze(session_path)

    if use_json:
        print(json.dumps(result, indent=2))
    else:
        print("=== Token Usage Analysis ===")
        print("")
        print(f"  File:         {result['file']}")
        print(f"  Input tokens: {result['input_tokens']:,}")
        print(f"  Output tokens:{result['output_tokens']:,}")
        print(f"  Cache tokens: {result['cache_read_tokens']:,}")
        print(f"  Total tokens: {result['total_tokens']:,}")
        print(f"  Tool calls:   {result['tool_calls']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
