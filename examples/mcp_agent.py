#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MCP Agent Loop Client for vllm-mlx.

Connects to a vllm-mlx server with MCP tools enabled and runs a full
agent loop: the LLM can call tools multiple rounds until it produces
a final text response.

Usage:
    # Interactive chat mode
    python examples/mcp_agent.py

    # One-shot query
    python examples/mcp_agent.py "오늘 서울 날씨 검색해줘"

    # Custom server URL
    python examples/mcp_agent.py --port 8001 "Search for latest AI news"

    # With streaming output
    python examples/mcp_agent.py --stream "Python 3.13 new features"

Server setup:
    vllm-mlx serve $(python -c 'from vllm_mlx.config.models import DEFAULT_MODEL_ID; print(DEFAULT_MODEL_ID)') \
      --mllm \
      --enable-auto-tool-choice \
      --tool-call-parser qwen \
      --reasoning-parser qwen3 \
      --mcp-config mcp.json \
      --port 8001
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import httpx

from vllm_mlx.config.models import (
    DEFAULT_MODEL_ID,  # noqa: F401 — exposed for user scripts
)

DEFAULT_PORT = 8001
MAX_TOOL_ROUNDS = 10
REQUEST_TIMEOUT = 120
MAX_HISTORY_MESSAGES = 6
MAX_TOOL_RESULT_CHARS = 1200
MAX_ASSISTANT_CHARS = 1600

TOOL_NEEDED_RE = re.compile(
    r"(latest|current|today|news|weather|search|web|internet|url|link|browse|find|look up|"
    r"최신|현재|오늘|뉴스|날씨|검색|웹|인터넷|링크|찾아)",
    re.IGNORECASE,
)
SIMPLE_CHAT_RE = re.compile(
    r"^(hi|hello|hey|안녕|안녕하세요|고마워|감사|thanks|thank you)[!. ]*$",
    re.IGNORECASE,
)
CODING_RE = re.compile(
    r"(python|javascript|typescript|java|rust|go|c\+\+|code|함수|구현|코드)",
    re.IGNORECASE,
)


class MCPAgent:
    """Agent that orchestrates LLM + MCP tool calling loop."""

    def __init__(self, base_url: str, max_tokens: int = 1024, stream: bool = False):
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.stream = stream
        self.tools = []
        self.tool_summaries = []
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT)

    def connect(self) -> bool:
        """Fetch MCP tools from server. Returns True if tools are available."""
        try:
            resp = self.client.get(f"{self.base_url}/v1/mcp/tools")
            resp.raise_for_status()
        except httpx.ConnectError:
            print(f"[ERROR] Cannot connect to {self.base_url}")
            print("  Start the server with --mcp-config mcp.json")
            return False
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] MCP tools endpoint failed: {e.response.status_code}")
            return False

        data = resp.json()
        raw_tools = data.get("tools", [])

        if not raw_tools:
            print("[ERROR] No MCP tools available.")
            print("  Check that --mcp-config points to a valid mcp.json")
            return False

        # Convert to OpenAI tool format
        self.tools = []
        for t in raw_tools:
            self.tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t.get("parameters", {}),
                },
            })
            self.tool_summaries.append(
                f"- {t['name']}: {t['description'][:80]}"
            )

        print(f"[MCP] Connected - {len(self.tools)} tool(s) available:")
        for t in self.tools:
            print(f"  - {t['function']['name']}: {t['function']['description'][:80]}")
        print()
        return True

    def _build_system_prompt(self, include_tools: bool) -> str:
        if not include_tools:
            return (
                "You are a concise, helpful assistant. "
                "Answer directly. Keep responses focused on the user's request."
            )

        tools_desc = "\n".join(self.tool_summaries)
        return (
            "You are a helpful assistant with access to external tools.\n"
            "Use tools only when the user asks for real-time, external, or web-based "
            "information, or when a tool is necessary to complete the task.\n"
            "If the task can be answered directly from the conversation, do not call tools.\n\n"
            f"Available tools:\n{tools_desc}"
        )

    def _last_user_text(self, messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                return content if isinstance(content, str) else ""
        return ""

    def _should_include_tools(self, messages: list[dict[str, Any]]) -> bool:
        last_user = self._last_user_text(messages).strip()
        if not last_user:
            return False
        if SIMPLE_CHAT_RE.match(last_user):
            return False
        if TOOL_NEEDED_RE.search(last_user):
            return True
        return any(msg.get("role") == "tool" for msg in messages[-4:])

    def _estimate_max_tokens(
        self, last_user: str, include_tools: bool, message_count: int
    ) -> int:
        if include_tools:
            return min(self.max_tokens, 1024)
        if SIMPLE_CHAT_RE.match(last_user) or len(last_user) < 32:
            return min(self.max_tokens, 192)
        if CODING_RE.search(last_user):
            return min(self.max_tokens, 768)
        if message_count <= 2:
            return min(self.max_tokens, 384)
        return min(self.max_tokens, 512)

    def _compact_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        compact = dict(msg)
        role = compact.get("role")
        content = compact.get("content")

        if isinstance(content, str):
            if role == "tool" and len(content) > MAX_TOOL_RESULT_CHARS:
                compact["content"] = (
                    content[:MAX_TOOL_RESULT_CHARS]
                    + "\n...[truncated tool result]"
                )
            elif role == "assistant" and len(content) > MAX_ASSISTANT_CHARS:
                compact["content"] = (
                    content[:MAX_ASSISTANT_CHARS]
                    + "\n...[truncated assistant content]"
                )

        return compact

    def _build_payload_messages(
        self, messages: list[dict[str, Any]], include_tools: bool
    ) -> list[dict[str, Any]]:
        system_message = {
            "role": "system",
            "content": self._build_system_prompt(include_tools),
        }
        non_system = [m for m in messages if m.get("role") != "system"]
        recent = non_system[-MAX_HISTORY_MESSAGES:]
        compact_recent = [self._compact_message(m) for m in recent]
        return [system_message, *compact_recent]

    def _parse_xml_tool_calls(self, content: str) -> list | None:
        """Parse XML-style tool calls from content as fallback.

        Some models (e.g. Qwen with reasoning mode) may emit tool calls
        as XML tags in content instead of structured tool_calls.

        Supported formats:
        1. Direct XML: <tool_name><param>value</param></tool_name>
        2. Qwen3 style: <tool_call>function=tool_name<parameter=key>value</parameter></tool_call>
        """
        if not content:
            return None

        tool_calls = []

        # Format 1: Qwen3 <tool_call> block
        # <tool_call>\nfunction=web-search__tavily_search\n<parameter=query>\nvalue\n</parameter>\n</tool_call>
        tc_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        for i, block in enumerate(tc_blocks):
            # Extract function name
            func_match = re.search(r"function\s*=\s*(\S+)", block)
            if not func_match:
                continue
            func_name = func_match.group(1).strip()

            # Extract parameters: <parameter=key>value</parameter>
            args = {}
            for param_match in re.finditer(
                r"<parameter=(\w+)>(.*?)</parameter>", block, re.DOTALL
            ):
                args[param_match.group(1)] = param_match.group(2).strip()

            tool_calls.append({
                "id": f"xml_call_{i}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })

        if tool_calls:
            return tool_calls

        # Format 2: Direct XML <tool_name><param>value</param></tool_name>
        known_tools = {t["function"]["name"] for t in self.tools}
        pattern = re.compile(
            r"<(" + "|".join(re.escape(t) for t in known_tools) + r")>(.*?)</\1>",
            re.DOTALL,
        )
        for i, m in enumerate(pattern.finditer(content)):
            tool_name, inner_xml = m.group(1), m.group(2)
            args = {}
            for param_match in re.finditer(r"<(\w+)>(.*?)</\1>", inner_xml, re.DOTALL):
                args[param_match.group(1)] = param_match.group(2).strip()

            tool_calls.append({
                "id": f"xml_call_{i}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })

        return tool_calls if tool_calls else None

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a single MCP tool and return result as string."""
        try:
            resp = self.client.post(
                f"{self.base_url}/v1/mcp/execute",
                json={"tool_name": tool_name, "arguments": arguments},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            return f"[Tool Error] HTTP {e.response.status_code}: {e.response.text[:200]}"
        except httpx.TimeoutException:
            return "[Tool Error] Execution timed out"

        result = resp.json()

        if result.get("is_error"):
            err = result.get("error_message") or result.get("content") or "Unknown error"
            return f"[Tool Error] {err}"

        content = result.get("content")
        if content is None:
            return "[Tool returned empty result]"
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=False, indent=2)
        return str(content)

    def _chat_completion(self, messages: list) -> dict:
        """Send chat completion request with tools."""
        include_tools = self._should_include_tools(messages)
        payload_messages = self._build_payload_messages(messages, include_tools)
        last_user = self._last_user_text(payload_messages)
        payload = {
            "model": "default",
            "messages": payload_messages,
            "max_tokens": self._estimate_max_tokens(
                last_user, include_tools, len(payload_messages)
            ),
        }
        if include_tools:
            payload["tools"] = self.tools

        if self.stream:
            return self._chat_completion_stream(payload)

        resp = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def _chat_completion_stream(self, payload: dict) -> dict:
        """Handle streaming chat completion, printing tokens as they arrive."""
        payload["stream"] = True

        collected_content = ""
        collected_reasoning = ""
        tool_calls_map = {}  # index -> {id, function: {name, arguments}}
        finish_reason = None

        with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                fr = chunk.get("choices", [{}])[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                # Reasoning tokens
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    if not collected_reasoning:
                        print("[Thinking] ", end="", flush=True)
                    print(reasoning, end="", flush=True)
                    collected_reasoning += reasoning

                # Content tokens
                content = delta.get("content")
                if content:
                    if collected_reasoning and not collected_content:
                        print("\n\n", end="")  # separator after reasoning
                    print(content, end="", flush=True)
                    collected_content += content

                # Tool calls (accumulated across chunks)
                for tc_delta in delta.get("tool_calls", []):
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_delta.get("id", f"call_{idx}"),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_map[idx]
                    if tc_delta.get("id"):
                        tc["id"] = tc_delta["id"]
                    fn = tc_delta.get("function", {})
                    if fn.get("name"):
                        tc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tc["function"]["arguments"] += fn["arguments"]

        if collected_content or collected_reasoning:
            print()  # newline after streaming

        # Build a synthetic response matching non-streaming format
        tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)] if tool_calls_map else None

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": collected_content or None,
                    "reasoning": collected_reasoning or None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason or "stop",
            }]
        }

    def run(self, user_message: str, messages: list | None = None) -> str:
        """
        Run the agent loop for a single user query.

        Returns the final text response from the LLM.
        """
        if messages is None:
            messages = [{
                "role": "system",
                "content": self._build_system_prompt(include_tools=False),
            }]

        messages.append({"role": "user", "content": user_message})

        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
            try:
                result = self._chat_completion(messages)
            except httpx.HTTPStatusError as e:
                return f"[Error] HTTP {e.response.status_code}: {e.response.text[:300]}"
            except httpx.TimeoutException:
                return "[Error] Request timed out"

            choice = result.get("choices", [{}])[0]
            assistant_msg = choice.get("message", {})
            tool_calls = assistant_msg.get("tool_calls")

            if not tool_calls:
                # Fallback: check if content contains XML-style tool calls
                content = assistant_msg.get("content", "")
                tool_calls = self._parse_xml_tool_calls(content)

                if not tool_calls:
                    # Truly no tool calls -> final response
                    reasoning = assistant_msg.get("reasoning") or assistant_msg.get("reasoning_content")
                    messages.append({"role": "assistant", "content": content})

                    if not self.stream:
                        if reasoning:
                            print(f"[Thinking] {reasoning}\n")
                        if content:
                            print(content)

                    return content

                # XML tool calls found - strip them from content for display
                if not self.stream:
                    reasoning = assistant_msg.get("reasoning") or assistant_msg.get("reasoning_content")
                    if reasoning:
                        print(f"[Thinking] {reasoning}\n")

            # Tool calls detected -> execute them
            if not self.stream:
                print(f"\n[Round {round_num}] LLM requested {len(tool_calls)} tool call(s)")

            # Add assistant message with tool_calls to history
            messages.append({
                "role": "assistant",
                "content": assistant_msg.get("content"),
                "tool_calls": tool_calls,
            })

            # Execute each tool call
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                print(f"  -> {func_name}({json.dumps(func_args, ensure_ascii=False)[:120]})")

                tool_result = self._execute_tool(func_name, func_args)
                print(f"     Result: {tool_result[:200]}{'...' if len(tool_result) > 200 else ''}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

        return "[Error] Max tool rounds exceeded"

    def chat_loop(self):
        """Interactive chat loop with conversation history."""
        print("=" * 60)
        print("MCP Agent Chat")
        print("=" * 60)
        print("Type 'exit' to quit, 'clear' to reset conversation\n")

        messages = [{
            "role": "system",
            "content": self._build_system_prompt(include_tools=False),
        }]

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                messages = messages[:1]  # keep system prompt
                print("[Conversation cleared]")
                continue

            print()
            response = self.run(user_input, messages)

            if not response:
                print("[No response]")

    def close(self):
        """Close HTTP client."""
        self.client.close()


def main():
    parser = argparse.ArgumentParser(
        description="MCP Agent Loop Client for vllm-mlx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/mcp_agent.py                          # Interactive chat
  python examples/mcp_agent.py "Search for AI news"     # One-shot query
  python examples/mcp_agent.py --port 8001 --stream "Seoul weather"
        """,
    )
    parser.add_argument("query", nargs="?", help="One-shot query (omit for interactive mode)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per response (default: 1024)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    agent = MCPAgent(base_url, max_tokens=args.max_tokens, stream=args.stream)

    if not agent.connect():
        sys.exit(1)

    try:
        if args.query:
            # One-shot mode
            agent.run(args.query)
        else:
            # Interactive mode
            agent.chat_loop()
    finally:
        agent.close()


if __name__ == "__main__":
    main()
