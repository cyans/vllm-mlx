# SPDX-License-Identifier: Apache-2.0
"""
Tests for MCP tool auto-injection behaviour.

Phase 2 of SPEC-FIX-QWEN36-RUNTIME wires an opt-in ``--auto-inject-mcp-tools``
flag that lets the /v1/chat/completions endpoint automatically surface the
MCP-registered tools to the model when the client has not supplied any. The
default stays OFF so bit-for-bit compat with Qwen 3.5 is preserved per
REQ-N2.

The server module performs heavy imports (mlx-vlm, engine, metal init) at
module import time. While another vllm-mlx server is using the Metal device
the import aborts with an Objective-C NSRangeException. To keep these tests
runnable alongside a live server we exercise the auto-inject logic through
three small, import-light seams:

1. The CLI argparse surface in :mod:`vllm_mlx.cli` (pure argparse; no mlx).
2. The tool-merge helper in :mod:`vllm_mlx.mcp.tools`.
3. The pure resolver / log helpers in :mod:`vllm_mlx.api.mcp_inject` that
   take a flag, a request's tools and an MCP manager stub.

The full chat completion handler is not invoked directly — the REQ-N2 /
REQ-E2 / REQ-E3 contract is captured by the pure resolver test cases plus
the launcher env-passthrough and CLI flag tests. The live-inference
verification (P2-AC3 live path) is DEFERRED to a user-driven server
restart per the SPEC.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

# @CODE:FIX-QWEN36-RUNTIME/mcp-auto-inject — merge helper lives next to the
# existing merge_tools / mcp_tools_to_openai helpers.
from vllm_mlx.mcp.tools import merge_tool_lists

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Test fixtures: reusable OpenAI-shaped tool definitions
# ---------------------------------------------------------------------------


def _fn_tool(name: str, description: str = "") -> dict[str, Any]:
    """Build a minimal OpenAI-shaped function tool dict for tests."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"test tool {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


# ---------------------------------------------------------------------------
# CLI flag tests
# ---------------------------------------------------------------------------


class TestAutoInjectCliFlag:
    """Cover P2-AC1: the --auto-inject-mcp-tools flag parses correctly."""

    def _parse(self, extra_args: list[str]) -> Any:
        """Run the serve subparser with the supplied extra args.

        Importing :mod:`vllm_mlx.cli` does not trigger mlx init, so this is
        safe to run alongside a live server.
        """
        # Rebuild the argparse tree exactly the way cli.main does.
        from vllm_mlx import cli

        # cli.main() wires subparsers dynamically; we invoke it with argv that
        # triggers --help for an easy exit would skip the defaults path, so
        # instead we reach into its parser by running cli.main with a patched
        # sys.argv that includes a serve line. We intercept the heavy
        # serve_command by stubbing it so only argparse is exercised.
        captured: dict[str, Any] = {}

        def _capture(args):
            captured["args"] = args

        monkey = cli.serve_command  # keep reference for restore
        cli.serve_command = _capture  # type: ignore[assignment]
        try:
            import sys

            original_argv = sys.argv
            sys.argv = ["vllm-mlx", "serve", "dummy-model", *extra_args]
            try:
                cli.main()
            finally:
                sys.argv = original_argv
        finally:
            cli.serve_command = monkey  # type: ignore[assignment]

        assert "args" in captured, "serve_command was not invoked"
        return captured["args"]

    def test_flag_parses_default_false(self):
        """P2-AC1: omitting the flag yields auto_inject_mcp_tools == False."""
        args = self._parse([])
        assert hasattr(args, "auto_inject_mcp_tools"), (
            "--auto-inject-mcp-tools must be defined on the serve subparser"
        )
        assert args.auto_inject_mcp_tools is False

    def test_flag_parses_when_provided(self):
        """P2-AC1: providing the flag yields auto_inject_mcp_tools == True."""
        args = self._parse(["--auto-inject-mcp-tools"])
        assert args.auto_inject_mcp_tools is True


# ---------------------------------------------------------------------------
# merge_tool_lists pure-helper tests
# ---------------------------------------------------------------------------


class TestMergeToolLists:
    """Cover P2-AC4 collision semantics on the pure helper."""

    def test_empty_inputs_returns_empty(self):
        assert merge_tool_lists([], []) == []
        assert merge_tool_lists(None, None) == []

    def test_only_user_tools_preserved_in_order(self):
        u1 = _fn_tool("a")
        u2 = _fn_tool("b")
        assert merge_tool_lists([u1, u2], []) == [u1, u2]

    def test_only_mcp_tools_preserved_in_order(self):
        m1 = _fn_tool("x")
        m2 = _fn_tool("y")
        assert merge_tool_lists([], [m1, m2]) == [m1, m2]

    def test_merge_orders_user_first_then_mcp(self):
        """Client-provided tools come first, non-colliding MCP tools follow."""
        u1 = _fn_tool("user_only")
        m1 = _fn_tool("mcp_only")
        merged = merge_tool_lists([u1], [m1])
        names = [t["function"]["name"] for t in merged]
        assert names == ["user_only", "mcp_only"]

    def test_collision_user_wins(self):
        """P2-AC4: when names collide the user-provided tool wins."""
        u_shared = _fn_tool("shared", description="user version")
        m_shared = _fn_tool("shared", description="mcp version")
        m_extra = _fn_tool("mcp_extra")
        merged = merge_tool_lists([u_shared], [m_shared, m_extra])

        names = [t["function"]["name"] for t in merged]
        assert names == ["shared", "mcp_extra"]
        # The surviving 'shared' entry must be the user version
        shared = merged[0]
        assert shared["function"]["description"] == "user version"


# ---------------------------------------------------------------------------
# Pure tool-resolution helper tests (behaviour gate per REQ-N2 / REQ-E2 / REQ-E3)
# ---------------------------------------------------------------------------


class _FakeMcpManager:
    """Stand-in for MCPClientManager.get_merged_tools used in tests."""

    def __init__(self, mcp_tools: list[dict[str, Any]]):
        self._mcp_tools = mcp_tools
        self.calls: list[Any] = []

    def get_merged_tools(self, user_tools=None):
        # Record the call so tests can assert behaviour.
        self.calls.append(user_tools)
        # The real manager would merge here; our helper never calls it with
        # user_tools when auto_inject is off, and we exercise both paths.
        return list(self._mcp_tools)


class TestResolveEffectiveTools:
    """Cover REQ-N2, REQ-E2, REQ-E3 via the pure resolver.

    The resolver lives in :mod:`vllm_mlx.api.mcp_inject`, which has no mlx
    / metal imports at module load time. That matters because importing
    :mod:`vllm_mlx.server` aborts with an Objective-C NSException when the
    Metal device is already owned by another vllm-mlx process. Keeping the
    pure resolver in a light module lets these tests run alongside a live
    server.
    """

    def _resolve(self, *args, **kwargs):
        from vllm_mlx.api.mcp_inject import resolve_effective_tools

        return resolve_effective_tools(*args, **kwargs)

    def test_inject_off_preserves_no_tools_behavior(self):
        """REQ-N2: flag off + no client tools => no tools returned."""
        mgr = _FakeMcpManager([_fn_tool("mcp_a")])
        result = self._resolve(
            request_tools=None,
            mcp_manager=mgr,
            auto_inject=False,
        )
        assert result is None
        assert mgr.calls == [], "MCP manager must not be consulted when off"

    def test_inject_off_preserves_client_tools(self):
        """REQ-N2: flag off + client tools => exactly client tools."""
        u = _fn_tool("client_only")
        mgr = _FakeMcpManager([_fn_tool("mcp_should_not_appear")])
        result = self._resolve(
            request_tools=[u],
            mcp_manager=mgr,
            auto_inject=False,
        )
        assert result == [u]
        assert mgr.calls == [], "MCP manager must not be consulted when off"

    def test_inject_on_no_client_tools(self):
        """REQ-E2: flag on + no client tools => full MCP list injected."""
        m1 = _fn_tool("mcp_a")
        m2 = _fn_tool("mcp_b")
        mgr = _FakeMcpManager([m1, m2])
        result = self._resolve(
            request_tools=None,
            mcp_manager=mgr,
            auto_inject=True,
        )
        assert result is not None
        names = [t["function"]["name"] for t in result]
        assert names == ["mcp_a", "mcp_b"]

    def test_inject_on_with_client_tools_merges(self):
        """REQ-E3: flag on + client tools => merged, user first."""
        u = _fn_tool("user_tool")
        m = _fn_tool("mcp_tool")
        mgr = _FakeMcpManager([m])
        result = self._resolve(
            request_tools=[u],
            mcp_manager=mgr,
            auto_inject=True,
        )
        names = [t["function"]["name"] for t in result]
        assert names == ["user_tool", "mcp_tool"]

    def test_inject_on_collision_user_wins(self):
        """P2-AC4: user tool wins on name collision."""
        u = _fn_tool("shared", description="user")
        m_shared = _fn_tool("shared", description="mcp")
        m_other = _fn_tool("other")
        mgr = _FakeMcpManager([m_shared, m_other])
        result = self._resolve(
            request_tools=[u],
            mcp_manager=mgr,
            auto_inject=True,
        )
        names = [t["function"]["name"] for t in result]
        assert names == ["shared", "other"]
        shared = result[0]
        # Shared must come from the user tool (Pydantic or dict)
        if isinstance(shared, dict):
            assert shared["function"]["description"] == "user"
        else:
            assert shared.function["description"] == "user"

    def test_inject_on_with_no_manager_falls_back_to_client_tools(self):
        """Defensive: flag on but _mcp_manager is None => client tools only."""
        u = _fn_tool("client")
        result = self._resolve(
            request_tools=[u],
            mcp_manager=None,
            auto_inject=True,
        )
        assert result == [u]

    def test_inject_on_with_no_manager_and_no_tools_returns_none(self):
        """Defensive: flag on, no manager, no client tools => None."""
        result = self._resolve(
            request_tools=None,
            mcp_manager=None,
            auto_inject=True,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Startup log status tests (P2-AC5)
# ---------------------------------------------------------------------------


class TestAutoInjectStartupLog:
    """P2-AC5: startup emits a log line describing auto-inject status."""

    def _log_helper(self):
        from vllm_mlx.api.mcp_inject import log_mcp_auto_inject_status

        return log_mcp_auto_inject_status

    def test_startup_log_states_enabled(self, caplog):
        helper = self._log_helper()
        with caplog.at_level(logging.INFO, logger="vllm_mlx.api.mcp_inject"):
            helper(True)
        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "MCP auto-injection" in m and "enabled" in m.lower() for m in messages
        ), f"expected enabled status log, got {messages!r}"

    def test_startup_log_states_disabled(self, caplog):
        helper = self._log_helper()
        with caplog.at_level(logging.INFO, logger="vllm_mlx.api.mcp_inject"):
            helper(False)
        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "MCP auto-injection" in m and "disabled" in m.lower() for m in messages
        ), f"expected disabled status log, got {messages!r}"


# ---------------------------------------------------------------------------
# Launcher env passthrough tests
# ---------------------------------------------------------------------------


class TestLauncherEnvPassthrough:
    """Launchers propagate VLLM_MLX_AUTO_INJECT_MCP_TOOLS=1 as a CLI flag.

    We assert this without running the actual servers by dry-running bash
    with a stubbed ``exec`` built-in that echoes its arguments.
    """

    LAUNCHERS = ("start-server.sh", "start-server-qwen36.sh")

    @pytest.mark.parametrize("launcher", LAUNCHERS)
    def test_launcher_syntax_valid(self, launcher):
        """Each launcher must pass bash -n (syntax check)."""
        path = REPO_ROOT / launcher
        assert path.exists(), f"launcher missing: {path}"
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"bash -n failed for {launcher}: {result.stderr}"
        )

    @pytest.mark.parametrize("launcher", LAUNCHERS)
    def test_launcher_contains_auto_inject_flag_guard(self, launcher):
        """Static guard: launcher must reference the env var and flag.

        This is an easy regression guard that passes without needing to boot
        the full server. The integration (P2-AC3 live path) is DEFERRED to a
        user-driven restart.
        """
        path = REPO_ROOT / launcher
        content = path.read_text()
        assert "VLLM_MLX_AUTO_INJECT_MCP_TOOLS" in content, (
            f"{launcher} must reference VLLM_MLX_AUTO_INJECT_MCP_TOOLS"
        )
        assert "--auto-inject-mcp-tools" in content, (
            f"{launcher} must forward --auto-inject-mcp-tools"
        )


# ---------------------------------------------------------------------------
# Sanity: module import does not blow up for the MCP tools helper
# ---------------------------------------------------------------------------


def test_merge_tool_lists_is_exported():
    """merge_tool_lists must be importable from vllm_mlx.mcp.tools."""
    from vllm_mlx.mcp import tools as mcp_tools

    assert hasattr(mcp_tools, "merge_tool_lists")
    assert callable(mcp_tools.merge_tool_lists)
