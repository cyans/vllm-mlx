# SPDX-License-Identifier: Apache-2.0
"""
MCP tool auto-injection helpers for /v1/chat/completions.

This module owns the opt-in tool auto-injection logic for Phase 2 of
SPEC-FIX-QWEN36-RUNTIME. The decision is factored out of ``vllm_mlx.server``
into this import-light module so the rules can be exercised without booting
mlx / Metal / the full FastAPI app.

Default behaviour: auto-injection is OFF. Bit-for-bit compat with Qwen 3.5
is preserved per REQ-N2.

@CODE:FIX-QWEN36-RUNTIME/mcp-auto-inject
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_mcp_auto_inject_status(enabled: bool) -> None:
    """Emit a single INFO line describing the MCP auto-inject status.

    Called once at server startup so operators can see at a glance whether
    the /v1/chat/completions endpoint will surface MCP-registered tools on
    requests that do not supply their own 'tools' field.

    Args:
        enabled: True if --auto-inject-mcp-tools was passed.
    """
    if enabled:
        logger.info("MCP auto-injection: enabled")
    else:
        logger.info("MCP auto-injection: disabled (default)")


def resolve_effective_tools(
    request_tools: list[Any] | None,
    mcp_manager: Any,
    auto_inject: bool,
) -> list[Any] | None:
    """Compute the effective tool list for a chat completion.

    This is the single decision point for whether MCP-registered tools are
    surfaced to the model. It is a pure function of its inputs so it can
    be unit-tested without booting the FastAPI app.

    Resolution rules:
      * ``auto_inject=False`` (default): behaviour is bit-for-bit identical
        to the pre-Phase-2 server — return ``request_tools`` unchanged when
        the client supplied tools, otherwise return ``None``. The MCP
        manager is NOT consulted (REQ-N2).
      * ``auto_inject=True`` + no client tools: return the full MCP tool
        list in OpenAI format (REQ-E2). If ``mcp_manager`` is ``None``
        (MCP not configured), return ``None``.
      * ``auto_inject=True`` + client tools present: return a merged list
        with client tools first, then non-colliding MCP tools. On name
        collision the client tool wins (REQ-E3 / P2-AC4).

    Args:
        request_tools: The request's ``tools`` field; a list of Pydantic
            ``ToolDefinition`` objects (or dicts) or ``None``.
        mcp_manager: The module-level ``_mcp_manager`` or a compatible
            stub exposing ``get_merged_tools()``.
        auto_inject: Effective auto-inject flag.

    Returns:
        A list of tools (Pydantic objects and/or dicts) to pass to
        ``convert_tools_for_template``, or ``None`` when no tools should
        be attached to the prompt.
    """
    # Fast path: auto-inject off. Preserve exact pre-Phase-2 behaviour.
    if not auto_inject:
        return request_tools if request_tools else None

    # Auto-inject on but MCP not configured: degrade gracefully to the
    # pre-Phase-2 behaviour instead of blowing up.
    if mcp_manager is None:
        return request_tools if request_tools else None

    # Pull the MCP-registered tools in OpenAI format. get_merged_tools()
    # with no user_tools argument returns the pure MCP tool list.
    try:
        mcp_tools = mcp_manager.get_merged_tools()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "MCP auto-injection: failed to fetch tools, falling back "
            "to client-only tools: %s",
            exc,
        )
        return request_tools if request_tools else None

    if not request_tools:
        return mcp_tools if mcp_tools else None

    # Both client and MCP tools present — merge with client priority.
    # Lazy import avoids circular imports at package load time.
    from ..mcp.tools import merge_tool_lists

    # Coerce client Pydantic ToolDefinition models to dicts for the
    # collision check; convert_tools_for_template handles both forms.
    client_as_dicts: list[Any] = []
    for tool in request_tools:
        if hasattr(tool, "model_dump"):
            client_as_dicts.append(tool.model_dump(exclude_none=True))
        else:
            client_as_dicts.append(tool)

    return merge_tool_lists(client_as_dicts, mcp_tools)
