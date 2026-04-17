# @CODE:FIX-QWEN36-AUTO-TOOL-CHOICE/launcher-tests
# Phase 1 tests: verify that start-server*.sh launchers pass
# --enable-auto-tool-choice by default, with opt-out via the
# VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE environment variable. The 122B mint
# launcher is explicitly out of scope for Phase 1 (REQ-N1).
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def _assert_gated_auto_tool_choice_block(content: str, name: str) -> None:
    """Shared assertions: an opt-out gate plus a --enable-auto-tool-choice
    EXTRA_FLAGS append that is reached only when the gate is not set. We
    also tolerate an optional --tool-call-parser argument next to the
    --enable-auto-tool-choice flag, since the server-side CLI requires it.
    """
    assert "VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE" in content, (
        f"{name}: missing opt-out env var"
    )
    assert "--enable-auto-tool-choice" in content, (
        f"{name}: missing --enable-auto-tool-choice flag"
    )
    # Ensure opt-out gate is a conditional append, not unconditional.
    # Allow any number of intervening setup lines inside the if-block (e.g.
    # resolving --tool-call-parser via Python) before the EXTRA_FLAGS append.
    assert re.search(
        r'if\s+\[\[\s*"\$\{VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE[^}]*\}"\s*!=\s*"1"\s*\]\];\s*then'
        r".*?EXTRA_FLAGS\+=\([^)]*--enable-auto-tool-choice[^)]*\)"
        r".*?fi",
        content,
        re.DOTALL,
    ), f"{name}: expected opt-out gated append of --enable-auto-tool-choice"


def test_start_server_has_auto_tool_choice_block():
    _assert_gated_auto_tool_choice_block(_read("start-server.sh"), "start-server.sh")


def test_start_server_qwen36_has_auto_tool_choice_block():
    _assert_gated_auto_tool_choice_block(
        _read("start-server-qwen36.sh"), "start-server-qwen36.sh"
    )


def test_auto_tool_choice_is_not_hardcoded_in_exec():
    # The flag should be added via EXTRA_FLAGS, not appear twice in the exec line.
    for name in ("start-server.sh", "start-server-qwen36.sh"):
        content = _read(name)
        exec_line = re.search(r"exec vllm-mlx serve[^\n]*\\", content)
        assert (
            exec_line is None or "--enable-auto-tool-choice" not in exec_line.group(0)
        ), (
            f"{name}: --enable-auto-tool-choice should not be hardcoded in exec line; "
            "use EXTRA_FLAGS"
        )


def test_122b_launcher_untouched():
    # start-server-122b-mint.sh is OUT OF SCOPE per REQ-N1.
    content = _read("start-server-122b-mint.sh")
    assert "VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE" not in content
    assert "--enable-auto-tool-choice" not in content
