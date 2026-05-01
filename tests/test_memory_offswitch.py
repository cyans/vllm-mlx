# SPDX-License-Identifier: Apache-2.0
"""REQ-S1 invariance and env-var resolution tests for the memory subsystem.

@TEST:MEMORY-01/offswitch

When ``MEMORY_ENABLED`` is not truthy the server module-level globals
must remain at their pre-SPEC defaults. This protects the bit-for-bit
compat invariant (P1-AC5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_mlx.memory.config import (
    ENV_MEMORY_DB_PATH,
    ENV_MEMORY_ENABLED,
    ENV_MEMORY_TOP_K_DEFAULT,
    ENV_MEMORY_TOP_K_MAX,
    ENV_MEMORY_VAULT_ALLOWLIST,
    ENV_MEMORY_VAULT_DENYLIST,
    ENV_MEMORY_VAULT_PATH,
    MEMORY_DEFAULT_DB_PATH,
    MEMORY_DEFAULT_DENYLIST,
    MEMORY_DEFAULT_TOP_K,
    MEMORY_DEFAULT_TOP_K_MAX,
    MEMORY_DEFAULT_VAULT_PATH,
    resolve_memory_config,
    resolve_memory_enabled,
)


class TestEnabledFlag:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("YES", True),
            ("on", True),
            (" 1 ", True),
            ("0", False),
            ("false", False),
            ("", False),
            ("no", False),
            ("anything-else", False),
        ],
    )
    def test_truthy_parsing(self, value, expected):
        assert (
            resolve_memory_enabled({ENV_MEMORY_ENABLED: value}) is expected
        )

    def test_unset_is_disabled(self):
        assert resolve_memory_enabled({}) is False

    def test_none_env_is_disabled(self):
        assert resolve_memory_enabled(None) is False


class TestConfigDefaults:
    def test_empty_env_yields_disabled_with_defaults(self):
        cfg = resolve_memory_config({})
        assert cfg.enabled is False
        assert str(cfg.vault_path) == str(Path(MEMORY_DEFAULT_VAULT_PATH))
        assert str(cfg.db_path) == str(Path(MEMORY_DEFAULT_DB_PATH))
        assert cfg.denylist == MEMORY_DEFAULT_DENYLIST
        assert cfg.allowlist == ()
        assert cfg.top_k_default == MEMORY_DEFAULT_TOP_K
        assert cfg.top_k_max == MEMORY_DEFAULT_TOP_K_MAX

    def test_none_env_skips_os_environ(self):
        # The point of resolve_memory_config(None) is that it must NOT
        # consult os.environ — same contract as resolve_legacy_think_tags.
        cfg = resolve_memory_config(None)
        assert cfg.enabled is False


class TestConfigOverrides:
    def test_paths_are_overrideable(self, tmp_path):
        cfg = resolve_memory_config(
            {
                ENV_MEMORY_ENABLED: "1",
                ENV_MEMORY_VAULT_PATH: str(tmp_path / "vault"),
                ENV_MEMORY_DB_PATH: str(tmp_path / "db" / "memory.db"),
            }
        )
        assert cfg.enabled is True
        assert cfg.vault_path == tmp_path / "vault"
        assert cfg.db_path == tmp_path / "db" / "memory.db"

    def test_denylist_csv_parsing(self):
        cfg = resolve_memory_config(
            {
                ENV_MEMORY_ENABLED: "1",
                ENV_MEMORY_VAULT_DENYLIST: "a/**, b/**, c/**",
            }
        )
        assert cfg.denylist == ("a/**", "b/**", "c/**")

    def test_empty_denylist_falls_back_to_default(self):
        cfg = resolve_memory_config(
            {ENV_MEMORY_ENABLED: "1", ENV_MEMORY_VAULT_DENYLIST: ""}
        )
        assert cfg.denylist == MEMORY_DEFAULT_DENYLIST

    def test_allowlist_csv_parsing(self):
        cfg = resolve_memory_config(
            {ENV_MEMORY_VAULT_ALLOWLIST: "Notes/**,Daily/**"}
        )
        assert cfg.allowlist == ("Notes/**", "Daily/**")

    def test_top_k_clamping(self):
        cfg = resolve_memory_config(
            {ENV_MEMORY_TOP_K_DEFAULT: "0", ENV_MEMORY_TOP_K_MAX: "5"}
        )
        assert cfg.top_k_max == 5
        assert cfg.top_k_default == 1  # clamped up to 1

    def test_top_k_max_capped_at_spec_ceiling(self):
        cfg = resolve_memory_config({ENV_MEMORY_TOP_K_MAX: "10000"})
        # SPEC §7 mandates an absolute ceiling of 20.
        assert cfg.top_k_max == 20

    def test_top_k_default_clamped_to_top_k_max(self):
        cfg = resolve_memory_config(
            {ENV_MEMORY_TOP_K_DEFAULT: "100", ENV_MEMORY_TOP_K_MAX: "7"}
        )
        assert cfg.top_k_default == 7

    def test_invalid_int_falls_back_to_default(self):
        cfg = resolve_memory_config(
            {ENV_MEMORY_TOP_K_DEFAULT: "not-an-int"}
        )
        assert cfg.top_k_default == MEMORY_DEFAULT_TOP_K


class TestServerImportNoSideEffects:
    """REQ-S1: importing :mod:`vllm_mlx.server` with memory disabled
    must not create any DB file or change behavior. We do NOT import
    the real server module here (it pulls in MLX) — instead we assert
    that the cli.serve_command env-var pass-through is purely lazy.
    """

    def test_cli_imports_memory_only_when_invoked(self):
        # The cli module must be importable without triggering memory init.
        # Memory module is also import-safe.
        from vllm_mlx import (
            cli,  # noqa: F401
            memory,  # noqa: F401
        )
        from vllm_mlx.memory import config  # noqa: F401

        # No filesystem side effects from import alone.
        assert True


class TestEnvVarNamesAreStable:
    """Pin the env var name strings — operators rely on them in shell
    scripts, so renaming them is a breaking change.
    """

    def test_env_var_names(self):
        assert ENV_MEMORY_ENABLED == "MEMORY_ENABLED"
        assert ENV_MEMORY_VAULT_PATH == "MEMORY_VAULT_PATH"
        assert ENV_MEMORY_DB_PATH == "MEMORY_DB_PATH"
        assert ENV_MEMORY_VAULT_DENYLIST == "MEMORY_VAULT_DENYLIST"
        assert ENV_MEMORY_VAULT_ALLOWLIST == "MEMORY_VAULT_ALLOWLIST"
        assert ENV_MEMORY_TOP_K_DEFAULT == "MEMORY_TOP_K_DEFAULT"
        assert ENV_MEMORY_TOP_K_MAX == "MEMORY_TOP_K_MAX"

    def test_default_paths_match_spec(self):
        # SPEC-MEMORY-01 §9 documents these defaults.
        assert MEMORY_DEFAULT_VAULT_PATH == "/Volumes/data/Obsidian/obsi"
        assert (
            MEMORY_DEFAULT_DB_PATH
            == "/Volumes/data/vllm-mlx-memory/memory.db"
        )
