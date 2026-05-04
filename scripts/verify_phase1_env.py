#!/usr/bin/env python3
"""PLAN.md Phase 1 성공 기준 검증: mlx-optiq / TurboQuant import."""
from __future__ import annotations

import sys


def _pkg_version(name: str) -> str | None:
    """배포 패키지 버전 (mlx는 __version__ 미노출인 경우가 있음)."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def main() -> int:
    errors: list[str] = []

    try:
        import mlx  # noqa: F401

        v = getattr(mlx, "__version__", None) or _pkg_version("mlx")
        if v:
            print(f"mlx: {v}")
        else:
            print("mlx: import OK (버전 문자열 없음)")
    except Exception as e:
        errors.append(f"mlx import 실패: {e}")

    try:
        import mlx_lm  # noqa: F401

        v = mlx_lm.__version__
        print(f"mlx-lm: {v}")
        parts = v.split(".")
        minor_ok = False
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            minor_ok = int(parts[0]) > 0 or int(parts[1]) >= 20
        if not minor_ok:
            try:
                from packaging.version import Version

                minor_ok = Version(v) >= Version("0.20.0")
            except Exception:
                minor_ok = True  # 파싱 실패 시 경고만
        if not minor_ok:
            errors.append(f"PLAN 권장: mlx-lm >= 0.20 (현재 {v})")
        else:
            print("mlx-lm 버전: 0.20 이상 요건 충족(또는 파싱 생략)")
    except Exception as e:
        errors.append(f"mlx_lm import 실패: {e}")

    try:
        import optiq

        print(f"optiq (mlx-optiq): {getattr(optiq, '__version__', 'unknown')}")
    except Exception as e:
        errors.append(f"optiq import 실패: {e}")

    try:
        from optiq.core.turbo_kv_cache import TurboQuantKVCache, patch_attention

        print("TurboQuant import OK: TurboQuantKVCache, patch_attention")
        _ = TurboQuantKVCache
        _ = patch_attention
    except Exception as e:
        errors.append(f"TurboQuant import 실패: {e}")

    if errors:
        print("\n[실패]", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    print("\n[Phase 1 검증] 모두 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
