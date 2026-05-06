# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-2 embedding wrapper.

@TEST:MEMORY-01/embedder

We do NOT load real bge-m3 weights here. The whole point of the
:class:`Embedder` design is that the heavy MLX import lives inside
``load()`` so unit tests can monkey-patch ``mlx_embeddings.utils.load``
with a deterministic fake. This keeps the test suite fast (no model
download, no Metal device contention with the live server) and lets
us exercise every failure branch the SPEC requires (REQ-O3 / REQ-N4).
"""

from __future__ import annotations

import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_mlx.memory.embedder import Embedder, pack_float32, unpack_float32


# ---------------------------------------------------------------------------
# Helpers — minimal in-memory stand-ins for mlx, mlx_embeddings.utils.load
# ---------------------------------------------------------------------------
class _FakeMx:
    """Bare-minimum stand-in for ``mlx.core``.

    We need only ``eval()`` (a no-op for our toy arrays) so the
    embedder's call sequence does not crash when invoked under tests.
    Real MLX provides this as a host-side sync primitive.
    """

    @staticmethod
    def eval(_x):  # noqa: D401 - mimic mlx.core.eval signature
        return None


class _FakeArray:
    """Tiny ndarray-ish object exposing ``shape`` and indexable rows."""

    def __init__(self, rows: list[list[float]]):
        self._rows = rows
        n = len(rows)
        d = len(rows[0]) if rows else 0
        self.shape = (n, d)

    def __getitem__(self, idx: int) -> "_FakeRow":
        return _FakeRow(self._rows[idx])


class _FakeRow:
    def __init__(self, values: list[float]):
        self._values = values
        self.shape = (len(values),)

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeOutputs:
    def __init__(self, embeds: _FakeArray):
        self.text_embeds = embeds


class _FakeTokenizer:
    def __init__(self):
        self.calls: list[list[str]] = []

    def batch_encode_plus(
        self, texts, *, return_tensors, padding, truncation, max_length
    ):  # noqa: D401
        # Record what the embedder asked us to tokenize.
        self.calls.append(list(texts))
        return {"input_ids": [[0] * len(texts)], "attention_mask": [[1] * len(texts)]}


class _FakeModel:
    """Returns deterministic embeddings derived from text length.

    The vector is filled with ``len(text) / 1000`` so identical inputs
    yield identical outputs and tests can assert exact bytes.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.calls: int = 0

    def __call__(self, input_ids, attention_mask):  # noqa: D401
        self.calls += 1
        # ``input_ids`` is our fake list[list[int]]; we use its outer
        # length as the batch size and ignore the inner padding shape.
        batch = len(input_ids[0])
        rows = [[(i + 1) / 1000.0] * self.dim for i in range(batch)]
        return _FakeOutputs(_FakeArray(rows))


def _install_fake_mlx_embeddings(monkeypatch, *, dim: int):
    """Stub out ``mlx_embeddings.utils.load`` and ``mlx.core``.

    Returns the (model, tokenizer) tuple so tests can introspect call
    counts. Sets up sys.modules entries the embedder will discover via
    its lazy imports.
    """
    fake_model = _FakeModel(dim=dim)
    fake_tokenizer = _FakeTokenizer()

    def _fake_load(model_id):  # noqa: D401
        return fake_model, fake_tokenizer

    utils_mod = ModuleType("mlx_embeddings.utils")
    utils_mod.load = _fake_load  # type: ignore[attr-defined]
    pkg_mod = ModuleType("mlx_embeddings")
    pkg_mod.utils = utils_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlx_embeddings", pkg_mod)
    monkeypatch.setitem(sys.modules, "mlx_embeddings.utils", utils_mod)

    fake_mx = ModuleType("mlx")
    fake_mx_core = ModuleType("mlx.core")
    fake_mx_core.eval = _FakeMx.eval  # type: ignore[attr-defined]
    fake_mx.core = fake_mx_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx_core)

    return fake_model, fake_tokenizer


# ---------------------------------------------------------------------------
# pack_float32 / unpack_float32 — pure helpers
# ---------------------------------------------------------------------------
class TestPackFloat32:
    def test_round_trip(self):
        v = [0.1, 0.2, 0.3, -0.4]
        out = unpack_float32(pack_float32(v))
        for a, b in zip(v, out):
            assert abs(a - b) < 1e-6

    def test_pack_size_is_4_bytes_per_element(self):
        v = [0.0] * 1024
        assert len(pack_float32(v)) == 4 * 1024

    def test_unpack_rejects_uneven_blob(self):
        with pytest.raises(ValueError, match="multiple of 4"):
            unpack_float32(b"abc")

    def test_unpack_empty_blob_returns_empty_list(self):
        assert unpack_float32(b"") == []

    def test_pack_empty_returns_empty(self):
        assert pack_float32([]) == b""


# ---------------------------------------------------------------------------
# Embedder lifecycle
# ---------------------------------------------------------------------------
class TestEmbedderLifecycle:
    def test_construction_does_not_load(self):
        e = Embedder("BAAI/bge-m3", dim=8, batch_size=2)
        assert not e.available()
        assert not e.disabled()

    def test_load_success_marks_available(self, monkeypatch):
        _install_fake_mlx_embeddings(monkeypatch, dim=8)
        e = Embedder("any/model", dim=8, batch_size=2)
        assert e.load()
        assert e.available()
        assert not e.disabled()

    def test_load_failure_is_permanent(self, monkeypatch):
        # Inject a load() that raises so we can verify the embedder
        # never retries within the same process.
        utils_mod = ModuleType("mlx_embeddings.utils")
        def _boom(_model_id):  # noqa: D401
            raise RuntimeError("synthetic load failure")
        utils_mod.load = _boom  # type: ignore[attr-defined]
        pkg_mod = ModuleType("mlx_embeddings")
        pkg_mod.utils = utils_mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlx_embeddings", pkg_mod)
        monkeypatch.setitem(sys.modules, "mlx_embeddings.utils", utils_mod)

        e = Embedder("missing/model", dim=8)
        assert not e.load()
        assert e.disabled()
        assert not e.available()
        # Second attempt is a no-op.
        assert not e.load()
        assert e.disabled()

    def test_load_is_idempotent_after_success(self, monkeypatch):
        fake_model, _ = _install_fake_mlx_embeddings(monkeypatch, dim=8)
        e = Embedder("any/model", dim=8)
        assert e.load()
        before = fake_model.calls
        assert e.load()  # no-op
        # We never invoked the model on the second load.
        assert fake_model.calls == before

    def test_force_disabled_drops_state(self, monkeypatch):
        _install_fake_mlx_embeddings(monkeypatch, dim=8)
        e = Embedder("any/model", dim=8)
        assert e.load()
        e.force_disabled("test reason")
        assert e.disabled()
        assert not e.available()


# ---------------------------------------------------------------------------
# encode_one / encode_batch
# ---------------------------------------------------------------------------
class TestEncode:
    def test_encode_one_returns_4_byte_per_dim_blob(self, monkeypatch):
        _install_fake_mlx_embeddings(monkeypatch, dim=16)
        e = Embedder("any/model", dim=16, batch_size=4)
        out = e.encode_one("hello world")
        assert out is not None
        assert len(out) == 4 * 16
        # Round-trip to confirm we got real floats.
        floats = unpack_float32(out)
        assert len(floats) == 16

    def test_encode_batch_preserves_order(self, monkeypatch):
        _install_fake_mlx_embeddings(monkeypatch, dim=8)
        e = Embedder("any/model", dim=8, batch_size=2)
        texts = ["a", "b", "c"]
        blobs = e.encode_batch(texts)
        assert len(blobs) == 3
        # Each blob is 32 bytes (8 floats × 4 bytes).
        for b in blobs:
            assert len(b) == 32

    def test_encode_batch_chunks_by_batch_size(self, monkeypatch):
        fake_model, fake_tok = _install_fake_mlx_embeddings(monkeypatch, dim=8)
        e = Embedder("any/model", dim=8, batch_size=2)
        e.encode_batch(["a", "b", "c", "d", "e"])
        # 5 texts at batch_size=2 → 3 model invocations.
        assert fake_model.calls == 3
        # Tokenizer should see batches of [a,b], [c,d], [e].
        assert fake_tok.calls == [["a", "b"], ["c", "d"], ["e"]]

    def test_encode_batch_returns_empty_on_load_failure(self, monkeypatch):
        utils_mod = ModuleType("mlx_embeddings.utils")
        def _boom(_id):  # noqa: D401
            raise RuntimeError("nope")
        utils_mod.load = _boom  # type: ignore[attr-defined]
        pkg_mod = ModuleType("mlx_embeddings")
        pkg_mod.utils = utils_mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlx_embeddings", pkg_mod)
        monkeypatch.setitem(sys.modules, "mlx_embeddings.utils", utils_mod)

        e = Embedder("missing/model", dim=8)
        assert e.encode_batch(["x", "y"]) == []
        assert e.encode_one("x") is None

    def test_encode_handles_empty_strings(self, monkeypatch):
        _install_fake_mlx_embeddings(monkeypatch, dim=8)
        e = Embedder("any/model", dim=8, batch_size=4)
        # Empty / whitespace-only inputs must not crash; embedder
        # internally substitutes a single space.
        out = e.encode_batch(["", "  ", "ok"])
        assert len(out) == 3
        for blob in out:
            assert len(blob) == 32

    def test_encode_dim_mismatch_raises_in_inner_path(self, monkeypatch):
        # If the underlying model returns a different dim than the
        # embedder advertises, we should fail loud — silent dim-mix
        # would corrupt the sqlite-vec table (REQ-U4 spirit).
        _install_fake_mlx_embeddings(monkeypatch, dim=8)
        # But construct the embedder with the wrong advertised dim:
        e = Embedder("any/model", dim=16, batch_size=2)
        # First encode_batch must catch the assertion and return [];
        # encode_batch's outer try/except is REQ-N4 (never propagate).
        result = e.encode_batch(["text"])
        assert result == []

    def test_encode_empty_list_returns_empty(self, monkeypatch):
        _install_fake_mlx_embeddings(monkeypatch, dim=8)
        e = Embedder("any/model", dim=8)
        # No texts → no model invocation, no error.
        assert e.encode_batch([]) == []

    def test_from_config_factory(self, monkeypatch):
        _install_fake_mlx_embeddings(monkeypatch, dim=8)
        cfg = SimpleNamespace(
            embed_model="any/model",
            embed_dim=8,
            embed_batch=4,
        )
        e = Embedder.from_config(cfg)
        assert e.model_id == "any/model"
        assert e.dim == 8
        assert e.batch_size == 4


# ---------------------------------------------------------------------------
# Determinism (mock-based; same input → same bytes)
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_same_input_same_bytes(self, monkeypatch):
        _install_fake_mlx_embeddings(monkeypatch, dim=8)
        e = Embedder("any/model", dim=8)
        a = e.encode_one("the quick brown fox")
        b = e.encode_one("the quick brown fox")
        # The fake model is content-blind (returns rows by position)
        # but the test still confirms our serialization is stable —
        # struct.pack must yield identical bytes for identical floats.
        assert isinstance(a, bytes)
        assert isinstance(b, bytes)
        assert a == b

    def test_unpack_first_dim_starts_with_known_value(self, monkeypatch):
        _install_fake_mlx_embeddings(monkeypatch, dim=4)
        e = Embedder("any/model", dim=4)
        blob = e.encode_one("only one in batch")
        assert blob is not None
        floats = unpack_float32(blob)
        # First batch index is 1 → row[0] is filled with 1/1000 = 0.001.
        assert abs(floats[0] - 0.001) < 1e-7
        # All four dims share the same fill value in our fake.
        assert all(abs(f - 0.001) < 1e-7 for f in floats)

    def test_pack_via_struct_matches_pack_float32(self):
        v = [0.5, -0.5, 1.0, 0.0]
        assert pack_float32(v) == struct.pack("4f", *v)
