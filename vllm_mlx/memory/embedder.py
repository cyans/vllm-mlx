# SPDX-License-Identifier: Apache-2.0
"""Embedding pipeline for the memory subsystem.

@CODE:MEMORY-01/embedder

Phase 2: lazy-load BAAI/bge-m3 (or any compatible XLM-RoBERTa model)
through ``mlx-embeddings`` and produce 1024-dim normalized dense
vectors for batches of text. Apple Silicon only — there is no fallback
to CPU torch and we do not pretend to support one (REQ-U2 / SPEC §4).

Failure isolation (REQ-N4 / REQ-O3):

* The model is loaded on first use, not at construction time, so an
  ``Embedder`` can be instantiated even when MLX is unavailable.
* Every entry point that touches MLX (``load``, ``encode_one``,
  ``encode_batch``) is wrapped with ``logger.exception`` + a sentinel
  return value (``None`` for one-shot, ``[]`` for batch) so a Metal
  failure cannot bubble into ``/v1/chat/completions``.
* Once a load fails, the embedder enters a permanent disabled state
  (``available()`` returns False) and never retries during the same
  process lifetime; callers must restart to reload weights.

Output format: each call returns ``bytes`` already serialized as
little-endian float32 via :func:`pack_float32`. This is the format
``sqlite-vec`` accepts directly via parameterized queries, so we
avoid an extra serialization step in the store.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def pack_float32(vector: Sequence[float]) -> bytes:
    """Serialize a vector of floats to ``sqlite-vec`` raw-bytes format.

    ``sqlite-vec`` expects little-endian float32 with no header. We use
    the builtin ``struct`` module rather than ``numpy.tobytes`` so this
    helper has no dependency on the array library.
    """
    return struct.pack(f"{len(vector)}f", *vector)


def unpack_float32(blob: bytes) -> list[float]:
    """Inverse of :func:`pack_float32`. Mainly for tests and debugging."""
    if not blob:
        return []
    if len(blob) % 4 != 0:
        raise ValueError(
            f"unpack_float32: blob length {len(blob)} is not a multiple of 4"
        )
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
class Embedder:
    """Lazy-loading wrapper around ``mlx-embeddings`` for a single model.

    Construct with :func:`Embedder` directly or via the convenience
    :func:`Embedder.from_config` helper. The model is not loaded until
    the first call to :meth:`encode_one` or :meth:`encode_batch`.
    """

    def __init__(
        self,
        model_id: str,
        *,
        dim: int,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id
        self.dim = int(dim)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(1, int(max_length))

        # State machine: None=unloaded, "loading"=in progress (single-thread
        # process so we do not need a real lock), "ready"=loaded,
        # "failed"=permanent disabled. Transitions are monotonic except
        # for unloaded → loading → (ready | failed).
        self._state: str = "unloaded"
        self._model: Any = None
        self._tokenizer: Any = None

    # -- factories ----------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> Embedder:
        """Construct an :class:`Embedder` from a :class:`MemoryRuntimeConfig`.

        The class avoids importing the config module to keep this
        module standalone (the only required attributes are documented
        in SPEC §9).
        """
        return cls(
            model_id=str(config.embed_model),
            dim=int(config.embed_dim),
            batch_size=int(config.embed_batch),
        )

    # -- introspection ------------------------------------------------------

    def available(self) -> bool:
        """Return True iff a successful load has completed.

        Note: returns False both before the first load attempt and
        after a failed load. Use :meth:`disabled` to distinguish.
        """
        return self._state == "ready"

    def disabled(self) -> bool:
        """Return True iff a load attempt has failed permanently."""
        return self._state == "failed"

    def force_disabled(self, reason: str) -> None:
        """Mark the embedder as permanently disabled.

        Used by REQ-U4 model-mismatch detection in the store: when the
        DB already contains vectors from a different model, we do not
        try to mix dimensions or model versions in one DB.
        """
        if self._state == "ready":
            logger.warning(
                "[memory] embedder forced disabled after load: %s", reason
            )
        else:
            logger.info("[memory] embedder disabled before load: %s", reason)
        self._state = "failed"
        # Drop refs so MLX can reclaim memory if it was loaded.
        self._model = None
        self._tokenizer = None

    # -- public API ---------------------------------------------------------

    def load(self) -> bool:
        """Load the underlying model + tokenizer.

        Returns True on success and False on any failure. Idempotent:
        calling :meth:`load` after a successful load is a no-op; calling
        it after a failed load is also a no-op (the embedder stays in
        the failed state). All exceptions are swallowed and logged.
        """
        if self._state == "ready":
            return True
        if self._state == "failed":
            return False

        self._state = "loading"
        try:
            # Heavy imports kept inside ``load`` so unit tests that mock
            # the Embedder do not need to import mlx_embeddings or MLX
            # itself (the live server holds the Metal device).
            from mlx_embeddings.utils import load as _ml_load

            logger.info(
                "[memory] loading embedding model %s (dim=%d, batch=%d) — "
                "this may take a while on first run while bge-m3 weights "
                "are downloaded (~2.3 GB)",
                self.model_id,
                self.dim,
                self.batch_size,
            )
            model, tokenizer = _ml_load(self.model_id)
            self._model = model
            self._tokenizer = tokenizer
            self._state = "ready"
            logger.info(
                "[memory] embedder ready: model=%s dim=%d", self.model_id, self.dim
            )
            return True
        except Exception:  # noqa: BLE001 — REQ-O3 + REQ-N4
            logger.exception(
                "[memory] embedder load failed for model=%s — falling back to "
                "BM25-only path. Hint: try MEMORY_EMBED_MODEL="
                "mlx-community/bge-m3-mlx-fp16 for the pre-converted MLX "
                "weights, or set MEMORY_EMBED_DISABLED=1 to silence this "
                "diagnostic.",
                self.model_id,
            )
            self._state = "failed"
            self._model = None
            self._tokenizer = None
            return False

    def encode_one(self, text: str) -> bytes | None:
        """Encode a single string. Returns ``None`` on any failure."""
        out = self.encode_batch([text])
        if not out:
            return None
        return out[0]

    def encode_batch(self, texts: Sequence[str]) -> list[bytes]:
        """Encode a list of strings to a list of float32-packed blobs.

        Order of the returned list matches ``texts``. On any failure
        the entire batch returns ``[]`` and the caller should treat
        that as a degraded result (REQ-O3): subsequent calls keep
        trying because transient failures may resolve.

        For batches larger than ``batch_size`` we chunk internally to
        keep MLX peak memory bounded.
        """
        if not texts:
            return []
        if not self.load():
            return []

        try:
            return self._encode_chunks(list(texts))
        except Exception:  # noqa: BLE001 — REQ-N4
            logger.exception(
                "[memory] encode_batch failed for %d texts (model=%s); "
                "returning empty so the caller can degrade to BM25",
                len(texts),
                self.model_id,
            )
            return []

    # -- internals ----------------------------------------------------------

    def _encode_chunks(self, texts: list[str]) -> list[bytes]:
        """Run the model on slices of at most ``batch_size`` texts."""
        # Heavy imports deferred to call time (see ``load``).
        import mlx.core as mx  # noqa: PLC0415 - intentional lazy import

        out: list[bytes] = []
        bs = self.batch_size

        # Sanitize: replace empty strings with a single space so the
        # tokenizer does not emit zero-length sequences (which crash
        # the XLM-RoBERTa attention). Empty inputs still get a vector
        # back, which simplifies callers (one-to-one shape contract).
        normalized = [t if t.strip() else " " for t in texts]

        for i in range(0, len(normalized), bs):
            batch = normalized[i : i + bs]
            inputs = self._tokenizer.batch_encode_plus(
                batch,
                return_tensors="mlx",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            outputs = self._model(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            # ``text_embeds`` are mean-pooled and L2-normalized inside
            # the model (see mlx_embeddings/models/xlm_roberta.py).
            embeds = outputs.text_embeds
            # Force-eval the lazy MLX graph and convert each row into
            # a python float list. ``mx.eval`` is cheap to call on
            # already-eval'd arrays.
            mx.eval(embeds)
            for j in range(embeds.shape[0]):
                row = embeds[j]
                # Sanity check: dim must match our advertised contract,
                # otherwise the store would silently insert wrong-sized
                # blobs. This is fast to assert and catches model swaps.
                if int(row.shape[-1]) != self.dim:
                    raise ValueError(
                        f"embedder dim mismatch: got {int(row.shape[-1])}, "
                        f"expected {self.dim} (model={self.model_id})"
                    )
                # ``tolist()`` triggers a host copy; that's the right
                # tradeoff for sqlite ingestion since we are already
                # leaving the GPU.
                out.append(pack_float32(row.tolist()))
        return out


__all__ = [
    "Embedder",
    "pack_float32",
    "unpack_float32",
]
