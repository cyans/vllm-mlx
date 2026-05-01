# Chunking strategy

Markdown headers naturally segment a document into coherent units. The
fallback for long sections is a 512-token sliding window with 64-token
overlap to preserve context across boundaries.

Chunk IDs need to be deterministic so re-indexing is idempotent.
