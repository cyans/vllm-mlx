# Security checklist

- Symlink-escape guard on the indexer (REQ-N2)
- Redaction patterns applied pre-write (REQ-N1)
- Local-only: no outbound HTTP from the memory server (REQ-U2)
- Opt-in: MEMORY_ENABLED defaults to 0 (REQ-S1)
