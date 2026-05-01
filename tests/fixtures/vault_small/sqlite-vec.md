# sqlite-vec adoption decision

sqlite-vec ships as a single C extension and co-locates with our chat
log table. No separate daemon, single backup target, FTS5 fallback if
the extension fails to load.

Risks: macOS system Python sometimes ships a different SQLite; verified
working on macOS 14+ with Python 3.10+.
