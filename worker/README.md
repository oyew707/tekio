# worker (Phase 2 placeholder)

This directory is reserved for the queue consumer that will process `trajectory_ready`
messages, pull full LangSmith trajectories by trace_id, summarize them into tips,
and call `app/rag/store.py::embed_and_upsert`.
