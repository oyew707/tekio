"""pgvector-backed retrieval store for self-improvement tips."""

from __future__ import annotations

import os
from typing import Any

import psycopg
from langchain_openai import OpenAIEmbeddings


class TipStore:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.environ["PGVECTOR_URL"]
        self.embeddings = OpenAIEmbeddings(
            base_url=os.environ["OPENAI_API_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    def _conn(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn)

    def embed_and_upsert(self, content: str, metadata: dict[str, Any] | None = None, trace_id: str | None = None) -> None:
        metadata = metadata or {}
        embedding = self.embeddings.embed_query(content)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tips (embedding, content, metadata, trace_id)
                    VALUES (%s, %s, %s::jsonb, %s)
                    """,
                    (embedding, content, psycopg.types.json.Jsonb(metadata), trace_id),
                )

    def query(self, text: str, k: int = 3) -> list[dict[str, Any]]:
        embedding = self.embeddings.embed_query(text)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, content, metadata, trace_id, created_at
                    FROM tips
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (embedding, k),
                )
                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "content": row[1],
                "metadata": row[2],
                "trace_id": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
            }
            for row in rows
        ]
