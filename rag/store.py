"""
-------------------------------------------------------
pgvector-backed retrieval store for self-improvement tips.
-------------------------------------------------------
Author:  Einstein O
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
import os
from typing import Any, List
import psycopg
from langchain_openai import OpenAIEmbeddings

# Constants

class TipStore:
    """
    -------------------------------------------------------
    A vector store implementation to manage and retrieve 
    self-improvement tips.
    -------------------------------------------------------
    """
    def __init__(self, dsn: str | None = None) -> None:
        """
        -------------------------------------------------------
        Initializes the TipStore with database credentials and 
        embedding model configuration.
        -------------------------------------------------------
        Parameters:
            dsn - PostgreSQL connection string defaults to PGVECTOR_URL 
                env variable (str)
        -------------------------------------------------------
        """
        self.dsn = dsn or os.environ["PGVECTOR_URL"]
        self.embeddings = OpenAIEmbeddings(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    def _conn(self) -> psycopg.Connection:
        """
        -------------------------------------------------------
        Creates and returns a new connection to the PostgreSQL database.
        -------------------------------------------------------
        Returns:
            psycopg.Connection: A connection object to the database.
        -------------------------------------------------------
        """
        return psycopg.connect(self.dsn)

    def embed_and_upsert(self, content: str, metadata: dict[str, Any] | None = None, trace_id: str | List[str] | None = None) -> None:
        """
        -------------------------------------------------------
        Generates an embedding for the provided text and stores 
        it in the database.
        -------------------------------------------------------
        Parameters:
            content: The text content to be embedded and stored (str)
            metadata: metadata to store alongside the content. (dict[str, Any] | None)
            trace_ids: identifier used for tracking the origin of the tip (str | List(str))
        -------------------------------------------------------
        """     
        metadata = metadata or {}
        embedding = self.embeddings.embed_query(content)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tips (embedding, content, metadata, trace_id)
                    VALUES (%s, %s, %s::jsonb, %s)
                    """,
                    (embedding, content, psycopg.types.json.Jsonb(metadata), str(trace_id)),
                )

    def query(self, text: str, k: int = 3) -> list[dict[str, Any]]:
        """
        -------------------------------------------------------
        Performs a vector similarity search to find the most 
        relevant tips for a given query.
        -------------------------------------------------------
        Parameters:
            text: The input query text to search against (str)
            k: The number of top results to retrieve. Defaults to 3 (int)
        Returns:
            list[dict[str, Any]]: A list of dictionaries containing 
                retrieved tip details (id, content, metadata, trace_id, created_at).
        -------------------------------------------------------
        """ 
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