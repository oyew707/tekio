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
import json
import psycopg
from langchain_openai import OpenAIEmbeddings
from utils.logger import get_logger

# Constants
logger = get_logger(__name__, "debug")


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
        self.logger = logger.getChild(__name__)
        self.dsn = dsn or os.environ["PGVECTOR_URL"]
        self.logger.debug(f"Initializing TipStore with dsn: {self.dsn}")
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
        self.logger.debug("Attempting database connection")
        try:
            conn = psycopg.connect(self.dsn)
            self.logger.info("Database connection established")
            return conn
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    def delete_tip(self, tip_id) -> bool:
        """
        -------------------------------------------------------
        Deletes a tip from the database by its ID.
        -------------------------------------------------------
        Parameters:
            tip_id - The UUID of the tip to delete (str)
        Returns:
            success - True if a row was deleted, False otherwise (bool)
        -------------------------------------------------------
        """
        self.logger.info(f"Deleting tip with id: {tip_id}")
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tips WHERE id = %s", (tip_id,))
                return cur.rowcount > 0

    def all_tips(self, include_embedding: bool = True) -> list[dict[str, Any]]:
        """
        -------------------------------------------------------
        Retrieves all tips from the database.
        -------------------------------------------------------
        Parameters:
            include_embedding - Whether to include the embedding vector
                in the results (bool)
        Returns:
            list[dict[str, Any]]: A list of dictionaries containing
                tip details
        -------------------------------------------------------
        """
        self.logger.info("Retrieving all tips from database")

        cols = "id::text, content, metadata, trace_id, created_at"
        if include_embedding:
            cols += ", embedding"

        query = f"SELECT {cols} FROM tips"

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()

        result = []
        for row in rows:
            # Map columns dynamically based on include_embedding
            base_data = {
                "id": row[0],
                "content": row[1],
                "metadata": row[2],
                "trace_id": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
            }
            if include_embedding:
                # row[5] is the embedding vector
                base_data["embedding"] = json.loads(row[5])
            result.append(base_data)

        return result

    def embed_and_upsert(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        trace_id: str | List[str] | None = None,
    ) -> None:
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
        self.logger.info(f"Upserting tip for content: {content[:50]}...")
        metadata = metadata or {}
        embedding = self.embeddings.embed_query(content)
        self.logger.debug("Embedding generated successfully")

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tips (embedding, content, metadata, trace_id)
                    VALUES (%s, %s, %s::jsonb, %s)
                    """,
                    (
                        embedding,
                        content,
                        psycopg.types.json.Jsonb(metadata),
                        str(trace_id),
                    ),
                )
                self.logger.info(f"Upsert successful for trace_id: {trace_id}")

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
        self.logger.info(f"Querying tips with k={k} for text: {text[:50]}...")
        embedding = self.embeddings.embed_query(text)
        self.logger.debug(f"Query embedding generated {len(embedding)}")

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id::text, content, metadata, trace_id, created_at
                    FROM tips
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, k),
                )
                rows = cur.fetchall()

        if not rows:
            self.logger.warning(f"No results found for query: {text}")
        else:
            self.logger.info(f"Query returned {len(rows)} results")

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
