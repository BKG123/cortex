from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from .store import SQLiteStore
from .vectors import create_vector_store, BaseVectorStore


@dataclass(frozen=True)
class Message:
    """Represents a single message in a conversation.

    - user_id: application-level user identifier
    - message_id: unique identifier for the message
    - content: raw text content of the message
    - role: role of the sender (user, assistant, system, etc.)
    - timestamp: ISO-8601 string (UTC recommended)
    - conversation_id: optional identifier for grouping messages
    - metadata_json: JSON string of arbitrary attributes
    """

    user_id: str
    message_id: str
    content: str
    role: str
    timestamp: str
    conversation_id: Optional[str] = None
    metadata_json: Optional[str] = None


class ConversationMemory:
    """Manages storage and retrieval of conversation messages with vector embeddings.

    Uses SQLite for metadata storage and configurable vector backends for similarity search.
    Supports local FAISS (default) and cloud vector databases like Pinecone.
    """

    def __init__(
        self,
        db_path: str = "cortex.db",
        vector_dir: str = "vectors",
        vector_backend: str = "faiss",
        **vector_kwargs,
    ) -> None:
        self.store = SQLiteStore(db_path)

        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

        # Initialize vector store backend
        if vector_backend == "faiss":
            vector_kwargs.setdefault("vector_dir", vector_dir)
        self.vector_store = create_vector_store(vector_backend, **vector_kwargs)

        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the database schema for storing conversation messages."""
        self.store.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message_id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                role TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                conversation_id TEXT,
                metadata_json TEXT,
                embedding_path TEXT
            );
            """
        )

        # Create indexes for efficient querying
        self.store.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_user_time
            ON messages (user_id, timestamp DESC);
            """
        )

        self.store.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages (conversation_id, timestamp ASC);
            """
        )

        self.store.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_role
            ON messages (user_id, role, timestamp DESC);
            """
        )

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence transformer."""
        embedding = self.embedding_model.encode([text], convert_to_tensor=False)
        return embedding.astype(np.float32)

    def add_message(self, message: Message) -> str:
        """Add a single message to the conversation memory."""
        # Generate embedding
        embedding = self._get_embedding(message.content)

        # Add to vector store
        self.vector_store.add_vectors(
            vectors=embedding.reshape(1, -1), ids=[message.message_id]
        )

        # Save embedding path (for future reference)
        embedding_path = f"embeddings/{message.message_id}.npy"

        # Store in SQLite
        cur = self.store.execute(
            """
            INSERT INTO messages (user_id, message_id, content, role, timestamp, 
                                conversation_id, metadata_json, embedding_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                message.user_id,
                message.message_id,
                message.content,
                message.role,
                message.timestamp,
                message.conversation_id,
                message.metadata_json,
                embedding_path,
            ],
        )

        return message.message_id

    def add_messages(self, messages: List[Message]) -> List[str]:
        """Add multiple messages to the conversation memory."""
        if not messages:
            return []

        # Generate embeddings for all messages
        contents = [msg.content for msg in messages]
        embeddings = self.embedding_model.encode(contents, convert_to_tensor=False)
        embeddings = embeddings.astype(np.float32)

        # Add to vector store
        message_ids = [msg.message_id for msg in messages]
        self.vector_store.add_vectors(vectors=embeddings, ids=message_ids)

        # Prepare data for SQLite
        params = [
            [
                msg.user_id,
                msg.message_id,
                msg.content,
                msg.role,
                msg.timestamp,
                msg.conversation_id,
                msg.metadata_json,
                f"embeddings/{msg.message_id}.npy",
            ]
            for msg in messages
        ]

        # Store in SQLite
        self.store.executemany(
            """
            INSERT INTO messages (user_id, message_id, content, role, timestamp, 
                                conversation_id, metadata_json, embedding_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params,
        )

        return message_ids

    def search_similar(
        self, user_id: str, query: str, limit: int = 10
    ) -> List[Tuple[Message, float]]:
        """Search for messages similar to the query using vector similarity."""
        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Search vector store
        similar_ids = self.vector_store.search_similar(
            query_vector=query_embedding, k=limit
        )

        # Retrieve messages from SQLite
        results = []
        for message_id, score in similar_ids:
            message = self._get_message_by_id(message_id)
            if message and message.user_id == user_id:
                results.append((message, score))

        # Sort by score (higher is better for cosine similarity)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def search_by_content(
        self, user_id: str, query: str, limit: int = 50
    ) -> List[Message]:
        """Search for messages containing specific text content."""
        results = self.store.query_all(
            """
            SELECT user_id, message_id, content, role, timestamp, 
                   conversation_id, metadata_json
            FROM messages 
            WHERE user_id = ? AND content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [user_id, f"%{query}%", limit],
        )

        return [self._row_to_message(row) for row in results]

    def get_conversation(
        self, user_id: str, limit: int = 100, conversation_id: Optional[str] = None
    ) -> List[Message]:
        """Retrieve conversation messages for a user."""
        if conversation_id:
            results = self.store.query_all(
                """
                SELECT user_id, message_id, content, role, timestamp, 
                       conversation_id, metadata_json
                FROM messages 
                WHERE user_id = ? AND conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                [user_id, conversation_id, limit],
            )
        else:
            results = self.store.query_all(
                """
                SELECT user_id, message_id, content, role, timestamp, 
                       conversation_id, metadata_json
                FROM messages 
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [user_id, limit],
            )

        return [self._row_to_message(row) for row in results]

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's conversations."""
        # Total message count
        total_messages = self.store.query_one(
            "SELECT COUNT(*) as count FROM messages WHERE user_id = ?", [user_id]
        )

        # Message count by role
        role_counts = self.store.query_all(
            """
            SELECT role, COUNT(*) as count 
            FROM messages 
            WHERE user_id = ? 
            GROUP BY role
            """,
            [user_id],
        )

        # Conversation count
        conversation_count = self.store.query_one(
            """
            SELECT COUNT(DISTINCT conversation_id) as count 
            FROM messages 
            WHERE user_id = ? AND conversation_id IS NOT NULL
            """,
            [user_id],
        )

        # Vector store stats
        vector_stats = self.vector_store.get_stats()

        return {
            "user_id": user_id,
            "total_messages": total_messages["count"] if total_messages else 0,
            "role_counts": {row["role"]: row["count"] for row in role_counts},
            "conversation_count": conversation_count["count"]
            if conversation_count
            else 0,
            "vector_store": vector_stats,
        }

    def delete_user_messages(self, user_id: str) -> int:
        """Delete all messages for a user."""
        # Get message IDs to delete from vector store
        message_ids = self.store.query_all(
            "SELECT message_id FROM messages WHERE user_id = ?", [user_id]
        )

        # Delete from vector store
        for row in message_ids:
            self.vector_store.delete_vector(row["message_id"])

        # Delete from SQLite
        cur = self.store.execute("DELETE FROM messages WHERE user_id = ?", [user_id])

        return cur.rowcount

    def _get_message_by_id(self, message_id: str) -> Optional[Message]:
        """Retrieve a message by its ID."""
        row = self.store.query_one(
            """
            SELECT user_id, message_id, content, role, timestamp, 
                   conversation_id, metadata_json
            FROM messages 
            WHERE message_id = ?
            """,
            [message_id],
        )

        return self._row_to_message(row) if row else None

    def _row_to_message(self, row) -> Message:
        """Convert a database row to a Message object."""
        return Message(
            user_id=row["user_id"],
            message_id=row["message_id"],
            content=row["content"],
            role=row["role"],
            timestamp=row["timestamp"],
            conversation_id=row["conversation_id"],
            metadata_json=row["metadata_json"],
        )

    def close(self) -> None:
        """Close the conversation memory and clean up resources."""
        self.store.close()
        self.vector_store.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
