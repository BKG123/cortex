from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .store import SQLiteStore


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

    Uses SQLite for metadata storage and FAISS for vector similarity search.
    All data is stored locally.
    """

    def __init__(self, db_path: str = "cortex.db", vector_dir: str = "vectors") -> None:
        self.store = SQLiteStore(db_path)
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(exist_ok=True)

        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

        # FAISS index for vector search
        self.index = faiss.IndexFlatIP(
            self.embedding_dim
        )  # Inner product for cosine similarity
        self.message_ids: List[str] = []  # Maps FAISS index to message_id

        self._ensure_schema()
        self._load_existing_vectors()

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

    def _load_existing_vectors(self) -> None:
        """Load existing vectors from disk and rebuild FAISS index."""
        vector_file = self.vector_dir / "faiss_index.bin"
        ids_file = self.vector_dir / "message_ids.pkl"

        if vector_file.exists() and ids_file.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(vector_file))

                # Load message IDs mapping
                with open(ids_file, "rb") as f:
                    self.message_ids = pickle.load(f)

                print(f"Loaded {len(self.message_ids)} existing vectors")
            except Exception as e:
                print(f"Error loading existing vectors: {e}")
                # Reset to empty index
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.message_ids = []

    def _save_vectors(self) -> None:
        """Save FAISS index and message IDs to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.vector_dir / "faiss_index.bin"))

            # Save message IDs mapping
            with open(self.vector_dir / "message_ids.pkl", "wb") as f:
                pickle.dump(self.message_ids, f)
        except Exception as e:
            print(f"Error saving vectors: {e}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence transformer."""
        embedding = self.embedding_model.encode([text], convert_to_tensor=False)
        return embedding.astype(np.float32)

    def add_message(self, message: Message) -> str:
        """Add a single message to the conversation memory."""
        # Generate embedding
        embedding = self._get_embedding(message.content)

        # Add to FAISS index
        self.index.add(embedding)
        self.message_ids.append(message.message_id)

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

        # Save vectors to disk
        self._save_vectors()

        return message.message_id

    def add_messages(self, messages: List[Message]) -> List[str]:
        """Add multiple messages to the conversation memory."""
        if not messages:
            return []

        # Generate embeddings for all messages
        contents = [msg.content for msg in messages]
        embeddings = self.embedding_model.encode(contents, convert_to_tensor=False)
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS index
        self.index.add(embeddings)

        # Add message IDs to mapping
        message_ids = [msg.message_id for msg in messages]
        self.message_ids.extend(message_ids)

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

        # Save vectors to disk
        self._save_vectors()

        return message_ids

    def get_conversation(
        self, user_id: str, conversation_id: Optional[str] = None, limit: int = 100
    ) -> List[Message]:
        """Retrieve conversation messages for a user."""
        if conversation_id:
            rows = self.store.query_all(
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
            rows = self.store.query_all(
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

        return [Message(**dict(row)) for row in rows]

    def search_similar(
        self, user_id: str, query: str, limit: int = 10
    ) -> List[Tuple[Message, float]]:
        """Search for messages similar to the query using vector similarity."""
        # Generate embedding for query
        query_embedding = self._get_embedding(query)

        # Search in FAISS index
        scores, indices = self.index.search(
            query_embedding, min(limit * 2, len(self.message_ids))
        )

        # Get messages for the found indices
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.message_ids):
                message_id = self.message_ids[idx]

                # Get message from database
                row = self.store.query_one(
                    """
                    SELECT user_id, message_id, content, role, timestamp, 
                           conversation_id, metadata_json
                    FROM messages
                    WHERE message_id = ? AND user_id = ?
                    """,
                    [message_id, user_id],
                )

                if row:
                    message = Message(**dict(row))
                    results.append((message, float(score)))

                    if len(results) >= limit:
                        break

        return results

    def search_by_content(
        self, user_id: str, query: str, limit: int = 50
    ) -> List[Message]:
        """Search messages by content using text search."""
        pattern = f"%{query}%"
        rows = self.store.query_all(
            """
            SELECT user_id, message_id, content, role, timestamp, 
                   conversation_id, metadata_json
            FROM messages
            WHERE user_id = ? AND content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [user_id, pattern, limit],
        )
        return [Message(**dict(row)) for row in rows]

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user's conversations."""
        # Total messages
        total_row = self.store.query_one(
            "SELECT COUNT(*) as count FROM messages WHERE user_id = ?", [user_id]
        )
        total_messages = total_row["count"] if total_row else 0

        # Messages by role
        role_rows = self.store.query_all(
            """
            SELECT role, COUNT(*) as count
            FROM messages
            WHERE user_id = ?
            GROUP BY role
            """,
            [user_id],
        )
        messages_by_role = {row["role"]: row["count"] for row in role_rows}

        # Conversations count
        conv_row = self.store.query_one(
            """
            SELECT COUNT(DISTINCT conversation_id) as count
            FROM messages
            WHERE user_id = ? AND conversation_id IS NOT NULL
            """,
            [user_id],
        )
        conversations_count = conv_row["count"] if conv_row else 0

        # First and last message timestamps
        first_row = self.store.query_one(
            """
            SELECT timestamp
            FROM messages
            WHERE user_id = ?
            ORDER BY timestamp ASC
            LIMIT 1
            """,
            [user_id],
        )
        last_row = self.store.query_one(
            """
            SELECT timestamp
            FROM messages
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            [user_id],
        )

        return {
            "total_messages": total_messages,
            "messages_by_role": messages_by_role,
            "conversations_count": conversations_count,
            "first_message": first_row["timestamp"] if first_row else None,
            "last_message": last_row["timestamp"] if last_row else None,
        }

    def delete_user_messages(self, user_id: str) -> int:
        """Delete all messages for a user."""
        # Get message IDs to remove from FAISS
        rows = self.store.query_all(
            "SELECT message_id FROM messages WHERE user_id = ?", [user_id]
        )
        message_ids_to_remove = {row["message_id"] for row in rows}

        # Remove from FAISS index (rebuild without these messages)
        new_message_ids = []
        new_embeddings = []

        for i, msg_id in enumerate(self.message_ids):
            if msg_id not in message_ids_to_remove:
                new_message_ids.append(msg_id)
                # Get embedding from index
                embedding = self.index.reconstruct(i)
                new_embeddings.append(embedding)

        # Rebuild FAISS index
        if new_embeddings:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            embeddings_array = np.array(new_embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.message_ids = new_message_ids

        # Delete from SQLite
        cur = self.store.execute("DELETE FROM messages WHERE user_id = ?", [user_id])

        # Save updated vectors
        self._save_vectors()

        return int(cur.rowcount or 0)

    def close(self) -> None:
        """Close the conversation memory and save vectors."""
        self._save_vectors()
        self.store.close()
