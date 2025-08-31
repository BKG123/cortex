#!/usr/bin/env python3
"""Command-line interface for Cortex conversation memory system."""

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from memory.conversation import ConversationMemory, Message


def create_message(
    user_id: str,
    content: str,
    role: str = "user",
    conversation_id: Optional[str] = None,
) -> Message:
    """Create a Message object with current timestamp."""
    return Message(
        user_id=user_id,
        message_id=str(uuid.uuid4()),
        content=content,
        role=role,
        timestamp=datetime.utcnow().isoformat(),
        conversation_id=conversation_id,
    )


def add_message_cmd(args):
    """Add a single message to conversation memory."""
    memory = ConversationMemory(args.db_path, args.vector_dir)

    try:
        message = create_message(
            user_id=args.user_id,
            content=args.content,
            role=args.role,
            conversation_id=args.conversation_id,
        )

        message_id = memory.add_message(message)
        print(f"Added message with ID: {message_id}")

    finally:
        memory.close()


def add_conversation_cmd(args):
    """Add a conversation from a JSON file."""
    memory = ConversationMemory(args.db_path, args.vector_dir)

    try:
        with open(args.file, "r") as f:
            conversation_data = json.load(f)

        messages = []
        conversation_id = args.conversation_id or str(uuid.uuid4())

        for msg_data in conversation_data:
            message = Message(
                user_id=args.user_id,
                message_id=str(uuid.uuid4()),
                content=msg_data["content"],
                role=msg_data.get("role", "user"),
                timestamp=msg_data.get("timestamp", datetime.utcnow().isoformat()),
                conversation_id=conversation_id,
                metadata_json=json.dumps(msg_data.get("metadata", {})),
            )
            messages.append(message)

        message_ids = memory.add_messages(messages)
        print(f"Added {len(message_ids)} messages to conversation {conversation_id}")

    finally:
        memory.close()


def get_conversation_cmd(args):
    """Retrieve conversation messages."""
    memory = ConversationMemory(args.db_path, args.vector_dir)

    try:
        messages = memory.get_conversation(
            user_id=args.user_id, conversation_id=args.conversation_id, limit=args.limit
        )

        print(f"Found {len(messages)} messages:")
        for msg in messages:
            print(f"[{msg.timestamp}] {msg.role}: {msg.content[:100]}...")

    finally:
        memory.close()


def search_similar_cmd(args):
    """Search for similar messages using vector similarity."""
    memory = ConversationMemory(args.db_path, args.vector_dir)

    try:
        results = memory.search_similar(
            user_id=args.user_id, query=args.query, limit=args.limit
        )

        print(f"Found {len(results)} similar messages:")
        for msg, score in results:
            print(f"[Score: {score:.3f}] {msg.role}: {msg.content[:100]}...")

    finally:
        memory.close()


def search_content_cmd(args):
    """Search messages by content using text search."""
    memory = ConversationMemory(args.db_path, args.vector_dir)

    try:
        messages = memory.search_by_content(
            user_id=args.user_id, query=args.query, limit=args.limit
        )

        print(f"Found {len(messages)} messages containing '{args.query}':")
        for msg in messages:
            print(f"[{msg.timestamp}] {msg.role}: {msg.content[:100]}...")

    finally:
        memory.close()


def stats_cmd(args):
    """Get user statistics."""
    memory = ConversationMemory(args.db_path, args.vector_dir)

    try:
        stats = memory.get_user_stats(args.user_id)

        print(f"Statistics for user {args.user_id}:")
        print(f"  Total messages: {stats['total_messages']}")
        print(f"  Conversations: {stats['conversations_count']}")
        print(f"  Messages by role: {stats['messages_by_role']}")
        print(f"  First message: {stats['first_message']}")
        print(f"  Last message: {stats['last_message']}")

    finally:
        memory.close()


def main():
    parser = argparse.ArgumentParser(description="Cortex Conversation Memory CLI")
    parser.add_argument("--db-path", default="cortex.db", help="SQLite database path")
    parser.add_argument(
        "--vector-dir", default="vectors", help="Vector storage directory"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add message command
    add_parser = subparsers.add_parser("add", help="Add a single message")
    add_parser.add_argument("user_id", help="User ID")
    add_parser.add_argument("content", help="Message content")
    add_parser.add_argument("--role", default="user", help="Message role")
    add_parser.add_argument("--conversation-id", help="Conversation ID")
    add_parser.set_defaults(func=add_message_cmd)

    # Add conversation command
    conv_parser = subparsers.add_parser(
        "add-conversation", help="Add conversation from JSON file"
    )
    conv_parser.add_argument("user_id", help="User ID")
    conv_parser.add_argument("file", help="JSON file with conversation data")
    conv_parser.add_argument(
        "--conversation-id", help="Conversation ID (auto-generated if not provided)"
    )
    conv_parser.set_defaults(func=add_conversation_cmd)

    # Get conversation command
    get_parser = subparsers.add_parser("get", help="Get conversation messages")
    get_parser.add_argument("user_id", help="User ID")
    get_parser.add_argument("--conversation-id", help="Conversation ID")
    get_parser.add_argument(
        "--limit", type=int, default=100, help="Maximum number of messages"
    )
    get_parser.set_defaults(func=get_conversation_cmd)

    # Search similar command
    similar_parser = subparsers.add_parser(
        "search-similar", help="Search for similar messages"
    )
    similar_parser.add_argument("user_id", help="User ID")
    similar_parser.add_argument("query", help="Search query")
    similar_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results"
    )
    similar_parser.set_defaults(func=search_similar_cmd)

    # Search content command
    content_parser = subparsers.add_parser(
        "search-content", help="Search messages by content"
    )
    content_parser.add_argument("user_id", help="User ID")
    content_parser.add_argument("query", help="Search query")
    content_parser.add_argument(
        "--limit", type=int, default=50, help="Maximum number of results"
    )
    content_parser.set_defaults(func=search_content_cmd)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get user statistics")
    stats_parser.add_argument("user_id", help="User ID")
    stats_parser.set_defaults(func=stats_cmd)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
