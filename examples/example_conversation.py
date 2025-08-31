#!/usr/bin/env python3
"""Example usage of the Cortex conversation ingestion system."""

import json
import uuid
from datetime import datetime
from memory.conversation import ConversationMemory, Message


def create_sample_conversation():
    """Create a sample conversation for testing."""
    user_id = "user123"
    conversation_id = str(uuid.uuid4())

    messages = [
        Message(
            user_id=user_id,
            message_id=str(uuid.uuid4()),
            content="Hello! I'm interested in learning about machine learning.",
            role="user",
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
        ),
        Message(
            user_id=user_id,
            message_id=str(uuid.uuid4()),
            content="Great! Machine learning is a fascinating field. What specific area interests you?",
            role="assistant",
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
        ),
        Message(
            user_id=user_id,
            message_id=str(uuid.uuid4()),
            content="I'm particularly interested in natural language processing and neural networks.",
            role="user",
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
        ),
        Message(
            user_id=user_id,
            message_id=str(uuid.uuid4()),
            content="Excellent choices! NLP and neural networks are fundamental to modern AI. Would you like to start with the basics of neural networks?",
            role="assistant",
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
        ),
        Message(
            user_id=user_id,
            message_id=str(uuid.uuid4()),
            content="Yes, please! I'd like to understand how neural networks work.",
            role="user",
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
        ),
    ]

    return messages


def main():
    """Demonstrate the conversation ingestion system."""
    print("Cortex Conversation Ingestion Example")
    print("=" * 50)

    # Initialize conversation memory
    memory = ConversationMemory()

    try:
        # Add sample conversation
        print("1. Adding sample conversation...")
        messages = create_sample_conversation()
        message_ids = memory.add_messages(messages)
        print(f"   Added {len(message_ids)} messages")

        # Get conversation
        print("\n2. Retrieving conversation...")
        user_id = "user123"
        conversation = memory.get_conversation(user_id, limit=10)
        print(f"   Found {len(conversation)} messages:")
        for msg in conversation:
            print(f"   [{msg.role}] {msg.content[:60]}...")

        # Search for similar messages
        print("\n3. Searching for similar messages...")
        similar_results = memory.search_similar(user_id, "neural networks", limit=3)
        print(f"   Found {len(similar_results)} similar messages:")
        for msg, score in similar_results:
            print(f"   [Score: {score:.3f}] {msg.content[:60]}...")

        # Search by content
        print("\n4. Searching by content...")
        content_results = memory.search_by_content(user_id, "machine learning", limit=3)
        print(
            f"   Found {len(content_results)} messages containing 'machine learning':"
        )
        for msg in content_results:
            print(f"   [{msg.role}] {msg.content[:60]}...")

        # Get user statistics
        print("\n5. User statistics...")
        stats = memory.get_user_stats(user_id)
        print(f"   Total messages: {stats['total_messages']}")
        print(f"   Messages by role: {stats['messages_by_role']}")
        print(f"   Conversations: {stats['conversations_count']}")

        # Add another conversation
        print("\n6. Adding another conversation...")
        conversation_id_2 = str(uuid.uuid4())
        messages_2 = [
            Message(
                user_id=user_id,
                message_id=str(uuid.uuid4()),
                content="Can you help me with Python programming?",
                role="user",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id=conversation_id_2,
            ),
            Message(
                user_id=user_id,
                message_id=str(uuid.uuid4()),
                content="Of course! Python is a great language. What specific topic do you need help with?",
                role="assistant",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id=conversation_id_2,
            ),
            Message(
                user_id=user_id,
                message_id=str(uuid.uuid4()),
                content="I'm having trouble with object-oriented programming concepts.",
                role="user",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id=conversation_id_2,
            ),
        ]

        memory.add_messages(messages_2)
        print("   Added 3 more messages")

        # Search across all conversations
        print("\n7. Searching across all conversations...")
        all_similar = memory.search_similar(user_id, "programming", limit=5)
        print(f"   Found {len(all_similar)} messages similar to 'programming':")
        for msg, score in all_similar:
            print(f"   [Score: {score:.3f}] {msg.content[:60]}...")

        # Updated statistics
        print("\n8. Updated statistics...")
        updated_stats = memory.get_user_stats(user_id)
        print(f"   Total messages: {updated_stats['total_messages']}")
        print(f"   Messages by role: {updated_stats['messages_by_role']}")
        print(f"   Conversations: {updated_stats['conversations_count']}")

    finally:
        memory.close()
        print("\nMemory closed successfully.")


if __name__ == "__main__":
    main()
