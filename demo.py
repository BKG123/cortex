#!/usr/bin/env python3
"""Simple demo of the Cortex conversation ingestion system."""

from memory.conversation import ConversationMemory, Message
from datetime import datetime
import uuid
import json


def main():
    """Demonstrate basic usage of the conversation memory system."""
    print("🧠 Cortex Conversation Memory Demo")
    print("=" * 40)

    # Initialize the conversation memory
    memory = ConversationMemory()

    try:
        # Create a conversation about AI
        conversation_id = str(uuid.uuid4())
        user_id = "demo_user"

        messages = [
            Message(
                user_id=user_id,
                message_id=str(uuid.uuid4()),
                content="What is artificial intelligence?",
                role="user",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id=conversation_id,
            ),
            Message(
                user_id=user_id,
                message_id=str(uuid.uuid4()),
                content="Artificial intelligence (AI) is the simulation of human intelligence in machines. It includes machine learning, natural language processing, and computer vision.",
                role="assistant",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id=conversation_id,
            ),
            Message(
                user_id=user_id,
                message_id=str(uuid.uuid4()),
                content="Can you explain machine learning?",
                role="user",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id=conversation_id,
            ),
            Message(
                user_id=user_id,
                message_id=str(uuid.uuid4()),
                content="Machine learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
                role="assistant",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id=conversation_id,
            ),
        ]

        # Add messages to memory
        print("📝 Adding conversation messages...")
        message_ids = memory.add_messages(messages)
        print(f"✅ Added {len(message_ids)} messages")

        # Search for similar content
        print("\n🔍 Searching for messages similar to 'neural networks'...")
        similar_results = memory.search_similar(user_id, "neural networks", limit=3)
        for msg, score in similar_results:
            print(f"   [Score: {score:.3f}] {msg.content[:80]}...")

        # Search by content
        print("\n🔍 Searching for messages containing 'machine learning'...")
        content_results = memory.search_by_content(user_id, "machine learning", limit=5)
        for msg in content_results:
            print(f"   [{msg.role}] {msg.content[:80]}...")

        # Get user statistics
        print("\n📊 User Statistics:")
        stats = memory.get_user_stats(user_id)
        print(f"   Total messages: {stats['total_messages']}")
        print(f"   Messages by role: {stats['messages_by_role']}")
        print(f"   Conversations: {stats['conversations_count']}")

        # Get conversation
        print(f"\n💬 Conversation {conversation_id}:")
        conversation = memory.get_conversation(user_id, conversation_id=conversation_id)
        for i, msg in enumerate(conversation, 1):
            print(f"   {i}. [{msg.role}] {msg.content}")

    finally:
        memory.close()
        print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
