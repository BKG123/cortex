#!/usr/bin/env python3
"""Example demonstrating Cortex with Pinecone vector database.

This example shows how to use Cortex with Pinecone instead of local FAISS
for vector storage and similarity search.
"""

import os
import uuid
from datetime import datetime
from memory.conversation import ConversationMemory, Message


def main():
    """Demonstrate Pinecone integration with Cortex."""

    # Check if Pinecone credentials are available
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_ENVIRONMENT"):
        print("Pinecone credentials not found. Please set:")
        print("  export PINECONE_API_KEY='your-api-key'")
        print("  export PINECONE_ENVIRONMENT='your-environment'")
        print("\nYou can get these from https://app.pinecone.io/")
        return

    print("🚀 Initializing Cortex with Pinecone vector backend...")

    # Initialize Cortex with Pinecone
    memory = ConversationMemory(
        vector_backend="pinecone",
        index_name="cortex-demo",  # Pinecone index name
        dimension=384,  # Must match your embedding model
        metric="cosine",  # Distance metric
    )

    try:
        # Create sample messages
        messages = [
            Message(
                user_id="user123",
                message_id=str(uuid.uuid4()),
                content="I'm interested in learning about machine learning and AI.",
                role="user",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id="conv_001",
            ),
            Message(
                user_id="user123",
                message_id=str(uuid.uuid4()),
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
                role="assistant",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id="conv_001",
            ),
            Message(
                user_id="user123",
                message_id=str(uuid.uuid4()),
                content="What are some popular machine learning frameworks?",
                role="user",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id="conv_001",
            ),
            Message(
                user_id="user123",
                message_id=str(uuid.uuid4()),
                content="Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
                role="assistant",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id="conv_001",
            ),
            Message(
                user_id="user123",
                message_id=str(uuid.uuid4()),
                content="I also want to learn about deep learning and neural networks.",
                role="user",
                timestamp=datetime.utcnow().isoformat(),
                conversation_id="conv_002",
            ),
        ]

        print(f"📝 Adding {len(messages)} messages to Pinecone...")

        # Add messages to memory
        message_ids = memory.add_messages(messages)
        print(f"✅ Added messages with IDs: {message_ids[:3]}...")

        # Get conversation statistics
        print("\n📊 Getting user statistics...")
        stats = memory.get_user_stats("user123")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Role counts: {stats['role_counts']}")
        print(f"Conversation count: {stats['conversation_count']}")
        print(f"Vector store: {stats['vector_store']['backend']}")
        print(f"Pinecone index: {stats['vector_store']['index_name']}")

        # Search for similar messages
        print("\n🔍 Searching for messages similar to 'artificial intelligence'...")
        similar_results = memory.search_similar(
            "user123", "artificial intelligence", limit=3
        )

        for message, score in similar_results:
            print(f"[{score:.3f}] {message.role}: {message.content[:80]}...")

        # Search by content
        print("\n🔍 Searching for messages containing 'neural'...")
        content_results = memory.search_by_content("user123", "neural", limit=5)

        for message in content_results:
            print(f"{message.role}: {message.content}")

        # Get conversation history
        print("\n💬 Getting conversation history...")
        conversation = memory.get_conversation(
            "user123", conversation_id="conv_001", limit=10
        )

        for message in conversation:
            print(f"[{message.timestamp}] {message.role}: {message.content}")

        print("\n🎉 Pinecone integration demo completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(
            "Make sure your Pinecone credentials are correct and the service is accessible."
        )

    finally:
        memory.close()


if __name__ == "__main__":
    main()
