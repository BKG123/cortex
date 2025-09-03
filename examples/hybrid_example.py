#!/usr/bin/env python3
"""Example demonstrating Cortex with different vector backends.

This example shows how to use Cortex with different vector storage backends
for different use cases - local FAISS for development and Pinecone for production.
"""

import os
import uuid
from datetime import datetime
from memory.conversation import ConversationMemory, Message


def demo_local_backend():
    """Demonstrate local FAISS backend (default)."""
    print("🏠 Using Local FAISS Backend (Default)")
    print("=" * 50)

    memory = ConversationMemory(
        vector_backend="faiss",  # Default, can be omitted
        vector_dir="vectors_local",
    )

    try:
        # Add a test message
        message = Message(
            user_id="dev_user",
            message_id=str(uuid.uuid4()),
            content="This is a test message for local development.",
            role="user",
            timestamp=datetime.utcnow().isoformat(),
        )

        message_id = memory.add_message(message)
        print(f"✅ Added message with ID: {message_id}")

        # Search
        results = memory.search_similar("dev_user", "test message", limit=1)
        for msg, score in results:
            print(f"🔍 Found: [{score:.3f}] {msg.content}")

        # Get stats
        stats = memory.get_user_stats("dev_user")
        print(f"📊 Vector store: {stats['vector_store']['backend']}")
        print(f"📊 Total vectors: {stats['vector_store']['total_vectors']}")

    finally:
        memory.close()


def demo_pinecone_backend():
    """Demonstrate Pinecone backend (if credentials available)."""
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_ENVIRONMENT"):
        print("\n☁️ Pinecone Backend (Credentials not available)")
        print("=" * 50)
        print("Set PINECONE_API_KEY and PINECONE_ENVIRONMENT to test Pinecone")
        return

    print("\n☁️ Using Pinecone Backend")
    print("=" * 50)

    memory = ConversationMemory(
        vector_backend="pinecone", index_name="cortex-hybrid-demo", dimension=384
    )

    try:
        # Add a test message
        message = Message(
            user_id="prod_user",
            message_id=str(uuid.uuid4()),
            content="This is a production message stored in Pinecone.",
            role="user",
            timestamp=datetime.utcnow().isoformat(),
        )

        message_id = memory.add_message(message)
        print(f"✅ Added message with ID: {message_id}")

        # Search
        results = memory.search_similar("prod_user", "production message", limit=1)
        for msg, score in results:
            print(f"🔍 Found: [{score:.3f}] {msg.content}")

        # Get stats
        stats = memory.get_user_stats("prod_user")
        print(f"📊 Vector store: {stats['vector_store']['backend']}")
        print(f"📊 Pinecone index: {stats['vector_store']['index_name']}")

    finally:
        memory.close()


def demo_backend_comparison():
    """Compare different backends side by side."""
    print("\n🔄 Backend Comparison")
    print("=" * 50)

    # Test local backend
    local_memory = ConversationMemory(
        vector_backend="faiss", vector_dir="vectors_compare"
    )

    try:
        # Add test data to local
        local_message = Message(
            user_id="compare_user",
            message_id=str(uuid.uuid4()),
            content="This message is stored locally in FAISS.",
            role="user",
            timestamp=datetime.utcnow().isoformat(),
        )

        local_memory.add_message(local_message)
        local_stats = local_memory.get_user_stats("compare_user")
        print(f"🏠 Local FAISS: {local_stats['vector_store']['backend']}")
        print(f"   - Total vectors: {local_stats['vector_store']['total_vectors']}")
        print(f"   - Directory: {local_stats['vector_store']['vector_dir']}")

    finally:
        local_memory.close()

    # Test Pinecone if available
    if os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_ENVIRONMENT"):
        pinecone_memory = ConversationMemory(
            vector_backend="pinecone", index_name="cortex-compare-demo"
        )

        try:
            # Add test data to Pinecone
            pinecone_message = Message(
                user_id="compare_user",
                message_id=str(uuid.uuid4()),
                content="This message is stored in Pinecone cloud.",
                role="user",
                timestamp=datetime.utcnow().isoformat(),
            )

            pinecone_memory.add_message(pinecone_message)
            pinecone_stats = pinecone_memory.get_user_stats("compare_user")
            print(f"☁️ Pinecone: {pinecone_stats['vector_store']['backend']}")
            print(f"   - Index: {pinecone_stats['vector_store']['index_name']}")
            print(
                f"   - Total vectors: {pinecone_stats['vector_store']['total_vector_count']}"
            )

        finally:
            pinecone_memory.close()


def main():
    """Run all backend demonstrations."""
    print("🚀 Cortex Vector Backend Demo")
    print("=" * 60)
    print("This demo shows how to use different vector storage backends")
    print("with Cortex memory system.\n")

    # Demo local backend
    demo_local_backend()

    # Demo Pinecone backend
    demo_pinecone_backend()

    # Compare backends
    demo_backend_comparison()

    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("\nKey takeaways:")
    print("• Local FAISS: Great for development, testing, and small datasets")
    print("• Pinecone: Perfect for production, scaling, and team collaboration")
    print("• Same API: Switch backends without changing your code")
    print("• Hybrid approach: Use local for dev, cloud for production")


if __name__ == "__main__":
    main()
