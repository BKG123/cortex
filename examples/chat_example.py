#!/usr/bin/env python3
"""Enhanced chat interface using OpenAI and Cortex memory system."""

import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from dotenv import load_dotenv

from memory.conversation import ConversationMemory, Message

load_dotenv()


class EnhancedChatBot:
    """Enhanced chatbot using OpenAI and Cortex memory with advanced memory features."""

    def __init__(self, user_id: str, openai_api_key: Optional[str] = None):
        """Initialize the chatbot.

        Args:
            user_id: Unique identifier for the user
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.user_id = user_id
        self.memory = ConversationMemory()
        self.conversation_id = str(uuid.uuid4())

        # Get OpenAI API key
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter."
            )

        # Initialize OpenAI client
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "OpenAI library required. Install with: pip install openai"
            )

    def get_ai_response(self, user_message: str, context: str = "", memory_context: str = "") -> str:
        """Get response from OpenAI with enhanced context including memory."""
        try:
            # Build comprehensive system message
            system_message = (
                "You are a helpful AI assistant with access to conversation memory. "
                "Use the provided context to give personalized and contextual responses. "
                "If the user asks about previous conversations or topics, reference the memory context.\n\n"
            )
            
            if memory_context:
                system_message += f"Memory Context (from previous conversations):\n{memory_context}\n\n"
            
            if context:
                system_message += f"Current Conversation Context:\n{context}\n\n"
            
            system_message += (
                "Instructions:\n"
                "- Be conversational and remember previous interactions\n"
                "- If the user asks about something from memory, acknowledge it\n"
                "- Provide personalized responses based on conversation history\n"
                "- If you don't have relevant memory context, say so honestly"
            )

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=800,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def add_message(self, content: str, role: str = "user", metadata: Optional[dict] = None) -> str:
        """Add a message to memory with optional metadata."""
        message = Message(
            user_id=self.user_id,
            message_id=str(uuid.uuid4()),
            content=content,
            role=role,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=self.conversation_id,
            metadata_json=json.dumps(metadata) if metadata else None,
        )

        return self.memory.add_message(message)

    def get_recent_context(self, limit: int = 5) -> str:
        """Get recent conversation context for current session."""
        messages = self.memory.get_conversation(
            self.user_id, conversation_id=self.conversation_id, limit=limit
        )

        if not messages:
            return ""

        context_lines = []
        for msg in messages[-limit:]:
            context_lines.append(f"{msg.role}: {msg.content}")

        return "\n".join(context_lines)

    def get_memory_context(self, query: str, limit: int = 3) -> str:
        """Get relevant memory context from previous conversations."""
        # Search for semantically similar messages from previous conversations
        similar_messages = self.memory.search_similar(self.user_id, query, limit=limit)
        
        if not similar_messages:
            return ""
        
        context_lines = ["Relevant memories from previous conversations:"]
        for msg, score in similar_messages:
            if msg.conversation_id != self.conversation_id:  # Only from other conversations
                context_lines.append(f"- [{score:.2f}] {msg.role}: {msg.content}")
        
        return "\n".join(context_lines)

    def search_similar(self, query: str, limit: int = 5) -> List[Tuple[Message, float]]:
        """Search for similar messages in memory."""
        return self.memory.search_similar(self.user_id, query, limit=limit)

    def search_by_content(self, query: str, limit: int = 10) -> List[Message]:
        """Search messages by content using text search."""
        return self.memory.search_by_content(self.user_id, query, limit=limit)

    def get_user_stats(self) -> dict:
        """Get user conversation statistics."""
        return self.memory.get_user_stats(self.user_id)

    def get_conversation_history(self, days: int = 7) -> List[Message]:
        """Get conversation history from the last N days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        # Get all messages from the last N days
        messages = self.memory.get_conversation(self.user_id, limit=1000)
        recent_messages = [
            msg for msg in messages 
            if datetime.fromisoformat(msg.timestamp) > cutoff_date
        ]
        
        return recent_messages

    def chat(self, user_input: str) -> str:
        """Process user input with enhanced memory context."""
        # Add user message to memory
        self.add_message(user_input, "user")

        # Get recent conversation context
        recent_context = self.get_recent_context()

        # Get relevant memory context from previous conversations
        memory_context = self.get_memory_context(user_input)

        # Get AI response with both contexts
        ai_response = self.get_ai_response(user_input, recent_context, memory_context)

        # Add AI response to memory
        self.add_message(ai_response, "assistant")

        return ai_response

    def demonstrate_memory_persistence(self):
        """Demonstrate memory persistence across sessions."""
        print("\n💾 Memory Persistence Demo")
        print("=" * 40)
        
        # Add some sample conversations to demonstrate memory
        sample_conversations = [
            ("What's your favorite programming language?", "I don't have personal preferences, but I can help you with many programming languages like Python, JavaScript, Java, and more."),
            ("Tell me about machine learning", "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions without explicit programming."),
            ("What's the weather like?", "I don't have access to real-time weather data, but I can help you find weather information or explain weather-related concepts."),
            ("Remember my name is Alice", "Nice to meet you, Alice! I'll remember that for our future conversations."),
            ("What did we talk about earlier?", "We discussed programming languages, machine learning, weather, and I learned your name is Alice."),
        ]
        
        print("📝 Adding sample conversations to memory...")
        for user_msg, ai_msg in sample_conversations:
            self.add_message(user_msg, "user")
            self.add_message(ai_msg, "assistant")
            print(f"  Added: {user_msg[:50]}...")
        
        print("✅ Sample conversations added to memory!")

    def close(self):
        """Close the memory system."""
        self.memory.close()


def main():
    """Run the enhanced chat example."""
    print("🧠 Enhanced Cortex + OpenAI Chat Example")
    print("=" * 50)
    print("This example demonstrates advanced memory features:")
    print("• Persistent memory across sessions")
    print("• Semantic search across conversation history")
    print("• Context-aware responses using memory")
    print("• Conversation statistics and analytics")
    print()
    print("Make sure you have set your OPENAI_API_KEY environment variable.")
    print()

    try:
        # Initialize chatbot
        user_id = "enhanced_chat_user"
        chatbot = EnhancedChatBot(user_id)

        print("✅ Enhanced chatbot initialized successfully!")
        
        # Demonstrate memory persistence
        chatbot.demonstrate_memory_persistence()
        
        # Show user stats
        stats = chatbot.get_user_stats()
        print(f"\n📊 Current User Stats:")
        print(f"   Total messages: {stats['total_messages']}")
        print(f"   Messages by role: {stats['messages_by_role']}")
        print(f"   Conversations: {stats['conversations_count']}")
        
        print("\n📚 Available commands:")
        print("   'quit' - Exit the chat")
        print("   'search <query>' - Search memory semantically")
        print("   'find <text>' - Search memory by text content")
        print("   'context' - Show recent conversation context")
        print("   'stats' - Show user statistics")
        print("   'history' - Show recent conversation history")
        print("   'memory <query>' - Show relevant memories for a query")
        print()

        # Chat loop
        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    break
                elif user_input.lower() == "context":
                    context = chatbot.get_recent_context()
                    print(f"\n📝 Recent Context:\n{context}\n")
                    continue
                elif user_input.lower() == "stats":
                    stats = chatbot.get_user_stats()
                    print(f"\n📊 User Statistics:")
                    print(f"   Total messages: {stats['total_messages']}")
                    print(f"   Messages by role: {stats['messages_by_role']}")
                    print(f"   Conversations: {stats['conversations_count']}")
                    print(f"   First message: {stats['first_message']}")
                    print(f"   Last message: {stats['last_message']}")
                    print()
                    continue
                elif user_input.lower() == "history":
                    history = chatbot.get_conversation_history(days=1)
                    print(f"\n📚 Recent Conversation History (last 24 hours):")
                    for i, msg in enumerate(history[-10:], 1):  # Show last 10 messages
                        print(f"   {i}. [{msg.role}] {msg.content[:80]}...")
                    print()
                    continue
                elif user_input.lower().startswith("search "):
                    query = user_input[7:]  # Remove 'search ' prefix
                    print(f"\n🔍 Semantic search for: '{query}'")
                    results = chatbot.search_similar(query)
                    if results:
                        print("Found similar messages:")
                        for msg, score in results:
                            print(f"  [{score:.3f}] {msg.content[:80]}...")
                    else:
                        print("No similar messages found.")
                    print()
                    continue
                elif user_input.lower().startswith("find "):
                    query = user_input[5:]  # Remove 'find ' prefix
                    print(f"\n🔍 Text search for: '{query}'")
                    results = chatbot.search_by_content(query)
                    if results:
                        print("Found messages containing the text:")
                        for msg in results:
                            print(f"  [{msg.role}] {msg.content[:80]}...")
                    else:
                        print("No messages found containing that text.")
                    print()
                    continue
                elif user_input.lower().startswith("memory "):
                    query = user_input[7:]  # Remove 'memory ' prefix
                    print(f"\n📚 Memory context for: '{query}'")
                    memory_context = chatbot.get_memory_context(query)
                    if memory_context:
                        print(memory_context)
                    else:
                        print("No relevant memories found for that query.")
                    print()
                    continue
                elif not user_input:
                    continue

                # Get AI response with memory context
                print("🤖 AI: ", end="", flush=True)
                response = chatbot.chat(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()

    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("\n💡 Make sure to:")
        print("   1. Set OPENAI_API_KEY environment variable")
        print("   2. Install openai library: pip install openai")

    finally:
        if "chatbot" in locals():
            chatbot.close()
        print("✅ Enhanced chat session ended.")


if __name__ == "__main__":
    main()
