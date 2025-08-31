#!/usr/bin/env python3
"""Example chat interface using OpenAI and Cortex memory system."""

import os
import uuid
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from memory.conversation import ConversationMemory, Message

load_dotenv()


class ChatBot:
    """Simple chatbot using OpenAI and Cortex memory."""

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

    def get_ai_response(self, user_message: str, context: str = "") -> str:
        """Get response from OpenAI based on user message and context."""
        try:
            # Build system message with context
            system_message = (
                "You are a helpful AI assistant. Use the conversation context "
                "below to provide relevant and contextual responses.\n\n"
                f"Conversation Context:\n{context}"
            )

            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_completion_tokens=500,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def add_message(self, content: str, role: str = "user") -> str:
        """Add a message to memory and return message ID."""
        message = Message(
            user_id=self.user_id,
            message_id=str(uuid.uuid4()),
            content=content,
            role=role,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=self.conversation_id,
        )

        return self.memory.add_message(message)

    def get_context(self, limit: int = 5) -> str:
        """Get recent conversation context for AI."""
        messages = self.memory.get_conversation(
            self.user_id, conversation_id=self.conversation_id, limit=limit
        )

        if not messages:
            return ""

        context_lines = []
        for msg in messages[-limit:]:  # Get last N messages
            context_lines.append(f"{msg.role}: {msg.content}")

        return "\n".join(context_lines)

    def search_similar(self, query: str, limit: int = 3) -> list:
        """Search for similar messages in memory."""
        return self.memory.search_similar(self.user_id, query, limit=limit)

    def chat(self, user_input: str) -> str:
        """Process user input and return AI response."""
        # Add user message to memory
        self.add_message(user_input, "user")

        # Get conversation context
        context = self.get_context()

        # Get AI response
        ai_response = self.get_ai_response(user_input, context)

        # Add AI response to memory
        self.add_message(ai_response, "assistant")

        return ai_response

    def close(self):
        """Close the memory system."""
        self.memory.close()


def main():
    """Run the chat example."""
    print("🤖 Cortex + OpenAI Chat Example")
    print("=" * 40)
    print("This example demonstrates using Cortex memory with OpenAI for chat.")
    print("Make sure you have set your OPENAI_API_KEY environment variable.")
    print()

    try:
        # Initialize chatbot
        user_id = "chat_user"
        chatbot = ChatBot(user_id)

        print("✅ Chatbot initialized successfully!")
        print("💡 Type 'quit' to exit, 'search <query>' to search memory")
        print("💡 Type 'context' to see recent conversation")
        print()

        # Chat loop
        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    break
                elif user_input.lower() == "context":
                    context = chatbot.get_context()
                    print(f"\n📝 Recent Context:\n{context}\n")
                    continue
                elif user_input.lower().startswith("search "):
                    query = user_input[7:]  # Remove 'search ' prefix
                    print(f"\n🔍 Searching for: '{query}'")
                    results = chatbot.search_similar(query)
                    if results:
                        print("Found similar messages:")
                        for msg, score in results:
                            print(f"  [{score:.3f}] {msg.content[:80]}...")
                    else:
                        print("No similar messages found.")
                    print()
                    continue
                elif not user_input:
                    continue

                # Get AI response
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
        print("✅ Chat session ended.")


if __name__ == "__main__":
    main()
