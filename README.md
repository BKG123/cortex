# Cortex

A **flexible conversation memory system** that ingests and stores conversation messages with vector embeddings for semantic search. Built with SQLite and **pluggable vector backends** - use local FAISS for development or cloud vector databases like Pinecone for production.

## Features

- **Flexible Storage** — Choose between local SQLite + FAISS or cloud vector databases
- **Pluggable Backends** — Local FAISS (default), Pinecone, and more coming soon
- **Semantic Search** — Vector similarity search using sentence transformers
- **Multi-User Support** — Organize conversations by user ID
- **Rich Metadata** — Support for conversation grouping, roles, and custom metadata
- **Simple API** — Clean Python interface with CLI tools
- **Zero Breaking Changes** — Existing code works unchanged

## Installation

### From PyPI (Recommended)

```bash
pip install cortex-memory
```

### From Source

```bash
# Clone the repository
git clone https://github.com/BKG123/cortex.git
cd cortex

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e .
```

The sentence transformer model will be automatically downloaded on first use.

### Optional Dependencies

For cloud vector databases like Pinecone:

```bash
pip install cortex-memory[pinecone]
```

## Quick Start

### Local FAISS (Default)

```python
from memory.conversation import ConversationMemory, Message
from datetime import datetime
import uuid

# Initialize with local FAISS (default)
memory = ConversationMemory()

# Add a message
message = Message(
    user_id="user123",
    message_id=str(uuid.uuid4()),
    content="Hello! I'm interested in machine learning.",
    role="user",
    timestamp=datetime.utcnow().isoformat()
)

memory.add_message(message)

# Search for similar messages
results = memory.search_similar("user123", "artificial intelligence", limit=5)
for msg, score in results:
    print(f"[{score:.3f}] {msg.content}")

memory.close()
```

### Pinecone Cloud

```python
import os
os.environ["PINECONE_API_KEY"] = "your-api-key"
os.environ["PINECONE_ENVIRONMENT"] = "your-environment"

memory = ConversationMemory(
    vector_backend="pinecone",
    index_name="cortex-vectors",
    dimension=384
)

# Same API, different backend!
memory.add_message(message)
results = memory.search_similar("user123", "artificial intelligence", limit=5)
```

## CLI Usage

```bash
# Add a message
cortex add user123 "Hello, how are you?" --role user

# Import conversation from JSON
cortex add-conversation user123 sample_conversation.json

# Retrieve messages
cortex get user123 --limit 10

# Semantic search
cortex search-similar user123 "machine learning" --limit 5

# Content search
cortex search-content user123 "AI" --limit 10

# View statistics
cortex stats user123
```

## API Reference

### ConversationMemory

Main interface for conversation storage and retrieval with configurable vector backends.

```python
# Local FAISS (default)
memory = ConversationMemory(
    db_path="cortex.db",      # SQLite database path
    vector_dir="vectors"      # FAISS index directory
)

# Pinecone cloud
memory = ConversationMemory(
    vector_backend="pinecone",
    index_name="cortex-vectors",
    dimension=384,
    metric="cosine"
)

# Custom FAISS settings
memory = ConversationMemory(
    vector_backend="faiss",
    vector_dir="custom_vectors",
    dimension=384,
    metric="euclidean"
)
```

### Message Model

```python
@dataclass
class Message:
    user_id: str                    # User identifier
    message_id: str                 # Unique message ID
    content: str                    # Message content
    role: str                       # Message role (user, assistant, system)
    timestamp: str                  # ISO-8601 timestamp
    conversation_id: Optional[str]  # Optional conversation grouping
    metadata_json: Optional[str]    # JSON string for custom metadata
```

### Core Methods

```python
# Add messages
memory.add_message(message)
memory.add_messages([message1, message2])

# Search
memory.search_similar(user_id, query, limit=10)
memory.search_by_content(user_id, query, limit=50)

# Retrieve
memory.get_conversation(user_id, limit=100)
memory.get_conversation(user_id, conversation_id="conv_123")

# Analytics
memory.get_user_stats(user_id)

# Management
memory.delete_user_messages(user_id)
```

## Vector Backends

Cortex supports multiple vector storage backends for different use cases:

### Local FAISS (Default)
- **Use Case**: Development, testing, small datasets, offline usage
- **Features**: Fast, no external dependencies, full control
- **Limitations**: No metadata filtering, limited scalability

### Pinecone
- **Use Case**: Production, large datasets, team collaboration
- **Features**: Metadata filtering, automatic scaling, team access
- **Requirements**: Pinecone API credentials

### Backend Selection

```python
# Use local FAISS (default)
memory = ConversationMemory()

# Use Pinecone
memory = ConversationMemory(
    vector_backend="pinecone",
    index_name="my-index"
)

# Mix and match
local_memory = ConversationMemory(vector_backend="faiss")
cloud_memory = ConversationMemory(vector_backend="pinecone")
```

### Environment Configuration

```bash
# For Pinecone
export PINECONE_API_KEY="your-api-key-here"
export PINECONE_ENVIRONMENT="your-environment-here"
```

## Storage

- **SQLite**: Message metadata and relationships (local)
- **Vector Backends**: Configurable vector storage
  - **Local FAISS**: Vector embeddings stored locally (default)
  - **Pinecone**: Cloud-hosted vector database
  - **More coming soon**: Weaviate, Qdrant, Chroma
- **Sentence Transformers**: `all-MiniLM-L6-v2` model (384-dimensional embeddings)

Choose your storage strategy: local-only for development, cloud for production, or hybrid approaches.

## Examples

See the included examples in the `examples/` folder:

```bash
# Basic demonstration
python examples/demo.py

# Comprehensive example
python examples/example_conversation.py

# Vector backend examples
python examples/pinecone_example.py      # Pinecone integration
python examples/hybrid_example.py       # Backend comparison

# Chat with OpenAI (requires API key)
export OPENAI_API_KEY="your-api-key-here"
python examples/chat_example.py

# Sample data
cortex add-conversation user456 examples/sample_conversation.json
cortex search-similar user456 "recommendation systems"
```

### Chat Example with OpenAI

The `examples/chat_example.py` demonstrates a complete chat interface using OpenAI and Cortex memory:

```python
from examples.chat_example import ChatBot

# Initialize chatbot
chatbot = ChatBot("user123")

# Chat with memory
response = chatbot.chat("Hello! What did we talk about yesterday?")
print(response)

# Search memory
results = chatbot.search_similar("machine learning")
for msg, score in results:
    print(f"[{score:.3f}] {msg.content}")

chatbot.close()
```

**Features:**
- 🤖 Interactive chat with OpenAI GPT-3.5-turbo
- 🧠 Automatic conversation memory storage
- 🔍 Semantic search through conversation history
- 📝 Context-aware responses using recent conversation
- 💬 Commands: `context`, `search <query>`, `quit`

**Installation:**
```bash
pip install cortex-memory[chat]
# or
pip install cortex-memory openai
```

## Project Structure

```
cortex/
├── memory/
│   ├── conversation.py    # Main conversation memory system
│   ├── store.py          # SQLite storage layer
│   ├── cli.py            # Command-line interface
│   └── vectors/          # Vector store backends
│       ├── __init__.py   # Factory functions
│       ├── base.py       # Abstract interfaces
│       ├── local_faiss.py # Local FAISS implementation
│       └── pinecone_store.py # Pinecone integration
├── examples/
│   ├── demo.py           # Basic demonstration
│   ├── example_conversation.py # Comprehensive example
│   ├── chat_example.py   # OpenAI integration example
│   ├── pinecone_example.py # Pinecone integration demo
│   ├── hybrid_example.py # Backend comparison demo
│   └── sample_conversation.json # Sample data
├── config/               # Configuration examples
│   └── pinecone.yaml    # Pinecone configuration
├── pyproject.toml        # Project configuration
└── README.md            # Documentation
```

## Requirements

- Python 3.8+
- faiss-cpu>=1.7.4
- sentence-transformers>=2.2.2
- numpy>=1.24.0

### Optional Dependencies

For cloud vector databases:

```bash
# Pinecone support
pip install cortex-memory[pinecone]

# All optional dependencies
pip install cortex-memory[chat,pinecone]
```

## Migration & Best Practices

### From Local to Cloud

Start with local storage and migrate to cloud when needed:

```python
# Development: Local FAISS
memory = ConversationMemory()

# Production: Pinecone
memory = ConversationMemory(
    vector_backend="pinecone",
    index_name="production-vectors"
)
```

### Environment-Based Configuration

```python
import os

# Auto-select backend based on environment
if os.getenv("CORTEX_VECTOR_BACKEND") == "pinecone":
    memory = ConversationMemory(vector_backend="pinecone")
else:
    memory = ConversationMemory()  # Default to local
```

### Error Handling & Fallbacks

```python
try:
    memory = ConversationMemory(vector_backend="pinecone")
except Exception as e:
    print(f"Cloud backend failed: {e}")
    # Fallback to local
    memory = ConversationMemory(vector_backend="faiss")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.
