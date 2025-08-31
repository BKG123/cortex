# Cortex

A **local-first conversation memory system** that ingests and stores conversation messages with vector embeddings for semantic search. Built with SQLite and FAISS for complete local operation.

## Features

- **Local-First Storage** — All data stored locally using SQLite + FAISS
- **Semantic Search** — Vector similarity search using sentence transformers
- **Multi-User Support** — Organize conversations by user ID
- **Rich Metadata** — Support for conversation grouping, roles, and custom metadata
- **Simple API** — Clean Python interface with CLI tools

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

## Quick Start

```python
from memory.conversation import ConversationMemory, Message
from datetime import datetime
import uuid

# Initialize
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

Main interface for conversation storage and retrieval.

```python
memory = ConversationMemory(
    db_path="cortex.db",      # SQLite database path
    vector_dir="vectors"      # FAISS index directory
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

## Storage

- **SQLite**: Message metadata and relationships
- **FAISS**: Vector embeddings for semantic search
- **Sentence Transformers**: `all-MiniLM-L6-v2` model (384-dimensional embeddings)

All data is stored locally with no external dependencies or cloud services.

## Examples

See the included examples:

```bash
# Basic demonstration
python demo.py

# Comprehensive example
python example_conversation.py

# Sample data
python cli.py add-conversation user456 sample_conversation.json
python cli.py search-similar user456 "recommendation systems"
```

## Project Structure

```
cortex/
├── memory/
│   ├── conversation.py    # Main conversation memory system
│   └── store.py          # SQLite storage layer
├── cli.py                # Command-line interface
├── demo.py              # Basic demonstration
├── example_conversation.py # Comprehensive example
├── sample_conversation.json # Sample data
└── pyproject.toml       # Project configuration
```

## Requirements

- Python 3.8+
- faiss-cpu>=1.7.4
- sentence-transformers>=2.2.2
- numpy>=1.24.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.
