# 🧠 Cortex — Conversation Memory System

Cortex is a **local-first conversation memory system** that ingests and stores all conversation messages with vector embeddings for semantic search. Built with SQLite and FAISS for full local operation.

---

## ✨ Core Features

1. **Local-First Storage** — All data stored locally using SQLite + FAISS
2. **Vector Search** — Semantic similarity search using sentence transformers
3. **User-Based Organization** — All conversations organized by user_id
4. **Rich Metadata** — Support for conversation grouping, roles, and custom metadata
5. **Simple API** — Easy-to-use Python interface and CLI tools

---

## 🚀 Quick Start

### Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# The system will automatically download the sentence transformer model on first use
```

### Basic Usage

```python
from memory.conversation import ConversationMemory, Message
from datetime import datetime
import uuid

# Initialize conversation memory
memory = ConversationMemory()

# Add a message
message = Message(
    user_id="user123",
    message_id=str(uuid.uuid4()),
    content="Hello! I'm interested in machine learning.",
    role="user",
    timestamp=datetime.utcnow().isoformat()
)

message_id = memory.add_message(message)
print(f"Added message: {message_id}")

# Search for similar messages
similar_results = memory.search_similar("user123", "artificial intelligence", limit=5)
for msg, score in similar_results:
    print(f"[Score: {score:.3f}] {msg.content}")

memory.close()
```

### CLI Usage

```bash
# Add a single message
python cli.py add user123 "Hello, how are you?" --role user

# Add a conversation from JSON file
python cli.py add-conversation user123 sample_conversation.json

# Get conversation messages
python cli.py get user123 --limit 10

# Search for similar messages
python cli.py search-similar user123 "machine learning" --limit 5

# Search by content
python cli.py search-content user123 "AI" --limit 10

# Get user statistics
python cli.py stats user123
```

---

## 📁 Project Structure

```
cortex/
├── __init__.py                 # Root package init
├── cli.py                     # Command-line interface
├── demo.py                    # Demo script
├── example_conversation.py    # Example script
├── sample_conversation.json   # Sample conversation data
├── pyproject.toml            # Dependencies & configuration
├── README.md                 # This documentation
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
└── memory/
    ├── __init__.py          # Memory package init
    ├── conversation.py      # Main conversation memory system
    └── store.py            # SQLite storage wrapper
```

---

## 🔧 Configuration

### Database and Vector Storage

```python
# Custom database path and vector directory
memory = ConversationMemory(
    db_path="my_conversations.db",
    vector_dir="my_vectors"
)
```

### Embedding Model

The system uses `all-MiniLM-L6-v2` by default (384-dimensional embeddings). The model is automatically downloaded on first use.

---

## 📊 Data Model

### Message Structure

```python
@dataclass
class Message:
    user_id: str                    # User identifier
    message_id: str                 # Unique message ID
    content: str                    # Message text
    role: str                       # Role (user, assistant, system, etc.)
    timestamp: str                  # ISO-8601 timestamp
    conversation_id: Optional[str]  # Optional conversation grouping
    metadata_json: Optional[str]    # JSON string for custom metadata
```

### Storage

- **SQLite**: Stores message metadata, content, and relationships
- **FAISS**: Stores vector embeddings for semantic search
- **Local Files**: Vector indices and message ID mappings

---

## 🔍 Search Capabilities

### Vector Similarity Search

```python
# Find messages semantically similar to query
results = memory.search_similar("user123", "neural networks", limit=10)
for message, similarity_score in results:
    print(f"Score: {similarity_score:.3f} - {message.content}")
```

### Content Search

```python
# Find messages containing specific text
results = memory.search_by_content("user123", "machine learning", limit=50)
for message in results:
    print(f"{message.role}: {message.content}")
```

### Conversation Retrieval

```python
# Get all messages for a user
messages = memory.get_conversation("user123", limit=100)

# Get messages from a specific conversation
messages = memory.get_conversation("user123", conversation_id="conv_123", limit=50)
```

---

## 📈 Statistics and Analytics

```python
# Get comprehensive user statistics
stats = memory.get_user_stats("user123")
print(f"Total messages: {stats['total_messages']}")
print(f"Messages by role: {stats['messages_by_role']}")
print(f"Conversations: {stats['conversations_count']}")
print(f"First message: {stats['first_message']}")
print(f"Last message: {stats['last_message']}")
```

---

## 🧪 Testing and Examples

### Run the Demo

```bash
python demo.py
```

This demonstrates:
- Adding conversation messages
- Vector similarity search
- Content-based search
- User statistics

### Run the Full Example

```bash
python example_conversation.py
```

This shows:
- Multiple conversation scenarios
- Cross-conversation search
- Advanced usage patterns

### Test with Sample Data

```bash
# Add the sample conversation
python cli.py add-conversation user456 sample_conversation.json

# Search for similar content
python cli.py search-similar user456 "recommendation systems" --limit 3

# Get statistics
python cli.py stats user456
```

---

## 🔄 Data Management

### Adding Messages

```python
# Single message
memory.add_message(message)

# Multiple messages (more efficient)
memory.add_messages([message1, message2, message3])
```

### Deleting Data

```python
# Delete all messages for a user
deleted_count = memory.delete_user_messages("user123")
print(f"Deleted {deleted_count} messages")
```

### Data Persistence

- SQLite database: `cortex.db` (default)
- Vector storage: `vectors/` directory
- All data persists between sessions

---

## 🛠️ Development

### Dependencies

- `faiss-cpu>=1.7.4` - Vector similarity search
- `sentence-transformers>=2.2.2` - Text embeddings
- `numpy>=1.24.0` - Numerical operations
- `sqlite3` - Built into Python

### Local Development

```bash
# Install in development mode
pip install -e .

# Run tests
python demo.py
python example_conversation.py

# Use CLI
python cli.py --help
```

---

## 📝 License

MIT License - see LICENSE file for details.

---

## 🤝 Contributing

This is Phase 1 of the Cortex conversation memory system. The repository has been cleaned to focus solely on conversation ingestion with local vector storage.
