## Cortex (MVP)

Minimal ingestion pipeline wired per `tech_doc.txt`.

### Quickstart

```bash
python -m cortex.cli ingest --user u1 --role user --message "I prefer morning meetings and avoid Friday"
```

# 🧠 OmniMem — Unified Memory Layer for LLM-Powered Apps

OmniMem is an **opinionated, developer-friendly Python library** for **automatic memory management** in LLM-based applications.  
It continuously ingests data (conversations, messages, events, etc.) and **auto-classifies** them into **Long-Term**, **Episodic**, **Semantic**, and **Preference** memory — **without requiring the developer to decide where to store what**.

---

## ✨ Core Principles
1. **Simplicity First** — Minimal, intuitive API. One entrypoint for ingestion, rich structured output.
2. **Zero Config Storage Decisions** — Library decides storage type (episodic, semantic, preference) automatically.
3. **Opinionated Infra** — Uses built-in, pre-selected storage backends (vector DB, graph DB, key-value) to reduce dependency chaos.
4. **Retrieval-First Design** — Optimized for hybrid search (vector + symbolic + filter).
5. **Pluggable, Not Overwhelming** — Defaults that “just work,” with extension points for power users.
6. **LLM-Agnostic** — Works with OpenAI, Anthropic, Mistral, local models, etc.

---

## 🚀 Example Usage

```python
from omnipmem import OmniMem

# Initialize (local dev mode)
mem = OmniMem()

# Ingest new conversation message
mem.ingest({
    "user_id": "u123",
    "content": "Book a meeting with John next Monday",
    "timestamp": "2025-08-10T09:12:00Z"
})

# Retrieve structured memories
memories = mem.retrieve(user_id="u123")

print(memories["preferences"])  # Extracted user preferences
print(memories["episodic"])     # Timeline of past events
print(memories["semantic"])     # Knowledge graph + embeddings
