# Tool Attention MCP 🧰✨

[![arXiv](https://img.shields.io/badge/arXiv-2604.21816-b31b1b.svg)](https://arxiv.org/abs/2604.21816)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-2024--11--05-purple)](https://modelcontextprotocol.io)

**Your AI agent shouldn't blow its entire context window just to know what tools exist.**

Tool Attention MCP is a lightweight middleware server that solves a simple but expensive problem: the **MCP Tax** — where every single message to an AI agent carries the full descriptions of *all* available tools, even when only 2-3 are relevant.

Based on the [Tool Attention paper](https://arxiv.org/abs/2604.21816) (Sadani, 2026). Works with any MCP-compatible agent — OpenClaw, Claude Desktop, VS Code, or your own client.

---

## The Problem: Why Your Agent Is Wasting Tokens

Here's what happens in a typical MCP setup:

**You:** "Hey, can you read the latest sales report?"

**Your agent's brain (the prompt sent to the LLM):**

```
Messages (your query) ...... 500 tokens
+ Tools: read_file ........ 1,200 tokens (full JSON schema)
+ Tools: send_email ....... 1,300 tokens (full JSON schema)
+ Tools: query_database ... 1,400 tokens (full JSON schema)
+ Tools: create_pdf ....... 1,200 tokens (full JSON schema)
+ Tools: deploy_server .... 1,500 tokens (full JSON schema)
+ ... 35 more tools ....... 50,000+ tokens
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~57,000 tokens  ← 99% of it completely irrelevant
```

The agent doesn't need `deploy_server`'s 10-parameter JSON schema just to read a file. But the MCP protocol sends everything, every time, because the APIs are stateless and can't remember what was sent before.

This causes three real problems:

1. **💸 Costs 5-10x more** — you're paying for unused tool descriptions every turn
2. **🧠 Worse reasoning** — the model gets buried in noise, context fills up, and it starts hallucinating tool names and parameters
3. **🔓 Security risk** — malicious tool descriptions can hijack the model's attention ("Tool Poisoning")

---

## The Solution: Only Load What You Need

Tool Attention replaces "shove everything in" with "figure out what's relevant, then load it."

```
┌─────────────────────────────────────────────┐
│  Your Query: "Read the sales report"        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  1. Quick Scan (Intent-Schema Overlap)      │
│     → "read_file" scores 0.92               │
│     → "send_email" scores 0.15              │
│     → "deploy_server" scores 0.02           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  2. Only useful tools make the cut          │
│     ✓ read_file (score > threshold)        │
│     ✗ send_email, query_database, ...      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  3. Send compact prompt to the LLM          │
│     Messages .............. 500 tokens      │
│     All tool summaries .... 700 tokens      │
│     Full schema: read_file  800 tokens      │
│     ─────────────────────────────────       │
│     Total: ~2,000 tokens                    │
│     (vs 57,000 before)                      │
└─────────────────────────────────────────────┘
```

**Result: 98.6% fewer tokens per turn.**

---

## How It Works (In Plain Terms)

### 1. Quick Scan (ISO Score)

Every tool has a **short summary** (~40 words). When you ask a question, Tool Attention converts both your query and each tool summary into "embeddings" — think of them as fingerprints in a 384-dimensional space. Tools that "smell" similar to your query score higher.

***Example:** * You say "list files" → `read_file` scores 0.92, `deploy_server` scores 0.02.

### 2. State Check (Gating)

Even if a tool is relevant, it might not be available right now (needs auth, wrong workflow stage). The gating function checks these preconditions.

### 3. Smart Loading (Two-Phase)

- **Phase 1 (always visible):** Just the short summaries of *all* tools — this stays in context permanently (and gets cached after the first turn, so it costs almost nothing).
- **Phase 2 (on-demand):** Only the full JSON schemas for the top-ranked tools (usually 3-5).

The agent can still *see* all tools it has via Phase 1, but only *loads* the ones it needs.

---

## The Numbers 📊

| Method | Tokens/Turn | Reduction |
|---|---:|---:|
| ❌ Full schemas for 120 tools | 57,452 | 0% |
| 😕 Just grab top-k by keyword | 5,390 | 90.6% |
| ✅ **Tool Attention (everyday use)** | **787** | **98.6%** |
| ✅ Tool Attention (first message only) | 5,459 | 90.5% |

> Phase 1 is cached after the first message. So for messages 2, 3, 4... you only pay ~800 tokens instead of ~5,400.

---

## Quick Start 🚀

### Install

```bash
git clone https://github.com/tanhaien/tool-attention-mcp.git
cd tool-attention-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

And if you want the full Tool Attention backend (recommended):

```bash
pip install sentence-transformers
```

### Check It Works

```bash
tool-attention-cli health
```

### Rank Some Tools

```bash
tool-attention-cli rank -i examples/rank_input.json
```

You'll see output like:

```json
{
  "ranked_tools": [
    {"id": "read_file", "score": 0.92, "confidence": "high"},
    {"id": "list_directory", "score": 0.78, "confidence": "medium"}
  ],
  "model_info": {"backend": "tool_attention"}
}
```

### Run Tests

```bash
pytest -q
```

---

## Project Structure

```
tool-attention-mcp/
├── src/tool_attention_mcp/     ← The MCP server (the thing you use)
│   ├── server.py               Handles JSON-RPC messages over stdin/stdout
│   ├── service.py              Picks the best ranking backend
│   ├── adapter_tool_attention.py  Connects to the reference implementation
│   ├── scorer_semantic.py      TF-IDF ranker (backup #2)
│   ├── scorer_ollama.py        LLM-based ranker (backup #1)
│   ├── scorer_fallback.py      Word-match ranker (always available)
│   ├── config.py               Setup from environment variables
│   └── models.py               Data types used everywhere
├── .vendor/tool-attention/     ← The paper's reference code (read-only)
│   └── code/
│       ├── vector_store.py     FAISS index for tool summaries
│       ├── intent_router.py    The core ranking algorithm
│       └── benchmark.py        Token-counting benchmark
├── examples/                   ← Example configs and test data
├── tests/                      ← Test suite
└── Dockerfile                  ← For container deployment
```

### Graceful Degradation

Even if the full Tool Attention backend fails, the server keeps working:

```
Preferred:    Tool Attention (sentence embeddings + FAISS)
Fallback #1:  LLM-based ranking (Ollama)
Fallback #2:  Semantic TF-IDF (scikit-learn)
Fallback #3:  Keyword matching (always works, no deps)
```

No single point of failure.

---

## MCP Tools Exposed

Once connected to OpenClaw or any MCP host, you get four tools:

| Tool | What It Does |
|---|---|
| `tool_attention.rank_tools` | "Here's my query, rank these tools by relevance" |
| `tool_attention.pick_tool` | "What's the single best tool for this query?" |
| `tool_attention.explain_ranking` | "Why did you pick that tool over this one?" |
| `tool_attention.health` | "Is everything working?" |

### Real Example

```json
{
  "query": "Find all customers from last month",
  "tools": [
    {"id": "query_database", "name": "Query Database", "description": "Run SQL queries on the database", "tags": ["sql", "read"]},
    {"id": "send_email", "name": "Send Email", "description": "Send an email to a recipient", "tags": ["communication"]}
  ],
  "top_k": 2
}
```

→ `pick_tool` returns `query_database` with score 0.95, `send_email` with 0.12.

---

## OpenClaw Configuration

Add this to your OpenClaw MCP servers config:

```json
{
  "mcpServers": {
    "tool-attention": {
      "command": "/path/to/tool-attention-mcp/.venv/bin/tool-attention-mcp",
      "args": [],
      "env": {
        "TA_USE_TOOL_ATTENTION": "1",
        "TA_THRESHOLD": "0.28",
        "TA_TOP_K_DEFAULT": "5",
        "TA_VENDOR_PATH": "/path/to/tool-attention-mcp/.vendor/tool-attention/code"
      }
    }
  }
}
```

Then use `tool_attention.pick_tool` in your agent's routing policy before calling any real tool.

### Environment Variables

| Variable | Default | What It Does |
|---|---|---|
| `TA_USE_TOOL_ATTENTION` | `1` | Turn on full Tool Attention backend (1=on, 0=fallback chain only) |
| `TA_THRESHOLD` | `0.28` | Minimum relevance score for a tool to be considered (0.0 to 1.0) |
| `TA_TOP_K_DEFAULT` | `5` | Default number of top tools to return |
| `TA_VENDOR_PATH` | — | Path to the reference implementation code |
| `TA_ENCODER` | `sentence-transformers/all-MiniLM-L6-v2` | Which embedding model to use for scoring |

---

## Docker

```bash
docker build -t tool-attention-mcp .
docker run -i tool-attention-mcp
```

Note: the slim Docker image skips `sentence-transformers` to keep size small. Extend it if you need the full backend.

---

## Credits 📖

- **Paper:** *"Tool Attention Is All You Need"* — Anuj Sadani, Deepak Kumar. [arXiv:2604.21816](https://arxiv.org/abs/2604.21816)
- **Reference Implementation:** [github.com/asadani/tool-attention](https://github.com/asadani/tool-attention) (vendored under `.vendor/`)
- **MCP Specification:** [modelcontextprotocol.io](https://modelcontextprotocol.io)

---

## License

MIT — see [`LICENSE`](LICENSE).

The vendored reference code (`.vendor/tool-attention/`) is MIT licensed by the original authors.
