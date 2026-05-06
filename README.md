# Tool Attention MCP 🧰✨

[![arXiv](https://img.shields.io/badge/arXiv-2604.21816-b31b1b.svg)](https://arxiv.org/abs/2604.21816)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-2024--11--05-purple)](https://modelcontextprotocol.io)

**Drop-in MCP middleware that eliminates the MCP Tax — from 57K tokens/turn down to ~800.**

Tool Attention MCP is a production-ready MCP server that wraps the **[Tool Attention](https://arxiv.org/abs/2604.21816)** algorithm (Sadani, 2026) into a drop-in service for any MCP-compatible agent host — OpenClaw, Claude Desktop, VS Code Cursor, or custom clients.

---

## The Problem: MCP Tax

Every MCP-compatible agent must re-serialize the **full** tool catalog on every conversational turn. With 40–120+ tools across multiple servers, this *MCP Tax* costs **15K–57K tokens per turn** — before your agent even starts thinking.

This means:
- 💸 **API costs skyrocket** (40–60% of spend goes to unused schemas)
- 🧠 **Reasoning degrades** once context utilization crosses ~70%
- 📦 **KV cache inflates**, slowing time-to-first-token
- 🔓 **Larger attack surface** for Tool Poisoning Attacks

---

## The Solution: Tool Attention

Tool Attention replaces flat schema injection with **dynamic, query-aware tool gating** — the "Attention Is All You Need" paradigm applied to tool selection rather than tokens.

### How It Works

```
                    ┌──────────────────┐
 User Query ──────▶ │  Intent Router   │
                    │  (ISO score)     │
                    └──────┬───────────┘
                           │
                    ┌──────▼───────────┐
                    │  State Gating    │
                    │  (preconditions) │
                    └──────┬───────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
     ┌────────▼────────┐      ┌────────▼────────┐
     │  Phase 1:       │      │  Phase 2:       │
     │  Summary Pool   │      │  Schema Loading │
     │  (always cached)│      │  (on-demand)    │
     └────────┬────────┘      └────────┬────────┘
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────────┐
                    │  Agent Prompt    │
                    │  (compact + full │
                    │   schemas for    │
                    │   top-k tools)   │
                    └──────────────────┘
```

Three core components:

1. **Intent–Schema Overlap (ISO)** — embeds the user query and tool summaries into a 384-dim vector space via `all-MiniLM-L6-v2`, then ranks by cosine similarity.

2. **Stateful Gating** — enforces preconditions (auth, workflow state, scopes) before tools are eligible.

3. **Two-Phase Lazy Schema Loading**:
   - **Phase 1 (Summary Pool)** — all N compact tool summaries (~40 tokens each) kept in context; **prompt-cache stable**
   - **Phase 2 (Schema Promotion)** — full JSON schemas only for the top-k gated tools, loaded on-demand

---

## Performance 📊

Measured on the paper's 120-tool, 6-server synthetic benchmark (calibrated to real MCP deployment audits):

| Method | Tokens/Turn | Reduction |
|---|---:|---:|
| Naive Full-Schema Injection | 57,452 | 0.0% |
| Simple Retrieval (top-k schemas) | 5,390 | 90.6% |
| **Tool Attention: Phase 1 only** | **787** | **98.6%** |
| **Tool Attention: Phase 2 only** | **4,672** | **91.9%** |
| Tool Attention: first turn (P1+P2) | 5,459 | 90.5% |

With prompt caching, Phase 1 is cached after the first turn, so steady-state cost is dominated by Phase 2's top-k schemas.

---

## Architecture

```
tool-attention-mcp/
├── src/tool_attention_mcp/     ◄── MCP server (your drop-in service)
│   ├── server.py                JSON-RPC stdio transport
│   ├── service.py               Orchestration + fallback chain
│   ├── adapter_tool_attention.py  Wraps the reference implementation
│   ├── scorer_semantic.py       TF-IDF semantic ranker (fallback)
│   ├── scorer_ollama.py         LLM-based ranker (fallback)
│   ├── scorer_fallback.py       Pure lexical ranker (always available)
│   ├── config.py                Environment-based config
│   └── models.py                Request/response types
├── .vendor/tool-attention/     ◄── Reference implementation (Sadani, 2026)
│   └── code/
│       ├── vector_store.py       FAISS-based summary index
│       ├── intent_router.py      ISO scorer + state-aware gate
│       ├── lazy_loader.py        LRU-cached schema loader
│       ├── tool_attention.py     before_model middleware
│       └── benchmark.py          Token-counting harness
├── examples/                   ◄── Example configs and test data
├── tests/                      ◄── Test suite
├── Dockerfile                  ◄── Dockerized deployment
└── pyproject.toml              ◄── pip-installable package
```

### Fallback Chain 🔁

No single point of failure. If the full Tool Attention stack is unavailable, the server gracefully degrades:

```
Tool Attention (sentence-transformers + FAISS)
    └── LLM-based ranker (Ollama)
        └── Semantic TF-IDF (scikit-learn)
            └── Pure lexical fallback (keyword overlap)
```

---

## Quick Start 🚀

### Installation

```bash
# Clone and install
git clone https://github.com/tanhaien/tool-attention-mcp.git
cd tool-attention-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional: enable full Tool Attention backend
pip install sentence-transformers
```

### Health Check

```bash
tool-attention-cli health
```

### Rank Tools

```bash
tool-attention-cli rank -i examples/rank_input.json
```

### Run the Benchmark

```bash
python scripts_benchmark.py
```

### Run Tests

```bash
pytest -q
```

---

## OpenClaw Configuration 🔧

Add this to your OpenClaw MCP server config:

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

Then in your agent routing policy, invoke `tool_attention.pick_tool` before executing a real tool call.

### Exposed Tools

| Tool | Purpose |
|---|---|
| `tool_attention.rank_tools` | Rank candidate tools by relevance to query |
| `tool_attention.pick_tool` | Pick the single best tool + top alternatives |
| `tool_attention.explain_ranking` | Get reasoning for the ranking |
| `tool_attention.health` | Check backend availability and status |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TA_USE_TOOL_ATTENTION` | `1` | Enable full Tool Attention backend |
| `TA_THRESHOLD` | `0.28` | ISO score threshold for gating |
| `TA_TOP_K_DEFAULT` | `5` | Default number of tools to rank |
| `TA_VENDOR_PATH` | — | Path to reference implementation code |
| `TA_ENCODER` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence encoder model |

---

## Docker 🐳

```bash
docker build -t tool-attention-mcp .
docker run -i tool-attention-mcp
```

The Docker image uses `TA_USE_TOOL_ATTENTION=0` by default (no `sentence-transformers` in the slim image). Extend the image if you need the full backend.

---

## References & Credits 🙏

- **Paper:** *"Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows"* — Anuj Sadani, Deepak Kumar. [arXiv:2604.21816](https://arxiv.org/abs/2604.21816)
- **Reference Implementation:** [github.com/asadani/tool-attention](https://github.com/asadani/tool-attention) (vendored under `.vendor/`)
- **MCP Specification:** [modelcontextprotocol.io](https://modelcontextprotocol.io)

---

## License

MIT — [`LICENSE`](LICENSE)

The vendored reference implementation under `.vendor/tool-attention/` retains its own MIT license from the original authors.
