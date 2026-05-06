#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

printf '%s\n' \
'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0.0"}}}' \
'{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"tool_attention.rank_tools","arguments":{ "query":"read file", "tools":[{"id":"read_file","name":"Read File","description":"Read file content","tags":["read"]}], "top_k":1 }}}' \
| tool-attention-mcp
