from __future__ import annotations

import json
import sys
from typing import Any

from .config import AppConfig
from .service import ToolAttentionService

SERVER_NAME = "tool-attention-mcp"
SERVER_VERSION = "0.1.0"


def _write(msg: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _tool_defs() -> list[dict[str, Any]]:
    request_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "inputs_schema": {"type": "object"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "name"],
                },
            },
            "top_k": {"type": "integer", "minimum": 1},
            "context": {"type": "object"},
        },
        "required": ["query", "tools"],
    }
    return [
        {
            "name": "tool_attention.rank_tools",
            "description": "Rank candidate tools for a user query",
            "inputSchema": request_schema,
        },
        {
            "name": "tool_attention.pick_tool",
            "description": "Pick the best tool for a user query",
            "inputSchema": request_schema,
        },
        {
            "name": "tool_attention.explain_ranking",
            "description": "Explain why tools were ranked in the returned order",
            "inputSchema": request_schema,
        },
        {
            "name": "tool_attention.health",
            "description": "Health and backend status",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


def _as_content(data: Any) -> list[dict[str, str]]:
    return [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]


def _handle_request(service: ToolAttentionService, req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": _tool_defs()},
        }

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            if name == "tool_attention.rank_tools":
                out = service.rank_tools(arguments)
            elif name == "tool_attention.pick_tool":
                out = service.pick_tool(arguments)
            elif name == "tool_attention.explain_ranking":
                out = service.explain_ranking(arguments)
            elif name == "tool_attention.health":
                out = service.health()
            else:
                raise ValueError(f"unknown tool: {name}")

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": _as_content(out),
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": _as_content({"error": str(exc)}),
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    cfg = AppConfig.from_env()
    service = ToolAttentionService(
        use_tool_attention=cfg.use_tool_attention,
        threshold=cfg.threshold,
        top_k_default=cfg.top_k_default,
        vendor_path=cfg.vendor_path,
        encoder_name=cfg.encoder_name,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = _handle_request(service, req)
            if resp is not None:
                _write(resp)
        except Exception as exc:
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
            )


if __name__ == "__main__":
    main()
