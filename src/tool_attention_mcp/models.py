from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Confidence = Literal["low", "medium", "high"]


@dataclass
class ToolSpec:
    id: str
    name: str
    description: str = ""
    inputs_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class RankRequest:
    query: str
    tools: list[ToolSpec]
    top_k: int = 5
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedTool:
    id: str
    score: float
    reason: str
    confidence: Confidence = "medium"


@dataclass
class RankResponse:
    ranked_tools: list[RankedTool]
    model_info: dict[str, Any]
    latency_ms: int


@dataclass
class PickResponse:
    tool: RankedTool
    alternatives: list[RankedTool]
    model_info: dict[str, Any]


def from_json_toolspec(payload: dict[str, Any]) -> ToolSpec:
    return ToolSpec(
        id=str(payload.get("id", "")),
        name=str(payload.get("name", payload.get("id", ""))),
        description=str(payload.get("description", "")),
        inputs_schema=dict(payload.get("inputs_schema", {}) or {}),
        tags=list(payload.get("tags", []) or []),
    )


def from_rank_request(payload: dict[str, Any]) -> RankRequest:
    query = str(payload.get("query", "")).strip()
    tools = [from_json_toolspec(x) for x in (payload.get("tools") or [])]
    top_k = int(payload.get("top_k", 5))
    context = dict(payload.get("context", {}) or {})
    if not query:
        raise ValueError("query is required")
    if not tools:
        raise ValueError("tools must be a non-empty list")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    return RankRequest(query=query, tools=tools, top_k=top_k, context=context)


def to_json(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_json(v) for v in obj]
    return obj
