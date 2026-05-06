from __future__ import annotations

import json
import math
import os
import urllib.request

from .models import RankedTool, ToolSpec


def _embed(text: str, base_url: str, model: str, timeout: float = 10.0) -> list[float]:
    data = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/embeddings",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local Ollama only by config
        payload = json.loads(resp.read().decode("utf-8"))
    emb = payload.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError("Ollama embeddings response missing embedding")
    return [float(x) for x in emb]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def rank_tools_ollama(query: str, tools: list[ToolSpec], top_k: int = 5) -> list[RankedTool]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if not (base_url.startswith("http://127.0.0.1:") or base_url.startswith("http://localhost:")):
        raise RuntimeError("Refusing non-local Ollama base URL")
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    q = _embed(query, base_url, model)
    ranked: list[RankedTool] = []
    for t in tools:
        text = " ".join([t.name, t.description, " ".join(t.tags)]).strip()
        score = max(0.0, min(0.9999, _cos(q, _embed(text, base_url, model))))
        ranked.append(
            RankedTool(
                id=t.id,
                score=round(score, 4),
                reason=f"Semantic cosine from local Ollama embeddings ({model})",
                confidence=("high" if score >= 0.75 else ("medium" if score >= 0.45 else "low")),  # type: ignore[arg-type]
            )
        )
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]
