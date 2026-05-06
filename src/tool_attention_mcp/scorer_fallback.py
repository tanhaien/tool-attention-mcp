from __future__ import annotations

import re

from .models import RankedTool, ToolSpec


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", s.lower()))


def rank_tools_fallback(query: str, tools: list[ToolSpec], top_k: int = 5) -> list[RankedTool]:
    q = _tokenize(query)
    ranked: list[RankedTool] = []
    for t in tools:
        bag = _tokenize(" ".join([t.name, t.description, " ".join(t.tags)]))
        overlap = len(q & bag)
        score = min(0.99, 0.1 + (overlap / max(1, len(q))) * 1.2)
        score = max(0.01, score)
        reason = "Keyword overlap between query and tool metadata"
        confidence = "high" if score >= 0.8 else ("medium" if score >= 0.45 else "low")
        ranked.append(
            RankedTool(
                id=t.id,
                score=round(score, 4),
                reason=reason,
                confidence=confidence,  # type: ignore[arg-type]
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]
