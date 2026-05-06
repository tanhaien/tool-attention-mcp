from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from .adapter_tool_attention import ToolAttentionAdapter
from .models import PickResponse, RankRequest, RankResponse, RankedTool, from_rank_request
from .scorer_fallback import rank_tools_fallback
from .scorer_ollama import rank_tools_ollama
from .scorer_semantic import rank_tools_semantic_tfidf


class ToolAttentionService:
    def __init__(
        self,
        use_tool_attention: bool = True,
        threshold: float = 0.28,
        top_k_default: int = 5,
        vendor_path: str | None = None,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.top_k_default = top_k_default
        self.use_tool_attention = use_tool_attention
        self.threshold = threshold
        self.vendor_path = vendor_path
        self.encoder_name = encoder_name
        self._adapter: ToolAttentionAdapter | None = None

    @property
    def adapter(self) -> ToolAttentionAdapter | None:
        if not self.use_tool_attention:
            return None
        if self._adapter is None:
            self._adapter = ToolAttentionAdapter(
                threshold=self.threshold,
                top_k=max(10, self.top_k_default),
                vendor_path=self.vendor_path,
                encoder_name=self.encoder_name,
            )
        return self._adapter

    def rank_tools(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = from_rank_request(payload)
        return asdict(self._rank(req))

    def pick_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = from_rank_request(payload)
        ranked = self._rank(req)
        best = ranked.ranked_tools[0]
        out = PickResponse(tool=best, alternatives=ranked.ranked_tools[1:3], model_info=ranked.model_info)
        return asdict(out)

    def explain_ranking(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = from_rank_request(payload)
        ranked = self._rank(req)
        reasons = [{"id": r.id, "reason": r.reason, "score": r.score} for r in ranked.ranked_tools]
        return {
            "query": req.query,
            "reasons": reasons,
            "model_info": ranked.model_info,
        }

    def health(self) -> dict[str, Any]:
        semantic_available = self._semantic_available()
        adapter = self.adapter
        return {
            "status": "ok",
            "tool_attention_enabled": self.use_tool_attention,
            "tool_attention_available": (adapter.available if adapter else False),
            "tool_attention_error": (adapter.error if adapter else "disabled"),
            "semantic_tfidf_available": semantic_available,
            "version": "0.1.0",
        }

    def _semantic_available(self) -> bool:
        try:
            import sklearn  # noqa: F401

            return True
        except Exception:
            return False

    def _rank(self, req: RankRequest) -> RankResponse:
        t0 = time.time()

        backend = "fallback"
        ranked: list[RankedTool]
        adapter = self.adapter

        if self.use_tool_attention and adapter and adapter.available:
            try:
                ranked = adapter.rank(req.query, req.tools, top_k=req.top_k)
                backend = "tool_attention"
            except Exception as exc:
                try:
                    ranked = rank_tools_ollama(req.query, req.tools, req.top_k)
                    backend = f"ollama_after_error:{type(exc).__name__}"
                except Exception:
                    try:
                        ranked = rank_tools_semantic_tfidf(req.query, req.tools, req.top_k)
                        backend = f"semantic_tfidf_after_error:{type(exc).__name__}"
                    except Exception:
                        ranked = rank_tools_fallback(req.query, req.tools, req.top_k)
                        backend = f"fallback_after_error:{type(exc).__name__}"
        else:
            try:
                ranked = rank_tools_ollama(req.query, req.tools, req.top_k)
                backend = "ollama"
            except Exception:
                try:
                    ranked = rank_tools_semantic_tfidf(req.query, req.tools, req.top_k)
                    backend = "semantic_tfidf"
                except Exception:
                    ranked = rank_tools_fallback(req.query, req.tools, req.top_k)
                    backend = "fallback"

        if not ranked:
            ranked = rank_tools_fallback(req.query, req.tools, req.top_k)
            backend = "fallback_for_empty"

        return RankResponse(
            ranked_tools=ranked[: req.top_k or self.top_k_default],
            model_info={"backend": backend, "version": "0.1.0"},
            latency_ms=int((time.time() - t0) * 1000),
        )
