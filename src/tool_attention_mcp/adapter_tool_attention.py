from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

from .models import RankedTool, ToolSpec


class ToolAttentionAdapter:
    """Best-effort adapter around the reference tool-attention implementation.

    Falls back to lexical scorer if dependencies are unavailable.
    """

    def __init__(
        self,
        threshold: float = 0.28,
        top_k: int = 10,
        vendor_path: str | None = None,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.threshold = threshold
        self.top_k = top_k
        self.vendor_path = vendor_path
        self.encoder_name = encoder_name

        self.available = False
        self._err: str | None = None
        self._store_cls: Any = None
        self._router_cls: Any = None

        self._init_backend()

    @property
    def error(self) -> str | None:
        return self._err

    def _init_backend(self) -> None:
        try:
            if importlib.util.find_spec("sentence_transformers") is None:
                self._err = "sentence_transformers not installed"
                return
            base = (
                Path(self.vendor_path)
                if self.vendor_path
                else Path(__file__).resolve().parents[2] / ".vendor" / "tool-attention" / "code"
            )
            vector_store_path = base / "vector_store.py"
            intent_router_path = base / "intent_router.py"

            if not vector_store_path.exists() or not intent_router_path.exists():
                self._err = f"vendor code not found under {base}"
                return

            vector_store = _load_module("vendor_vector_store", vector_store_path)
            intent_router = _load_module("vendor_intent_router", intent_router_path)

            self._store_cls = getattr(vector_store, "ToolVectorStore")
            self._router_cls = getattr(intent_router, "IntentRouter")
            self.available = True
        except Exception as exc:  # pragma: no cover
            self._err = f"adapter init failed: {exc}"
            self.available = False

    def rank(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RankedTool]:
        if not self.available:
            raise RuntimeError(self._err or "tool-attention backend unavailable")

        tool_summaries = [
            {
                "id": t.id,
                "summary": " ".join([t.name, t.description, " ".join(t.tags)]).strip(),
            }
            for t in tools
        ]

        dim = 384
        store = self._store_cls(dim=dim)

        st_module = importlib.import_module("sentence_transformers")
        encoder = st_module.SentenceTransformer(self.encoder_name)
        store.add_tools(tool_summaries, encoder)

        router = self._router_cls(
            store=store,
            encoder=encoder,
            threshold=self.threshold,
            top_k=min(self.top_k, max(top_k, 1)),
        )

        routed = router.route(query)

        reason = "Semantic ISO score from tool-attention intent router"
        ranked = [
            RankedTool(
                id=r.tool_id,
                score=round(float(r.score), 4),
                reason=reason,
                confidence=("high" if r.score >= 0.8 else ("medium" if r.score >= 0.45 else "low")),  # type: ignore[arg-type]
            )
            for r in routed
        ]
        if len(ranked) > top_k:
            ranked = ranked[:top_k]
        return ranked


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
