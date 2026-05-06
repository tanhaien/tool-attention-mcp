from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import RankedTool, ToolSpec


def rank_tools_semantic_tfidf(query: str, tools: list[ToolSpec], top_k: int = 5) -> list[RankedTool]:
    corpus = [query]
    for t in tools:
        corpus.append(" ".join([t.name, t.description, " ".join(t.tags)]).strip())

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vectorizer.fit_transform(corpus)

    qv = mat[0]
    tv = mat[1:]
    sims = cosine_similarity(qv, tv).flatten()

    ranked: list[RankedTool] = []
    for i, t in enumerate(tools):
        score = float(max(0.01, min(0.999, sims[i])))
        conf = "high" if score >= 0.65 else ("medium" if score >= 0.35 else "low")
        ranked.append(
            RankedTool(
                id=t.id,
                score=round(score, 4),
                reason="Semantic similarity (TF-IDF cosine ISO proxy)",
                confidence=conf,  # type: ignore[arg-type]
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:top_k]
