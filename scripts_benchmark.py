from __future__ import annotations

import json
from pathlib import Path

from tool_attention_mcp.models import from_rank_request
from tool_attention_mcp.scorer_fallback import rank_tools_fallback
from tool_attention_mcp.scorer_semantic import rank_tools_semantic_tfidf


def run(eval_path: Path) -> dict:
    cases = json.loads(eval_path.read_text())
    total = len(cases)

    def eval_backend(fn):
        hit1 = 0
        mrr = 0.0
        details = []
        for c in cases:
            req = from_rank_request({"query": c["query"], "tools": c["tools"], "top_k": 5})
            ranked = fn(req.query, req.tools, 5)
            ids = [r.id for r in ranked]
            exp = c["expected"]
            if ids and ids[0] == exp:
                hit1 += 1
            rr = 0.0
            for i, tid in enumerate(ids, 1):
                if tid == exp:
                    rr = 1.0 / i
                    break
            mrr += rr
            details.append({"query": c["query"], "expected": exp, "pred": ids[:3]})
        return {"hit@1": hit1 / total, "mrr": mrr / total, "details": details}

    fb = eval_backend(rank_tools_fallback)
    sem = eval_backend(rank_tools_semantic_tfidf)
    return {"total": total, "fallback": fb, "semantic_tfidf": sem}


if __name__ == "__main__":
    out = run(Path("examples/eval_cases.json"))
    print(json.dumps(out, ensure_ascii=False, indent=2))
