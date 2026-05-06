from pathlib import Path
import importlib.util


def test_benchmark_runs_and_returns_metrics():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts_benchmark.py"
    spec = importlib.util.spec_from_file_location("scripts_benchmark", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)

    out = mod.run(root / "examples/eval_cases.json")
    assert out["total"] >= 1
    assert "fallback" in out
    assert "semantic_tfidf" in out
    assert 0 <= out["semantic_tfidf"]["hit@1"] <= 1
