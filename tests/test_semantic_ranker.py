from tool_attention_mcp.models import ToolSpec
from tool_attention_mcp.scorer_semantic import rank_tools_semantic_tfidf


def test_semantic_ranker_prefers_close_tool():
    tools = [
        ToolSpec(id="run_tests", name="Run Tests", description="Run unit tests", tags=["test"]),
        ToolSpec(id="search_web", name="Search Web", description="Search internet", tags=["web"]),
    ]
    out = rank_tools_semantic_tfidf("please run unit tests", tools, top_k=2)
    assert out[0].id == "run_tests"
