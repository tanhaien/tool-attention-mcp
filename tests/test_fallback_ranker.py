from tool_attention_mcp.scorer_fallback import rank_tools_fallback
from tool_attention_mcp.models import ToolSpec


def test_ranker_returns_ordered_scores():
    tools = [
        ToolSpec(id="read_file", name="Read File", description="Read file content", tags=["filesystem", "read"]),
        ToolSpec(id="edit_file", name="Edit File", description="Edit file content", tags=["filesystem", "write"]),
        ToolSpec(id="send_email", name="Send Email", description="Send emails", tags=["communication"]),
    ]
    out = rank_tools_fallback("read file config before editing", tools, top_k=2)
    assert len(out) == 2
    assert out[0].score >= out[1].score
    ids = [x.id for x in out]
    assert "read_file" in ids
