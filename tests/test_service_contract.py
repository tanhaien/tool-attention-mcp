from tool_attention_mcp.service import ToolAttentionService


def _payload():
    return {
        "query": "read config file",
        "tools": [
            {"id": "read_file", "name": "Read File", "description": "Read file content", "tags": ["read"]},
            {"id": "edit_file", "name": "Edit File", "description": "Edit file content", "tags": ["write"]},
        ],
        "top_k": 2,
    }


def test_rank_tools_contract():
    svc = ToolAttentionService(use_tool_attention=False)
    out = svc.rank_tools(_payload())
    assert "ranked_tools" in out
    assert len(out["ranked_tools"]) == 2
    assert out["model_info"]["backend"] in {"semantic_tfidf", "fallback"}


def test_pick_tool_contract():
    svc = ToolAttentionService(use_tool_attention=False)
    out = svc.pick_tool(_payload())
    assert "tool" in out
    assert "id" in out["tool"]


def test_health_contract():
    svc = ToolAttentionService(use_tool_attention=False)
    out = svc.health()
    assert out["status"] == "ok"
    assert "tool_attention_available" in out
