from __future__ import annotations

import argparse
import json
import sys

from .config import AppConfig
from .service import ToolAttentionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-attention ranker CLI")
    parser.add_argument("command", choices=["rank", "pick", "explain", "health"])
    parser.add_argument("--input", "-i", help="Path to JSON payload. Use '-' for stdin", default="-")
    args = parser.parse_args()

    cfg = AppConfig.from_env()
    svc = ToolAttentionService(
        use_tool_attention=cfg.use_tool_attention,
        threshold=cfg.threshold,
        top_k_default=cfg.top_k_default,
        vendor_path=cfg.vendor_path,
        encoder_name=cfg.encoder_name,
    )

    if args.command == "health":
        print(json.dumps(svc.health(), ensure_ascii=False, indent=2))
        return

    if args.input == "-":
        payload = json.load(sys.stdin)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)

    if args.command == "rank":
        out = svc.rank_tools(payload)
    elif args.command == "pick":
        out = svc.pick_tool(payload)
    else:
        out = svc.explain_ranking(payload)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
