from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    use_tool_attention: bool = True
    threshold: float = 0.28
    top_k_default: int = 5
    vendor_path: str | None = None
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    @staticmethod
    def from_env() -> "AppConfig":
        use = os.getenv("TA_USE_TOOL_ATTENTION", "1").strip().lower() not in {"0", "false", "no", "off"}
        threshold = float(os.getenv("TA_THRESHOLD", "0.28"))
        top_k = int(os.getenv("TA_TOP_K_DEFAULT", "5"))
        vendor_path = os.getenv("TA_VENDOR_PATH") or None
        encoder_name = os.getenv("TA_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
        return AppConfig(
            use_tool_attention=use,
            threshold=threshold,
            top_k_default=top_k,
            vendor_path=vendor_path,
            encoder_name=encoder_name,
        )
