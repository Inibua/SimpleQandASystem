from typing import Mapping, Any

from common.base_config import BaseConfig


class UIConfig(BaseConfig):
    """Configuration for the Chainlit UI."""

    _FILENAME = "ui.json"

    _REQUIRED: Mapping[str, Any] = {
        "host": "localhost",
        "port": 8001,
        "title": "Knowledge‑Base Q&A",
        "favicon_path": "static/favicon.ico",
        # UI is deliberately minimal; nothing else needed.
    }