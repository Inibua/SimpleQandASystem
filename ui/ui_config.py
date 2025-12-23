"""
config.py

Configuration for the Chainlit UI.
"""

from typing import Mapping, Any
from common.base_config import BaseConfig


class UIConfig(BaseConfig):
    """Configuration for the Chainlit UI."""

    _FILENAME = "ui_config.json"

    # required keys → default (None means "no sensible default, must be present")
    _REQUIRED: Mapping[str, Any] = {
        # --- Chainlit settings -----------------------------------------------
        "host": "localhost",
        "port": 8001,
        "debug": False,

        # --- Backend connection ----------------------------------------------
        "backend_host": "localhost",
        "backend_port": 8000,
        "backend_path": "/ws",
    }
