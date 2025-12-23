"""
backend_config.py

Configuration for the backend service including WebSocket, session, and retrieval settings.
"""

from typing import Mapping, Any
from common.base_config import BaseConfig


class BackendConfig(BaseConfig):
    """Configuration for the backend service."""

    _FILENAME = "backend_config.json"

    # required keys → default (None means “no sensible default, must be present”)
    _REQUIRED: Mapping[str, Any] = {
        # --- WebSocket server settings ------------------------------------
        "host": "localhost",
        "port": 8000,
        "websocket_path": "/ws",

        # --- Query validation ---------------------------------------------
        "max_query_tokens": 1000,
        "disable_token_limit": False,

        # --- Session management -------------------------------------------
        "max_conversation_turns": 20,
        "summarization_threshold": 10,

        # --- Retrieval parameters -----------------------------------------
        "top_dense": 10,
        "top_sparse": 10,
        "top_rerank": 5,

        # --- LLM settings -------------------------------------------------
        "llm_model": "llama2",  # Ollama model name
        "llm_host": "localhost",
        "llm_port": 11434,
        "max_tokens": 4000,

        # --- Qdrant settings ----------------------------------------------
        "qdrant_host": "localhost",
        "qdrant_http_port": 6333,
        "qdrant_grpc_port": 6333,
        "collection_name": "simple_kb",
    }
