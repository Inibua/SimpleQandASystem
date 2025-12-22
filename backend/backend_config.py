from typing import Mapping, Any

from common.base_config import BaseConfig


class BackendConfig(BaseConfig):
    """Configuration for the backend / LLM orchestration."""

    _FILENAME = "backend_config.json"

    _REQUIRED: Mapping[str, Any] = {
        #--- websocket --------------------------------------------------
        "host": "127.0.0.1",
        "port": 8000,
        "allow_origins": ["*"],          # Chainlit will enforce CORS later
        #--- LLM ---------------------------------------------------------
        "model_name": "llama3:8b",
        "max_input_tokens": 2048,
        "max_output_tokens": 512,
        "temperature": 0.0,
        #--- retrieval ---------------------------------------------------
        "top_dense": 10,
        "top_sparse": 10,
        "top_rerank": 5,
        #--- conversation handling ---------------------------------------
        "conversation_max_turns": 12,    # after this we summarise older turns
        "summary_prompt": "Summarize the following conversation briefly, preserving facts:",
        #--- citation enforcement -----------------------------------------
        "require_citations": True,
        #--- optional behaviour -----------------------------------------
        "disable_input_token_limit": False,
    }
