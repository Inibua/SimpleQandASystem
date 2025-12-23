"""
ollama_client.py

Client for managing Ollama LLM interactions.
"""

import logging
import requests
from typing import List


class OllamaClient:
    """Client for Ollama LLM operations."""

    def __init__(self, host: str = "localhost", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self.logger = logging.getLogger(__name__)

    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Ollama generation failed: {response.status_code}")

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        available_models = self.list_models()
        return any(model_name in model for model in available_models)

    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available."""
        if self.is_model_available(model_name):
            return True

        self.logger.info(f"Pulling model: {model_name}")
        response = requests.post(
            f"{self.base_url}/api/pull",
            json={"name": model_name},
            timeout=300  # 5 minutes for model download
        )

        return response.status_code == 200
