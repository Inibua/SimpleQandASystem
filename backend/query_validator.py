"""
query_validator.py

Query validation and token counting functionality.
"""

import logging
from typing import Optional


class QueryValidator:
    """Validates user queries before processing."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_query(self, query: str, session_id: str) -> Optional[str]:
        """
        Validate a user query.

        Args:
            query: The user's query string
            session_id: The session ID for logging

        Returns:
            Error message if invalid, None if valid
        """
        if not query or not query.strip():
            return "Query cannot be empty"

        # Check token limit if not disabled
        if not self.config.disable_token_limit:
            token_count = self._count_tokens(query)
            if token_count > self.config.max_query_tokens:
                return f"Query too long ({token_count} tokens, max {self.config.max_query_tokens})"

        # Additional validation can be added here
        if len(query.strip()) < 2:
            return "Query is too short"

        self.logger.info(f"Query validated for session {session_id}: {len(query)} chars")
        return None

    def _count_tokens(self, text: str) -> int:
        """
        Simple token counting (approximate).
        In production, you might want to use a proper tokenizer.
        """
        # Simple word-based token count as approximation
        words = text.split()
        return len(words)
