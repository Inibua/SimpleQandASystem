"""
session_manager.py

Session state management for WebSocket connections.
"""

import logging
from typing import Dict, List, Optional
from uuid import uuid4
from dataclasses import dataclass, field
import time


@dataclass
class ConversationTurn:
    """Represents a single question/answer turn in a conversation."""
    question: str
    answer: str
    timestamp: float
    retrieved_chunks: List[Dict] = field(default_factory=list)


@dataclass
class SessionState:
    """Represents the state of a WebSocket session."""
    session_id: str
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())
    last_activity: float = field(default_factory=lambda: time.time())

    def add_turn(self, question: str, answer: str, retrieved_chunks: List[Dict]) -> None:
        """Add a new conversation turn."""
        turn = ConversationTurn(
            question=question,
            answer=answer,
            timestamp=time.time(),
            retrieved_chunks=retrieved_chunks
        )
        self.conversation_history.append(turn)
        self.last_activity = time.time()

    def get_recent_context(self, max_turns: int = 3) -> List[ConversationTurn]:
        """Get the most recent conversation turns."""
        return self.conversation_history[-max_turns:]

    def should_summarize(self, threshold: int) -> bool:
        """Check if conversation should be summarized."""
        return len(self.conversation_history) >= threshold

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation (to be implemented with LLM)."""
        # Placeholder - will be implemented with LLM summarization
        if len(self.conversation_history) <= 3:
            return ""

        # For now, return a simple summary of older turns
        older_turns = self.conversation_history[:-3]
        summary = f"Conversation started with {len(older_turns)} previous turns."
        return summary


class SessionManager:
    """Manages WebSocket session states."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sessions: Dict[str, SessionState] = {}

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid4())
        self.sessions[session_id] = SessionState(session_id=session_id)
        self.logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID."""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = time.time()
        return session

    def cleanup_inactive_sessions(self, max_age_seconds: int = 3600) -> None:
        """Clean up sessions that have been inactive for too long."""
        current_time = time.time()
        inactive_sessions = []

        for session_id, session in self.sessions.items():
            if current_time - session.last_activity > max_age_seconds:
                inactive_sessions.append(session_id)

        for session_id in inactive_sessions:
            del self.sessions[session_id]
            self.logger.info(f"Cleaned up inactive session: {session_id}")

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions
