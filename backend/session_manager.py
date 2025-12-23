"""
session_manager.py

Enhanced session management with conversation summarization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4
from dataclasses import dataclass, field
import time

from backend.backend_config import BackendConfig


@dataclass
class ConversationTurn:
    """Represents a single question/answer turn in a conversation."""
    question: str
    answer: str
    timestamp: float
    retrieved_chunks: List[Dict] = field(default_factory=list)


@dataclass
class SessionState:
    """Represents the state of a WebSocket session with summarization."""
    session_id: str
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    conversation_summary: str = ""
    created_at: float = field(default_factory=lambda: time.time())
    last_activity: float = field(default_factory=lambda: time.time())
    summarization_count: int = 0

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

    def get_turns_for_summarization(self) -> Tuple[List[ConversationTurn], List[ConversationTurn]]:
        """
        Split conversation into turns to summarize and turns to keep intact.

        Returns:
            Tuple of (turns_to_summarize, turns_to_keep)
        """
        total_turns = len(self.conversation_history)
        if total_turns <= 3:
            return [], self.conversation_history

        # Keep last 3 turns intact, summarize the rest
        cutoff_index = max(0, total_turns - 3)
        turns_to_summarize = self.conversation_history[:cutoff_index]
        turns_to_keep = self.conversation_history[cutoff_index:]

        return turns_to_summarize, turns_to_keep

    def apply_summarization(self, summary: str, turns_to_keep: List[ConversationTurn]) -> None:
        """Apply summarization to the conversation history."""
        self.conversation_summary = summary
        self.conversation_history = turns_to_keep.copy()
        self.summarization_count += 1
        self.last_activity = time.time()

    def get_conversation_context(self) -> Dict[str, Any]:
        """Get the complete conversation context for answer generation."""
        return {
            "summary": self.conversation_summary,
            "recent_turns": [
                {
                    "question": turn.question,
                    "answer": turn.answer
                }
                for turn in self.conversation_history
            ],
            "total_turns_before_summarization": self.summarization_count * 3,  # Approximate
            "summarization_count": self.summarization_count
        }


class ConversationSummarizer:
    """Handles conversation summarization using LLM."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def summarize_conversation(self, turns: List[ConversationTurn]) -> str:
        """Summarize a list of conversation turns using LLM."""
        if not turns:
            return ""

        self.logger.info(f"Summarizing {len(turns)} conversation turns")

        try:
            # Prepare conversation text for summarization
            conversation_text = self._prepare_conversation_text(turns)

            # Generate summary using LLM
            summary = await self._call_summarization_llm(conversation_text)

            self.logger.info("Conversation summarization completed")
            return summary.strip()

        except Exception as e:
            self.logger.error(f"Conversation summarization failed: {e}")
            # Fallback: create a basic summary
            return self._create_fallback_summary(turns)

    def _prepare_conversation_text(self, turns: List[ConversationTurn]) -> str:
        """Prepare conversation text for summarization."""
        conversation_lines = []
        for i, turn in enumerate(turns, 1):
            conversation_lines.append(f"Turn {i}:")
            conversation_lines.append(f"Q: {turn.question}")
            conversation_lines.append(f"A: {turn.answer}")
            conversation_lines.append("")  # Empty line for separation

        return "\n".join(conversation_lines)

    async def _call_summarization_llm(self, conversation_text: str) -> str:
        """Call LLM for conversation summarization."""
        prompt = self._build_summarization_prompt(conversation_text)

        # Use the same Ollama infrastructure as answer generator
        try:
            import requests
            response = requests.post(
                f"http://{self.config.llm_host}:{self.config.llm_port}/api/generate",
                json={
                    "model": self.config.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Slightly higher temperature for summarization
                        "top_p": 0.9,
                        "num_predict": 500  # Shorter output for summaries
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                raise Exception(f"LLM summarization error: {response.status_code}")

        except Exception as e:
            raise Exception(f"LLM call for summarization failed: {e}")

    def _build_summarization_prompt(self, conversation_text: str) -> str:
        """Build prompt for conversation summarization."""
        return f"""Please provide a concise summary of the following conversation. Focus on the main topics discussed, key questions asked, and important information exchanged.

CONVERSATION:
{conversation_text}

Your summary should be brief but capture the essential points. Avoid unnecessary details.

Summary:"""

    def _create_fallback_summary(self, turns: List[ConversationTurn]) -> str:
        """Create a fallback summary when LLM summarization fails."""
        topics = set()
        for turn in turns:
            # Extract simple topics from questions (very basic)
            words = turn.question.lower().split()
            if words:
                # Take first few significant words as topic
                topic = " ".join(words[:4])
                topics.add(topic)

        topics_list = list(topics)[:5]  # Limit to 5 topics
        return f"Conversation covered topics including: {', '.join(topics_list)}."


class SessionManager:
    """Manages WebSocket session states with summarization."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sessions: Dict[str, SessionState] = {}
        self.summarizer = ConversationSummarizer(config)

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

    async def maybe_summarize_conversation(self, session_id: str) -> bool:
        """
        Check if conversation should be summarized and perform summarization if needed.

        Returns:
            True if summarization was performed, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return False

        if not session.should_summarize(self.config.summarization_threshold):
            return False

        self.logger.info(f"Starting summarization for session {session_id}")

        try:
            # Split conversation into parts to summarize and keep
            turns_to_summarize, turns_to_keep = session.get_turns_for_summarization()

            if not turns_to_summarize:
                return False

            # Generate summary
            summary = await self.summarizer.summarize_conversation(turns_to_summarize)

            # Apply summarization to session
            session.apply_summarization(summary, turns_to_keep)

            self.logger.info(f"Summarization completed for session {session_id}. "
                           f"Kept {len(turns_to_keep)} recent turns, summarized {len(turns_to_summarize)} turns.")
            return True

        except Exception as e:
            self.logger.error(f"Summarization failed for session {session_id}: {e}")
            return False

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

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        session = self.get_session(session_id)
        if not session:
            return {}

        return {
            "session_id": session_id,
            "total_turns": len(session.conversation_history),
            "summarization_count": session.summarization_count,
            "has_summary": bool(session.conversation_summary),
            "session_age_seconds": time.time() - session.created_at,
            "last_activity_seconds": time.time() - session.last_activity
        }
