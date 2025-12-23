"""
websocket_server.py

Updated WebSocket server with answer generation integration.
"""

import logging
import websockets
from typing import Dict, Any
from backend.backend_config import BackendConfig
from backend.session_manager import SessionManager
from backend.query_validator import QueryValidator
from backend.retrieval_engine import RetrievalEngine
from backend.answer_generator import AnswerGenerator  # Added import


class WebSocketServer:
    """WebSocket server with complete answer generation pipeline."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session_manager = SessionManager()
        self.query_validator = QueryValidator(config)

        # Initialize retrieval and answer generation
        self.retrieval_engine = RetrievalEngine(config)
        self.answer_generator = AnswerGenerator(config)
        self.server = None

    async def start_server(self):
        """Start the WebSocket server."""
        self.logger.info(f"Starting WebSocket server on {self.config.host}:{self.config.port}")

        # Health checks
        if not self.retrieval_engine.health_check():
            self.logger.error("Retrieval engine health check failed")
            raise RuntimeError("Retrieval engine not ready")

        if not self.answer_generator.health_check():
            self.logger.error("Answer generator health check failed")
            raise RuntimeError("Ollama not available")

        # Create and start the server
        self.server = await websockets.serve(
            self.handle_connection,
            self.config.host,
            self.config.port,
            ping_interval=20,
            ping_timeout=10
        )

        self.logger.info(f"WebSocket server running on ws://{self.config.host}:{self.config.port}{self.config.websocket_path}")

        # Keep the server running
        await self.server.wait_closed()

    # ... (handle_connection and handle_message remain the same) ...

    async def process_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Process a user query with full retrieval and answer generation pipeline.
        """
        try:
            # Get session and conversation context
            session = self.session_manager.get_session(session_id)
            if not session:
                return {
                    "type": "error",
                    "message": "Session not found",
                    "session_id": session_id
                }

            # Build conversation context from recent turns
            conversation_context = []
            recent_turns = session.get_recent_context(max_turns=2)
            for turn in recent_turns:
                conversation_context.append({
                    "question": turn.question,
                    "answer": turn.answer
                })

            # Perform retrieval
            retrieved_chunks = await self.retrieval_engine.retrieve(query, conversation_context)

            # Generate answer with citation enforcement
            generation_result = await self.answer_generator.generate_answer(
                query, conversation_context, retrieved_chunks
            )

            # Handle different outcomes
            if not generation_result["should_answer"] and generation_result["clarification_question"]:
                # Ask clarification question
                return {
                    "type": "clarification",
                    "question": generation_result["clarification_question"],
                    "retrieved_chunks": [],
                    "session_id": session_id
                }
            elif generation_result.get("error"):
                # Error case
                return {
                    "type": "error",
                    "message": generation_result["answer"],
                    "session_id": session_id
                }
            else:
                # Successful answer with citations
                # Add to conversation history
                session.add_turn(
                    query,
                    generation_result["answer"],
                    retrieved_chunks
                )

                return {
                    "type": "response",
                    "answer": generation_result["answer"],
                    "citations": generation_result["citations"],
                    "retrieved_chunks_count": generation_result["retrieved_chunks_count"],
                    "session_id": session_id
                }

        except Exception as e:
            self.logger.error(f"Error processing query for session {session_id}: {e}")
            return {
                "type": "error",
                "message": "Error processing query",
                "session_id": session_id
            }
