"""
websocket_server.py

Updated WebSocket server with answer generation integration.
"""

import logging
import websockets
import json
from typing import Dict, Any

from websockets import ServerConnection
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
        self.session_manager = SessionManager(config)
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

        self.logger.info(
            f"WebSocket server running on ws://{self.config.host}:{self.config.port}{self.config.websocket_path}")

        # Keep the server running
        await self.server.wait_closed()

    async def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket server stopped")

    async def handle_connection(self, websocket: ServerConnection):
        """Handle a new WebSocket connection."""
        session_id = self.session_manager.create_session()
        client_address = websocket.remote_address
        self.logger.info(f"New connection from {client_address}, session: {session_id}")

        try:
            # Send session creation confirmation
            await websocket.send(json.dumps({
                "type": "session_created",
                "session_id": session_id,
                "status": "connected"
            }))

            # Handle incoming messages
            async for message in websocket:
                await self.handle_message(websocket, session_id, message)

        except Exception as e:
            self.logger.error(f"Error handling connection for session {session_id}: {e}")
            raise e

    async def handle_message(self, websocket: ServerConnection, session_id: str, message: str):
        """Handle an incoming message from a client."""
        try:
            data = json.loads(message)

            if data.get("type") != "query":
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid message type. Expected 'query'."
                }))
                return

            query = data.get("query", "").strip()

            # Validate query
            validation_error = self.query_validator.validate_query(query, session_id)
            if validation_error:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": validation_error
                }))
                return

            # Send acknowledgment
            await websocket.send(json.dumps({
                "type": "processing",
                "message": "Processing your query...",
                "session_id": session_id
            }))

            # Process the query (retrieval and answer generation will be implemented in Steps 8-9)
            response = await self.process_query(session_id, query)

            # Send response
            await websocket.send(json.dumps(response))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON format"
            }))
        except Exception as e:
            self.logger.error(f"Error processing message for session {session_id}: {e}")
            raise e

    async def process_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Process a user query with summarization support.
        """
        try:
            # Get session
            session = self.session_manager.get_session(session_id)
            if not session:
                return {
                    "type": "error",
                    "message": "Session not found",
                    "session_id": session_id
                }

            # Check if we need to summarize the conversation
            summarization_performed = await self.session_manager.maybe_summarize_conversation(session_id)
            if summarization_performed:
                self.logger.info(f"Summarization performed for session {session_id}")

            # Get conversation context (now includes summary)
            conversation_context = session.get_conversation_context()

            # Perform retrieval (still uses recent turns for context, not summary)
            recent_turns_for_retrieval = []
            recent_turns = session.get_recent_context(max_turns=2)
            for turn in recent_turns:
                recent_turns_for_retrieval.append({
                    "question": turn.question,
                    "answer": turn.answer
                })

            retrieved_chunks = await self.retrieval_engine.retrieve(query, recent_turns_for_retrieval)

            # Generate answer with full conversation context (summary + recent turns)
            generation_result = await self.answer_generator.generate_answer(
                query, conversation_context, retrieved_chunks
            )

            # Handle outcomes
            if not generation_result["should_answer"] and generation_result["clarification_question"]:
                return {
                    "type": "clarification",
                    "question": generation_result["clarification_question"],
                    "retrieved_chunks": [],
                    "session_id": session_id,
                    "summarization_performed": summarization_performed
                }
            elif generation_result.get("error"):
                return {
                    "type": "error",
                    "message": generation_result["answer"],
                    "session_id": session_id
                }
            else:
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
                    "session_id": session_id,
                    "summarization_performed": summarization_performed,
                    "session_stats": self.session_manager.get_session_stats(session_id)
                }

        except Exception as e:
            self.logger.error(f"Error processing query for session {session_id}: {e}")
            return {
                "type": "error",
                "message": "Error processing query",
                "session_id": session_id
            }
