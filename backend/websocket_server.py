"""
websocket_server.py

WebSocket server with integrated retrieval engine.
"""

import logging
import websockets
from typing import Dict, Any, List
from backend.backend_config import BackendConfig
from backend.session_manager import SessionManager, SessionState
from backend.query_validator import QueryValidator
from backend.retrieval_engine import RetrievalEngine  # Added import


class WebSocketServer:
    """WebSocket server for handling client connections and queries."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session_manager = SessionManager()
        self.query_validator = QueryValidator(config)

        # Initialize retrieval engine
        self.retrieval_engine = RetrievalEngine(config)
        self.server = None

        # Will be set by answer generation component
        self.answer_generator = None

    async def start_server(self):
        """Start the WebSocket server."""
        self.logger.info(f"Starting WebSocket server on {self.config.host}:{self.config.port}")

        # Health check retrieval engine
        if not self.retrieval_engine.health_check():
            self.logger.error("Retrieval engine health check failed")
            raise RuntimeError("Retrieval engine not ready")

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

    # ... (rest of the methods remain the same until process_query) ...

    async def process_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Process a user query with retrieval engine.
        """
        try:
            # Get session and conversation context
            session = self.session_manager.get_session(session_id)
            if not session:
                return {
                    "type": "error",
                    "message": "Session not found"
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

            if not retrieved_chunks:
                # No relevant context found - ask clarification question
                return {
                    "type": "clarification",
                    "question": "I couldn't find relevant information in the documents. Could you provide more specific details or rephrase your question?",
                    "retrieved_chunks": [],
                    "session_id": session_id
                }

            # For now, return retrieved chunks with placeholder answer
            # This will be replaced with actual answer generation in Step 9
            answer = self._generate_placeholder_answer(retrieved_chunks)

            # Add to conversation history
            session.add_turn(query, answer, retrieved_chunks)

            return {
                "type": "response",
                "answer": answer,
                "citations": self._extract_citations(retrieved_chunks),
                "retrieved_chunks": retrieved_chunks,  # For debugging
                "session_id": session_id
            }

        except Exception as e:
            self.logger.error(f"Error processing query for session {session_id}: {e}")
            return {
                "type": "error",
                "message": "Error processing query",
                "session_id": session_id
            }

    def _generate_placeholder_answer(self, retrieved_chunks: List[Dict]) -> str:
        """Generate a placeholder answer showing retrieval worked."""
        if not retrieved_chunks:
            return "No relevant information found."

        # Show that retrieval is working
        doc_names = set(chunk["document_name"] for chunk in retrieved_chunks)
        pages = set(chunk["page_number"] for chunk in retrieved_chunks)

        return f"Retrieval successful! Found {len(retrieved_chunks)} relevant chunks from documents: {', '.join(doc_names)} (pages: {', '.join(map(str, pages))}). Answer generation will be implemented in the next step."

    def _extract_citations(self, retrieved_chunks: List[Dict]) -> List[Dict]:
        """Extract citation information from retrieved chunks."""
        citations = []
        for chunk in retrieved_chunks:
            citations.append({
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "section_heading": chunk.get("section_heading", ""),
                "chunk_id": chunk["chunk_id"]
            })
        return citations

    def set_answer_generator(self, generator):
        """Set the answer generator (to be called from main)."""
        self.answer_generator = generator
