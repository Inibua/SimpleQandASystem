"""
websocket_server.py

WebSocket server for handling real-time query intake.
"""

import logging
import json
import websockets
from typing import Dict, Any
from websockets.exceptions import ConnectionClosed
from backend.backend_config import BackendConfig
from backend.session_manager import SessionManager
from backend.query_validator import QueryValidator


class WebSocketServer:
    """WebSocket server for handling client connections and queries."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session_manager = SessionManager()
        self.query_validator = QueryValidator(config)
        self.server = None

        # Will be set by retrieval and answer generation components
        self.retrieval_handler = None
        self.answer_generator = None

    async def start_server(self):
        """Start the WebSocket server."""
        self.logger.info(f"Starting WebSocket server on {self.config.host}:{self.config.port}")

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

    async def stop_server(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket server stopped")

    async def handle_connection(self, websocket):
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

        except ConnectionClosed:
            self.logger.info(f"Connection closed for session {session_id}")
        except Exception as e:
            self.logger.error(f"Error handling connection for session {session_id}: {e}")
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Internal server error"
                }))
            except ConnectionClosed:
                pass  # Client already disconnected

    async def handle_message(self, websocket, session_id: str, message: str):
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
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Error processing query"
                }))
            except ConnectionClosed:
                pass  # Client already disconnected

    async def process_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Process a user query (placeholder for now).
        Will be implemented with retrieval and answer generation in Steps 8-9.
        """
        # Placeholder response
        return {
            "type": "response",
            "answer": "This is a placeholder response. Retrieval and answer generation will be implemented in the next steps.",
            "citations": [],
            "session_id": session_id
        }

    def set_retrieval_handler(self, handler):
        """Set the retrieval handler (to be called from main)."""
        self.retrieval_handler = handler

    def set_answer_generator(self, generator):
        """Set the answer generator (to be called from main)."""
        self.answer_generator = generator
