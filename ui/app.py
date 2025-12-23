"""
app.py

Minimal Chainlit UI for the Q&A system with WebSocket communication.
"""

import chainlit as cl
import json
import asyncio
import websockets
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend WebSocket configuration
BACKEND_WS_URL = "ws://localhost:8000"


class WebSocketClient:
    """WebSocket client for communicating with the backend."""

    def __init__(self, url: str = BACKEND_WS_URL):
        self.url = url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None

    async def connect(self):
        """Connect to the backend WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.url)
            logger.info(f"Connected to backend at {self.url}")

            # Wait for session creation message
            session_msg = await self.websocket.recv()
            session_data = json.loads(session_msg)

            if session_data.get("type") == "session_created":
                self.session_id = session_data["session_id"]
                logger.info(f"Session created: {self.session_id}")
                return True
            else:
                raise Exception("Unexpected message from server")

        except Exception as e:
            logger.error(f"Failed to connect to backend: {e}")
            return False

    async def send_query(self, query: str) -> dict:
        """Send a query to the backend and wait for response."""
        if not self.websocket or not self.session_id:
            raise Exception("Not connected to backend")

        try:
            # Send query
            message = {
                "type": "query",
                "query": query,
                "session_id": self.session_id
            }
            await self.websocket.send(json.dumps(message))

            # Wait for processing acknowledgment
            processing_msg = await self.websocket.recv()
            processing_data = json.loads(processing_msg)

            if processing_data.get("type") != "processing":
                raise Exception("Unexpected response type")

            # Wait for final response
            response_msg = await self.websocket.recv()
            response_data = json.loads(response_msg)

            return response_data

        except Exception as e:
            logger.error(f"Error sending query: {e}")
            return {"type": "error", "message": f"Communication error: {e}"}

    async def close(self):
        """Close the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.session_id = None


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    # Create WebSocket client
    ws_client = WebSocketClient()

    # Try to connect to backend
    connected = await ws_client.connect()

    if not connected:
        await cl.Message(
            content="❌ Failed to connect to the backend service. Please ensure the backend is running on localhost:8000."
        ).send()
        return

    # Store WebSocket client in user session
    cl.user_session.set("ws_client", ws_client)

    # Send welcome message
    await cl.Message(
        content="🚀 Connected to the Q&A system! You can now ask questions about your documents.\n\n**Features:**\n- Grounded answers with citations\n- Conversation memory with automatic summarization\n- Clarification questions when context is missing"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""
    # Get WebSocket client from session
    ws_client = cl.user_session.get("ws_client")

    if not ws_client or not ws_client.websocket:
        await cl.Message(
            content="❌ Not connected to backend. Please refresh the page."
        ).send()
        return

    # Send thinking indicator
    thinking_msg = cl.Message(content="Processing your question...")
    await thinking_msg.send()

    try:
        # Send query to backend
        response = await ws_client.send_query(message.content)

        # Remove thinking message
        await thinking_msg.remove()

        # Handle different response types
        response_type = response.get("type")

        if response_type == "response":
            await handle_response_message(response)
        elif response_type == "clarification":
            await handle_clarification_message(response)
        elif response_type == "error":
            await handle_error_message(response)
        else:
            await handle_unknown_message(response)

    except Exception as e:
        await thinking_msg.remove()
        await cl.Message(
            content=f"❌ Error processing your message: {str(e)}"
        ).send()


async def handle_response_message(response: dict):
    """Handle a successful response with answer and citations."""
    answer = response.get("answer", "No answer provided.")
    citations = response.get("citations", [])
    chunks_count = response.get("retrieved_chunks_count", 0)
    summarization_performed = response.get("summarization_performed", False)

    # Build message content
    content_parts = [f"{answer}"]

    # Add citations if available
    if citations:
        content_parts.append("\n\n**Citations:**")
        for i, citation in enumerate(citations, 1):
            doc_name = citation.get("document_name", "Unknown")
            page_num = citation.get("page_number", "N/A")
            section = citation.get("section_heading", "")

            citation_text = f"{i}. {doc_name}"
            if page_num != "N/A":
                citation_text += f" (page {page_num})"
            if section:
                citation_text += f" - {section}"

            content_parts.append(citation_text)

    # Add metadata
    metadata_parts = []
    if chunks_count > 0:
        metadata_parts.append(f"Retrieved {chunks_count} relevant chunks")
    if summarization_performed:
        metadata_parts.append("Conversation summarized")

    if metadata_parts:
        content_parts.append(f"\n*{', '.join(metadata_parts)}*")

    # Send the message
    await cl.Message(content="\n".join(content_parts)).send()


async def handle_clarification_message(response: dict):
    """Handle a clarification question when context is missing."""
    question = response.get("question", "Could you provide more details?")
    summarization_performed = response.get("summarization_performed", False)

    content = f"❓ {question}"

    if summarization_performed:
        content += "\n\n*Note: I summarized our previous conversation to focus on the most relevant information.*"

    await cl.Message(content=content).send()


async def handle_error_message(response: dict):
    """Handle error responses from the backend."""
    error_msg = response.get("message", "An unknown error occurred.")
    await cl.Message(content=f"❌ Error: {error_msg}").send()


async def handle_unknown_message(response: dict):
    """Handle unexpected response types."""
    await cl.Message(
        content=f"⚠️ Received unexpected response type: {response.get('type', 'unknown')}"
    ).send()


@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat session ends."""
    ws_client = cl.user_session.get("ws_client")
    if ws_client:
        await ws_client.close()
        logger.info("WebSocket connection closed")


if __name__ == "__main__":
    # This is for running the Chainlit app directly
    # Normally you'd run with: chainlit run app.py
    pass
