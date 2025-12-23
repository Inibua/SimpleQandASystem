"""
main.py

Main backend service that coordinates WebSocket server, session management, and query processing.
"""

import logging
import asyncio
import sys
from backend.backend_config import BackendConfig
from backend.websocket_server import WebSocketServer


class BackendService:
    """Main backend service coordinating all components."""

    def __init__(self):
        self.config = BackendConfig()
        self.logger = logging.getLogger(__name__)
        self.websocket_server = WebSocketServer(self.config)
        self._shutdown_event = asyncio.Event()

    async def run(self):
        """Run the backend service."""
        self.logger.info("Starting backend service")

        # Set up signal handlers for graceful shutdown (cross-platform)
        self._setup_signal_handlers()

        try:
            # Start WebSocket server
            server_task = asyncio.create_task(self.websocket_server.start_server())

            # Wait for shutdown signal or server completion
            await self._shutdown_event.wait()

            # Clean shutdown
            self.logger.info("Initiating graceful shutdown...")
            await self.websocket_server.stop_server()
            server_task.cancel()

            try:
                await server_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            self.logger.error(f"Backend service failed: {e}")
            raise
        finally:
            self.logger.info("Backend service stopped")

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown (cross-platform)."""
        if sys.platform == 'win32':
            # Windows doesn't support add_signal_handler, use a different approach
            self.logger.info("Running on Windows - using alternative shutdown handling")
            # We'll rely on KeyboardInterrupt for Windows
        else:
            # Unix-like systems can use signal handlers
            try:
                import signal
                loop = asyncio.get_running_loop()
                for sig in [signal.SIGINT, signal.SIGTERM]:
                    loop.add_signal_handler(sig, self.initiate_shutdown)
                self.logger.info("Signal handlers set up for graceful shutdown")
            except (ImportError, NotImplementedError) as e:
                self.logger.warning(f"Could not set up signal handlers: {e}")

    def initiate_shutdown(self):
        """Initiate graceful shutdown."""
        self._shutdown_event.set()


async def main():
    """Main entry point."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    service = BackendService()

    try:
        await service.run()
    except KeyboardInterrupt:
        service.initiate_shutdown()
        # Wait a moment for cleanup
        await asyncio.sleep(0.1)
    except Exception as e:
        logging.error(f"Backend service crashed: {e}")


if __name__ == "__main__":
    # Windows-specific: Use ProactorEventLoop for better performance
    if sys.platform == 'win32':
        asyncio.set_event_loop(asyncio.ProactorEventLoop())

    asyncio.run(main())
