"""
run.py

Launcher script for the Chainlit UI.
"""

import os
import sys
import logging
from ui.ui_config import UIConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Launch the Chainlit UI."""
    try:
        # Load configuration
        config = UIConfig()
        logger.info("UI configuration loaded")

        # Set environment variables for Chainlit
        os.environ["CHAINLIT_HOST"] = config.host
        os.environ["CHAINLIT_PORT"] = str(config.port)

        # Import and run Chainlit
        from chainlit.cli import run_chainlit

        # Get the path to app.py
        app_path = os.path.join(os.path.dirname(__file__), "app.py")

        # Run Chainlit
        run_chainlit(app_path)

    except Exception as e:
        logger.error(f"Failed to launch UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
