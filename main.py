"""Backward-compatible CLI entrypoint."""

import asyncio

from research_agent.cli import intelligent_file_router, interactive_chat


if __name__ == "__main__":
    asyncio.run(interactive_chat())

