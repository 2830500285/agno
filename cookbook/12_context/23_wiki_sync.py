"""
Wiki Context Provider — Sync Usage
==================================

Demonstrates using WikiContextProvider with sync agent.run().
The agent auto-expands ContextProvider to sync tools.

Requires: OPENAI_API_KEY
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agno.agent import Agent
from agno.context.wiki import FileSystemBackend, WikiContextProvider
from agno.models.openai import OpenAIResponses

WIKI_PATH = Path(__file__).resolve().parent / "demo-wiki"
if WIKI_PATH.exists():
    shutil.rmtree(WIKI_PATH)
WIKI_PATH.mkdir()
(WIKI_PATH / "README.md").write_text(
    "# Demo Wiki\n\nA tiny wiki for testing sync context providers.\n"
)
(WIKI_PATH / "architecture.md").write_text(
    "# Architecture\n\n"
    "The system uses a three-tier architecture:\n"
    "1. Frontend (React)\n"
    "2. API (FastAPI)\n"
    "3. Database (PostgreSQL)\n"
)

wiki = WikiContextProvider(
    id="wiki",
    backend=FileSystemBackend(path=WIKI_PATH),
    model=OpenAIResponses(id="gpt-5.4-mini"),
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.4"),
    tools=wiki.get_tools(),
    instructions=wiki.instructions(),
    markdown=True,
)


def main() -> None:
    print(f"\nwiki.status() = {wiki.status()}\n")

    prompt = "What is our system architecture? List the tiers."
    print(f"> {prompt}\n")
    agent.print_response(prompt, stream=True)


if __name__ == "__main__":
    main()
