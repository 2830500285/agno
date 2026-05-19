"""Test: Does sync agent.run() work with async context provider tools?"""

from agno.agent import Agent
from agno.context import Answer, ContextProvider, Status
from agno.models.openai import OpenAIResponses
from agno.run import RunContext


class SimpleProvider(ContextProvider):
    """Minimal provider that just echoes."""

    def status(self) -> Status:
        return Status(ok=True)

    async def astatus(self) -> Status:
        return Status(ok=True)

    def query(self, question: str, *, run_context: RunContext | None = None) -> Answer:
        return Answer(text=f"sync: {question}")

    async def aquery(self, question: str, *, run_context: RunContext | None = None) -> Answer:
        return Answer(text=f"async: {question}")


def test_sync_agent_with_context_provider():
    """Use sync agent.run() with a context provider — does it break?"""
    provider = SimpleProvider(id="test", stream_sub_agent_events=False)

    agent = Agent(
        model=OpenAIResponses(id="gpt-4o-mini"),
        tools=provider.get_tools(),
        instructions="Use query_test to answer questions.",
    )

    # SYNC run — this is what the reviewer says is broken
    print("Testing sync agent.run()...")
    response = agent.run("What is 2+2?", stream=False)

    print(f"Response type: {type(response)}")
    print(f"Response content: {response.content}")
    print(f"Tool calls: {response.tools}")

    return response


if __name__ == "__main__":
    result = test_sync_agent_with_context_provider()
    print("\n--- RESULT ---")
    print(f"Success: {result.content is not None}")
