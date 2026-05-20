"""Unit tests for the ContextProvider ABC.

Focus on the tool-wrapping layer — what callers actually see when
`aquery` / `aupdate` raise, when `aupdate` is not overridden, etc.
These are the contract edges that _query_tool / _update_tool have to
catch before reaching the calling agent.
"""

from __future__ import annotations

import json

import pytest

from agno.context import Answer, ContextMode, ContextProvider, Status
from agno.context.provider import _sanitize_id
from agno.run import RunContext


def _collect_tool_output_sync(tool, **kwargs) -> str:
    """Collect final string output from a sync generator tool."""
    result = ""
    for chunk in tool.entrypoint(**kwargs):
        if isinstance(chunk, str):
            result = chunk
    return result


async def _collect_tool_output_async(tool, **kwargs) -> str:
    """Collect final string output from an async tool.

    Handles both regular async functions and async generators.
    The @tool decorator wraps async generators in a coroutine that
    returns the generator, so we need to await first.
    """
    result = ""
    gen = tool.entrypoint(**kwargs)
    # Await the wrapper coroutine to get the actual generator/result
    inner = await gen
    if hasattr(inner, "__anext__"):
        async for chunk in inner:
            if isinstance(chunk, str):
                result = chunk
    else:
        result = inner
    return result


async def _collect_streaming_chunks(tool, **kwargs) -> list:
    """Collect all chunks from an async streaming tool.

    The @tool decorator wraps async generators in a coroutine,
    so we await first to get the actual generator.
    """
    chunks = []
    gen = tool.entrypoint(**kwargs)
    inner = await gen
    if hasattr(inner, "__anext__"):
        async for chunk in inner:
            chunks.append(chunk)
    else:
        chunks.append(inner)
    return chunks


# ---------------------------------------------------------------------------
# Test fixtures — minimal providers that pass / raise on demand
# ---------------------------------------------------------------------------


class _EchoProvider(ContextProvider):
    """Returns the question back as text. For happy-path exercises."""

    def status(self) -> Status:
        return Status(ok=True, detail="echo")

    async def astatus(self) -> Status:
        return self.status()

    def query(self, question: str, *, run_context: RunContext | None = None) -> Answer:
        return Answer(text=f"q:{question}")

    async def aquery(self, question: str, *, run_context: RunContext | None = None) -> Answer:
        return Answer(text=f"q:{question}")


class _RaisingQueryProvider(_EchoProvider):
    async def aquery(self, question: str, *, run_context: RunContext | None = None) -> Answer:
        raise RuntimeError("aquery boom")


class _WritableProvider(_EchoProvider):
    def update(self, instruction: str, *, run_context: RunContext | None = None) -> Answer:
        return Answer(text=f"u:{instruction}")

    async def aupdate(self, instruction: str, *, run_context: RunContext | None = None) -> Answer:
        return Answer(text=f"u:{instruction}")


class _RaisingWritableProvider(_EchoProvider):
    def update(self, instruction: str, *, run_context: RunContext | None = None) -> Answer:
        raise ValueError("update boom")

    async def aupdate(self, instruction: str, *, run_context: RunContext | None = None) -> Answer:
        raise ValueError("aupdate boom")


# ---------------------------------------------------------------------------
# _sanitize_id
# ---------------------------------------------------------------------------


def test_sanitize_id_normalizes_case_and_punctuation():
    assert _sanitize_id("My-Provider.2") == "my_provider_2"


def test_sanitize_id_empty_input_defaults():
    assert _sanitize_id("!!!") == "context"


# ---------------------------------------------------------------------------
# Tool-name derivation
# ---------------------------------------------------------------------------


def test_default_tool_names_derive_from_id():
    p = _EchoProvider(id="MyThing")
    assert p.query_tool_name == "query_mything"
    assert p.update_tool_name == "update_mything"


def test_explicit_tool_names_override():
    p = _EchoProvider(id="x", query_tool_name="ask_x", update_tool_name="write_x")
    assert p.query_tool_name == "ask_x"
    assert p.update_tool_name == "write_x"


# ---------------------------------------------------------------------------
# get_tools() — mode resolution
# ---------------------------------------------------------------------------


def test_mode_default_returns_default_tools():
    p = _EchoProvider(id="e")
    tools = p.get_tools()
    assert [t.name for t in tools] == ["query_e"]


def test_mode_agent_returns_just_query_tool():
    p = _EchoProvider(id="e", mode=ContextMode.agent)
    tools = p.get_tools()
    assert [t.name for t in tools] == ["query_e"]


def test_mode_tools_returns_all_tools():
    p = _EchoProvider(id="e", mode=ContextMode.tools)
    tools = p.get_tools()
    # base class _all_tools returns [_query_tool()]
    assert [t.name for t in tools] == ["query_e"]


def test_format_tools_expands_toolkit_from_provider():
    """format_tools() should expand Toolkit returned by mode=tools provider."""
    from agno.os.utils import format_tools
    from agno.tools import Toolkit

    class _ToolkitProvider(_EchoProvider):
        """Provider that returns a Toolkit from _all_tools (like FilesystemContextProvider)."""

        def _all_tools(self, async_mode=False):
            toolkit = Toolkit(name="mock_toolkit")

            @toolkit.register
            def read_file(path: str) -> str:
                """Read a file."""
                return f"content of {path}"

            @toolkit.register
            def list_files(directory: str) -> str:
                """List files in directory."""
                return f"files in {directory}"

            return [toolkit]

    p = _ToolkitProvider(id="tk", mode=ContextMode.tools)
    formatted = format_tools([p])

    # Should expand to 2 functions from the toolkit
    assert len(formatted) == 2
    names = {t["name"] for t in formatted}
    assert names == {"read_file", "list_files"}


def test_mode_tools_get_tools_sync_returns_toolkit():
    """get_tools(async_mode=False) with mode=tools returns Toolkit for sync iteration."""
    from agno.tools import Toolkit

    class _ToolkitProvider(_EchoProvider):
        def _all_tools(self, async_mode=False):
            toolkit = Toolkit(name="sync_toolkit")

            @toolkit.register
            def sync_tool(x: str) -> str:
                """Sync tool."""
                return x

            return [toolkit]

    p = _ToolkitProvider(id="s", mode=ContextMode.tools)
    tools = p.get_tools(async_mode=False)
    assert len(tools) == 1
    assert isinstance(tools[0], Toolkit)
    assert "sync_tool" in tools[0].functions


def test_mode_tools_get_tools_async_returns_toolkit():
    """get_tools(async_mode=True) with mode=tools returns Toolkit for async iteration."""
    from agno.tools import Toolkit

    class _ToolkitProvider(_EchoProvider):
        def _all_tools(self, async_mode=False):
            toolkit = Toolkit(name="async_toolkit")

            @toolkit.register
            def sync_tool(x: str) -> str:
                """Tool available in both sync and async mode."""
                return x

            return [toolkit]

    p = _ToolkitProvider(id="a", mode=ContextMode.tools)
    # async_mode=True still returns the toolkit; agent uses get_async_functions()
    tools = p.get_tools(async_mode=True)
    assert len(tools) == 1
    assert isinstance(tools[0], Toolkit)
    assert "sync_tool" in tools[0].functions


def test_mode_agent_returns_query_function_not_toolkit():
    """mode=agent always returns a query Function, never a Toolkit."""
    from agno.tools.function import Function

    class _ToolkitProvider(_EchoProvider):
        def _all_tools(self, async_mode=False):
            from agno.tools import Toolkit

            toolkit = Toolkit(name="ignored")

            @toolkit.register
            def ignored_tool(x: str) -> str:
                return x

            return [toolkit]

    p = _ToolkitProvider(id="ag", mode=ContextMode.agent)
    tools = p.get_tools()
    # mode=agent ignores _all_tools and returns [_query_tool()]
    assert len(tools) == 1
    assert isinstance(tools[0], Function)
    assert tools[0].name == "query_ag"


def test_mode_default_returns_function_not_toolkit():
    """mode=default returns Functions from _default_tools, not raw Toolkit."""
    from agno.tools.function import Function

    p = _EchoProvider(id="d", mode=ContextMode.default)
    tools = p.get_tools()
    assert len(tools) == 1
    assert isinstance(tools[0], Function)
    assert tools[0].name == "query_d"


def test_parse_tools_expands_toolkit_from_context_provider():
    """parse_tools() correctly expands Toolkit from ContextProvider mode=tools."""
    from unittest.mock import MagicMock

    from agno.agent._tools import parse_tools
    from agno.tools import Toolkit
    from agno.tools.function import Function

    class _ToolkitProvider(_EchoProvider):
        def _all_tools(self, async_mode=False):
            toolkit = Toolkit(name="fs_toolkit")

            @toolkit.register
            def read_file(path: str) -> str:
                """Read a file."""
                return f"content of {path}"

            @toolkit.register
            def list_files(directory: str) -> str:
                """List files."""
                return f"files in {directory}"

            return [toolkit]

    provider = _ToolkitProvider(id="fs", mode=ContextMode.tools)

    # Mock agent with minimal required attributes
    mock_agent = MagicMock()
    mock_agent._team = None
    mock_agent.tool_hooks = None
    mock_agent._tool_instructions = []
    mock_agent.structured_outputs = False
    mock_agent.use_json_mode = False

    # Mock model
    mock_model = MagicMock()
    mock_model.supports_native_tools = True
    mock_model.supports_parallel_tool_calls = True
    mock_model.supports_native_structured_outputs = False

    functions = parse_tools(
        agent=mock_agent,
        tools=[provider],
        model=mock_model,
        async_mode=False,
    )

    # Should have expanded to 2 functions from the toolkit
    assert len(functions) == 2
    names = {f.name for f in functions}
    assert names == {"read_file", "list_files"}

    # Each function should have _agent set
    for func in functions:
        assert func._agent == mock_agent
        assert isinstance(func, Function)


# ---------------------------------------------------------------------------
# read / write flags — applied via the _read_write_tools helper that
# read+write subclasses call from their _default_tools override.
# ---------------------------------------------------------------------------


class _TwoToolProvider(_EchoProvider):
    """Subclass that exposes both query and update — uses the helper."""

    async def aupdate(self, instruction: str, *, run_context: RunContext | None = None) -> Answer:
        return Answer(text=f"u:{instruction}")

    def _default_tools(self, async_mode: bool = False) -> list:
        return self._read_write_tools(async_mode=async_mode)


def test_read_write_helper_default_returns_both_tools():
    p = _TwoToolProvider(id="x")
    assert [t.name for t in p.get_tools()] == ["query_x", "update_x"]


def test_read_write_helper_drops_update_when_write_false():
    p = _TwoToolProvider(id="x", write=False)
    assert [t.name for t in p.get_tools()] == ["query_x"]


def test_read_write_helper_drops_query_when_read_false():
    p = _TwoToolProvider(id="x", read=False)
    assert [t.name for t in p.get_tools()] == ["update_x"]


def test_both_flags_false_raises():
    with pytest.raises(ValueError, match="at least one of `read` or `write`"):
        _TwoToolProvider(id="x", read=False, write=False)


def test_read_write_flags_default_to_true():
    p = _EchoProvider(id="e")
    assert p.read is True
    assert p.write is True


def test_mode_agent_silently_ignores_read_false():
    """Per the design call: mode=agent + read=False is silently allowed.

    Behaviour-locking test — if we ever decide to raise, this is the
    test that flips. mode-mode interactions stay in their lane today.
    """
    p = _EchoProvider(id="e", mode=ContextMode.agent, read=False)
    tools = p.get_tools()
    # mode=agent always returns [query_tool] regardless of read.
    assert [t.name for t in tools] == ["query_e"]


# ---------------------------------------------------------------------------
# _query_tool — happy + error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_tool_serializes_answer_text():
    p = _EchoProvider(id="e")
    query_tool = p._build_query_tool(async_mode=True)
    out = await _collect_tool_output_async(query_tool, question="hello")
    payload = json.loads(out)
    # Empty `results` is omitted — no provider populates Document
    # results today, and shipping `"results": []` on every call is
    # filler the calling agent has to read past.
    assert payload == {"text": "q:hello"}


@pytest.mark.asyncio
async def test_query_tool_catches_aquery_exceptions():
    p = _RaisingQueryProvider(id="e")
    query_tool = p._build_query_tool(async_mode=True)
    out = await _collect_tool_output_async(query_tool, question="hello")
    payload = json.loads(out)
    # Error is reported as a string — the calling agent sees it but
    # isn't crashed.
    assert "error" in payload
    assert "RuntimeError" in payload["error"]
    assert "aquery boom" in payload["error"]


@pytest.mark.asyncio
async def test_query_tool_omits_both_when_answer_is_empty():
    """Both fields absent → empty payload. Honest "tool returned nothing" signal."""

    class _DocsOnly(_EchoProvider):
        async def aquery(self, question: str, *, run_context: RunContext | None = None) -> Answer:
            return Answer()

    tool_ = _DocsOnly(id="e")._build_query_tool(async_mode=True)
    out = await _collect_tool_output_async(tool_, question="hello")
    payload = json.loads(out)
    assert payload == {}


@pytest.mark.asyncio
async def test_query_tool_includes_results_when_populated():
    """When a provider does populate Document results, they're serialized."""
    from agno.context.provider import Document

    class _WithDocs(_EchoProvider):
        async def aquery(self, question: str, *, run_context: RunContext | None = None) -> Answer:
            return Answer(
                results=[Document(id="d1", name="Page 1", uri="/p/1", snippet="hello")],
                text="see results",
            )

    tool_ = _WithDocs(id="e")._build_query_tool(async_mode=True)
    out = await _collect_tool_output_async(tool_, question="hello")
    payload = json.loads(out)
    assert payload["text"] == "see results"
    assert payload["results"] == [{"id": "d1", "name": "Page 1", "uri": "/p/1", "source": None, "snippet": "hello"}]


# ---------------------------------------------------------------------------
# _update_tool — happy, error, read-only paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_tool_happy_path():
    p = _WritableProvider(id="w")
    tool_ = p._build_update_tool(async_mode=True)
    out = await tool_.entrypoint(instruction="add x")
    payload = json.loads(out)
    assert payload == {"text": "u:add x"}


@pytest.mark.asyncio
async def test_update_tool_reports_read_only_when_not_overridden():
    p = _EchoProvider(id="ro")  # no aupdate override -> base raises NotImplementedError
    tool_ = p._build_update_tool(async_mode=True)
    out = await tool_.entrypoint(instruction="add x")
    payload = json.loads(out)
    # Specifically a read-only message, not a generic exception string —
    # the calling agent should be able to learn from this and not retry.
    assert payload == {"error": f"{p.name} is read-only"}


@pytest.mark.asyncio
async def test_update_tool_catches_aupdate_exceptions():
    p = _RaisingWritableProvider(id="w")
    tool_ = p._build_update_tool(async_mode=True)
    out = await tool_.entrypoint(instruction="add x")
    payload = json.loads(out)
    assert "error" in payload
    assert "ValueError" in payload["error"]
    assert "aupdate boom" in payload["error"]


def test_sync_update_tool_happy_path():
    """Sync update tool returns JSON via update()."""
    p = _WritableProvider(id="w")
    tool_ = p._build_update_tool(async_mode=False)
    out = tool_.entrypoint(instruction="add x")
    payload = json.loads(out)
    assert payload == {"text": "u:add x"}


def test_sync_update_tool_reports_read_only():
    """Sync update tool reports read-only error."""
    p = _EchoProvider(id="ro")
    tool_ = p._build_update_tool(async_mode=False)
    out = tool_.entrypoint(instruction="add x")
    payload = json.loads(out)
    assert payload == {"error": f"{p.name} is read-only"}


def test_sync_update_tool_catches_exceptions():
    """Sync update tool catches update() exceptions."""
    p = _RaisingWritableProvider(id="w")
    tool_ = p._build_update_tool(async_mode=False)
    out = tool_.entrypoint(instruction="add x")
    payload = json.loads(out)
    assert "error" in payload
    assert "ValueError" in payload["error"]
    assert "update boom" in payload["error"]


# ---------------------------------------------------------------------------
# RunContext propagation — the wrapper should thread run_context from the
# calling agent's auto-injection into provider.aquery / aupdate, and the
# `_run_kwargs_for_sub_agent` helper should extract the right fields.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_tool_forwards_run_context_to_aquery():
    captured: dict = {}

    class _Captor(_EchoProvider):
        async def aquery(self, question: str, *, run_context: RunContext | None = None) -> Answer:
            captured["run_context"] = run_context
            return Answer(text=f"q:{question}")

    p = _Captor(id="c")
    query_tool = p._build_query_tool(async_mode=True)
    rc = RunContext(run_id="r-1", user_id="u-1", session_id="s-1", metadata={"action_token": "xoxa-abc"})
    # Framework would normally inject run_context via Function._run_context;
    # calling the entrypoint directly with run_context= simulates that path.
    await _collect_tool_output_async(query_tool, question="hello", run_context=rc)
    assert captured["run_context"] is rc


@pytest.mark.asyncio
async def test_update_tool_forwards_run_context_to_aupdate():
    captured: dict = {}

    class _WCaptor(_EchoProvider):
        async def aupdate(self, instruction: str, *, run_context: RunContext | None = None) -> Answer:
            captured["run_context"] = run_context
            return Answer(text=f"u:{instruction}")

    p = _WCaptor(id="w")
    update_tool = p._build_update_tool(async_mode=True)
    rc = RunContext(run_id="r-2", session_id="s-2", user_id="u-2", dependencies={"db_url": "postgres://..."})
    await update_tool.entrypoint(instruction="write x", run_context=rc)
    assert captured["run_context"] is rc


def test_run_kwargs_for_sub_agent_extracts_only_populated_fields():
    # None -> empty dict (no kwargs injected)
    assert _EchoProvider(id="e")._run_kwargs_for_sub_agent(None) == {}

    # Fields with truthy values are extracted
    rc = RunContext(
        run_id="r-3",
        user_id="u-1",
        session_id="s-1",
        metadata={"action_token": "xoxa-abc"},
        dependencies={"tenant": "acme"},
    )
    kwargs = _EchoProvider(id="e")._run_kwargs_for_sub_agent(rc)
    assert kwargs == {
        "user_id": "u-1",
        "session_id": "s-1",
        "metadata": {"action_token": "xoxa-abc"},
        "dependencies": {"tenant": "acme"},
    }


def test_run_kwargs_for_sub_agent_drops_empty_fields():
    # Empty dict / empty string / None values should NOT be propagated,
    # so sub-agent defaults aren't silently overridden with empty data.
    rc = RunContext(run_id="r-4", user_id="", session_id="only-session", metadata={}, dependencies=None)
    kwargs = _EchoProvider(id="e")._run_kwargs_for_sub_agent(rc)
    assert kwargs == {"session_id": "only-session"}


# ---------------------------------------------------------------------------
# Base asetup() / aclose() are safe no-ops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_aclose_is_noop():
    p = _EchoProvider(id="e")
    # Must not raise even though no session was ever opened.
    await p.aclose()


@pytest.mark.asyncio
async def test_base_asetup_is_noop():
    p = _EchoProvider(id="e")
    # Providers without async resources get a free pass on the hook.
    await p.asetup()


@pytest.mark.asyncio
async def test_base_asetup_is_idempotent():
    p = _EchoProvider(id="e")
    # Calling asetup() multiple times must be safe so callers can wire it
    # into a lifespan without tracking state themselves.
    await p.asetup()
    await p.asetup()


# ---------------------------------------------------------------------------
# Streaming vs non-streaming tool selection
# ---------------------------------------------------------------------------


def test_simple_tool_returns_string_directly():
    """Non-streaming tool returns JSON string via query()."""
    p = _EchoProvider(id="e", stream_sub_agent_events=False)
    out = _collect_tool_output_sync(p._build_query_tool(async_mode=False), question="hello")
    payload = json.loads(out)
    assert payload == {"text": "q:hello"}


def test_streaming_tool_yields_final_answer():
    """Streaming tool yields JSON answer when no sub-agent is configured."""
    p = _EchoProvider(id="e", stream_sub_agent_events=True)
    out = _collect_tool_output_sync(p._build_query_tool(async_mode=False), question="hello")
    payload = json.loads(out)
    assert payload == {"text": "q:hello"}


# ---------------------------------------------------------------------------
# Sub-agent event streaming — _aget_query_agent hook
# ---------------------------------------------------------------------------


class _SubAgentProvider(_EchoProvider):
    """Provider that returns a mock sub-agent from _aget_query_agent."""

    def __init__(self, sub_agent_events=None, **kwargs):
        super().__init__(**kwargs)
        self._sub_agent_events = sub_agent_events or []
        self._aquery_called = False
        self._aget_query_agent_called = False
        self._get_query_agent_called = False
        self._asetup_called = False
        self._setup_called = False

    def setup(self):
        self._setup_called = True

    async def asetup(self):
        self._asetup_called = True

    async def aquery(self, question: str, *, run_context=None) -> Answer:
        self._aquery_called = True
        return Answer(text=f"aquery:{question}")

    def _get_query_agent(self, run_context=None):
        """Sync sub-agent hook for sync streaming tests."""
        self._get_query_agent_called = True
        if not self._sub_agent_events:
            return None
        return _MockSubAgent(self._sub_agent_events)

    async def _aget_query_agent(self, run_context=None):
        self._aget_query_agent_called = True
        if not self._sub_agent_events:
            return None
        return _MockSubAgent(self._sub_agent_events)


class _MockSubAgent:
    """Mock agent that yields predefined events and captures call kwargs."""

    def __init__(self, events):
        self._events = events
        self.last_call_kwargs = None

    def run(self, question, **kwargs):
        from agno.run.agent import RunOutput

        self.last_call_kwargs = kwargs
        for event in self._events:
            yield event
        yield RunOutput(run_id="sub-run-1", content=f"sub-agent answer: {question}")

    async def arun(self, question, **kwargs):
        from agno.run.agent import RunOutput

        self.last_call_kwargs = kwargs
        for event in self._events:
            yield event
        yield RunOutput(run_id="sub-run-1", content=f"sub-agent answer: {question}")


@pytest.mark.asyncio
async def test_streaming_tool_yields_sub_agent_events():
    """When sub-agent is configured, streaming tool yields its events with correct identity."""
    from agno.run.agent import RunStartedEvent

    mock_event = RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent")
    p = _SubAgentProvider(id="e", stream_sub_agent_events=True, sub_agent_events=[mock_event])
    chunks = await _collect_streaming_chunks(p._build_query_tool(async_mode=True), question="test", run_context=None)

    # Should yield: 1 event + 1 final JSON answer
    assert len(chunks) == 2
    # First is the exact event we passed in (with parent_run_id added)
    assert chunks[0].run_id == "sub-run-1"
    assert chunks[0].agent_id == "sub-agent"
    # Last is JSON answer
    payload = json.loads(chunks[1])
    assert "text" in payload
    assert "sub-agent answer" in payload["text"]


@pytest.mark.asyncio
async def test_streaming_tool_passes_correct_flags_to_sub_agent():
    """Streaming tool passes stream=True, stream_events=True, yield_run_output=True."""
    from agno.run.agent import RunStartedEvent

    mock_event = RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent")
    mock_agent = _MockSubAgent([mock_event])

    class _KwargsCapturingProvider(_SubAgentProvider):
        async def _aget_query_agent(self, run_context=None):
            return mock_agent

    p = _KwargsCapturingProvider(id="e", stream_sub_agent_events=True)
    await _collect_streaming_chunks(p._build_query_tool(async_mode=True), question="test", run_context=None)

    assert mock_agent.last_call_kwargs["stream"] is True
    assert mock_agent.last_call_kwargs["stream_events"] is True
    assert mock_agent.last_call_kwargs["yield_run_output"] is True


@pytest.mark.asyncio
async def test_streaming_tool_sets_parent_run_id_on_events():
    """Sub-agent events get parent_run_id set to the calling agent's run_id."""
    from agno.run.agent import RunStartedEvent

    mock_events = [RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent")]
    p = _SubAgentProvider(id="e", stream_sub_agent_events=True, sub_agent_events=mock_events)
    rc = RunContext(run_id="parent-run-123", session_id="s-1", user_id="u-1")
    chunks = await _collect_streaming_chunks(p._build_query_tool(async_mode=True), question="test", run_context=rc)

    assert chunks[0].parent_run_id == "parent-run-123"


@pytest.mark.asyncio
async def test_streaming_tool_calls_asetup_before_running():
    """Streaming tool calls asetup() before running sub-agent, same as aquery()."""
    from agno.run.agent import RunStartedEvent

    mock_events = [RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent")]
    p = _SubAgentProvider(id="e", stream_sub_agent_events=True, sub_agent_events=mock_events)
    await _collect_streaming_chunks(p._build_query_tool(async_mode=True), question="test", run_context=None)

    assert p._asetup_called


@pytest.mark.asyncio
async def test_streaming_tool_does_not_call_aquery_when_sub_agent_exists():
    """When sub-agent streams, aquery() is bypassed entirely."""
    from agno.run.agent import RunStartedEvent

    mock_events = [RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent")]
    p = _SubAgentProvider(id="e", stream_sub_agent_events=True, sub_agent_events=mock_events)
    await _collect_streaming_chunks(p._build_query_tool(async_mode=True), question="test", run_context=None)

    assert not p._aquery_called


@pytest.mark.asyncio
async def test_streaming_tool_falls_back_to_aquery_when_no_sub_agent():
    """When _aget_query_agent returns None, streaming tool calls aquery()."""
    p = _SubAgentProvider(id="e", stream_sub_agent_events=True, sub_agent_events=None)
    out = await _collect_tool_output_async(p._build_query_tool(async_mode=True), question="hello")

    assert p._aquery_called
    payload = json.loads(out)
    assert payload == {"text": "aquery:hello"}


@pytest.mark.asyncio
async def test_simple_tool_never_calls_aget_query_agent():
    """Non-streaming tool uses aquery() directly, never checks for sub-agent."""
    p = _SubAgentProvider(id="e", stream_sub_agent_events=False, sub_agent_events=[])
    await _collect_tool_output_async(p._build_query_tool(async_mode=True), question="hello")

    assert p._aquery_called
    assert not p._aget_query_agent_called


# ---------------------------------------------------------------------------
# Sync sub-agent streaming — _get_query_agent hook
# ---------------------------------------------------------------------------


def _collect_sync_streaming_chunks(tool, **kwargs) -> list:
    """Collect all chunks from a sync streaming tool."""
    chunks = []
    for chunk in tool.entrypoint(**kwargs):
        chunks.append(chunk)
    return chunks


def test_sync_streaming_tool_yields_sub_agent_events():
    """Sync streaming tool yields sub-agent events via _get_query_agent."""
    from agno.run.agent import RunStartedEvent

    mock_event = RunStartedEvent(run_id="sub-run-sync", agent_id="sync-agent")
    p = _SubAgentProvider(id="s", stream_sub_agent_events=True, sub_agent_events=[mock_event])
    chunks = _collect_sync_streaming_chunks(p._build_query_tool(async_mode=False), question="test", run_context=None)

    # Should yield: 1 event + 1 final JSON answer
    assert len(chunks) == 2
    assert chunks[0].run_id == "sub-run-sync"
    assert chunks[0].agent_id == "sync-agent"
    payload = json.loads(chunks[1])
    assert "sub-agent answer" in payload["text"]


def test_sync_streaming_tool_calls_setup_before_running():
    """Sync streaming tool calls setup() before running sub-agent."""
    from agno.run.agent import RunStartedEvent

    mock_events = [RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent")]
    p = _SubAgentProvider(id="s", stream_sub_agent_events=True, sub_agent_events=mock_events)
    _collect_sync_streaming_chunks(p._build_query_tool(async_mode=False), question="test", run_context=None)

    assert p._setup_called
    assert p._get_query_agent_called


def test_sync_streaming_tool_sets_parent_run_id():
    """Sync streaming tool sets parent_run_id on events."""
    from agno.run.agent import RunStartedEvent

    mock_events = [RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent")]
    p = _SubAgentProvider(id="s", stream_sub_agent_events=True, sub_agent_events=mock_events)
    rc = RunContext(run_id="parent-sync-123", session_id="s-1", user_id="u-1")
    chunks = _collect_sync_streaming_chunks(p._build_query_tool(async_mode=False), question="test", run_context=rc)

    assert chunks[0].parent_run_id == "parent-sync-123"


def test_sync_streaming_preserves_existing_parent_run_id():
    """Sync: Events with existing parent_run_id are not overwritten."""
    from agno.run.agent import RunStartedEvent

    event_with_parent = RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent")
    event_with_parent.parent_run_id = "already-set-parent"

    p = _SubAgentProvider(id="p", stream_sub_agent_events=True, sub_agent_events=[event_with_parent])
    rc = RunContext(run_id="outer-run-456", session_id="s-1", user_id="u-1")
    chunks = _collect_sync_streaming_chunks(p._build_query_tool(async_mode=False), question="test", run_context=rc)

    assert chunks[0].parent_run_id == "already-set-parent"


def test_sync_streaming_filters_run_content_events():
    """Sync: RunContentEvent is filtered out to prevent content contamination."""
    from agno.run.agent import RunContentEvent, RunStartedEvent

    events = [
        RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent"),
        RunContentEvent(run_id="sub-run-1", content="partial content"),
    ]
    p = _SubAgentProvider(id="f", stream_sub_agent_events=True, sub_agent_events=events)
    chunks = _collect_sync_streaming_chunks(p._build_query_tool(async_mode=False), question="test", run_context=None)

    # RunContentEvent should be filtered: 1 RunStartedEvent + 1 final JSON
    assert len(chunks) == 2
    assert chunks[0].run_id == "sub-run-1"
    assert isinstance(chunks[1], str)
    payload = json.loads(chunks[1])
    assert "text" in payload


def test_sync_streaming_yields_multiple_event_types():
    """Sync: Yields various event types (except RunContentEvent)."""
    from agno.run.agent import RunStartedEvent, ToolCallStartedEvent

    events = [
        RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent"),
        ToolCallStartedEvent(run_id="sub-run-1", tool=None),
    ]
    p = _SubAgentProvider(id="m", stream_sub_agent_events=True, sub_agent_events=events)
    chunks = _collect_sync_streaming_chunks(p._build_query_tool(async_mode=False), question="test", run_context=None)

    # Should yield: 2 events + 1 final JSON answer
    assert len(chunks) == 3
    assert chunks[0].run_id == "sub-run-1"
    assert hasattr(chunks[1], "tool")
    payload = json.loads(chunks[2])
    assert "text" in payload


@pytest.mark.asyncio
async def test_streaming_preserves_existing_parent_run_id():
    """Events with existing parent_run_id are not overwritten."""
    from agno.run.agent import RunStartedEvent

    # Event already has a parent_run_id set
    event_with_parent = RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent")
    event_with_parent.parent_run_id = "already-set-parent"

    p = _SubAgentProvider(id="p", stream_sub_agent_events=True, sub_agent_events=[event_with_parent])
    rc = RunContext(run_id="outer-run-456", session_id="s-1", user_id="u-1")
    chunks = await _collect_streaming_chunks(p._build_query_tool(async_mode=True), question="test", run_context=rc)

    # Existing parent_run_id should be preserved via the `or run_id` fallback
    assert chunks[0].parent_run_id == "already-set-parent"


# ---------------------------------------------------------------------------
# Event type filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_filters_run_content_events():
    """RunContentEvent is filtered out to prevent content contamination."""
    from agno.run.agent import RunContentEvent, RunStartedEvent

    # Include both a RunStartedEvent (should pass) and RunContentEvent (should be filtered)
    events = [
        RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent"),
        RunContentEvent(run_id="sub-run-1", content="partial content"),
    ]
    p = _SubAgentProvider(id="f", stream_sub_agent_events=True, sub_agent_events=events)
    chunks = await _collect_streaming_chunks(p._build_query_tool(async_mode=True), question="test", run_context=None)

    # RunContentEvent should be filtered, so we get: 1 RunStartedEvent + 1 final JSON
    assert len(chunks) == 2
    assert chunks[0].run_id == "sub-run-1"
    # Second chunk is JSON answer, not RunContentEvent
    assert isinstance(chunks[1], str)
    payload = json.loads(chunks[1])
    assert "text" in payload


@pytest.mark.asyncio
async def test_streaming_yields_multiple_event_types():
    """Streaming yields various event types (except RunContentEvent)."""
    from agno.run.agent import RunStartedEvent, ToolCallStartedEvent

    events = [
        RunStartedEvent(run_id="sub-run-1", agent_id="sub-agent"),
        ToolCallStartedEvent(run_id="sub-run-1", tool=None),
    ]
    p = _SubAgentProvider(id="m", stream_sub_agent_events=True, sub_agent_events=events)
    chunks = await _collect_streaming_chunks(p._build_query_tool(async_mode=True), question="test", run_context=None)

    # Should yield: 2 events + 1 final JSON answer
    assert len(chunks) == 3
    assert chunks[0].run_id == "sub-run-1"
    assert hasattr(chunks[1], "tool")  # ToolCallStartedEvent has tool field
    payload = json.loads(chunks[2])
    assert "text" in payload


# ---------------------------------------------------------------------------
# Exception handling in streaming paths
# ---------------------------------------------------------------------------


class _FailingSetupProvider(_EchoProvider):
    """Provider that raises in setup/asetup."""

    def setup(self):
        raise ConnectionError("Failed to connect to backend")

    async def asetup(self):
        raise ConnectionError("Failed to connect to backend")


class _FailingAgentHookProvider(_EchoProvider):
    """Provider that raises in _get_query_agent/_aget_query_agent."""

    def setup(self):
        pass

    async def asetup(self):
        pass

    def _get_query_agent(self, run_context=None):
        raise RuntimeError("Agent initialization failed")

    async def _aget_query_agent(self, run_context=None):
        raise RuntimeError("Agent initialization failed")


@pytest.mark.asyncio
async def test_async_streaming_handles_asetup_exception():
    """Async streaming yields error JSON when asetup() raises."""
    p = _FailingSetupProvider(id="fs", stream_sub_agent_events=True)
    out = await _collect_tool_output_async(p._build_query_tool(async_mode=True), question="test")

    payload = json.loads(out)
    assert "error" in payload
    assert "ConnectionError" in payload["error"]
    assert "Failed to connect" in payload["error"]


@pytest.mark.asyncio
async def test_async_streaming_handles_aget_query_agent_exception():
    """Async streaming yields error JSON when _aget_query_agent() raises."""
    p = _FailingAgentHookProvider(id="fa", stream_sub_agent_events=True)
    out = await _collect_tool_output_async(p._build_query_tool(async_mode=True), question="test")

    payload = json.loads(out)
    assert "error" in payload
    assert "RuntimeError" in payload["error"]
    assert "Agent initialization failed" in payload["error"]


def test_sync_streaming_handles_setup_exception():
    """Sync streaming yields error JSON when setup() raises."""
    p = _FailingSetupProvider(id="fs", stream_sub_agent_events=True)
    out = _collect_tool_output_sync(p._build_query_tool(async_mode=False), question="test")

    payload = json.loads(out)
    assert "error" in payload
    assert "ConnectionError" in payload["error"]
    assert "Failed to connect" in payload["error"]


def test_sync_streaming_handles_get_query_agent_exception():
    """Sync streaming yields error JSON when _get_query_agent() raises."""
    p = _FailingAgentHookProvider(id="fa", stream_sub_agent_events=True)
    out = _collect_tool_output_sync(p._build_query_tool(async_mode=False), question="test")

    payload = json.loads(out)
    assert "error" in payload
    assert "RuntimeError" in payload["error"]
    assert "Agent initialization failed" in payload["error"]


# ---------------------------------------------------------------------------
# Integration: parse_tools expands ContextProvider correctly
# ---------------------------------------------------------------------------


def test_parse_tools_expands_context_provider_sync():
    """parse_tools() correctly expands ContextProvider for sync agents."""
    from agno.agent._tools import parse_tools
    from agno.tools.function import Function
    from unittest.mock import MagicMock

    p = _EchoProvider(id="test", stream_sub_agent_events=True)

    # Create a minimal mock agent
    agent = MagicMock()
    agent._team = None
    agent.tool_hooks = None
    agent._tool_instructions = []

    model = MagicMock()
    functions = parse_tools(agent, [p], model, async_mode=False)

    # Should have expanded to query_test tool
    assert len(functions) >= 1
    tool_names = [f.name for f in functions]
    assert "query_test" in tool_names

    # Tool should be a Function with correct structure
    query_tool = next(f for f in functions if f.name == "query_test")
    assert isinstance(query_tool, Function)
    assert query_tool.entrypoint is not None


@pytest.mark.asyncio
async def test_parse_tools_expands_context_provider_async():
    """parse_tools() correctly expands ContextProvider for async agents."""
    from agno.agent._tools import parse_tools
    from agno.tools.function import Function
    from unittest.mock import MagicMock

    p = _EchoProvider(id="test", stream_sub_agent_events=True)

    agent = MagicMock()
    agent._team = None
    agent.tool_hooks = None
    agent._tool_instructions = []

    model = MagicMock()
    functions = parse_tools(agent, [p], model, async_mode=True)

    assert len(functions) >= 1
    tool_names = [f.name for f in functions]
    assert "query_test" in tool_names

    query_tool = next(f for f in functions if f.name == "query_test")
    assert isinstance(query_tool, Function)


def test_parse_tools_expands_toolkit_from_context_provider_mode_tools():
    """When mode=tools, parse_tools expands the Toolkit returned by provider."""
    from agno.agent._tools import parse_tools
    from agno.context.mode import ContextMode
    from agno.tools.function import Function
    from unittest.mock import MagicMock

    # _EchoProvider with mode=tools returns a Toolkit
    p = _EchoProvider(id="test", mode=ContextMode.tools, stream_sub_agent_events=True)

    agent = MagicMock()
    agent._team = None
    agent.tool_hooks = None
    agent._tool_instructions = []

    model = MagicMock()
    functions = parse_tools(agent, [p], model, async_mode=False)

    # Toolkit should have at least one function
    assert len(functions) >= 1
    # All parsed items should be Function instances
    for f in functions:
        assert isinstance(f, Function)
