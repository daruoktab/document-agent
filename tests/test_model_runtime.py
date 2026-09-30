"""Offline failover, tool binding, and agent execution regression tests."""

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from app.model_runtime import ModelRuntime


class ScriptedModel(BaseChatModel):
    model_name: str
    outcomes: list[Any]
    calls: list[Any] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "offline-scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        result = self.outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        message = result if isinstance(result, AIMessage) else AIMessage(content=result)
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.mark.parametrize("preferred", ["vision", "agent"])
def test_bidirectional_failover_and_cooldown(preferred):
    vision = ScriptedModel(model_name="vision", outcomes=["ok", "again"])
    agent = ScriptedModel(model_name="agent", outcomes=["ok", "again"])
    runtime = ModelRuntime(vision, agent)
    runtime.models[preferred].outcomes = [ConnectionError("offline")]
    routed = runtime.routed(preferred)
    assert routed.invoke("read").content == "ok"
    assert routed.invoke("read again").content == "again"
    assert len(runtime.models[preferred].calls) == 1
    assert runtime.events[-1]["fallback"] is True
    assert not runtime.dual_available()


def test_both_models_fail_cleanly_and_no_repeated_attempts():
    vision = ScriptedModel(model_name="vision", outcomes=[TimeoutError("vision")])
    agent = ScriptedModel(model_name="agent", outcomes=[ConnectionError("agent")])
    runtime = ModelRuntime(vision, agent)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Tidak ada VLM"):
            runtime.routed("vision").invoke("read")
    assert len(vision.calls) == len(agent.calls) == 1


def test_programming_error_is_not_hidden_by_fallback():
    vision = ScriptedModel(model_name="vision", outcomes=[ValueError("bad request")])
    agent = ScriptedModel(model_name="agent", outcomes=["unused"])
    with pytest.raises(ValueError):
        ModelRuntime(vision, agent).routed("vision").invoke("read")
    assert agent.calls == []


def test_single_model_is_not_retried_as_its_own_fallback():
    vision = ScriptedModel(model_name="vision", outcomes=[TimeoutError("offline")])
    with pytest.raises(RuntimeError):
        ModelRuntime(vision).routed("agent").invoke("read")
    assert len(vision.calls) == 1


def test_same_endpoint_model_is_not_retried_through_other_role():
    vision = ScriptedModel(model_name="shared", outcomes=["must not retry"])
    agent = ScriptedModel(model_name="shared", outcomes=[TimeoutError("offline")])
    runtime = ModelRuntime(vision, agent)
    with pytest.raises(RuntimeError):
        runtime.routed("agent").invoke("read")
    # Reverse role must still honor the cooldown for this identical endpoint/model.
    with pytest.raises(RuntimeError):
        runtime.routed("vision").invoke("read")
    assert len(agent.calls) == 1 and vision.calls == []


def test_preferred_model_can_recover_after_cooldown():
    vision = ScriptedModel(
        model_name="vision", outcomes=[TimeoutError("offline"), "recovered"]
    )
    agent = ScriptedModel(model_name="agent", outcomes=["fallback"])
    runtime = ModelRuntime(vision, agent)
    with patch("app.model_runtime.time.monotonic", return_value=0):
        assert runtime.routed("vision").invoke("read").content == "fallback"
    with patch("app.model_runtime.time.monotonic", return_value=61):
        assert runtime.routed("vision").invoke("read").content == "recovered"


def test_async_failover_preserves_tool_calls_and_metadata():
    expected = AIMessage(
        content="", tool_calls=[{"name": "save", "args": {}, "id": "1"}]
    )
    vision = ScriptedModel(model_name="vision", outcomes=[expected])
    agent = ScriptedModel(model_name="agent", outcomes=[TimeoutError("offline")])
    routed = ModelRuntime(vision, agent).routed("agent").bind_tools([{"name": "save"}])
    result = asyncio.run(routed.ainvoke("save"))
    assert result.tool_calls == expected.tool_calls
    assert result.response_metadata["vlm_route"]["model"] == "vision"
    assert len(vision.bound_tools) == 1


def test_fallback_does_not_replay_agent_tools():
    writes = []

    @tool
    def save_result(value: str) -> str:
        """Store the extraction once."""
        writes.append(value)
        return "saved"

    vision = ScriptedModel(
        model_name="vision",
        outcomes=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "save_result",
                        "args": {"value": "data"},
                        "id": "save-1",
                    }
                ],
            ),
            "completed",
        ],
    )
    agent = ScriptedModel(model_name="agent", outcomes=[TimeoutError("offline")])
    runnable = create_agent(
        ModelRuntime(vision, agent).routed("agent"), tools=[save_result]
    )
    result = runnable.invoke({"messages": [{"role": "user", "content": "save"}]})
    assert result["messages"][-1].content == "completed"
    assert writes == ["data"]


def test_real_deep_agent_accepts_routed_model_without_api():
    from app.config import Settings
    from app.deep_agent import build_deep_agent

    vision = ScriptedModel(model_name="vision", outcomes=["finished"])
    agent = ScriptedModel(model_name="agent", outcomes=[ConnectionError("offline")])
    with (
        patch("app.deep_agent.build_vlm", return_value=vision),
        patch("app.deep_agent.build_language_vlm", return_value=agent),
        patch("app.learning_store.LearningStore") as store,
    ):
        store.return_value.active_run.return_value = None
        runnable = build_deep_agent(Settings(language_vlm_model="agent", ocr_model=""))
        result = runnable.invoke({"messages": [{"role": "user", "content": "hello"}]})
    assert result["messages"][-1].content == "finished"


@pytest.mark.parametrize("subagent", ["layout-classifier", "general-purpose"])
def test_real_subagents_inherit_per_call_failover(subagent):
    from app.config import Settings
    from app.deep_agent import build_deep_agent

    agent = ScriptedModel(
        model_name="agent",
        outcomes=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Classify this request.",
                            "subagent_type": subagent,
                        },
                        "id": "delegate-1",
                    }
                ],
            ),
            TimeoutError("agent went offline during delegation"),
        ],
    )
    vision = ScriptedModel(model_name="vision", outcomes=["plain", "finished"])
    with (
        patch("app.deep_agent.build_vlm", return_value=vision),
        patch("app.deep_agent.build_language_vlm", return_value=agent),
        patch("app.learning_store.LearningStore") as store,
    ):
        store.return_value.active_run.return_value = None
        runnable = build_deep_agent(Settings(language_vlm_model="agent", ocr_model=""))
        result = runnable.invoke(
            {"messages": [{"role": "user", "content": "Delegate classification."}]}
        )
    assert result["messages"][-1].content == "finished"
    assert len(agent.calls) == 2 and len(vision.calls) == 2
