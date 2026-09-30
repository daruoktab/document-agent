"""Per-call VLM failover shared by document tools and autonomous agents."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openai import APIConnectionError, APIStatusError
from pydantic import Field

logger = logging.getLogger(__name__)
Role = Literal["vision", "agent"]


def service_failure(exc: Exception) -> bool:
    """Do not conceal malformed requests or application programming errors."""
    return isinstance(exc, (APIConnectionError, TimeoutError, ConnectionError)) or (
        isinstance(exc, APIStatusError)
        and (exc.status_code in {401, 403, 404, 408, 429} or exc.status_code >= 500)
    )


class ModelRuntime:
    def __init__(
        self,
        vision: Any,
        agent: Any = None,
        *,
        cooldown: float = 60,
        settings: Any = None,
    ) -> None:
        self.models = {"vision": vision, "agent": agent}
        self.settings = settings
        self.cooldown = cooldown
        self._blocked: dict[tuple[Any, ...], float] = {}
        self._lock = threading.RLock()
        self.events: list[dict[str, Any]] = []

    def reset(self) -> None:
        with self._lock:
            self._blocked.clear()
            self.events.clear()

    @staticmethod
    def identity(model: Any) -> tuple[Any, ...]:
        identity = (
            getattr(model, "openai_api_base", None),
            getattr(model, "model_name", None),
        )
        return (id(model),) if identity == (None, None) else identity

    def candidates(self, role: Role) -> list[tuple[str, Any]]:
        order = (role, "agent" if role == "vision" else "vision")
        result: list[tuple[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        with self._lock:
            for name in order:
                model = self.models[name]
                if model is None:
                    continue
                identity = self.identity(model)
                if identity in seen:
                    continue
                seen.add(identity)
                if self._blocked.get(identity, 0) > time.monotonic():
                    continue
                result.append((name, model))
        return result

    def dual_available(self) -> bool:
        return len(self.candidates("vision")) > 1

    def record(
        self, role: Role, actual: str, model: Any, error: str = ""
    ) -> dict[str, Any]:
        event = {
            "requested_role": role,
            "actual_role": actual,
            "model": str(getattr(model, "model_name", type(model).__name__)),
            "fallback": actual != role,
            "error": error,
        }
        with self._lock:
            self.events.append(event)
            if error:
                self._blocked[self.identity(model)] = time.monotonic() + self.cooldown
        logger.info("VLM route: %s", event)
        return event

    def routed(self, role: Role) -> RoutedChatModel:
        return RoutedChatModel(runtime=self, role=role)


class RoutedChatModel(BaseChatModel):
    """Buffer each model response before exposing it; never replay tool execution."""

    runtime: Any = Field(exclude=True)
    role: Role
    tools: list[Any] | None = Field(default=None, exclude=True)
    tool_options: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "routed-vlm"

    def bind_tools(self, tools: Any, **kwargs: Any) -> RoutedChatModel:
        return self.model_copy(update={"tools": list(tools), "tool_options": kwargs})

    def _target(self, model: Any) -> Any:
        return (
            model.bind_tools(self.tools, **self.tool_options)
            if self.tools is not None
            else model
        )

    def _result(self, response: Any, actual: str, model: Any) -> ChatResult:
        event = self.runtime.record(self.role, actual, model)
        message = (
            response
            if isinstance(response, AIMessage)
            else AIMessage(content=str(response.content))
        )
        message = message.model_copy(
            update={
                "response_metadata": {
                    **message.response_metadata,
                    "vlm_route": event.copy(),
                }
            }
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        failure: Exception | None = None
        for actual, model in self.runtime.candidates(self.role):
            try:
                config = (
                    {
                        "callbacks": CallbackManager(
                            handlers=run_manager.inheritable_handlers,
                            parent_run_id=run_manager.run_id,
                        )
                    }
                    if run_manager
                    else None
                )
                response = self._target(model).invoke(
                    messages, config=config, stop=stop, **kwargs
                )
                return self._result(response, actual, model)
            except Exception as exc:
                if not service_failure(exc):
                    raise
                failure = exc
                self.runtime.record(self.role, actual, model, str(exc))
        raise RuntimeError(
            "Tidak ada VLM yang tersedia untuk menyelesaikan pemanggilan."
        ) from failure

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        failure: Exception | None = None
        for actual, model in self.runtime.candidates(self.role):
            try:
                config = (
                    {
                        "callbacks": AsyncCallbackManager(
                            handlers=run_manager.inheritable_handlers,
                            parent_run_id=run_manager.run_id,
                        )
                    }
                    if run_manager
                    else None
                )
                response = await self._target(model).ainvoke(
                    messages, config=config, stop=stop, **kwargs
                )
                return self._result(response, actual, model)
            except Exception as exc:
                if not service_failure(exc):
                    raise
                failure = exc
                self.runtime.record(self.role, actual, model, str(exc))
        raise RuntimeError(
            "Tidak ada VLM yang tersedia untuk menyelesaikan pemanggilan."
        ) from failure
