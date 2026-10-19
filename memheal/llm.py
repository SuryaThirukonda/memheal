"""Thin wrapper around the OpenAI Responses API.

Every call is pinned (model, temperature, reasoning effort) and appended to a
per-run JSONL log, so each reported run keeps every prompt and raw output
(proposal Section 6.6). The agent and memory code depend only on LLMResponse,
so tests can substitute a scripted fake.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gpt-5.4-mini"

# A sweep is hundreds of sequential calls, so a transient network blip is close to
# certain over a full catalogue run. Without this one of them aborts the variant and
# silently loses every run in it.
MAX_ATTEMPTS = 5
BACKOFF_BASE = 2.0


@dataclass
class ToolCall:
    call_id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    tool_calls: list[ToolCall]
    text: str
    model: str
    # Items to append to the conversation input, rebuilt without server-side ids
    # so they can be replayed with store=False.
    history_items: list[dict[str, Any]] = field(default_factory=list)


class LLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        log_path: str | Path | None = None,
        temperature: float | None = 0,
        reasoning_effort: str | None = "none",
        client: Any = None,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.log_path = Path(log_path) if log_path else None
        self.returned_model: str | None = None
        self.retries = 0

    def params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }

    def respond(
        self,
        instructions: str,
        input: list[dict[str, Any]] | str,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        json_schema: dict[str, Any] | None = None,
        tag: str = "",
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input,
            "store": False,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if tools:
            request["tools"] = tools
            request["parallel_tool_calls"] = False
            if tool_choice:
                request["tool_choice"] = tool_choice
        if json_schema:
            request["text"] = {"format": {"type": "json_schema", "strict": True, **json_schema}}

        response = self._create_with_retry(request, tag)
        self.returned_model = response.model

        calls: list[ToolCall] = []
        history: list[dict[str, Any]] = []
        for item in response.output:
            if item.type == "function_call":
                calls.append(ToolCall(item.call_id, item.name, json.loads(item.arguments)))
                history.append({
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                })
        text = response.output_text or ""
        if text:
            history.insert(0, {"role": "assistant", "content": text})

        self._log(tag, request, response)
        return LLMResponse(calls, text, response.model, history)

    def _create_with_retry(self, request: dict[str, Any], tag: str) -> Any:
        """Retry the transient failures; let a genuine bad request surface immediately."""
        import openai

        transient = (openai.APIConnectionError, openai.APITimeoutError,
                     openai.RateLimitError, openai.InternalServerError)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self.client.responses.create(**request)
            except transient as exc:
                self.retries += 1
                if attempt == MAX_ATTEMPTS:
                    raise
                # Jitter so parallel workers do not all come back at the same moment.
                delay = BACKOFF_BASE ** (attempt - 1) * (1 + random.random() * 0.3)
                print(f"    [retry {attempt}/{MAX_ATTEMPTS - 1} after "
                      f"{type(exc).__name__} on {tag}; sleeping {delay:.1f}s]", flush=True)
                time.sleep(delay)

    def _log(self, tag: str, request: dict[str, Any], response: Any) -> None:
        if not self.log_path:
            return
        entry = {
            "ts": time.time(),
            "tag": tag,
            "request": request,
            "returned_model": response.model,
            "output": [item.model_dump(exclude_none=True) for item in response.output],
            "usage": response.usage.model_dump() if response.usage else None,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
