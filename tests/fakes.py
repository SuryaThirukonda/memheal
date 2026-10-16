"""Offline stand-ins for the LLM and embedder."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from memheal.llm import LLMResponse, ToolCall


class FakeEmbedder:
    name = "fake-hash-embedder"

    def encode(self, texts):
        vecs = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
            v = np.random.default_rng(seed).standard_normal(16).astype(np.float32)
            vecs.append(v / np.linalg.norm(v))
        return np.stack(vecs)


class FakeLLM:
    """Plays back scripted tool calls for agent turns; extraction returns the input's
    first line of text as a single fact."""

    def __init__(self, script: list[tuple[str, dict]], consolidation=None, reflection=None):
        self.script = list(script)
        # consolidation: the {"summaries": [...]} payload to return; reflection: list of facts.
        self.consolidation = consolidation
        self.reflection = reflection
        self.model = "fake-model"
        self.returned_model = "fake-model"
        self.calls = 0
        self.tags: list[str] = []

    def params(self):
        return {"model": "fake-model"}

    def respond(self, instructions, input, *, tools=None, tool_choice=None, json_schema=None, tag=""):
        self.calls += 1
        self.tags.append(tag)
        if json_schema and json_schema["name"] == "consolidated_memories":
            return LLMResponse([], json.dumps({"summaries": self.consolidation or []}), "fake-model")
        if json_schema and tag.startswith("reflect:"):
            return LLMResponse([], json.dumps({"facts": self.reflection or []}), "fake-model")
        if json_schema:
            text = input.split("Text:\n", 1)[-1].strip()
            return LLMResponse([], json.dumps({"facts": [text]}), "fake-model")
        if not tools:
            return LLMResponse([], "fake reply", "fake-model")
        name, args = self.script.pop(0)
        call = ToolCall(f"call_{self.calls}", name, args)
        item = {"type": "function_call", "call_id": call.call_id, "name": name, "arguments": json.dumps(args)}
        return LLMResponse([call], "", "fake-model", [item])
