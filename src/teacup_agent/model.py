"""Model — the only part that thinks. Everything else is plumbing.

The model is behind one interface, `complete(messages, tools) -> Reply`, so there
can be several implementations:

* OpenAIModel / ResponsesModel  real API calls (need OPENAI_API_KEY)
* ScriptedModel                 replies from a script, for offline demos and evals
                                (no dependencies, no cost)

This abstraction is not over-engineering: without it there is no way to test the
control loop without spending money.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # a JSON **string**, not a dict


@dataclass
class Reply:
    """One turn of model output, normalized.

    `items` are "the entries to append back into the context", not "one assistant
    message": Chat Completions produces exactly one per turn, while Responses can
    produce several (a reasoning item plus function_call items). Carrying them back
    verbatim is precisely what preserves the reasoning state.
    """

    items: list[dict[str, Any]]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    cost: float = 0.0  # dollars spent on this turn
    input_tokens: int = 0  # size of the context sent in (drives compaction)
    cached_tokens: int = 0  # of which served from the prompt cache (10x cheaper)


class Model(Protocol):
    def set_cache_key(self, key: str) -> None:
        """Give the prompt cache a stable grouping key (optional to implement)."""
        ...

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply: ...

    def tool_result_item(self, call: ToolCall, result: str) -> dict[str, Any]:
        """The shape a tool result takes in the context — the two APIs differ here."""
        ...


def chat_tool_result(call: ToolCall, result: str) -> dict[str, Any]:
    """Tool-result shape for Chat Completions."""
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": result,
    }


Price = tuple[float, float, float]  # (input, cached input, output), USD per 1M tokens


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    prices: Price | None = None,
) -> float:
    """input_tokens is the **total** input (cache hits included); cached_tokens is
    the part of it served from cache.

    `prices` overrides the name-keyed table. PRICES only knows OpenAI's own models,
    and `_DEFAULT_PRICE` guesses for everything else — which is fine for a demo and
    actively misleading the moment a run points at a local or third-party endpoint
    (#15's `base_url` reaches vLLM, Ollama and OpenRouter for free). A profile that
    states its own rate gets cost tracking that means something instead of gpt-5's
    price attached to a model that is not gpt-5.
    """
    pin, pcached, pout = prices if prices is not None else PRICES.get(model, _DEFAULT_PRICE)
    fresh = max(0, input_tokens - cached_tokens)
    return (fresh * pin + cached_tokens * pcached + output_tokens * pout) / 1_000_000


def cached_from(usage: Any, field: str) -> int:
    """Dig the cached-token count out of `usage` (the two APIs name the field differently)."""
    details = getattr(usage, field, None)
    return getattr(details, "cached_tokens", 0) or 0 if details else 0


# --------------------------------------------------------------------------
# Real models
# --------------------------------------------------------------------------

# Price per million tokens in USD: (input, cached input, output). For the budget
# demo only, and it goes stale — check the official price list.
# Cached input usually costs a tenth of fresh input: that is the direct payoff for
# keeping the context prefix stable.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
}
_DEFAULT_PRICE = (1.25, 0.125, 10.00)


class OpenAIModel:
    """The Chat Completions path.

    Why keep it when Responses is the default? Because the `messages` list *is* the
    agent's state here, so "append the tool result and ask again" is visible at a
    glance — the clearest version for learning. Switching APIs only touches this
    class; the loop does not move.
    """

    def __init__(self, model: str = "gpt-5", client: Any = None, prices: Price | None = None):
        self.model = model
        self.prices = prices
        self.cache_key: str | None = None
        if client is None:
            from openai import OpenAI  # lazy import: offline runs need no openai

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is missing. Set it in .env, or run offline "
                    "(drop --live)."
                )
            client = OpenAI()
        self.client = client

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
        }
        if self.cache_key:  # route same-prefix requests together for better hits
            kwargs["prompt_cache_key"] = self.cache_key
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        cached = cached_from(usage, "prompt_tokens_details") if usage else 0
        return Reply(
            items=[msg.model_dump(exclude_none=True)],
            text=msg.content or "",
            tool_calls=calls,
            cost=estimate_cost(
                self.model,
                prompt_tokens,
                getattr(usage, "completion_tokens", 0) if usage else 0,
                cached,
                self.prices,
            ),
            input_tokens=prompt_tokens,
            cached_tokens=cached,
        )

    def tool_result_item(self, call: ToolCall, result: str) -> dict[str, Any]:
        return chat_tool_result(call, result)

    def set_cache_key(self, key: str) -> None:
        # This was missing until routing (#21) went looking for it: `complete()` has
        # always read `self.cache_key`, but nothing could ever write it, so loop.py's
        # `getattr(model, "set_cache_key", None)` skipped this class and
        # `prompt_cache_key` was dead on the Chat path.
        self.cache_key = key


class ResponsesModel:
    """The Responses API — the recommended path for reasoning models (gpt-5 family).

    All four shape differences from Chat Completions are sealed inside this class,
    so loop.py needs no changes:

    1. Tool definitions are **flat**: {"type": "function", "name": ...}, with no
       nested "function" layer;
    2. Output is the resp.output list (which may contain a reasoning item plus
       several function_call items) rather than a single message;
    3. The tool-call id field is called call_id, not id;
    4. Tool results go back as {"type": "function_call_output", "call_id", "output"}
       instead of a role="tool" message.

    Why it is worth switching: carrying the reasoning items from `output` back into
    the next request's `input` **verbatim** preserves the model's reasoning state
    across tool calls. Chat Completions throws it away every turn. OpenAI's
    migration guide reports roughly +3% on SWE-bench with the same prompt.

    State management here is "stateless resend": the whole context (including last
    turn's reasoning items) goes out each time. The alternative is
    previous_response_id plus deltas, which is cheaper — see roadmap #2.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        client: Any = None,
        reasoning_effort: str | None = None,
        prices: Price | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.prices = prices
        self.cache_key: str | None = None
        if client is None:
            from openai import OpenAI  # lazy import: offline runs need no openai

            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is missing. Set it in .env, or run offline "
                    "(drop --live)."
                )
            client = OpenAI()
        self.client = client

    @staticmethod
    def _flatten_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten Chat-shaped tool definitions into the Responses shape."""
        flat = []
        for t in tools:
            fn = t.get("function", t)
            flat.append(
                {
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn["parameters"],
                }
            )
        return flat

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "tools": self._flatten_tools(tools),
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.cache_key:  # route same-prefix requests together for better hits
            kwargs["prompt_cache_key"] = self.cache_key

        resp = self.client.responses.create(**kwargs)

        items: list[dict[str, Any]] = []
        calls: list[ToolCall] = []
        for item in resp.output:
            data = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
            items.append(data)  # reasoning items go back verbatim — that is the win
            if data.get("type") == "function_call":
                calls.append(
                    ToolCall(
                        id=data["call_id"],
                        name=data["name"],
                        arguments=data.get("arguments", "{}"),
                    )
                )

        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        cached = cached_from(usage, "input_tokens_details") if usage else 0
        return Reply(
            items=items,
            text=getattr(resp, "output_text", "") or "",
            tool_calls=calls,
            cost=estimate_cost(
                self.model,
                input_tokens,
                getattr(usage, "output_tokens", 0) if usage else 0,
                cached,
                self.prices,
            ),
            input_tokens=input_tokens,
            cached_tokens=cached,
        )

    def tool_result_item(self, call: ToolCall, result: str) -> dict[str, Any]:
        return {"type": "function_call_output", "call_id": call.id, "output": result}

    def set_cache_key(self, key: str) -> None:
        self.cache_key = key


# --------------------------------------------------------------------------
# Scripted model (offline demos / evals)
# --------------------------------------------------------------------------


def assistant_says(text: str, cost: float = 0.001) -> Reply:
    """Build a reply that answers directly and calls no tools."""
    return Reply(items=[{"role": "assistant", "content": text}], text=text, cost=cost)


def assistant_calls(calls: list[tuple[str, Any]], cost: float = 0.001) -> Reply:
    """Build a reply that requests tool calls.

    calls: [(tool name, arguments)]. Arguments may be a dict, or a raw string when
    you want to simulate malformed JSON.
    """
    tool_calls = []
    for i, (name, args) in enumerate(calls):
        raw = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        tool_calls.append(ToolCall(id=f"call_{i}", name=name, arguments=raw))
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ],
    }
    return Reply(items=[message], tool_calls=tool_calls, cost=cost)


class ScriptedModel:
    """Return scripted replies one by one; once the script runs out, keep returning
    a closing answer.

    It also answers the framework's own side calls — planning and compaction — so a
    script only has to cover the turns of the actual conversation. Those side calls
    are not recorded in `calls`, which keeps assertions about "what the model saw on
    turn N" honest.
    """

    def __init__(
        self,
        script: list[Reply],
        fallback: str = "(script exhausted)",
        plan_items: list[str] | None = None,
        reflection: dict[str, str] | None = None,
    ):
        self.script = list(script)
        self.fallback = fallback
        self.plan_items = plan_items or ["research the topic", "email the result"]
        self.reflection = reflection or {}  # {"experience": ...} and/or {"lesson": ...}
        self.summaries = 0
        self.calls: list[list[dict[str, Any]]] = []  # messages seen per call, for assertions
        self.tool_specs: list[list[dict[str, Any]]] = []  # tool list seen per call

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        first = str(messages[0].get("content", "")) if messages else ""
        if first.startswith("You are compacting"):
            self.summaries += 1
            return assistant_says("[summary] Facts A and B verified; attempt C failed.")
        if first.startswith("Break the user's request"):
            return assistant_says(json.dumps(self.plan_items, ensure_ascii=False))
        if first.startswith("Below is the full trajectory of one completed agent run"):
            return assistant_says(json.dumps(self.reflection, ensure_ascii=False))

        self.calls.append(list(messages))
        self.tool_specs.append(list(tools))
        if self.script:
            return self.script.pop(0)
        return assistant_says(self.fallback)

    def tool_result_item(self, call: ToolCall, result: str) -> dict[str, Any]:
        return chat_tool_result(call, result)


class AnthropicModel:
    """The Anthropic Messages API — a genuinely different request/response shape,
    added the same way `ResponsesModel` was added beside `OpenAIModel`: a new class
    behind the unchanged `Model` Protocol, with no change to the loop.

    `base_url` on an OpenAI-compatible endpoint (#15) already reaches a great many
    providers for free, and that is the right tool when a gateway speaks OpenAI's
    protocol. This is for when it does not. Five shape differences are sealed in here:

    1. **The system prompt is a parameter, not a message.** Messages carry only
       `user` and `assistant`. The loop writes several `role: "system"` entries — the
       persona up front, then run-status notes and completion push-backs mid-run — so
       the first becomes `system=` and the later ones become user turns. They are
       positional feedback about what just happened; hoisting them all to the top
       would move them away from the turn they are about.
    2. **Tools are flat and the schema key is `input_schema`,** not a nested
       `function` object with `parameters`.
    3. **Output is a list of content blocks** (`text`, `tool_use`), not one message
       with a separate `tool_calls` field.
    4. **A tool result is a `user` message** containing a `tool_result` block keyed by
       `tool_use_id` — not a role of its own.
    5. **Consecutive same-role messages must be merged.** A tool result followed by a
       run-status note is two user turns in this repo's model and one here.

    Cost: Anthropic models are not in `PRICES`, and inventing numbers that go stale is
    worse than saying so. Give the profile `price_input`/`price_cached`/`price_output`
    (#16) and the accounting is exact; leave them out and it falls back to the table's
    default guess, which will be wrong.
    """

    # Required by the API, unlike the OpenAI paths where it is optional.
    DEFAULT_MAX_TOKENS = 8192

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        client: Any = None,
        prices: Price | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.model = model
        self.prices = prices
        self.max_tokens = max_tokens
        self.cache_key: str | None = None
        if client is None:
            from anthropic import Anthropic  # lazy: offline runs need no SDK

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is missing. Set it in .env, or run offline "
                    "(no --live), or point the profile at a different provider."
                )
            client = Anthropic()
        self.client = client

    def set_cache_key(self, key: str) -> None:
        self.cache_key = key

    # -- shape translation --------------------------------------------------

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for t in tools:
            fn = t.get("function", t)
            out.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return out

    @staticmethod
    def _split(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Return (system prompt, messages) in Messages API shape.

        Only the *first* system entry becomes the system prompt; later ones are the
        loop's mid-run notes and stay where they are, as user turns.
        """
        system = ""
        converted: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                if not system and not converted:
                    system = str(m.get("content", ""))
                    continue
                role = "user"
            content = m.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif content is None:
                content = []
            converted.append({"role": role or "user", "content": list(content)})

        # Merge consecutive same-role turns; the API rejects them separately.
        merged: list[dict[str, Any]] = []
        for m in converted:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"].extend(m["content"])
            else:
                merged.append(m)
        return system, merged

    def tool_result_item(self, call: ToolCall, result: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call.id, "content": result}
            ],
        }

    # -- the call -----------------------------------------------------------

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        system, msgs = self._split(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": msgs,
        }
        if system:
            block: dict[str, Any] = {"type": "text", "text": system}
            if self.cache_key:
                # Anthropic's prompt cache is explicit, marked on the block rather than
                # inferred from a stable prefix the way OpenAI's is. The system prompt is
                # the one part that is identical on every turn of a run, so it is what a
                # cache key can actually buy anything on.
                block["cache_control"] = {"type": "ephemeral"}
            kwargs["system"] = [block]
        if tools:
            kwargs["tools"] = self._tools(tools)

        resp = self.client.messages.create(**kwargs)

        blocks = [
            b if isinstance(b, dict) else b.model_dump(exclude_none=True)
            for b in (getattr(resp, "content", None) or [])
        ]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = [
            ToolCall(
                id=b["id"],
                name=b["name"],
                # The loop hands `arguments` to json.loads; this API already parsed it.
                arguments=json.dumps(b.get("input") or {}, ensure_ascii=False),
            )
            for b in blocks
            if b.get("type") == "tool_use"
        ]

        usage = getattr(resp, "usage", None)
        fresh = getattr(usage, "input_tokens", 0) if usage else 0
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0 if usage else 0
        output = getattr(usage, "output_tokens", 0) if usage else 0
        # This API reports fresh and cached input separately; estimate_cost wants the
        # total with the cached part named inside it.
        total_input = fresh + cached
        return Reply(
            items=[{"role": "assistant", "content": blocks}],
            text=text,
            tool_calls=calls,
            cost=estimate_cost(self.model, total_input, output, cached, self.prices),
            input_tokens=total_input,
            cached_tokens=cached,
        )
