# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for core/llm.py resource accounting.

Reasoning-token accounting is intentionally NOT done at sim time -- DD's
normalised ``Usage`` dataclass exposes only ``input_tokens`` /
``output_tokens`` / ``total_tokens`` and strips provider-specific
reasoning telemetry (OpenAI's ``completion_tokens_details``, Gemini's
``thoughts_token_count``, etc.) at the abstraction boundary. Reasoning
is estimated post-hoc at report-build time via tiktoken on the
trajectory's visible content. See ``reporting/_reasoning.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from usersim.engine.core import llm
from usersim.engine.core.llm import (
    ContextWindowError,
    _dicts_to_chat_messages,
    acall_llm,
    set_current_outcome_builder,
)
from usersim.engine.core.outcomes import OutcomeBuilder, OutcomeStatus


async def _noop_sleep(_delay):
    """Stand in for the awaited backoff, returning at once."""
    return None


def _collect(into):
    """Record each backoff without waiting it out."""

    def _sleep(delay):
        into.append(delay)
        return None

    return _sleep


class TestAsyncModelCalls:
    """The awaiting call path, which shares its retry policy with the sync one."""

    class _Facade:
        def __init__(self, failures: int = 0) -> None:
            self.attempts = 0
            self._failures = failures

        async def acompletion(self, messages, **kwargs):
            self.attempts += 1
            if self.attempts <= self._failures:
                raise RuntimeError("transient")
            return SimpleNamespace(
                message=SimpleNamespace(content="ok", reasoning_content=None, tool_calls=None),
                usage=SimpleNamespace(input_tokens=3, output_tokens=4),
            )

    async def test_returns_the_reply_as_a_message_dict(self) -> None:
        from usersim.engine.core.llm import acall_llm

        result = await acall_llm({"m": self._Facade()}, "m", [{"role": "user", "content": "hi"}])

        assert result["role"] == "assistant"
        assert result["content"] == "ok"

    async def test_closes_nested_json_schema_objects_for_strict_providers(self) -> None:
        class CapturingFacade(self._Facade):
            kwargs: dict

            async def acompletion(self, messages, **kwargs):
                self.kwargs = kwargs
                return await super().acompletion(messages, **kwargs)

        schema = {
            "type": "object",
            "properties": {"result": {"$ref": "#/$defs/result"}},
            "$defs": {
                "result": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "$ref": "#/$defs/score",
                            "description": "Quality score",
                        }
                    },
                    "required": ["score"],
                },
                "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            },
            "required": ["result"],
        }
        facade = CapturingFacade()

        await acall_llm(
            {"m": facade},
            "m",
            [{"role": "user", "content": "hi"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "evaluation", "schema": schema},
            },
        )

        sent_schema = facade.kwargs["response_format"]["json_schema"]["schema"]
        assert sent_schema["additionalProperties"] is False
        assert sent_schema["$defs"]["result"]["additionalProperties"] is False
        score_schema = sent_schema["$defs"]["result"]["properties"]["score"]
        assert score_schema == {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "Quality score",
        }
        assert "additionalProperties" not in schema
        assert "additionalProperties" not in schema["$defs"]["result"]
        assert schema["$defs"]["result"]["properties"]["score"]["$ref"] == "#/$defs/score"

    async def test_waits_without_blocking_the_runtime(self) -> None:
        """Backoff has to yield, or every other conversation waits with it."""
        import asyncio

        from usersim.engine.core import llm as llm_module

        slept: list[float] = []

        async def _record(delay: float) -> None:
            slept.append(delay)

        facade = self._Facade(failures=1)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", _record)
            mp.setattr(llm_module.time, "sleep", lambda _: pytest.fail("backoff blocked the runtime"))
            result = await llm_module.acall_llm({"m": facade}, "m", [{"role": "user", "content": "hi"}])

        assert facade.attempts == 2, f"expected one retry, made {facade.attempts} attempts"
        assert slept, "retried with no backoff at all"
        assert result["content"] == "ok"

    async def test_a_defective_call_is_not_retried(self) -> None:
        """A call built wrong cannot be repaired by sending it again."""
        from usersim.engine.core.llm import acall_llm

        class _Broken:
            def __init__(self) -> None:
                self.attempts = 0

            async def acompletion(self, messages, **kwargs):
                self.attempts += 1
                raise AttributeError("no such method")

        facade = _Broken()
        with pytest.raises(AttributeError):
            await acall_llm({"m": facade}, "m", [{"role": "user", "content": "hi"}])
        assert facade.attempts == 1, f"tried {facade.attempts} times for a failure no retry can fix"


class TestConversationStateIsPerContext:
    """Conversation state follows the unit of work, not the runtime thread.

    Two trajectories can share a thread. When they do, state keyed by thread
    identity collapses into one slot and each trajectory's model calls are
    attributed to whichever ran most recently.
    """

    @staticmethod
    def _in_its_own_context(fn, *args):
        import contextvars

        return contextvars.copy_context().run(fn, *args)

    def test_one_context_cannot_see_another_conversation_id(self) -> None:
        from usersim.engine.core.llm import get_conversation_id, set_conversation_id

        observed: dict[str, str | None] = {}

        def work(name: str) -> None:
            set_conversation_id(name)
            observed[name] = get_conversation_id()

        self._in_its_own_context(work, "traj-a")
        self._in_its_own_context(work, "traj-b")

        assert observed == {"traj-a": "traj-a", "traj-b": "traj-b"}
        assert get_conversation_id() is None, (
            f"a trajectory's conversation id escaped into the caller as "
            f"{get_conversation_id()!r}, so the next one inherits it"
        )

    def test_one_context_cannot_see_another_outcome_builder(self) -> None:
        from usersim.engine.core.llm import get_current_outcome_builder, set_current_outcome_builder
        from usersim.engine.core.outcomes import OutcomeBuilder, Provenance

        def work() -> None:
            set_current_outcome_builder(OutcomeBuilder(provenance=Provenance()))

        self._in_its_own_context(work)

        assert get_current_outcome_builder() is None, (
            "a trajectory's outcome builder escaped into the caller, so calls "
            "made after it record against the wrong trajectory"
        )

    def test_a_buffer_is_not_inherited_from_the_caller(self) -> None:
        """Naming a conversation gives it a buffer of its own.

        Copying a context copies the reference to the buffer, not the buffer,
        so a conversation that reused an inherited one would append into
        whatever its caller was still holding.
        """
        from usersim.engine.core import llm as llm_module
        from usersim.engine.core.llm import append_debug_record, set_conversation_id

        set_conversation_id("outer")
        append_debug_record({"alias": "user_model"})
        outer_buffer = llm_module._pending_debug_records()

        def inner() -> list:
            set_conversation_id("inner")
            append_debug_record({"alias": "judge_model"})
            return llm_module._pending_debug_records()

        inner_buffer = self._in_its_own_context(inner)

        assert inner_buffer is not outer_buffer, "both conversations share one buffer"
        assert len(outer_buffer) == 1, f"the caller's buffer grew to {len(outer_buffer)} records"
        set_conversation_id(None)

    def test_a_flush_drains_only_its_own_records(self) -> None:
        """One trajectory's flush must not consume another's pending records."""
        from usersim.engine.core.llm import append_debug_record, flush_debug_log, set_conversation_id

        drained: dict[str, int] = {}

        def record_two_then_flush(name: str) -> None:
            set_conversation_id(name)
            append_debug_record({"alias": "user_model"})
            append_debug_record({"alias": "judge_model"})
            from usersim.engine.core import llm as llm_module

            pending = llm_module._pending_debug_records()
            drained[name] = len(pending)
            flush_debug_log()

        self._in_its_own_context(record_two_then_flush, "traj-a")
        self._in_its_own_context(record_two_then_flush, "traj-b")

        assert drained == {"traj-a": 2, "traj-b": 2}, f"a trajectory saw records that were not its own: {drained}"


class _Facade:
    async def acompletion(self, messages, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content="ok",
                reasoning_content=None,
                tool_calls=None,
            ),
            usage=SimpleNamespace(input_tokens=12, output_tokens=7),
        )


class _RejectsMaxTokensFacade:
    """A reasoning endpoint: refuses ``max_tokens``, takes the other name."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def acompletion(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        if "max_tokens" in kwargs:
            raise RuntimeError(
                "ProviderError: Unsupported parameter: 'max_tokens' is not "
                "supported with this model. Use 'max_completion_tokens' instead."
            )
        return SimpleNamespace(
            message=SimpleNamespace(content="ok", reasoning_content=None, tool_calls=None),
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


class _BrokenFacade:
    """Stands in for a malformed models dict: no ``completion`` to call."""

    def __init__(self) -> None:
        self.attempts = 0

    async def acompletion(self, messages, **kwargs):
        self.attempts += 1
        raise AttributeError("'object' object has no attribute 'completion'")


async def test_a_malformed_call_fails_immediately_instead_of_backing_off() -> None:
    """Retrying cannot repair a call that was built wrong.

    The backoff is there for a provider having a bad minute. Spending it on
    an AttributeError delays the traceback by the full budget and buries the
    cause under warnings blaming a provider that was never asked anything.
    """
    facade = _BrokenFacade()
    slept: list[float] = []

    with patch.object(llm.asyncio, "sleep", _collect(slept)), pytest.raises(AttributeError):
        await acall_llm({"summary_model": facade}, "summary_model", [{"role": "user", "content": "hi"}])

    assert facade.attempts == 1, f"tried {facade.attempts} times; a defective call should be attempted once"
    assert slept == [], f"backed off {slept} before reporting a failure that no retry could fix"


async def test_transient_failures_are_still_retried() -> None:
    """The guard above must not disable the backoff it sits next to."""
    calls: list[int] = []

    class _FlakyFacade:
        async def acompletion(self, messages, **kwargs):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("503 Service Unavailable")
            return SimpleNamespace(
                message=SimpleNamespace(content="ok", reasoning_content=None, tool_calls=None),
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    with patch.object(llm.asyncio, "sleep", _noop_sleep):
        result = await acall_llm({"m": _FlakyFacade()}, "m", [{"role": "user", "content": "hi"}])

    assert result["content"] == "ok"
    assert len(calls) == 3


def test_non_ascii_budget_scales_up_never_down() -> None:
    """Denser scripts must get MORE room than English, not less.

    This was inverted: the call site substituted a constant 4096, which was a
    raise when written and silently became a cap once the catalog default
    rose above it. Japanese, Korean and Devanagari then ran on half of what
    English got, and the resulting truncation reads as a model quality
    failure rather than a budget one.
    """
    from usersim.engine.core.llm import NON_ASCII_TOKEN_SCALE, scaled_max_tokens

    def facade(max_tokens):
        return SimpleNamespace(
            _model_config=SimpleNamespace(inference_parameters=SimpleNamespace(max_tokens=max_tokens))
        )

    for configured in (2048, 4096, 8192):
        scaled = scaled_max_tokens(facade(configured), NON_ASCII_TOKEN_SCALE)
        assert scaled["max_tokens"] > configured, f"non-ASCII budget {scaled} must exceed the configured {configured}"

    # A model that sets no budget must not acquire one: that is how reasoning
    # models are configured, and they reject the parameter outright.
    assert scaled_max_tokens(facade(None), NON_ASCII_TOKEN_SCALE) == {}


async def test_call_llm_translates_max_tokens_for_reasoning_models() -> None:
    """Callers pass a token budget without knowing the model's spelling.

    The engine injects ``max_tokens`` at the call site for non-ASCII locales,
    where denser tokenization truncates replies. Reasoning endpoints reject
    that field outright, which is a permanent error: retrying unchanged just
    burned all three attempts and failed the turn.
    """
    facade = _RejectsMaxTokensFacade()
    result = await acall_llm(
        {"assistant_model": facade},
        "assistant_model",
        [{"role": "user", "content": "hi"}],
        max_tokens=4096,
    )
    assert result["content"] == "ok"
    assert len(facade.calls) == 2, "should retry once, not exhaust the backoff"
    assert facade.calls[0]["max_tokens"] == 4096
    assert facade.calls[1]["max_completion_tokens"] == 4096
    assert "max_tokens" not in facade.calls[1]


async def test_call_llm_records_data_designer_usage_tokens() -> None:
    builder = OutcomeBuilder()
    set_current_outcome_builder(builder)
    try:
        result = await acall_llm(
            {"judge_model": _Facade()},
            "judge_model",
            [{"role": "user", "content": "hello"}],
        )
    finally:
        set_current_outcome_builder(None)

    assert result["content"] == "ok"
    outcome = builder.finalize(status=OutcomeStatus.OK)
    assert outcome.per_model_calls == {"judge_model": 1}
    assert outcome.per_model_input_tokens == {"judge_model": 12}
    assert outcome.per_model_output_tokens == {"judge_model": 7}


class _OpenAIShapedFacade:
    """Provider that exposes prompt_tokens / completion_tokens (the
    OpenAI-shaped names DD's ``extract_usage`` accepts as a fallback)."""

    async def acompletion(self, messages, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content="ok",
                reasoning_content=None,
                tool_calls=None,
            ),
            # OpenAI shape: prompt_tokens / completion_tokens fall back
            # to input_tokens / output_tokens via getattr-or-getattr.
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=15),
        )


async def test_call_llm_falls_back_to_openai_shaped_token_names() -> None:
    """The wrapper accepts either DD-canonical (input/output_tokens) or
    OpenAI-shaped (prompt/completion_tokens) Usage objects. This is
    behaviour we share with DD's own ``extract_usage`` parser and care
    about because some test mocks emit the OpenAI shape directly."""
    builder = OutcomeBuilder()
    set_current_outcome_builder(builder)
    try:
        await acall_llm(
            {"assistant_model": _OpenAIShapedFacade()},
            "assistant_model",
            [{"role": "user", "content": "hello"}],
        )
    finally:
        set_current_outcome_builder(None)

    outcome = builder.finalize(status=OutcomeStatus.OK)
    assert outcome.per_model_input_tokens == {"assistant_model": 20}
    assert outcome.per_model_output_tokens == {"assistant_model": 15}


class _ReasoningFieldFacade:
    """Provider that exposes reasoning fields the wrapper deliberately
    does NOT read (regression lock that we don't accidentally re-enable
    Phase A capture against fields DD would have stripped anyway)."""

    async def acompletion(self, messages, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content="ok",
                reasoning_content=None,
                tool_calls=None,
            ),
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=80,
                # These fields would matter under Phase A capture, but
                # DD's normalised Usage strips them. We make sure the
                # wrapper doesn't try to read them either.
                completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=42,
                ),
            ),
        )


async def test_call_llm_ignores_provider_reasoning_telemetry() -> None:
    """Regression lock: even if a mock exposes
    ``usage.completion_tokens_details.reasoning_tokens``, the wrapper
    must NOT attempt to capture it. Reasoning lives at the report-build
    layer (post-hoc tiktoken on visible content); doing it at sim time
    would tie us to a field DD strips before we see it in real runs."""
    builder = OutcomeBuilder()
    set_current_outcome_builder(builder)
    try:
        await acall_llm(
            {"assistant_model": _ReasoningFieldFacade()},
            "assistant_model",
            [{"role": "user", "content": "hello"}],
        )
    finally:
        set_current_outcome_builder(None)

    outcome = builder.finalize(status=OutcomeStatus.OK)
    # Output tokens are still the gross 80 (correct).
    assert outcome.per_model_output_tokens == {"assistant_model": 80}
    # Critically: SimulationOutcome no longer carries a per-model
    # reasoning-tokens dict at all.
    assert not hasattr(outcome, "per_model_reasoning_tokens")


async def test_context_window_error_is_not_retried() -> None:
    class _ContextFacade:
        calls = 0

        async def acompletion(self, messages, **kwargs):
            self.calls += 1
            raise RuntimeError(
                "ProviderError: litellm.ContextWindowExceededError: "
                "This model's maximum context length is 32768 tokens."
            )

    facade = _ContextFacade()
    with patch("usersim.engine.core.llm.asyncio.sleep", new_callable=AsyncMock) as sleep:
        with pytest.raises(ContextWindowError) as exc:
            await acall_llm(
                {"assistant_model": facade},
                "assistant_model",
                [{"role": "user", "content": "large prompt"}],
            )
    assert exc.value.alias == "assistant_model"
    assert facade.calls == 1
    sleep.assert_not_called()


async def test_transient_error_still_retries() -> None:
    class _FlakyFacade:
        calls = 0

        async def acompletion(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary endpoint failure")
            return SimpleNamespace(
                message=SimpleNamespace(
                    content="ok",
                    reasoning_content=None,
                    tool_calls=None,
                ),
                usage=None,
            )

    facade = _FlakyFacade()
    with patch("usersim.engine.core.llm.asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await acall_llm(
            {"assistant_model": facade},
            "assistant_model",
            [{"role": "user", "content": "hello"}],
        )
    assert result["content"] == "ok"
    assert facade.calls == 2
    sleep.assert_called_once()


# ---------------------------------------------------------------------------
# Thinking traces are stored, never replayed
# ---------------------------------------------------------------------------


def test_trace_does_not_reach_the_chat_message() -> None:
    """A stored thinking trace must not travel back into a model call.

    Trajectories keep ``reasoning_content`` on the assistant message for
    analysis, so the guarantee lives at the conversion boundary: whatever
    is on the stored message, what leaves ``_dicts_to_chat_messages`` for
    a model call carries no prior thinking. Pinned here because this is
    the narrowest point the claim can be made.

    Asserted as falsy rather than ``is None`` because the wire contract is
    what matters: ``ChatMessage.to_dict`` must omit the key entirely.
    """
    trace = "Let me think. The user wants X, so I do Y."
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer", "reasoning_content": trace},
        {"role": "user", "content": "second question"},
    ]
    chat = _dicts_to_chat_messages(messages)

    assistants = [m for m in chat if m.role == "assistant"]
    assert assistants, "fixture must contain an assistant message"
    assert not any(m.reasoning_content for m in assistants)
    # The wire payload is what actually matters.
    payload = [m.to_dict() for m in chat]
    assert not any("reasoning_content" in m for m in payload)
    assert trace not in str(payload)
