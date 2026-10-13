# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from data_designer.engine.models.clients.types import TransportKwargs
from data_designer.engine.models.facade import _COMPLETION_REQUEST_FIELDS, ModelFacade

import usersim
from usersim.engine.core import llm, realized_audit
from usersim.engine.core.llm import (
    ContextWindowError,
    _dicts_to_chat_messages,
    acall_llm,
    replay_reasoning_scope,
    set_current_outcome_builder,
)
from usersim.engine.core.outcomes import OutcomeBuilder, OutcomeStatus
from usersim.engine.probes.health_disclosure import move_runtime


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
# Thinking traces are stored, and replayed only to the assistant, on request
# ---------------------------------------------------------------------------


def test_trace_does_not_reach_the_chat_message() -> None:
    """By default, a stored thinking trace must not travel back into a model call.

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


class _RecordingFacade:
    """Records the wire form of every request it receives."""

    def __init__(self) -> None:
        self.payloads: list[list[dict]] = []

    async def acompletion(self, messages, **kwargs):
        self.payloads.append([m.to_dict() for m in messages])
        return SimpleNamespace(
            message=SimpleNamespace(content="ok", reasoning_content=None, tool_calls=None),
            usage=None,
        )


_REPLAY_ON = SimpleNamespace(replay_assistant_reasoning=True)
_REPLAY_OFF = SimpleNamespace(replay_assistant_reasoning=False)
_TRACE = "Plan: ask about sleep first."
_HISTORY = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer", "reasoning_content": _TRACE},
    {"role": "user", "content": "second question"},
]


async def _sent(alias: str, config) -> str:
    facade = _RecordingFacade()
    with replay_reasoning_scope(config):
        await acall_llm({alias: facade}, alias, _HISTORY)
    return str(facade.payloads)


async def test_replay_sends_the_assistant_its_earlier_reasoning() -> None:
    assert _TRACE in await _sent("assistant_model", _REPLAY_ON)
    assert _TRACE not in await _sent("assistant_model", _REPLAY_OFF)


@pytest.mark.parametrize("alias", ["user_model", "judge_model", "summary_model", "api_response_model"])
async def test_replay_never_sends_reasoning_to_another_model(alias: str) -> None:
    """Even a request that holds an assistant message with reasoning: only the assistant's own history replays."""
    assert _TRACE not in await _sent(alias, _REPLAY_ON)


async def test_the_replay_setting_is_restored_after_a_failure() -> None:
    with pytest.raises(RuntimeError), replay_reasoning_scope(_REPLAY_ON):
        raise RuntimeError("the episode failed")

    # A call outside any episode, in the same context, must not replay.
    facade = _RecordingFacade()
    await acall_llm({"assistant_model": facade}, "assistant_model", _HISTORY)
    assert _TRACE not in str(facade.payloads)


async def test_concurrent_episodes_each_keep_their_own_setting() -> None:
    import asyncio

    async def episode(config) -> str:
        facade = _RecordingFacade()
        with replay_reasoning_scope(config):
            await asyncio.sleep(0)
            await acall_llm({"assistant_model": facade}, "assistant_model", _HISTORY)
        return str(facade.payloads)

    on, off = await asyncio.gather(episode(_REPLAY_ON), episode(_REPLAY_OFF))

    assert _TRACE in on and _TRACE not in off


# ---------------------------------------------------------------------------
# Settings Data Designer's facade would drop
# ---------------------------------------------------------------------------

_HI = [{"role": "user", "content": "hi"}]


def _data_designer_facade(extra_body: dict | None = None, *, rejects_max_tokens: bool = False) -> MagicMock:
    """A facade of the type Data Designer builds, configured with ``extra_body``,
    that records each call and answers like ``_Facade`` or ``_RejectsMaxTokensFacade``."""
    answers = _RejectsMaxTokensFacade() if rejects_max_tokens else _Facade()
    facade = MagicMock(spec=ModelFacade)
    facade._model_config = SimpleNamespace(inference_parameters=SimpleNamespace(extra_body=extra_body, max_tokens=None))
    facade.calls = []

    async def acompletion(messages, **kwargs):
        facade.calls.append(kwargs)
        return await answers.acompletion(messages, **kwargs)

    facade.acompletion = acompletion
    return facade


class TestSettingsDataDesignerWouldDrop:
    """Data Designer forwards a fixed set of request fields and drops the rest
    without an error, so a reasoning setting only reaches the provider inside
    ``extra_body``."""

    async def test_a_setting_the_model_already_uses_is_sent_over_its_config(self) -> None:
        facade = _data_designer_facade({"reasoning_effort": "high", "top_k": 20})
        await acall_llm({"judge_model": facade}, "judge_model", _HI, reasoning_effort="low")
        (call,) = facade.calls
        assert "reasoning_effort" not in call
        assert call["extra_body"] == {"reasoning_effort": "low", "top_k": 20}

    async def test_a_setting_the_model_does_not_use_is_not_sent(self) -> None:
        """A provider can reject a setting outright, and the model config is the
        record of which ones this model takes."""
        facade = _data_designer_facade({"chat_template_kwargs": {"enable_thinking": True}})
        await acall_llm({"judge_model": facade}, "judge_model", _HI, reasoning_effort="low")
        (call,) = facade.calls
        assert "reasoning_effort" not in call
        assert "reasoning_effort" not in call.get("extra_body", {})

    async def test_chat_template_kwargs_merge_with_the_models_own(self) -> None:
        facade = _data_designer_facade({"chat_template_kwargs": {"enable_thinking": True, "keep": 1}})
        await acall_llm({"judge_model": facade}, "judge_model", _HI, chat_template_kwargs={"enable_thinking": False})
        (call,) = facade.calls
        assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False, "keep": 1}}

    async def test_the_max_tokens_retry_sends_its_budget_inside_extra_body(self) -> None:
        facade = _data_designer_facade(rejects_max_tokens=True)
        await acall_llm({"assistant_model": facade}, "assistant_model", _HI, max_tokens=4096)
        first, retry = facade.calls
        assert first["max_tokens"] == 4096
        assert "max_tokens" not in retry and "max_completion_tokens" not in retry
        assert retry["extra_body"] == {"max_completion_tokens": 4096}

    async def test_other_facades_get_the_arguments_as_given(self) -> None:
        """A hosted run's facade hands its parameters to the host, which applies them."""
        seen = []

        class _HostedFacade(_Facade):
            async def acompletion(self, messages, **kwargs):
                seen.append(kwargs)
                return await super().acompletion(messages, **kwargs)

        await acall_llm({"judge_model": _HostedFacade()}, "judge_model", _HI, reasoning_effort="low")
        assert seen == [{"reasoning_effort": "low"}]

    def test_data_designer_builds_the_settings_into_the_request_body(self) -> None:
        """Through Data Designer's own request building, so an upstream change
        to either step fails here instead of silently dropping the setting."""
        configured = {"reasoning_effort": "high"}
        kwargs = llm._request_kwargs(
            _data_designer_facade(configured), "judge_model", {"reasoning_effort": "low", "max_tokens": 100}
        )
        owner = SimpleNamespace(
            _model_config=SimpleNamespace(
                inference_parameters=SimpleNamespace(generate_kwargs={"extra_body": configured})
            ),
            model_provider=SimpleNamespace(name="nvidia", extra_body=None, extra_headers=None),
            model_name="vendor/model",
        )
        consolidated = ModelFacade.consolidate_kwargs(owner, **kwargs)
        request = ModelFacade._build_chat_completion_request(owner, _HI, consolidated)
        body = TransportKwargs.from_request(request).body
        assert body["reasoning_effort"] == "low"
        assert body["max_tokens"] == 100

    def test_data_designer_still_drops_the_settings_routed_around_it(self) -> None:
        """If Data Designer starts forwarding one of these itself, it can be
        passed as a plain argument."""
        forwarded = set(_COMPLETION_REQUEST_FIELDS) & {*llm._EXTRA_BODY_SETTINGS, "max_completion_tokens"}
        assert not forwarded, f"Data Designer now forwards {sorted(forwarded)}; stop routing them through extra_body."


def test_every_argument_passed_to_a_model_can_reach_it() -> None:
    """Data Designer drops keyword arguments outside its request fields without
    an error, so an argument added at a call site has to be one it forwards or
    one ``acall_llm`` routes."""
    root = Path(usersim.__file__).parent
    reaches = set(_COMPLETION_REQUEST_FIELDS) | set(llm._EXTRA_BODY_SETTINGS)
    offenders = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", getattr(node.func, "attr", None)) != "acall_llm":
                continue
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg in reaches:
                    continue
                # The default whenever tools are passed, so dropping it changes nothing.
                if keyword.arg == "tool_choice" and getattr(keyword.value, "value", None) == "auto":
                    continue
                offenders.append(f"{path.relative_to(root)}:{node.lineno} {keyword.arg}")
    assert not offenders, f"these arguments never reach the model: {offenders}"


@pytest.mark.parametrize("value", ["off", "low", "medium", "high"])
def test_the_reasoning_overrides_use_only_settings_acall_llm_routes(monkeypatch, value) -> None:
    monkeypatch.setenv("USERSIM_AUDIT_REASONING_EFFORT", value)
    monkeypatch.setattr(move_runtime, "_MOVE_REASONING_EFFORT", value)
    for produced in (realized_audit._audit_reasoning_kwargs(), move_runtime._move_reasoning_kwargs()):
        assert produced and set(produced) <= set(llm._EXTRA_BODY_SETTINGS)
