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

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from usersim.engine.core.llm import (
    ContextWindowError,
    _dicts_to_chat_messages,
    call_llm,
    set_current_outcome_builder,
)
from usersim.engine.core.outcomes import OutcomeBuilder, OutcomeStatus


class _Facade:
    def completion(self, messages, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content="ok",
                reasoning_content=None,
                tool_calls=None,
            ),
            usage=SimpleNamespace(input_tokens=12, output_tokens=7),
        )


def test_call_llm_records_data_designer_usage_tokens() -> None:
    builder = OutcomeBuilder()
    set_current_outcome_builder(builder)
    try:
        result = call_llm(
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

    def completion(self, messages, **kwargs):
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


def test_call_llm_falls_back_to_openai_shaped_token_names() -> None:
    """The wrapper accepts either DD-canonical (input/output_tokens) or
    OpenAI-shaped (prompt/completion_tokens) Usage objects. This is
    behaviour we share with DD's own ``extract_usage`` parser and care
    about because some test mocks emit the OpenAI shape directly."""
    builder = OutcomeBuilder()
    set_current_outcome_builder(builder)
    try:
        call_llm(
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

    def completion(self, messages, **kwargs):
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


def test_call_llm_ignores_provider_reasoning_telemetry() -> None:
    """Regression lock: even if a mock exposes
    ``usage.completion_tokens_details.reasoning_tokens``, the wrapper
    must NOT attempt to capture it. Reasoning lives at the report-build
    layer (post-hoc tiktoken on visible content); doing it at sim time
    would tie us to a field DD strips before we see it in real runs."""
    builder = OutcomeBuilder()
    set_current_outcome_builder(builder)
    try:
        call_llm(
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


def test_context_window_error_is_not_retried() -> None:
    class _ContextFacade:
        calls = 0

        def completion(self, messages, **kwargs):
            self.calls += 1
            raise RuntimeError(
                "ProviderError: litellm.ContextWindowExceededError: "
                "This model's maximum context length is 32768 tokens."
            )

    facade = _ContextFacade()
    with patch("usersim.engine.core.llm.time.sleep") as sleep:
        with pytest.raises(ContextWindowError) as exc:
            call_llm(
                {"assistant_model": facade},
                "assistant_model",
                [{"role": "user", "content": "large prompt"}],
            )
    assert exc.value.alias == "assistant_model"
    assert facade.calls == 1
    sleep.assert_not_called()


def test_transient_error_still_retries() -> None:
    class _FlakyFacade:
        calls = 0

        def completion(self, messages, **kwargs):
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
    with patch("usersim.engine.core.llm.time.sleep") as sleep:
        result = call_llm(
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
        {"role": "assistant", "content": "first answer",
         "reasoning_content": trace},
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
