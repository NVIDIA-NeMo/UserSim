# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for judge-gated assistant-turn resampling.

Covers ``ConversationLoop._generate_and_judge_assistant_turn``:
the default single-attempt path (behavior preservation), the
resample-on-rejection path, budget exhaustion, transcript rollback,
the probe-level ``supports_assistant_resampling`` opt-out, and the
``n_assistant_retries`` outcome counter.
"""

from __future__ import annotations

from dataclasses import dataclass

import usersim.engine.core.simulation as sim
from usersim.engine.core.outcomes import OutcomeBuilder, OutcomeStatus
from usersim.engine.core.probes import (
    AgenticMixin,
    BaseProbe,
    ToolCallingMixin,
    assistant_message,
)
from usersim.engine.core.simulation import (
    ConversationLoop,
    ConversationState,
)


@dataclass
class _StubProbe:
    """Minimal probe: appends exactly one assistant message (BaseProbe
    default shape), no side effects — resampling-safe."""

    label: str = "stub_probe"
    supports_assistant_resampling: bool = True

    def after_assistant_turn(self, models, state, response, cfg):
        content = response.get("content", "")
        state.messages.append({"role": "assistant", "content": content})
        return content

    def format_assistant_judge_prompt(self, assistant_response, conversation_history):
        """Part of the ProbeAdapter protocol; BaseProbe supplies the real one."""
        return f"judge: {assistant_response}"


@dataclass
class _StubConfig:
    max_assistant_attempts: int = 1
    store_reasoning: bool = True


def _make_state() -> ConversationState:
    state = ConversationState(outcome=OutcomeBuilder())
    state.messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q1"},
    ]
    return state


class _Harness:
    """Monkeypatch bundle: scripted assistant responses + judge verdicts."""

    def __init__(
        self,
        monkeypatch,
        responses: list[str],
        verdicts: list[bool],
        traces: list[str] | None = None,
    ):
        self.assistant_calls = 0
        self.judge_calls = 0
        #: Prompt text the loop actually handed the judge, per call.
        self.judge_prompts: list[str] = []

        def fake_call_llm(models, alias, messages, **kwargs):
            i = min(self.assistant_calls, len(responses) - 1)
            content = responses[i]
            self.assistant_calls += 1
            resp = {"content": content}
            if traces is not None:
                resp["reasoning_content"] = traces[min(i, len(traces) - 1)]
            return resp

        def fake_judge_ex(models, alias, prompt):
            ok = verdicts[min(self.judge_calls, len(verdicts) - 1)]
            self.judge_prompts.append(prompt)
            self.judge_calls += 1
            rating = "success" if ok else "failure"
            # parse_ok=True: scripted verdicts are genuine judgments,
            # not parser defaults (judge flakiness is tested separately).
            return f"verdict {self.judge_calls}", rating, ok, True

        monkeypatch.setattr(sim, "call_llm", fake_call_llm)
        monkeypatch.setattr(sim, "run_inline_judge_ex", fake_judge_ex)


def _run(state, cfg, probe):
    loop = ConversationLoop()
    return loop._generate_and_judge_assistant_turn(
        models={},
        probe=probe,
        state=state,
        cfg=cfg,
        turn_idx=0,
        assistant_msgs=list(state.messages),
        call_kwargs={},
    )


class TestDefaultSingleAttempt:
    """max_assistant_attempts=1 must reproduce the original behavior."""

    def test_one_call_one_judgment_on_pass(self, monkeypatch):
        h = _Harness(monkeypatch, ["good answer"], [True])
        state = _make_state()
        content, expl, rating, ok = _run(state, _StubConfig(1), _StubProbe())
        assert (h.assistant_calls, h.judge_calls) == (1, 1)
        assert ok and rating == "success" and content == "good answer"
        assert state.messages[-1]["content"] == "good answer"

    def test_failed_judgment_is_flag_only_no_resample(self, monkeypatch):
        h = _Harness(monkeypatch, ["bad answer"], [False])
        state = _make_state()
        content, _, rating, ok = _run(state, _StubConfig(1), _StubProbe())
        assert h.assistant_calls == 1
        assert not ok and rating == "failure"
        # The flagged candidate stays in the transcript (flag-only).
        assert state.messages[-1]["content"] == "bad answer"
        out = state.outcome.finalize(OutcomeStatus.OK)
        assert out.n_assistant_retries == 0

    def test_metadata_entry_has_attempt_index(self, monkeypatch):
        _Harness(monkeypatch, ["a"], [True])
        state = _make_state()
        _run(state, _StubConfig(1), _StubProbe())
        ratings = state.metadata["assistant_judge_ratings"]
        assert len(ratings) == 1
        assert ratings[0]["turn_idx"] == 0
        assert ratings[0]["attempt"] == 0
        assert ratings[0]["success"] is True


class TestResampling:
    def test_rejected_candidate_is_replaced(self, monkeypatch):
        h = _Harness(monkeypatch, ["bad", "bad2", "good"], [False, False, True])
        state = _make_state()
        content, _, _, ok = _run(state, _StubConfig(3), _StubProbe())
        assert (h.assistant_calls, h.judge_calls) == (3, 3)
        assert ok and content == "good"
        # Transcript contains ONLY the accepted candidate.
        assistant_msgs = [m for m in state.messages if m["role"] == "assistant"]
        assert assistant_msgs == [{"role": "assistant", "content": "good"}]

    def test_all_attempts_logged_with_attempt_indices(self, monkeypatch):
        _Harness(monkeypatch, ["bad", "good"], [False, True])
        state = _make_state()
        _run(state, _StubConfig(3), _StubProbe())
        ratings = state.metadata["assistant_judge_ratings"]
        assert [r["attempt"] for r in ratings] == [0, 1]
        assert [r["success"] for r in ratings] == [False, True]

    def test_retry_counter_counts_discarded_candidates(self, monkeypatch):
        _Harness(monkeypatch, ["bad", "good"], [False, True])
        state = _make_state()
        _run(state, _StubConfig(3), _StubProbe())
        out = state.outcome.finalize(OutcomeStatus.OK)
        assert out.n_assistant_retries == 1

    def test_stops_early_on_first_pass(self, monkeypatch):
        h = _Harness(monkeypatch, ["good"], [True])
        state = _make_state()
        _run(state, _StubConfig(3), _StubProbe())
        assert (h.assistant_calls, h.judge_calls) == (1, 1)

    def test_exhaustion_keeps_last_candidate_flagged(self, monkeypatch):
        h = _Harness(monkeypatch, ["bad1", "bad2", "bad3"], [False, False, False])
        state = _make_state()
        content, _, _, ok = _run(state, _StubConfig(3), _StubProbe())
        assert h.assistant_calls == 3
        assert not ok and content == "bad3"
        # Last candidate kept (same semantics as flag-only), budget spent.
        assert state.messages[-1]["content"] == "bad3"
        out = state.outcome.finalize(OutcomeStatus.OK)
        # Only discarded-and-regenerated candidates count as retries.
        assert out.n_assistant_retries == 2

    def test_transcript_rollback_leaves_prior_turns_intact(self, monkeypatch):
        _Harness(monkeypatch, ["bad", "good"], [False, True])
        state = _make_state()
        prior = list(state.messages)
        _run(state, _StubConfig(2), _StubProbe())
        assert state.messages[: len(prior)] == prior
        assert len(state.messages) == len(prior) + 1


class TestJudgePromptComesFromTheProbe:
    """The loop must judge with the PROBE's prompt, not a module constant."""

    def test_loop_judges_with_the_probes_prompt(self, monkeypatch):
        h = _Harness(monkeypatch, ["the answer"], [True])
        state = _make_state()
        _run(state, _StubConfig(1), _StubProbe())

        # _StubProbe returns f"judge: {assistant_response}" -- a shape no
        # module constant produces, so a match can only come from the hook.
        assert h.judge_prompts == ["judge: the answer"]

    def test_every_resample_attempt_re_renders_it(self, monkeypatch):
        """Each attempt judges its own candidate, so the prompt is rebuilt
        per attempt rather than computed once and reused."""
        h = _Harness(monkeypatch, ["bad", "good"], [False, True])
        state = _make_state()
        _run(state, _StubConfig(2), _StubProbe())

        assert h.judge_prompts == ["judge: bad", "judge: good"]


class TestProbeOptOut:
    def test_unsafe_probe_capped_to_one_attempt(self, monkeypatch):
        h = _Harness(monkeypatch, ["bad"], [False])
        state = _make_state()
        probe = _StubProbe(supports_assistant_resampling=False)
        _run(state, _StubConfig(3), probe)
        assert h.assistant_calls == 1
        out = state.outcome.finalize(OutcomeStatus.OK)
        assert out.n_assistant_retries == 0

    def test_base_probe_is_resampling_safe(self):
        assert BaseProbe.supports_assistant_resampling is True

    def test_tool_calling_mixin_opts_out(self):
        assert ToolCallingMixin.supports_assistant_resampling is False

    def test_agentic_mixin_opts_out(self):
        assert AgenticMixin.supports_assistant_resampling is False

    def test_mixin_wins_in_mro(self):
        class _ToolProbe(ToolCallingMixin, BaseProbe):
            pass

        assert _ToolProbe.supports_assistant_resampling is False


class _TracingProbe:
    """Attributes the trace per call via ``assistant_message()``.

    ``_StubProbe`` hand-builds its dict and so exercises the
    ``ConversationLoop`` fallback; this one exercises the per-call path
    and the fallback's "already attributed" guard.
    """

    label: str = "tracing_probe"
    supports_assistant_resampling: bool = True

    def after_assistant_turn(self, models, state, response, cfg):
        content = response.get("content", "")
        state.messages.append(
            assistant_message(
                response,
                content,
                store_reasoning=getattr(cfg, "store_reasoning", True),
            )
        )
        return content

    def format_assistant_judge_prompt(self, assistant_response, conversation_history):
        """Part of the ProbeAdapter protocol; BaseProbe supplies the real one."""
        return f"judge: {assistant_response}"


class _MultiCallProbe:
    """One turn, several assistant messages from several calls.

    The tool-calling shape: the loop makes the first call, the probe
    makes the rest internally. Each message is attributed with the trace
    of the call that produced it.
    """

    label: str = "multi_call_probe"
    supports_assistant_resampling: bool = True
    SECOND_TRACE: str = "TRACE_SECOND_CALL"

    def after_assistant_turn(self, models, state, response, cfg):
        # The loop's call (a tool call, in the real shape).
        store = getattr(cfg, "store_reasoning", True)
        state.messages.append(
            assistant_message(
                response,
                response.get("content", ""),
                store_reasoning=store,
            )
        )
        # A second, probe-internal call: the synthesised reply.
        second = {"content": "final synthesis", "reasoning_content": self.SECOND_TRACE}
        state.messages.append(
            assistant_message(
                second,
                second["content"],
                store_reasoning=store,
            )
        )
        return second["content"]

    def format_assistant_judge_prompt(self, assistant_response, conversation_history):
        """Part of the ProbeAdapter protocol; BaseProbe supplies the real one."""
        return f"judge: {assistant_response}"


class TestRejectedCandidateTraces:
    """A rolled-back candidate must take its thinking trace with it.

    This behaviour spans two changes and is owned by neither: the
    rollback (``del state.messages[messages_checkpoint:]``) is the
    resampling work, and what rides on the deleted message is the
    trace-capture work. ``messages_checkpoint`` is the only thing joining
    them, so these tests sit on that seam.
    """

    A, B, C = "TRACE_A", "TRACE_B", "TRACE_C"

    def _assistants(self, state):
        return [m for m in state.messages if m["role"] == "assistant"]

    @staticmethod
    def _no_trace(msg) -> bool:
        """ "No trace" is any falsy reading: key absent, None, or "".

        The capture path omits the key entirely, but the property under
        test is that nothing readable survives -- an implementation that
        wrote an explicit null or empty string would satisfy it just as
        well, and should not fail these tests.
        """
        return not msg.get("reasoning_content")

    def test_rejected_candidates_trace_is_discarded(self, monkeypatch):
        """The surviving message carries its OWN trace, and the rejected
        candidate's trace is gone from the transcript entirely."""
        _Harness(monkeypatch, ["bad", "good"], [False, True], traces=[self.A, self.B])
        state = _make_state()
        _run(state, _StubConfig(2), _StubProbe())

        assistants = self._assistants(state)
        assert [m["content"] for m in assistants] == ["good"]
        assert assistants[0]["reasoning_content"] == self.B
        assert self.A not in str(state.messages)

    def test_scoped_to_this_attempt_not_the_whole_transcript(self, monkeypatch):
        """The fallback's "already attributed?" check must look only at
        messages THIS attempt appended.

        Widen it to the whole transcript and an earlier turn's trace
        satisfies the check, so the fallback skips and the current turn
        silently loses its trace. A prior traced message is therefore
        load-bearing here, not decoration.
        """
        _Harness(monkeypatch, ["good"], [True], traces=[self.B])
        state = _make_state()
        # A prior, already-settled turn that DID emit a trace.
        state.messages.append(
            {
                "role": "assistant",
                "content": "earlier reply",
                "reasoning_content": self.A,
            }
        )
        state.messages.append({"role": "user", "content": "Q2"})
        prior = state.messages[2]

        _run(state, _StubConfig(2), _StubProbe())

        # This turn still got its own trace ...
        survivor = self._assistants(state)[-1]
        assert survivor["content"] == "good"
        assert survivor["reasoning_content"] == self.B
        # ... and the earlier turn's is untouched.
        assert prior["reasoning_content"] == self.A

    def test_exhaustion_keeps_the_last_trace(self, monkeypatch):
        """Budget spent with every candidate rejected: the kept message
        carries the last attempt's trace, not the first's."""
        _Harness(monkeypatch, ["bad1", "bad2", "bad3"], [False, False, False], traces=[self.A, self.B, self.C])
        state = _make_state()
        _run(state, _StubConfig(3), _StubProbe())

        assistants = self._assistants(state)
        assert [m["content"] for m in assistants] == ["bad3"]
        assert assistants[0]["reasoning_content"] == self.C
        for gone in (self.A, self.B):
            assert gone not in str(state.messages)

    def test_per_call_attribution_survives_rollback(self, monkeypatch):
        """Same guarantee when the probe attributes via
        ``assistant_message()`` rather than leaving it to the fallback."""
        _Harness(monkeypatch, ["bad", "good"], [False, True], traces=[self.A, self.B])
        state = _make_state()
        _run(state, _StubConfig(2), _TracingProbe())

        assistants = self._assistants(state)
        assert [m["content"] for m in assistants] == ["good"]
        assert assistants[0]["reasoning_content"] == self.B
        assert self.A not in str(state.messages)

    def test_fallback_does_not_overwrite_an_attributed_trace(self, monkeypatch):
        """The fallback's "already attributed" guard.

        Only bites on a multi-message turn. ``assistant_resp`` is the
        loop's own call -- the FIRST of the turn -- so without the guard
        the fallback files that first trace against the last assistant
        message, which is the misattribution ``assistant_message()``
        exists to prevent.
        """
        _Harness(monkeypatch, ["tool call"], [True], traces=[self.B])
        state = _make_state()
        _run(state, _StubConfig(1), _MultiCallProbe())

        first, final = self._assistants(state)
        assert first["reasoning_content"] == self.B
        assert final["reasoning_content"] == _MultiCallProbe.SECOND_TRACE

    def test_store_reasoning_off_drops_the_trace(self, monkeypatch):
        """The loop fallback honours ``cfg.store_reasoning`` too, not just
        ``assistant_message`` -- otherwise probes that hand-build their
        message would ignore the flag."""
        _Harness(monkeypatch, ["good"], [True], traces=[self.B])
        state = _make_state()
        _run(state, _StubConfig(1, store_reasoning=False), _StubProbe())

        assistants = self._assistants(state)
        assert [m["content"] for m in assistants] == ["good"]
        assert all(self._no_trace(m) for m in assistants)

    def test_store_reasoning_off_on_the_per_call_path(self, monkeypatch):
        _Harness(monkeypatch, ["good"], [True], traces=[self.B])
        state = _make_state()
        _run(state, _StubConfig(1, store_reasoning=False), _TracingProbe())

        assert all(self._no_trace(m) for m in self._assistants(state))

    def test_traceless_model_is_unaffected_by_rollback(self, monkeypatch):
        """No trace emitted => nothing readable, before or after a
        resample."""
        _Harness(monkeypatch, ["bad", "good"], [False, True])
        state = _make_state()
        _run(state, _StubConfig(2), _StubProbe())

        assistants = self._assistants(state)
        assert [m["content"] for m in assistants] == ["good"]
        assert all(self._no_trace(m) for m in assistants)


class TestOutcomeSerialization:
    def test_counter_serialized_in_outcome_dict(self):
        b = OutcomeBuilder()
        b.inc_assistant_retries()
        b.inc_assistant_retries()
        out = b.finalize(OutcomeStatus.OK)
        assert out.n_assistant_retries == 2
        assert out.to_dict()["n_assistant_retries"] == 2

    def test_counter_defaults_to_zero(self):
        out = OutcomeBuilder().finalize(OutcomeStatus.OK)
        assert out.to_dict()["n_assistant_retries"] == 0


class TestJudgeParseErrors:
    """A parser-default 'failure' must not be charged as a rejection."""

    @staticmethod
    def _harness(monkeypatch, responses, judge_script):
        """judge_script: list of (ok, parse_ok) tuples, consumed per call."""
        calls = {"assistant": 0, "judge": 0}

        def fake_call_llm(models, alias, messages, **kwargs):
            content = responses[min(calls["assistant"], len(responses) - 1)]
            calls["assistant"] += 1
            return {"content": content}

        def fake_judge_ex(models, alias, prompt):
            ok, parse_ok = judge_script[min(calls["judge"], len(judge_script) - 1)]
            calls["judge"] += 1
            rating = "success" if ok else "failure"
            return f"verdict {calls['judge']}", rating, ok, parse_ok

        monkeypatch.setattr(sim, "call_llm", fake_call_llm)
        monkeypatch.setattr(sim, "run_inline_judge_ex", fake_judge_ex)
        return calls

    def test_parse_error_rejudges_same_candidate_without_resample(self, monkeypatch):
        # Judge flakes (unparseable), then passes the SAME candidate on
        # the re-ask: one assistant call, two judge calls, no retry burned.
        calls = self._harness(monkeypatch, ["only answer"], [(False, False), (True, True)])
        state = _make_state()
        content, _, rating, ok = _run(state, _StubConfig(3), _StubProbe())
        assert calls["assistant"] == 1 and calls["judge"] == 2
        assert ok and content == "only answer"
        out = state.outcome.finalize(OutcomeStatus.OK)
        assert out.n_assistant_retries == 0
        ratings = state.metadata["assistant_judge_ratings"]
        assert len(ratings) == 1 and "judge_parse_error" not in ratings[0]

    def test_persistent_parse_error_is_marked_and_counts_as_failure(self, monkeypatch):
        # Judge never produces a parseable verdict: after one re-ask the
        # attempt proceeds as a failure (documented budget charge), and
        # the audit entry is distinguishable from a genuine rejection.
        calls = self._harness(
            monkeypatch,
            ["candidate a", "candidate b"],
            [(False, False)] * 4,
        )
        state = _make_state()
        content, _, rating, ok = _run(state, _StubConfig(2), _StubProbe())
        assert calls["assistant"] == 2  # budget consumed as documented
        assert not ok
        ratings = state.metadata["assistant_judge_ratings"]
        assert all(r.get("judge_parse_error") for r in ratings)
