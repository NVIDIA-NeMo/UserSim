# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for core/outcomes.py — structured simulation outcome records.

Schema tests:
- Roundtrip: a built outcome serializes to a dict that contains all
  expected fields with the right types.
- Enum exhaustiveness: every failure class and failure attribution
  round-trips through the to_dict/JSON path.
- Status promotion: finalizing an OK outcome with accumulated warnings
  or user-role violations bumps status to COMPLETED_WITH_WARNINGS.
- Counters: incrementer methods add up correctly.
- Builder -> SimulationOutcome handoff: finalize returns an immutable
  dataclass with the right failure fields.
- make_failed / make_result emit ``simulation_outcome`` +
  ``simulation_traces`` alongside the per-row side-effect columns.
"""

from __future__ import annotations

import json

import pytest

from usersim.engine.core.outcomes import (
    FailureAttribution,
    FailureClass,
    OutcomeBuilder,
    OutcomeStatus,
    Provenance,
    SimulationOutcome,
    SimulationTrace,
    TraceKind,
    WarningKind,
    serialize_traces,
)
from usersim.engine.core.simulation import make_failed, make_result


class TestSimulationOutcomeSchema:
    def test_default_is_ok_with_empty_counters(self) -> None:
        outcome = SimulationOutcome()
        assert outcome.status == OutcomeStatus.OK
        assert outcome.failure_class is None
        assert outcome.failure_attribution is None
        assert outcome.n_turns == 0
        assert outcome.n_tool_calls == 0
        assert outcome.n_user_query_attempts == 0
        assert outcome.n_user_followup_retries == 0
        assert outcome.n_assistant_inline_failures == 0
        assert outcome.n_api_response_rerolls == 0
        assert outcome.n_fourth_wall_triggers == 0
        assert outcome.n_user_role_violations == 0
        assert outcome.warnings == []
        assert outcome.early_stop is False

    def test_to_dict_contains_all_documented_fields(self) -> None:
        outcome = SimulationOutcome()
        d = outcome.to_dict()
        expected = {
            "status",
            "failure_class",
            "failure_attribution",
            "failure_detail",
            "n_turns",
            "n_tool_calls",
            "n_user_query_attempts",
            "n_user_followup_retries",
            "n_assistant_inline_failures",
            "n_assistant_retries",
            "n_api_response_rerolls",
            "n_fourth_wall_triggers",
            "n_user_role_violations",
            "n_user_language_violations",
            "warnings",
            "early_stop",
            "per_model_input_tokens",
            "per_model_output_tokens",
            "per_model_calls",
            "wall_clock_s_by_alias",
            "wall_clock_s",
            "provenance",
        }
        assert set(d.keys()) == expected

    def test_to_json_roundtrips(self) -> None:
        outcome = SimulationOutcome(
            status=OutcomeStatus.FAILED,
            failure_class=FailureClass.USER_QUERY_GATE_EXHAUSTED,
            failure_attribution=FailureAttribution.USER_MODEL,
            failure_detail="exhausted after 3 attempts",
            n_turns=2,
            n_user_query_attempts=3,
            per_model_calls={"user_model": 3},
            per_model_input_tokens={"user_model": 150},
            per_model_output_tokens={"user_model": 80},
            wall_clock_s=12.34,
            wall_clock_s_by_alias={"user_model": 9.1},
            provenance=Provenance(
                nemotron_personas_version="v1.0",
                scenario_prompt_version="v1.0",
                code_sha="abc123",
            ),
        )
        parsed = json.loads(outcome.to_json())
        assert parsed["status"] == "failed"
        assert parsed["failure_class"] == "user_query_gate_exhausted"
        assert parsed["failure_attribution"] == "user_model"
        assert parsed["failure_detail"] == "exhausted after 3 attempts"
        assert parsed["n_turns"] == 2
        assert parsed["n_user_query_attempts"] == 3
        assert parsed["per_model_calls"] == {"user_model": 3}
        assert parsed["wall_clock_s"] == 12.34
        assert parsed["provenance"]["code_sha"] == "abc123"
        assert parsed["provenance"]["nemotron_personas_version"] == "v1.0"


class TestEnumExhaustiveness:
    """Every failure_class and failure_attribution must survive to_dict()."""

    @pytest.mark.parametrize("fc", list(FailureClass))
    def test_every_failure_class_serializes(self, fc: FailureClass) -> None:
        outcome = SimulationOutcome(
            status=OutcomeStatus.FAILED,
            failure_class=fc,
            failure_attribution=FailureAttribution.SCENARIO_LOGIC,
        )
        d = outcome.to_dict()
        assert d["failure_class"] == fc.value

    @pytest.mark.parametrize("fa", list(FailureAttribution))
    def test_every_failure_attribution_serializes(self, fa: FailureAttribution) -> None:
        outcome = SimulationOutcome(
            status=OutcomeStatus.FAILED,
            failure_class=FailureClass.INFRASTRUCTURE_ERROR,
            failure_attribution=fa,
        )
        d = outcome.to_dict()
        assert d["failure_attribution"] == fa.value

    @pytest.mark.parametrize("tk", list(TraceKind))
    def test_every_trace_kind_serializes(self, tk: TraceKind) -> None:
        tr = SimulationTrace(kind=tk, turn_idx=0)
        assert tr.to_dict()["kind"] == tk.value


class TestBuilderCounters:
    def test_increment_counters(self) -> None:
        b = OutcomeBuilder()
        b.inc_user_query_attempts()
        b.inc_user_query_attempts()
        b.inc_user_followup_retries(3)
        b.inc_assistant_inline_failures()
        b.inc_api_response_rerolls()
        b.inc_fourth_wall_triggers()
        b.inc_user_role_violations()
        out = b.finalize(OutcomeStatus.OK)
        assert out.n_user_query_attempts == 2
        assert out.n_user_followup_retries == 3
        assert out.n_assistant_inline_failures == 1
        assert out.n_api_response_rerolls == 1
        assert out.n_fourth_wall_triggers == 1

    def test_record_call_aggregates_per_alias(self) -> None:
        b = OutcomeBuilder()
        b.record_call("user_model", input_tokens=100, output_tokens=50, elapsed_s=1.5)
        b.record_call("user_model", input_tokens=120, output_tokens=60, elapsed_s=2.0)
        b.record_call("judge_model", input_tokens=200, output_tokens=30, elapsed_s=0.8)
        out = b.finalize(OutcomeStatus.OK)
        assert out.per_model_calls == {"user_model": 2, "judge_model": 1}
        assert out.per_model_input_tokens == {"user_model": 220, "judge_model": 200}
        assert out.per_model_output_tokens == {"user_model": 110, "judge_model": 30}
        assert out.wall_clock_s_by_alias["user_model"] == pytest.approx(3.5)
        assert out.wall_clock_s_by_alias["judge_model"] == pytest.approx(0.8)

    def test_record_call_does_not_carry_reasoning_tokens(self) -> None:
        """Regression lock: SimulationOutcome no longer carries
        per-alias reasoning telemetry. Reasoning is estimated post-hoc
        at report build via tiktoken on visible content (DD strips
        provider-captured reasoning at the Usage abstraction layer, so
        sim-time capture would always be empty anyway)."""
        b = OutcomeBuilder()
        b.record_call(
            "assistant_model",
            input_tokens=10,
            output_tokens=80,
            elapsed_s=1.0,
        )
        out = b.finalize(OutcomeStatus.OK)
        # The field is gone -- not even an empty dict.
        assert not hasattr(out, "per_model_reasoning_tokens")
        assert "per_model_reasoning_tokens" not in out.to_dict()


class TestStatusPromotion:
    def test_ok_with_no_warnings_stays_ok(self) -> None:
        b = OutcomeBuilder()
        out = b.finalize(OutcomeStatus.OK)
        assert out.status == OutcomeStatus.OK

    def test_ok_with_warning_promotes_to_completed_with_warnings(self) -> None:
        b = OutcomeBuilder()
        b.add_warning(
            kind=WarningKind.ID_FABRICATION,
            turn_idx=1,
            detail="CPF not in persona snapshot",
        )
        out = b.finalize(OutcomeStatus.OK)
        assert out.status == OutcomeStatus.COMPLETED_WITH_WARNINGS
        assert len(out.warnings) == 1
        assert out.warnings[0].kind == WarningKind.ID_FABRICATION
        assert out.warnings[0].turn_idx == 1

    def test_ok_with_user_role_violations_promotes(self) -> None:
        b = OutcomeBuilder()
        b.inc_user_role_violations()
        out = b.finalize(OutcomeStatus.OK)
        assert out.status == OutcomeStatus.COMPLETED_WITH_WARNINGS

    def test_failed_status_is_not_promoted(self) -> None:
        b = OutcomeBuilder()
        b.add_warning(WarningKind.POOR_ASSISTANT_RESPONSE, turn_idx=0, detail="")
        out = b.finalize(
            OutcomeStatus.FAILED,
            failure_class=FailureClass.INFRASTRUCTURE_ERROR,
            failure_attribution=FailureAttribution.INFRASTRUCTURE,
            failure_detail="boom",
        )
        assert out.status == OutcomeStatus.FAILED
        assert out.failure_class == FailureClass.INFRASTRUCTURE_ERROR


class TestTraces:
    def test_add_and_serialize(self) -> None:
        b = OutcomeBuilder()
        b.add_trace(
            SimulationTrace(
                kind=TraceKind.USER_QUERY_GATE,
                turn_idx=0,
                call_idx=0,
                model_alias="judge_model",
                rating="success",
                detail="ok",
            )
        )
        b.add_trace(
            SimulationTrace(
                kind=TraceKind.API_RESPONSE_REROLL,
                turn_idx=1,
                call_idx=2,
                model_alias="api_response_model",
                detail="invalid json",
            )
        )
        traces = b.traces()
        assert len(traces) == 2
        blob = serialize_traces(traces)
        parsed = json.loads(blob)
        assert parsed[0]["kind"] == "user_query_gate"
        assert parsed[0]["rating"] == "success"
        assert parsed[1]["kind"] == "api_response_reroll"
        assert parsed[1]["call_idx"] == 2


class TestMakeFailedWithOutcome:
    """make_failed must synthesize a valid outcome when none is supplied."""

    def test_synthesizes_outcome_on_bare_call(self) -> None:
        result = make_failed("reason here")
        assert result["simulation_outcome"] is not None
        parsed = json.loads(result["simulation_outcome"])
        assert parsed["status"] == "failed"
        assert parsed["failure_class"] == "scenario_aborted"
        assert parsed["failure_attribution"] == "scenario_logic"
        assert parsed["failure_detail"] == "reason here"
        assert result["simulation_traces"] == "[]"

    def test_carries_supplied_outcome(self) -> None:
        b = OutcomeBuilder()
        b.inc_user_query_attempts()
        b.inc_user_query_attempts()
        b.inc_user_query_attempts()
        outcome = b.finalize(
            OutcomeStatus.FAILED,
            failure_class=FailureClass.USER_QUERY_GATE_EXHAUSTED,
            failure_attribution=FailureAttribution.USER_MODEL,
            failure_detail="exhausted",
        )
        result = make_failed("exhausted", outcome=outcome, traces=[])
        parsed = json.loads(result["simulation_outcome"])
        assert parsed["failure_class"] == "user_query_gate_exhausted"
        assert parsed["n_user_query_attempts"] == 3

    def test_side_effect_columns_present(self) -> None:
        result = make_failed("whatever")
        for key in [
            "user_query",
            "conversation_messages",
            "conversation_metadata",
            "conversation_status",
            "num_turns",
            "num_tool_calls",
            "tool_subset",
            "disclosure_style",
            "user_interaction_style",
            "simulation_outcome",
            "simulation_traces",
        ]:
            assert key in result


class TestMakeResultWithOutcome:
    def test_outcome_none_serialized_as_none(self) -> None:
        result = make_result([], {}, True)
        assert result["simulation_outcome"] is None
        assert result["simulation_traces"] is None

    def test_outcome_and_traces_serialize(self) -> None:
        outcome = OutcomeBuilder().finalize(OutcomeStatus.OK)
        traces = [SimulationTrace(kind=TraceKind.EARLY_STOP, turn_idx=1)]
        result = make_result([], {}, True, outcome=outcome, traces=traces)
        parsed_outcome = json.loads(result["simulation_outcome"])
        parsed_traces = json.loads(result["simulation_traces"])
        assert parsed_outcome["status"] == "ok"
        assert parsed_traces[0]["kind"] == "early_stop"
