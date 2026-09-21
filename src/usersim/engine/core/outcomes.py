# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Structured outcome records emitted by the simulator.

The simulator's contract with every downstream consumer (reporting,
evaluator, training-data extraction) rests on two artifacts:

- ``SimulationOutcome`` — one record per trajectory. The row-level
  aggregate: status, failure attribution, counters, per-model tokens
  and calls, wall-clock, provenance, warnings.
- ``SimulationTrace`` — zero-or-more records per trajectory, keyed
  (``trajectory_id`` x ``turn_idx`` x ``call_idx``), discriminated by
  ``TraceKind``. Per-turn / per-call telemetry lives here: in-sim judge
  traces, tool-call timings, API-response reroll details, etc. This is
  the debugging surface, not the assistant-evaluation surface.

The principle: the simulator **emits** structured outcomes; the evaluator
**interprets** them. Anything that requires a model call to determine
its value is the simulator's responsibility (it cannot be reconstructed
later). Anything that is a derivation, aggregation, or judgment over
what's already captured belongs to the evaluator.

In-sim judge raw outputs (user-query gate ratings, assistant-quality
judge ratings, early-stop completion verdicts) are written to the
``simulation_traces`` sidecar with the appropriate ``TraceKind`` — never
to ``simulation_outcome`` and never to the evaluator's ``eval_store``.

The load-bearing decisions around this
separation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OutcomeStatus(str, Enum):
    """Row-level outcome status."""

    OK = "ok"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"


class FailureClass(str, Enum):
    """Structured classification of how a trajectory failed (if it did)."""

    USER_QUERY_GATE_EXHAUSTED = "user_query_gate_exhausted"
    USER_FOLLOWUP_GATE_EXHAUSTED = "user_followup_gate_exhausted"
    USER_LANGUAGE_GATE_EXHAUSTED = "user_language_gate_exhausted"
    API_RESPONSE_INVALID = "api_response_invalid"
    TOOL_CALL_SCHEMA_ERROR = "tool_call_schema_error"
    JUDGE_PARSE_ERROR = "judge_parse_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    SCENARIO_ABORTED = "scenario_aborted"
    TURN_LIMIT_REACHED = "turn_limit_reached"


class FailureAttribution(str, Enum):
    """Which actor caused the failure. Orthogonal to FailureClass."""

    USER_MODEL = "user_model"
    ASSISTANT_MODEL = "assistant_model"
    API_RESPONSE_MODEL = "api_response_model"
    JUDGE_MODEL = "judge_model"
    SCENARIO_LOGIC = "scenario_logic"
    INFRASTRUCTURE = "infrastructure"


class TraceKind(str, Enum):
    """Discriminator for ``SimulationTrace`` entries.

    These are *simulation-internal* events — they drive behavioral
    primitives (frustration escalation, gate retries, flow control) or
    capture per-call telemetry. They are NEVER read by the evaluator as
    assistant-quality signal.
    """

    USER_QUERY_GATE = "user_query_gate"
    USER_FOLLOWUP_GATE = "user_followup_gate"
    FOURTH_WALL_PREFILTER = "fourth_wall_prefilter"
    # User turn regenerated because its script did not match the
    # locale's expected script (e.g. romanized Hindi in an hi_Deva_IN
    # run). Deterministic, no model call. See the script-compliance
    # gate in ``core/simulation.py``.
    LANGUAGE_PREFILTER = "language_prefilter"
    ASSISTANT_QUALITY = "assistant_quality"
    EARLY_STOP = "early_stop"
    API_RESPONSE_REROLL = "api_response_reroll"
    TOOL_CALL_VERIFIER = "tool_call_verifier"
    USER_ROLE_VIOLATION = "user_role_violation"
    SUMMARIZATION = "summarization"
    # Per-assistant-turn capitulation classifier check (yes/no
    # decision made by the probe's ``is_capitulation_detected`` hook).
    # Emitted regardless of result so the trace log shows the
    # classifier's signal at every turn for audit purposes.
    CAPITULATION_CHECK = "capitulation_check"
    # A retrieval query was machine-translated into the borrowed corpus's
    # language before searching (stop-gap locales, where the conversation
    # language and the asset language differ). Carries both strings so a
    # retrieval miss can be attributed to the translator rather than the
    # assistant — the query the assistant wrote is not the query that ranked.
    KB_QUERY_TRANSLATION = "kb_query_translation"


class WarningKind(str, Enum):
    """Structured warning kinds for ``SimulationOutcome.warnings``."""

    ID_FABRICATION = "id_fabrication"
    COMPRESSION_FALLBACK = "compression_fallback"
    POOR_ASSISTANT_RESPONSE = "poor_assistant_response"
    # Flagged when a trajectory probed a fact that was marked
    # ``placeholder: true`` in its bank. Downstream scorecards refuse to
    # render a "ready for sovereign-AI claim" headline whenever any
    # trajectory carries this warning.
    USED_PLACEHOLDER_FACT = "used_placeholder_fact"
    # Flagged when a trajectory probed against a sov_ai_dynamic
    # taxonomy that was marked ``placeholder: true`` at the file level.
    # Same epistemic discipline as ``USED_PLACEHOLDER_FACT`` but for the
    # `sov_ai_dynamic` probe, where the gate is at taxonomy
    # granularity (the invitation + sub-topic hints share one reviewer
    # audit, not per-hint).
    USED_PLACEHOLDER_TAXONOMY = "used_placeholder_taxonomy"
    # Flagged when a trajectory probed against a query that was marked
    # ``placeholder: true`` in its query bank. Same epistemic discipline
    # as ``USED_PLACEHOLDER_FACT`` and ``USED_PLACEHOLDER_TAXONOMY``, but
    # for the cross-locale ``sov_ai_multilingual_parity`` probe. The
    # gate is at single-query granularity (each entry holds N locale
    # renderings, all sharing one reviewer audit).
    USED_PLACEHOLDER_QUERY = "used_placeholder_query"
    # Flagged when a safety_chat_pressure trajectory probed against a
    # target_request marked ``placeholder: true`` in its
    # pressure-strategy bank. Downstream readers (capability dashboard,
    # scorers) treat placeholder-tainted safety trajectories as
    # preview-only — same epistemic discipline as the sovereign-AI
    # placeholder warnings.
    USED_PLACEHOLDER_TARGET = "used_placeholder_target"
    # Flagged when a safety_agentic trajectory probed against an
    # action_request marked ``placeholder: true`` in its agentic-action
    # bank. Downstream readers treat placeholder-tainted agentic-safety
    # trajectories as preview-only.
    USED_PLACEHOLDER_AGENTIC_ACTION = "used_placeholder_agentic_action"
    # Flagged when a financial_services trajectory ran against a task
    # template or document corpus marked ``placeholder: true`` (AI-authored
    # beta content). Downstream scorecards treat placeholder-tainted
    # financial-services trajectories as preview-only — same epistemic
    # discipline as the sovereign-AI / safety placeholder warnings.
    USED_PLACEHOLDER_FINANCE = "used_placeholder_finance"
    # Flagged when a health_disclosure trajectory used a clinical profile
    # marked ``placeholder: true`` in its clinical-profile bank (the bundled
    # synthetic banks are all placeholders until a customer ships real ground
    # truth). Same epistemic discipline: scorecards stay preview-only for any
    # trajectory whose concealment ground truth was synthetic.
    USED_PLACEHOLDER_CLINICAL_PROFILE = "used_placeholder_clinical_profile"
    # Flagged when a trajectory's verbatim turn-1 was machine-translated
    # from an India base-locale (en_IN) bank into the conversation
    # language because no native per-language bank exists yet (India
    # language-variant stop-gap). Downstream scorecards treat these cells
    # as preview-only, same discipline as the placeholder warnings above.
    # Automatically stops appearing once a native asset lands (the probe
    # resolves the variant's own asset -> no translation).
    USED_MACHINE_TRANSLATION = "used_machine_translation"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Provenance:
    """Everything needed to audit what a trajectory was built against.

    A panel / cohort / canary parquet built against v1 that is replayed
    against a v2 persona dataset must be a schema-mismatch error at
    load time, not a silent drift — this record is what makes that
    possible.
    """

    nemotron_personas_version: str | None = None
    scenario_prompt_version: str | None = None
    code_sha: str | None = None
    bank_version: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nemotron_personas_version": self.nemotron_personas_version,
            "scenario_prompt_version": self.scenario_prompt_version,
            "code_sha": self.code_sha,
            "bank_version": dict(self.bank_version),
        }


@dataclass
class SimulationWarning:
    """Structured warning entry. Stored in ``SimulationOutcome.warnings``."""

    kind: WarningKind
    turn_idx: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "turn_idx": self.turn_idx,
            "detail": self.detail,
        }


@dataclass
class SimulationOutcome:
    """The canonical per-trajectory row-level outcome record.

    Everything a downstream consumer needs to answer "what happened with
    this trajectory at an aggregate level?" — without re-parsing
    conversation messages or invoking any model.
    """

    status: OutcomeStatus = OutcomeStatus.OK
    failure_class: FailureClass | None = None
    failure_attribution: FailureAttribution | None = None
    failure_detail: str = ""

    n_turns: int = 0
    n_tool_calls: int = 0
    n_user_query_attempts: int = 0
    n_user_followup_retries: int = 0
    n_assistant_inline_failures: int = 0
    # Judge-rejected assistant candidates that were discarded and
    # regenerated (``max_assistant_attempts > 1``). Distinct from
    # ``n_assistant_inline_failures``, which counts turns whose FINAL
    # candidate is flagged in the delivered transcript.
    n_assistant_retries: int = 0
    n_api_response_rerolls: int = 0
    n_fourth_wall_triggers: int = 0
    n_user_role_violations: int = 0
    n_user_language_violations: int = 0

    warnings: list[SimulationWarning] = field(default_factory=list)
    early_stop: bool = False

    per_model_input_tokens: dict[str, int] = field(default_factory=dict)
    per_model_output_tokens: dict[str, int] = field(default_factory=dict)
    per_model_calls: dict[str, int] = field(default_factory=dict)
    wall_clock_s_by_alias: dict[str, float] = field(default_factory=dict)
    wall_clock_s: float = 0.0

    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "failure_class": self.failure_class.value if self.failure_class else None,
            "failure_attribution": (self.failure_attribution.value if self.failure_attribution else None),
            "failure_detail": self.failure_detail,
            "n_turns": self.n_turns,
            "n_tool_calls": self.n_tool_calls,
            "n_user_query_attempts": self.n_user_query_attempts,
            "n_user_followup_retries": self.n_user_followup_retries,
            "n_assistant_inline_failures": self.n_assistant_inline_failures,
            "n_assistant_retries": self.n_assistant_retries,
            "n_api_response_rerolls": self.n_api_response_rerolls,
            "n_fourth_wall_triggers": self.n_fourth_wall_triggers,
            "n_user_role_violations": self.n_user_role_violations,
            "n_user_language_violations": self.n_user_language_violations,
            "warnings": [w.to_dict() for w in self.warnings],
            "early_stop": self.early_stop,
            "per_model_input_tokens": dict(self.per_model_input_tokens),
            "per_model_output_tokens": dict(self.per_model_output_tokens),
            "per_model_calls": dict(self.per_model_calls),
            "wall_clock_s_by_alias": dict(self.wall_clock_s_by_alias),
            "wall_clock_s": round(self.wall_clock_s, 3),
            "provenance": self.provenance.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


@dataclass
class SimulationTrace:
    """A single per-turn / per-call telemetry entry.

    Stored in the sidecar table keyed (``trajectory_id``, ``turn_idx``,
    ``call_idx``). ``kind`` discriminates which event it captures.
    """

    kind: TraceKind
    turn_idx: int
    call_idx: int = 0
    model_alias: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    prompt_version: str | None = None
    rating: str | None = None  # for judge kinds: "success" / "warn" / "failure"
    detail: str | None = None  # free-text: short explanation, error, etc.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "turn_idx": self.turn_idx,
            "call_idx": self.call_idx,
            "model_alias": self.model_alias,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 3),
            "prompt_version": self.prompt_version,
            "rating": self.rating,
            "detail": self.detail,
            "extra": dict(self.extra),
        }


# ---------------------------------------------------------------------------
# Builder — mutated during simulation; frozen into SimulationOutcome at end
# ---------------------------------------------------------------------------


class OutcomeBuilder:
    """Mutable accumulator for the sim's structured outcome record.

    Owned by ``ConversationState``. Mutated by the simulator as it
    progresses. ``finalize(status, ...)`` returns an immutable
    ``SimulationOutcome``; ``traces()`` returns the list of
    ``SimulationTrace`` entries for the sidecar.
    """

    def __init__(self, provenance: Provenance | None = None) -> None:
        self._outcome = SimulationOutcome(provenance=provenance or Provenance())
        self._traces: list[SimulationTrace] = []

    # ── counters ───────────────────────────────────────────────────

    def inc_user_query_attempts(self) -> None:
        self._outcome.n_user_query_attempts += 1

    def inc_user_followup_retries(self, n: int = 1) -> None:
        self._outcome.n_user_followup_retries += n

    def inc_assistant_inline_failures(self) -> None:
        self._outcome.n_assistant_inline_failures += 1

    def inc_assistant_retries(self) -> None:
        self._outcome.n_assistant_retries += 1

    def inc_api_response_rerolls(self) -> None:
        self._outcome.n_api_response_rerolls += 1

    def inc_fourth_wall_triggers(self) -> None:
        self._outcome.n_fourth_wall_triggers += 1

    def inc_user_role_violations(self) -> None:
        self._outcome.n_user_role_violations += 1

    def inc_user_language_violations(self) -> None:
        self._outcome.n_user_language_violations += 1

    # ── warnings ──────────────────────────────────────────────────

    def add_warning(
        self,
        kind: WarningKind,
        turn_idx: int | None = None,
        detail: str = "",
    ) -> None:
        self._outcome.warnings.append(SimulationWarning(kind=kind, turn_idx=turn_idx, detail=detail))

    # ── traces ────────────────────────────────────────────────────

    def add_trace(self, trace: SimulationTrace) -> None:
        self._traces.append(trace)

    def traces(self) -> list[SimulationTrace]:
        return list(self._traces)

    # ── per-model aggregation (called from within the LLM layer) ──

    def record_call(
        self,
        alias: str,
        input_tokens: int,
        output_tokens: int,
        elapsed_s: float,
    ) -> None:
        self._outcome.per_model_calls[alias] = self._outcome.per_model_calls.get(alias, 0) + 1
        self._outcome.per_model_input_tokens[alias] = self._outcome.per_model_input_tokens.get(alias, 0) + input_tokens
        self._outcome.per_model_output_tokens[alias] = (
            self._outcome.per_model_output_tokens.get(alias, 0) + output_tokens
        )
        self._outcome.wall_clock_s_by_alias[alias] = self._outcome.wall_clock_s_by_alias.get(alias, 0.0) + elapsed_s

    # ── finalization ──────────────────────────────────────────────

    def set_early_stop(self, value: bool = True) -> None:
        self._outcome.early_stop = value

    def set_n_turns(self, n: int) -> None:
        self._outcome.n_turns = n

    def set_n_tool_calls(self, n: int) -> None:
        self._outcome.n_tool_calls = n

    def set_wall_clock_s(self, s: float) -> None:
        self._outcome.wall_clock_s = s

    def finalize(
        self,
        status: OutcomeStatus,
        failure_class: FailureClass | None = None,
        failure_attribution: FailureAttribution | None = None,
        failure_detail: str = "",
    ) -> SimulationOutcome:
        self._outcome.status = status
        self._outcome.failure_class = failure_class
        self._outcome.failure_attribution = failure_attribution
        self._outcome.failure_detail = failure_detail
        # Promote to COMPLETED_WITH_WARNINGS if we finished OK but
        # accumulated warnings (identifier fabrication, etc).
        if status == OutcomeStatus.OK and (
            self._outcome.warnings
            or self._outcome.n_user_role_violations > 0
            or self._outcome.n_user_language_violations > 0
        ):
            self._outcome.status = OutcomeStatus.COMPLETED_WITH_WARNINGS
        return self._outcome


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def serialize_traces(traces: list[SimulationTrace]) -> str:
    """JSON-encode a list of traces for storage as a string column."""
    return json.dumps([t.to_dict() for t in traces], ensure_ascii=False, default=str)
