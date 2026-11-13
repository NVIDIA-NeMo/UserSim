# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
import re
from dataclasses import dataclass, field, fields
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
    # The judge's verdict on one simulated tool response (tool_calling):
    # ``rating`` holds it, ``detail`` the explanation.
    API_RESPONSE_JUDGE = "api_response_judge"
    TOOL_CALL_VERIFIER = "tool_call_verifier"
    USER_ROLE_VIOLATION = "user_role_violation"
    SUMMARIZATION = "summarization"
    # Per-assistant-turn capitulation classifier check (yes/no
    # decision made by the probe's ``is_capitulation_detected`` hook).
    # Emitted regardless of result so the trace log shows the
    # classifier's signal at every turn for audit purposes.
    CAPITULATION_CHECK = "capitulation_check"
    # A retrieval query was machine-translated into the borrowed corpus's
    # language before searching (India variants, where the conversation
    # language and the asset language differ). Carries both strings so a
    # retrieval miss can be attributed to the translator rather than the
    # assistant — the query the assistant wrote is not the query that ranked.
    KB_QUERY_TRANSLATION = "kb_query_translation"


class WarningKind(str, Enum):
    """The warning kinds the engine itself records in ``SimulationOutcome.warnings``.

    A probe records a kind of its own as a string (see ``warning_kind_problem``),
    such as the ``placeholder_warning_kind`` it declares.
    """

    COMPRESSION_FALLBACK = "compression_fallback"
    POOR_ASSISTANT_RESPONSE = "poor_assistant_response"
    # Flagged when a trajectory's verbatim turn-1 was machine-translated
    # from an India base-locale (en_IN) bank into the conversation
    # language, because that language has no native bank of its own;
    # financial_services also flags it when kb_search queries are
    # translated to search the base-locale corpus. Recorded on the outcome
    # so a reader can tell which findings rest on machine-translated
    # content. A locale with its own native asset never translates, so its
    # rows never carry this warning.
    USED_MACHINE_TRANSLATION = "used_machine_translation"


#: The start of a built-in probe's placeholder warning kind, such as
#: ``used_placeholder_fact``; a probe outside this package writes it after its
#: prefix, as in ``my_used_placeholder_booking``.
PLACEHOLDER_KIND_MARKER = "used_placeholder_"

#: A warning kind a probe outside this package defines: lowercase words joined
#: by underscores, led by the package's own prefix, such as ``my_slow_tool``.
_EXTENSION_WARNING_KIND = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")


def warning_kind_problem(kind: object) -> str | None:
    """Why ``kind`` cannot be a warning kind, or ``None`` when it can.

    A ``WarningKind`` member, or a lowercase name of words joined by
    underscores: a built-in probe's kind such as ``used_placeholder_fact``, or
    a prefixed one such as ``my_slow_tool``.
    """
    if isinstance(kind, WarningKind):
        return None
    if isinstance(kind, str) and _EXTENSION_WARNING_KIND.fullmatch(kind):
        return None
    return f"must be a WarningKind or a lowercase name such as 'my_slow_tool', got {kind!r}"


def is_placeholder_warning_kind(kind: object) -> bool:
    """Whether a warning kind marks placeholder content.

    A built-in probe's ``used_placeholder_<thing>`` kind, or a kind of the
    form ``<prefix>_used_placeholder_<thing>`` from a probe outside this
    package. Stored rows hold the kind as a string; either form is read.
    """
    value = kind.value if isinstance(kind, WarningKind) else kind
    return isinstance(value, str) and (
        value.startswith(PLACEHOLDER_KIND_MARKER) or f"_{PLACEHOLDER_KIND_MARKER}" in value
    )


def placeholder_warning_kind_problem(kind: object) -> str | None:
    """Why ``kind`` cannot be a probe's ``placeholder_warning_kind``, or ``None`` when it can.

    It must be a valid warning kind that marks placeholder content, so the
    report counts the rows that carry it as resting on placeholder content.
    """
    problem = warning_kind_problem(kind)
    if problem:
        return problem
    if not is_placeholder_warning_kind(kind):
        return (
            "must mark placeholder content: a name of the form 'used_placeholder_<thing>', or "
            f"'<prefix>_used_placeholder_<thing>' for a probe outside this package, got {kind!r}"
        )
    return None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Provenance:
    """Everything needed to audit what a trajectory was built against.

    Recorded on every row, so rows built against different persona
    dataset, prompt, code or bank versions can be told apart.
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
    """Structured warning entry. Stored in ``SimulationOutcome.warnings``.

    A string equal to a built-in kind's value becomes that ``WarningKind``;
    any other string must pass ``warning_kind_problem``.
    """

    kind: WarningKind | str
    turn_idx: int | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.kind, WarningKind):
            return
        try:
            self.kind = WarningKind(self.kind)
            return
        except ValueError:
            pass
        problem = warning_kind_problem(self.kind)
        if problem:
            raise ValueError(f"warning kind {problem}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value if isinstance(self.kind, WarningKind) else self.kind,
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
    # Simulated tool responses the judge rejected; each was regenerated
    # while the probe's attempt budget allowed.
    n_api_response_rejections: int = 0
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
            "n_api_response_rejections": self.n_api_response_rejections,
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


@dataclass(frozen=True)
class OutcomeCheckpoint:
    """An ``OutcomeBuilder``'s counters, warnings and traces at one moment."""

    counters: dict[str, int]
    n_warnings: int
    n_traces: int


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

    def inc_api_response_rejections(self) -> None:
        self._outcome.n_api_response_rejections += 1

    def inc_fourth_wall_triggers(self) -> None:
        self._outcome.n_fourth_wall_triggers += 1

    def inc_user_role_violations(self) -> None:
        self._outcome.n_user_role_violations += 1

    def inc_user_language_violations(self) -> None:
        self._outcome.n_user_language_violations += 1

    # ── warnings ──────────────────────────────────────────────────

    def add_warning(
        self,
        kind: WarningKind | str,
        turn_idx: int | None = None,
        detail: str = "",
    ) -> None:
        self._outcome.warnings.append(SimulationWarning(kind=kind, turn_idx=turn_idx, detail=detail))

    # ── traces ────────────────────────────────────────────────────

    def add_trace(self, trace: SimulationTrace) -> None:
        self._traces.append(trace)

    def traces(self) -> list[SimulationTrace]:
        return list(self._traces)

    # ── rollback ──────────────────────────────────────────────────

    def checkpoint(self) -> OutcomeCheckpoint:
        """The counters, warnings and traces as they stand, for :meth:`rollback`."""
        return OutcomeCheckpoint(
            counters={
                f.name: getattr(self._outcome, f.name) for f in fields(SimulationOutcome) if f.name.startswith("n_")
            },
            n_warnings=len(self._outcome.warnings),
            n_traces=len(self._traces),
        )

    def rollback(self, checkpoint: OutcomeCheckpoint) -> None:
        """Undo every counter, warning and trace recorded since ``checkpoint``.

        Model usage (calls, tokens, wall clock) stays: those calls were made.
        """
        for name, value in checkpoint.counters.items():
            setattr(self._outcome, name, value)
        del self._outcome.warnings[checkpoint.n_warnings :]
        del self._traces[checkpoint.n_traces :]

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
