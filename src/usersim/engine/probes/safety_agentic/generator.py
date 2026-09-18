# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_agentic probe.

Multi-turn loop with a **mock tool-call interceptor**:

1. Resolve the action_request from the input row's
   ``action_request_id`` override, or derive from the persona
   (smoke-test path).
2. Inject ``action_request.initial_user_prompt`` as turn 1 verbatim.
3. Loop turn 1..max_turns:
   - Call the assistant with the running message history + the
     action request's ``simulated_tools`` formatted as OpenAI-style
     tool specs.
   - If the assistant emits ``tool_calls``: for each call, **record
     it as an attempted action** (the load-bearing side channel for
     the safety_agentic scorer), and append the matching
     ``mock_tool_response`` from the bank as the tool's reply (no
     execution, no separate LLM call to synthesize a response —
     every response is pre-canned).
   - If the assistant emits no tool calls AND non-empty content:
     the trajectory is allowed to continue (the assistant might be
     asking for confirmation or refusing). On the next turn the
     simulator would normally let the user-agent reply, but this
     probe is **single-user-turn** by design (the user states their
     request once; what follows is the assistant's tool-using
     response). Stop the loop.
4. Finalise.

**Single-user-turn design rationale.** The ``safety_chat_pressure`` probe
runs multi-user-turn loops because the user-agent's pressure tactics
are the load. ``safety_agentic`` runs single-user-turn loops because
the load is the *assistant's* tool-using response — the user stated
their request once, and what we're testing is what the assistant
chose to do with the available tools across the turns the assistant
takes internally. Multi-user-turn variants (e.g., user confirms after
the assistant discloses) are out of scope for this probe; they would
be a separate sibling probe.

**Why this probe overrides ``run_dispatch`` instead of going through
``ConversationLoop.run``.** The unified loop's machinery — fourth-wall
prefilter, user-judge gate retries, role-violation telemetry,
follow-up generation, frustration escalation — assumes a
back-and-forth conversation with a user-LLM driving the user side.
This probe has NO user-LLM at all (single user turn from the bank,
zero follow-ups), so threading the agentic shape through the
unified loop would either skip every step (every hook returning
False) or call user-LLM machinery uselessly. The substrate
(BaseProbe + BankBackedProbe + AgenticMixin) still provides the
load-bearing pieces — bank load + cache + placeholder warning +
``provenance.bank_version["agentic"]`` pinning + metadata seeding
+ ``should_succeed`` / ``build_result_extras`` hooks — and this
probe owns the simulate loop directly.

Built on ``BaseProbe`` + ``BankBackedProbe`` + ``AgenticMixin``.

Side channels emitted on the trajectory:

- ``state.metadata["action_request_id"]`` — primary join key.
- ``state.metadata["sub_protocol"]`` — gating dimension for the bundle.
- ``state.metadata["sanctioned_action_name"]`` /
  ``tempted_action_name`` — pinned for cross-trajectory comparison.
- ``state.metadata["attempted_actions"]`` — list of
  ``{tool_name, tool_args, turn_idx, was_in_simulated_tools}``
  records, one per assistant tool call. The safety_agentic scorer
  reads this to compute confirmation_rate / boundary_violation_rate
  / consequence_disclosure_rate / etc.
- ``state.metadata["bank_id"]`` / ``bank_version`` for drift detection.
- ``result["probe_variant"]`` —
  ``f"{sub_protocol}::{sanctioned_action_name or 'no_action'}"``.
- ``SimulationOutcome.provenance.bank_version["agentic"]`` — bank
  version under stable non-locale key (this probe is not
  locale-stratified; same convention `safety_chat_pressure` uses
  with the ``"safety"`` key).
- ``SimulationOutcome.warnings`` —
  ``WarningKind.USED_PLACEHOLDER_AGENTIC_ACTION`` appended when the
  picked action_request had ``placeholder: true`` (substrate-driven).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from usersim.engine.core.agentic_bank import (
    ActionRequest,
    ToolSpec,
    load_agentic_bank_default,
    reset_agentic_bank_cache,
)
from usersim.engine.core.llm import call_llm, set_current_outcome_builder
from usersim.engine.core.tool_calls import parse_tool_call
from usersim.engine.core.outcomes import (
    FailureAttribution,
    FailureClass,
    OutcomeBuilder,
    OutcomeStatus,
    SimulationTrace,
    TraceKind,
    WarningKind,
)
from usersim.engine.core.probes import (
    AgenticMixin,
    BankBackedProbe,
    assistant_message,
    register_probe,
)
from usersim.engine.core.simulation import (
    ConversationState,
    make_failed,
    make_result,
)
from usersim.engine.probes.safety_agentic.task_derivation import (
    resolve_task_from_row,
)

logger = logging.getLogger("usersim.engine")

# ── Probe metadata ────────────────────────────────────────────────────

PROBE_FAMILY: str = "safety_agentic"
PROBE_VARIANTS: list[str] = ["default"]
PROMPT_VERSION: str = "v1.0"

_MODEL_ASSISTANT = "assistant_model"

# Locale-independent — bank version lives under "agentic" so
# downstream readers (capability dashboard, scorers) know where to
# look. Different from the sovereign-AI convention (locale-keyed).
_BANK_VERSION_KEY = "agentic"

# ---------------------------------------------------------------------------
# Bank-loading aliases
# ---------------------------------------------------------------------------

_load_bank = load_agentic_bank_default
_reset_bank_cache = reset_agentic_bank_cache


def _bank_loader():
    """Indirection through the module-level alias for test patchability.

    ``BankBackedProbe._load_bank_for_locale`` calls ``loader(locale)``
    first and falls back to no-arg on TypeError, so this no-arg
    loader is invoked via the fallback path.
    """
    return _load_bank()


def should_succeed(state: ConversationState) -> bool:
    """Sim-side guardrail: did we actually probe an action request?

    Returns ``False`` when the trajectory carries no
    ``action_request_id`` — that would happen only if the probe
    somehow produced a trajectory record without resolving a request.
    Module-level mirror of the probe method so existing test surfaces
    can call this directly without an instance.
    """
    return bool(state.metadata.get("action_request_id"))


class SafetyAgenticProbeError(ValueError):
    """Raised when the probe cannot construct (no resolvable action_request).

    The shim ``simulate_safety_agentic`` catches this and returns a
    structured ``make_failed`` outcome with
    ``failure_class=SCENARIO_ABORTED``.
    """


@dataclass
class _PickedActionRequest:
    """Substrate-shaped wrapper around the resolved ``ActionRequest``."""

    id: str
    placeholder: bool
    action_request: ActionRequest
    bank_id: str
    bank_version: str


# ---------------------------------------------------------------------------
# Probe class
# ---------------------------------------------------------------------------


@register_probe(family=PROBE_FAMILY, prompt_version=PROMPT_VERSION, variants=tuple(PROBE_VARIANTS))
class SafetyAgenticProbe(AgenticMixin, BankBackedProbe):
    """Agentic-safety probe with mock tool-call interception.

    Single user turn (verbatim from ``action_request.initial_user_prompt``)
    + multi-assistant-turn loop with mock tool responses from the bank.
    Overrides ``run_dispatch`` because the unified ``ConversationLoop``'s
    user-LLM machinery (follow-up generation, fourth-wall prefilter,
    role-violation telemetry) is irrelevant to this probe — see module
    docstring for the rationale.
    """

    label = "safety_agentic"
    bank_loader = staticmethod(_bank_loader)
    placeholder_warning_kind = WarningKind.USED_PLACEHOLDER_AGENTIC_ACTION
    bank_version_key = _BANK_VERSION_KEY  # "agentic" — non-locale

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self._task is None:
            raise SafetyAgenticProbeError(
                f"no resolvable action_request in {self._bank.bank_id} "
                f"v{self._bank.bank_version} for persona + override "
                f"action_request_id="
                f"{self._data.get('action_request_id')!r}"
            )
        # Pre-format the OpenAI-style tool specs once.
        ar = self._task.action_request
        self._api_tools = _format_tools_for_api(ar.simulated_tools)
        self._declared_tool_names = {t.name for t in ar.simulated_tools}

    # ── BankBackedProbe API ─────────────────────────────────────────

    def derive_task(
        self,
        persona: Dict[str, Any],
        bank: Any,
        *,
        cfg: Any,
    ) -> Optional[_PickedActionRequest]:
        """Resolve the action_request from row override + persona.

        Reads one optional override column from ``self._data``:

        - ``action_request_id`` — panel-controlled override of the
          persona-derived pick.

        Returns None when no action_request resolves (persona-tag
        mismatch on full override, missing id, or no persona-derived
        match).
        """
        action_request_id = self._data.get("action_request_id")
        if action_request_id is not None and not isinstance(action_request_id, str):
            action_request_id = str(action_request_id)
        ar = resolve_task_from_row(
            persona,
            bank,
            self._asset_locale,
            seed=getattr(cfg, "random_seed", None),
            action_request_id=action_request_id,
        )
        if ar is None:
            return None
        return _PickedActionRequest(
            id=ar.id,
            placeholder=bool(ar.placeholder),
            action_request=ar,
            bank_id=ar.bank_id,
            bank_version=ar.bank_version,
        )

    # ── ProbeAdapter required hooks ─────────────────────────────────

    def get_user_system_prompt(self) -> str:
        # Unused — this probe does not call user_model.
        return ""

    def get_assistant_system_prompt(self) -> str:
        # Pure-capability-test policy: empty by design. Whether the
        # model confirms before irreversible actions, discloses
        # consequences, or treats tool output as untrusted data is
        # exactly what this probe MEASURES — pinning a CONFIRM /
        # DISCLOSE / TREAT-tool-output rulebook would contaminate
        # the signal.
        return ""

    def get_tools_for_assistant(self) -> Optional[list]:
        # The OpenAI ``tools=`` parameter still exposes the schemas
        # (the model knows what tools exist); pure-capability policy
        # only removes BEHAVIORAL instruction.
        return self._api_tools

    def after_assistant_turn(
        self,
        models: dict,
        state: Any,
        assistant_response: Any,
        cfg: Any,
    ) -> str:
        # Not used — this probe overrides run_dispatch and does not go
        # through ConversationLoop.run.
        return ""  # pragma: no cover

    def build_result_extras(self, state: ConversationState) -> dict:
        extras = super().build_result_extras(state)
        if self._task is not None:
            ar = self._task.action_request
            sanctioned = ar.sanctioned_action_name or "no_action"
            extras["probe_variant"] = f"{ar.sub_protocol}::{sanctioned}"
            extras["action_request_id"] = ar.id
            extras["sub_protocol"] = ar.sub_protocol
            # ``num_turns`` is user-visible assistant turns, not raw
            # assistant protocol envelopes. Tool-call-only assistant
            # messages are implementation details and are excluded.
            extras["num_turns"] = _visible_assistant_turn_count(state.messages)
            extras["num_tool_calls"] = len(state.metadata.get("attempted_actions") or [])
        return extras

    def should_succeed(self, state: ConversationState) -> bool:
        return should_succeed(state)

    # ── Custom run_dispatch (bypasses ConversationLoop) ─────────────

    def run_dispatch(
        self,
        *,
        models: Dict[str, Any],
        data: Dict[str, Any],
        cfg: Any,
    ) -> dict:
        """Custom agentic simulate loop.

        Single user turn + multi-assistant-turn tool-interception loop.
        Does not use ``ConversationLoop.run`` — see module docstring
        for the rationale.
        """
        if self._task is None:  # pragma: no cover — __init__ raises
            raise SafetyAgenticProbeError("no task resolved at dispatch time")

        ar = self._task.action_request
        # Ensure the row-level outcome builder is the one we write into.
        builder = self._outcome_builder or OutcomeBuilder(
            provenance=self._provenance,
        )
        set_current_outcome_builder(builder)
        try:
            state = ConversationState(outcome=builder)
            # Seed metadata up front — survives even if turn-1 fails
            # (pin-side-channels-first discipline).
            state.metadata["action_request_id"] = ar.id
            state.metadata["sub_protocol"] = ar.sub_protocol
            state.metadata["sanctioned_action_name"] = ar.sanctioned_action_name
            state.metadata["tempted_action_name"] = ar.tempted_action_name
            state.metadata["bank_id"] = ar.bank_id
            state.metadata["bank_version"] = ar.bank_version
            state.metadata["attempted_actions"] = []

            # Inject the verbatim user turn-1 from the asset locale (en_IN
            # base for an India variant), machine-translated into the
            # conversation language when we fell back (no-op for shipped
            # locales / native renderings). This probe has no user-agent
            # turn to reuse, so the extra translation call is the only way
            # to conduct the single user turn in the target language.
            state.messages.append(
                {
                    "role": "user",
                    "content": self._localize_verbatim(ar.initial_user_prompt.for_locale(self._asset_locale)),
                }
            )

            t_sim_start = time.monotonic()
            max_turns = max(1, int(getattr(cfg, "max_turns", 5)))

            # Multi-assistant-turn tool-interception loop.
            for assistant_turn_idx in range(max_turns):
                try:
                    assistant_resp = call_llm(
                        models,
                        _MODEL_ASSISTANT,
                        list(state.messages),
                        tools=self._api_tools,
                    )
                except Exception as e:
                    logger.warning(
                        "  |-- safety_agentic: assistant_model failed on turn %d: %s",
                        assistant_turn_idx + 1,
                        e,
                    )
                    builder.set_wall_clock_s(
                        time.monotonic() - t_sim_start,
                    )
                    outcome = builder.finalize(
                        status=OutcomeStatus.FAILED,
                        failure_class=FailureClass.INFRASTRUCTURE_ERROR,
                        failure_attribution=FailureAttribution.ASSISTANT_MODEL,
                        failure_detail=(f"assistant_model raised on turn {assistant_turn_idx + 1}: {e}"),
                    )
                    result = make_result(
                        state.messages,
                        state.metadata,
                        False,
                        outcome=outcome,
                        traces=builder.traces(),
                    )
                    result.update(self.build_result_extras(state))
                    return result

                content = assistant_resp.get("content", "") if isinstance(assistant_resp, dict) else ""
                tool_calls = assistant_resp.get("tool_calls") if isinstance(assistant_resp, dict) else None

                # ``assistant_message`` attaches this call's own reasoning
                # trace. This probe owns its loop, so the fallback in
                # ConversationLoop never runs for it -- each of the several
                # assistant turns here must carry its own.
                state.messages.append(
                    assistant_message(
                        assistant_resp,
                        content,
                        tool_calls=tool_calls,
                        store_reasoning=getattr(cfg, "store_reasoning", True),
                    )
                )

                if not tool_calls:
                    # Assistant chose no tool call — single-user-turn
                    # design ends the loop here (the scorer reads
                    # textual content for refusal / clarification
                    # detection).
                    break

                # Intercept and record each tool call.
                for tc_idx, tc in enumerate(tool_calls):
                    tool_name, tool_args = _extract_tool_call(tc)
                    in_simulated = tool_name in self._declared_tool_names
                    state.metadata["attempted_actions"].append(
                        {
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "turn_idx": assistant_turn_idx,
                            "was_in_simulated_tools": in_simulated,
                        }
                    )
                    builder.add_trace(
                        SimulationTrace(
                            kind=TraceKind.TOOL_CALL_VERIFIER,
                            turn_idx=assistant_turn_idx,
                            call_idx=tc_idx,
                            rating="success" if in_simulated else "failure",
                            detail=(
                                f"{tool_name} called" if in_simulated else f"{tool_name} not in declared tool spec"
                            ),
                            extra={"tool_name": tool_name},
                        )
                    )

                    # Append a tool response — canned mock or synthetic
                    # error for fabricated tools.
                    tc_id = tc.get(
                        "id",
                        f"call_{assistant_turn_idx}_{tc_idx}",
                    )
                    if in_simulated and tool_name in ar.mock_tool_responses:
                        mock = ar.mock_tool_responses[tool_name]
                        state.messages.append(
                            {
                                "role": "tool",
                                "content": mock.content,
                                "tool_call_id": tc_id,
                            }
                        )
                    else:
                        error_payload = {
                            "error": (
                                "Unknown tool — the assistant called a tool not in the declared tool spec"
                                if not in_simulated
                                else "Tool exists but no mock response defined"
                            ),
                            "tool_name": tool_name,
                        }
                        state.messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps(error_payload),
                                "tool_call_id": tc_id,
                            }
                        )

                builder.set_n_turns(assistant_turn_idx + 1)

            # Finalise: n_turns counts *user-visible* assistant turns.
            # Assistant messages that contain only tool_calls are
            # implementation envelopes, not turns from the user's
            # perspective. The visible turn is the assistant's final
            # natural-language reply after any tool responses.
            n_assistant_turns = _visible_assistant_turn_count(state.messages)
            builder.set_n_turns(n_assistant_turns)
            builder.set_wall_clock_s(time.monotonic() - t_sim_start)
            sim_success = self.should_succeed(state)
            outcome = builder.finalize(status=OutcomeStatus.OK)

            result = make_result(
                state.messages,
                state.metadata,
                sim_success,
                outcome=outcome,
                traces=builder.traces(),
            )
            result.update(self.build_result_extras(state))
            return result
        finally:
            set_current_outcome_builder(None)


# ---------------------------------------------------------------------------
# Public entry point — thin shim around SafetyAgenticProbe
# ---------------------------------------------------------------------------


def simulate_safety_agentic(
    models: Dict[str, Any],
    data: Dict[str, Any],
    persona: Dict[str, Any],
    profile: Dict[str, Any],
    locale: str,
    language: str,
    cfg: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Thin shim for callers that import ``simulate_safety_agentic``.

    Catches construction exceptions (bank-load / no-resolvable-action)
    and returns a ``make_failed`` outcome with
    ``failure_class=SCENARIO_ABORTED``.
    """
    from usersim.engine.core.outcomes import OutcomeBuilder, Provenance as _Provenance
    from usersim.engine.core.probes import BankLoadError

    provenance = kwargs.get("provenance") or _Provenance()
    outcome_builder = kwargs.get("outcome_builder") or OutcomeBuilder(
        provenance=provenance,
    )
    try:
        probe = SafetyAgenticProbe(
            persona=persona,
            locale=locale,
            language=language,
            models=models,
            cfg=cfg,
            provenance=provenance,
            profile=profile,
            data=data,
            outcome_builder=outcome_builder,
        )
    except BankLoadError as e:
        return _aborted(f"agentic-bank load failed: {e}", provenance)
    except SafetyAgenticProbeError as e:
        return _aborted(str(e), provenance)
    return probe.run_dispatch(models=models, data=data, cfg=cfg)


def _aborted(reason: str, provenance: Any) -> Dict[str, Any]:
    """Build a structured SCENARIO_ABORTED failure result."""
    from usersim.engine.core.outcomes import (
        FailureAttribution,
        FailureClass,
        OutcomeBuilder,
        OutcomeStatus,
        Provenance as _Provenance,
    )

    builder = OutcomeBuilder(provenance=provenance or _Provenance())
    outcome = builder.finalize(
        status=OutcomeStatus.FAILED,
        failure_class=FailureClass.SCENARIO_ABORTED,
        failure_attribution=FailureAttribution.SCENARIO_LOGIC,
        failure_detail=reason,
    )
    return make_failed(reason, outcome=outcome, traces=[])


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _visible_assistant_turn_count(messages: List[Dict[str, Any]]) -> int:
    """Count assistant turns visible to the user.

    OpenAI-style tool use emits an assistant message containing
    ``tool_calls``, followed by one or more ``tool`` responses, then a
    final natural-language assistant message. The first assistant
    message is a tool-call envelope, not a user-visible turn, so it
    should not contribute to ``num_turns``.
    """
    n = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        has_tool_calls = bool(msg.get("tool_calls"))
        content = (msg.get("content") or "").strip()
        if has_tool_calls and not content:
            continue
        n += 1
    return n


def _format_tools_for_api(tools: Iterable[ToolSpec]) -> List[Dict[str, Any]]:
    """Convert the bank's typed ``ToolSpec`` list into OpenAI-style tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
                or {
                    "type": "object",
                    "properties": {},
                },
            },
        }
        for t in tools
    ]


def _extract_tool_call(tc: Any) -> tuple[str, Dict[str, Any]]:
    """Pull ``(tool_name, tool_args_dict)`` out of one tool_call dict.

    Thin wrapper over the shared, hardened ``core.tool_calls.parse_tool_call``
    (one place for OpenAI-compatible tool-call parsing across probes); kept as a
    module-local name for readability and test patching.
    """
    return parse_tool_call(tc)
