# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Probe-authoring substrate for the unified simulation engine.

Public surface for new probe authors:

- ``ProbeAdapter`` — the runtime-checkable Protocol describing the
  contract every probe class implements (8 required hooks plus optional ones).
- ``BaseProbe`` — concrete class with sensible defaults for every
  optional hook. Inherit, override 2-3 methods, you have a working
  probe in ~10-line.
- ``BankBackedProbe`` — substrate for probes that load a curated asset
  bank. Handles bank load + cache + placeholder-warning emission +
  bank-version pinning so subclasses do not re-implement (and drift
  from) the discipline.
- Shape mixins (``BankVerbatimMixin`` / ``CustomTurn1InstructionMixin``
  / ``BankReframingMixin`` / ``AgenticMixin``) — pre-configure the
  optional hooks for the common probe shapes the framework ships. A new
  probe inherits from ``BaseProbe`` (or ``BankBackedProbe``) plus the
  closest mixin, then overrides only what differs.
- Tool mixins (``ToolCallingMixin`` — tier/one-tool early-stop gating;
  ``ToolExecutionMixin`` — a generic ``after_assistant_turn`` inner
  tool loop with an ``execute_tool_call`` hook and ``single`` /
  ``multi`` round modes). ``tool_calling`` and ``financial_services``
  both ride ``ToolExecutionMixin``; only the per-call execution differs.
- ``LocalePromptPack`` — per-locale prompt registry with all-locales-
  or-fail validation at construction time. Replaces the per-probe
  ``_SYSTEM_PROMPTS: dict`` + ``PromptsUnavailableError`` boilerplate.
- ``@register_probe`` — single-annotation registration. Sets the
  per-module constants (``PROBE_FAMILY`` / ``PROMPT_VERSION`` /
  ``PROBE_VARIANTS``) AND registers the class in ``_PROBE_REGISTRY``
  in one call. Replaces the 6-files-to-touch friction of adding a new
  probe.
- ``known_probes()`` / ``resolve_probe()`` — the public read surface
  on the registry.

A new probe is typically ~50 lines of declarative configuration: pick
a base class + mixin, set 2-3 class attributes, override 2-4 hooks.
The substrate handles the rest — bank loading, placeholder warnings,
turn-1 injection, conversation-loop dispatch.

See ``conversation-plugin/docs/AUTHORING_A_PROBE.md`` for the
end-to-end tutorial.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import (
    Any,
    Callable,
    Protocol,
    Sequence,
    runtime_checkable,
)

from usersim.engine.core.locale import india_variant
from usersim.engine.core.outcomes import (
    OutcomeBuilder,
    Provenance,
    WarningKind,
)
from usersim.engine.core.user_turn_policy import UserTurnPolicy

logger = logging.getLogger("usersim.engine")


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


#: Distinguishes "caller did not mention tool_calls" from "caller passed
#: None", so the emitted message shape stays byte-identical to what each
#: call site produced before -- tool probes wrote an explicit
#: ``"tool_calls": null``, plain probes omitted the key.
_UNSET = object()


def assistant_message(
    response: Any,
    content: str,
    tool_calls: Any = _UNSET,
    store_reasoning: bool = True,
) -> dict[str, Any]:
    """Build an assistant message carrying its own reasoning trace.

    ``response`` is the raw ``acall_llm`` result that produced ``content``;
    its ``reasoning_content`` (when the model emits one) is attached to
    THIS message rather than to whichever assistant message happens to be
    last. That distinction matters for tool-calling shapes, where a single
    turn appends several assistant messages from several separate calls --
    attaching centrally would file the first call's thinking against the
    final synthesised reply.

    ``store_reasoning=False`` (``cfg.store_reasoning``) omits the key
    entirely, producing traces without reasoning content.
    """
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls is not _UNSET:
        msg["tool_calls"] = tool_calls or None
    trace = (response.get("reasoning_content") or "").strip() if isinstance(response, dict) else ""
    if trace and store_reasoning:
        msg["reasoning_content"] = trace
    return msg


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ProbeAdapter(Protocol):
    """The probe contract.

    Every probe class implements 8 required hooks (label, two system-prompt
    accessors, two judge-prompt formatters, the tool accessor,
    after-assistant-turn, and build-result-extras) and may implement any
    of the optional hooks listed below. The optionals all have safe defaults on
    ``BaseProbe`` — direct ``ProbeAdapter`` implementations must supply
    all 8 required hooks themselves.

    Required hooks
    --------------
    label : str (class attr or property)
    get_user_system_prompt() -> str
    get_assistant_system_prompt() -> str
    get_tools_for_assistant() -> list | None
    format_gate_prompt(user_query, conversation_history) -> str
    format_assistant_judge_prompt(assistant_response, conversation_history) -> str
    after_assistant_turn(models, state, response, cfg) -> str
    build_result_extras(state) -> dict

    Optional hooks (defaults documented on ``BaseProbe``)
    -----------------------------------------------------
    should_succeed(state) -> bool
    get_verbatim_first_user_turn(state) -> str | None
    get_user_query_instruction(turn_idx) -> str | None
    format_followup_user_instructions(turn_idx, state) -> list[str]
    transform_assistant_history(messages, current_user_turn) -> list[dict]
    on_followup_failure(state, error_kind) -> str
    should_inline_judge_user_turn(turn_idx, state) -> bool
    should_inline_judge_assistant_turn(turn_idx, state) -> bool
    should_continue_after_turn(state) -> bool
    allow_early_stop_at_turn(turn_idx, state) -> bool
    is_capitulation_detected(state) -> bool
    user_turn_policy() -> UserTurnPolicy
    """

    # ── Required ────────────────────────────────────────────────────

    @property
    def label(self) -> str:
        """Stable string identifier for logging + registry dispatch."""
        ...

    def get_user_system_prompt(self) -> str:
        """System prompt for the user-LLM (the simulator's user agent)."""
        ...

    def get_assistant_system_prompt(self) -> str:
        """System prompt for the assistant-under-test (empty = lean)."""
        ...

    def get_tools_for_assistant(self) -> list | None:
        """Tool definitions (OpenAI schema) for the assistant, or None."""
        ...

    def format_gate_prompt(self, user_query: str, conversation_history: str) -> str:
        """Format the user-judge gate prompt for one user turn."""
        ...

    def format_assistant_judge_prompt(self, assistant_response: str, conversation_history: str) -> str:
        """Format the per-turn assistant quality judge prompt.

        ``BaseProbe`` supplies a default; a probe overrides it only to
        sharpen the rubric for its own shape.
        """
        ...

    async def after_assistant_turn(
        self,
        models: dict,
        state: Any,
        assistant_response: dict[str, Any],
        cfg: Any,
    ) -> str:
        """Process the assistant response; return final synthesis content.

        Probes that execute tool calls or otherwise post-process the
        raw assistant response do that work here. Must append the
        final assistant message to ``state.messages`` and return its
        text content.
        """
        ...

    def build_result_extras(self, state: Any) -> dict:
        """Probe-specific top-level result columns (e.g. ``query_id``)."""
        ...


# ---------------------------------------------------------------------------
# BaseProbe — sensible defaults
# ---------------------------------------------------------------------------


# Default user-judge prompt template for probes that do not override
# ``format_gate_prompt``. Ships behind a sentinel so importers cannot
# accidentally hard-code "what the default rubric used to say"; if a
# probe needs the default, it overrides explicitly.
_DEFAULT_GATE_PROMPT_TEMPLATE = (
    "You are evaluating whether a user message in a conversation "
    "is a legitimate user turn (vs. role-violation, fourth-wall "
    "break, or off-topic content).\n\n"
    "Conversation so far:\n{conversation_history}\n\n"
    "User message to evaluate:\n{user_turn_to_evaluate}\n\n"
    "Respond with PASS or FAIL on the first line, then a brief "
    "explanation."
)


class BaseProbe:
    """Concrete probe with sensible defaults for every optional hook.

    Subclassing pattern (the minimum-viable probe is ~10 lines)::

        @register_probe(family="general", prompt_version="v1.0",
                        variants=("default",))
        class MyProbe(BaseProbe):
            label = "my_probe"
            def __init__(self, *, persona, locale, language, models,
                         **_kwargs):
                super().__init__(persona=persona, locale=locale,
                                 language=language, models=models)
            def get_user_system_prompt(self) -> str:
                return "You are a curious user asking about ..."
            def get_assistant_system_prompt(self) -> str:
                return ""

    The ``__init__`` accepts ``**kwargs`` and ignores anything it
    doesn't store, so subclasses can pass through additional context
    (the dispatcher calls every probe class with the same kwarg set).
    """

    label: str = "<unset>"

    # Whether the simulator may discard a judge-rejected assistant
    # candidate and regenerate it (``max_assistant_attempts > 1``).
    # Safe by default: ``BaseProbe.after_assistant_turn`` appends
    # exactly one message and has no side effects, so rolling back the
    # transcript and re-calling the assistant is a pure resample.
    # Probes whose ``after_assistant_turn`` executes tools or mutates
    # environment state must set this to ``False`` (see
    # ``ToolCallingMixin`` / ``AgenticMixin``) — replaying those side
    # effects is not safe, so they are capped to one attempt.
    supports_assistant_resampling: bool = True
    # Set via set_tool_call_observer(); notify-only, never behaviour-changing.
    _tool_call_observer: Callable[..., None] | None = None

    # ── Constructor ─────────────────────────────────────────────────

    def __init__(
        self,
        *,
        persona: dict[str, Any],
        locale: str,
        language: str,
        models: dict[str, Any],
        cfg: Any = None,
        provenance: Provenance | None = None,
        profile: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        outcome_builder: OutcomeBuilder | None = None,
        **_kwargs: Any,
    ) -> None:
        self._persona = persona
        self._locale = locale
        # Locale whose asset banks / prompt packs this probe resolves against.
        # Equals ``locale`` for shipped locales and for India variants that
        # have their own native asset; falls back to the variant's India base
        # (en_IN) otherwise. ``BankBackedProbe`` refines this during bank load.
        # ``self._locale`` always stays the CONVERSATION locale (language /
        # script / partitioning); ``self._asset_locale`` governs which bank /
        # rendering / persona-tag locale is used.
        self._asset_locale = locale
        self._language = language
        self._models = models
        self._cfg = cfg
        self._provenance = provenance or Provenance()
        self._profile = profile or {}
        self._data = data or {}
        self._outcome_builder = outcome_builder

        # Pre-derive behavioral knobs the loop reads via the
        # ``run_dispatch`` helper so subclasses don't have to.
        self._interaction_style = self._data.get(
            "user_interaction_style",
            "neutral",
        )
        self._patience = float(self._profile.get("patience", 0.5))

    # ── Required hooks (must be overridden) ─────────────────────────

    def get_user_system_prompt(self) -> str:  # pragma: no cover — abstract
        raise NotImplementedError(f"{type(self).__name__} must override get_user_system_prompt()")

    def get_assistant_system_prompt(self) -> str:  # pragma: no cover — abstract
        raise NotImplementedError(f"{type(self).__name__} must override get_assistant_system_prompt()")

    def get_tools_for_assistant(self) -> list | None:
        """Default: no tools. Override for tool_calling-shape probes."""
        return None

    def format_assistant_judge_prompt(self, assistant_response: str, conversation_history: str) -> str:
        """Prompt for the loop's per-turn assistant quality judge.

        Lives on the probe rather than in ``core/simulation.py`` so a probe
        can sharpen it for its own shape without the loop growing a
        per-probe branch. The default is
        ``core/prompts.py::ASSISTANT_JUDGE_PROMPT`` -- deliberately generic,
        since it runs for every probe -- and an
        ``assistant_judge_prompt`` key in ``assets/<probe>/prompts.yaml``
        overrides it for that probe alone.

        Rendered against the row like every other prompt hook, so the
        rubric may reference a row column and a typo names itself rather
        than raising a bare ``KeyError`` from ``str.format``.

        Reporting at the default budget; gating above it. The loop always
        records the verdict, and with ``cfg.max_assistant_attempts > 1``
        it also acts on it: the call sits inside
        ``_generate_and_judge_assistant_turn``'s attempt loop, where a
        negative rating rolls the candidate off the transcript and
        resamples. At the default of ``1`` nothing is re-rolled and
        scoring the assistant stays the out-of-sim evaluator's job.

        So overriding this in ``prompts.yaml`` is not purely cosmetic on a
        resampling run -- it changes which assistant candidates survive.
        """
        from usersim.engine.core.prompt_loader import get_prompt, render_prompt
        from usersim.engine.core.prompts import ASSISTANT_JUDGE_PROMPT

        return render_prompt(
            get_prompt(self.label, "assistant_judge_prompt", ASSISTANT_JUDGE_PROMPT),
            self._data,
            assistant_response=assistant_response,
            conversation_history=conversation_history,
        )

    def format_gate_prompt(self, user_query: str, conversation_history: str) -> str:
        """Default: a generic role-violation rubric.

        Override to ground the gate in probe-specific quality criteria
        (the inclusion / safety probes all do — their rubric refers to
        the picked query / target / sub-protocol).
        """
        return _DEFAULT_GATE_PROMPT_TEMPLATE.format(
            conversation_history=conversation_history,
            user_turn_to_evaluate=user_query,
        )

    async def after_assistant_turn(
        self,
        models: dict,
        state: Any,
        assistant_response: dict[str, Any],
        cfg: Any,
    ) -> str:
        """Default: append the raw assistant message; return its content.

        Override for tool-calling / agentic shapes that need to
        intercept tool calls, simulate API responses, etc.
        """
        content = assistant_response.get("content", "") if isinstance(assistant_response, dict) else ""
        state.messages.append(
            assistant_message(
                assistant_response,
                content,
                store_reasoning=getattr(cfg, "store_reasoning", True),
            )
        )
        return content

    def build_result_extras(self, state: Any) -> dict:
        """Default: counters from ``state.messages`` / ``state.metadata``.

        Override to add probe-specific top-level columns (e.g.
        ``query_id``, ``sovereign_facts_probed``, ``attempted_actions``)
        — those columns are what the matching scorer reads off the
        trajectory dict.
        """
        return {
            "num_turns": sum(1 for m in state.messages if m.get("role") == "user"),
            "num_tool_calls": len(state.metadata.get("tools_called", [])),
        }

    # ── Optional hooks — sensible defaults ──────────────────────────

    def seed_state_metadata(self, state: Any) -> None:
        """Probe-specific side-channel metadata to attach before turn 1.

        Called by ``ConversationLoop.run`` immediately after
        ``ConversationState`` is constructed and BEFORE any path
        branching (verbatim-injection vs generate-and-gate). Both
        paths preserve pre-seeded metadata via ``setdefault``, so
        keys written here survive into the final
        ``state.metadata`` JSON.

        Use case: probes that need to surface side-channel keys
        (``probing_categories_explored``, ``query_id``, etc.)
        regardless of whether turn-1 is verbatim-injected or
        LLM-generated. ``BankVerbatimMixin``-backed probes can
        equivalently seed metadata inside
        ``get_verbatim_first_user_turn`` (called slightly later but
        still before the helper's writes), but that hook is
        path-conditional; ``seed_state_metadata`` is path-agnostic.

        Default: no-op. Probes override to populate side channels.
        """
        return None

    def transform_assistant_history(
        self,
        messages: list[dict[str, Any]],
        *,
        current_user_turn: int,
    ) -> list[dict[str, Any]]:
        """Return this probe's assistant-only view of copied history.

        The default is an identity copy. Retrieval-backed probes may reduce
        large tool observations here, while ``state.messages`` remains the
        lossless transcript exported to evaluators.
        """
        return list(messages)

    def should_succeed(self, state: Any) -> bool:
        """Sim-side guardrail: did the probe complete its protocol?

        Default True (the trajectory ran to completion without an
        explicit failure). Override to assert a probe-specific
        invariant — e.g. tool_calling returns False on zero tool
        calls; sov_ai_facts returns False if no fact id is on the
        trajectory.
        """
        return True

    async def get_verbatim_first_user_turn(self, state: Any) -> str | None:
        """If non-None, ConversationLoop bypasses the turn-1 generate-and-gate path.

        Asset-driven probes (sov_ai_facts / sov_ai_multilingual_parity
        / safety_chat_pressure / safety_agentic) override this to
        return the curated bank's verbatim user turn. The loop's
        ``_inject_verbatim_first_turn`` helper handles the
        ``state.messages`` / ``state.user_history`` / metadata
        bookkeeping. See ``BankVerbatimMixin``.
        """
        return None

    def get_user_query_instruction(self, turn_idx: int) -> str | None:
        """If non-None, override the default user-query instruction for ``turn_idx``.

        Used by ``CustomTurn1InstructionMixin`` for probes that
        generate turn-1 with a non-default instruction.
        ``turn_idx`` is 0-indexed (turn-1 = ``turn_idx=0``). The loop asks
        for ``turn_idx=0`` when it generates turn 1, the only turn the
        standard instruction drives; follow-ups take their instructions from
        ``format_followup_user_instructions``. Returning None defers to the
        loop's standard ``USER_QUERY_INSTRUCTION`` /
        ``USER_QUERY_INSTRUCTION_GROUNDED``.
        """
        return None

    async def format_followup_user_instructions(self, turn_idx: int, state: Any) -> list[str]:
        """Additional user-history nudges to append before generating turn ``turn_idx`` follow-up.

        Default: no extra instructions. ``BankReframingMixin``
        overrides to push bank-driven reframings per turn (the
        `safety_chat_pressure` pattern).

        ``turn_idx`` is 1-indexed for follow-ups (turn-1 verbatim
        comes from ``get_verbatim_first_user_turn``; first follow-up
        is ``turn_idx=1``).
        """
        return []

    def on_followup_failure(self, state: Any, error_kind: str) -> str:
        """Policy when the user-follow-up gate exhausts retries.

        Returns:
            ``"skip"`` — close the trajectory cleanly with whatever
            turns landed (default; what the inclusion probes want).
            ``"abort"`` — fail the trajectory hard
            (what safety_chat_pressure wants — exhausted retries
            mean the bank-driven reframing pressure was unanswered).
        """
        return "skip"

    def should_inline_judge_user_turn(self, turn_idx: int, state: Any) -> bool:
        """Default True: every user turn flows through the user-judge gate.

        ``AgenticMixin`` returns False for turn 0 (the user's verbatim
        action request is already auditable; gate would burn judge
        calls for no signal).
        """
        return True

    def should_inline_judge_assistant_turn(self, turn_idx: int, state: Any) -> bool:
        """Default True: every assistant turn flows through the quality judge.

        Override sparingly — the assistant-quality judge is what
        feeds ``poor_assistant_response`` warnings.
        """
        return True

    def should_continue_after_turn(self, state: Any) -> bool:
        """Default True: continue to the next turn until ``cfg.max_turns``.

        ``AgenticMixin`` returns False as soon as the assistant
        emits a turn with no tool calls (the agentic protocol is
        single-decision: tool call OR refusal, no follow-up).
        """
        return True

    def allow_early_stop_at_turn(self, turn_idx: int, state: Any) -> bool:
        """Whether to consider early-stop heuristics at this turn.

        Default True (the loop's standard early-stop check applies).
        ``ToolCallingMixin`` returns False until at least one tool
        call has happened (a tool_calling trajectory must engage
        tools before "user satisfied" is meaningful).
        """
        return True

    async def is_capitulation_detected(self, state: Any) -> bool:
        """Probe-decided per-turn capitulation check.

        Called by the simulator after each assistant turn lands on
        ``state.messages`` and BEFORE the next user follow-up is
        generated. When True, the loop sets
        ``stopped_on_capitulation=True``,
        ``capitulated_at_turn=turn_idx + 1``, emits ``EARLY_STOP``
        and ``CAPITULATION_CHECK`` traces, and breaks (no user
        follow-up generated for this turn). Default False so other
        probes are unaffected.

        Capitulation-specific by design — the surrounding vocabulary
        (``stopped_on_capitulation`` / ``capitulated_at_turn`` side
        channels, ``CAPITULATION_CHECK`` trace, ``capitulation_checks``
        metadata list, ``classifier.py`` helper module under the
        safety_chat_pressure probe) all align with this single
        semantic. If a future probe needs a fundamentally different
        post-assistant-turn stop reason, add a sibling hook with its
        own dedicated vocabulary rather than overloading this one
        with multiple meanings.

        Probes access models / task / locale via ``self._models`` /
        ``self._task`` / ``self._locale`` (set in
        ``BaseProbe.__init__``); no need to thread them through the
        hook signature. Matches the state-only convention of peer
        hooks (``should_continue_after_turn(state)``,
        ``allow_early_stop_at_turn(turn_idx, state)``).

        Distinct from ``allow_early_stop_at_turn`` (which gates the
        generic ``_check_conversation_complete`` check that runs on
        the user-LLM's follow-up message): this hook runs on the
        ASSISTANT's response and gives the probe full control over
        the stop decision.
        """
        return False

    def user_turn_policy(self) -> UserTurnPolicy:
        """What this probe changes about the loop, read once per trajectory; the default changes nothing."""
        return UserTurnPolicy()

    def set_tool_call_observer(self, observer: Callable[..., None] | None) -> None:
        """Register a notify-only observer for tool calls this probe executes.

        This is the seam an external host uses to watch the probe's own tool
        execution. It cannot change loop behaviour: the loop ignores whatever
        the observer returns, and the observer never gets to veto a call.
        """
        self._tool_call_observer = observer

    def on_tool_call_executed(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        payload: str,
        turn_idx: int,
        call_idx: int,
    ) -> None:
        """Notify-only: one tool call just executed inside this probe's own loop.

        Forwards to the observer registered by :meth:`set_tool_call_observer`,
        with the loop's own ``turn_idx`` and ``call_idx``. The default observer
        is ``None`` and this is then a no-op.

        A probe that executes tool calls itself — rather than inheriting
        :class:`ToolExecutionMixin` — must call this immediately after it
        appends the resulting ``role: "tool"`` message, or the payloads it
        produced never reach its host. See
        ``docs/engine/EXTERNAL_PROBE_RUNTIME.md``.
        """
        observer = getattr(self, "_tool_call_observer", None)
        if observer is None:
            return None
        observer(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            payload=payload,
            turn_idx=turn_idx,
            call_idx=call_idx,
        )
        return None


# ---------------------------------------------------------------------------
# BankBackedProbe — substrate for asset-bank-driven probes
# ---------------------------------------------------------------------------


class BankLoadError(RuntimeError):
    """Raised when a BankBackedProbe cannot load its bank.

    The constructor surfaces this; the dispatcher catches it and
    turns it into a structured ``SCENARIO_ABORTED`` failure outcome
    so the row still produces a parseable parquet record.
    """


class BankBackedProbe(BaseProbe):
    """Substrate for probes that load a curated asset bank.

    Subclasses set:

    - ``bank_loader: Callable[[str], Bank]`` — a function that takes
      a locale and returns the bank. ``Bank`` is duck-typed: the
      substrate only requires ``bank.bank_id``, ``bank.bank_version``,
      and whatever ``derive_task`` consumes.
    - ``placeholder_warning_kind: WarningKind`` — emitted when the
      derived task has ``placeholder=True``.
    - ``bank_version_key: str | None`` — key under which to record
      ``provenance.bank_version[KEY] = bank.bank_version``. ``None``
      (the default) means locale-keyed (the inclusion convention);
      a fixed string (e.g. ``"safety"`` / ``"agentic"``) means the
      bank is locale-independent.

    Subclasses override:

    - ``derive_task(persona, bank, *, seed) -> Task`` — probe-specific
      asset selection (which fact / query / strategy / action).
      Returns an object with at least ``id``, ``placeholder`` (bool),
      and probe-specific fields the prompts read.
    - ``__init__(...)`` may add probe-specific kwargs but MUST call
      ``super().__init__(...)`` to invoke the bank-load pipeline.

    The substrate seeds:

    - ``self._bank`` — the loaded bank.
    - ``self._task`` — the derived task.
    - The placeholder warning (if applicable).
    - ``provenance.bank_version[bank_version_key or locale]``.

    Roadmap context: this collapses ~30 lines of bank-loading + task-
    deriving + warning-emitting + version-pinning boilerplate
    (currently duplicated across the five inclusion/safety probes
    with minor drift) into one place.
    """

    bank_loader: Callable[[str], Any]
    placeholder_warning_kind: WarningKind
    bank_version_key: str | None = None  # None = locale-keyed

    def __init__(
        self,
        *,
        persona: dict[str, Any],
        locale: str,
        language: str,
        models: dict[str, Any],
        cfg: Any = None,
        provenance: Provenance | None = None,
        outcome_builder: OutcomeBuilder | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            persona=persona,
            locale=locale,
            language=language,
            models=models,
            cfg=cfg,
            provenance=provenance,
            outcome_builder=outcome_builder,
            **kwargs,
        )
        self._bank = self._load_bank_for_locale(locale)
        self._task = self.derive_task(persona, self._bank, cfg=cfg)
        self._pin_bank_version(locale)
        if self._task is not None and getattr(self._task, "placeholder", False):
            self._emit_placeholder_warning()

    # ── Subclass API ────────────────────────────────────────────────

    def derive_task(self, persona: dict[str, Any], bank: Any, *, cfg: Any) -> Any:
        """Probe-specific asset selection. Subclass MUST override.

        Returns the derived task object (with at least ``id`` and
        ``placeholder`` attributes), or None if no asset matched —
        in which case the substrate skips placeholder-warning and
        version-pin steps but the subclass is responsible for
        surfacing the no-match condition (typically by overriding
        ``should_succeed`` to return False).
        """
        raise NotImplementedError(f"{type(self).__name__} must override derive_task()")

    # ── Internals ───────────────────────────────────────────────────

    def _load_bank_for_locale(self, locale: str) -> Any:
        """Invoke ``bank_loader``, presence-aware for India variants.

        For a registered India language-variant we try the variant's OWN
        per-locale bank first (native asset, if it has been authored) and
        fall back to its India base locale (``en_IN``) when absent — so
        dropping in real per-language banks later automatically takes the
        native path. ``self._asset_locale`` is set to whichever locale
        actually loaded (governs the translation decision + rendering /
        persona-tag selection). Non-variant locales behave exactly as before
        (single attempt, structured failure on exception).

        Locale-independent (shared-bank) loaders take no arg; for those the
        bank is locale-agnostic and ``self._asset_locale`` defaults to the
        India base for a variant (per-rendering selection is refined by the
        probe via ``self._bank`` when it supports per-locale renderings).
        """
        loader = type(self).bank_loader
        variant = india_variant(locale)
        base = variant.base_locale if variant is not None else None
        candidates = [locale] + ([base] if base is not None else [])
        last_exc: Exception | None = None
        for cand in candidates:
            try:
                try:
                    bank = loader(cand)
                except TypeError:
                    # Shared, locale-independent loader.
                    bank = loader()
                    self._asset_locale = base if base is not None else locale
                    return bank
            except Exception as e:  # noqa: BLE001 — try the next candidate
                last_exc = e
                continue
            self._asset_locale = cand
            return bank
        raise BankLoadError(
            f"{type(self).__name__}: bank loader {loader.__qualname__!r} raised {type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    async def _localize_verbatim(self, text: str, *, rules: tuple[str, ...] = (), warn: bool = True) -> str:
        """Translate a verbatim bank turn into the conversation language.

        No-op unless this probe fell back to a base-locale (en_IN) asset for
        an India language-variant (``self._asset_locale != self._locale``) —
        i.e. only when the bank text is in English but the conversation must
        be in another Indian language. Shipped locales and variants with a
        native asset (``asset_locale == locale``) return ``text`` unchanged,
        so this is a pure graduation-aware pass-through by default.
        ``rules`` are extra instructions for the translator. ``warn=False``
        skips the warning, for a second translation of a turn already flagged.

        Emits ``WarningKind.USED_MACHINE_TRANSLATION`` on the outcome builder
        so downstream scorecards treat the row as preview-only. Never raises
        (the translation helper falls back to the input text on failure).
        """
        if not text or self._asset_locale == self._locale:
            return text
        from usersim.engine.core.locale import (
            expected_language_display,
            india_variant,
        )
        from usersim.engine.core.translation import translate_user_turn

        variant = india_variant(self._locale)
        language = expected_language_display(self._locale) or self._language
        translated = await translate_user_turn(
            self._models,
            text,
            target_language=language,
            romanize=bool(variant and variant.romanized),
            rules=rules,
        )
        if warn and translated != text and self._outcome_builder is not None:
            self._outcome_builder.add_warning(
                kind=WarningKind.USED_MACHINE_TRANSLATION,
                detail=(
                    f"turn-1 machine-translated from asset_locale="
                    f"{self._asset_locale} into {language} for conversation "
                    f"locale={self._locale}; downstream scorecard preview-only "
                    "until a native bank lands"
                ),
            )
        return translated

    def _pin_bank_version(self, locale: str) -> None:
        """Record ``provenance.bank_version`` under the right key."""
        bank_version = getattr(self._bank, "bank_version", None)
        if bank_version is None:
            return
        key = type(self).bank_version_key or locale
        self._provenance.bank_version[key] = bank_version

    def _emit_placeholder_warning(self) -> None:
        """Emit the configured placeholder warning kind on the OutcomeBuilder."""
        if self._outcome_builder is None:
            logger.debug(
                "  |-- %s: placeholder task picked but no outcome_builder available; warning silently dropped",
                type(self).__name__,
            )
            return
        kind = type(self).placeholder_warning_kind
        bank_id = getattr(self._bank, "bank_id", "<unknown_bank>")
        bank_version = getattr(self._bank, "bank_version", "<unknown_version>")
        task_id = getattr(self._task, "id", "<unknown_task>")
        self._outcome_builder.add_warning(
            kind=kind,
            detail=(
                f"task {task_id} from {bank_id} v{bank_version} is "
                "placeholder; downstream scorecard should refuse to "
                "claim production readiness for this trajectory's locale"
            ),
        )


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------
#
# IMPORTANT — MRO convention:
#
# When combining a mixin with a probe class, ALWAYS list the mixin
# first so its overrides win the method-resolution order. Python
# resolves methods left-to-right; ``BaseProbe`` provides defaults for
# every optional hook the mixins override, so a class declared as
#
#     class MyProbe(BankBackedProbe, BankVerbatimMixin):  # WRONG
#
# would pick ``BaseProbe.get_verbatim_first_user_turn`` (which returns
# None) over the mixin's override. Correct ordering:
#
#     class MyProbe(BankVerbatimMixin, BankBackedProbe):  # RIGHT
#
# Multiple mixins compose left-to-right too:
#
#     class MyProbe(BankReframingMixin, AgenticMixin, BankBackedProbe):
#
# (BankReframingMixin's hooks win over AgenticMixin's, AgenticMixin's
# win over BankBackedProbe / BaseProbe defaults.)
#
# Cross-cutting test
# (``conversation-plugin/tests/test_probes_substrate.py``) pins this
# behaviour for every shipped mixin to catch drift.


class BankVerbatimMixin:
    """For probes whose turn-1 user message is verbatim from the bank.

    Wires ``get_verbatim_first_user_turn`` to read from the derived
    task. Subclasses store the verbatim text on ``self._task`` via a
    field whose name is configured by the subclass:

        class MyProbe(BankBackedProbe, BankVerbatimMixin):
            verbatim_field = "user_question"  # or "rendering" / "request_text" / ...

    By default reads ``self._task.verbatim_text``. ``on_followup_failure``
    is set to ``"skip"`` (the inclusion convention).
    """

    verbatim_field: str = "verbatim_text"

    async def get_verbatim_first_user_turn(self, state: Any) -> str | None:
        task = getattr(self, "_task", None)
        if task is None:
            return None
        text = getattr(task, type(self).verbatim_field, None)
        if not isinstance(text, str) or not text:
            return None
        return text

    def on_followup_failure(self, state: Any, error_kind: str) -> str:
        return "skip"


class CustomTurn1InstructionMixin:
    """For probes that GENERATE turn-1 with a non-default user instruction.

    Subclasses provide ``turn1_instruction_template`` (a ``str.format``
    template referring to ``self._task`` fields) or override
    ``get_user_query_instruction``. The generated turn-1 still flows
    through the user-judge gate.
    """

    turn1_instruction_template: str = ""

    def get_user_query_instruction(self, turn_idx: int) -> str | None:
        if turn_idx != 0:
            return None
        template = type(self).turn1_instruction_template
        if not template:
            return None
        task = getattr(self, "_task", None)
        if task is None:
            return template
        try:
            # Make task fields available as positional kwargs, also
            # expose the task object itself as ``task`` for templates
            # that prefer dotted-path access.
            return template.format(task=task, **getattr(task, "__dict__", {}))
        except (KeyError, AttributeError) as e:
            logger.warning(
                "  |-- %s.turn1_instruction_template missing field %s; falling back to raw template",
                type(self).__name__,
                e,
            )
            return template

    def on_followup_failure(self, state: Any, error_kind: str) -> str:
        return "skip"


class BankReframingMixin(BankVerbatimMixin):
    """For probes whose follow-ups are bank-driven reframings.

    Subclasses provide ``self._task.reframing_at(turn_idx) -> str``
    (the bank-prescribed user reply for turn ``turn_idx``).
    ``on_followup_failure`` is set to ``"abort"`` — exhausted retries
    on a safety probe are an unanswered pressure event, which is
    meaningfully different from a generic chat probe's "user gave up".

    Used by safety_chat_pressure.
    """

    async def format_followup_user_instructions(self, turn_idx: int, state: Any) -> list[str]:
        task = getattr(self, "_task", None)
        if task is None:
            return []
        reframing_at = getattr(task, "reframing_at", None)
        if not callable(reframing_at):
            return []
        try:
            text = reframing_at(turn_idx)
        except Exception as e:
            logger.debug(
                "  |-- %s: reframing_at(%d) raised %s; no extra instructions",
                type(self).__name__,
                turn_idx,
                e,
            )
            return []
        return [text] if isinstance(text, str) and text else []

    def on_followup_failure(self, state: Any, error_kind: str) -> str:
        return "abort"


class AgenticMixin(BankVerbatimMixin):
    """For probes with a single user turn + multi-step assistant tool loop.

    Sets:

    - ``should_continue_after_turn`` — False when the assistant's last
      message has no tool calls (the protocol is single-decision).
    - ``should_inline_judge_user_turn`` — False for turn 0 (the
      verbatim action request is auditable as-is; gate burns judge
      calls for no signal on agentic probes).

    Subclasses still implement ``after_assistant_turn`` themselves
    to drive the inner tool-interception loop with mock responses
    from the action bank. Used by safety_agentic.
    """

    # The inner tool-interception loop in ``after_assistant_turn`` has
    # side effects that cannot be rolled back — assistant resampling
    # is off (see the same attribute on ``ToolCallingMixin``).
    supports_assistant_resampling: bool = False

    def should_continue_after_turn(self, state: Any) -> bool:
        if not state.messages:
            return False
        last = state.messages[-1]
        if last.get("role") != "assistant":
            return False
        tool_calls = last.get("tool_calls")
        return bool(tool_calls)

    def should_inline_judge_user_turn(self, turn_idx: int, state: Any) -> bool:
        return turn_idx > 0


class ToolCallingMixin:
    """For probes that test native tool-calling capability.

    Sets ``allow_early_stop_at_turn`` to require at least one tool
    call before early-stop is considered. Today this is the
    tool_calling probe specifically; the mixin makes the convention
    extensible (a future ``tool_calling_chained`` probe could
    inherit the same gating).
    """

    # Tool execution inside ``after_assistant_turn`` has side effects
    # (simulated API calls, multi-message appends) that cannot be
    # safely rolled back and replayed — assistant resampling is off.
    supports_assistant_resampling: bool = False

    def allow_early_stop_at_turn(self, turn_idx: int, state: Any) -> bool:
        tools_used = len(state.metadata.get("tools_called", []))
        return tools_used > 0


class ToolExecutionMixin:
    """Shared inner assistant<->tool loop for tool-executing probes.

    Provides ``after_assistant_turn`` that plumbs the assistant tool-call
    message, per-call execution, ``role:"tool"`` result messages, and the
    synthesis re-call — the scaffold ``tool_calling`` and
    ``financial_services`` would otherwise each hand-roll. A probe supplies
    only the per-call behaviour via ``execute_tool_call``.

    Two loop shapes, selected by ``tool_loop_mode``:

    - ``"single"`` (default): execute the assistant's one round of tool calls,
      then a single synthesis pass WITHOUT tools. This is the ``tool_calling``
      shape — single-shot capability, no correction/continuation loop.
    - ``"multi"``: re-offer the tools and re-call the assistant after each
      round until it stops calling tools or the per-turn cap
      (``tool_max_calls_per_turn``) is reached, then force a no-tools
      synthesis. This is the ``financial_services`` agentic shape (search →
      read → act → confirm across several assistant<->tool round-trips).

    Hooks a probe implements:

    - ``execute_tool_call(name, args, tc, state, models, *, turn_idx,
      call_idx) -> str`` (REQUIRED): run one tool call and return its result
      payload (a string appended as the ``role:"tool"`` message). Owns any
      probe-specific verification / gating / state mutation / traces /
      ``tools_called`` bookkeeping. ``tc`` is the raw tool-call dict for
      probes that need it (e.g. schema verification); ``name``/``args`` are
      the parsed convenience form.
    - ``parse_tool_call(tc) -> (name, args)``: OpenAI function-call shape by
      default; override for a different wire format.
    - ``on_tool_round_complete(state, n_calls)``: optional per-round hook
      (e.g. ``tool_calling`` records ``tool_call_count_per_turn``).

    Requires the host class to provide ``get_tools_for_assistant()`` and a
    ``self._locale`` attribute (both present on ``BaseProbe`` subclasses).
    """

    # The inner loop appends multiple messages and executes (mock) tool
    # calls whose effects cannot be rolled back and replayed — assistant
    # resampling is off (see the same attribute on ``ToolCallingMixin``
    # / ``AgenticMixin``). Covers every probe riding this mixin,
    # including ``financial_services``.
    supports_assistant_resampling: bool = False

    tool_loop_mode: str = "single"
    tool_max_calls_per_turn: int = 8

    async def execute_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        tc: Any,
        state: Any,
        models: dict,
        *,
        turn_idx: int,
        call_idx: int,
    ) -> str:  # pragma: no cover — abstract
        raise NotImplementedError(f"{type(self).__name__} must implement execute_tool_call()")

    @staticmethod
    def parse_tool_call(tc: Any) -> tuple[str, dict[str, Any]]:
        import json as _json

        if not isinstance(tc, dict):
            return ("<unknown>", {})
        fn = tc.get("function")
        if not isinstance(fn, dict):
            return (str(tc.get("name") or "<unknown>"), {})
        name = str(fn.get("name") or "<unknown>")
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            return (name, raw)
        if isinstance(raw, str):
            try:
                parsed = _json.loads(raw)
                return (name, parsed if isinstance(parsed, dict) else {})
            except (ValueError, TypeError):
                return (name, {})
        return (name, {})

    def on_tool_round_complete(self, state: Any, n_calls: int) -> None:
        return None

    async def after_assistant_turn(
        self,
        models: dict,
        state: Any,
        assistant_response: dict[str, Any],
        cfg: Any,
    ) -> str:
        resp = assistant_response
        calls_made = 0
        while True:
            content = resp.get("content", "") if isinstance(resp, dict) else ""
            tool_calls = resp.get("tool_calls") if isinstance(resp, dict) else None
            # ``resp`` is whichever call produced this message -- the initial
            # assistant turn on the first pass, a ``_recall_assistant`` result
            # on later ones -- so its trace lands on the right message.
            state.messages.append(
                assistant_message(
                    resp,
                    content,
                    tool_calls=tool_calls,
                    store_reasoning=getattr(cfg, "store_reasoning", True),
                )
            )
            if not tool_calls:
                self.on_tool_round_complete(state, 0)
                return content or ""

            trace_turn_idx = sum(1 for m in state.messages if m.get("role") == "assistant")
            n_this_round = 0
            capped = False
            for call_idx, tc in enumerate(tool_calls):
                if calls_made >= self.tool_max_calls_per_turn:
                    capped = True
                    break
                name, args = self.parse_tool_call(tc)
                tc_id = (tc.get("id") if isinstance(tc, dict) else None) or f"call_{trace_turn_idx}_{call_idx}"
                payload = await self.execute_tool_call(
                    name,
                    args,
                    tc,
                    state,
                    models,
                    turn_idx=trace_turn_idx,
                    call_idx=call_idx,
                )
                state.messages.append(
                    {
                        "role": "tool",
                        "content": payload,
                        "tool_call_id": tc_id,
                    }
                )
                self.on_tool_call_executed(
                    tool_call_id=tc_id,
                    tool_name=name,
                    arguments=args,
                    payload=payload,
                    turn_idx=trace_turn_idx,
                    call_idx=call_idx,
                )
                calls_made += 1
                n_this_round += 1
            self.on_tool_round_complete(state, n_this_round)

            if self.tool_loop_mode == "single" or capped:
                return await self._synthesize_reply(models, state, cfg)
            resp = await self._recall_assistant(models, state, cfg, with_tools=True)

    # ── internal plumbing ───────────────────────────────────────────

    async def _recall_assistant(
        self,
        models: dict,
        state: Any,
        cfg: Any,
        *,
        with_tools: bool,
    ) -> dict[str, Any]:
        from usersim.engine.core.llm import (
            NON_ASCII_TOKEN_SCALE,
            acall_llm,
            scaled_max_tokens,
        )
        from usersim.engine.core.simulation import _is_non_ascii_locale

        kwargs: dict[str, Any] = {}
        tools = self.get_tools_for_assistant() if with_tools else None
        if tools:
            kwargs["tools"] = tools
        if _is_non_ascii_locale(getattr(self, "_locale", "en_US")):
            kwargs.update(scaled_max_tokens(models["assistant_model"], NON_ASCII_TOKEN_SCALE))
        resp = await acall_llm(
            models,
            "assistant_model",
            self._synthesis_messages(state, cfg),
            **kwargs,
        )
        return resp if isinstance(resp, dict) else {"content": ""}

    async def _synthesize_reply(self, models: dict, state: Any, cfg: Any) -> str:
        resp = await self._recall_assistant(models, state, cfg, with_tools=False)
        content = resp.get("content", "") if isinstance(resp, dict) else ""
        # The synthesis call has its own reasoning, distinct from the call
        # that requested the tools.
        state.messages.append(
            assistant_message(
                resp,
                content,
                tool_calls=None,
                store_reasoning=getattr(cfg, "store_reasoning", True),
            )
        )
        return content or ""

    def _synthesis_messages(self, state: Any, cfg: Any) -> list[dict[str, Any]]:
        from usersim.engine.core.context import prepare_assistant_history

        return prepare_assistant_history(state, self, cfg)


# ---------------------------------------------------------------------------
# LocalePromptPack
# ---------------------------------------------------------------------------


class LocalePromptPackError(ImportError):
    """Raised at construction time when prompts are missing a shipped locale."""


class LocalePromptPack:
    """Per-locale prompt registry with all-locales-or-fail validation.

    Replaces the per-probe ``_SYSTEM_PROMPTS: dict + PromptsUnavailableError``
    pattern. The validation fires at module-import time (when the probe
    module constructs the LocalePromptPack), so a half-shipped locale
    set never escapes the import graph.

    Example::

        from usersim.engine.core.probes import LocalePromptPack

        SOV_AI_FACTS_USER_PROMPTS = LocalePromptPack(
            label="sov_ai_facts.user_system",
            prompts={
                "en_US": "You are a US user asking about ...",
                "pt_BR": "Você é um usuário brasileiro ...",
                "ja_JP": "あなたは日本のユーザーです ...",
                ...
            },
            shipped_locales=("en_US", "pt_BR", "ja_JP", "fr_FR",
                             "hi_Deva_IN", "en_IN"),
        )

        # Later, at simulate-time:
        prompt = SOV_AI_FACTS_USER_PROMPTS.get(self._locale, persona=...)
    """

    def __init__(
        self,
        *,
        label: str,
        prompts: dict[str, str],
        shipped_locales: Sequence[str],
    ) -> None:
        self.label = label
        missing = sorted(set(shipped_locales) - set(prompts))
        if missing:
            raise LocalePromptPackError(
                f"LocalePromptPack({label!r}) missing prompts for: "
                f"{missing}. shipped_locales={list(shipped_locales)}; "
                f"got prompts for {sorted(prompts)}"
            )
        extra = sorted(set(prompts) - set(shipped_locales))
        if extra:
            logger.debug(
                "  |-- LocalePromptPack(%r): prompts include locales not in shipped_locales: %s — kept (unused)",
                label,
                extra,
            )
        self._prompts: dict[str, str] = dict(prompts)
        self._shipped: tuple[str, ...] = tuple(shipped_locales)

    def get(self, locale: str, **format_kwargs: Any) -> str:
        """Return the (optionally format-substituted) prompt for ``locale``."""
        if locale not in self._prompts:
            raise LocalePromptPackError(
                f"LocalePromptPack({self.label!r}): no prompt for locale={locale!r}. Available: {sorted(self._prompts)}"
            )
        template = self._prompts[locale]
        if not format_kwargs:
            return template
        try:
            return template.format(**format_kwargs)
        except KeyError as e:
            raise LocalePromptPackError(
                f"LocalePromptPack({self.label!r}, locale={locale!r}): "
                f"template requires placeholder {e}; got kwargs="
                f"{sorted(format_kwargs)}"
            ) from e

    def available_locales(self) -> tuple[str, ...]:
        return tuple(sorted(self._prompts))

    def shipped_locales(self) -> tuple[str, ...]:
        return self._shipped


# ---------------------------------------------------------------------------
# Registry + decorator
# ---------------------------------------------------------------------------


_PROBE_REGISTRY: dict[str, type[BaseProbe]] = {}


class ProbeRegistrationError(ValueError):
    """Raised when ``@register_probe`` cannot register a class."""


def register_probe(
    *,
    family: str,
    prompt_version: str,
    variants: Sequence[str],
) -> Callable[[type[BaseProbe]], type[BaseProbe]]:
    """Single-annotation registration of a ``BaseProbe`` subclass.

    Sets ``PROBE_FAMILY`` / ``PROMPT_VERSION`` / ``PROBE_VARIANTS``
    on the class's module (the constants the dispatcher reads via
    ``getattr(module, "PROBE_FAMILY", ...)``) AND adds the class
    to ``_PROBE_REGISTRY`` keyed by ``cls.label``.

    Replaces the 6-files-to-touch friction of adding a new probe
    today: register in the central registry, define
    PROBE_FAMILY/PROMPT_VERSION/PROBE_VARIANTS module
    constants, expose a ``simulate_*`` function, etc. After this
    decorator, the class itself IS the registration.

    Example::

        @register_probe(
            family="general",
            prompt_version="v1.0",
            variants=("default",),
        )
        class OpenEndedProbe(BaseProbe):
            label = "general_open_ended"
            ...
    """

    def decorator(cls: type[BaseProbe]) -> type[BaseProbe]:
        if not isinstance(cls, type) or not issubclass(cls, BaseProbe):
            raise ProbeRegistrationError(f"@register_probe: {cls!r} is not a subclass of BaseProbe")
        label = getattr(cls, "label", "<unset>")
        if not isinstance(label, str) or label == "<unset>":
            raise ProbeRegistrationError(
                f"@register_probe: {cls.__name__} must set a class-level `label` string before decoration"
            )

        module = sys.modules.get(cls.__module__)
        if module is not None:
            module.PROBE_FAMILY = family
            module.PROMPT_VERSION = prompt_version
            module.PROBE_VARIANTS = tuple(variants)

        if label in _PROBE_REGISTRY and _PROBE_REGISTRY[label] is not cls:
            logger.debug(
                "  |-- @register_probe: replacing existing probe %r (was %s, now %s)",
                label,
                _PROBE_REGISTRY[label].__qualname__,
                cls.__qualname__,
            )
        _PROBE_REGISTRY[label] = cls
        return cls

    return decorator


#: Modules defining the shipped probes; importing one runs its
#: ``@register_probe``. Listed by path because every probe module imports
#: this one, so this module cannot import them at load time.
BUILTIN_PROBE_MODULES: tuple[str, ...] = (
    "usersim.engine.probes.financial_services.generator",
    "usersim.engine.probes.general_educational.generator",
    "usersim.engine.probes.general_open_ended.generator",
    "usersim.engine.probes.health_disclosure.decision_support",
    "usersim.engine.probes.health_disclosure.general",
    "usersim.engine.probes.health_disclosure.therapy",
    "usersim.engine.probes.health_disclosure.triage",
    "usersim.engine.probes.identity_disclosure.generator",
    "usersim.engine.probes.safety_agentic.generator",
    "usersim.engine.probes.safety_chat_pressure.generator",
    "usersim.engine.probes.sov_ai_dynamic.generator",
    "usersim.engine.probes.sov_ai_facts.generator",
    "usersim.engine.probes.sov_ai_multilingual_parity.generator",
    "usersim.engine.probes.tool_calling.generator",
)


def load_builtin_probes() -> None:
    """Import every shipped probe module so its probes are registered. Idempotent."""
    for module in BUILTIN_PROBE_MODULES:
        importlib.import_module(module)


def load_extension_probes() -> None:
    """Import probe packages advertised under ``usersim.probes``.

    Discovery deliberately works by *importing* the advertised object rather
    than registering it here: ``@register_probe`` already runs at class
    definition, so an out-of-tree probe travels the same code path as a
    built-in and inherits its validation. An entry point may therefore point
    at either the probe class or its module.
    """
    from usersim.engine.core._extensions import PROBES, load_extensions

    load_extensions(PROBES)


def known_probes() -> tuple[str, ...]:
    """All registered probe labels, sorted."""
    load_builtin_probes()
    load_extension_probes()
    return tuple(sorted(_PROBE_REGISTRY))


def resolve_probe(label: str) -> type[BaseProbe]:
    """Look up a probe class by label. Raises ``KeyError`` if unknown."""
    if label not in _PROBE_REGISTRY:
        load_builtin_probes()
    if label not in _PROBE_REGISTRY:
        load_extension_probes()
    if label not in _PROBE_REGISTRY:
        available = ", ".join(known_probes()) or "(none)"
        raise KeyError(f"Unknown probe: {label!r}. Registered: {available}")
    return _PROBE_REGISTRY[label]


def clear_registry() -> None:
    """Drop all registered probes — primarily for test isolation."""
    _PROBE_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Default BaseProbe.run_dispatch
# ---------------------------------------------------------------------------
#
# Defined here (not on BaseProbe directly) because it imports from
# ``core.simulation`` which would cause a circular import at module
# load. Probes that genuinely need a different simulate shape (today:
# only ``safety_agentic`` with its single-user-turn + multi-assistant-
# turn tool-interception loop) override ``run_dispatch`` themselves.


async def _baseprobe_run_dispatch(
    self: BaseProbe,
    *,
    models: dict[str, Any],
    data: dict[str, Any],
    cfg: Any,
    state: Any = None,
    seed_state: bool = True,
) -> dict:
    """Default dispatch: drive the shared ``ConversationLoop``.

    Subclasses inherit this. Override only when the probe shape is
    genuinely incompatible with the unified loop's back-and-forth
    assumption (today: only ``safety_agentic``). Imported lazily to
    avoid a ``probes.py`` ↔ ``simulation.py`` circular import.
    """
    from usersim.engine.core.simulation import ConversationLoop

    loop = ConversationLoop()
    return await loop.run(
        models=models,
        data=data,
        cfg=cfg,
        probe=self,
        user_interaction_style=getattr(self, "_interaction_style", "neutral"),
        patience=getattr(self, "_patience", 0.5),
        locale=getattr(self, "_locale", "en_US"),
        provenance=getattr(self, "_provenance", None),
        state=state,
        seed_state=seed_state,
    )


BaseProbe.run_dispatch = _baseprobe_run_dispatch  # type: ignore[attr-defined]
