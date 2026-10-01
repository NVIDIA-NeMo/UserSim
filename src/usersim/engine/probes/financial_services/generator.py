# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services probe — hybrid-agentic, knowledge-grounded.

Rides the shared :class:`ConversationLoop` (like ``tool_calling`` /
``sov_ai_dynamic``) rather than owning a custom ``run_dispatch``. The loop
provides the multi-turn persona user-LLM, the user-judge gate, termination
(capitulation / user-satisfied / max-turns), traces, and resource/outcome
accounting. This probe contributes only the pieces that are finance-specific:

- ``seed_state_metadata`` seeds the gold/task side channels + the mutable
  per-conversation account state and discovered-tool set (all JSON-safe).
- ``get_verbatim_first_user_turn`` injects the task's param-locked opening as
  turn 1 (the ``dynamic`` tier will later generate turn 1 instead).
- ``get_user_system_prompt`` is the persona user-simulator prompt that unlocks
  multi-turn follow-ups.
- ``after_assistant_turn`` runs the inner scoped tool loop: ``kb_search``
  retrieval + the discoverable-tool gate + deterministic account-state
  mutation, re-calling the assistant until it produces a user-facing reply.
- ``allow_early_stop_at_turn`` is tier-aware (verifiable tasks must complete
  at least one domain tool call before "user satisfied" can end the run).

Gold metadata (gold docs / gold tool sequence / expected state deltas) is
derived deterministically at sim time by ``core/finance_tasks.py`` and
persisted on the trajectory as side channels + top-level columns, so the
deterministic scorer is self-contained (no re-derivation).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from usersim.engine.core.behavioral import (
    format_behavioral_profile_for_prompt,
    format_disclosure_instructions,
    format_interaction_style_instructions,
)
from usersim.engine.core.finance_bank import (
    DEFAULT_DOMAIN_ACCOUNT_LABELS,
    DEFAULT_IDENTITY_REQUIRED_FIELDS,
    FinanceBank,
    load_finance_bank_for_locale,
    reset_finance_bank_cache,
)
from usersim.engine.core.finance_tasks import FinanceInstance
from usersim.engine.core.identity import persona_uuid as compute_persona_uuid
from usersim.engine.core.locale import (
    expected_language_display,
    persona_dataset_locale,
)
from usersim.engine.core.outcomes import (
    FailureAttribution,
    FailureClass,
    OutcomeBuilder,
    OutcomeStatus,
    Provenance,
    SimulationTrace,
    TraceKind,
    WarningKind,
)
from usersim.engine.core.persona import format_persona_for_prompt
from usersim.engine.core.probes import (
    BankBackedProbe,
    ToolExecutionMixin,
    register_probe,
)
from usersim.engine.core.simulation import (
    ConversationState,
    make_failed,
)
from usersim.engine.core.simulation import (
    language_instruction as _language_instruction,
)
from usersim.engine.probes.financial_services import retrieval as _retrieval
from usersim.engine.probes.financial_services import tools as _tools
from usersim.engine.probes.financial_services.prompts import (
    DYNAMIC_FOLLOWUP_INSTRUCTION,
    DYNAMIC_USER_SYSTEM_PROMPT,
    USER_SYSTEM_PROMPT,
    financial_literacy_instructions,
)
from usersim.engine.probes.financial_services.task_derivation import (
    derive_financial_literacy,
    derive_instance,
)

logger = logging.getLogger("usersim.engine")

PROBE_FAMILY: str = "financial_services"
PROBE_VARIANTS: list[str] = ["default"]
PROMPT_VERSION: str = "v1.1"

_KB_BODY_RETENTION_USER_TURNS = 8

#: Task param slots whose value names a PERSON the customer brings into the request --
#: a relative, a professional, an ex. Tasks name the RELATIONSHIP ("my spouse") and the
#: customer supplies the identity, so these are what ``_counterparty_context`` keys off.
#:
#: Excludes ``payee``, whose authored values mix companies ("Pacific Gas & Electric",
#: "Northstar Telecom") with people ("my landlord", "my son's tuition teacher") -- a
#: company payee needs no personal details. Also excludes ``biller`` and ``operator``,
#: which are services. A test binds this set to the authored task YAML so a new locale
#: cannot introduce a seventh slot unnoticed (the first draft of this work missed
#: ``contact`` and would have left named counterparties behind).
PERSON_PARAM_SLOTS: tuple[str, ...] = (
    "person",
    "beneficiary",
    "contact",
    "nominee",
    "relation",
    "ex",
)

#: Values in those slots that are not an individual: a trust, or a group. There is no
#: single name or date of birth to give for "my two children equally".
_NON_INDIVIDUAL_MARKERS: tuple[str, ...] = ("trust", "equally", "children")

#: Granting another person ACCESS to an account is the one counterparty action with a
#: hard age floor in real policy, so a minor there can be CORRECTLY refused -- which
#: would read as a task failure and make tool selection uninterpretable. Every other
#: counterparty action takes a minor perfectly normally: nominees and beneficiaries are
#: routinely children, and en_IN's claim task admits a relative to hospital who may
#: well be one. So the adult constraint is scoped to this tool rather than applied
#: everywhere, which would cost realism for no gain.
_ADULT_REQUIRED_TOOLS: frozenset[str] = frozenset({"manage_authorized_users"})

# Canonical finance gold-metadata side-channel columns emitted on EVERY
# trajectory by ``build_result_extras`` (both tiers). This is the single source
# of truth for the training-data flywheel: ``selection/schemas.py``'s ``finance``
# group projects exactly these (a schema-parity test binds the two across the
# package boundary, and an emit-parity test binds this list to
# ``build_result_extras``). Keep the three in sync via those tests.
#   verifiable -> verifier-labeled tool trajectories (gold_tool_sequence +
#     expected_state_deltas; PRM from per-turn traces)
#   dynamic    -> grounded SFT + preference pairs (labeled by the dynamic.* judge)
# RAFT-style extraction uses gold_document_ids + institution_id (same-institution
# distractors); notebooks/03 owns the actual record shaping.
FINANCE_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "finance_task_id",
    "task_tier",
    "task_type",
    "task_contract_version",
    "institution_id",
    "institution_type",
    "domain",
    "dynamic_category_id",
    "dynamic_subtopic_hint",
    "taxonomy_version",
    "gold_document_ids",
    "gold_tool_sequence",
    "expected_state_deltas",
    "retrieved_document_ids",
    "attempted_tool_names",
    "num_tool_calls",
    "kb_search_queries",
    "num_kb_searches",
)

# Module-level alias for test patchability (see AUTHORING_A_PROBE.md).
_load_bank = load_finance_bank_for_locale
_reset_bank_cache = reset_finance_bank_cache


def _bank_loader(locale: str = "en_US"):
    return _load_bank(locale)


def should_succeed(state: ConversationState) -> bool:
    """Sim-side guardrail: did we actually instantiate a finance task?"""
    return bool(state.metadata.get("finance_task_id"))


class FinancialServicesProbeError(ValueError):
    """Raised when the probe cannot construct (no resolvable task template)."""


def _kb_search_payload(content: Any) -> dict[str, Any] | None:
    """Parse only the exact finance document-result shape, else return None."""
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    required = {"id", "title", "body", "tools"}
    if not all(
        isinstance(document, dict)
        and required <= set(document)
        and isinstance(document.get("id"), str)
        and bool(document["id"])
        for document in results
    ):
        return None
    return payload


def _tool_result_names(
    messages: list[dict[str, Any]],
) -> dict[int, str]:
    """Map tool-message positions to their originating assistant function."""
    result: dict[int, str] = {}
    pending: list[tuple[str | None, str]] = []
    for message_index, message in enumerate(messages):
        if message.get("role") == "assistant":
            pending = []
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                pending.append(
                    (
                        str(tool_call.get("id")) if tool_call.get("id") else None,
                        str(function.get("name") or ""),
                    )
                )
        elif message.get("role") == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            match_index = next(
                (index for index, (call_id, _name) in enumerate(pending) if call_id and call_id == tool_call_id),
                0 if pending else None,
            )
            if match_index is not None:
                _call_id, name = pending.pop(match_index)
                result[message_index] = name
        elif message.get("role") in {"user", "system"}:
            pending = []
    return result


def _project_finance_document_history(
    messages: list[dict[str, Any]],
    *,
    current_user_turn: int,
) -> list[dict[str, Any]]:
    """Keep a deduplicated, refreshable document working set for the assistant."""
    retrievals: list[tuple[int, int, dict[str, Any]]] = []
    tool_result_names = _tool_result_names(messages)
    retrieval_turn = 0
    for message_index, message in enumerate(messages):
        if message.get("role") == "user":
            retrieval_turn += 1
        if message.get("role") != "tool" or tool_result_names.get(message_index) != "kb_search":
            continue
        payload = _kb_search_payload(message.get("content"))
        if payload is not None:
            retrievals.append((message_index, retrieval_turn, payload))

    if not retrievals:
        return list(messages)

    newest_occurrence: dict[str, tuple[int, int]] = {}
    for message_index, _turn, payload in retrievals:
        for document_index, document in enumerate(payload["results"]):
            newest_occurrence[document["id"]] = (message_index, document_index)

    projected = list(messages)
    for message_index, document_turn, payload in retrievals:
        documents: list[dict[str, Any]] = []
        changed = False
        for document_index, document in enumerate(payload["results"]):
            keep_body = (
                newest_occurrence[document["id"]] == (message_index, document_index)
                and current_user_turn - document_turn < _KB_BODY_RETENTION_USER_TURNS
            )
            if keep_body:
                documents.append(document)
                continue
            changed = True
            documents.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "tools": document["tools"],
                }
            )
        if changed:
            projected[message_index] = {
                **messages[message_index],
                "content": json.dumps(
                    {**payload, "results": documents},
                    ensure_ascii=False,
                ),
            }
    return projected


@dataclass
class _PickedTask:
    """Substrate-shaped wrapper around the instantiated task."""

    id: str
    placeholder: bool
    instance: FinanceInstance
    bank_version: str


@register_probe(
    family=PROBE_FAMILY,
    prompt_version=PROMPT_VERSION,
    variants=tuple(PROBE_VARIANTS),
)
class FinancialServicesProbe(ToolExecutionMixin, BankBackedProbe):
    label = "financial_services"
    bank_loader = staticmethod(_bank_loader)
    placeholder_warning_kind = WarningKind.USED_PLACEHOLDER_FINANCE
    bank_version_key = None  # per-locale
    # Agentic shape: re-offer tools until the assistant stops (search -> read
    # -> act -> confirm), capped per turn by ToolExecutionMixin.
    tool_loop_mode = "multi"
    tool_max_calls_per_turn = 8

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self._task is None:
            raise FinancialServicesProbeError(
                f"no resolvable financial_services template in "
                f"{getattr(self._bank, 'bank_id', '?')} for locale={self._locale} "
                f"(asset_locale={self._asset_locale})"
            )
        self._instance: FinanceInstance = self._task.instance
        self._institution = self._bank.institution(self._instance.institution_id)
        if self._institution is None:
            raise FinancialServicesProbeError(
                f"instance references unknown institution {self._instance.institution_id!r}"
            )
        self._offered_tools = _tools.build_offered_tools(
            self._institution,
            identity_required_fields=self._bank.identity_required_fields(),
        )
        # Read tools are answered with a definitive account view rather than a
        # bare ack, so the assistant resolves and proceeds instead of re-querying.
        self._read_only_tools: frozenset[str] = frozenset(
            t.name for t in self._institution.tools if t.side_effect_class == "read_only"
        )
        self._retrieval_mode = getattr(self._cfg, "finance_retrieval_mode", "hybrid")
        self._embedding_alias = getattr(
            self._cfg,
            "finance_embedding_model_alias",
            "embedding_model",
        )
        # Tripped the first time dense mode has to fall back to lexical (no query
        # vector) so a misconfigured run -- dense requested but no embedding model
        # -- is loud instead of silently degrading to word overlap.
        self._dense_fallback_warned = False
        # Tripped once per row so a translated-query warning is emitted once
        # rather than per kb_search call.
        self._query_translation_warned = False
        self._user_system_prompt = self._build_user_system_prompt()

    # ── BankBackedProbe API ─────────────────────────────────────────

    def _persona_locale(self) -> str:
        """Locale whose PERSONA-ATTRIBUTE vocabulary this run's personas use.

        Three locales are in play and they are not interchangeable:

        - ``self._locale`` — the CONVERSATION locale (language / script /
          partitioning). Drives the language instruction.
        - ``self._asset_locale`` — the locale whose BANK / taxonomy / renderings
          loaded (an India language-variant falls back to its base).
        - this — the locale whose persona parquet the personas were sampled
          from, which is what ``_education_ordinal`` is keyed on. A variant like
          ``ta_Taml_IN`` samples ``en_IN`` personas, so deriving literacy against
          the raw conversation locale would miss the exact education map and
          silently fall through to the keyword heuristic. Hindi resolves to
          itself (it has its own native panels, with romanized attribute values
          on ``hi_Latn_IN``), so this is a no-op there.

        Computed on demand rather than cached in ``__init__`` because
        ``derive_task`` runs during ``super().__init__()``.
        """
        return persona_dataset_locale(self._locale)

    def derive_task(
        self,
        persona: dict[str, Any],
        bank: FinanceBank,
        *,
        cfg: Any,
    ) -> _PickedTask | None:
        puuid = self._data.get("persona_uuid") or compute_persona_uuid(persona)
        tier_mix = float(getattr(cfg, "finance_tier_mix", 0.0) or 0.0)
        forced_tier = getattr(cfg, "finance_tier", None)
        taxonomy = None
        if tier_mix > 0.0 or forced_tier == "dynamic":
            taxonomy = self._load_dynamic_taxonomy()
        instance = derive_instance(
            persona,
            bank,
            self._persona_locale(),
            persona_uuid=puuid,
            seed=getattr(cfg, "random_seed", None),
            tier_mix=tier_mix,
            forced_tier=forced_tier,
            taxonomy=taxonomy,
        )
        if instance is None:
            return None
        return _PickedTask(
            id=instance.template_id,
            placeholder=bool(instance.placeholder),
            instance=instance,
            bank_version=bank.bank_version,
        )

    def _load_dynamic_taxonomy(self) -> Any:
        """Load the locale's dynamic-tier taxonomy (best-effort, non-fatal).

        A missing taxonomy is not an error — ``derive_instance`` falls back to
        the verifiable tier — so a locale can ship verifiable-only until its
        ``dynamic.yaml`` is authored.
        """
        try:
            from usersim.engine.core.probing_taxonomy import (
                load_probing_taxonomy_for,
            )

            return load_probing_taxonomy_for(
                "financial_services",
                self._asset_locale,
                filename="dynamic.yaml",
                env_prefix="USERSIM_FINANCIAL_SERVICES_DYNAMIC_TAXONOMY",
            )
        except Exception as e:  # noqa: BLE001 — soft-fail to verifiable-only
            logger.warning(
                "  |-- financial_services: no dynamic taxonomy for locale=%s "
                "(asset_locale=%s) (%s: %s); dynamic tier unavailable",
                self._locale,
                self._asset_locale,
                type(e).__name__,
                e,
            )
            return None

    # ── ProbeAdapter required hooks ─────────────────────────────────

    def get_user_system_prompt(self) -> str:
        return self._user_system_prompt

    def get_assistant_system_prompt(self) -> str:
        return ""  # pure-capability policy: the assistant gets no system prompt

    def get_tools_for_assistant(self) -> list | None:
        return self._offered_tools

    # ── Optional hooks ──────────────────────────────────────────────

    def transform_assistant_history(
        self,
        messages: list[dict[str, Any]],
        *,
        current_user_turn: int,
    ) -> list[dict[str, Any]]:
        """Bound retrieved bodies in assistant requests, not in the transcript."""
        return _project_finance_document_history(
            messages,
            current_user_turn=current_user_turn,
        )

    def seed_state_metadata(self, state: ConversationState) -> None:
        """Seed all finance side channels BEFORE turn 1 (path-agnostic).

        ``ConversationLoop.run`` calls this immediately after constructing
        state; both the verbatim-injection and generate-and-gate paths
        preserve these keys via ``setdefault``, so they survive onto every
        row (including failed ones). Side channels are kept JSON-serializable
        (``discovered_tools`` a list, ``account_state`` a dict) so the
        trajectory metadata dumps cleanly.
        """
        inst = self._instance
        gold = inst.gold
        md = state.metadata
        md["finance_task_id"] = inst.template_id
        md["task_tier"] = inst.tier
        md["task_type"] = inst.task_type
        md["institution_id"] = inst.institution_id
        md["institution_type"] = inst.institution_type
        md["domain"] = inst.domain
        md["task_contract_version"] = inst.task_contract_version
        md["bank_id"] = self._bank.bank_id
        md["bank_version"] = self._bank.bank_version
        md["gold_document_ids"] = list(gold.gold_document_ids) if gold else []
        md["gold_tool_sequence"] = list(gold.gold_tool_sequence) if gold else []
        md["expected_state_deltas"] = gold.expected_state_deltas if gold else {}
        md["retrieved_document_ids"] = []
        md["attempted_actions"] = []
        md["tools_called"] = []
        # kb_search is a FRAMEWORK primitive, so it is deliberately absent from
        # attempted_actions (which the verifier reads for the unauthorized-action
        # check -- a search is not a domain action). Recorded separately here, or
        # retrieval EFFORT is invisible in the export: without it a run where the
        # agent never searched and one where it searched ten times look identical.
        md["kb_search_queries"] = []
        md["identity_verified"] = False
        # Mutable in-memory account state + discovered-tool set (JSON-safe).
        md["account_state"] = dict(inst.account_state)
        # Resolve the ONE canonical account up front and pin it on the state, so
        # the read tools return exactly the account the user is grounded to (see
        # _build_user_system_prompt) -- no type/last4 contradiction that would
        # make the assistant refuse to act. Dynamic tasks get a synthesized
        # account here too (they carry none), so account-specific dynamic topics
        # don't dead-end on "can't locate your account".
        accounts = self._grounding_accounts(inst)
        md["account_state"]["accounts"] = accounts
        if accounts and not md["account_state"].get("account_id"):
            md["account_state"]["account_id"] = accounts[0]["account_id"]
        md["discovered_tools"] = [t.name for t in self._institution.permanent_tools()]
        # Dynamic-tier side channels (scorer grounding-reference + reporting).
        if inst.is_dynamic:
            md["dynamic_category_id"] = inst.dynamic_category_id
            md["dynamic_subtopic_hint"] = inst.dynamic_subtopic_hint
            md["taxonomy_id"] = inst.taxonomy_id
            md["taxonomy_version"] = inst.taxonomy_version

    async def get_verbatim_first_user_turn(self, state: ConversationState) -> str | None:
        """Tier-branched turn-1.

        ``verifiable`` ships a param-locked opening (returned verbatim to keep
        the gold well-posed). The ``dynamic`` tier has no scripted opening
        (``opening_user_message`` is empty) so we return ``None`` and the loop
        generates turn-1 from the persona + invitation + subtopic hint carried
        in ``get_user_system_prompt``.

        The opening is authored in the bank's language, so it is routed through
        ``_localize_verbatim``: a pure pass-through whenever the bank we loaded
        matches the conversation locale (every locale with its own bank,
        including native Hindi), and a machine translation only when we fell
        back to a base-locale bank for a language variant.
        """
        return await self._localize_verbatim(self._instance.opening_user_message) or None

    def allow_early_stop_at_turn(
        self,
        turn_idx: int,
        state: ConversationState,
    ) -> bool:
        """Tier-aware early-stop gate.

        A ``verifiable`` task is not "done" until the assistant has completed
        at least one domain tool call, so we block the generic user-satisfied
        early-stop until ``tools_called`` is non-empty (mirrors
        ``ToolCallingMixin``). Open-ended / advisory tasks have no required
        tool, so the standard early-stop applies.
        """
        if state.metadata.get("task_tier") == "verifiable":
            return len(state.metadata.get("tools_called", [])) > 0
        return True

    async def format_followup_user_instructions(
        self,
        turn_idx: int,
        state: ConversationState,
    ) -> list[str]:
        """Dynamic-tier depth nudge (collapse avoidance).

        Open-ended conversations otherwise tend to close after one generic
        answer; nudging the persona to press for the specifics that apply to
        their situation keeps the grounding/numeric-faithfulness signal alive.
        Verifiable tasks have their own scripted flow, so no nudge there.
        """
        if self._instance.is_dynamic:
            return [DYNAMIC_FOLLOWUP_INSTRUCTION]
        return []

    def should_succeed(self, state: ConversationState) -> bool:
        return should_succeed(state)

    def build_result_extras(self, state: ConversationState) -> dict:
        extras = super().build_result_extras(state)
        inst = self._instance
        gold = inst.gold
        attempted = state.metadata.get("attempted_actions") or []
        kb_queries = state.metadata.get("kb_search_queries") or []
        # For the dynamic tier the per-row discriminator is the taxonomy
        # category (mirrors sov_ai_dynamic); verifiable rows use domain::type.
        probe_variant = inst.dynamic_category_id if inst.is_dynamic else f"{inst.domain}::{inst.task_type}"
        extras.update(
            {
                "probe_variant": probe_variant,
                "finance_task_id": inst.template_id,
                "task_tier": inst.tier,
                "task_type": inst.task_type,
                "task_contract_version": inst.task_contract_version,
                "institution_id": inst.institution_id,
                "institution_type": inst.institution_type,
                "domain": inst.domain,
                "dynamic_category_id": inst.dynamic_category_id,
                "dynamic_subtopic_hint": inst.dynamic_subtopic_hint,
                "taxonomy_version": inst.taxonomy_version,
                "region": str(self._bank.region_meta.get("region", self._asset_locale)),
                "gold_document_ids": json.dumps(list(gold.gold_document_ids) if gold else [], ensure_ascii=False),
                "gold_tool_sequence": json.dumps(list(gold.gold_tool_sequence) if gold else [], ensure_ascii=False),
                "expected_state_deltas": json.dumps(gold.expected_state_deltas if gold else {}, ensure_ascii=False),
                "retrieved_document_ids": json.dumps(
                    state.metadata.get("retrieved_document_ids") or [],
                    ensure_ascii=False,
                ),
                "attempted_tool_names": json.dumps([a.get("tool_name") for a in attempted], ensure_ascii=False),
                "num_tool_calls": len(attempted),
                # Retrieval effort, separate from the domain-action list above.
                # The queries themselves are kept (not just the count) because the
                # failure modes are query-shaped: an empty query, the same query
                # repeated, or one lookup where the task needed several.
                "kb_search_queries": json.dumps(kb_queries, ensure_ascii=False),
                "num_kb_searches": len(kb_queries),
            }
        )
        return extras

    # ── Inner tool loop (ToolExecutionMixin hooks) ─────────────────

    @staticmethod
    def parse_tool_call(tc: Any) -> tuple:
        """Finance uses ``call_tool``-style wrapping, so parse via helper."""
        return _tools.extract_call(tc)

    def _corpus_language(self) -> str:
        """Language the loaded corpus is written in, or "" if it is already the
        conversation language.

        Non-empty only when this row borrowed someone else's bank — a stop-gap
        locale on the en_IN corpus. Computed on demand rather than cached in
        ``__init__`` for the same reason as ``_persona_locale``, and because a
        cached copy would go stale against ``_asset_locale``.
        """
        if self._asset_locale == self._locale:
            return ""
        return expected_language_display(self._asset_locale) or ""

    async def _search_query(
        self,
        query: str,
        state: ConversationState,
        models: dict,
        *,
        turn_idx: int,
        call_idx: int,
    ) -> str:
        """The string retrieval actually ranks on, translated when it has to be.

        A stop-gap locale borrows the en_IN corpus, so the assistant is talking
        to (say) a Tamil user about English documents and tends to search in
        Tamil. Lexical ranking tokenizes on ``[a-z0-9]+``: a native-script query
        produces zero tokens, which makes the lexical fallback return the
        corpus's first k documents irrespective of the query and reduces
        ``hybrid`` to pure dense — the mode this corpus measures at 4%
        precision@8 against hybrid's 88%. Translating the query into the
        corpus's language restores both halves of the ranking.

        No-op whenever the corpus is already in the conversation language, so
        en_US / en_IN runs are untouched and a graduated native bank turns this
        off by itself.
        """
        corpus_language = self._corpus_language()
        if not corpus_language or not query.strip():
            return query
        from usersim.engine.core.translation import translate_search_query

        translated = await translate_search_query(
            models,
            query,
            target_language=corpus_language,
        )
        if translated == query:
            return query
        state.outcome.add_trace(
            SimulationTrace(
                kind=TraceKind.KB_QUERY_TRANSLATION,
                turn_idx=turn_idx,
                call_idx=call_idx,
                rating="success",
                detail=(f"kb_search query translated into {corpus_language} for the {self._asset_locale} corpus"),
                extra={"query": query, "translated_query": translated},
            )
        )
        # The dynamic tier has no verbatim turn-1, so without this a dynamic row
        # on a stop-gap locale would involve machine translation and carry no
        # warning at all. Emitted once per row; the kind is a set member.
        if not self._query_translation_warned and self._outcome_builder is not None:
            self._query_translation_warned = True
            self._outcome_builder.add_warning(
                kind=WarningKind.USED_MACHINE_TRANSLATION,
                detail=(
                    f"kb_search queries machine-translated into "
                    f"{corpus_language} to search the "
                    f"{self._asset_locale} corpus for conversation "
                    f"locale={self._locale}; retrieval metrics are "
                    "preview-only until a native bank lands"
                ),
            )
        return translated

    async def execute_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        tc: Any,
        state: ConversationState,
        models: dict,
        *,
        turn_idx: int,
        call_idx: int,
    ) -> str:
        """Execute one tool call against the scoped bank + in-memory state.

        Returns the tool result payload (a JSON string). Reads/writes the
        JSON-safe side channels on ``state.metadata``
        (``discovered_tools`` / ``account_state`` / ``retrieved_document_ids``
        / ``attempted_actions`` / ``tools_called``). The mixin owns the
        message plumbing + assistant re-calls.
        """
        md = state.metadata
        gold = self._instance.gold

        if name == "kb_search":
            query = str(args.get("query", ""))
            # Record what the ASSISTANT asked for, not what we searched with:
            # the query is assistant behaviour, the translation is ours.
            md.setdefault("kb_search_queries", []).append(query)
            search_query = await self._search_query(
                query,
                state,
                models,
                turn_idx=turn_idx,
                call_idx=call_idx,
            )
            query_embedding = None
            if self._retrieval_mode in ("hybrid", "dense"):
                from usersim.engine.core.embeddings import aembed_query

                query_embedding = await aembed_query(
                    models,
                    self._embedding_alias,
                    search_query,
                )
                if query_embedding is None and not self._dense_fallback_warned:
                    self._dense_fallback_warned = True
                    logger.warning(
                        "financial_services: retrieval_mode=%r but query "
                        "embedding via alias %r returned None -- retrieval is "
                        "DEGRADING TO LEXICAL word-overlap for this run. Add an "
                        "'%s' entry to your --models config (the same embedding "
                        "model that produced embeddings.parquet) to enable the "
                        "semantic half of retrieval.",
                        self._retrieval_mode,
                        self._embedding_alias,
                        self._embedding_alias,
                    )
            # Ranking scores every document in the institution corpus, so it
            # runs off the event loop: left inline it would stall every other
            # conversation sharing the runtime for the whole pass.
            docs = await asyncio.to_thread(
                _retrieval.retrieve,
                self._institution,
                search_query,
                k=8,
                mode=self._retrieval_mode,
                gold_document_ids=(gold.gold_document_ids if gold else ()),
                query_embedding=query_embedding,
            )
            retrieved = md.setdefault("retrieved_document_ids", [])
            discovered = md.setdefault("discovered_tools", [])
            for d in docs:
                if d.id not in retrieved:
                    retrieved.append(d.id)
                for t in d.mentions_tools:  # discover documented tools
                    if t not in discovered:
                        discovered.append(t)
            payload = [{"id": d.id, "title": d.title, "body": d.body, "tools": list(d.mentions_tools)} for d in docs]
            return json.dumps({"results": payload}, ensure_ascii=False)

        if name == "verify_identity":
            verified, payload = _verify_identity(args, self._bank.identity_required_fields())
            md["identity_verified"] = verified
            return json.dumps(payload)

        # Everything else targets a domain tool (via call_tool or directly).
        target, target_args = _tools.resolve_invoked_tool(name, args)
        discovered = md.setdefault("discovered_tools", [])
        is_discovered = target in discovered
        md.setdefault("attempted_actions", []).append(
            {
                "tool_name": target,
                "tool_args": target_args,
                "turn_idx": turn_idx,
                "discovered": is_discovered,
            }
        )
        spec = self._institution.tool_by_name(target)
        state.outcome.add_trace(
            SimulationTrace(
                kind=TraceKind.TOOL_CALL_VERIFIER,
                turn_idx=turn_idx,
                call_idx=call_idx,
                rating="success" if (spec and is_discovered) else "failure",
                detail=(f"{target} invoked" if (spec and is_discovered) else f"{target} not discovered/known"),
                extra={"tool_name": target},
            )
        )
        if spec is None:
            return json.dumps(
                {
                    "error": f"Unknown tool {target!r}.",
                    "tool_name": target,
                }
            )
        if not is_discovered:
            return json.dumps(
                {
                    "error": (
                        f"Tool {target!r} is not available yet — retrieve its "
                        "documentation via kb_search before calling it."
                    ),
                    "tool_name": target,
                }
            )
        # Successful domain tool execution: mutate state + count it.
        md.setdefault("tools_called", []).append(target)
        result_payload = _apply_tool(
            target,
            target_args,
            md["account_state"],
            self._read_only_tools,
        )
        return json.dumps(result_payload, ensure_ascii=False)

    # ── Prompt construction ─────────────────────────────────────────

    def _build_user_system_prompt(self) -> str:
        prof = self._profile or {}
        if "tech_literacy" in prof and "error_proneness" in prof:
            behavioral = format_behavioral_profile_for_prompt(
                prof,
                probe_type="financial_services",
                language=self._language,
            )
        else:
            behavioral = ""
        persona_name = (
            " ".join(str(self._persona.get(k, "")).strip() for k in ("first_name", "last_name")).strip()
            or "the customer"
        )
        inst = self._instance
        literacy = derive_financial_literacy(self._persona, self._persona_locale())
        common = dict(
            persona_name=persona_name,
            # The brand as this locale's own corpus writes it, so the customer
            # names their institution the same way the retrieved documents do.
            institution_name=self._institution.brand_name(),
            persona_block=format_persona_for_prompt(self._persona),
            financial_literacy_instructions=financial_literacy_instructions(literacy),
            behavioral_instructions=behavioral,
            disclosure_instructions=format_disclosure_instructions(
                self._data.get("disclosure_style", "upfront"),
            ),
            interaction_style_instructions=format_interaction_style_instructions(
                self._interaction_style,
            ),
            language_instruction=_language_instruction(
                self._language,
                self._locale,
            ),
            identity_context=self._identity_context(persona_name),
            # Empty unless this task named someone other than the customer. Dynamic
            # instances carry no params, so it is naturally empty there.
            counterparty_context=self._counterparty_context(inst),
        )
        if inst.is_dynamic:
            # No scripted opening: the loop generates turn-1 in the persona's
            # voice from the taxonomy invitation + a subtopic hint. Still ground
            # the customer to their on-file account so account-specific dynamic
            # topics (beneficiaries, closing, etc.) don't dead-end on the agent
            # "not being able to locate" an account the user invented.
            return DYNAMIC_USER_SYSTEM_PROMPT.format(
                invitation=inst.dynamic_invitation,
                subtopic_hint=inst.dynamic_subtopic_hint,
                account_context=self._account_context(inst),
                **common,
            )
        return USER_SYSTEM_PROMPT.format(
            task_goal=inst.opening_user_message,
            account_context=self._account_context(inst),
            **common,
        )

    def _synth_dob(self) -> str:
        """A stable date of birth (MM/DD/YYYY) for the persona. Derived from the
        persona's age when present; the verifier only checks that a DOB is
        provided, so this just needs to be consistent within the conversation."""
        import datetime

        p = self._persona
        for key in ("date_of_birth", "dob", "birth_date"):
            if p.get(key):
                return str(p[key])
        try:
            age = int(p.get("age") or 40)
        except (TypeError, ValueError):
            age = 40
        year = datetime.date.today().year - max(18, min(age, 95))
        h = int(hashlib.sha1(f"{p.get('first_name', '')}{p.get('last_name', '')}".encode()).hexdigest(), 16)
        return f"{h % 12 + 1:02d}/{h // 12 % 28 + 1:02d}/{year}"

    def _synth_pan(self) -> str:
        """A stable, well-formed but fictional Indian PAN (``AAAAA9999A``).

        Deterministic from the persona so the same customer quotes the same PAN
        across turns. Only the FORMAT matters: the mock gate is presence-based,
        and a real PAN must never appear in synthetic data.
        """
        p = self._persona
        h = hashlib.sha1(f"pan:{p.get('first_name', '')}{p.get('last_name', '')}".encode()).hexdigest()
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        head = "".join(letters[int(h[i : i + 2], 16) % 26] for i in range(0, 10, 2))
        digits = f"{int(h[10:14], 16) % 10000:04d}"
        tail = letters[int(h[14:16], 16) % 26]
        return f"{head}{digits}{tail}"

    def _identity_value(self, field: str, persona_name: str) -> str | None:
        """The customer's value for one region-required identity field."""
        if field == "full_name":
            return persona_name
        if field == "date_of_birth":
            return self._synth_dob()
        if field == "pan":
            return self._synth_pan()
        if field in ("aadhaar_last4", "national_id_last4"):
            h = hashlib.sha1(f"{field}:{persona_name}".encode()).hexdigest()
            return f"{int(h[:4], 16) % 10000:04d}"
        return None

    def _identity_context(self, persona_name: str) -> str:
        """Give the customer their own identity credentials AND instruct them to
        provide them when asked to verify.

        Driven by the REGION's required fields (``region_meta.identity_verification``),
        so an India run grounds a PAN alongside name + DOB while a US run stays at
        name + DOB. Without this the privacy-cautious user-sim declines to share a
        DOB and the conversation dead-locks before any account action (mirrors
        tau-Knowledge's user-simulator 'Verification info' block)."""
        required = self._bank.identity_required_fields()
        pairs = [(f.replace("_", " "), self._identity_value(f, persona_name)) for f in required]
        known = [(label, val) for label, val in pairs if val]
        if not known:
            return ""
        listed = ", ".join(f"{label} {val}" for label, val in known)
        names = " AND ".join(label for label, _ in known)
        return (
            "Your identity, for when the agent asks to verify you before acting on "
            f"your account: {listed}. Identity verification is required, and you "
            "are comfortable giving these to your own institution — so provide "
            f"your {names} promptly when asked, even if you normally share "
            f"details gradually. Do not refuse or stall on any of them."
        )

    def _counterparty_context(self, inst: "FinanceInstance") -> str:
        """License + rules for the OTHER person in the request (an authorized user, a
        nominee, a beneficiary, an accountant).

        Tasks name a RELATIONSHIP and never a person, so the customer supplies the
        identity. Without this the agent asks for the nominee's date of birth and the
        conversation dies there: `incremental` disclosure tells the user to reveal
        details only when asked, and nothing told them they know a third party's. Eight
        en_US rows stalled exactly that way, which is why `manage_authorized_users` was
        the most-missed tool in the run.

        Deliberately keyed on the PRESENCE of a person slot, never on parsing its value.
        These values are user-visible (interpolated into the opening), so a locale
        authoring in its own language writes "ma femme" -- English keyword matching
        would silently stop working. The model can read the relationship itself.
        """
        mentioned = [
            str(v).strip() for k, v in (inst.params or {}).items() if k in PERSON_PARAM_SLOTS and str(v).strip()
        ]
        # A trust or "my two children equally" is not someone with a date of birth.
        individuals = [v for v in mentioned if not any(m in v.lower() for m in _NON_INDIVIDUAL_MARKERS)]
        if not individuals:
            return ""
        gold_tools = set(inst.gold.gold_tool_sequence) if inst.gold else set()
        age_rule = (
            "They are an adult."
            if gold_tools & _ADULT_REQUIRED_TOOLS
            else "Their age suits that relationship and your own."
        )
        several = (
            " If you brought up more than one person, they are different people with different names."
            if len(individuals) > 1
            else ""
        )
        return (
            "About the other person in your request: you know exactly who they are. "
            "When the agent asks for their name, date of birth, or contact details, "
            "give them — do not say you are unsure and do not stall, and keep the same "
            "details for the rest of the conversation."
            f"{several} Their name fits your own cultural and linguistic background, "
            "written the way you would write it in this conversation, and shares your "
            "family name only where that is the normal custom for that relationship "
            "here — a professional such as an accountant or attorney never shares it. "
            f"Their name and any pronouns match the relationship you named. {age_rule} "
            "Ordinary details only: you would not recite a government ID number for "
            "someone else."
        )

    def _synth_account_id(self, inst: "FinanceInstance") -> str:
        """A stable per-(persona, institution) account id for dynamic instances,
        which carry no authored account_state -- so the customer is an existing
        account holder rather than an unknown."""
        puuid = self._data.get("persona_uuid") or compute_persona_uuid(self._persona)
        digest = hashlib.sha1(f"{inst.institution_id}:{puuid}".encode()).hexdigest()[:8]
        return f"acct_{inst.institution_id}_{digest}"

    def _grounding_accounts(self, inst: "FinanceInstance") -> list:
        """The canonical account view for this instance -- shared by the read
        tools (via ``seed_state_metadata``) and the user prompt. Dynamic tasks
        carry no account_state, so synthesize one stable account (the persona is
        an existing customer). New-account verifiable flows keep no account
        (grounding to an existing one would be wrong)."""
        state = dict(inst.account_state)
        if inst.is_dynamic and not state.get("accounts") and not state.get("account_id"):
            state["account_id"] = self._synth_account_id(inst)
        seed = "" if inst.is_dynamic else inst.template_id
        return _resolve_accounts(
            state,
            domain=inst.domain,
            institution_type=inst.institution_type,
            seed=seed,
            account_label=self._bank.domain_account_label(inst.domain),
        )

    def _account_context(self, inst: "FinanceInstance") -> str:
        """Ground the user-sim to the SAME account the read tools return, so the
        customer quotes a consistent account id / last-4 / type instead of
        inventing details that contradict the mock and stall the agent."""
        accts = self._grounding_accounts(inst)
        if not accts:
            return ""
        a = accts[0]
        return (
            "For reference, your account on file with this institution is your "
            f"{a['account_type']} ending {a['last4']} (current balance "
            f"${a['balance']}). If the agent needs specific account details, use "
            "THESE exact ones — never invent a different account number or last-4."
        )


# ---------------------------------------------------------------------------
# Public shim
# ---------------------------------------------------------------------------


async def simulate_financial_services(
    models: dict[str, Any],
    data: dict[str, Any],
    persona: dict[str, Any],
    profile: dict[str, Any],
    locale: str,
    language: str,
    cfg: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Thin shim: catch construction errors -> structured SCENARIO_ABORTED."""
    from usersim.engine.core.probes import BankLoadError

    provenance = kwargs.get("provenance") or Provenance()
    outcome_builder = kwargs.get("outcome_builder") or OutcomeBuilder(
        provenance=provenance,
    )
    try:
        probe = FinancialServicesProbe(
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
        return _aborted(f"financial_services bank load failed: {e}", provenance)
    except FinancialServicesProbeError as e:
        return _aborted(str(e), provenance)
    return await probe.run_dispatch(models=models, data=data, cfg=cfg)


def _aborted(reason: str, provenance: Any) -> dict[str, Any]:
    builder = OutcomeBuilder(provenance=provenance or Provenance())
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


# Precedence for the label a listing tool reports, most specific first:
#   1. the task's authored ``account_state['account_type']``;
#   2. the REGION's own product vocabulary (``region_meta.account_labels``, via
#      ``FinanceBank.domain_account_label``) — "checking account" in the US,
#      "savings account" in India;
#   3. a region-neutral per-domain label.
# Step 2 matters because naming an account after a product the market does not
# have reads as the wrong institution; step 3 keeps a listing tool from offering a
# "checking account" for a retirement/brokerage request, which the assistant
# rightly refuses to act on -> a read-loop.


def _seeded_last4(seed: str) -> str:
    """Deterministic 4-digit account-number suffix from ``seed``. Stable within a
    task (same seed -> same last4) and distinct across tasks, so trajectories
    don't all quote the same '0001' when a task doesn't author a last4."""
    return f"{int(hashlib.sha1(seed.encode()).hexdigest(), 16) % 10000:04d}"


def _resolve_accounts(
    account_state: dict[str, Any],
    *,
    domain: str = "",
    institution_type: str = "",
    seed: str = "",
    account_label: str = "",
) -> list:
    """The ONE canonical account view the read tools return AND the user is
    grounded to, so both sides reference the same account_id / last4 / type
    (a mismatch makes the assistant refuse to act -> a read-loop).

    Prefers an explicit ``accounts`` list, else an authored ``account_type`` /
    ``last4``, else ``account_label`` (the region's own product vocabulary, passed
    in by the caller) or a region-neutral domain label, plus a ``seed``-derived
    last4 (the printed suffix is intentionally independent of the internal
    account_id, and distinct per task). Returns ``[]`` when there is no concrete
    account to ground on (e.g. new-account / rollover flows), so those tools
    honestly report none."""
    explicit = account_state.get("accounts")
    if isinstance(explicit, list) and explicit:
        return explicit
    account_id = account_state.get("account_id")
    if not account_id:
        return []
    account_id = str(account_id)
    last4 = account_state.get("last4")
    if not last4:
        last4 = _seeded_last4(seed or account_id)
    return [
        {
            "account_id": account_id,
            "account_type": account_state.get("account_type")
            or account_label
            or DEFAULT_DOMAIN_ACCOUNT_LABELS.get(domain, "account"),
            "last4": str(last4),
            "balance": account_state.get("balance", "0.00"),
            "status": account_state.get("status", "open"),
        }
    ]


def _accounts_view(account_state: dict[str, Any]) -> list:
    """Accounts as a listing tool returns them. ``seed_state_metadata`` populates
    ``account_state['accounts']`` with the resolved canonical view, so this simply
    returns it (falling back to a domain-agnostic synth if it wasn't seeded)."""
    return _resolve_accounts(account_state)


def _verify_identity(
    args: dict[str, Any],
    required_fields: Sequence[str] = (),
) -> tuple:
    """Identity gate over the REGION's required fields.

    Every required field must be present or verification fails, so an assistant
    that skips one (or a customer who declines) cannot proceed to a
    state-changing action. The set is regional data — the US stops at name +
    DOB, India also expects a PAN — and is advertised in the ``verify_identity``
    schema, so this enforces a requirement the agent was told about. Presence-
    based (the mock holds no ground-truth PII), which is enough to force the
    collection step.

    Returns ``(verified: bool, payload: dict)``."""
    required = tuple(required_fields) or DEFAULT_IDENTITY_REQUIRED_FIELDS
    missing = [f for f in required if not str(args.get(f) or "").strip()]
    if not missing:
        return True, {"status": "verified"}
    return False, {
        "status": "unverified",
        "reason": (
            "identity verification requires "
            f"{' and '.join(f.replace('_', ' ') for f in missing)}; "
            "ask the customer to provide it"
        ),
    }


def _resolve_card(account_state: dict[str, Any], args: dict[str, Any]):
    """Resolve the card a card action targets. Prefer an explicit ``card_id`` that
    matches a card on file; otherwise fall back to the single card on file. There
    is no card-listing tool, so the customer references "the card ending X" and the
    bank acts on their card -- this lets the action hit the REAL card (applying the
    state delta) instead of a fabricated / "unknown" id."""
    cards = account_state.get("cards") or []
    want = str(args.get("card_id") or "").strip()
    for card in cards:
        if card.get("card_id") == want:
            return card
    return cards[0] if cards else None


def _apply_tool(
    tool_name: str,
    args: dict[str, Any],
    account_state: dict[str, Any],
    read_only_tools: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Deterministic mock tool execution mutating the in-memory account state.

    Acknowledges the call and applies the obvious state transition for the tools
    that have one, so within-trajectory responses stay consistent (not static
    mocks). Anything else is answered by its SIDE-EFFECT CLASS — reads get a
    definitive account view, state changes get a referenced acknowledgement —
    which keeps every locale's tool taxonomy usable without enumerating it here.
    """
    if tool_name == "file_transaction_dispute":
        txn_id = str(args.get("transaction_id", "unknown"))
        for txn in account_state.get("transactions", []):
            if txn.get("transaction_id") == txn_id:
                txn["status"] = "UNDER_REVIEW"
        return {"dispute_id": f"dsp_{txn_id}", "status": "UNDER_REVIEW"}
    if tool_name == "freeze_card":
        card = _resolve_card(account_state, args)
        if card is not None:
            card["status"] = "frozen"
        return {"card_id": (card or {}).get("card_id", "unknown"), "status": "frozen"}
    if tool_name == "activate_card":
        card = _resolve_card(account_state, args)
        if card is not None:
            card["status"] = "active"
        return {"card_id": (card or {}).get("card_id", "unknown"), "status": "active"}
    # Account-listing / holdings reads: return a DEFINITIVE account view so the
    # assistant resolves the account_id once and proceeds to the action, instead
    # of re-querying (a generic {"status": "ok"} invited an endless read-loop that
    # never reached the state-changing gold tool). Cards are included so card
    # actions can target a real card_id (there is no separate card-listing tool).
    if tool_name in ("get_accounts", "get_retirement_accounts"):
        payload: dict[str, Any] = {"accounts": _accounts_view(account_state)}
        if account_state.get("cards"):
            payload["cards"] = account_state["cards"]
        return payload
    if tool_name == "get_positions":
        return {"accounts": _accounts_view(account_state), "positions": account_state.get("positions", [])}
    if tool_name == "get_portfolio":
        return {"accounts": _accounts_view(account_state), "holdings": account_state.get("holdings", [])}
    if tool_name == "get_account_transactions":
        return {"transactions": account_state.get("transactions", [])}
    if tool_name == "close_account":
        return {"account_id": args.get("account_id"), "status": "CLOSED"}
    if tool_name == "transfer_funds":
        account_state["transfer_status"] = "COMPLETED"
        return {"transfer_id": "txf_0001", "amount": args.get("amount"), "status": "COMPLETED"}
    if tool_name == "order_replacement_card":
        card = _resolve_card(account_state, args)
        if card is not None:
            card["replacement"] = "ordered"
        return {
            "card_id": (card or {}).get("card_id", "card_nb_0001"),
            "replacement_status": "ordered",
            "eta_business_days": 7,
        }
    if tool_name == "open_account":
        account_state["evergreen_account"] = "acct_new_0001"
        return {"account_id": "acct_new_0001", "status": "OPEN", "account_type": args.get("account_type")}
    if tool_name == "initiate_rollover":
        account_state["rollover_status"] = "IN_PROGRESS"
        return {"rollover_id": "rlv_0001", "status": "IN_PROGRESS"}
    if tool_name == "request_withdrawal":
        account_state["withdrawal_status"] = "PROCESSING"
        return {"withdrawal_id": "wd_0001", "amount": args.get("amount"), "status": "PROCESSING"}

    # ── Fallbacks keyed on the tool's SIDE-EFFECT CLASS ──────────────────
    # Enumerating tools does not scale across locales (and already left
    # get_statements / dispute_status / get_order_status on the generic path),
    # so anything not special-cased above is answered by class.
    if tool_name in read_only_tools:
        # A read must come back DEFINITIVE. A bare {"status": "ok"} tells the
        # assistant nothing, so it re-queries and burns the turn budget without
        # ever reaching the state-changing gold tool.
        payload: dict[str, Any] = {"accounts": _accounts_view(account_state)}
        if account_state.get("cards"):
            payload["cards"] = account_state["cards"]
        for key in ("transactions", "positions", "holdings"):
            if account_state.get(key):
                payload[key] = account_state[key]
        return payload

    # State-changing: acknowledge with a reference the assistant can quote back,
    # so it treats the action as done and moves to confirmation instead of
    # retrying an apparently-inert call. Deterministic per (tool, args) so the
    # same call yields the same reference within a trajectory.
    ref = hashlib.sha1(f"{tool_name}:{sorted((str(k), str(v)) for k, v in args.items())}".encode()).hexdigest()[:8]
    ack: dict[str, Any] = {
        "status": "COMPLETED",
        "reference_id": f"ref_{ref}",
        "tool_name": tool_name,
    }
    # Echo the salient arguments back; a confirmation that repeats the amount /
    # account reads as a real receipt and lets the assistant summarize accurately.
    for key in ("amount", "account_id", "policy_id", "loan_id", "sip_id"):
        if args.get(key) is not None:
            ack[key] = args[key]
    return ack
