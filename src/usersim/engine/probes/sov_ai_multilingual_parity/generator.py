# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sov_ai_multilingual_parity probe.

Probes the assistant for **respect parity** within a locale and
**quality consistency** across locales. Built on ``BaseProbe`` +
``BankBackedProbe`` + ``BankVerbatimMixin`` — the substrate handles
bank load + cache + placeholder warning + bank-version pinning;
``BankVerbatimMixin`` handles turn-1 verbatim injection;
``ConversationLoop`` drives the turn loop with shared guardrails
(fourth-wall prefilter, user-judge gate retries, role-violation
telemetry, frustration). What lives in this module is the
configuration that makes this probe *this* probe: the bank loader,
the placeholder warning kind, the locale-specific prompts, the
sim-side ``should_succeed`` invariant, and the per-row metadata.

Side channels emitted on the trajectory:

- ``state.metadata["query_id"]`` — the cross-locale matched-pair join
  key (load-bearing for downstream cross-locale parity analysis).
- ``state.metadata["query_domain"]`` / ``query_difficulty`` /
  ``query_concern`` — for stratified reporting and reviewer triage.
- ``state.metadata["bank_id"]`` / ``bank_version`` — so the scorer
  can re-load the bank and the reporter can stratify by bank version.
- ``result["probe_variant"]`` — set to the picked ``query.id``
  (per-row discriminator).
- ``result["query_id"]`` — top-level for evaluator scorer + matched-
  pair join (the load-bearing column).
- ``SimulationOutcome.provenance.bank_version["<locale>"]`` — bank
  version used (locale-keyed; ``BankBackedProbe`` substrate handles
  this via ``bank_version_key=None``).
- ``SimulationOutcome.warnings`` — ``WarningKind.USED_PLACEHOLDER_QUERY``
  appended whenever the picked query had ``placeholder: true``
  (substrate handles this).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from usersim.engine.core.outcomes import WarningKind
from usersim.engine.core.persona import format_persona_for_prompt
from usersim.engine.core.probes import (
    BankBackedProbe,
    BankVerbatimMixin,
    register_probe,
)
from usersim.engine.core.query_bank import (
    Query,
    load_query_bank_default,
    reset_query_bank_cache,
)
from usersim.engine.core.simulation import ConversationState, make_failed
from usersim.engine.probes.sov_ai_multilingual_parity.prompts import (
    PromptsUnavailableError,
    get_followup_instruction,
    get_system_prompt,
)
from usersim.engine.probes.sov_ai_multilingual_parity.task_derivation import (
    derive_task,
)

logger = logging.getLogger("usersim.engine")

# ── Probe metadata ────────────────────────────────────────────────────
# Bumping ``PROMPT_VERSION`` invalidates prior evaluator rows on
# re-run, since the dedup key includes it. Bump on any semantic change
# to the user-agent system prompts or follow-up instruction.

PROBE_FAMILY: str = "sov_ai_multilingual_parity"
# Probe discriminator is the query's id (e.g. ``Q-MEDS-001``). The
# dispatcher seeds ``probe_variant = "default"``; the probe overrides
# it to the picked query's id once known.
PROBE_VARIANTS: list[str] = ["default"]
PROMPT_VERSION: str = "v1.0"

# ---------------------------------------------------------------------------
# Bank-loading aliases (parallel to sov_ai_facts / sov_ai_dynamic modules)
# ---------------------------------------------------------------------------
#
# Thin re-exports so tests can patch on the probe module's surface
# while the cache still lives in ``core.query_bank`` (where the
# scorer also reads it from). Single shared cache prevents the
# simulator and the scorer from disagreeing on what bank version was
# used.

_load_bank = load_query_bank_default
_reset_bank_cache = reset_query_bank_cache


def _bank_loader(locale: str):
    """Indirection through the module-level ``_load_bank`` alias.

    Resolves ``_load_bank`` at call time (not at probe-class
    construction time), so tests that monkeypatch the module-level
    ``_load_bank`` work transparently. The substrate's
    ``BankBackedProbe`` calls ``type(self).bank_loader(locale)`` —
    pointing at this thin wrapper preserves the patchable surface.
    """
    return _load_bank()


def should_succeed(state: ConversationState) -> bool:
    """Sim-side guardrail: did we actually probe a query?

    Returns ``False`` when the trajectory carries no ``query_id`` —
    that would happen if the probe somehow produced a trajectory
    record without picking a query. Structural-invariant check, not
    assistant-quality scoring (the latter lives in the evaluator's
    respect-parity scorer at
    ``evaluator/scorers/sov_ai_multilingual_parity.py``). Module-level
    mirror of the probe method so existing test surfaces
    (``module.should_succeed``) can call this directly without an
    instance.
    """
    return bool(state.metadata.get("query_id"))


class SovAiMultilingualParityProbeError(ValueError):
    """Raised when the probe cannot construct (no task, missing prompts, etc.).

    The shim ``simulate_sov_ai_multilingual_parity`` catches this and
    returns a structured ``make_failed`` outcome with
    ``failure_class=SCENARIO_ABORTED``.
    """


# ---------------------------------------------------------------------------
# Probe class
# ---------------------------------------------------------------------------


@register_probe(family=PROBE_FAMILY, prompt_version=PROMPT_VERSION,
                variants=tuple(PROBE_VARIANTS))
class SovAiMultilingualParityProbe(BankVerbatimMixin, BankBackedProbe):
    """Multilingual respect-parity / quality-consistency probe.

    Bank-driven turn-1 (verbatim from the locale's authored query
    rendering); user-LLM follow-up with the locale-specific persona
    role-play prompt + per-turn follow-up instruction.
    """

    label = "sov_ai_multilingual_parity"
    # Routes through the module-level ``_load_bank`` alias so tests
    # that monkeypatch ``_load_bank`` propagate correctly.
    bank_loader = staticmethod(_bank_loader)
    placeholder_warning_kind = WarningKind.USED_PLACEHOLDER_QUERY
    bank_version_key = None  # locale-keyed (the inclusion convention)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Fail-fast contract: ``BankBackedProbe.derive_task`` returns
        # None when (a) the persona's tags match no shipped query or
        # (b) the requested locale is not in the bank. Either case
        # is a structural SCENARIO_ABORTED — the shim catches and
        # turns it into a make_failed outcome.
        if self._task is None:
            raise SovAiMultilingualParityProbeError(
                f"no query in {self._bank.bank_id} v{self._bank.bank_version} "
                f"matched persona tags + locale={self._locale}"
            )
        # Resolve rendering + prompt pack against the ASSET locale (the
        # variant's own rendering if the shared bank has it, else en_IN),
        # not the conversation locale. ``derive_task`` finalized
        # ``self._asset_locale`` above.
        self._rendering = self._task.rendering_for(self._asset_locale)
        if not self._rendering:
            # Defence in depth: derive_task already filters to
            # supports_locale, so this branch is unreachable in
            # practice. If it ever fires, the loader invariant is
            # broken — fail loudly.
            raise SovAiMultilingualParityProbeError(
                f"derive_task picked query {self._task.id} which has no "
                f"rendering for locale={self._asset_locale} — loader invariant "
                "violated"
            )
        try:
            user_system_template = get_system_prompt(self._asset_locale)
            self._followup_instruction = get_followup_instruction(self._asset_locale)
        except PromptsUnavailableError as e:
            raise SovAiMultilingualParityProbeError(str(e)) from e
        self._user_system_prompt = user_system_template.format(
            persona=format_persona_for_prompt(self._persona),
            domain=self._task.domain,
            difficulty=self._task.difficulty,
        )

    # ── BankBackedProbe API ─────────────────────────────────────────

    def derive_task(
        self, persona: Dict[str, Any], bank: Any, *, cfg: Any,
    ) -> Optional[Query]:
        """Pick the matched query for this persona's locale.

        Returns None on no-match or unsupported-locale; the substrate
        skips placeholder-warning / version-pin in that case and
        ``should_succeed`` returns False.

        Presence-aware: if this is an India variant whose own rendering the
        shared bank already ships, use it (graduation); otherwise use the
        India base (en_IN) that the substrate resolved into
        ``self._asset_locale``. The matched-pair join key (``query_id``) is
        identical across India variants because they share the same en_IN
        persona hash + bank salt + seed.
        """
        # Presence-aware asset-locale resolution (robust regardless of what
        # the shared-bank loader set): use the variant's own rendering if the
        # bank ships it (graduation), else the India base (en_IN), else the
        # locale itself (shipped locales; unknown -> None below = fail-fast).
        from usersim.engine.core.locale import india_variant
        loc = self._locale
        if loc not in bank.locales:
            variant = india_variant(loc)
            if variant is not None:
                loc = variant.base_locale
        self._asset_locale = loc
        if loc not in bank.locales:
            logger.warning(
                "  |-- sov_ai_multilingual_parity: asset_locale=%s not in bank "
                "%s v%s (locales: %s)",
                loc, bank.bank_id, bank.bank_version,
                sorted(bank.locales),
            )
            return None
        return derive_task(
            persona, bank, loc,
            seed=getattr(cfg, "random_seed", None),
        )

    # ── ProbeAdapter required hooks ─────────────────────────────────

    def get_user_system_prompt(self) -> str:
        return self._user_system_prompt or ""

    def get_assistant_system_prompt(self) -> str:
        # PURE-CAPABILITY-TEST POLICY: empty by design.
        #
        # We deliberately do NOT pin a language affordance like
        # "Respond in {language}". Whether the model responds in the
        # right language is exactly part of what `language_compliance`
        # and the multilingual scorer measure — an out-of-the-box
        # response that defaults to English IS a finding, not a
        # missing affordance. See conversation-plugin/README.md
        # "Assistant under test: pure-capability policy".
        return ""

    # ── Optional hooks (verbatim + follow-up + invariants) ──────────

    def get_verbatim_first_user_turn(
        self, state: ConversationState,
    ) -> Optional[str]:
        if self._rendering is None or self._task is None:
            return None
        # Seed the side-channel metadata before the loop's
        # _inject_verbatim_first_turn helper runs (it preserves
        # existing keys via setdefault).
        state.metadata["query_id"] = self._task.id
        state.metadata["query_domain"] = self._task.domain
        state.metadata["query_difficulty"] = self._task.difficulty
        state.metadata["query_concern"] = self._task.concern
        state.metadata["bank_id"] = self._task.bank_id
        state.metadata["bank_version"] = self._task.bank_version
        # The rendering is the en_IN (base) authored text for an India
        # variant without its own rendering; machine-translate it into the
        # conversation language (no-op for shipped locales / native
        # renderings). query_id is unchanged, so matched pairs still join.
        return self._localize_verbatim(self._rendering)

    def format_followup_user_instructions(
        self, turn_idx: int, state: ConversationState,
    ) -> List[str]:
        if not self._followup_instruction:
            return []
        return [self._followup_instruction]

    def should_succeed(self, state: ConversationState) -> bool:
        """Sim-side guardrail: did we actually probe a query?"""
        return should_succeed(state)

    def build_result_extras(self, state: ConversationState) -> dict:
        extras = super().build_result_extras(state)
        if self._task is not None:
            extras["probe_variant"] = self._task.id
            extras["query_id"] = self._task.id
        return extras


# ---------------------------------------------------------------------------
# Public entry point — thin shim around SovAiMultilingualParityProbe
# ---------------------------------------------------------------------------


def simulate_sov_ai_multilingual_parity(
    models: Dict[str, Any],
    data: Dict[str, Any],
    persona: Dict[str, Any],
    profile: Dict[str, Any],
    locale: str,
    language: str,
    cfg: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Thin shim for callers that import ``simulate_sov_ai_multilingual_parity``.

    The dispatcher in ``usersim.engine.generator`` instantiates
    ``SovAiMultilingualParityProbe`` and calls ``probe.run_dispatch``
    directly; this function exists for backward compat with per-probe
    tests that import the symbol.

    Catches construction exceptions (bank-load / no-matching-query /
    locale-not-in-bank / prompt-pack-missing) and returns a
    ``make_failed`` outcome with ``failure_class=SCENARIO_ABORTED``.
    """
    from usersim.engine.core.outcomes import OutcomeBuilder, Provenance as _Provenance
    from usersim.engine.core.probes import BankLoadError

    provenance = kwargs.get("provenance") or _Provenance()
    # Construct an OutcomeBuilder so placeholder warnings + bank-version
    # pins emitted at probe construction time survive into the final
    # outcome. The dispatcher in ``usersim.engine.generator`` does
    # this itself; the shim mirrors the contract for direct callers
    # (per-probe tests).
    outcome_builder = kwargs.get("outcome_builder") or OutcomeBuilder(
        provenance=provenance,
    )
    try:
        probe = SovAiMultilingualParityProbe(
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
        return _aborted(f"query-bank load failed: {e}", provenance)
    except SovAiMultilingualParityProbeError as e:
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
