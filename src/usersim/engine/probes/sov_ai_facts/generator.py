# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""sov_ai_facts probe.

Probes the assistant for locale-specific factual knowledge using a
curated fact bank. The first user turn is the fact's question
**verbatim** — no LLM call synthesizes it. Preserving exact wording
matters for two reasons:

- Factual-recall entries deliberately embed *false premises*;
  rewording them in paraphrase could soften or drop the premise and
  quietly defang the probe.
- Completion entries supply a literal partial verse; paraphrasing
  would change which text the assistant is asked to continue.

Built on ``BaseProbe`` + ``BankBackedProbe`` + ``BankVerbatimMixin``;
the substrate handles bank load + cache + placeholder warning +
bank-version pinning + verbatim turn-1 injection.
``ConversationLoop`` drives the turn loop with shared
guardrails (fourth-wall prefilter, user-judge gate retries,
role-violation telemetry, frustration). What remains in this module
is the configuration that makes this probe *this* probe: the bank
loader, the placeholder warning kind, the locale-specific prompts
(question-type-conditioned), the sim-side ``should_succeed``
invariant, and the per-row metadata.

Side channels emitted on the trajectory:

- ``state.metadata["facts_probed"]`` — list of fact ids probed in
  this trajectory (today always ``[fact.id]``; future extensions can
  probe multiple facts in one conversation).
- ``state.metadata["fact_category"]`` / ``fact_question_type`` /
  ``fact_difficulty`` / ``bank_id`` / ``bank_version`` — for
  stratified reporting and reviewer triage.
- ``result["probe_variant"]`` — set to the picked ``fact.category``
  (per-row discriminator).
- ``result["sovereign_facts_probed"]`` — top-level for the
  ``sov_ai_facts`` scorer (the load-bearing column).
- ``SimulationOutcome.provenance.bank_version["<locale>"]`` — bank
  version used (locale-keyed; ``BankBackedProbe`` substrate handles
  this via ``bank_version_key=None``).
- ``SimulationOutcome.warnings`` —
  ``WarningKind.USED_PLACEHOLDER_FACT`` appended whenever the picked
  fact had ``placeholder: true`` (substrate-driven).
"""

from __future__ import annotations

import logging
from typing import Any

from usersim.engine.core.fact_bank import (
    Fact,
    load_fact_bank_for_locale,
    reset_fact_bank_cache,
)
from usersim.engine.core.outcomes import WarningKind
from usersim.engine.core.persona import format_persona_for_prompt
from usersim.engine.core.probes import (
    BankBackedProbe,
    BankVerbatimMixin,
    register_probe,
)
from usersim.engine.core.simulation import ConversationState, make_failed
from usersim.engine.probes.sov_ai_facts.prompts import (
    PromptsUnavailableError,
    get_followup_instruction,
    get_system_prompt,
)
from usersim.engine.probes.sov_ai_facts.task_derivation import (
    derive_task,
)

logger = logging.getLogger("usersim.engine")

# ── Probe metadata ────────────────────────────────────────────────────
# Bumping ``PROMPT_VERSION`` invalidates prior evaluator rows on
# re-run, since the dedup key includes it. Bump on any semantic change
# to the user-agent prompts (or to the follow-up instruction).

PROBE_FAMILY: str = "sov_ai_facts"
# Probe discriminator is the picked fact's ``category`` (e.g.
# ``brazil-geography``). Because the category is only known after
# ``derive_task`` runs, the probe overrides ``probe_variant`` to the
# fact's category in ``build_result_extras``. The dispatcher seeds
# ``probe_variant = "default"`` initially; that seed is reflected in
# ``trajectory_id`` but the final parquet row carries the real
# discriminator. (Trajectory-id stability vs per-row category split
# is a known trade-off — `trajectory_id` is computed before
# `derive_task` runs, so it carries the seed default; the parquet row
# carries the per-fact category.)
PROBE_VARIANTS: list[str] = ["default"]
PROMPT_VERSION: str = "v1.0"

# ---------------------------------------------------------------------------
# Bank-loading aliases (parallel to sov_ai_multilingual_parity / sov_ai_dynamic)
# ---------------------------------------------------------------------------
#
# Thin re-exports so tests can patch on the probe module's surface
# while the cache still lives in ``core.fact_bank`` (where the scorer
# also reads it from). Single shared cache prevents the simulator and
# the scorer from disagreeing on what bank version was used.

_load_bank_for_locale = load_fact_bank_for_locale
_reset_bank_cache = reset_fact_bank_cache


def _bank_loader(locale: str):
    """Indirection through the module-level alias.

    Resolves ``_load_bank_for_locale`` at call time (not at probe-class
    construction time), so tests that monkeypatch the module-level
    ``_load_bank_for_locale`` work transparently.
    """
    return _load_bank_for_locale(locale)


def should_succeed(state: ConversationState) -> bool:
    """Sim-side guardrail: did we actually probe a fact?

    Returns ``False`` when the trajectory carries no probed facts —
    that happens if ``derive_task`` returned ``None`` *and* the
    probe somehow still produced a trajectory record. Structural
    invariant check, not assistant-quality scoring (the latter lives
    in the evaluator's ``sov_ai_facts`` scorer). Module-level mirror
    of the probe method so existing test surfaces can call this
    directly without an instance.
    """
    return bool(state.metadata.get("facts_probed"))


class SovAiFactsProbeError(ValueError):
    """Raised when the probe cannot construct (no fact, missing prompts).

    The shim ``simulate_sov_ai_facts`` catches this and returns
    a structured ``make_failed`` outcome with
    ``failure_class=SCENARIO_ABORTED``.
    """


# ---------------------------------------------------------------------------
# Probe class
# ---------------------------------------------------------------------------


@register_probe(family=PROBE_FAMILY, prompt_version=PROMPT_VERSION, variants=tuple(PROBE_VARIANTS))
class SovAiFactsProbe(BankVerbatimMixin, BankBackedProbe):
    """Sovereign-AI factual-knowledge probe.

    Bank-driven turn-1 (verbatim from the curated fact's
    ``question`` field, including any embedded false premise);
    user-LLM follow-up with the locale-specific persona role-play
    prompt + per-turn follow-up instruction.
    """

    label = "sov_ai_facts"
    bank_loader = staticmethod(_bank_loader)
    placeholder_warning_kind = WarningKind.USED_PLACEHOLDER_FACT
    bank_version_key = None  # locale-keyed (the inclusion convention)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Fail-fast contract: ``derive_task`` returns None when the
        # persona's tags match no shipped fact. This is a structural
        # SCENARIO_ABORTED — the shim catches and turns it into a
        # make_failed outcome.
        if self._task is None:
            raise SovAiFactsProbeError(
                f"no fact in {self._bank.bank_id} v{self._bank.bank_version} "
                f"matched persona tags for locale={self._locale}"
            )
        try:
            # Resolve prompts against the ASSET locale (en_IN for an India
            # variant that reuses the en_IN fact bank), not the conversation
            # locale — en_IN is a shipped pack key so this never aborts, and
            # ``available_locales()`` is untouched. The fact bank's own
            # ``derive_task`` already uses ``fact_bank.locale`` for tags.
            user_system_template = get_system_prompt(
                self._asset_locale,
                self._task.question_type,
            )
            self._followup_instruction = get_followup_instruction(self._asset_locale)
        except PromptsUnavailableError as e:
            raise SovAiFactsProbeError(str(e)) from e
        self._user_system_prompt = user_system_template.format(
            persona=format_persona_for_prompt(self._persona),
            category=self._task.category,
            difficulty=self._task.difficulty,
        )

    # ── BankBackedProbe API ─────────────────────────────────────────

    def derive_task(
        self,
        persona: dict[str, Any],
        bank: Any,
        *,
        cfg: Any,
    ) -> Fact | None:
        """Pick the matched fact for this persona; bank is locale-keyed."""
        return derive_task(
            persona,
            bank,
            seed=getattr(cfg, "random_seed", None),
        )

    # ── ProbeAdapter required hooks ─────────────────────────────────

    def get_user_system_prompt(self) -> str:
        return self._user_system_prompt

    def get_assistant_system_prompt(self) -> str:
        # Pure-capability-test policy: see
        # conversation-plugin/README.md "Assistant under test:
        # pure-capability policy" for the rationale.
        return ""

    # ── Optional hooks (verbatim + follow-up + invariants) ──────────

    def get_verbatim_first_user_turn(
        self,
        state: ConversationState,
    ) -> str | None:
        if self._task is None:
            return None
        # Seed the side-channel metadata before the loop's
        # _inject_verbatim_first_turn helper runs (it preserves
        # existing keys via setdefault).
        state.metadata["facts_probed"] = [self._task.id]
        state.metadata["fact_category"] = self._task.category
        state.metadata["fact_question_type"] = self._task.question_type
        state.metadata["fact_difficulty"] = self._task.difficulty
        state.metadata["bank_id"] = self._task.bank_id
        state.metadata["bank_version"] = self._task.bank_version
        # Verbatim English fact question; machine-translated into the
        # conversation language for India variants that fell back to the
        # en_IN bank (no-op for shipped locales / native variant assets).
        # False premises are preserved by the translation prompt.
        return self._localize_verbatim(self._task.question)

    def format_followup_user_instructions(
        self,
        turn_idx: int,
        state: ConversationState,
    ) -> list[str]:
        if not self._followup_instruction:
            return []
        return [self._followup_instruction]

    def should_succeed(self, state: ConversationState) -> bool:
        return should_succeed(state)

    def build_result_extras(self, state: ConversationState) -> dict:
        extras = super().build_result_extras(state)
        if self._task is not None:
            extras["probe_variant"] = self._task.category
            extras["sovereign_facts_probed"] = list(state.metadata.get("facts_probed") or [])
        return extras


# ---------------------------------------------------------------------------
# Public entry point — thin shim around SovAiFactsProbe
# ---------------------------------------------------------------------------


def simulate_sov_ai_facts(
    models: dict[str, Any],
    data: dict[str, Any],
    persona: dict[str, Any],
    profile: dict[str, Any],
    locale: str,
    language: str,
    cfg: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Thin shim for callers that import ``simulate_sov_ai_facts``.

    The dispatcher in ``usersim.engine.generator`` instantiates
    ``SovAiFactsProbe`` and calls ``probe.run_dispatch`` directly;
    this function exists for backward compat with per-probe tests that
    import the symbol.

    Catches construction exceptions (bank-load / no-matching-fact /
    prompt-pack-missing) and returns a ``make_failed`` outcome with
    ``failure_class=SCENARIO_ABORTED``.
    """
    from usersim.engine.core.outcomes import OutcomeBuilder
    from usersim.engine.core.outcomes import Provenance as _Provenance
    from usersim.engine.core.probes import BankLoadError

    provenance = kwargs.get("provenance") or _Provenance()
    # Construct an OutcomeBuilder so placeholder warnings + bank-version
    # pins emitted at probe construction time survive into the final
    # outcome. The dispatcher in ``usersim.engine.generator`` does
    # this itself; the shim mirrors the contract for direct callers.
    outcome_builder = kwargs.get("outcome_builder") or OutcomeBuilder(
        provenance=provenance,
    )
    try:
        probe = SovAiFactsProbe(
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
        return _aborted(
            f"fact-bank load failed for locale={locale}: {e}",
            provenance,
        )
    except SovAiFactsProbeError as e:
        return _aborted(str(e), provenance)
    return probe.run_dispatch(models=models, data=data, cfg=cfg)


def _aborted(reason: str, provenance: Any) -> dict[str, Any]:
    """Build a structured SCENARIO_ABORTED failure result."""
    from usersim.engine.core.outcomes import (
        FailureAttribution,
        FailureClass,
        OutcomeBuilder,
        OutcomeStatus,
    )
    from usersim.engine.core.outcomes import (
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
