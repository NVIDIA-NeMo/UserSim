# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
  the ``used_placeholder_fact`` warning appended whenever the picked
  fact had ``placeholder: true`` (substrate-driven).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from usersim.engine.probes.sov_ai_facts.bank import (
    Fact,
    default_fact_bank_path,
    load_fact_bank,
    load_fact_bank_for_locale,
    reset_fact_bank_cache,
)
from usersim.engine.probes.sov_ai_facts.prompts import (
    PromptsUnavailableError,
    get_followup_instruction,
    get_system_prompt,
)
from usersim.engine.probes.sov_ai_facts.task_derivation import (
    derive_task,
)
from usersim.extension import (
    SHIPPED_LOCALES,
    BankBackedProbe,
    BankVerbatimMixin,
    ConversationState,
    PreviewField,
    format_persona_for_prompt,
    register_probe,
)

logger = logging.getLogger("usersim.engine")

# ---------------------------------------------------------------------------
# Bank-loading aliases (parallel to sov_ai_multilingual_parity / sov_ai_dynamic)
# ---------------------------------------------------------------------------
#
# Thin re-exports so tests can patch on the probe module's surface
# while the cache lives in ``bank.py`` beside this module, where the scorer
# also reads it from. Single shared cache prevents the simulator and
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

    The generator records it as a failed row with
    ``failure_class=SCENARIO_ABORTED``.
    """


# ---------------------------------------------------------------------------
# Probe class
# ---------------------------------------------------------------------------


@register_probe
class SovAiFactsProbe(BankVerbatimMixin, BankBackedProbe):
    """Sovereign-AI factual-knowledge probe.

    Bank-driven turn-1 (verbatim from the curated fact's
    ``question`` field, including any embedded false premise);
    user-LLM follow-up with the locale-specific persona role-play
    prompt + per-turn follow-up instruction.
    """

    label = "sov_ai_facts"
    # Bump on any semantic change to the user-agent prompts or the follow-up
    # instruction. The variant stays "default": the picked fact's category
    # (e.g. ``brazil-geography``) is known only after ``derive_task`` runs, so
    # ``build_result_extras`` records it as the row's probe_variant, while
    # ``trajectory_id`` carries "default".
    prompt_version = "v1.0"
    assets_root = Path(__file__).parent / "assets"
    description = (
        "Tests recall of locale-specific (sovereign) facts drawn from a curated bank, "
        "graded against ground-truth answers. Some questions embed a false premise to "
        "correct; others give a partial verse for the assistant to continue."
    )
    default_scorers = ("sov_ai_facts",)
    preview_fields = (
        PreviewField("facts_probed", "Facts probed", from_metadata=True),
        PreviewField("fact_category", "Fact category", from_metadata=True),
        PreviewField("fact_question_type", "Question type", from_metadata=True),
        PreviewField("fact_difficulty", "Difficulty", from_metadata=True),
    )
    output_columns = ("sovereign_facts_probed",)
    bank_loader = staticmethod(_bank_loader)
    placeholder_warning_kind = "used_placeholder_fact"
    bank_version_key = None  # locale-keyed (the inclusion convention)

    @classmethod
    def check_locale(cls, locale_dir: Path) -> None:
        """Load the locale's fact bank, ``<locale_dir>/sample.yaml``."""
        load_fact_bank(locale_dir / "sample.yaml")

    @classmethod
    def asset_problems(cls) -> list[str]:
        """A fact bank for every shipped locale."""
        problems = []
        for locale in SHIPPED_LOCALES:
            path = default_fact_bank_path(locale)
            if not path.exists():
                problems.append(f"no fact bank for {locale} at {path}")
        return problems

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Fail-fast contract: ``derive_task`` returns None when the
        # persona's tags match no shipped fact. This is a structural
        # SCENARIO_ABORTED, which the generator records as a failed row.
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
        # docs/engine/README.md "Assistant under test:
        # pure-capability policy" for the rationale.
        return ""

    # ── Optional hooks (verbatim + follow-up + invariants) ──────────

    async def get_verbatim_first_user_turn(
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
        return await self._localize_verbatim(self._task.question)

    async def format_followup_user_instructions(
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
