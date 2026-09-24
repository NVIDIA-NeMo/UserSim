# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""identity_disclosure probe: does the model name the developer that made it?

A persona user asks the model under test about itself, in the row's language,
with no assistant system prompt. Each row records the answers that model may
give (its expected identity, resolved from the spec by the assistant model's
id) together with the vocabulary and rules they are graded against, so
scoring needs nothing beyond the row.

How a row is built:

- The tactic and strategy come from ``task_derivation``: deterministic in the
  persona, the spec digest and the run seed, or pinned by a panel.
- The model under test is the id of the row's ``assistant_model``, as the
  models config names it. A spec rule declaring that exact id decides its
  expected identity; otherwise it is inferred from a model family name in the
  id. A model neither declared nor inferable aborts the row before any model
  call, and the message shows the declaration that would fix it. A facade
  that exposes no model id (a test stub) gets a neutral question and no
  expected identity, so its row cannot be graded.
- The competitor pool drops the expected developer and its lineage, so a
  leading question never offers the true developer as the false premise.

Columns written, all declared in ``ConversationSimulatorConfig.side_effect_columns``:
``identity_tactic_id``, ``identity_tactic_kind`` (``neutral`` or ``leading``),
``identity_pair``, ``identity_competitor``, ``strategy_id``,
``expected_identity`` and ``identity_spec`` (both JSON), and
``probe_variant = "<tactic>::<strategy>"``.

A probe built on this one subclasses it with its own ``label``. Its spec is
``assets/<label>/spec.yaml`` (or ``USERSIM_<LABEL>_SPEC``), and its
``bank_version_key`` follows the label unless the subclass sets one.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from typing import Any

from usersim.engine.config import MODEL_ASSISTANT
from usersim.engine.core.behavioral import (
    format_behavioral_profile_for_prompt,
    format_interaction_style_instructions,
)
from usersim.engine.core.identity import persona_uuid, resolve_model_name
from usersim.engine.core.identity_spec import (
    PROBE_LABEL,
    ExpectedIdentity,
    IdentitySpec,
    load_identity_spec_default,
    spec_env_var,
)
from usersim.engine.core.locale import SHIPPED_LOCALES, india_variant
from usersim.engine.core.outcomes import WarningKind
from usersim.engine.core.persona import format_persona_for_prompt
from usersim.engine.core.probes import BankBackedProbe, BankVerbatimMixin, register_probe
from usersim.engine.core.simulation import ConversationState, language_instruction
from usersim.engine.core.user_turn_policy import (
    FIRST_MESSAGE_PHRASES,
    LANGUAGE_MODEL_NOUNS,
    PUSHBACK_OPENERS,
    UserTurnPolicy,
)
from usersim.engine.probes.identity_disclosure.prompts import GATE_PROMPTS, USER_SYSTEM_PROMPTS
from usersim.engine.probes.identity_disclosure.task_derivation import (
    COMPETITOR_PLACEHOLDERS,
    IdentityTask,
    IdentityTaskError,
    fill,
    placeholder_values,
    placeholders_in,
    select_competitor,
    select_strategy,
    select_tactic,
)

logger = logging.getLogger("usersim.engine")

PROBE_FAMILY: str = PROBE_LABEL
PROBE_VARIANTS: list[str] = ["default"]
PROMPT_VERSION: str = "v1.0"

#: (label, model id, spec digest) already announced in this process.
_announced: set[tuple[str, str, str]] = set()


def should_succeed(state: ConversationState) -> bool:
    """Sim-side guardrail: did the row pick a tactic?"""
    return bool(state.metadata.get("identity_tactic_id"))


class IdentityDisclosureProbeError(ValueError):
    """The probe cannot build this row; the dispatcher records it as scenario_aborted."""


@register_probe(family=PROBE_FAMILY, prompt_version=PROMPT_VERSION, variants=tuple(PROBE_VARIANTS))
class IdentityDisclosureProbe(BankVerbatimMixin, BankBackedProbe):
    """Persona users ask the model under test who it is; rows carry what it may truthfully say."""

    label = PROBE_LABEL
    placeholder_warning_kind = WarningKind.USED_PLACEHOLDER_IDENTITY
    bank_version_key = PROBE_LABEL

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A probe built on this one records its spec version under its own label.
        if "bank_version_key" not in cls.__dict__:
            cls.bank_version_key = cls.label

    @classmethod
    def bank_loader(cls) -> IdentitySpec:  # type: ignore[override]
        """The spec named by the probe's label. One spec serves every locale, so it takes none."""
        return load_identity_spec_default(cls.label)

    # ── BankBackedProbe API ─────────────────────────────────────────

    def derive_task(self, persona: dict[str, Any], bank: Any, *, cfg: Any) -> IdentityTask:
        spec: IdentitySpec = bank
        label = type(self).label
        persona_key = persona_uuid(persona)
        seed = getattr(cfg, "random_seed", None)

        model_id = resolve_model_name(self._models.get(MODEL_ASSISTANT), MODEL_ASSISTANT)
        identified = model_id != MODEL_ASSISTANT
        expected = spec.resolve(model_id) if identified else None
        if identified and expected is None:
            raise IdentityDisclosureProbeError(_unmatched_model_message(label, model_id, spec))
        _announce(label, model_id, expected, spec)

        try:
            tactic = select_tactic(
                spec,
                persona_key,
                seed=seed,
                override=_panel_value(self._data.get("identity_tactic_id")),
                neutral_only=not identified,
            )
            strategy = select_strategy(
                spec,
                persona_key,
                seed=seed,
                override=_panel_value(self._data.get("identity_strategy_id")),
            )
        except IdentityTaskError as e:
            raise IdentityDisclosureProbeError(f"{label}: {e}") from e

        # An India variant opens with its own rendering when the spec has one,
        # and otherwise with the base locale's, which _localize_verbatim translates.
        if self._asset_locale != self._locale and self._locale in tactic.text.renderings:
            self._asset_locale = self._locale

        excluded = (*expected.developers, *expected.lineage) if expected else ()
        pool = spec.competitor_pool(self._asset_locale, exclude=excluded)
        names_competitor = bool(placeholders_in(tactic) & COMPETITOR_PLACEHOLDERS)
        competitor = select_competitor(spec, pool, persona_key, seed=seed) if names_competitor else None
        if names_competitor and competitor is None:
            raise IdentityDisclosureProbeError(
                f"{label}: tactic {tactic.id!r} names a competitor, but spec {spec.version} leaves none "
                f"for locale {self._asset_locale} once {', '.join(excluded)} are excluded"
            )
        values = placeholder_values(spec, expected, competitor=competitor, pool=pool, language=self._language)
        return IdentityTask(
            id=tactic.id,
            placeholder=spec.placeholder or tactic.placeholder or strategy.placeholder,
            tactic=tactic,
            strategy=strategy,
            competitor=competitor,
            expected=expected,
            opening=fill(tactic.text.for_locale(self._asset_locale), values),
        )

    # ── ProbeAdapter hooks ──────────────────────────────────────────

    def get_user_system_prompt(self) -> str:
        return USER_SYSTEM_PROMPTS.get(
            _prompt_locale(self._locale),
            persona=format_persona_for_prompt(self._persona),
            language_instruction=language_instruction(self._language, self._locale),
            behavioral_instructions=(
                format_behavioral_profile_for_prompt(
                    self._profile, probe_type=type(self).label, language=self._language
                )
                if self._profile
                else ""
            ),
            interaction_style_instructions=format_interaction_style_instructions(self._interaction_style),
        )

    def get_assistant_system_prompt(self) -> str:
        # Pure-capability policy: what the model says about itself unprompted
        # is the finding, so it gets no system prompt.
        return ""

    def user_turn_policy(self) -> UserTurnPolicy:
        """Earlier replies stay as written, no follow-up wraps up, and the words these users write pass the filters.

        The script check skips the spec's names, products and models, which
        are written in Latin script in every locale.
        """
        spec: IdentitySpec | None = self._bank
        names = (
            {
                name
                for lists in spec.vocabulary().values()
                for key in ("names", "products", "models")
                for name in lists.get(key) or []
            }
            if spec is not None
            else set()
        )
        return UserTurnPolicy(
            context_compression=False,
            wrap_up=False,
            allowed_phrases=LANGUAGE_MODEL_NOUNS | FIRST_MESSAGE_PHRASES | PUSHBACK_OPENERS,
            script_check_ignores=tuple(sorted(names)),
        )

    def format_gate_prompt(self, user_query: str, conversation_history: str) -> str:
        return GATE_PROMPTS.get(
            _prompt_locale(self._locale),
            conversation_history=conversation_history,
            user_turn_to_evaluate=user_query,
        )

    def seed_state_metadata(self, state: ConversationState) -> None:
        task = self._task
        if task is None:
            return
        state.metadata["identity_tactic_id"] = task.tactic.id
        state.metadata["identity_tactic_kind"] = task.tactic.kind.value
        state.metadata["identity_pair"] = task.tactic.pair or None
        state.metadata["identity_competitor"] = task.competitor
        state.metadata["strategy_id"] = task.strategy.id

    async def get_verbatim_first_user_turn(self, state: ConversationState) -> str | None:
        if self._task is None:
            return None
        return await self._localize_verbatim(self._task.opening)

    def should_succeed(self, state: ConversationState) -> bool:
        return should_succeed(state)

    def build_result_extras(self, state: ConversationState) -> dict:
        extras = super().build_result_extras(state)
        task = self._task
        if task is None:
            return extras
        spec: IdentitySpec = self._bank
        extras.update(
            {
                "probe_variant": f"{task.tactic.id}::{task.strategy.id}",
                "identity_tactic_id": task.tactic.id,
                "identity_tactic_kind": task.tactic.kind.value,
                "identity_pair": task.tactic.pair or None,
                "identity_competitor": task.competitor,
                "strategy_id": task.strategy.id,
                "expected_identity": (
                    json.dumps(asdict(task.expected), ensure_ascii=False) if task.expected is not None else None
                ),
                "identity_spec": json.dumps(
                    {
                        "version": spec.version,
                        "digest": spec.digest,
                        "vocabulary": spec.vocabulary(),
                        "rules": spec.rules(),
                    },
                    ensure_ascii=False,
                ),
            }
        )
        return extras


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prompt_locale(locale: str) -> str:
    """The locale whose prompt scaffold a row uses: an India variant uses its base."""
    if locale in SHIPPED_LOCALES:
        return locale
    variant = india_variant(locale)
    return variant.base_locale if variant is not None else "en_US"


def _panel_value(value: Any) -> str | None:
    """A panel override as a string, or None when the column is absent or empty."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _announce(label: str, model_id: str, expected: ExpectedIdentity | None, spec: IdentitySpec) -> None:
    """Log, once per model and spec in this process, what rows are graded against."""
    key = (label, model_id, spec.digest)
    if key in _announced:
        return
    _announced.add(key)
    if expected is None:
        logger.warning(
            "  |-- %s: the assistant model facade exposes no model id, so rows ask a neutral "
            "question and carry no expected identity; they cannot be graded",
            label,
        )
        return
    how = "declared" if expected.declared else f"inferred from {expected.rule!r}"
    logger.info(
        "  |-- %s: expected identity for %s is %s (%s in %s)",
        label,
        model_id,
        ", ".join(expected.developers),
        how,
        expected.layer,
    )


def _unmatched_model_message(label: str, model_id: str, spec: IdentitySpec) -> str:
    return (
        f"{label}: spec {spec.version} cannot infer who made assistant model {model_id!r}. "
        f"Declare it in a spec that extends {label} and point {spec_env_var(label)} at that file:\n"
        f'  schema_version: "1"\n'
        f"  extends: {label}\n"
        f"  bank_id: my_models\n"
        f'  bank_version: "1"\n'
        f"  expected_identities:\n"
        f"    - model: {json.dumps(model_id)}\n"
        f"      developer: <one of {', '.join(sorted(spec.developers))}, or a new one under developers>"
    )
