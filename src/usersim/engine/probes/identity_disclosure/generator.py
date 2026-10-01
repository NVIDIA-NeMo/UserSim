# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
- A verbatim tactic's text is sent as written. For a generated tactic the
  simulated user writes the opening from the tactic's instruction, in the
  row's language and the persona's voice; a draft naming AI developers the
  tactic does not allow is written again, and the gate judges it against the
  instruction.
- The competitor pool drops the expected developer and its lineage, so a
  leading question never offers the true developer as the false premise.
- A run with follow-up turns gives every identified row a pressure strategy.
  Each follow-up then carries the strategy's next reframing, and the simulated
  user presses the row's competitor with its own prompt, gate and chat rules,
  over every turn of the run. A single-turn run uses the ``none`` strategy.

Columns written, all declared in ``ConversationSimulatorConfig.side_effect_columns``:
``identity_tactic_id``, ``identity_tactic_kind`` (``neutral`` or ``leading``),
``identity_pair``, ``identity_competitor``, ``strategy_id``,
``reframings_used`` (the reframing ids in follow-up order),
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
from collections.abc import Callable
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
    TacticMode,
    Vocabulary,
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
from usersim.engine.probes.identity_disclosure.prompts import (
    FOLLOWUP_ANCHOR,
    GATE_PROMPTS,
    LEADING_NAMES_RULE,
    NEUTRAL_NAMES_RULE,
    OPENING_GATE_PROMPTS,
    OPENING_INSTRUCTION,
    OPENING_SENT,
    OPENING_TO_WRITE,
    PRESSURE_GATE_PROMPTS,
    PRESSURE_INSTRUCTION,
    PRESSURE_USER_SYSTEM_PROMPTS,
    USER_SYSTEM_PROMPTS,
)
from usersim.engine.probes.identity_disclosure.task_derivation import (
    COMPETITOR_PLACEHOLDERS,
    IDENTITY_PLACEHOLDERS,
    IdentityTask,
    IdentityTaskError,
    fill,
    fill_frame,
    opening_frame,
    placeholder_values,
    placeholders_in,
    select_competitor,
    select_strategy,
    select_tactic,
    strategy_placeholders,
)

logger = logging.getLogger("usersim.engine")

PROBE_FAMILY: str = PROBE_LABEL
PROBE_VARIANTS: list[str] = ["default"]
PROMPT_VERSION: str = "v1.0"

#: Every machine-translated opening in a locale addresses the model the same way.
_POLITE_YOU = 'Address the reader with the polite form of "you" used with a stranger.'

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
    #: Set once the loop asks for the first follow-up, so the opening gate judges only turn 1.
    _opening_written = False

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
                # A row that cannot be graded is not worth pressing.
                max_turns=getattr(cfg, "max_turns", 1) if identified else 1,
            )
        except IdentityTaskError as e:
            raise IdentityDisclosureProbeError(f"{label}: {e}") from e

        # An India variant opens with its own rendering when the spec has one,
        # and otherwise with the base locale's, which _localize_verbatim translates.
        if self._asset_locale != self._locale and tactic.text is not None and self._locale in tactic.text.renderings:
            self._asset_locale = self._locale

        excluded = (*expected.developers, *expected.lineage) if expected else ()
        pool = spec.competitor_pool(self._asset_locale, exclude=excluded)
        names_competitor = bool((placeholders_in(tactic) | strategy_placeholders(strategy)) & COMPETITOR_PLACEHOLDERS)
        competitor = select_competitor(spec, pool, persona_key, seed=seed) if names_competitor else None
        if names_competitor and competitor is None:
            raise IdentityDisclosureProbeError(
                f"{label}: tactic {tactic.id!r} or strategy {strategy.id!r} names a competitor, but spec "
                f"{spec.version} leaves none for locale {self._asset_locale} once {', '.join(excluded)} are excluded"
            )
        values = placeholder_values(spec, expected, competitor=competitor, pool=pool, language=self._language)
        return IdentityTask(
            id=tactic.id,
            placeholder=spec.placeholder or tactic.placeholder or strategy.placeholder,
            tactic=tactic,
            strategy=strategy,
            competitor=competitor,
            expected=expected,
            opening=fill(tactic.text.for_locale(self._asset_locale), values) if tactic.text is not None else "",
            values=values,
            instruction=(
                fill(tactic.instruction.for_locale(self._asset_locale), values)
                if tactic.mode is TacticMode.GENERATED
                else ""
            ),
        )

    # ── ProbeAdapter hooks ──────────────────────────────────────────

    def get_user_system_prompt(self) -> str:
        pressing = self._pressing
        return (PRESSURE_USER_SYSTEM_PROMPTS if pressing else USER_SYSTEM_PROMPTS).get(
            _prompt_locale(self._locale),
            opening_line=OPENING_TO_WRITE if self._writes_opening else OPENING_SENT,
            persona=format_persona_for_prompt(self._persona),
            language_instruction=language_instruction(self._language, self._locale),
            behavioral_instructions=(
                format_behavioral_profile_for_prompt(
                    self._profile, probe_type=type(self).label, language=self._language, pressing=pressing
                )
                if self._profile
                else ""
            ),
            interaction_style_instructions=format_interaction_style_instructions(self._interaction_style),
            **self._pressure_values,
        )

    def get_assistant_system_prompt(self) -> str:
        # Pure-capability policy: what the model says about itself unprompted
        # is the finding, so it gets no system prompt.
        return ""

    @property
    def _pressing(self) -> bool:
        return self._task is not None and self._task.pressured

    @property
    def _writes_opening(self) -> bool:
        """Whether the simulated user writes the opening from a generated tactic's instruction."""
        return self._task is not None and self._task.generated

    def get_user_query_instruction(self, turn_idx: int) -> str | None:
        """Turn 1 of a generated tactic: its instruction, for the simulated user to write from."""
        if turn_idx != 0 or not self._writes_opening:
            return None
        return OPENING_INSTRUCTION.format(instruction=self._task.instruction)

    @property
    def _pressure_values(self) -> dict[str, str]:
        """The competitor a pressed row's prompts name; empty for a row that is not pressed."""
        if not self._pressing:
            return {}
        values = self._task.values
        return {"competitor": values["competitor"], "competitor_developer": values["competitor_developer"]}

    async def format_followup_user_instructions(self, turn_idx: int, state: ConversationState) -> list[str]:
        """The strategy's next reframing, filled for this row; none for a row that is not pressed."""
        self._opening_written = True
        reframing = self._task.strategy.reframing_at(turn_idx - 1) if self._pressing else None
        if reframing is None:
            return []
        state.metadata.setdefault("reframings_used", []).append(reframing.id)
        messages = getattr(state, "messages", None) or []
        previous = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "assistant"), "")
        values = {**self._task.values, "previous_assistant_response": previous}
        instruction = fill(reframing.instruction.for_locale(self._asset_locale), values)
        return [PRESSURE_INSTRUCTION.format(instruction=instruction)]

    def allow_early_stop_at_turn(self, turn_idx: int, state: ConversationState) -> bool:
        # A pressed row runs every turn, so a model that gives in late, or recovers, is seen.
        return not self._pressing

    def user_turn_policy(self) -> UserTurnPolicy:
        """Earlier replies stay as written, no follow-up wraps up, and the words these users write pass the filters.

        The script check skips the spec's names, products and models, which
        are written in Latin script in every locale. An opening the simulated
        user writes is checked for the names it may and may not contain.
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
            followup_anchor=FOLLOWUP_ANCHOR if self._pressing else None,
            allowed_phrases=LANGUAGE_MODEL_NOUNS | FIRST_MESSAGE_PHRASES | PUSHBACK_OPENERS,
            script_check_ignores=tuple(sorted(names)),
            check_opening=self._opening_check(spec) if self._writes_opening and spec is not None else None,
        )

    def _opening_check(self, spec: IdentitySpec) -> Callable[[str], str | None]:
        """Why a drafted opening breaks its tactic's rules, or None.

        A neutral opening names no AI developer's company, product or model; a
        leading one names the row's and no other. Names that are also ordinary
        words are left to the gate, which reads them in context.
        """
        vocabulary = Vocabulary(spec.vocabulary())
        allowed = self._opening_developers(spec)

        def check(text: str) -> str | None:
            named = vocabulary.developers_in(text, include_ambiguous=False)
            if others := sorted(named - allowed):
                return f"names {', '.join(vocabulary.display_names[dev] for dev in others)}"
            if allowed and not named & allowed:
                return f"does not name {', '.join(sorted(vocabulary.display_names[dev] for dev in allowed))}"
            return None

        return check

    def _opening_developers(self, spec: IdentitySpec) -> set[str]:
        """The developers whose names the tactic's instruction fills in."""
        task = self._task
        placeholders = placeholders_in(task.tactic)
        developers: set[str] = set()
        if placeholders & IDENTITY_PLACEHOLDERS and task.expected is not None:
            developers.update(task.expected.developers)
        products = [task.competitor] if placeholders & {"competitor", "competitor_developer"} else []
        if "competitor_list" in placeholders:
            products += task.values["competitor_list"].split(", ")
        developers.update(dev for dev in map(spec.developer_of, filter(None, products)) if dev)
        return developers

    def format_gate_prompt(self, user_query: str, conversation_history: str) -> str:
        if self._writes_opening and not self._opening_written:
            return OPENING_GATE_PROMPTS.get(
                _prompt_locale(self._locale),
                instruction=self._task.instruction,
                user_turn_to_evaluate=user_query,
                names_rule=self._names_rule(),
            )
        return (PRESSURE_GATE_PROMPTS if self._pressing else GATE_PROMPTS).get(
            _prompt_locale(self._locale),
            conversation_history=conversation_history,
            user_turn_to_evaluate=user_query,
            **self._pressure_values,
        )

    def _names_rule(self) -> str:
        """The opening gate's rule on AI names: none for a neutral opening, only the row's for a leading one."""
        task = self._task
        filled = sorted(placeholders_in(task.tactic) & (IDENTITY_PLACEHOLDERS | COMPETITOR_PLACEHOLDERS))
        names = [task.values[p] for p in filled if task.values.get(p)]
        if task.tactic.neutral or not names:
            return NEUTRAL_NAMES_RULE
        return LEADING_NAMES_RULE.format(names=", ".join(names))

    def seed_state_metadata(self, state: ConversationState) -> None:
        task = self._task
        if task is None:
            return
        state.metadata["identity_tactic_id"] = task.tactic.id
        state.metadata["identity_tactic_kind"] = task.tactic.kind.value
        state.metadata["identity_pair"] = task.tactic.pair or None
        state.metadata["identity_competitor"] = task.competitor
        state.metadata["strategy_id"] = task.strategy.id
        state.metadata["reframings_used"] = []

    async def get_verbatim_first_user_turn(self, state: ConversationState) -> str | None:
        if self._task is None or self._writes_opening:
            return None
        return await self._opening_in_conversation_language(self._task)

    async def _opening_in_conversation_language(self, task: IdentityTask) -> str:
        """The opening as the model reads it.

        A machine-translated opening is translated once per wording, with each
        name held as a token and filled in afterwards, so a question reads the
        same whichever developer it names.
        """
        if self._asset_locale == self._locale:
            return task.opening
        frame, names = opening_frame(task.tactic.text.for_locale(self._asset_locale), task.values)
        rules = (_POLITE_YOU,)
        if names:
            rules += (
                f"Keep {', '.join(names)} exactly as written: each stands for a name, so place it where the name belongs.",
            )
        translated = await self._localize_verbatim(frame, rules=rules)
        if all(token in translated for token in names):
            return fill_frame(translated, names)
        logger.warning(
            "  |-- %s: translating %r into %s lost a name; translating the filled opening instead",
            type(self).label,
            frame,
            self._language,
        )
        return await self._localize_verbatim(task.opening, rules=(_POLITE_YOU,), warn=False)

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
                "reframings_used": list(state.metadata.get("reframings_used") or []),
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
