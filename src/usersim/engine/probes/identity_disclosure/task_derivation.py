# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which tactic, strategy and competitor a row gets, and the text it opens with.

Picks are deterministic in ``(persona, spec digest, seed)``: the same persona
sees the same cell for a given spec and run seed, and a spec edit reshuffles
them. Each slot draws from its own stream, so the tactic a persona gets does
not constrain its strategy or competitor. A panel can pin the tactic and the
strategy through the ``identity_tactic_id`` and ``identity_strategy_id``
input columns instead.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Mapping

from usersim.engine.core.identity_spec import (
    ExpectedIdentity,
    IdentitySpec,
    Strategy,
    Tactic,
    TacticMode,
)

#: Placeholders that need the expected identity of the model under test.
IDENTITY_PLACEHOLDERS = frozenset({"true_developer", "true_model"})
#: Placeholders that need a competitor.
COMPETITOR_PLACEHOLDERS = frozenset({"competitor", "competitor_developer", "competitor_list"})

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


class IdentityTaskError(ValueError):
    """A row cannot be given a tactic, strategy or competitor as configured."""


@dataclass(frozen=True)
class IdentityTask:
    """Everything one row asks with, and what it is graded against."""

    #: The tactic id; ``BankBackedProbe`` names the task by it in warnings.
    id: str
    placeholder: bool
    tactic: Tactic
    strategy: Strategy
    #: The competitor the opening or the pressure names, or None when neither does.
    competitor: str | None
    #: None when the model under test cannot be identified.
    expected: ExpectedIdentity | None
    #: The opening turn in the asset locale, placeholders filled; empty for a generated tactic.
    opening: str
    #: What the row's placeholders stand for, for the opening and each reframing.
    values: Mapping[str, str] = field(default_factory=dict)
    #: What the simulated user is told to write for a generated tactic, placeholders filled.
    instruction: str = ""

    @property
    def pressured(self) -> bool:
        """Whether follow-up turns push back, following the strategy's reframings."""
        return bool(self.strategy.reframings)

    @property
    def generated(self) -> bool:
        """Whether the simulated user writes the opening rather than the spec."""
        return self.tactic.mode is TacticMode.GENERATED


def placeholders_in(tactic: Tactic) -> set[str]:
    """Every placeholder the tactic's opening uses: its text, or a generated tactic's instruction."""
    localized = tactic.instruction if tactic.mode is TacticMode.GENERATED else tactic.text
    if localized is None:
        return set()
    return {name for text in localized.renderings.values() for name in _PLACEHOLDER.findall(text)}


def strategy_placeholders(strategy: Strategy) -> set[str]:
    """Every placeholder any rendering of the strategy's reframings uses."""
    return {
        name
        for reframing in strategy.reframings
        for text in reframing.instruction.renderings.values()
        for name in _PLACEHOLDER.findall(text)
    }


def _stream(persona_key: str, spec: IdentitySpec, seed: int | None, slot: str) -> random.Random:
    material = f"{persona_key}|{spec.digest}|{'' if seed is None else seed}|{slot}"
    return random.Random(int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big"))


def select_tactic(
    spec: IdentitySpec,
    persona_key: str,
    *,
    seed: int | None,
    override: str | None = None,
    neutral_only: bool = False,
) -> Tactic:
    """The row's opening tactic.

    ``neutral_only`` restricts the draw to neutral tactics that need neither
    the expected identity nor a competitor, for a row whose model under test
    cannot be identified.
    """
    eligible = list(spec.tactics)
    if neutral_only:
        needs = IDENTITY_PLACEHOLDERS | COMPETITOR_PLACEHOLDERS
        eligible = [t for t in eligible if t.neutral and not placeholders_in(t) & needs]
    if override is not None and not neutral_only:
        tactic = spec.tactic_by_id(override)
        if tactic is None:
            raise IdentityTaskError(
                f"identity_tactic_id {override!r} is not a tactic in spec {spec.version}; "
                f"choose one of {[t.id for t in eligible]}"
            )
        return tactic
    if not eligible:
        kind = "neutral tactic" if neutral_only else "tactic"
        raise IdentityTaskError(f"spec {spec.version} has no {kind} to open with")
    return _stream(persona_key, spec, seed, "tactic").choice(eligible)


def select_strategy(
    spec: IdentitySpec,
    persona_key: str,
    *,
    seed: int | None,
    override: str | None = None,
    max_turns: int = 1,
) -> Strategy:
    """The row's pressure strategy.

    A run with follow-up turns draws from the strategies that have
    reframings, so every row is pressed; a single-turn run, which has no
    follow-ups to press with, draws from those without.
    """
    if override is not None:
        strategy = spec.strategy_by_id(override)
        if strategy is None:
            raise IdentityTaskError(
                f"identity_strategy_id {override!r} is not a strategy in spec {spec.version}; "
                f"choose one of {[s.id for s in spec.strategies]}"
            )
        return strategy
    pressing = [s for s in spec.strategies if s.reframings]
    calm = [s for s in spec.strategies if not s.reframings]
    pool = pressing if max_turns > 1 and pressing else calm or pressing
    if not pool:
        raise IdentityTaskError(f"spec {spec.version} has no strategy")
    return _stream(persona_key, spec, seed, "strategy").choice(pool)


def select_competitor(
    spec: IdentitySpec,
    pool: tuple[str, ...],
    persona_key: str,
    *,
    seed: int | None,
) -> str | None:
    """The competitor product the row may name, or None when the pool is empty."""
    if not pool:
        return None
    return _stream(persona_key, spec, seed, "competitor").choice(pool)


def placeholder_values(
    spec: IdentitySpec,
    expected: ExpectedIdentity | None,
    *,
    competitor: str | None,
    pool: tuple[str, ...],
    language: str,
) -> dict[str, str]:
    """The values a row's text placeholders take."""
    true_developer = spec.developers[expected.developers[0]].display_name if expected else ""
    competitor_developer = ""
    if competitor is not None:
        dev_id = spec.developer_of(competitor)
        competitor_developer = spec.developers[dev_id].display_name if dev_id else competitor
    return {
        "true_developer": true_developer,
        "true_model": expected.model_names[0] if expected and expected.model_names else true_developer,
        "competitor": competitor or "",
        "competitor_developer": competitor_developer,
        "competitor_list": ", ".join(pool),
        "other_language": "French" if language == "English" else "English",
        "previous_assistant_response": "",
    }


def fill(text: str, values: Mapping[str, str]) -> str:
    """Substitute known placeholders and leave any other braces untouched."""
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def opening_frame(text: str, values: Mapping[str, str]) -> tuple[str, dict[str, str]]:
    """``text`` with each known placeholder held as a numbered token, and the name each token stands for.

    Tokens are numbered in order of first appearance rather than named after
    the placeholder, so openings that differ only in which name they ask
    about share one frame, and so one translation.
    """
    tokens: dict[str, str] = {}
    names: dict[str, str] = {}

    def hold(match: re.Match[str]) -> str:
        placeholder = match.group(1)
        if placeholder not in values:
            return match.group(0)
        if placeholder not in tokens:
            tokens[placeholder] = f"⟦{len(tokens) + 1}⟧"
            names[tokens[placeholder]] = values[placeholder]
        return tokens[placeholder]

    return _PLACEHOLDER.sub(hold, text), names


def fill_frame(text: str, names: Mapping[str, str]) -> str:
    """``text`` with each token from :func:`opening_frame` replaced by its name."""
    for token, name in names.items():
        text = text.replace(token, name)
    return text
