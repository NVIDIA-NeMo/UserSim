# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe template — verbatim turn-1 from a curated asset bank.

Use this shape when the probe needs to ask a SPECIFIC question
verbatim (so false-premise wording, exact phrasing for matched-pair
joins, or reviewer-authored content survives unchanged). The
``sov_ai_facts``, ``sov_ai_multilingual_parity``, and
``safety_chat_pressure`` probes all use this shape.

What ``BankVerbatimMixin`` does:
- Wires the optional ``get_verbatim_first_user_turn`` hook to
  return your bank-derived text (you override
  ``get_verbatim_first_user_turn``; the mixin is mostly a marker
  for intent + future shared logic).
- The ``ConversationLoop`` checks this hook at turn-1 and bypasses
  the generate-and-gate path: the curated question is injected
  directly, with a ``USER_QUERY_GATE`` trace marked ``rating="bypassed"``.

What ``BankBackedProbe`` provides:
- Calls your ``bank_loader`` once (with locale fallback to no-arg).
- Catches load failures and re-raises as ``BankLoadError`` (the
  shim catches and returns SCENARIO_ABORTED).
- Calls your ``derive_task`` to pick the asset for this trajectory.
- Emits ``placeholder_warning_kind`` if the picked asset is
  placeholder.
- Pins ``provenance.bank_version[bank_version_key or locale]``.

What you provide:
- ``derive_task(persona, bank, *, cfg)`` — returns an object with
  at least ``.id`` and ``.placeholder``, and probe-specific fields
  the prompts read.
- ``get_verbatim_first_user_turn(state)`` — returns the verbatim
  user-turn-1 text AND seeds metadata side-channels.
- ``should_succeed(state)`` — assert the side-channel join key is
  present.
- ``build_result_extras(state)`` — surface probe_variant + the
  top-level columns the matching scorer reads.

Total LOC: ~80. ``sov_ai_facts`` is a real example at ~190 LOC
(190 vs 30 because facts is multi-locale + question-type-conditioned
+ has a 2nd turn nudge — most of that is configuration, not logic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from usersim.engine.core.outcomes import WarningKind
from usersim.engine.core.probes import (
    BankBackedProbe,
    BankVerbatimMixin,
    register_probe,
)

# After you copy this file into ``src/usersim/engine/probes/<your_probe>/``,
# also copy ``prompts.py`` next to it and change this import to ``from .prompts``.
from usersim.engine.templates.probe.prompts import DEMO_USER_AGENT_PROMPTS  # type: ignore[import-not-found]


@dataclass
class _DemoTask:
    """Substrate-shaped task — must have .id and .placeholder."""

    id: str
    placeholder: bool
    question: str
    category: str


def _demo_bank_loader(locale: str = "en_US"):
    """Stub loader — your real probe loads a YAML bank here.

    Real probes use a module-level alias indirection (the
    ``_bank_loader`` pattern from sov_ai_facts) so test patching
    works cleanly.
    """
    raise NotImplementedError("replace with your real bank loader")


@register_probe(
    family="demo_bank_verbatim",
    prompt_version="v1.0",
    variants=("default",),
)
class DemoBankVerbatimProbe(BankVerbatimMixin, BankBackedProbe):
    """Verbatim-turn-1 probe driven by a curated asset bank.

    NOTE the MRO: mixin BEFORE BankBackedProbe. The mixin's hooks
    must take precedence over BaseProbe defaults.
    """

    label = "demo_bank_verbatim"
    bank_loader = staticmethod(_demo_bank_loader)
    placeholder_warning_kind = WarningKind.USED_PLACEHOLDER_FACT
    bank_version_key: str | None = None  # locale-keyed (inclusion convention)

    def derive_task(
        self,
        persona: dict,
        bank: Any,
        *,
        cfg: Any,
    ) -> _DemoTask | None:
        # Your real probe filters by persona-tag intersection,
        # excludes already-probed ids, deterministic salt, etc.
        # See sov_ai_facts/task_derivation.py for the canonical pattern.
        return _DemoTask(
            id="DEMO-001",
            placeholder=True,
            question="Demo question?",
            category="demo",
        )

    def get_user_system_prompt(self) -> str:
        return DEMO_USER_AGENT_PROMPTS.get(
            self._locale,
            persona=self._persona.get("first_name", "the user"),
        )

    def get_assistant_system_prompt(self) -> str:
        return ""  # Pure-capability-test policy.

    def get_verbatim_first_user_turn(self, state: Any) -> str:
        # Seed all the metadata keys the scorer joins on. Use
        # setdefault if the substrate's seed_state_metadata might
        # have already populated them.
        state.metadata["facts_probed"] = [self._task.id]
        state.metadata["fact_category"] = self._task.category
        state.metadata["bank_id"] = getattr(self._bank, "bank_id", "<unknown>")
        state.metadata["bank_version"] = getattr(
            self._bank,
            "bank_version",
            "<unknown>",
        )
        return self._task.question

    def should_succeed(self, state: Any) -> bool:
        return bool(state.metadata.get("facts_probed"))

    def build_result_extras(self, state: Any) -> dict:
        extras = super().build_result_extras(state)
        if self._task is not None:
            extras["probe_variant"] = self._task.category
            extras["sovereign_facts_probed"] = list(state.metadata.get("facts_probed") or [])
        return extras
