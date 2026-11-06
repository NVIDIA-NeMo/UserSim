# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe template — BaseProbe only, no asset bank.

Use this shape when the probe sets topics for the simulated user to
explore, without needing reviewer-curated content (the
``general_open_ended`` and ``general_educational`` probes use this
shape). Each row draws one topic from ``themes(locale)``, and the user
explores it in the persona's own words: the probe sets the area, never
the message.

What the unified ``ConversationLoop`` provides for you:
- Multi-turn back-and-forth with the user-LLM.
- Fourth-wall pre-filter, role-violation telemetry.
- User-judge gate retries (default ``"skip"`` on exhaustion).
- Per-assistant-turn quality judge.
- Frustration escalation, early-stop heuristics.

What you provide:
- ``themes(locale)``: the topics a run draws from, one per row, each a
  mapping with at least a ``description``.
- The user-agent system prompt: one instruction in ``prompts.yaml``
  (copy ``prompts.yaml`` to ``assets/<your_probe>/``), filled with the
  persona and the row's topic, plus ``language_instruction`` so the user
  writes in the row's language.
- The assistant system prompt (empty per pure-capability-test policy;
  see docs/engine/README.md "Assistant under test").

Total LOC: ~30 (this file). The ``general_open_ended`` probe is a
real example at ~50 LOC.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# After you copy this file into ``src/usersim/engine/probes/<your_probe>/``,
# also copy ``prompts.py`` next to it and import from
# ``usersim.engine.probes.<your_probe>.prompts``.
from usersim.engine.templates.probe.prompts import (  # type: ignore[import-not-found]
    DEMO_GATE_PROMPT_TEMPLATE,
)
from usersim.extension import (
    BaseProbe,
    format_persona_for_prompt,
    get_prompt,
    language_instruction,
    register_probe,
    render_prompt,
)


@register_probe
class DemoGeneralProbe(BaseProbe):
    """Smallest viable probe — no asset bank, default everything else."""

    label = "demo_general"
    assets_root = Path(__file__).parent / "assets"  # holds demo_general/, beside this module

    @classmethod
    def themes(cls, locale: str) -> list[dict]:
        # One file of topics per locale, written in its language. A run
        # draws one per row and stores it in the row's ``theme`` column.
        return yaml.safe_load((cls.assets_dir() / locale / "themes.yaml").read_text(encoding="utf-8"))

    def get_user_system_prompt(self) -> str:
        # The topic bounds the conversation; the persona picks the question.
        prompt = render_prompt(
            get_prompt(self.label, "user_system_prompt"),
            self._data,
            persona=format_persona_for_prompt(self._persona),
            topic=self.theme()["description"],
        )
        return f"{prompt}\n\n{language_instruction(self._language, self._locale)}"

    def get_assistant_system_prompt(self) -> str:
        # Pure-capability-test policy. Empty system prompt;
        # the assistant under test gets only conversation history.
        # See docs/engine/README.md "Assistant under test".
        return ""

    def format_gate_prompt(
        self,
        user_query: str,
        conversation_history: str,
    ) -> str:
        # Optional override — the BaseProbe default is a generic
        # role-violation rubric. Override when the gate should be
        # grounded in probe-specific quality criteria.
        return DEMO_GATE_PROMPT_TEMPLATE.format(
            user_turn=user_query,
            history=conversation_history,
        )
