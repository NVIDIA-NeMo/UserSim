# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe template — BaseProbe only, no asset bank.

Use this shape when the probe asks general questions of the assistant
without needing reviewer-curated content (the ``general_open_ended``
and ``general_educational`` probes use this shape).

What the unified ``ConversationLoop`` provides for you:
- Multi-turn back-and-forth with the user-LLM.
- Fourth-wall pre-filter, role-violation telemetry.
- User-judge gate retries (default ``"skip"`` on exhaustion).
- Per-assistant-turn quality judge.
- Frustration escalation, early-stop heuristics.

What you provide:
- The user-agent system prompt (how the simulated user behaves).
- The assistant system prompt (empty per pure-capability-test policy
  — see conversation-plugin/README.md "Assistant under test").

Total LOC: ~30 (this file). The ``general_open_ended`` probe is a
real example at ~50 LOC.
"""

from __future__ import annotations

from usersim.engine.core.probes import BaseProbe, register_probe

# After you copy this file into ``src/usersim/engine/probes/<your_probe>/``,
# also copy ``prompts.py`` next to it and change this import to a
# relative one: ``from .prompts import ...``.
from usersim.engine.templates.probe.prompts import (  # type: ignore[import-not-found]
    DEMO_GATE_PROMPT_TEMPLATE,
    DEMO_USER_AGENT_PROMPTS,
)


@register_probe(
    family="demo_general",
    prompt_version="v1.0",
    variants=("default",),
)
class DemoGeneralProbe(BaseProbe):
    """Smallest viable probe — no asset bank, default everything else."""

    label = "demo_general"

    def get_user_system_prompt(self) -> str:
        return DEMO_USER_AGENT_PROMPTS.get(
            self._locale,
            persona=self._persona.get("first_name", "the user"),
        )

    def get_assistant_system_prompt(self) -> str:
        # Pure-capability-test policy. Empty system prompt;
        # the assistant under test gets only conversation history.
        # See conversation-plugin/README.md "Assistant under test".
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
