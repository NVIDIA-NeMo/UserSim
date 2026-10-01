# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Educational probe (tutoring, explanation, practice).

A thin probe class plus a shim function: ``EducationalProbe`` defines
the user-agent prompt and delegates everything else to ``BaseProbe`` +
``ConversationLoop``; ``simulate_general_educational`` is the back-compat
shim that constructs the probe and runs it through ``run_dispatch``.
"""

from __future__ import annotations

from typing import Any

from usersim.engine.core.behavioral import (
    format_behavioral_profile_for_prompt,
    format_disclosure_instructions,
    format_interaction_style_instructions,
)
from usersim.engine.core.messages import _parse_theme
from usersim.engine.core.persona import format_persona_for_prompt
from usersim.engine.core.probes import BaseProbe, register_probe
from usersim.engine.core.prompt_loader import render_prompt
from usersim.engine.core.simulation import language_instruction
from usersim.engine.probes.general_educational.prompts import (
    assistant_system_prompt,
    user_agent_system_prompt,
    user_judge_turn_prompt,
)

# ── Probe metadata ──────────────────────────────────────────────────
# Bumping ``PROMPT_VERSION`` invalidates prior evaluator rows on
# re-run, since the dedup key includes it. Bump on any semantic change
# to the probe's prompts or task derivation.

PROBE_FAMILY: str = "general_educational"
PROBE_VARIANTS: list[str] = ["default"]  # subject-conditioned via theme; single variant today
PROMPT_VERSION: str = "v1.0"


@register_probe(family=PROBE_FAMILY, prompt_version=PROMPT_VERSION, variants=tuple(PROBE_VARIANTS))
class EducationalProbe(BaseProbe):
    """Educational / tutoring probe.

    Test the assistant's tutoring / explanation / practice behavior
    with persona-grounded, subject-conditioned conversations. No
    tools; no curated asset bank; quality judgment lives in the
    out-of-sim trajectory evaluator.
    """

    label = "general_educational"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        theme = _parse_theme(self._data.get(getattr(self._cfg, "theme_column", "theme"), "{}"))
        # Parsed once; the prompt itself is rendered per call in
        # get_user_system_prompt so all three prompt hooks resolve alike.
        self._subject = theme.get("description", theme.get("type", "general subject"))
        self._disclosure_style = self._data.get("disclosure_style", "upfront")

    def get_user_system_prompt(self) -> str:
        return render_prompt(
            user_agent_system_prompt(),
            self._data,
            persona=format_persona_for_prompt(self._persona),
            subject=self._subject,
            language_instruction=language_instruction(self._language, self._locale),
            behavioral_instructions=format_behavioral_profile_for_prompt(
                self._profile,
                language=self._language,
            ),
            disclosure_instructions=format_disclosure_instructions(
                self._disclosure_style,
            ),
            interaction_style_instructions=format_interaction_style_instructions(
                self._interaction_style,
            ),
        )

    def get_assistant_system_prompt(self) -> str:
        # Empty as shipped; see the accessor's docstring.
        return render_prompt(assistant_system_prompt(), self._data)

    def format_gate_prompt(
        self,
        user_query: str,
        conversation_history: str,
    ) -> str:
        return render_prompt(
            user_judge_turn_prompt(),
            self._data,
            conversation_history=conversation_history,
            user_turn_to_evaluate=user_query,
        )


async def simulate_general_educational(
    models: dict,
    data: dict,
    persona: dict,
    profile: dict,
    locale: str,
    language: str,
    cfg: Any,
    **kwargs: Any,
) -> dict:
    """Thin shim for callers that import ``simulate_general_educational`` directly.

    See ``usersim.engine.probes.general_open_ended.generator.simulate_general_open_ended``
    for the rationale; preserved for backward-compat with the dispatcher
    bootstrap and per-probe tests.
    """
    probe = EducationalProbe(
        persona=persona,
        locale=locale,
        language=language,
        models=models,
        cfg=cfg,
        provenance=kwargs.get("provenance"),
        profile=profile,
        data=data,
        outcome_builder=kwargs.get("outcome_builder"),
    )
    return await probe.run_dispatch(models=models, data=data, cfg=cfg)
