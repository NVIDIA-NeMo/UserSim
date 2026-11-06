# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Educational probe (tutoring, explanation, practice).

A thin probe class: ``EducationalProbe`` defines the user-agent prompt and
delegates everything else to ``BaseProbe`` and ``ConversationLoop``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from usersim.engine.probes.general_educational.prompts import (
    _SCENARIO,
    assistant_system_prompt,
    user_agent_system_prompt,
    user_judge_turn_prompt,
)
from usersim.extension import (
    SHIPPED_LOCALES,
    BaseProbe,
    PreviewField,
    format_behavioral_profile_for_prompt,
    format_disclosure_instructions,
    format_interaction_style_instructions,
    format_persona_for_prompt,
    language_instruction,
    load_prompt_file,
    load_seeds,
    register_probe,
    render_prompt,
    theme_problems,
)


@register_probe
class EducationalProbe(BaseProbe):
    """Educational / tutoring probe.

    Test the assistant's tutoring / explanation / practice behavior
    with persona-grounded, subject-conditioned conversations. No
    tools; no curated asset bank; quality judgment lives in the
    out-of-sim trajectory evaluator.
    """

    label = "general_educational"
    # Bump on any semantic change to the probe's prompts or task derivation.
    # One variant: the subject comes from the row's theme.
    prompt_version = "v1.0"
    assets_root = Path(__file__).parent / "assets"
    description = (
        "Tests tutoring: a persona-grounded learner asks about a subject, and the probe "
        "looks at whether explanations are accurate, pitched at the learner's level and "
        "scaffolded with examples, without condescension."
    )
    preview_fields = (PreviewField("theme", "Subject"),)

    @classmethod
    def themes(cls, locale: str) -> list[dict[str, Any]]:
        """The subjects in ``locale``'s seed file, or in the ``en_US`` file when the locale has none."""
        return load_seeds(locale, cls.label, "subjects")

    @classmethod
    def asset_problems(cls) -> list[str]:
        """Subjects for every shipped locale, and the prompts file the prompt defaults would mask."""
        problems = theme_problems(cls, SHIPPED_LOCALES)
        if not load_prompt_file(_SCENARIO):
            problems.append(f"{_SCENARIO}/prompts.yaml is missing or empty")
        return problems

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        theme = self.theme()
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
