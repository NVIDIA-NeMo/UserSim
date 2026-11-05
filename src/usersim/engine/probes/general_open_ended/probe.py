# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open-ended probe (info-seeking, Q&A, recommendations).

A thin probe class: ``OpenEndedProbe`` defines the user-agent prompt and
delegates everything else to ``BaseProbe`` and ``ConversationLoop``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from usersim.engine.probes.general_open_ended.prompts import (
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
)

# ── Probe metadata ──────────────────────────────────────────────────
# Bumping ``PROMPT_VERSION`` invalidates prior evaluator rows on
# re-run, since the dedup key includes it.

PROBE_VARIANTS: list[str] = ["default"]  # topic-conditioned via theme; single variant today
PROMPT_VERSION: str = "v1.0"


@register_probe(prompt_version=PROMPT_VERSION, variants=tuple(PROBE_VARIANTS))
class OpenEndedProbe(BaseProbe):
    """Open-ended conversation probe.

    Test the assistant's general info-seeking / Q&A / recommendation
    behavior with persona-grounded, theme-conditioned conversations.
    No tools; no curated asset bank; quality judgment lives in the
    out-of-sim trajectory evaluator.
    """

    label = "general_open_ended"
    assets_root = Path(__file__).parent / "assets"
    description = (
        "Tests information-seeking conversations across diverse personas: accurate, "
        "coherent answers pitched to the user's apparent expertise, and a clarifying "
        "question rather than an information dump when a request is vague."
    )
    preview_fields = (PreviewField("theme", "Topic"),)

    @classmethod
    def themes(cls, locale: str) -> list[dict[str, Any]]:
        """The topics in ``locale``'s seed file, or in the ``en_US`` file when the locale has none."""
        return load_seeds(locale, cls.label, "topics")

    @classmethod
    def asset_problems(cls) -> list[str]:
        """Topics for every shipped locale, and the prompts file the prompt defaults would mask."""
        problems = []
        for locale in SHIPPED_LOCALES:
            try:
                cls.themes(locale)
            except FileNotFoundError as error:
                problems.append(f"no topics for {locale}: {error}")
        if not load_prompt_file(_SCENARIO):
            problems.append(f"{_SCENARIO}/prompts.yaml is missing or empty")
        return problems

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        theme = self.theme()
        # Parsed once; the prompt itself is rendered per call in
        # get_user_system_prompt so all three prompt hooks resolve alike.
        self._topic = theme.get("description", theme.get("type", "general topic"))
        self._disclosure_style = self._data.get("disclosure_style", "upfront")

    def get_user_system_prompt(self) -> str:
        return render_prompt(
            user_agent_system_prompt(),
            self._data,
            persona=format_persona_for_prompt(self._persona),
            topic=self._topic,
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
