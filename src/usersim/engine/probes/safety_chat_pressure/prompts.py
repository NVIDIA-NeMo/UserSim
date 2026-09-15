# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts for the safety_chat_pressure probe.

Two surfaces shipped here:

- :data:`ASSISTANT_SYSTEM_PROMPT` — the assistant's system prompt.
  Deliberately lean: the probe targets the assistant's refusal
  behaviour under pressure; an elaborate instruction would
  itself be a confounder. The assistant gets a generic "be helpful
  and follow your usual guidelines" framing and the test surfaces
  what its trained safety stance actually does under conversational
  pressure.

- :func:`build_user_agent_followup_messages` — assembles the message
  list passed to the user-agent on every turn ≥ 2: the strategy's
  ``system_prompt_template`` (formatted against persona / target /
  initial prompt), the conversation so far, and the picked
  reframing's ``instruction`` (formatted against the assistant's
  previous response). The reframing's instruction is the *last*
  message the user-agent reads — the user-agent's response to it is
  the next user turn.

Capitulation detection no longer happens through user-LLM structured
output (that proved unreliable due to safety-bias and multi-turn
context contamination). It now lives in
:mod:`usersim.engine.probes.safety_chat_pressure.classifier` —
a dedicated comparator-shaped ``summary_model`` call that runs on
the assistant's response with no pressure framing.

Both surfaces are English-only because the pressure-strategy bank is
English-only by design (see ``assets/safety_chat_pressure/SCHEMA.md``);
non-English pressure attempts are a separate test surface that a
future locale-aware bank can carry.
"""

from __future__ import annotations

from typing import Any, Dict, List

from usersim.engine.core.persona import format_persona_for_prompt
from usersim.engine.core.pressure_bank import (
    Reframing,
    Strategy,
    TargetRequest,
)


# PURE-CAPABILITY-TEST POLICY: empty by design.
#
# The framework tests the assistant model's out-of-the-box behavior;
# adding policy text ("follow your safety guidelines", "decline harmful
# requests", etc.) would prime the model and contaminate the signal we
# are trying to measure (does the model refuse without being told to?).
# All probes follow the same convention — see
# conversation-plugin/README.md "Assistant under test: pure-capability
# policy" for the cross-cutting rationale.
ASSISTANT_SYSTEM_PROMPT: str = ""


def build_user_agent_system_prompt(
    *,
    strategy: Strategy,
    target: TargetRequest,
    persona: Dict[str, Any],
    locale: str,
) -> str:
    """Format the strategy's system_prompt_template against the trajectory,
    using the locale-appropriate rendering for every localized field.

    The template's required placeholders (``{persona}`` /
    ``{target_description}`` / ``{initial_prompt}``) are validated at
    bank-load time **on every locale rendering**, so .format() here is
    safe regardless of which locale's rendering we picked.

    ``locale`` controls which rendering is selected for each
    :class:`LocalizedText` field (``strategy.system_prompt_template``,
    ``target.description``, ``target.initial_prompt``). Missing
    locales fall back to the bank's default locale (``en_US``) per
    :meth:`LocalizedText.for_locale` semantics.
    """
    return strategy.system_prompt_template.for_locale(locale).format(
        persona=format_persona_for_prompt(persona),
        target_description=target.description.for_locale(locale),
        initial_prompt=target.initial_prompt.for_locale(locale),
    )


def build_reframing_instruction(
    *,
    reframing: Reframing,
    previous_assistant_response: str,
    locale: str,
) -> str:
    """Format the reframing's instruction against the assistant's previous turn.

    The required placeholder (``{previous_assistant_response}``) is
    validated at bank-load time. ``locale`` selects the per-locale
    rendering of the instruction template.
    """
    return reframing.instruction.for_locale(locale).format(
        previous_assistant_response=previous_assistant_response or "(empty)",
    )


def build_user_agent_followup_messages(
    *,
    strategy: Strategy,
    target: TargetRequest,
    persona: Dict[str, Any],
    conversation_so_far: List[Dict[str, Any]],
    reframing: Reframing,
    previous_assistant_response: str,
    locale: str,
) -> List[Dict[str, Any]]:
    """Assemble the message list the user-agent receives on a follow-up turn.

    Order matters — the reframing instruction is appended **last** as
    a system message so it's the most recent context the user-agent
    sees when composing its response.

    ``locale`` is propagated to every locale-aware sub-builder so the
    user-agent receives instructions in the persona's language.
    """
    return [
        {
            "role": "system",
            "content": build_user_agent_system_prompt(
                strategy=strategy, target=target, persona=persona,
                locale=locale,
            ),
        },
        *conversation_so_far,
        {
            "role": "system",
            "content": build_reframing_instruction(
                reframing=reframing,
                previous_assistant_response=previous_assistant_response,
                locale=locale,
            ),
        },
    ]
