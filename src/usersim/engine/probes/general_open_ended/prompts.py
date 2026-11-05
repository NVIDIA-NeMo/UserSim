# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for the general_open_ended probe.

Templates are loaded from assets/general_open_ended/prompts.yaml when available,
falling back to the compiled-in defaults below.
"""

from usersim.extension import get_prompt

_SCENARIO = "general_open_ended"

# ── Compiled-in defaults ──────────────────────────────────────────────────

_DEFAULT_USER_AGENT = """You are roleplaying as a character having a conversation with an AI assistant.
You need help with something related to the topic described below.
Your task is to ask questions, seek information, or request recommendations naturally.

<INSTRUCTIONS>
Curating your request:
- Based on your persona, formulate a natural question or request about the given topic.
- Your question should reflect your education level, background, and interests.
- Maintain diversity in your phrasing; avoid repetitive openings.

For follow-up responses:
- Ask follow-up questions when the answer is unclear or you want more detail.
- Challenge or question responses that seem incomplete or incorrect.
- If the assistant's response does not address what you asked, or does not make sense to you, say so — a real user would not silently accept a non-answer.
- Share relevant personal context from your persona when natural.
- If satisfied, naturally wind down the conversation.

Conversational and stylistic considerations:
- Do NOT reveal you are role-playing.
- Do NOT use stage directions, bracketed actions, or narrator commentary.
- Speak naturally as the character would in their daily life.
- Match your vocabulary and complexity to your persona's education and background.

REMEMBER: YOU ARE THE CHARACTER. Stay in character throughout.
</INSTRUCTIONS>

{language_instruction}

Act as the following character:
<YOUR_PERSONA>
{persona}
</YOUR_PERSONA>

{behavioral_instructions}

{disclosure_instructions}

{interaction_style_instructions}

Your question or request should be about this topic:
<TOPIC>
{topic}
</TOPIC>"""


_DEFAULT_USER_JUDGE_RUBRIC = """Criteria for success:
- The user's message is natural and coherent for their persona.
- The message stays on topic or naturally extends the conversation.
- The user provides relevant follow-up questions or clarifications.
- The language and vocabulary match the persona's background.

Criteria for failure:
- The user breaks character or acts like an assistant.
- The message is incoherent or contradicts the persona.
- The user provides assistant-like analysis or structured output.
- The user offers help, assistance, or expertise to the assistant — a real user receiving help does not offer to help the helper.
- The user references the conversation itself, the simulation, or how messages are being constructed — this is a fourth-wall break."""


_INLINE_JUDGE_RESPONSE_FORMAT = """
Strictly respond in the following format:
<explanation>
[Your detailed explanation and justification for the rating.]
</explanation>
<rating>
[Must be either 'success' or 'failure' strictly]
</rating>"""

# ── Public templates (YAML overrides compiled-in defaults) ────────────────

OPEN_ENDED_USER_AGENT_SYSTEM_PROMPT = get_prompt(_SCENARIO, "user_agent_system_prompt", _DEFAULT_USER_AGENT)

OPEN_ENDED_USER_JUDGE_RUBRIC = get_prompt(_SCENARIO, "user_judge_rubric", _DEFAULT_USER_JUDGE_RUBRIC)


USER_JUDGE_TURN_PROMPT = (
    """You are an expert evaluator. Judge the user's message in this conversation.

Previous conversation:
<PREV_CONVERSATION>
{conversation_history}
</PREV_CONVERSATION>

User message to judge:
<USER_MESSAGE>
{user_turn_to_evaluate}
</USER_MESSAGE>

<RUBRIC>
"""
    + OPEN_ENDED_USER_JUDGE_RUBRIC
    + """
</RUBRIC>
"""
    + _INLINE_JUDGE_RESPONSE_FORMAT
)


# ── Call-time resolution ─────────────────────────────────────────────
# The module-level constants above bind whatever asset root was active at
# IMPORT time. Probes are imported at plugin bootstrap, before a run's
# ``assets_dir`` is known, so a per-run root would never reach them. These
# accessors re-resolve on every call, which is what makes
# ``--assets-dir`` / ``USERSIM_ASSET_PATH`` govern prompts the same way it
# already governs seeds and banks. The constants are kept for callers that
# import them directly.


def user_agent_system_prompt() -> str:
    """Resolve ``user_agent_system_prompt`` against the currently-active asset root."""
    return get_prompt(_SCENARIO, "user_agent_system_prompt", _DEFAULT_USER_AGENT)


def assistant_system_prompt() -> str:
    """Resolve ``assistant_system_prompt`` against the currently-active asset root.

    Defaults to ``""``: the assistant under test gets no system prompt, so
    that whether it behaves helpfully, answers in the right language or
    refuses is what the probe measures rather than what it was told. See
    the pure-capability policy in ``docs/engine/README.md``.
    Resolved like every other prompt so the mechanism is uniform, not
    because filling it in is expected -- a non-empty value here changes
    what the probe measures and wants a rationale in the module docstring.
    """
    return get_prompt(_SCENARIO, "assistant_system_prompt", "")


def user_judge_rubric() -> str:
    """Resolve ``user_judge_rubric`` against the currently-active asset root."""
    return get_prompt(_SCENARIO, "user_judge_rubric", _DEFAULT_USER_JUDGE_RUBRIC)


def user_judge_turn_prompt() -> str:
    """``USER_JUDGE_TURN_PROMPT`` with the rubric resolved at call time."""
    return USER_JUDGE_TURN_PROMPT.replace(OPEN_ENDED_USER_JUDGE_RUBRIC, user_judge_rubric())
