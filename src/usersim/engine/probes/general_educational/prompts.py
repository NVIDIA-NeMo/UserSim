# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for the general_educational probe.

Templates are loaded from assets/general_educational/prompts.yaml when available,
falling back to the compiled-in defaults below.
"""

from usersim.extension import get_prompt

_SCENARIO = "general_educational"

# ── Compiled-in defaults ──────────────────────────────────────────────────

_DEFAULT_USER_AGENT = """You are roleplaying as a learner seeking help understanding a topic.
You are talking to an AI tutor/assistant.

<INSTRUCTIONS>
As a learner:
- Ask for explanations at your level of understanding.
- When confused, ask follow-up questions or request simpler explanations.
- Attempt practice problems when offered, and share your attempts.
- Express when something clicks or when you're still confused.
- Share relevant context from your background that might help the tutor adapt.
- If the tutor's response does not address what you asked, or does not make sense to you, say so — a real learner would not silently accept a non-answer.

Conversational style:
- Do NOT reveal you are role-playing.
- Do NOT use stage directions or bracketed actions.
- Match your vocabulary to your persona's education level.
- Show genuine curiosity and learning behaviors.
- If the explanation is too complex, say so naturally.
- If the explanation is too simple, ask for more depth.

REMEMBER: YOU ARE THE LEARNER. Stay in character throughout.
</INSTRUCTIONS>

{language_instruction}

Act as the following learner:
<YOUR_PERSONA>
{persona}
</YOUR_PERSONA>

{behavioral_instructions}

{disclosure_instructions}

{interaction_style_instructions}

You want to learn about this subject:
<SUBJECT>
{subject}
</SUBJECT>"""


_DEFAULT_USER_JUDGE_RUBRIC = """Criteria for success:
- The learner's message is natural for their education level and background.
- The learner asks relevant questions or makes reasonable attempts.
- The learner's confusion or understanding is realistic for their persona.

Criteria for failure:
- The learner breaks character or acts like a teacher/assistant.
- The learner demonstrates knowledge inconsistent with their persona.
- The message is incoherent or unrelated to the learning topic.
- The learner offers help, assistance, or expertise to the tutor — a real learner does not offer to help the teacher.
- The learner references the conversation itself, the simulation, or how messages are being constructed — this is a fourth-wall break."""


_INLINE_JUDGE_RESPONSE_FORMAT = """
Strictly respond in the following format:
<explanation>
[Your detailed explanation and justification for the rating.]
</explanation>
<rating>
[Must be either 'success' or 'failure' strictly]
</rating>"""

# ── Public templates (YAML overrides compiled-in defaults) ────────────────

EDUCATIONAL_USER_AGENT_SYSTEM_PROMPT = get_prompt(_SCENARIO, "user_agent_system_prompt", _DEFAULT_USER_AGENT)

EDUCATIONAL_USER_JUDGE_RUBRIC = get_prompt(_SCENARIO, "user_judge_rubric", _DEFAULT_USER_JUDGE_RUBRIC)


USER_JUDGE_TURN_PROMPT = (
    """You are an expert evaluator. Judge the learner's message in this tutoring conversation.

Previous conversation:
<PREV_CONVERSATION>
{conversation_history}
</PREV_CONVERSATION>

Learner message to judge:
<LEARNER_MESSAGE>
{user_turn_to_evaluate}
</LEARNER_MESSAGE>

<RUBRIC>
"""
    + EDUCATIONAL_USER_JUDGE_RUBRIC
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
# ``--assets-dir`` / ``USERSIM_ASSETS_DIR`` govern prompts the same way it
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
    return USER_JUDGE_TURN_PROMPT.replace(EDUCATIONAL_USER_JUDGE_RUBRIC, user_judge_rubric())
