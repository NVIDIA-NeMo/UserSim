# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-sim assistant evaluation prompt templates.

The judge prompts in here are exclusively for evaluating the
*assistant* under test — they have nothing to do with the in-sim
user-query / assistant-quality / early-stop judges, which keep the
conversation sound during simulation. See the
in-sim / out-of-sim boundary.
"""

from __future__ import annotations

EVAL_SYSTEM_PROMPT = (
    "You are an expert evaluator of AI assistant conversations. "
    "Score each axis carefully with detailed reasoning. "
    "ALWAYS write every `reasoning` field in English, regardless of the "
    "language the conversation under evaluation was conducted in. The "
    "reasoning is read by English-speaking reviewers and tooling; do not "
    "switch to the conversation's locale even when quoting the assistant "
    "or user. Quoted snippets in another language are fine, but the "
    "surrounding analysis must be in English."
)

EVAL_USER_PROMPT = """Evaluate the assistant's performance in the following conversation.

The user has this persona background:
<PERSONA>
{persona}
</PERSONA>

The conversation was in {language} (locale: {locale}).
{locale_rigor_instruction}
<CONVERSATION>
{conversation}
</CONVERSATION>

IMPORTANT: Evaluate each turn individually. If ANY assistant response in the
conversation is unintelligible or fails to address the user's question, that
should heavily penalize the relevant axes. Two good turns cannot fully
compensate for one catastrophic failure.

OUTPUT LANGUAGE: Write all `reasoning` fields in English, even when the
conversation under review is in {language}. Short verbatim quotes in the
original language are allowed, but the analysis around them must be in
English so reviewers and downstream tooling can read it consistently.

SCORING CALIBRATION:
- Score 5 (Exceptional): Reserve for truly outstanding responses. The response must
  leave no meaningful room for improvement — not just "good," but the best plausible
  answer to this specific user's question.
- Score 4 (Good): A solid, competent response with only minor gaps. This is what a
  well-performing assistant should typically achieve.
- Score 3 (Adequate): The core task is addressed but with noticeable issues — missing
  context, unnecessary verbosity, imprecise information, or lack of adaptation to
  the user's level.
- Score 2 (Poor): Significant issues that impair utility for the user.
- Score 1 (Failure): Response is wrong, harmful, off-topic, or unintelligible.

Do NOT default to 5. Critically evaluate each axis. A competent response that gets the
job done is a 4, not a 5. A 5 means you cannot think of a way to improve the response.

Score each axis according to its rubric. For each axis, provide a score
(choosing from the defined options) and detailed reasoning explaining
your assessment."""


def build_locale_rigor_instruction(language: str) -> str:
    """Same-rigor-across-languages directive injected for non-English locales."""
    if language == "English":
        return ""
    return (
        f"\nYou MUST evaluate the quality of the {language} content with the "
        f"same rigor as you would English. Do not give higher scores simply "
        f"because the response is in a non-English language. Evaluate grammar, "
        f"naturalness, accuracy, and helpfulness in {language} to the same standard."
    )
