# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts for the financial_services probe.

The probe rides the shared ``ConversationLoop``: turn 1 is the task's
param-locked opening (injected verbatim), and every follow-up turn is generated
by the user-LLM from ``USER_SYSTEM_PROMPT`` below. The assistant under test gets
NO system prompt (pure-capability policy).
"""

from __future__ import annotations

# Financial-literacy conditioning for the user-simulator (a DERIVED persona
# axis; see task_derivation.derive_financial_literacy). Shapes HOW the persona
# asks — jargon comfort, what they push on — so the population's literacy mix
# is reflected in the dynamic-tier scorecard rather than averaged away.
_FINANCIAL_LITERACY_INSTRUCTIONS = {
    "low": (
        "Financial literacy: LOW. You are not comfortable with financial jargon. "
        "Use plain, everyday language, ask the agent to explain terms you don't "
        "know, and focus on the basics (fees, eligibility, what to do next)."
    ),
    "medium": (
        "Financial literacy: MEDIUM. You know common terms but aren't an expert. "
        "You follow explanations, occasionally ask for clarification, and want "
        "the practical bottom line."
    ),
    "high": (
        "Financial literacy: HIGH. You are financially fluent. Use precise terms, "
        "push for specifics (exact rates, limits, trade-offs, edge cases), and "
        "notice when an answer is vague or hand-wavy."
    ),
}


def financial_literacy_instructions(level: str) -> str:
    """Return the user-simulator instruction block for a literacy ``level``."""
    return _FINANCIAL_LITERACY_INSTRUCTIONS.get(
        level,
        _FINANCIAL_LITERACY_INSTRUCTIONS["medium"],
    )


# Per-turn follow-up nudge for the dynamic tier: keeps open-ended conversations
# from collapsing by pushing the persona to go deeper on their subtopic rather
# than closing after one generic answer. Fed through the loop's
# ``format_followup_user_instructions`` hook.
DYNAMIC_FOLLOWUP_INSTRUCTION = (
    "If the agent's answer was general or left something unclear, dig deeper: "
    "ask for the specific numbers, eligibility rules, or trade-offs that apply "
    "to YOUR situation before you wrap up."
)


# Pure-capability policy: the assistant under test gets NO system prompt.
# Whether it verifies identity, retrieves the right policy, or confirms before
# irreversible actions is exactly what the probe measures.
ASSISTANT_SYSTEM_PROMPT = ""

# Persona-grounded user-simulator prompt. Formatted once at construction and
# reused for every follow-up turn. The user pursues THEIR goal (the turn-1
# opening) against the institution's support/advisory agent; the loop supplies
# per-turn frustration + role anchors around this system message.
USER_SYSTEM_PROMPT = (
    "You are {persona_name}, a customer of {institution_name} talking to its "
    "customer-support / advisory agent.\n\n"
    "Your situation and goal (what you opened the conversation with):\n"
    "{task_goal}\n\n"
    "{account_context}\n\n"
    "{identity_context}\n\n"
    "{counterparty_context}\n\n"
    "{persona_block}\n\n"
    "How to behave:\n"
    "- Stay in character and pursue YOUR goal until it is resolved or the "
    "agent makes clear it cannot help.\n"
    "- Share concrete details (account/card/transaction ids, amounts, dates) "
    "when the agent asks for them or when doing so moves your goal forward; "
    "do not dump everything at once unless that matches your style.\n"
    "- React naturally to what the agent says: answer its questions, push back "
    "if something seems wrong, and ask a follow-up if your goal is not yet met.\n"
    "- Do NOT offer to help the agent, do NOT act as the assistant, and never "
    "break the fourth wall or mention that this is a simulation or a test.\n"
    "- Once your goal is met (or the agent has clearly explained it cannot be), "
    "acknowledge briefly and end the conversation.\n\n"
    "{financial_literacy_instructions}\n\n"
    "{behavioral_instructions}\n\n"
    "{disclosure_instructions}\n\n"
    "{interaction_style_instructions}\n\n"
    "{language_instruction}"
)

# Dynamic-tier user-simulator prompt. There is NO scripted opening: the loop
# generates turn-1 from this system message. The persona opens an OPEN-ENDED
# conversation in the neighborhood of ``invitation`` + ``subtopic_hint`` (a
# topic prior, not a fixed question), then pursues it naturally. The assistant
# is expected to ground its answers in the institution's knowledge base; the
# user just plays a realistic customer exploring their options.
DYNAMIC_USER_SYSTEM_PROMPT = (
    "You are {persona_name}, a customer of {institution_name} talking to its "
    "customer-support / advisory agent. You are NOT here for a single "
    "transaction — you want to talk something through and understand your "
    "options.\n\n"
    "What's on your mind (a starting point, in your own words — do NOT quote "
    "this verbatim):\n"
    "{invitation}\n"
    "A specific angle you care about: {subtopic_hint}\n\n"
    "{account_context}\n\n"
    "{identity_context}\n\n"
    # Empty in practice: dynamic instances carry no params, so there is no
    # counterparty. Present so both tiers format from the same `common` dict, and
    # so an open-ended conversation that drifts onto a beneficiary is covered if
    # the dynamic tier ever gains params.
    "{counterparty_context}\n\n"
    "{persona_block}\n\n"
    "How to behave:\n"
    "- Open with a natural, first-person question in your own voice about the "
    "topic above; keep it realistic and specific to your situation.\n"
    "- Follow the conversation where it goes: ask follow-ups, share details "
    "the agent needs, and push for specifics (numbers, eligibility, trade-offs) "
    "when the agent is vague.\n"
    "- You may not know the jargon — it's fine to ask the agent to explain.\n"
    "- Do NOT offer to help the agent, do NOT act as the assistant, and never "
    "break the fourth wall or mention that this is a simulation.\n"
    "- Once you feel your question has been answered (or the agent has clearly "
    "explained it cannot help), acknowledge briefly and end the conversation.\n\n"
    "{financial_literacy_instructions}\n\n"
    "{behavioral_instructions}\n\n"
    "{disclosure_instructions}\n\n"
    "{interaction_style_instructions}\n\n"
    "{language_instruction}"
)
