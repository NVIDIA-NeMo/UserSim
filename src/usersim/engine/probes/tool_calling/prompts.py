# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Prompt templates for the tool_calling probe.

Templates are loaded from assets/tool_calling/prompts.yaml when available,
falling back to the compiled-in defaults below.

All simulation agents (user, tool simulator) and the trajectory judge
use a strong frontier model. The assistant model (under test) receives
only ASSISTANT_SYSTEM_PROMPT and the tool schemas via the API.
"""

from usersim.engine.core.prompt_loader import get_prompt

_SCENARIO = "tool_calling"

# ── Compiled-in defaults ──────────────────────────────────────────────────

_DEFAULT_USER_AGENT = """You are roleplaying as a real person who needs help with a task.
You have access to the description of tools the assistant can use, but you must NOT
reference the assistant's specific tool function names, parameter names, or enum values.

<INSTRUCTIONS>
Crafting your request:
- Based on your persona, formulate a natural request related to the described theme.
- Your request MUST be something that genuinely requires the assistant to use one or
  more of the available tools to fulfill. Do NOT ask pure knowledge or opinion questions
  that could be answered without any tool.
- Frame requests as a real person would in everyday language.
- It is OK to be vague, underspecify, or omit details a real user wouldn't know.
- Let the assistant figure out the specifics — that's what you're evaluating.
- Draw from your persona's background, interests, and context to make the request realistic.
- Do NOT reference the assistant's specific tool function names, parameter names,
  or enum values (e.g., don't say "use get_weather" or "set source_lang to en").
- General technology terms, file formats, and domain knowledge ARE fine — a real
  user would naturally say things like ".ics file", "CSV", "REST API", or "JSON".

For follow-up responses:
- Respond naturally to the assistant's questions or results.
- If the assistant asks for clarification, provide it in natural language.
- If the result seems wrong or incomplete, say so as a real user would.
- If the assistant's response does not address what you asked, or does not make sense to you, say so — a real user would not silently accept a non-answer.
- If satisfied, naturally wind down the conversation.

Conversational style:
- Do NOT reveal you are roleplaying.
- Do NOT use stage directions, bracketed actions, or narrator commentary.
- Speak naturally as the character would in their daily life.
- Match vocabulary and complexity to your persona's education and background.

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

The assistant has access to tools in this domain:
<TOOL_CONTEXT>
{tool_context}
</TOOL_CONTEXT>

Your request should be about this theme:
<THEME>
{theme}
</THEME>"""


_DEFAULT_TOOL_SIMULATOR = """You are simulating a realistic API response for a tool call.
Given the tool specification and the arguments provided, generate a realistic JSON response
that the tool would return. The response should be consistent with the tool's description
and the arguments provided.

Tool specification:
{tool_spec}

Tool call arguments:
{tool_args}

Conversation context (for understanding what the user needs):
{conversation_context}

Respond with ONLY the JSON that this API would return. Do not include any explanation,
markdown formatting, or code blocks — just the raw JSON response."""


# The trajectory-judge prompt for tool-calling lives in
# usersim.engine.evaluator.scorers.tool_use as
# ``_TRAJECTORY_JUDGE_PROMPT``. The simulator no longer judges the
# assistant under test; the trajectory-evaluator plugin does.


_DEFAULT_USER_JUDGE = """You are an expert evaluator. Judge the user's message in this conversation.

The assistant has access to these tools:
<TOOLS>
{tools}
</TOOLS>

Previous conversation:
<PREV_CONVERSATION>
{conversation_history}
</PREV_CONVERSATION>

User message to judge:
<USER_MESSAGE>
{user_turn_to_evaluate}
</USER_MESSAGE>

<RUBRIC>
Criteria for success:
- The user's message is natural and coherent for their persona.
- The user does NOT directly reference the assistant's tool function names, parameter
  names, or enum values from the tool definitions above (e.g., saying
  "call get_translation_nllb" or "pass source_lang='en'" would be a violation).
- General technology vocabulary, file formats, protocols, and domain terminology
  (e.g., ".ics file", "CSV export", "API key", "webhook") are NOT violations —
  real users know these terms.
- The message stays on topic or naturally extends the conversation.
- The language and vocabulary match the persona's background.
{extra_criteria}
Criteria for failure:
- The user breaks character or acts like an assistant.
- The user directly references tool function names, parameter names, or enum values
  from the tool definitions above.
- The message is incoherent or contradicts the persona.
- The user offers help, assistance, or expertise to the assistant — a real user receiving help does not offer to help the helper.
- The user references the conversation itself, the simulation, or how messages are being constructed — this is a fourth-wall break.
{extra_failure_criteria}
</RUBRIC>

Strictly respond in the following format:
<explanation>
[Your detailed explanation and justification for the rating.]
</explanation>
<rating>
[Must be either 'success' or 'failure' strictly]
</rating>"""


_DEFAULT_TOOL_RELEVANCE_SUCCESS = (
    "- The user's request naturally requires the assistant to use at least one of\n"
    "  the available tools to properly fulfill it. Pure knowledge or opinion questions\n"
    "  that the assistant could answer without any tool do NOT qualify.\n"
)

_DEFAULT_TOOL_RELEVANCE_FAILURE = (
    "- The request can be fully answered from general knowledge or opinion alone,\n"
    "  without needing any of the available tools.\n"
)

# ── Public templates (YAML overrides compiled-in defaults) ────────────────

USER_AGENT_SYSTEM_PROMPT = get_prompt(_SCENARIO, "user_agent_system_prompt", _DEFAULT_USER_AGENT)

# PURE-CAPABILITY-TEST POLICY: default empty.
#
# tool_calling tests the assistant model's native tool-use behavior.
# An elaborate system prompt ("use tools when they help, format args
# correctly, etc.") would prime the model and contaminate the signal.
# The OpenAI ``tools=`` parameter exposes the tool schemas — the model
# knows what tools exist and what they do (each tool's own description
# is part of its schema), but no behavioral instruction is added. See
# conversation-plugin/README.md "Assistant under test:
# pure-capability policy".
#
# A YAML override is still honored (``assistant_system_prompt`` key in
# the per-locale prompts file) for explicit research-driven exceptions
# — those should be deliberate, documented, and reviewed.
ASSISTANT_SYSTEM_PROMPT = get_prompt(_SCENARIO, "assistant_system_prompt", "")

TOOL_SIMULATOR_PROMPT = get_prompt(_SCENARIO, "tool_simulator_prompt", _DEFAULT_TOOL_SIMULATOR)

USER_JUDGE_PROMPT = get_prompt(_SCENARIO, "user_judge_prompt", _DEFAULT_USER_JUDGE)

TOOL_RELEVANCE_SUCCESS = get_prompt(_SCENARIO, "tool_relevance_success", _DEFAULT_TOOL_RELEVANCE_SUCCESS)

TOOL_RELEVANCE_FAILURE = get_prompt(_SCENARIO, "tool_relevance_failure", _DEFAULT_TOOL_RELEVANCE_FAILURE)


# ── Call-time resolution ─────────────────────────────────────────────
# The module-level constants above bind whatever asset root was active at
# IMPORT time. Probes are imported at plugin bootstrap, before a run's
# ``assets_dir`` is known, so a per-run root would never reach them. These
# accessors re-resolve on every call, which is what makes ``--assets-dir`` /
# ``USERSIM_ASSETS_DIR`` govern prompts the same way it already governs
# seeds and banks. The constants are kept for callers that import them.


def user_agent_system_prompt() -> str:
    return get_prompt(_SCENARIO, "user_agent_system_prompt", _DEFAULT_USER_AGENT)


def assistant_system_prompt() -> str:
    return get_prompt(_SCENARIO, "assistant_system_prompt", "")


def tool_simulator_prompt() -> str:
    return get_prompt(_SCENARIO, "tool_simulator_prompt", _DEFAULT_TOOL_SIMULATOR)


def user_judge_prompt() -> str:
    return get_prompt(_SCENARIO, "user_judge_prompt", _DEFAULT_USER_JUDGE)


def tool_relevance_success() -> str:
    return get_prompt(_SCENARIO, "tool_relevance_success", _DEFAULT_TOOL_RELEVANCE_SUCCESS)


def tool_relevance_failure() -> str:
    return get_prompt(_SCENARIO, "tool_relevance_failure", _DEFAULT_TOOL_RELEVANCE_FAILURE)
