# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-calling 8-axis trajectory scorer.

The 8 axes (task_completion, tool_selection, argument_quality,
unnecessary_tool_use, information_gathering, role_adherence,
architecture_leaking, overall) are *assistant-under-test* judgments,
not simulation infrastructure — they belong on the evaluator side of
the in-sim / out-of-sim boundary.

The scorer is registered as ``"tool_use"`` and is invoked by the
``TrajectoryEvaluator`` when a config carries ``scorers=["tool_use"]``.
Configure this scorer for any tool-calling-family trajectory you want
the 8-axis breakdown for. Future probes (e.g., agentic safety) will
register their own scorers alongside.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from data_designer.config.column_configs import Score
from data_designer.engine.column_generators.utils.judge_score_factory import (
    create_judge_response_model,
    create_judge_structured_output_model,
)

from usersim.engine.core.llm import call_llm
from usersim.engine.core.messages import (
    format_conversation_history_for_prompt,
    format_tools_for_prompt,
)
from usersim.engine.evaluator.scorers import register_scorer

logger = logging.getLogger("usersim.engine")


# ── 8-axis trajectory schema (unchanged from the in-sim version) ──────

TRAJECTORY_SCORES: List[Score] = [
    Score(
        name="task_completion",
        description="Was the user's request resolved?",
        options={
            1: "Fail — request unresolved",
            3: "Acceptable — partially resolved",
            5: "Good — fully resolved",
        },
    ),
    Score(
        name="tool_selection",
        description="Did the model choose the right tool(s) for the task?",
        options={
            1: "Fail — wrong tools chosen",
            3: "Acceptable — suboptimal but functional",
            5: "Good — optimal tool selection",
        },
    ),
    Score(
        name="argument_quality",
        description="Were arguments reasonable given the conversation?",
        options={
            1: "Fail — incorrect or missing arguments",
            3: "Acceptable — mostly correct with issues",
            5: "Good — correct and well-formed",
        },
    ),
    Score(
        name="unnecessary_tool_use",
        description="Did the model avoid calling tools when not needed?",
        options={
            1: "Fail — called tools unnecessarily",
            3: "Acceptable — minor unnecessary calls",
            5: "Good — appropriate restraint",
        },
    ),
    Score(
        name="information_gathering",
        description=("Did the model ask for missing info when needed, without re-asking for info already provided?"),
        options={
            1: "Fail — failed to gather or re-asked known info",
            3: "Acceptable — gathered info with some issues",
            5: "Good — efficient information gathering",
        },
    ),
    Score(
        name="role_adherence",
        description=(
            "Did the user and assistant each stay in their role? "
            "The user should only ask questions, make requests, and provide requested details. "
            "The assistant should help, answer, and call tools — and should always respond "
            "with a coherent reply that addresses the user. Producing unintelligible content "
            "is a role failure. "
            "Penalize if the user explains things, provides structured data, calls tools, "
            "takes over the assistant's task, offers help to the assistant, or "
            "references the conversation itself or how it is being conducted."
        ),
        options={
            1: "Fail — clear role inversion",
            3: "Acceptable — minor role blurring",
            5: "Good — both stay in role",
        },
    ),
    Score(
        name="architecture_leaking",
        description=(
            "Did the assistant expose internal tool names, API endpoints, or system "
            "architecture to the user? A real assistant never says things like "
            "'the API only supports metadata' or 'the available endpoints are...' "
            "or names specific tool functions in its responses. If it cannot fulfill "
            "a request, it should say so without technical justification."
        ),
        options={
            1: "Fail — named tools, APIs, or endpoints",
            3: "Acceptable — vague system references",
            5: "Good — no internal details exposed",
        },
    ),
    Score(
        name="overall",
        description=(
            "Holistic trajectory quality considering all axes above. "
            "A single unintelligible or non-responsive assistant reply should "
            "significantly lower the overall score."
        ),
        options={
            1: "Fail — poor overall trajectory",
            3: "Acceptable — functional but flawed",
            5: "Good — strong overall performance",
        },
    ),
]


_TRAJECTORY_JUDGE_PROMPT = """\
Evaluate the assistant's tool-calling performance in this conversation.

The user interacted with an AI assistant that had access to the following tools:
<TOOLS>
{tools}
</TOOLS>

<CONVERSATION>
{conversation}
</CONVERSATION>

Evaluate the assistant's performance on each axis according to the scoring rubric.
Consider the full trajectory — not just individual calls, but the overall approach
to solving the user's request.

For each axis, provide:
- A score from the defined options
- Detailed reasoning explaining your assessment"""


# Default judge alias for the relocated tool-use scorer. Honours the
# same alias the evaluator config uses for its judge ensemble — but
# falls back to a sensible default if absent.
_DEFAULT_JUDGE_ALIAS = "judge_model"


def _build_schema():
    response_models = [create_judge_response_model(s) for s in TRAJECTORY_SCORES]
    return create_judge_structured_output_model(response_models)


def _normalize_tool_subset(tool_subset_field: Any) -> List[Dict[str, Any]]:
    """The trajectory carries `tool_subset` as a JSON-encoded list. Decode."""
    if tool_subset_field is None:
        return []
    if isinstance(tool_subset_field, list):
        return tool_subset_field
    if isinstance(tool_subset_field, str):
        try:
            parsed = json.loads(tool_subset_field)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def score_tool_use_trajectory(
    trajectory: Dict[str, Any],
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the 8-axis tool-use trajectory judge to one trajectory.

    Inputs (from the trajectory row dict):
      - ``conversation_messages``: list of messages.
      - ``tool_subset``: optional, the tool definitions the assistant
        was given (lives on the simulator's trajectory row for tool_calling).
      - implicitly, ``models`` (from the evaluator) holds the judge
        facades; we pick the first one as the scoring judge.

    Output: a dict shaped like::

        {
          "judge_alias": "...",
          "scores": {
            "task_completion": {"score": 5, "reasoning": "..."},
            ...
          },
          "status_proposal": True | False,   # False if a critical axis scored 1
          "error": "..."  (only present if scoring failed)
        }

    ``status_proposal`` is a heuristic single-bit pass/fail derived from
    the raw scores: False if any of ``overall`` / ``tool_selection`` /
    ``architecture_leaking`` was scored 1. Consumers that need finer
    grain should derive their own pass/fail from ``scores``.
    """
    messages = trajectory.get("conversation_messages") or []
    tool_subset = _normalize_tool_subset(trajectory.get("tool_subset"))

    # Pick the first available judge facade (the evaluator passes its
    # full {alias: facade} map; for ensemble usage the evaluator
    # generator already runs the LLM-judge ensemble per axis — this
    # scorer is for the *separate* 8-axis structured judgment).
    judge_alias = next(iter(models)) if models else _DEFAULT_JUDGE_ALIAS

    prompt = _TRAJECTORY_JUDGE_PROMPT.format(
        tools=format_tools_for_prompt(tool_subset),
        conversation=format_conversation_history_for_prompt(messages),
    )
    msgs = [
        {"role": "system", "content": "You are an expert evaluator of AI tool-calling trajectories."},
        {"role": "user", "content": prompt},
    ]
    schema_model = _build_schema()
    try:
        resp = call_llm(
            models,
            judge_alias,
            msgs,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trajectory_judgment",
                    "schema": schema_model.model_json_schema(),
                },
            },
        )
    except Exception as e:
        logger.warning(f"  |-- evaluator/scorers.tool_use: judge {judge_alias!r} raised {type(e).__name__}: {e}")
        return {"judge_alias": judge_alias, "scores": {}, "status_proposal": True, "error": f"{type(e).__name__}: {e}"}

    content = resp.get("content", "") if isinstance(resp, dict) else ""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("  |-- evaluator/scorers.tool_use: failed to parse structured output")
        return {
            "judge_alias": judge_alias,
            "scores": {s.name: {"score": None, "reasoning": "Parse failure"} for s in TRAJECTORY_SCORES},
            "status_proposal": True,
            "error": "parse_failure",
        }

    scores: Dict[str, Dict[str, Any]] = {}
    for s in TRAJECTORY_SCORES:
        cell = parsed.get(s.name)
        if isinstance(cell, dict):
            scores[s.name] = {
                "score": cell.get("score"),
                "reasoning": cell.get("reasoning", ""),
            }
        else:
            scores[s.name] = {"score": None, "reasoning": ""}

    # Critical-axis heuristic: a 1 on overall, tool_selection, or
    # architecture_leaking flips the proposed pass/fail bit.
    status_proposal = True
    for k in ("overall", "tool_selection", "architecture_leaking"):
        v = scores.get(k, {})
        if v.get("score") == 1:
            status_proposal = False
            break

    return {
        "judge_alias": judge_alias,
        "scores": scores,
        "status_proposal": status_proposal,
    }


# Auto-register on import so the scorer registry sees it as soon as
# any code imports `usersim.engine.evaluator.scorers.tool_use`.
register_scorer("tool_use", score_tool_use_trajectory)
