# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge utilities for conversation simulation: parsing, inline judging."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


from usersim.engine.core.llm import call_llm


JUDGE_FOLLOWUP_PROMPT = """Please reformat your previous response to strictly follow this format:
<explanation>
[Your detailed explanation and justification for the rating.]
</explanation>
<rating>
[Must be either 'success', 'warn', or 'failure' strictly]
</rating>"""


def _parse_judge_response(text: str) -> Tuple[str, str, bool]:
    """Parse <explanation>/<rating> XML tags from judge LLM output.

    Returns (explanation, rating, parsed_ok).
    Accepts 'success', 'warn', and 'failure' as valid ratings.
    """
    expl_match = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
    rating_match = re.search(r"<rating>\s*(.*?)\s*</rating>", text, re.DOTALL)

    explanation = expl_match.group(1).strip() if expl_match else text.strip()
    rating_raw = rating_match.group(1).strip().lower() if rating_match else ""

    if rating_raw in ("success", "warn", "failure"):
        return explanation, rating_raw, True
    return explanation, "failure", False


def run_inline_judge(
    models: Dict[str, Any],
    alias: str,
    prompt_text: str,
) -> Tuple[str, str, bool]:
    """Run an inline judge call and parse its XML-tagged output.

    If the first response doesn't contain valid tags, sends a follow-up
    asking the model to reformat.

    Returns (explanation, rating, passed) where passed is True for both
    'success' and 'warn' ratings (only 'failure' terminates).
    """
    explanation, rating, passed, _ = run_inline_judge_ex(models, alias, prompt_text)
    return explanation, rating, passed


def run_inline_judge_ex(
    models: Dict[str, Any],
    alias: str,
    prompt_text: str,
) -> Tuple[str, str, bool, bool]:
    """``run_inline_judge`` plus the final parse status.

    Returns ``(explanation, rating, passed, parse_ok)``. ``parse_ok`` is
    False when even the reformat follow-up produced no valid ``<rating>``
    tag — the "failure" verdict is then a parser default, not a judgment.
    Callers that spend budget on failure verdicts (assistant-turn
    resampling) use this to tell a genuine rejection from judge flakiness.
    """
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are an expert judge."},
        {"role": "user", "content": prompt_text},
    ]
    resp = call_llm(models, alias, msgs)
    content = resp.get("content", "") if isinstance(resp, dict) else ""

    explanation, rating, parsed_ok = _parse_judge_response(content)
    if parsed_ok:
        return explanation, rating, rating in ("success", "warn"), True

    msgs.append({"role": "assistant", "content": content})
    msgs.append({"role": "user", "content": JUDGE_FOLLOWUP_PROMPT})
    resp2 = call_llm(models, alias, msgs)
    content2 = resp2.get("content", "") if isinstance(resp2, dict) else ""
    explanation, rating, parsed_ok2 = _parse_judge_response(content2)
    return explanation, rating, rating in ("success", "warn"), parsed_ok2
