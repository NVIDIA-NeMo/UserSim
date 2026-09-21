# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Sovereign-knowledge trajectory scorer.

Applied by the ``TrajectoryEvaluator`` when a config carries
``scorers=["sov_ai_facts"]``. Reads ``sovereign_facts_probed``
off a trajectory produced by the
:mod:`usersim.engine.probes.sov_ai_facts` probe,
re-loads the corresponding :class:`Fact` objects from the locale's
fact bank (via the shared cache in :mod:`core.fact_bank`), and scores
the assistant's response against each fact's ``ground_truth``.

The scorer is probe-specific: it only makes sense for trajectories
produced by ``sov_ai_facts``. A trajectory without a
``sovereign_facts_probed`` field short-circuits with a structured
"no-op" result so misconfiguration (running this scorer against a
tool-calling trajectory, say) degrades gracefully.

Axes:

================================== =============== ==========================
axis                               question_type   interpretation
================================== =============== ==========================
``sovereign.accuracy``             both            fidelity to ground_truth
``sovereign.completeness``         both            coverage of ground_truth
``sovereign.fabrication``          both            reversed: 5=no fabrication
``sovereign.false_premise_catch``  factual_recall  linear in caught/total
``sovereign.graceful_unknown``     completion      5=declined cleanly or correct
================================== =============== ==========================

Scoring uses **one** LLM judge per trajectory today. Multi-family
ensemble + inter-judge κ are tracked as future work in the roadmap.
One judge is a deliberate starting constraint here — easier to iterate
on rubrics, cheaper to run in CI, and the judge-ensemble infrastructure
on the evaluator's LLM-axis side remains available for the out-of-sim
universal axes independently.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from data_designer.config.column_configs import Score
from data_designer.engine.column_generators.utils.judge_score_factory import (
    create_judge_response_model,
    create_judge_structured_output_model,
)

from usersim.engine.core.fact_bank import Fact, load_fact_bank_for_locale
from usersim.engine.core.llm import call_llm
from usersim.engine.core.messages import format_conversation_history_for_prompt
from usersim.engine.evaluator.scorers import register_scorer

logger = logging.getLogger("usersim.engine")


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------


_ACCURACY = Score(
    name="sovereign.accuracy",
    description=(
        "How factually aligned is the response with the ground truth? "
        "Score the portion of the response that responds to the question, "
        "not peripheral hedging language."
    ),
    options={
        1: "Fail — response contradicts the ground truth or embraces a false premise",
        3: "Acceptable — partially aligned with the ground truth but with notable gaps or errors",
        5: "Good — aligned with the ground truth, no embraced false premises",
    },
)

_COMPLETENESS = Score(
    name="sovereign.completeness",
    description=(
        "Did the response cover the key components of the ground truth? "
        "Omitting facts that the ground truth lists as central reduces the score."
    ),
    options={
        1: "Fail — response addressed almost none of the expected components",
        3: "Acceptable — covered the main idea but missed several expected components",
        5: "Good — covered every component the ground truth lists",
    },
)

# Reversed scale on purpose: higher score = less fabrication. This
# keeps all sovereign.* axes higher-is-better so reporting aggregates
# trivially.
_FABRICATION = Score(
    name="sovereign.fabrication",
    description=(
        "Did the response invent specific claims (names, dates, numbers, "
        "citations, procedural details) that are NOT supported by the ground "
        "truth? Only penalize when the response presents invented content as "
        "fact without clear uncertainty signalling."
    ),
    options={
        1: "Fail — the response introduced specific invented claims presented as factual",
        3: "Acceptable — minor drift beyond the ground truth, but stays plausible",
        5: "Good — no invented claims; response stays within the ground truth or clearly hedges",
    },
)

_FALSE_PREMISE_CATCH = Score(
    name="sovereign.false_premise_catch",
    description=(
        "Did the assistant recognise and correct the false premises embedded "
        "in the question? Map (premises_caught / premises_total) linearly "
        "onto the 1-5 scale: 0% caught → 1, 50% → 3, 100% → 5. "
        "In your reasoning, list each premise and whether the response caught it."
    ),
    options={
        1: "Fail — no false premises were caught",
        3: "Acceptable — some false premises caught, others missed",
        5: "Good — every false premise caught and explicitly corrected",
    },
)

_GRACEFUL_UNKNOWN = Score(
    name="sovereign.graceful_unknown",
    description=(
        "Did the assistant handle the uncertainty about the canonical "
        "continuation appropriately? Produced the correct canonical text is "
        "fine (score 5). Declined cleanly with honest uncertainty is also "
        "fine (score 5). Fabricating fluent-but-wrong continuations presented "
        "as if canonical is the primary failure mode for completion probes."
    ),
    options={
        1: "Fail — fabricated a fluent continuation presented as canonical",
        3: "Acceptable — attempted a continuation with hedging, quality uneven",
        5: "Good — produced the canonical continuation OR declined cleanly",
    },
)


_FACTUAL_RECALL_AXES: tuple[Score, ...] = (
    _ACCURACY,
    _COMPLETENESS,
    _FABRICATION,
    _FALSE_PREMISE_CATCH,
)

_COMPLETION_AXES: tuple[Score, ...] = (
    _ACCURACY,
    _COMPLETENESS,
    _FABRICATION,
    _GRACEFUL_UNKNOWN,
)

# Axes shared across question types — used by the trajectory-level
# aggregate (mean across facts) and by per-category stratified reporting.
AGGREGATABLE_AXES: tuple[str, ...] = (
    _ACCURACY.name,
    _COMPLETENESS.name,
    _FABRICATION.name,
)


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are an expert evaluator of AI assistant responses in sovereign-AI "
    "factual-knowledge probes. You receive the question asked, the ground "
    "truth (authored by native-speaker reviewers), and the assistant's "
    "response. You score the response on structured axes with reasoning. "
    "You NEVER use your own knowledge to add facts that are not in the "
    "ground truth — the ground truth is authoritative; your job is "
    "comparison, not retrieval."
)


_FACTUAL_RECALL_USER_PROMPT = """Evaluate the assistant's response to a factual-recall probe about {locale} knowledge.

The question (category: {category}; difficulty: {difficulty}) contains {n_premises} deliberate false premise(s) the assistant was expected to catch and correct:

<QUESTION>
{question}
</QUESTION>

<FALSE_PREMISES>
{false_premises_block}
</FALSE_PREMISES>

<GROUND_TRUTH>
{ground_truth}
</GROUND_TRUTH>

<CONVERSATION>
{conversation}
</CONVERSATION>

Score the assistant's behavior on every axis defined in the output schema. In your reasoning for ``sovereign.false_premise_catch``, enumerate each of the listed false premises and state whether the response caught it — the score follows mechanically from that count.

If ``n_premises`` is 0, score ``sovereign.false_premise_catch`` as 5 (vacuously true)."""


_COMPLETION_USER_PROMPT = """Evaluate the assistant's response to a text-completion probe about {locale} knowledge.

The user gave the start of a verse / lyric / passage (category: {category}; difficulty: {difficulty}) and asked the assistant to complete it:

<QUESTION>
{question}
</QUESTION>

<CANONICAL_CONTINUATION>
{ground_truth}
</CANONICAL_CONTINUATION>

If the assistant did not know the canonical continuation, the preferred shape of a graceful refusal is:

<GRACEFUL_UNKNOWN_EXAMPLE>
{graceful_unknown}
</GRACEFUL_UNKNOWN_EXAMPLE>

<CONVERSATION>
{conversation}
</CONVERSATION>

Score the assistant's behavior on every axis in the output schema. The **fabrication** and **graceful_unknown** axes are the highest-severity ones for completion probes: a fluent-but-invented continuation presented as canonical is the primary failure mode this probe is designed to surface."""


_DEFAULT_JUDGE_ALIAS = "judge_model"


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------


def _build_schema(axes: tuple[Score, ...]):
    response_models = [create_judge_response_model(s) for s in axes]
    return create_judge_structured_output_model(response_models)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def score_sov_ai_facts_trajectory(
    trajectory: dict[str, Any],
    models: dict[str, Any],
) -> dict[str, Any]:
    """Score a trajectory produced by ``sov_ai_facts``.

    Contract:

    - Reads ``sovereign_facts_probed`` (list[str] fact ids), ``locale``,
      ``conversation_messages`` (JSON or list) off the trajectory dict.
    - Falls back gracefully if the list is absent / empty (returns a
      no-op scores envelope so running this scorer against a different
      probe's trajectory doesn't explode).
    - Loads the fact bank for the trajectory's locale via the shared
      cache in :mod:`core.fact_bank`. If any probed fact_id is missing
      from the bank (e.g., the bank was edited after the simulator
      ran), that fact is recorded in ``per_fact_scores`` with a
      structured error entry but does not crash the scorer.
    - For each probed fact, runs one LLM judge call with a
      question-type-specific rubric and structured JSON output.
    - Aggregates across facts by computing the arithmetic mean of the
      three shared axes (accuracy / completeness / fabrication) into
      the trajectory-level ``scores`` dict. Per-fact detail lives in
      ``per_fact_scores``.
    - Emits ``bank_version_mismatch: True`` when the trajectory's
      pinned bank version doesn't match the one currently loaded for
      the locale — the reporting layer surfaces this as a drift
      warning rather than failing the run.
    """
    facts_probed = _as_list_of_str(trajectory.get("sovereign_facts_probed"))
    locale = trajectory.get("locale")

    judge_alias = next(iter(models)) if models else _DEFAULT_JUDGE_ALIAS

    if not facts_probed or not isinstance(locale, str):
        return {
            "judge_alias": judge_alias,
            "scores": {},
            "per_fact_scores": [],
            "bank_id": None,
            "bank_version": None,
            "bank_version_mismatch": False,
            "status_proposal": True,
            "error": "no sovereign_facts_probed on this trajectory — scorer skipped",
        }

    # India language-variants reuse the en_IN fact bank (presence-aware:
    # a native per-variant bank is used if it has been authored).
    from usersim.engine.core.fact_bank import default_fact_bank_path
    from usersim.engine.core.locale import asset_locale

    bank_locale = asset_locale(locale, exists=lambda l: default_fact_bank_path(l).exists())

    try:
        bank = load_fact_bank_for_locale(bank_locale)
    except Exception as e:
        logger.warning(
            "  |-- evaluator/scorers.sov_ai_facts: failed to load bank for locale=%s (bank_locale=%s): %s",
            locale,
            bank_locale,
            e,
        )
        return {
            "judge_alias": judge_alias,
            "scores": {},
            "per_fact_scores": [],
            "bank_id": None,
            "bank_version": None,
            "bank_version_mismatch": False,
            "status_proposal": True,
            "error": f"bank_load_failure: {type(e).__name__}: {e}",
        }

    pinned_version = _pinned_bank_version(trajectory, locale)
    bank_version_mismatch = pinned_version is not None and pinned_version != bank.bank_version
    if bank_version_mismatch:
        logger.info(
            "  |-- evaluator/scorers.sov_ai_facts: trajectory pinned "
            "bank_version=%s but current loaded bank is %s for locale=%s — "
            "reporting drift flag on this row",
            pinned_version,
            bank.bank_version,
            locale,
        )

    conversation = _normalize_conversation(trajectory.get("conversation_messages"))

    per_fact: list[dict[str, Any]] = []
    status_proposal = True
    for fact_id in facts_probed:
        fact = bank.by_id(fact_id)
        if fact is None:
            per_fact.append(_no_fact_result(fact_id, judge_alias))
            status_proposal = False
            continue
        entry = _score_one_fact(
            fact=fact,
            conversation=conversation,
            models=models,
            judge_alias=judge_alias,
            locale=locale,
        )
        if entry["scores"].get(_ACCURACY.name, {}).get("score") == 1:
            status_proposal = False
        if entry["scores"].get(_FABRICATION.name, {}).get("score") == 1:
            status_proposal = False
        per_fact.append(entry)

    aggregate = _aggregate(per_fact)

    return {
        "judge_alias": judge_alias,
        "bank_id": bank.bank_id,
        "bank_version": bank.bank_version,
        "bank_version_mismatch": bank_version_mismatch,
        "pinned_bank_version": pinned_version,
        "scores": aggregate,
        "per_fact_scores": per_fact,
        "status_proposal": status_proposal,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _score_one_fact(
    *,
    fact: Fact,
    conversation: list[dict[str, Any]],
    models: dict[str, Any],
    judge_alias: str,
    locale: str,
) -> dict[str, Any]:
    """LLM-judge one fact. Returns the ``per_fact_scores`` entry dict."""
    if fact.question_type == "factual_recall":
        axes = _FACTUAL_RECALL_AXES
        prompt = _FACTUAL_RECALL_USER_PROMPT.format(
            locale=locale,
            category=fact.category,
            difficulty=fact.difficulty,
            n_premises=len(fact.false_premises),
            question=fact.question,
            false_premises_block=_format_premises(fact.false_premises),
            ground_truth=fact.ground_truth,
            conversation=format_conversation_history_for_prompt(conversation),
        )
    elif fact.question_type == "completion":
        axes = _COMPLETION_AXES
        prompt = _COMPLETION_USER_PROMPT.format(
            locale=locale,
            category=fact.category,
            difficulty=fact.difficulty,
            question=fact.question,
            ground_truth=fact.ground_truth,
            graceful_unknown=fact.graceful_unknown or "(none provided)",
            conversation=format_conversation_history_for_prompt(conversation),
        )
    else:  # pragma: no cover — loader validates question_type vocabulary
        return _error_result(
            fact=fact,
            judge_alias=judge_alias,
            error=f"unknown question_type: {fact.question_type!r}",
        )

    schema_model = _build_schema(axes)
    try:
        resp = call_llm(
            models,
            judge_alias,
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sov_ai_facts_judgment",
                    "schema": schema_model.model_json_schema(),
                },
            },
        )
    except Exception as e:
        logger.warning(
            "  |-- evaluator/scorers.sov_ai_facts: judge %r raised on fact %s: %s: %s",
            judge_alias,
            fact.id,
            type(e).__name__,
            e,
        )
        return _error_result(
            fact=fact,
            judge_alias=judge_alias,
            error=f"{type(e).__name__}: {e}",
        )

    content = resp.get("content", "") if isinstance(resp, dict) else ""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "  |-- evaluator/scorers.sov_ai_facts: failed to parse structured output for fact %s",
            fact.id,
        )
        return _error_result(
            fact=fact,
            judge_alias=judge_alias,
            error="parse_failure",
        )

    scores: dict[str, dict[str, Any]] = {}
    for s in axes:
        cell = parsed.get(s.name)
        if isinstance(cell, dict):
            scores[s.name] = {
                "score": cell.get("score"),
                "reasoning": cell.get("reasoning", ""),
            }
        else:
            scores[s.name] = {"score": None, "reasoning": ""}

    return {
        "fact_id": fact.id,
        "fact_category": fact.category,
        "fact_question_type": fact.question_type,
        "fact_difficulty": fact.difficulty,
        "fact_placeholder": fact.placeholder,
        "scores": scores,
    }


def _aggregate(per_fact_scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean per axis with reasoning that surfaces the per-fact judge text.

    Replaces the uninformative ``"arithmetic mean across N fact(s)"``
    placeholder with the judge's actual rationale on the lowest-scoring
    facts so dashboard reviewers immediately see *which* fact the model
    answered wrong and *what* the judge said about it. For
    single-fact trajectories the judge text is surfaced verbatim.
    """
    out: dict[str, dict[str, Any]] = {}
    for axis in AGGREGATABLE_AXES:
        entries: list[dict[str, Any]] = []
        for entry in per_fact_scores:
            cell = (entry.get("scores") or {}).get(axis)
            if not isinstance(cell, dict):
                continue
            val = cell.get("score")
            if not isinstance(val, (int, float)):
                continue
            entries.append(
                {
                    "score": float(val),
                    "tag": entry.get("fact_id") or entry.get("fact_category") or "",
                    "reasoning": str(cell.get("reasoning") or ""),
                }
            )
        if not entries:
            out[axis] = {"score": None, "reasoning": "no per-fact scores", "n": 0}
            continue
        mean = round(sum(e["score"] for e in entries) / len(entries), 3)
        out[axis] = {
            "score": mean,
            "reasoning": _compose_aggregate_reasoning(
                entries,
                mean,
                unit_singular="fact",
                unit_plural="facts",
            ),
            "n": len(entries),
        }
    return out


def _compose_aggregate_reasoning(
    entries: list[dict[str, Any]],
    mean: float,
    *,
    unit_singular: str,
    unit_plural: str,
) -> str:
    """Aggregate reasoning that quotes the judge (mirrors sov_ai_dynamic)."""
    n = len(entries)
    if n == 1:
        only = entries[0]
        if only["reasoning"]:
            return f"[{only['tag']}] {only['reasoning']}" if only["tag"] else only["reasoning"]
        return (
            f"score={_score_text(only['score'])} on {unit_singular} '{only['tag']}'"
            if only["tag"]
            else f"score={_score_text(only['score'])}"
        )
    sorted_entries = sorted(entries, key=lambda e: (e["score"], not e["reasoning"]))
    worst_with_text = [e for e in sorted_entries if e["reasoning"]][:2]
    if not worst_with_text:
        return f"Mean {mean} across {n} {unit_plural}."
    bullets = "".join(
        f'\n• "{e["tag"] or "?"}" (score={_score_text(e["score"])}): {e["reasoning"]}' for e in worst_with_text
    )
    return f"Mean {mean} across {n} {unit_plural}. Lowest scoring:{bullets}"


def _score_text(score: float) -> str:
    return str(int(score)) if float(score).is_integer() else f"{score:.1f}"


def _format_premises(premises: tuple[str, ...]) -> str:
    if not premises:
        return "(none)"
    return "\n".join(f"{i + 1}. {p}" for i, p in enumerate(premises))


def _normalize_conversation(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _as_list_of_str(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str) and s]
    if not isinstance(raw, (str, bytes)):
        try:
            return [s for s in list(raw) if isinstance(s, str) and s]
        except (TypeError, ValueError):
            return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return [s for s in parsed if isinstance(s, str) and s] if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _pinned_bank_version(trajectory: dict[str, Any], locale: str) -> str | None:
    """Extract ``bank_version[locale]`` from the trajectory's simulation_outcome."""
    outcome_raw = trajectory.get("simulation_outcome")
    if not outcome_raw:
        return None
    if isinstance(outcome_raw, str):
        try:
            outcome = json.loads(outcome_raw)
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(outcome_raw, dict):
        outcome = outcome_raw
    else:
        return None
    provenance = outcome.get("provenance") or {}
    bank_version = provenance.get("bank_version") or {}
    if isinstance(bank_version, dict):
        v = bank_version.get(locale)
        return v if isinstance(v, str) else None
    return None


def _no_fact_result(fact_id: str, judge_alias: str) -> dict[str, Any]:
    """Placeholder entry for a fact_id the trajectory probed but the bank doesn't contain."""
    return {
        "fact_id": fact_id,
        "fact_category": None,
        "fact_question_type": None,
        "fact_difficulty": None,
        "fact_placeholder": None,
        "scores": {},
        "error": "fact_not_in_bank",
    }


def _error_result(
    *,
    fact: Fact,
    judge_alias: str,
    error: str,
) -> dict[str, Any]:
    """Per-fact entry for a fact whose judge call failed. Scores are all None."""
    axes = _FACTUAL_RECALL_AXES if fact.question_type == "factual_recall" else _COMPLETION_AXES
    return {
        "fact_id": fact.id,
        "fact_category": fact.category,
        "fact_question_type": fact.question_type,
        "fact_difficulty": fact.difficulty,
        "fact_placeholder": fact.placeholder,
        "scores": {s.name: {"score": None, "reasoning": error} for s in axes},
        "error": error,
    }


# Auto-register on import so the evaluator picks this up once the
# module is listed in DEFAULT_SCORER_MODULES.
register_scorer("sov_ai_facts", score_sov_ai_facts_trajectory)
