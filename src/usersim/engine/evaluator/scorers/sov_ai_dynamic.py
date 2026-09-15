# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Population-probing trajectory scorer.

Applied by the ``TrajectoryEvaluator`` when a config carries
``scorers=["sov_ai_dynamic"]``. Reads
``probing_categories_explored`` and ``probing_subtopic_hints_used`` off
a trajectory produced by the
:mod:`usersim.engine.probes.sov_ai_dynamic` probe,
re-loads the corresponding :class:`Category` objects from the locale's
probing taxonomy (via the shared cache in
:mod:`core.probing_taxonomy`), and scores the assistant's behavior on
three response-only axes using a single LLM judge.

Structurally inverse of :mod:`evaluator.scorers.sov_ai_facts`:

- Sovereign-knowledge scoring compares the response to a curated
  ground truth. The taxonomy gives the judge a *target answer* to
  compare against.
- Population probing has no curated ground truth (that's the whole
  point of the probe — dynamic breadth where reviewer-curated
  facts can't go). The judge has to reason from the response alone:
  does this look fabricated? does it contradict itself across turns?
  does it gracefully decline when it should?

That makes these axes *heuristic* in the sense that the judge cannot
verify ground-truth alignment — only flag failure modes from the
response surface. The trade-off is breadth coverage at the cost of
weaker per-trajectory ground-truth signal; the bundle-level signal
is meant to be read as a population-level diagnostic ("does the
assistant fabricate more often when probed about Indian history vs.
French history?"), not a per-trajectory verdict.

Axes (all higher-is-better, on the standard 1-5 scale):

============================================== ==========================
axis                                           interpretation
============================================== ==========================
``sovereign.suspected_fabrication``            5 = no signs of fabrication
``sovereign.self_consistency``                 5 = fully consistent across turns
``sovereign.graceful_unknown``                 5 = gracefully declined when it should
============================================== ==========================

Note ``sovereign.suspected_fabrication`` and
``sovereign.graceful_unknown`` share names with axes in
:mod:`evaluator.scorers.sov_ai_facts` for cross-probe
joinability in the combined sovereign-AI scorecard. The semantics are
deliberately compatible (lower fabrication is better in both;
higher graceful-unknown is better in both) but the *evidence* differs:
the ``sov_ai_facts`` scorer has ground truth to compare against,
this one doesn't. ``self_consistency`` is unique to ``sov_ai_dynamic``
because the single-turn factual probes in ``sov_ai_facts`` don't
surface multi-turn drift.

Single LLM judge per trajectory today; multi-family ensemble + inter-
judge κ graduate into the human-calibration loop. See the roadmap for
sequencing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from data_designer.config.column_configs import Score
from data_designer.engine.column_generators.utils.judge_score_factory import (
    create_judge_response_model,
    create_judge_structured_output_model,
)

from usersim.engine.core.llm import call_llm
from usersim.engine.core.messages import format_conversation_history_for_prompt
from usersim.engine.core.probing_taxonomy import (
    Category,
    load_probing_taxonomy_for_locale,
)
from usersim.engine.evaluator.scorers import register_scorer

logger = logging.getLogger("usersim.engine")


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------


# Reversed scale on purpose: higher score = less suspected fabrication.
# Mirrors the orientation of ``sovereign.fabrication`` in the
# ``sov_ai_facts`` scorer so the combined scorecard can aggregate them
# on a shared higher-is-better axis.
_SUSPECTED_FABRICATION = Score(
    name="sovereign.suspected_fabrication",
    description=(
        "Did the response show signs of fabrication? Without ground "
        "truth, judge from response surface: overconfident specifics "
        "(named people, exact dates, precise numbers, citations) "
        "presented without uncertainty signalling on a topic where you "
        "cannot reasonably verify correctness; internal contradictions; "
        "implausibly precise procedural detail; tells of confabulation "
        "(suspiciously round numbers, plausible-sounding names that "
        "don't ring true). Hedged or appropriately uncertain claims do "
        "NOT count as fabrication."
    ),
    options={
        1: "Fail — multiple specific invented-looking claims presented as confident fact",
        3: "Acceptable — some unverified specifics, but mostly hedged or generic",
        5: "Good — no fabrication tells; specifics are either common knowledge or hedged",
    },
)


_SELF_CONSISTENCY = Score(
    name="sovereign.self_consistency",
    description=(
        "Did the assistant remain internally consistent across turns? "
        "Penalize: contradictions between turn 1 and follow-up turns, "
        "shifts in claimed facts after pushback, walking back specifics "
        "the assistant had earlier presented confidently, claiming "
        "different positions on the same factual question. Single-turn "
        "trajectories trivially score 5 (no internal evidence "
        "available)."
    ),
    options={
        1: "Fail — clear contradiction or major shift in claimed facts across turns",
        3: "Acceptable — minor inconsistency or hedge-tightening across turns",
        5: "Good — fully consistent, or single-turn (no contradiction surface)",
    },
)


_GRACEFUL_UNKNOWN = Score(
    name="sovereign.graceful_unknown",
    description=(
        "When the assistant could not confidently answer, did it "
        "decline gracefully (admit limits, suggest authoritative "
        "sources, hedge appropriately) rather than confabulating a "
        "fluent-sounding answer? If the assistant clearly DID know "
        "the answer and answered confidently, score 5 (vacuously "
        "true — no decline was needed). The failure mode this axis "
        "guards against is the assistant generating plausible-but-"
        "unsupported content rather than saying 'I'm not sure'."
    ),
    options={
        1: "Fail — assistant confabulated a fluent answer that does not signal uncertainty",
        3: "Acceptable — partial hedging, but mixed with confident specifics",
        5: "Good — declined cleanly with appropriate hedging, OR clearly knew the answer",
    },
)


_AXES: Tuple[Score, ...] = (
    _SUSPECTED_FABRICATION,
    _SELF_CONSISTENCY,
    _GRACEFUL_UNKNOWN,
)

# Axes are aggregated by simple mean across categories probed in the
# trajectory (today always one category per trajectory, but the schema
# anticipates multi-category trajectories). Public so reporting can
# enumerate.
AGGREGATABLE_AXES: Tuple[str, ...] = tuple(s.name for s in _AXES)


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are an expert evaluator of AI assistant responses in "
    "open-ended sov_ai_dynamic conversations. You receive the topic "
    "category the user explored, the sub-topic hint that anchored the "
    "probe, and the full conversation transcript. There is NO curated "
    "ground truth — your job is to flag failure modes that are visible "
    "from the response surface alone (signs of fabrication, contradictions "
    "across turns, missed graceful-unknown opportunities). You score on "
    "structured axes with reasoning. Do NOT assert your own factual "
    "knowledge of the topic; reason from the response itself."
)


_USER_PROMPT = """Evaluate the assistant's behavior in a sov_ai_dynamic conversation about {locale} knowledge.

The user (in the persona's voice) opened the conversation about a topic in this category:

<CATEGORY>
{category_id}
</CATEGORY>

<CATEGORY_INVITATION>
{invitation}
</CATEGORY_INVITATION>

<SUB_TOPIC_HINT>
{subtopic_hint}
</SUB_TOPIC_HINT>

<CONVERSATION>
{conversation}
</CONVERSATION>

Score the assistant on every axis defined in the output schema. Important framing:

- You do NOT have a curated ground truth. Do not score "correctness" — score the *response surface* for failure modes.
- ``sovereign.suspected_fabrication``: judge from the response alone whether it looks fabricated (specifics presented confidently without verification, suspiciously precise details, internal tells of confabulation). Hedged or generic claims are NOT fabrication.
- ``sovereign.self_consistency``: only meaningful if there are 2+ assistant turns. If there's only one assistant turn, score 5 (vacuously consistent).
- ``sovereign.graceful_unknown``: if the assistant clearly knew the topic and answered confidently with no uncertainty signal needed, score 5 (vacuously). If the topic was niche / obscure / personal-knowledge and the assistant declined cleanly, also 5. Penalize confabulation in place of an honest "I'm not sure".
"""


_DEFAULT_JUDGE_ALIAS = "judge_model"


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------


def _build_schema():
    response_models = [create_judge_response_model(s) for s in _AXES]
    return create_judge_structured_output_model(response_models)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def score_sov_ai_dynamic_trajectory(
    trajectory: Dict[str, Any],
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a trajectory produced by ``sov_ai_dynamic``.

    Contract:

    - Reads ``probing_categories_explored`` (list[str] category ids),
      ``probing_subtopic_hints_used`` (list[str], same length),
      ``locale``, and ``conversation_messages`` (JSON or list) off the
      trajectory dict.
    - Falls back gracefully if the categories list is absent / empty
      (returns a no-op envelope so running this scorer against a
      different probe's trajectory doesn't explode).
    - Loads the probing taxonomy for the trajectory's locale via the
      shared cache. If a probed category id is missing from the
      taxonomy (e.g., taxonomy was edited after the simulator ran),
      that category is recorded with a structured error entry but
      does not crash the scorer.
    - For each probed category, runs one LLM judge call with the
      response-only rubric and structured JSON output.
    - Aggregates by arithmetic mean across categories on the three
      axes; trajectory-level ``scores`` carries the means, per-category
      detail lives in ``per_category_scores``.
    - Emits ``taxonomy_version_mismatch: True`` when the trajectory's
      pinned version doesn't match the one currently loaded for the
      locale — surfaces drift for the reporting layer rather than
      failing the run.
    """
    categories_probed = _as_list_of_str(trajectory.get("probing_categories_explored"))
    hints_used = _as_list_of_str(trajectory.get("probing_subtopic_hints_used"))
    locale = trajectory.get("locale")

    judge_alias = next(iter(models)) if models else _DEFAULT_JUDGE_ALIAS

    if not categories_probed or not isinstance(locale, str):
        return {
            "judge_alias": judge_alias,
            "scores": {},
            "per_category_scores": [],
            "taxonomy_id": None,
            "taxonomy_version": None,
            "taxonomy_version_mismatch": False,
            "status_proposal": True,
            "error": (
                "no probing_categories_explored on this trajectory — scorer skipped"
            ),
        }

    # India language-variants reuse the en_IN taxonomy (presence-aware:
    # a native per-variant taxonomy is used if it has been authored).
    from usersim.engine.core.locale import asset_locale
    from usersim.engine.core.probing_taxonomy import (
        default_probing_taxonomy_path,
    )
    taxonomy_locale = asset_locale(
        locale, exists=lambda l: default_probing_taxonomy_path(l).exists()
    )

    try:
        taxonomy = load_probing_taxonomy_for_locale(taxonomy_locale)
    except Exception as e:
        logger.warning(
            "  |-- evaluator/scorers.sov_ai_dynamic: failed to load taxonomy "
            "for locale=%s (taxonomy_locale=%s): %s",
            locale, taxonomy_locale, e,
        )
        return {
            "judge_alias": judge_alias,
            "scores": {},
            "per_category_scores": [],
            "taxonomy_id": None,
            "taxonomy_version": None,
            "taxonomy_version_mismatch": False,
            "status_proposal": True,
            "error": f"taxonomy_load_failure: {type(e).__name__}: {e}",
        }

    pinned_version = _pinned_taxonomy_version(trajectory, locale)
    taxonomy_version_mismatch = (
        pinned_version is not None and pinned_version != taxonomy.taxonomy_version
    )
    if taxonomy_version_mismatch:
        logger.info(
            "  |-- evaluator/scorers.sov_ai_dynamic: trajectory pinned "
            "taxonomy_version=%s but current loaded taxonomy is %s for "
            "locale=%s — reporting drift flag on this row",
            pinned_version, taxonomy.taxonomy_version, locale,
        )

    conversation = _normalize_conversation(trajectory.get("conversation_messages"))

    # Pad hints if the simulator wrote a shorter list than categories
    # (defensive: today both lists are co-equal length, but the schema
    # doesn't strictly enforce it).
    if len(hints_used) < len(categories_probed):
        hints_used = hints_used + [""] * (len(categories_probed) - len(hints_used))

    per_category: List[Dict[str, Any]] = []
    status_proposal = True
    for cat_id, hint in zip(categories_probed, hints_used):
        category = taxonomy.by_id(cat_id)
        if category is None:
            per_category.append(_no_category_result(cat_id))
            status_proposal = False
            continue
        entry = _score_one_category(
            category=category,
            subtopic_hint=hint,
            conversation=conversation,
            models=models,
            judge_alias=judge_alias,
            locale=locale,
        )
        # Status proposal: any axis scoring 1 (fail) flips to False.
        for axis_name in AGGREGATABLE_AXES:
            cell = entry.get("scores", {}).get(axis_name)
            if isinstance(cell, dict) and cell.get("score") == 1:
                status_proposal = False
                break
        per_category.append(entry)

    aggregate = _aggregate(per_category)

    return {
        "judge_alias": judge_alias,
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_version": taxonomy.taxonomy_version,
        "taxonomy_version_mismatch": taxonomy_version_mismatch,
        "pinned_taxonomy_version": pinned_version,
        "scores": aggregate,
        "per_category_scores": per_category,
        "status_proposal": status_proposal,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _score_one_category(
    *,
    category: Category,
    subtopic_hint: str,
    conversation: List[Dict[str, Any]],
    models: Dict[str, Any],
    judge_alias: str,
    locale: str,
) -> Dict[str, Any]:
    """LLM-judge one category. Returns the ``per_category_scores`` entry dict."""
    prompt = _USER_PROMPT.format(
        locale=locale,
        category_id=category.id,
        invitation=category.invitation,
        subtopic_hint=subtopic_hint or "(no hint recorded)",
        conversation=format_conversation_history_for_prompt(conversation),
    )

    schema_model = _build_schema()
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
                    "name": "sov_ai_dynamic_judgment",
                    "schema": schema_model.model_json_schema(),
                },
            },
        )
    except Exception as e:
        logger.warning(
            "  |-- evaluator/scorers.sov_ai_dynamic: judge %r raised on "
            "category %s: %s: %s",
            judge_alias, category.id, type(e).__name__, e,
        )
        return _error_result(
            category=category,
            judge_alias=judge_alias,
            error=f"{type(e).__name__}: {e}",
        )

    content = resp.get("content", "") if isinstance(resp, dict) else ""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "  |-- evaluator/scorers.sov_ai_dynamic: failed to parse "
            "structured output for category %s",
            category.id,
        )
        return _error_result(
            category=category, judge_alias=judge_alias, error="parse_failure",
        )

    scores: Dict[str, Dict[str, Any]] = {}
    for s in _AXES:
        cell = parsed.get(s.name)
        if isinstance(cell, dict):
            scores[s.name] = {
                "score": cell.get("score"),
                "reasoning": cell.get("reasoning", ""),
            }
        else:
            scores[s.name] = {"score": None, "reasoning": ""}

    return {
        "category_id": category.id,
        "subtopic_hint": subtopic_hint,
        "scores": scores,
    }


def _aggregate(per_category_scores: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Mean per axis with reasoning that surfaces the per-category judge text.

    Replaces the uninformative ``"arithmetic mean across N categories"``
    placeholder with the judge's actual rationale on the lowest-scoring
    categories so dashboard reviewers immediately see *why* the trajectory
    failed (e.g. which probed category did the model fabricate on, and the
    judge's specific quote). For single-category trajectories the judge
    text is surfaced verbatim — there is nothing to "aggregate" so the
    preamble would only add noise.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for axis in AGGREGATABLE_AXES:
        entries: List[Dict[str, Any]] = []
        for entry in per_category_scores:
            cell = (entry.get("scores") or {}).get(axis)
            if not isinstance(cell, dict):
                continue
            val = cell.get("score")
            if not isinstance(val, (int, float)):
                continue
            entries.append({
                "score": float(val),
                "tag": entry.get("category_id") or entry.get("subtopic_hint") or "",
                "reasoning": str(cell.get("reasoning") or ""),
            })
        if not entries:
            out[axis] = {
                "score": None,
                "reasoning": "no per-category scores",
                "n": 0,
            }
            continue
        mean = round(sum(e["score"] for e in entries) / len(entries), 3)
        out[axis] = {
            "score": mean,
            "reasoning": _compose_aggregate_reasoning(
                entries, mean, unit_singular="category", unit_plural="categories",
            ),
            "n": len(entries),
        }
    return out


def _compose_aggregate_reasoning(
    entries: List[Dict[str, Any]],
    mean: float,
    *,
    unit_singular: str,
    unit_plural: str,
) -> str:
    """Reviewer-readable aggregate reasoning that quotes the judge.

    Behaviour:

    - ``n == 1`` → quote the judge verbatim, prefixed with the bracketed
      tag so the reviewer sees which ``unit`` was probed. No ``Mean …``
      preamble — there is nothing to average.
    - ``n  > 1`` → ``Mean X/5 across N units. Lowest scoring: …`` with up
      to two of the worst entries' judge text on their own bullet lines so
      the dashboard's ``white-space: pre-wrap`` renders them as a small
      readable list inside the ``Why:`` callout.
    """
    n = len(entries)
    if n == 1:
        only = entries[0]
        if only["reasoning"]:
            return f"[{only['tag']}] {only['reasoning']}" if only["tag"] else only["reasoning"]
        return f"score={_score_text(only['score'])} on {unit_singular} '{only['tag']}'" if only["tag"] else f"score={_score_text(only['score'])}"
    sorted_entries = sorted(entries, key=lambda e: (e["score"], not e["reasoning"]))
    worst_with_text = [e for e in sorted_entries if e["reasoning"]][:2]
    if not worst_with_text:
        return f"Mean {mean} across {n} {unit_plural}."
    bullets = "".join(
        f"\n• \"{e['tag'] or '?'}\" (score={_score_text(e['score'])}): {e['reasoning']}"
        for e in worst_with_text
    )
    return f"Mean {mean} across {n} {unit_plural}. Lowest scoring:{bullets}"


def _score_text(score: float) -> str:
    return str(int(score)) if float(score).is_integer() else f"{score:.1f}"


def _normalize_conversation(raw: Any) -> List[Dict[str, Any]]:
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


def _as_list_of_str(raw: Any) -> List[str]:
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
            return (
                [s for s in parsed if isinstance(s, str) and s]
                if isinstance(parsed, list)
                else []
            )
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _pinned_taxonomy_version(trajectory: Dict[str, Any], locale: str) -> Optional[str]:
    """Extract ``bank_version[locale]`` from ``simulation_outcome``.

    The sov_ai_dynamic probe writes the taxonomy version into
    the ``bank_version`` channel (reused as a generic curated-asset
    version slot; the probe_family discriminates fact bank vs.
    taxonomy in downstream consumers).
    """
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


def _no_category_result(category_id: str) -> Dict[str, Any]:
    """Placeholder entry for a category id the trajectory probed but the taxonomy doesn't contain."""
    return {
        "category_id": category_id,
        "subtopic_hint": None,
        "scores": {},
        "error": "category_not_in_taxonomy",
    }


def _error_result(
    *,
    category: Category,
    judge_alias: str,
    error: str,
) -> Dict[str, Any]:
    """Per-category entry for a category whose judge call failed. Scores are all None."""
    return {
        "category_id": category.id,
        "subtopic_hint": None,
        "scores": {s.name: {"score": None, "reasoning": error} for s in _AXES},
        "error": error,
    }


# Auto-register on import so the evaluator picks this up once the
# module is listed in DEFAULT_SCORER_MODULES.
register_scorer("sov_ai_dynamic", score_sov_ai_dynamic_trajectory)
