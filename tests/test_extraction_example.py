# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic training-data extraction example — schema-sufficiency contract.

The 03 notebook (``notebooks/03_extract_training_data.ipynb``) is a
placeholder; the public extraction API ships separately. This test
file implements **toy versions** of the two simplest extractor types
from the planned five-type taxonomy (SFT and pairwise), exercises
them on the existing synthetic fixtures, and pins the contract that
the trajectory + evaluator schemas are sufficient to drive these
extractors.

Why this matters now:

- If the schemas were missing a field that an extractor would need
  (e.g., no ``persona_uuid`` on the trajectory, no per-axis-per-judge
  breakdown on the evaluator cell, no ``trajectory_id`` join key), we
  would discover it here — *before* anyone outside this repo writes
  extraction code against the wrong assumptions.
- The extractor functions defined in this file are intentionally
  **not** exported from a public module. Publishing the API is the 03
  notebook's job once the open design questions are
  answered (threshold tuning, pairwise sourcing, governance). Pinning
  the contract via tests now is the cheap version.

Out of scope here: verifier / PRM / online-RL extractors. They need
richer fixtures (per-turn judge data, branching) that the placeholder
notebook does not yet ship.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Helpers — toy extractors. These are illustrative; the public extractors
# live in the extraction module the 03 notebook will publish.
# ---------------------------------------------------------------------------


def _decode_json(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _per_axis_ensemble_mean(eval_cell: Any, axis: str) -> Optional[float]:
    """Return the mean score across all judges for one axis, or None.

    Returns None if the cell is missing, skipped, or doesn't carry the
    axis. Skipping (`skipped=True`) is treated as "no signal" rather
    than zero so it doesn't poison aggregations.
    """
    cell = _decode_json(eval_cell)
    if not cell or cell.get("skipped"):
        return None
    axis_block = (cell.get("axes") or {}).get(axis)
    if not isinstance(axis_block, dict) or not axis_block:
        return None
    scores: List[float] = []
    for judge_block in axis_block.values():
        if not isinstance(judge_block, dict):
            continue
        s = judge_block.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))
    if not scores:
        return None
    return sum(scores) / len(scores)


def _last_user_assistant_pair(messages: Any) -> Optional[Tuple[str, str]]:
    """Pull the last (user, assistant) turn from a conversation, or None.

    The simplest SFT format: a single (prompt, response) pair per
    trajectory. Multi-turn variants will use ``prompt`` = full
    conversation prefix and ``response`` = final assistant turn.
    """
    msgs = _decode_json(messages)
    if not isinstance(msgs, list):
        return None
    last_assistant_idx: Optional[int] = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None or last_assistant_idx == 0:
        return None
    user_msg = msgs[last_assistant_idx - 1]
    if user_msg.get("role") != "user":
        return None
    return (str(user_msg.get("content", "")), str(msgs[last_assistant_idx].get("content", "")))


def extract_sft_examples(
    joined,
    *,
    axis: str,
    min_score: float,
) -> List[Dict[str, Any]]:
    """Emit (prompt, response, provenance) records for high-scoring trajectories.

    Filters:
    - simulation_outcome.status == "ok" or "completed_with_warnings"
      (anything that ran to completion, not just gate-exhausted runs)
    - evaluator's per-axis ensemble mean on ``axis`` >= ``min_score``
    - the trajectory has at least one (user, assistant) turn pair
    """
    out: List[Dict[str, Any]] = []
    for _, row in joined.iterrows():
        outcome = _decode_json(row.get("simulation_outcome")) or {}
        if outcome.get("status") not in {"ok", "completed_with_warnings"}:
            continue
        score = _per_axis_ensemble_mean(row.get("assistant_eval"), axis)
        if score is None or score < min_score:
            continue
        pair = _last_user_assistant_pair(row.get("conversation_messages"))
        if pair is None:
            continue
        out.append(
            {
                "prompt": pair[0],
                "response": pair[1],
                "axis": axis,
                "score": score,
                "trajectory_id": row.get("trajectory_id"),
                "persona_uuid": row.get("persona_uuid"),
                "provenance": {
                    "code_sha": (outcome.get("provenance") or {}).get("code_sha"),
                    "probe_family": row.get("probe_family"),
                    "probe_variant": row.get("probe_variant"),
                    "scenario_prompt_version": (outcome.get("provenance") or {}).get("scenario_prompt_version"),
                    "locale": row.get("locale"),
                },
            }
        )
    return out


def extract_pairwise_examples(
    joined,
    *,
    axis: str,
    group_keys: Tuple[str, ...],
    min_score_delta: float = 1.0,
) -> List[Dict[str, Any]]:
    """Emit (prompt, chosen, rejected) pairs from matched trajectories.

    Within each group (defined by ``group_keys``, typically
    ``("persona_uuid", "probe_variant")``), pair every trajectory
    with every other trajectory in the same group whose score on
    ``axis`` differs by at least ``min_score_delta``. The higher-scored
    trajectory becomes ``chosen``; the lower becomes ``rejected``. The
    prompt is taken from the higher-scored trajectory's last user turn
    (assumed to be representative of the group's question — in
    practice the 03 notebook will join on a per-turn key).
    """
    out: List[Dict[str, Any]] = []
    enriched = joined.copy()
    enriched["_score"] = enriched["assistant_eval"].apply(lambda c: _per_axis_ensemble_mean(c, axis))
    enriched = enriched[enriched["_score"].notna()]
    if enriched.empty:
        return out

    for _, group in enriched.groupby(list(group_keys), sort=False):
        if len(group) < 2:
            continue
        rows = group.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if abs(a["_score"] - b["_score"]) < min_score_delta:
                    continue
                hi, lo = (a, b) if a["_score"] > b["_score"] else (b, a)
                hi_pair = _last_user_assistant_pair(hi.get("conversation_messages"))
                lo_pair = _last_user_assistant_pair(lo.get("conversation_messages"))
                if hi_pair is None or lo_pair is None:
                    continue
                out.append(
                    {
                        "prompt": hi_pair[0],
                        "chosen": hi_pair[1],
                        "rejected": lo_pair[1],
                        "axis": axis,
                        "score_delta": hi["_score"] - lo["_score"],
                        "persona_uuid": hi.get("persona_uuid"),
                        "chosen_trajectory_id": hi.get("trajectory_id"),
                        "rejected_trajectory_id": lo.get("trajectory_id"),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Schema-sufficiency contract assertions
# ---------------------------------------------------------------------------


REQUIRED_TRAJECTORY_COLUMNS = {
    "trajectory_id",
    "persona_uuid",
    "probe_family",
    "probe_variant",
    "locale",
    "conversation_messages",
    "simulation_outcome",
}


REQUIRED_EVAL_CELL_KEYS = {"envelope", "axes", "scorers", "skipped"}


class TestSchemaSufficiency:
    """Verifies the trajectory + evaluator schemas carry every field the
    toy extractors above need. If a future schema change drops one of
    these columns / keys, this test fires before extractor code breaks
    in the wild."""

    def test_trajectory_schema_has_required_columns(self, trajectory_df):
        missing = REQUIRED_TRAJECTORY_COLUMNS - set(trajectory_df.columns)
        assert not missing, f"trajectory schema missing required columns: {missing}"

    def test_evaluator_cell_has_required_keys(self, evaluator_df):
        for cell_str in evaluator_df["assistant_eval"]:
            cell = json.loads(cell_str)
            missing = REQUIRED_EVAL_CELL_KEYS - set(cell.keys())
            assert not missing, f"eval cell missing keys: {missing}"

    def test_evaluator_envelope_carries_versioning(self, evaluator_df):
        # Without prompt_version + evaluator_version we cannot detect
        # cross-run drift. Pin presence here.
        for cell_str in evaluator_df["assistant_eval"]:
            envelope = json.loads(cell_str)["envelope"]
            assert "prompt_version" in envelope
            assert "judge_aliases" in envelope
            assert "axes" in envelope


# ---------------------------------------------------------------------------
# SFT extraction
# ---------------------------------------------------------------------------


class TestSFTExtraction:
    """The simplest extractor: keep trajectories that scored well on a
    chosen axis, emit one (prompt, response, provenance) record each."""

    def _join(self, trajectory_df, evaluator_df):
        # evaluator_df is built from trajectory_df and shares all columns,
        # so the join is effectively a no-op pass-through. Kept explicit
        # to mirror what the 03 notebook will do with two parquets.
        return evaluator_df.merge(trajectory_df[["trajectory_id"]], on="trajectory_id", how="inner").merge(
            evaluator_df.drop(columns=[c for c in evaluator_df.columns if c == "trajectory_id"]),
            left_index=True,
            right_index=True,
            suffixes=("", "_dup"),
        )

    def test_threshold_filters(self, trajectory_df, evaluator_df):
        joined = evaluator_df  # already enriched with all trajectory columns
        # Helpfulness axis means in the fixture (skipped rows excluded):
        #   row 0 (general_open_ended OK):     (5+4)/2 = 4.5
        #   row 2 (general_open_ended warn):   (2+3)/2 = 2.5
        #   row 3 (tool_calling OK):   (5+5)/2 = 5.0
        #   row 4 (tool_calling skip): SKIPPED -> excluded
        #   row 5 (tool_calling OK):   (4+4)/2 = 4.0
        thresholds_to_count = {3.0: 3, 4.0: 3, 4.5: 2, 5.0: 1, 6.0: 0}
        for thresh, expected in thresholds_to_count.items():
            out = extract_sft_examples(joined, axis="helpfulness", min_score=thresh)
            assert len(out) == expected, (
                f"helpfulness >= {thresh} expected {expected} examples, "
                f"got {len(out)}: {[o['trajectory_id'] for o in out]}"
            )

    def test_skipped_and_failed_rows_excluded(self, evaluator_df):
        out = extract_sft_examples(evaluator_df, axis="helpfulness", min_score=0.0)
        ids = {o["trajectory_id"] for o in out}
        # Failed row (idx 1) excluded by status filter.
        assert "t000000000000002" not in ids
        # Skipped row (idx 4) excluded by score=None filter.
        assert "t000000000000005" not in ids

    def test_axis_selection(self, evaluator_df):
        # Accuracy on the confrontational row was higher than helpfulness
        # (4.0 vs 2.5), so different axis -> different selection.
        helpful = extract_sft_examples(evaluator_df, axis="helpfulness", min_score=4.0)
        accurate = extract_sft_examples(evaluator_df, axis="accuracy", min_score=4.0)
        helpful_ids = {o["trajectory_id"] for o in helpful}
        accurate_ids = {o["trajectory_id"] for o in accurate}
        # Confrontational row (t...3): helpfulness 2.5, accuracy 4.0.
        assert "t000000000000003" not in helpful_ids
        assert "t000000000000003" in accurate_ids

    def test_unknown_axis_returns_empty(self, evaluator_df):
        out = extract_sft_examples(evaluator_df, axis="not_an_axis", min_score=0.0)
        assert out == []

    def test_record_shape(self, evaluator_df):
        out = extract_sft_examples(evaluator_df, axis="helpfulness", min_score=4.5)
        assert out, "expected at least one example"
        for rec in out:
            for k in ("prompt", "response", "axis", "score", "trajectory_id", "persona_uuid", "provenance"):
                assert k in rec, f"missing key {k}"
            assert rec["axis"] == "helpfulness"
            assert isinstance(rec["score"], float)
            assert isinstance(rec["prompt"], str) and rec["prompt"]
            assert isinstance(rec["response"], str) and rec["response"]
            for k in ("code_sha", "probe_family", "probe_variant", "locale"):
                assert k in rec["provenance"], f"missing provenance key {k}"

    def test_drift_detectable_via_provenance(self, evaluator_df):
        """The provenance block must let downstream code spot rows
        produced by a different code version — required for safe
        training-data curation."""
        out = extract_sft_examples(evaluator_df, axis="accuracy", min_score=0.0)
        shas = {rec["provenance"]["code_sha"] for rec in out}
        # The fixture intentionally seeds two distinct shas across rows
        # so this assertion fires when we lose code_sha during refactors.
        assert len(shas) >= 2, (
            "extracted SFT records all share the same code_sha; if the "
            "fixture really has uniform provenance the test is wrong, "
            "but more likely the extractor has lost the field"
        )


# ---------------------------------------------------------------------------
# Pairwise extraction
# ---------------------------------------------------------------------------


def _make_pairwise_fixture():
    """Build a tiny pairwise-ready frame: 2 personas × 2 model variants.

    Each persona is run once against assistant_a (high score) and once
    against assistant_b (low score), with a third persona getting
    only one trajectory (so it produces no pair).
    """
    pd = pytest.importorskip("pandas")

    # Reuse the existing _outcome / _messages helpers via direct JSON
    # construction so the fixture is self-contained.
    def msgs(u, a):
        return json.dumps(
            [
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
            ]
        )

    base_envelope = {
        "judge_aliases": ["judge_a", "judge_b"],
        "judge_families": ["openai", "nvidia_nemotron"],
        "axes": ["helpfulness"],
        "scorers": [],
        "prompt_version": "v1.0",
        "evaluator_version": "v1.0",
    }

    def cell(score_a, score_b):
        return json.dumps(
            {
                "envelope": base_envelope,
                "axes": {
                    "helpfulness": {
                        "judge_a": {"score": score_a, "reasoning": "x"},
                        "judge_b": {"score": score_b, "reasoning": "y"},
                    },
                },
                "scorers": {},
                "skipped": False,
                "skipped_reason": None,
            }
        )

    def outcome():
        return json.dumps(
            {
                "status": "ok",
                "failure_class": None,
                "failure_attribution": None,
                "failure_detail": "",
                "n_turns": 1,
                "n_tool_calls": 0,
                "n_user_query_attempts": 1,
                "n_user_followup_retries": 0,
                "n_assistant_inline_failures": 0,
                "n_api_response_rerolls": 0,
                "n_fourth_wall_triggers": 0,
                "n_user_role_violations": 0,
                "warnings": [],
                "early_stop": False,
                "per_model_input_tokens": {},
                "per_model_output_tokens": {},
                "per_model_calls": {},
                "wall_clock_s_by_alias": {},
                "wall_clock_s": 1.0,
                "provenance": {
                    "code_sha": "abc",
                    "nemotron_personas_version": "v",
                    "scenario_prompt_version": "v1.0",
                    "bank_version": {},
                },
            }
        )

    rows = []
    # Persona A — clear preference (chosen vs rejected)
    rows.append(
        {
            "trajectory_id": "tA_high",
            "persona_uuid": "pA",
            "probe_family": "general_open_ended",
            "probe_variant": "default",
            "locale": "en_US",
            "assistant_model_id": "assistant_v2",
            "conversation_messages": msgs("Help me with X.", "Here's a great answer."),
            "simulation_outcome": outcome(),
            "assistant_eval": cell(5, 5),
        }
    )
    rows.append(
        {
            "trajectory_id": "tA_low",
            "persona_uuid": "pA",
            "probe_family": "general_open_ended",
            "probe_variant": "default",
            "locale": "en_US",
            "assistant_model_id": "assistant_v1",
            "conversation_messages": msgs("Help me with X.", "I don't know."),
            "simulation_outcome": outcome(),
            "assistant_eval": cell(2, 3),
        }
    )

    # Persona B — identical scores (no informative pair)
    rows.append(
        {
            "trajectory_id": "tB_high",
            "persona_uuid": "pB",
            "probe_family": "general_open_ended",
            "probe_variant": "default",
            "locale": "en_US",
            "assistant_model_id": "assistant_v2",
            "conversation_messages": msgs("Question.", "Same answer."),
            "simulation_outcome": outcome(),
            "assistant_eval": cell(4, 4),
        }
    )
    rows.append(
        {
            "trajectory_id": "tB_low",
            "persona_uuid": "pB",
            "probe_family": "general_open_ended",
            "probe_variant": "default",
            "locale": "en_US",
            "assistant_model_id": "assistant_v1",
            "conversation_messages": msgs("Question.", "Same answer."),
            "simulation_outcome": outcome(),
            "assistant_eval": cell(4, 4),
        }
    )

    # Persona C — only one trajectory (no pair possible)
    rows.append(
        {
            "trajectory_id": "tC_only",
            "persona_uuid": "pC",
            "probe_family": "general_open_ended",
            "probe_variant": "default",
            "locale": "en_US",
            "assistant_model_id": "assistant_v2",
            "conversation_messages": msgs("Solo.", "Answer."),
            "simulation_outcome": outcome(),
            "assistant_eval": cell(3, 3),
        }
    )

    return pd.DataFrame(rows)


@pytest.fixture
def pairwise_fixture():
    return _make_pairwise_fixture()


class TestPairwiseExtraction:
    def test_emits_chosen_rejected_for_score_gap(self, pairwise_fixture):
        out = extract_pairwise_examples(
            pairwise_fixture,
            axis="helpfulness",
            group_keys=("persona_uuid", "probe_variant"),
            min_score_delta=1.0,
        )
        # Persona A delta = 5.0 - 2.5 = 2.5 → 1 pair.
        # Persona B delta = 0 → no pair.
        # Persona C single trajectory → no pair.
        assert len(out) == 1
        rec = out[0]
        assert rec["persona_uuid"] == "pA"
        assert rec["chosen_trajectory_id"] == "tA_high"
        assert rec["rejected_trajectory_id"] == "tA_low"
        assert rec["score_delta"] == pytest.approx(2.5)
        assert rec["chosen"] == "Here's a great answer."
        assert rec["rejected"] == "I don't know."

    def test_threshold_suppresses_close_pairs(self, pairwise_fixture):
        # Bumping the delta requirement above the actual gap should
        # eliminate the persona A pair too.
        out = extract_pairwise_examples(
            pairwise_fixture,
            axis="helpfulness",
            group_keys=("persona_uuid", "probe_variant"),
            min_score_delta=3.0,
        )
        assert out == []

    def test_no_cross_persona_pairs(self, pairwise_fixture):
        # Group key is (persona_uuid, probe_variant), so trajectories
        # for different personas should never be paired even if their
        # scores look pair-able in isolation.
        out = extract_pairwise_examples(
            pairwise_fixture,
            axis="helpfulness",
            group_keys=("persona_uuid", "probe_variant"),
            min_score_delta=0.0,  # accept any non-zero delta
        )
        ids = {(rec["chosen_trajectory_id"], rec["rejected_trajectory_id"]) for rec in out}
        # No pair should mix personas A and B.
        for chosen, rejected in ids:
            chosen_persona = chosen.split("_")[0]
            rejected_persona = rejected.split("_")[0]
            assert chosen_persona == rejected_persona, f"pair {chosen}/{rejected} mixes personas across the group key"

    def test_record_shape(self, pairwise_fixture):
        out = extract_pairwise_examples(
            pairwise_fixture,
            axis="helpfulness",
            group_keys=("persona_uuid", "probe_variant"),
            min_score_delta=1.0,
        )
        for rec in out:
            for k in (
                "prompt",
                "chosen",
                "rejected",
                "axis",
                "score_delta",
                "persona_uuid",
                "chosen_trajectory_id",
                "rejected_trajectory_id",
            ):
                assert k in rec, f"missing key {k}"
            assert rec["score_delta"] > 0
            assert rec["chosen"] != rec["rejected"]
