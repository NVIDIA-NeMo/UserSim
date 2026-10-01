# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``usersim rescore`` — recomputing deterministic scorers in place.

The whole point of this command is that it is cheap and non-destructive: it
recomputes the model-free scorers over an existing eval run without paying for
the judge ensemble again, and without disturbing anything it was not asked
about. Both halves of that are tested here, because either one silently
breaking would be expensive: losing the judge axes means a real re-eval, and
letting a hybrid scorer through means spending money on what advertises itself
as an offline operation.
"""

from __future__ import annotations

import argparse
import json

import pytest

from usersim.cli.rescore import DETERMINISTIC_SCORERS, run


def _judge_axes():
    return {
        "helpfulness": {"judge_model": {"score": 4, "reasoning": "solid"}},
        "language_appropriateness": {"judge_model": {"score": 2, "reasoning": "off"}},
    }


def _eval_cell():
    """An eval cell whose language_compliance skipped, as the pre-fix scorer did."""
    return json.dumps(
        {
            "envelope": {
                "judge_aliases": ["judge_model"],
                "axes": ["helpfulness", "language_appropriateness"],
                "scorers": ["language_compliance"],
                "prompt_version": "v1.0",
                "evaluator_version": "v1.0",
            },
            "axes": _judge_axes(),
            "scorers": {
                "language_compliance": {
                    "scorer_kind": "deterministic",
                    "scores": {},
                    "error": "locale='kn_Knda_IN' not registered — scorer skipped",
                },
                # A scorer we will NOT ask about; must come back untouched.
                "tool_use": {"scores": {"overall": {"score": 5}}, "status_proposal": True},
            },
            "skipped": False,
            "skipped_reason": None,
        }
    )


_KANNADA = (
    "\u0c97\u0ccd\u0cb0\u0cbe\u0cb9\u0c95\u0cb0 \u0c96\u0cbe\u0ca4\u0cc6\u0caf \u0cb5\u0cbf\u0cb5\u0cb0 \u0caa\u0ca1\u0cc6\u0caf\u0cb2\u0cc1 \u0ca8\u0cbe\u0cb5\u0cc1 \u0caa\u0cb0\u0cbf\u0cb6\u0cc0\u0cb2\u0cbf\u0cb8\u0cbf "
    * 5
)


@pytest.fixture()
def run_on_disk(tmp_path):
    """A one-row eval run plus its trajectory, laid out as the CLI expects."""
    pd = pytest.importorskip("pandas")
    from usersim.engine.core.storage import write_run_partition

    run_id = "1700000000"
    evaluations = tmp_path / "evaluations"
    trajectories = tmp_path / "trajectories"

    eval_df = pd.DataFrame(
        {
            "trajectory_id": ["t1"],
            "locale": ["kn_Knda_IN"],
            "probe_family": ["general_open_ended"],
            "assistant_eval": [_eval_cell()],
        }
    )
    traj_df = pd.DataFrame(
        {
            "trajectory_id": ["t1"],
            "locale": ["kn_Knda_IN"],
            "probe_family": ["general_open_ended"],
            "conversation_messages": [
                json.dumps(
                    [
                        {"role": "user", "content": "\u0cb9\u0cc7\u0c97\u0cbf\u0ca6\u0ccd\u0ca6\u0cc0\u0cb0\u0cbf"},
                        {"role": "assistant", "content": _KANNADA},
                    ],
                    ensure_ascii=False,
                )
            ],
        }
    )
    write_run_partition(eval_df, evaluations, run_id)
    write_run_partition(traj_df, trajectories, run_id)
    return argparse.Namespace(
        evaluations=evaluations,
        trajectories=trajectories,
        run=run_id,
        scorers=["language_compliance"],
        eval_column="assistant_eval",
        dry_run=False,
    )


def _read_cell(evaluations, run_id="1700000000"):
    pytest.importorskip("pandas")
    from usersim.engine.core.storage import read_partitioned_dataset

    df = read_partitioned_dataset(evaluations, run=run_id)
    return json.loads(df.iloc[0]["assistant_eval"])


class TestRoundTrip:
    def test_judge_axes_survive_untouched(self, run_on_disk):
        """The judge axes are the expensive part. If a rescore could disturb
        them it would not be a cheap operation at all -- it would mean a real
        re-eval to recover them."""
        assert run(run_on_disk) == 0
        cell = _read_cell(run_on_disk.evaluations)
        assert cell["axes"] == _judge_axes()

    def test_envelope_is_left_alone(self, run_on_disk):
        """``TrajectoryEvaluatorGenerator`` skips a row whose stored envelope
        matches the config's. Rewriting it here would make the next full eval
        re-judge every patched row at full cost."""
        assert run(run_on_disk) == 0
        cell = _read_cell(run_on_disk.evaluations)
        assert cell["envelope"]["scorers"] == ["language_compliance"]
        assert cell["envelope"]["prompt_version"] == "v1.0"

    def test_scorers_not_requested_are_untouched(self, run_on_disk):
        assert run(run_on_disk) == 0
        cell = _read_cell(run_on_disk.evaluations)
        assert cell["scorers"]["tool_use"] == {
            "scores": {"overall": {"score": 5}},
            "status_proposal": True,
        }

    def test_the_target_scorer_is_recomputed(self, run_on_disk):
        """The stored block had skipped with an error; the fixed scorer scores
        the script axes for this detector-less locale."""
        assert run(run_on_disk) == 0
        block = _read_cell(run_on_disk.evaluations)["scorers"]["language_compliance"]
        assert "error" not in block
        assert block["detector_available"] is False
        assert "language.script_integrity_rate" in block["scores"]

    def test_the_recomputed_block_is_data_not_a_pending_call(self, run_on_disk):
        """A scorer block has to be the result, not the call that produces it.

        Scoring is awaited. Storing whatever the call returns without waiting
        for it writes a placeholder into the cell, and because a placeholder
        never equals the block it replaced, every row also reports as changed.
        """
        assert run(run_on_disk) == 0
        block = _read_cell(run_on_disk.evaluations)["scorers"]["language_compliance"]
        assert isinstance(block, dict), f"stored {type(block).__name__} where a scorer result belongs"
        assert block.get("scores"), "the stored block carries no scores, so nothing was actually scored"

    def test_marks_which_scorers_were_patched(self, run_on_disk):
        """Otherwise a reader hits a cell whose scorers disagree with its
        envelope and has no way to tell why."""
        assert run(run_on_disk) == 0
        assert _read_cell(run_on_disk.evaluations)["rescored"] == ["language_compliance"]

    def test_dry_run_writes_nothing(self, run_on_disk):
        run_on_disk.dry_run = True
        assert run(run_on_disk) == 0
        block = _read_cell(run_on_disk.evaluations)["scorers"]["language_compliance"]
        assert "error" in block


class TestRefusesToSpendMoney:
    @pytest.mark.parametrize("scorer", ["financial_services", "safety_agentic", "health_disclosure_concealment"])
    def test_hybrid_scorers_are_refused(self, run_on_disk, scorer):
        """Each looks deterministic but branches into an LLM on some rows:
        ``financial_services`` and ``safety_agentic`` emit mechanical axes with no
        ``scorer_kind`` marker yet judge dynamic-tier rows / certain sub-protocols,
        and ``health_disclosure_concealment`` runs its realized-behavior audit
        whenever a judge model is wired. Allowing any of them would silently bill
        an operation that advertises itself as offline.
        """
        run_on_disk.scorers = [scorer]
        with pytest.raises(SystemExit) as excinfo:
            run(run_on_disk)
        assert scorer in str(excinfo.value)
        assert "LLM-free" in str(excinfo.value)

    def test_unknown_scorer_is_refused(self, run_on_disk):
        run_on_disk.scorers = ["not_a_scorer"]
        with pytest.raises(SystemExit) as excinfo:
            run(run_on_disk)
        assert "not a known deterministic scorer" in str(excinfo.value)

    def test_allowlist_is_only_the_model_free_scorers(self):
        assert set(DETERMINISTIC_SCORERS) == {
            "language_compliance",
            "refusal_basics",
            "response_shape",
        }
