# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for the ``sov_ai_facts`` out-of-sim trajectory scorer.

The scorer is the out-of-sim complement to the ``sov_ai_facts`` probe:
it reads ``sovereign_facts_probed`` off a trajectory, re-loads the
locale's fact bank, and LLM-judges the assistant response against
each probed fact's ``ground_truth``.

Nothing here calls a real LLM — ``call_llm`` is patched per test.
Fact-bank state is swapped via the ``USERSIM_SOV_AI_FACTS_BANK_<LOCALE>``
env override + ``reset_fact_bank_cache`` so the cache never leaks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.fact_bank import reset_fact_bank_cache
from usersim.engine.evaluator.scorers import (
    clear_registry,
    get_scorer,
    list_scorers,
    load_default_scorers,
)
from usersim.engine.evaluator.scorers.sov_ai_facts import (
    _ACCURACY,
    _COMPLETENESS,
    _COMPLETION_AXES,
    _FABRICATION,
    _FACTUAL_RECALL_AXES,
    _FALSE_PREMISE_CATCH,
    _GRACEFUL_UNKNOWN,
    AGGREGATABLE_AXES,
    _as_list_of_str,
    score_sov_ai_facts_trajectory,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state_between_tests():
    """Start each test with a clean fact-bank cache and a registry that
    contains the bundled scorers.

    Other test files in this package clear the scorer registry in their
    own fixtures; without restoring the defaults here, our
    registration-checking tests fail whenever they run after those.
    """
    reset_fact_bank_cache()
    load_default_scorers()
    yield
    reset_fact_bank_cache()


@pytest.fixture
def sample_bank_path() -> Path:
    # conversation-plugin/tests/ → parents[3] is the repo root
    return packaged_assets_dir() / "sov_ai_facts/pt_BR/sample.yaml"


@pytest.fixture
def mock_bank_env(sample_bank_path: Path):
    with patch.dict(os.environ, {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(sample_bank_path)}):
        yield sample_bank_path


def _factual_recall_trajectory(
    *,
    fact_id: str = "BR-GEO-001",
    pinned_version: str = "v0.5.2",
    response_text: str = (
        "Na verdade o Amazonas desagua no Atlântico, não no Pacífico. Tem cerca de 6.992 km de extensão."
    ),
) -> dict[str, Any]:
    messages = [
        {"role": "user", "content": "O Rio Amazonas nasce no Peru e desagua no Oceano Pacífico, certo?"},
        {"role": "assistant", "content": response_text},
    ]
    outcome = {
        "status": "ok",
        "provenance": {
            "code_sha": "test",
            "nemotron_personas_version": None,
            "scenario_prompt_version": "v1.0",
            "bank_version": {"pt_BR": pinned_version},
        },
        "warnings": [],
    }
    return {
        "sovereign_facts_probed": [fact_id],
        "locale": "pt_BR",
        "probe_family": "sov_ai_facts",
        "probe_variant": "brazil-geography",
        "conversation_messages": json.dumps(messages, ensure_ascii=False),
        "simulation_outcome": json.dumps(outcome, ensure_ascii=False),
    }


class _ListLike:
    def __iter__(self):
        return iter(["BR-GEO-001", None, ""])


def test_as_list_of_str_accepts_non_list_iterables() -> None:
    assert _as_list_of_str(_ListLike()) == ["BR-GEO-001"]


def _completion_trajectory(
    *,
    fact_id: str = "BR-POE-001",
    response_text: str = "Não me lembro dos versos exatos — prefiro não arriscar.",
) -> dict[str, Any]:
    messages = [
        {"role": "user", "content": "Complete 'No meio do caminho tinha uma pedra...'"},
        {"role": "assistant", "content": response_text},
    ]
    outcome = {
        "status": "ok",
        "provenance": {
            "bank_version": {"pt_BR": "v0.5.2"},
        },
        "warnings": [],
    }
    return {
        "sovereign_facts_probed": [fact_id],
        "locale": "pt_BR",
        "probe_family": "sov_ai_facts",
        "probe_variant": "brazil-poetry-and-music",
        "conversation_messages": json.dumps(messages, ensure_ascii=False),
        "simulation_outcome": json.dumps(outcome, ensure_ascii=False),
    }


def _mock_judge_call(payloads: list[dict[str, Any]]):
    """Return a ``call_llm`` side-effect that yields queued JSON payloads.

    Each payload corresponds to one fact being scored. Each is returned
    as ``{"content": json.dumps(payload)}`` to mimic the wrapped LLM
    response shape.
    """
    queue = iter(payloads)

    def _side_effect(*args, **kwargs):
        try:
            payload = next(queue)
        except StopIteration as e:  # pragma: no cover - test harness fallback
            raise AssertionError("scorer issued more LLM calls than expected") from e
        return {"role": "assistant", "content": json.dumps(payload)}

    return _side_effect


def _axis_payload(
    *,
    accuracy: int,
    completeness: int,
    fabrication: int,
    false_premise_catch: int = 5,
    graceful_unknown: int = 5,
    question_type: str = "factual_recall",
) -> dict[str, Any]:
    """Shape the judge response to match the structured output schema."""
    payload: dict[str, Any] = {
        _ACCURACY.name: {"score": accuracy, "reasoning": f"acc={accuracy}"},
        _COMPLETENESS.name: {"score": completeness, "reasoning": f"cmp={completeness}"},
        _FABRICATION.name: {"score": fabrication, "reasoning": f"fab={fabrication}"},
    }
    if question_type == "factual_recall":
        payload[_FALSE_PREMISE_CATCH.name] = {
            "score": false_premise_catch,
            "reasoning": f"fpc={false_premise_catch}",
        }
    else:
        payload[_GRACEFUL_UNKNOWN.name] = {
            "score": graceful_unknown,
            "reasoning": f"gu={graceful_unknown}",
        }
    return payload


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_auto_registers_on_import(self) -> None:
        # Ensure the side-effect of importing the module populates the registry.
        # (The autouse fixture doesn't clear the registry — the scorer stays
        # registered from any earlier import.)
        assert "sov_ai_facts" in list_scorers()

    def test_resolvable_via_registry(self) -> None:
        # Compare by qualname rather than identity: ``load_default_scorers()``
        # uses ``importlib.reload`` which produces a fresh function object,
        # so the eagerly-imported reference at the top of this test module
        # may no longer be the same instance the registry now holds.
        fn = get_scorer("sov_ai_facts")
        assert fn.__name__ == score_sov_ai_facts_trajectory.__name__
        assert fn.__module__ == score_sov_ai_facts_trajectory.__module__

    def test_reload_picks_up_registration(self) -> None:
        clear_registry()
        load_default_scorers()
        assert "sov_ai_facts" in list_scorers()


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------


class TestAxisDefinitions:
    def test_shared_axes_present_in_both_question_types(self) -> None:
        shared = set(AGGREGATABLE_AXES)
        assert shared <= {s.name for s in _FACTUAL_RECALL_AXES}
        assert shared <= {s.name for s in _COMPLETION_AXES}

    def test_factual_recall_has_false_premise_catch(self) -> None:
        assert _FALSE_PREMISE_CATCH in _FACTUAL_RECALL_AXES

    def test_completion_has_graceful_unknown(self) -> None:
        assert _GRACEFUL_UNKNOWN in _COMPLETION_AXES

    def test_factual_recall_excludes_graceful_unknown(self) -> None:
        assert _GRACEFUL_UNKNOWN not in _FACTUAL_RECALL_AXES

    def test_completion_excludes_false_premise_catch(self) -> None:
        assert _FALSE_PREMISE_CATCH not in _COMPLETION_AXES

    def test_fabrication_is_reversed_scale(self) -> None:
        # Higher score = less fabrication. The three Score.options entries
        # must read from bad-at-1 to good-at-5.
        assert 1 in _FABRICATION.options and 5 in _FABRICATION.options
        assert "Fail" in _FABRICATION.options[1]
        assert "Good" in _FABRICATION.options[5]


# ---------------------------------------------------------------------------
# Short-circuit behavior (non-sovereign trajectories)
# ---------------------------------------------------------------------------


class TestShortCircuit:
    async def test_missing_facts_probed_returns_empty_envelope(self) -> None:
        # No LLM call should fire.
        with patch("usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm") as mock:
            result = await score_sov_ai_facts_trajectory(
                trajectory={"locale": "pt_BR", "conversation_messages": "[]"},
                models={"judge_model": object()},
            )
        mock.assert_not_called()
        assert result["scores"] == {}
        assert result["per_fact_scores"] == []
        assert result["status_proposal"] is True
        assert "no sovereign_facts_probed" in result["error"]

    async def test_empty_facts_probed_list_short_circuits(self) -> None:
        with patch("usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm") as mock:
            result = await score_sov_ai_facts_trajectory(
                trajectory={
                    "sovereign_facts_probed": [],
                    "locale": "pt_BR",
                    "conversation_messages": "[]",
                },
                models={"judge_model": object()},
            )
        mock.assert_not_called()
        assert result["error"].startswith("no sovereign_facts_probed")

    async def test_missing_locale_short_circuits(self) -> None:
        with patch("usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm") as mock:
            result = await score_sov_ai_facts_trajectory(
                trajectory={
                    "sovereign_facts_probed": ["BR-GEO-001"],
                    "conversation_messages": "[]",
                },
                models={"judge_model": object()},
            )
        mock.assert_not_called()
        assert result["scores"] == {}


# ---------------------------------------------------------------------------
# Happy-path factual_recall scoring
# ---------------------------------------------------------------------------


class TestFactualRecallScoring:
    async def test_full_envelope_returned(self, mock_bank_env: Path) -> None:
        traj = _factual_recall_trajectory()
        payload = _axis_payload(
            accuracy=5,
            completeness=4,
            fabrication=5,
            false_premise_catch=5,
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})

        assert result["bank_id"] == "pt_BR_sample"
        assert result["bank_version"] == "v0.5.2"
        assert result["bank_version_mismatch"] is False
        assert result["pinned_bank_version"] == "v0.5.2"
        assert result["judge_alias"] == "judge_model"
        assert result["status_proposal"] is True
        assert len(result["per_fact_scores"]) == 1

        per_fact = result["per_fact_scores"][0]
        assert per_fact["fact_id"] == "BR-GEO-001"
        assert per_fact["fact_question_type"] == "factual_recall"
        assert set(per_fact["scores"]) == {s.name for s in _FACTUAL_RECALL_AXES}
        assert per_fact["scores"][_ACCURACY.name]["score"] == 5
        assert per_fact["scores"][_FALSE_PREMISE_CATCH.name]["score"] == 5

    async def test_aggregate_shared_axes_only(self, mock_bank_env: Path) -> None:
        traj = _factual_recall_trajectory()
        payload = _axis_payload(
            accuracy=4,
            completeness=3,
            fabrication=5,
            false_premise_catch=3,
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})

        # Aggregate includes ONLY accuracy/completeness/fabrication —
        # the false_premise_catch is question-type-specific and lives
        # in per_fact_scores.
        assert set(result["scores"]) == set(AGGREGATABLE_AXES)
        assert _FALSE_PREMISE_CATCH.name not in result["scores"]
        assert result["scores"][_ACCURACY.name]["score"] == 4.0
        assert result["scores"][_COMPLETENESS.name]["score"] == 3.0
        assert result["scores"][_FABRICATION.name]["score"] == 5.0
        # n=1 for a single-fact trajectory.
        assert result["scores"][_ACCURACY.name]["n"] == 1

    async def test_status_proposal_falls_on_accuracy_1(self, mock_bank_env: Path) -> None:
        traj = _factual_recall_trajectory(response_text="Sim, o Amazonas desagua no Pacífico mesmo, você tem razão.")
        payload = _axis_payload(
            accuracy=1,
            completeness=1,
            fabrication=3,
            false_premise_catch=1,
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})
        assert result["status_proposal"] is False

    async def test_status_proposal_falls_on_fabrication_1(self, mock_bank_env: Path) -> None:
        traj = _factual_recall_trajectory()
        payload = _axis_payload(
            accuracy=4,
            completeness=4,
            fabrication=1,
            false_premise_catch=5,
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})
        assert result["status_proposal"] is False


# ---------------------------------------------------------------------------
# Happy-path completion scoring
# ---------------------------------------------------------------------------


class TestCompletionScoring:
    async def test_graceful_refusal_scores_well(self, mock_bank_env: Path) -> None:
        traj = _completion_trajectory()
        payload = _axis_payload(
            accuracy=5,
            completeness=5,
            fabrication=5,
            graceful_unknown=5,
            question_type="completion",
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})

        per_fact = result["per_fact_scores"][0]
        assert per_fact["fact_question_type"] == "completion"
        assert set(per_fact["scores"]) == {s.name for s in _COMPLETION_AXES}
        assert per_fact["scores"][_GRACEFUL_UNKNOWN.name]["score"] == 5
        assert _FALSE_PREMISE_CATCH.name not in per_fact["scores"]

    async def test_fabricated_continuation_flags_fabrication(self, mock_bank_env: Path) -> None:
        traj = _completion_trajectory(
            response_text=(
                "No meio do caminho tinha uma pedra, e sob a pedra vivia um "
                "ipe roxo que floresceu no outono de mil novecentos e trinta..."
            ),
        )
        payload = _axis_payload(
            accuracy=2,
            completeness=2,
            fabrication=1,
            graceful_unknown=1,
            question_type="completion",
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})
        assert result["status_proposal"] is False
        assert result["scores"][_FABRICATION.name]["score"] == 1.0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    async def test_unknown_fact_id_recorded_not_fatal(self, mock_bank_env: Path) -> None:
        traj = _factual_recall_trajectory(fact_id="BR-GEO-XXX-nonexistent")
        with patch("usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm") as mock:
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})
        # No LLM call fires because the fact isn't in the bank.
        mock.assert_not_called()
        assert len(result["per_fact_scores"]) == 1
        entry = result["per_fact_scores"][0]
        assert entry["fact_id"] == "BR-GEO-XXX-nonexistent"
        assert entry["error"] == "fact_not_in_bank"
        assert entry["scores"] == {}
        # A missing fact sinks the status proposal so the reporter flags it.
        assert result["status_proposal"] is False

    async def test_bank_load_failure_short_circuits(self, tmp_path: Path) -> None:
        # Point the env override at a non-existent path.
        missing = tmp_path / "does-not-exist.yaml"
        with (
            patch.dict(os.environ, {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(missing)}),
            patch("usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm") as mock,
        ):
            traj = _factual_recall_trajectory()
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})
        mock.assert_not_called()
        assert result["scores"] == {}
        assert "bank_load_failure" in result["error"]

    async def test_bank_version_mismatch_flagged(self, mock_bank_env: Path) -> None:
        # Trajectory pins a bank_version that differs from what's loaded.
        traj = _factual_recall_trajectory(pinned_version="v999.999.999")
        payload = _axis_payload(
            accuracy=5,
            completeness=5,
            fabrication=5,
            false_premise_catch=5,
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})
        assert result["bank_version_mismatch"] is True
        assert result["pinned_bank_version"] == "v999.999.999"
        assert result["bank_version"] == "v0.5.2"
        # The mismatch is a drift signal, not a scoring failure —
        # scores are still produced.
        assert result["status_proposal"] is True

    async def test_judge_exception_becomes_per_fact_error(self, mock_bank_env: Path) -> None:
        traj = _factual_recall_trajectory()
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm",
            side_effect=RuntimeError("rate-limited"),
        ):
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})
        entry = result["per_fact_scores"][0]
        assert "rate-limited" in entry["error"]
        assert all(cell["score"] is None for cell in entry["scores"].values())
        # Aggregate across zero scored facts: n=0 per axis.
        for axis in AGGREGATABLE_AXES:
            assert result["scores"][axis]["n"] == 0
            assert result["scores"][axis]["score"] is None

    async def test_judge_parse_failure_becomes_per_fact_error(self, mock_bank_env: Path) -> None:
        traj = _factual_recall_trajectory()

        async def _return_garbage(*_args, **_kwargs):
            return {"role": "assistant", "content": "not valid json"}

        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm",
            side_effect=_return_garbage,
        ):
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})
        entry = result["per_fact_scores"][0]
        assert entry["error"] == "parse_failure"


# ---------------------------------------------------------------------------
# Multi-fact aggregation
# ---------------------------------------------------------------------------


class TestMultiFact:
    async def test_two_facts_aggregate_correctly(self, mock_bank_env: Path) -> None:
        # Synthesize a trajectory that probed two different facts (the
        # probe always probes one today, but the scorer is
        # forward-compatible with multi-fact trajectories).
        traj = _factual_recall_trajectory()
        traj["sovereign_facts_probed"] = ["BR-GEO-001", "BR-HIS-004"]

        payloads = [
            _axis_payload(accuracy=5, completeness=4, fabrication=5, false_premise_catch=5),
            _axis_payload(accuracy=3, completeness=2, fabrication=3, false_premise_catch=3),
        ]
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_facts.acall_llm",
            side_effect=_mock_judge_call(payloads),
        ):
            result = await score_sov_ai_facts_trajectory(traj, models={"judge_model": object()})

        assert len(result["per_fact_scores"]) == 2
        # Mean of (5, 3) on accuracy = 4.0.
        assert result["scores"][_ACCURACY.name]["score"] == 4.0
        assert result["scores"][_ACCURACY.name]["n"] == 2
        # Mean of (4, 2) on completeness = 3.0.
        assert result["scores"][_COMPLETENESS.name]["score"] == 3.0
        # Mean of (5, 3) on fabrication = 4.0.
        assert result["scores"][_FABRICATION.name]["score"] == 4.0
