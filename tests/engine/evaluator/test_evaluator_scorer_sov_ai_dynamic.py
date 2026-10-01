# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``sov_ai_dynamic`` out-of-sim trajectory scorer.

The scorer is the out-of-sim complement to the ``sov_ai_dynamic`` probe:
it reads ``probing_categories_explored`` + ``probing_subtopic_hints_used``
off a trajectory, re-loads the locale's probing taxonomy, and LLM-judges
the assistant response on three response-only axes (no ground truth).

Nothing here calls a real LLM — ``acall_llm`` is patched per test.
Taxonomy state is swapped via the ``USERSIM_SOV_AI_DYNAMIC_TAXONOMY_<LOCALE>``
env override + ``reset_probing_taxonomy_cache`` so the cache never leaks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.probing_taxonomy import reset_probing_taxonomy_cache
from usersim.engine.evaluator.scorers import (
    clear_registry,
    get_scorer,
    list_scorers,
    load_default_scorers,
)
from usersim.engine.evaluator.scorers.sov_ai_dynamic import (
    _AXES,
    _GRACEFUL_UNKNOWN,
    _SELF_CONSISTENCY,
    _SUSPECTED_FABRICATION,
    AGGREGATABLE_AXES,
    _as_list_of_str,
    score_sov_ai_dynamic_trajectory,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state_between_tests():
    """Start each test with a clean taxonomy cache and a registry that
    contains the bundled scorers.

    Other test files in this package clear the scorer registry in their
    own fixtures; restoring the defaults here keeps registration tests
    order-independent.
    """
    reset_probing_taxonomy_cache()
    load_default_scorers()
    yield
    reset_probing_taxonomy_cache()


@pytest.fixture
def sample_taxonomy_path() -> Path:
    return packaged_assets_dir() / "sov_ai_dynamic/pt_BR/categories.yaml"


@pytest.fixture
def mock_taxonomy_env(sample_taxonomy_path: Path):
    with patch.dict(
        os.environ,
        {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(sample_taxonomy_path)},
    ):
        yield sample_taxonomy_path


def _trajectory(
    *,
    category_id: str = "brazil-history",
    subtopic_hint: str = "Era Vargas",
    pinned_version: str = "v0.5.1",
    n_assistant_turns: int = 1,
) -> dict[str, Any]:
    """Build a minimal `sov_ai_dynamic` trajectory record for scoring.

    ``n_assistant_turns`` controls whether the conversation has 1 or 2
    assistant turns (relevant for ``self_consistency`` semantics).
    """
    messages: list[dict[str, str]] = [
        {"role": "user", "content": f"Aproveitando, queria entender melhor sobre {subtopic_hint}..."},
        {"role": "assistant", "content": "Resposta do assistente sobre o tópico."},
    ]
    if n_assistant_turns >= 2:
        messages.extend(
            [
                {"role": "user", "content": "Hmm, e quanto a..."},
                {"role": "assistant", "content": "Resposta refinada do assistente."},
            ]
        )
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
        "probing_categories_explored": [category_id],
        "probing_subtopic_hints_used": [subtopic_hint],
        "locale": "pt_BR",
        "probe_family": "sov_ai_dynamic",
        "probe_variant": category_id,
        "conversation_messages": json.dumps(messages, ensure_ascii=False),
        "simulation_outcome": json.dumps(outcome, ensure_ascii=False),
    }


class _ListLike:
    def __iter__(self):
        return iter(["brazil-history", None, ""])


def test_as_list_of_str_accepts_non_list_iterables() -> None:
    assert _as_list_of_str(_ListLike()) == ["brazil-history"]


def _axis_payload(
    *,
    suspected_fabrication: int,
    self_consistency: int,
    graceful_unknown: int,
) -> dict[str, Any]:
    """Shape the judge response to match the scorer's structured output schema."""
    return {
        _SUSPECTED_FABRICATION.name: {
            "score": suspected_fabrication,
            "reasoning": f"fab={suspected_fabrication}",
        },
        _SELF_CONSISTENCY.name: {
            "score": self_consistency,
            "reasoning": f"sc={self_consistency}",
        },
        _GRACEFUL_UNKNOWN.name: {
            "score": graceful_unknown,
            "reasoning": f"gu={graceful_unknown}",
        },
    }


def _mock_judge_call(payloads: list[dict[str, Any]]):
    """``acall_llm`` side-effect that yields queued JSON payloads in order."""
    queue = iter(payloads)

    def _side_effect(*args, **kwargs):
        try:
            payload = next(queue)
        except StopIteration as e:  # pragma: no cover — harness fallback
            raise AssertionError("scorer issued more LLM calls than expected") from e
        return {"role": "assistant", "content": json.dumps(payload)}

    return _side_effect


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_auto_registers_on_import(self) -> None:
        assert "sov_ai_dynamic" in list_scorers()

    def test_resolvable_via_registry(self) -> None:
        # Compare by qualname rather than identity: load_default_scorers()
        # uses importlib.reload, so the eagerly-imported reference may no
        # longer be the same instance the registry now holds.
        fn = get_scorer("sov_ai_dynamic")
        assert fn.__name__ == score_sov_ai_dynamic_trajectory.__name__
        assert fn.__module__ == score_sov_ai_dynamic_trajectory.__module__

    def test_reload_picks_up_registration(self) -> None:
        clear_registry()
        load_default_scorers()
        assert "sov_ai_dynamic" in list_scorers()

    def test_listed_alongside_sister_scorers(self) -> None:
        names = list_scorers()
        # ``sov_ai_facts`` sibling and the original tool-use scorer must
        # both still be present alongside the new one.
        assert "sov_ai_facts" in names
        assert "tool_use" in names
        assert "sov_ai_dynamic" in names


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------


class TestAxisDefinitions:
    def test_three_axes_defined(self) -> None:
        names = {s.name for s in _AXES}
        assert names == {
            "sovereign.suspected_fabrication",
            "sovereign.self_consistency",
            "sovereign.graceful_unknown",
        }

    def test_aggregatable_axes_match(self) -> None:
        assert set(AGGREGATABLE_AXES) == {s.name for s in _AXES}

    def test_axis_names_share_sovereign_namespace(self) -> None:
        """All axes prefixed ``sovereign.*`` so the combined scorecard
        can group ``sov_ai_facts`` + ``sov_ai_dynamic`` outputs by
        namespace without per-axis provenance plumbing."""
        for s in _AXES:
            assert s.name.startswith("sovereign.")

    def test_all_axes_use_1_to_5_options(self) -> None:
        for s in _AXES:
            assert 1 in s.options
            assert 5 in s.options


# ---------------------------------------------------------------------------
# Short-circuit / no-op behavior
# ---------------------------------------------------------------------------


class TestShortCircuit:
    async def test_missing_categories_returns_noop(self) -> None:
        result = await score_sov_ai_dynamic_trajectory(
            {"locale": "pt_BR"},  # no probing_categories_explored
            {"judge_model": object()},
        )
        assert result["scores"] == {}
        assert result["per_category_scores"] == []
        assert result["status_proposal"] is True
        assert "no probing_categories_explored" in result["error"]

    async def test_empty_categories_list_returns_noop(self) -> None:
        result = await score_sov_ai_dynamic_trajectory(
            {
                "probing_categories_explored": [],
                "locale": "pt_BR",
            },
            {"judge_model": object()},
        )
        assert result["scores"] == {}
        assert "no probing_categories_explored" in result["error"]

    async def test_missing_locale_returns_noop(self) -> None:
        result = await score_sov_ai_dynamic_trajectory(
            {"probing_categories_explored": ["brazil-geography"]},
            {"judge_model": object()},
        )
        assert result["scores"] == {}
        assert "no probing_categories_explored" in result["error"]


# ---------------------------------------------------------------------------
# Happy path — single category, single judge call
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_one_category_scored(self, mock_taxonomy_env: Path) -> None:
        payload = _axis_payload(
            suspected_fabrication=4,
            self_consistency=5,
            graceful_unknown=4,
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_dynamic.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_dynamic_trajectory(
                _trajectory(),
                {"judge_model": object()},
            )

        assert result["taxonomy_id"] == "pt_BR_sov_ai_dynamic"
        assert result["taxonomy_version"] == "v0.5.1"
        assert result["taxonomy_version_mismatch"] is False
        assert result["status_proposal"] is True
        assert len(result["per_category_scores"]) == 1

        per_cat = result["per_category_scores"][0]
        assert per_cat["category_id"] == "brazil-history"
        assert per_cat["subtopic_hint"] == "Era Vargas"
        assert per_cat["scores"][_SUSPECTED_FABRICATION.name]["score"] == 4
        assert per_cat["scores"][_SELF_CONSISTENCY.name]["score"] == 5
        assert per_cat["scores"][_GRACEFUL_UNKNOWN.name]["score"] == 4

        # Aggregate equals the per-category value when n=1.
        agg = result["scores"]
        assert agg[_SUSPECTED_FABRICATION.name]["score"] == 4
        assert agg[_SUSPECTED_FABRICATION.name]["n"] == 1
        assert agg[_SELF_CONSISTENCY.name]["score"] == 5
        assert agg[_GRACEFUL_UNKNOWN.name]["score"] == 4

    async def test_failing_score_flips_status_proposal(self, mock_taxonomy_env: Path) -> None:
        payload = _axis_payload(
            suspected_fabrication=1,  # fail
            self_consistency=5,
            graceful_unknown=5,
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_dynamic.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_dynamic_trajectory(
                _trajectory(),
                {"judge_model": object()},
            )
        assert result["status_proposal"] is False

    async def test_two_assistant_turns_judged_with_self_consistency(
        self,
        mock_taxonomy_env: Path,
    ) -> None:
        """Multi-turn trajectory should give the judge a self-consistency
        surface to score on — we just check the call goes through with
        the right number of messages in the prompt."""
        payload = _axis_payload(
            suspected_fabrication=5,
            self_consistency=3,  # mild inconsistency surfaced
            graceful_unknown=5,
        )
        captured_calls: list[Any] = []

        async def _capturing_side_effect(*args, **kwargs):
            captured_calls.append((args, kwargs))
            return {"role": "assistant", "content": json.dumps(payload)}

        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_dynamic.acall_llm",
            side_effect=_capturing_side_effect,
        ):
            result = await score_sov_ai_dynamic_trajectory(
                _trajectory(n_assistant_turns=2),
                {"judge_model": object()},
            )
        assert result["scores"][_SELF_CONSISTENCY.name]["score"] == 3
        # Exactly one judge call per category.
        assert len(captured_calls) == 1


# ---------------------------------------------------------------------------
# Multi-category aggregation (forward-looking — today probes emit one)
# ---------------------------------------------------------------------------


class TestMultiCategoryAggregation:
    async def test_arithmetic_mean_across_two_categories(
        self,
        mock_taxonomy_env: Path,
    ) -> None:
        traj = _trajectory()
        traj["probing_categories_explored"] = ["brazil-history", "brazil-geography"]
        traj["probing_subtopic_hints_used"] = ["Era Vargas", "o cerrado"]
        # Two judge calls, one per category, with different scores.
        payloads = [
            _axis_payload(suspected_fabrication=4, self_consistency=5, graceful_unknown=4),
            _axis_payload(suspected_fabrication=2, self_consistency=3, graceful_unknown=4),
        ]
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_dynamic.acall_llm",
            side_effect=_mock_judge_call(payloads),
        ):
            result = await score_sov_ai_dynamic_trajectory(
                traj,
                {"judge_model": object()},
            )

        assert len(result["per_category_scores"]) == 2
        agg = result["scores"]
        assert agg[_SUSPECTED_FABRICATION.name]["score"] == 3.0  # (4+2)/2
        assert agg[_SUSPECTED_FABRICATION.name]["n"] == 2
        assert agg[_SELF_CONSISTENCY.name]["score"] == 4.0  # (5+3)/2
        assert agg[_GRACEFUL_UNKNOWN.name]["score"] == 4.0  # (4+4)/2
        # Any score=1 would flip status; here lowest is 2.
        # But our scorer flips status_proposal on score==1 only, not <=2,
        # so this should still be True.
        assert result["status_proposal"] is True


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    async def test_taxonomy_load_failure_returns_structured_error(self, tmp_path: Path) -> None:
        # Point env at a nonexistent file.
        with patch.dict(
            os.environ,
            {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(tmp_path / "missing.yaml")},
        ):
            result = await score_sov_ai_dynamic_trajectory(
                _trajectory(),
                {"judge_model": object()},
            )
        assert result["taxonomy_id"] is None
        assert result["scores"] == {}
        assert "taxonomy_load_failure" in result["error"]

    async def test_unknown_category_id_records_per_category_error(
        self,
        mock_taxonomy_env: Path,
    ) -> None:
        traj = _trajectory(category_id="brazil-nonexistent")
        # No LLM call should fire — category lookup fails before the call.
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_dynamic.acall_llm",
            side_effect=AssertionError("scorer should not call LLM for unknown category"),
        ):
            result = await score_sov_ai_dynamic_trajectory(
                traj,
                {"judge_model": object()},
            )
        assert len(result["per_category_scores"]) == 1
        assert result["per_category_scores"][0]["category_id"] == "brazil-nonexistent"
        assert result["per_category_scores"][0]["error"] == "category_not_in_taxonomy"
        # Status flips to False because we couldn't actually score.
        assert result["status_proposal"] is False

    async def test_judge_exception_records_per_category_error(
        self,
        mock_taxonomy_env: Path,
    ) -> None:
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_dynamic.acall_llm",
            side_effect=RuntimeError("rate-limited"),
        ):
            result = await score_sov_ai_dynamic_trajectory(
                _trajectory(),
                {"judge_model": object()},
            )
        assert len(result["per_category_scores"]) == 1
        per_cat = result["per_category_scores"][0]
        assert "RuntimeError" in per_cat["error"]
        # All axes None on a failed judge call.
        for axis_name in AGGREGATABLE_AXES:
            assert per_cat["scores"][axis_name]["score"] is None

    async def test_judge_returns_unparseable_content(self, mock_taxonomy_env: Path) -> None:
        async def _bad_content(*args, **kwargs):
            return {"role": "assistant", "content": "not-json-{garbage"}

        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_dynamic.acall_llm",
            side_effect=_bad_content,
        ):
            result = await score_sov_ai_dynamic_trajectory(
                _trajectory(),
                {"judge_model": object()},
            )
        per_cat = result["per_category_scores"][0]
        assert per_cat["error"] == "parse_failure"
        for axis_name in AGGREGATABLE_AXES:
            assert per_cat["scores"][axis_name]["score"] is None


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


class TestVersionMismatch:
    async def test_pinned_version_mismatch_flagged_not_failed(
        self,
        mock_taxonomy_env: Path,
    ) -> None:
        # Trajectory pinned an older taxonomy version; current cache loads v0.5.0.
        traj = _trajectory(pinned_version="v0.0.9-dev")
        payload = _axis_payload(
            suspected_fabrication=5,
            self_consistency=5,
            graceful_unknown=5,
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_dynamic.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_dynamic_trajectory(
                traj,
                {"judge_model": object()},
            )
        assert result["taxonomy_version_mismatch"] is True
        assert result["pinned_taxonomy_version"] == "v0.0.9-dev"
        assert result["taxonomy_version"] == "v0.5.1"
        # Mismatch is a flag, not a failure — scoring still proceeds.
        assert result["scores"][_SUSPECTED_FABRICATION.name]["score"] == 5
        assert result["status_proposal"] is True

    async def test_no_pinned_version_no_mismatch_flag(self, mock_taxonomy_env: Path) -> None:
        traj = _trajectory()
        # Strip the bank_version block from the simulation_outcome.
        outcome = json.loads(traj["simulation_outcome"])
        outcome["provenance"]["bank_version"] = {}
        traj["simulation_outcome"] = json.dumps(outcome)
        payload = _axis_payload(
            suspected_fabrication=5,
            self_consistency=5,
            graceful_unknown=5,
        )
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_dynamic.acall_llm",
            side_effect=_mock_judge_call([payload]),
        ):
            result = await score_sov_ai_dynamic_trajectory(
                traj,
                {"judge_model": object()},
            )
        assert result["pinned_taxonomy_version"] is None
        assert result["taxonomy_version_mismatch"] is False
