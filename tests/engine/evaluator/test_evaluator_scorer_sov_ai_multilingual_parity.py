# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``sov_ai_multilingual_parity`` out-of-sim trajectory scorer.

The scorer is the out-of-sim complement to the
``sov_ai_multilingual_parity`` probe: it reads ``query_id`` off a
trajectory, re-loads the cross-locale query bank, and LLM-judges the
assistant response on three respect-parity axes
(``respect.condescension``, ``respect.complexity_calibration``,
``respect.helpfulness_at_proficiency``). One judge call per trajectory
(one query per trajectory; no per-item loop, unlike the
``sov_ai_dynamic`` scorer).

Nothing here calls a real LLM — ``acall_llm`` is patched per test.
Query-bank state is swapped via the ``USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK`` env
override + ``reset_query_bank_cache`` so the cache never leaks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.query_bank import reset_query_bank_cache
from usersim.engine.evaluator.scorers import (
    clear_registry,
    get_scorer,
    list_scorers,
    load_default_scorers,
)
from usersim.engine.evaluator.scorers.sov_ai_multilingual_parity import (
    _AXES,
    _COMPLEXITY_CALIBRATION,
    _CONDESCENSION,
    _HELPFULNESS_AT_PROFICIENCY,
    RESPECT_AXES,
    score_sov_ai_multilingual_parity_trajectory,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_BANK_PATH = packaged_assets_dir() / "sov_ai_multilingual_parity/sample.yaml"


@pytest.fixture(autouse=True)
def _reset_state_between_tests():
    """Start each test with a clean query-bank cache and the bundled
    scorers registered.

    Other test files clear the scorer registry in their own fixtures;
    restoring defaults here keeps registration tests order-independent.
    """
    reset_query_bank_cache()
    load_default_scorers()
    yield
    reset_query_bank_cache()


@pytest.fixture
def mock_bank_env():
    """Point ``load_query_bank_default`` at the shipped sample bank."""
    with patch.dict(os.environ, {"USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK": str(SAMPLE_BANK_PATH)}):
        yield SAMPLE_BANK_PATH


def _trajectory(
    *,
    query_id: str = "Q-MEDS-001",
    locale: str = "pt_BR",
    pinned_version: str = "v0.6.0",
    n_assistant_turns: int = 1,
    persona: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal `sov_ai_multilingual_parity` trajectory record for scoring."""
    if persona is None:
        persona = {
            "first_name": "Mariana",
            "last_name": "Silva",
            "age": 37,
            "education_level": "Ensino superior",
            "occupation": "Enfermeira",
            "state": "SP",
        }
    messages: list[dict[str, str]] = [
        {"role": "user", "content": "Pergunta verbatim do banco."},
        {"role": "assistant", "content": "Resposta do assistente."},
    ]
    if n_assistant_turns >= 2:
        messages.extend(
            [
                {"role": "user", "content": "Mas e quanto a..."},
                {"role": "assistant", "content": "Resposta refinada."},
            ]
        )
    outcome = {
        "status": "ok",
        "provenance": {
            "code_sha": "test",
            "nemotron_personas_version": None,
            "scenario_prompt_version": "v1.0",
            "bank_version": {locale: pinned_version},
        },
        "warnings": [],
    }
    return {
        "query_id": query_id,
        "locale": locale,
        "probe_family": "sov_ai_multilingual_parity",
        "probe_variant": query_id,
        "conversation_messages": json.dumps(messages, ensure_ascii=False),
        "simulation_outcome": json.dumps(outcome, ensure_ascii=False),
        "persona": json.dumps(persona, ensure_ascii=False),
    }


def _axis_payload(
    *,
    condescension: int,
    complexity_calibration: int,
    helpfulness: int,
) -> dict[str, Any]:
    """Shape the judge response to match the scorer's structured output schema."""
    return {
        _CONDESCENSION.name: {
            "score": condescension,
            "reasoning": f"cond={condescension}",
        },
        _COMPLEXITY_CALIBRATION.name: {
            "score": complexity_calibration,
            "reasoning": f"calib={complexity_calibration}",
        },
        _HELPFULNESS_AT_PROFICIENCY.name: {
            "score": helpfulness,
            "reasoning": f"help={helpfulness}",
        },
    }


def _mock_judge_call(payloads: list[dict[str, Any]]):
    """``acall_llm`` side-effect that yields queued JSON payloads."""
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
    def test_auto_registered_on_import(self) -> None:
        assert "sov_ai_multilingual_parity" in list_scorers()

    def test_resolvable_via_registry(self) -> None:
        # Compare by qualname rather than identity: load_default_scorers()
        # uses importlib.reload, so the eagerly-imported reference may no
        # longer be the same instance the registry now holds.
        fn = get_scorer("sov_ai_multilingual_parity")
        assert fn.__name__ == score_sov_ai_multilingual_parity_trajectory.__name__
        assert fn.__module__ == score_sov_ai_multilingual_parity_trajectory.__module__

    def test_clear_then_load_defaults_restores_registration(self) -> None:
        clear_registry()
        assert "sov_ai_multilingual_parity" not in list_scorers()
        load_default_scorers()
        assert "sov_ai_multilingual_parity" in list_scorers()


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------


class TestAxisDefinitions:
    def test_three_axes(self) -> None:
        assert len(_AXES) == 3

    def test_axes_keyed_under_respect_namespace(self) -> None:
        for s in _AXES:
            assert s.name.startswith("respect."), f"axis {s.name} not under the respect.* namespace"

    def test_respect_axes_constant_matches_axes_module(self) -> None:
        assert RESPECT_AXES == tuple(s.name for s in _AXES)

    def test_each_axis_has_three_options(self) -> None:
        # 1 / 3 / 5 anchors mirror the sovereign and sov_ai_dynamic
        # scorers; reporting depends on the same anchor scale to
        # aggregate across probes.
        for s in _AXES:
            assert set(s.options.keys()) == {1, 3, 5}, (
                f"axis {s.name} has options {sorted(s.options.keys())}, expected {{1, 3, 5}}"
            )


# ---------------------------------------------------------------------------
# Short-circuit
# ---------------------------------------------------------------------------


class TestShortCircuit:
    async def test_missing_query_id_returns_no_op(self) -> None:
        out = await score_sov_ai_multilingual_parity_trajectory({}, {})
        assert out["scores"] == {}
        assert "no query_id" in out["error"]
        assert out["status_proposal"] is True
        assert out["bank_id"] is None

    async def test_non_string_query_id_returns_no_op(self) -> None:
        out = await score_sov_ai_multilingual_parity_trajectory(
            {"query_id": 12345, "locale": "pt_BR"},
            {},
        )
        assert "no query_id" in out["error"]

    async def test_missing_locale_returns_no_op(self) -> None:
        out = await score_sov_ai_multilingual_parity_trajectory(
            {"query_id": "Q-MEDS-001"},
            {},
        )
        assert "no query_id" in out["error"]

    async def test_no_llm_call_made_on_short_circuit(self) -> None:
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
        ) as mock_call:
            await score_sov_ai_multilingual_parity_trajectory({}, {"alias": object()})
        mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_runs_one_judge_call_and_returns_scores(
        self,
        mock_bank_env: Path,
    ) -> None:
        traj = _trajectory(query_id="Q-MEDS-001", locale="pt_BR")
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_mock_judge_call(
                [
                    _axis_payload(
                        condescension=4,
                        complexity_calibration=4,
                        helpfulness=5,
                    ),
                ]
            ),
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )

        assert out["query_id"] == "Q-MEDS-001"
        assert out["query_domain"] == "health"
        assert out["query_difficulty"] == "easy"
        assert out["query_placeholder"] is False
        assert out["bank_id"] == "sample_v1"
        assert out["bank_version"] == "v0.6.0"
        assert out["bank_version_mismatch"] is False
        assert out["pinned_bank_version"] == "v0.6.0"
        assert out["status_proposal"] is True
        assert out["judge_alias"] == "judge_a"
        # Per-axis cells.
        assert out["scores"][_CONDESCENSION.name]["score"] == 4
        assert out["scores"][_COMPLEXITY_CALIBRATION.name]["score"] == 4
        assert out["scores"][_HELPFULNESS_AT_PROFICIENCY.name]["score"] == 5

    async def test_judge_alias_falls_back_to_default_when_models_empty(
        self,
        mock_bank_env: Path,
    ) -> None:
        # Empty models dict: scorer should still return a no-op-style
        # envelope rather than crashing on next(iter(models)). It will
        # fail at the LLM call step (KeyError under acall_llm), but we
        # patch acall_llm to avoid that and assert envelope shape.
        traj = _trajectory()
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_mock_judge_call(
                [
                    _axis_payload(
                        condescension=3,
                        complexity_calibration=3,
                        helpfulness=3,
                    ),
                ]
            ),
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(traj, {})
        assert out["judge_alias"] == "judge_model"

    async def test_judge_call_receives_locale_and_concern(
        self,
        mock_bank_env: Path,
    ) -> None:
        # The rubric's load-bearing input is the query.concern field;
        # confirm the scorer renders it into the user prompt the judge sees.
        traj = _trajectory(query_id="Q-MEDS-001", locale="pt_BR")
        captured = {}

        async def _capture_call(*args, **kwargs):
            # acall_llm signature: (models, alias, messages, **kwargs)
            captured["messages"] = args[2] if len(args) >= 3 else kwargs.get("messages")
            return {
                "role": "assistant",
                "content": json.dumps(
                    _axis_payload(
                        condescension=4,
                        complexity_calibration=4,
                        helpfulness=4,
                    )
                ),
            }

        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_capture_call,
        ):
            await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )

        user_msg = captured["messages"][1]["content"]
        # Locale and human-readable language label are interpolated.
        assert "pt_BR" in user_msg
        assert "Portuguese" in user_msg
        # query_id appears so the judge can refer to it in reasoning.
        assert "Q-MEDS-001" in user_msg
        # The concern block must be rendered between the markers.
        assert "<CONCERN>" in user_msg
        # Persona block must be rendered.
        assert "<PERSONA>" in user_msg
        assert "Enfermeira" in user_msg


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    async def test_query_not_in_bank_returns_structured_error(
        self,
        mock_bank_env: Path,
    ) -> None:
        traj = _trajectory(query_id="Q-DOES-NOT-EXIST", locale="pt_BR")
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
        ) as mock_call:
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        # No LLM call should have happened — we failed before scoring.
        mock_call.assert_not_called()
        assert out["error"] == "query_not_in_bank"
        assert out["status_proposal"] is False
        # Bank metadata still pinned for downstream drift detection.
        assert out["bank_id"] == "sample_v1"
        assert out["bank_version"] == "v0.6.0"

    async def test_bank_load_failure_returns_structured_error(self) -> None:
        traj = _trajectory()
        # Point env at a path that does not exist.
        with (
            patch.dict(
                os.environ,
                {"USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK": "/nonexistent/bank.yaml"},
            ),
            patch(
                "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            ) as mock_call,
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        mock_call.assert_not_called()
        assert "bank_load_failure" in out["error"]
        assert out["status_proposal"] is False
        assert out["bank_id"] is None
        assert out["scores"] == {}

    async def test_judge_exception_yields_per_axis_none_and_blocks_status(
        self,
        mock_bank_env: Path,
    ) -> None:
        async def _raises(*args, **kwargs):
            raise RuntimeError("judge API exploded")

        traj = _trajectory()
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_raises,
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        # The wrapping envelope flips status_proposal=False on judge
        # error (it counts as a 1 on a safety axis effectively).
        assert out["status_proposal"] is False
        # Every axis cell carries score=None and the error reason.
        for axis in RESPECT_AXES:
            cell = out["scores"][axis]
            assert cell["score"] is None
            assert "RuntimeError" in cell["reasoning"]

    async def test_unparseable_judge_response_yields_per_axis_none(
        self,
        mock_bank_env: Path,
    ) -> None:
        traj = _trajectory()
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            return_value={"role": "assistant", "content": "not-json-{"},
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        assert out["status_proposal"] is False
        for axis in RESPECT_AXES:
            cell = out["scores"][axis]
            assert cell["score"] is None
            assert cell["reasoning"] == "parse_failure"


# ---------------------------------------------------------------------------
# Status proposal gating
# ---------------------------------------------------------------------------


class TestStatusProposal:
    async def test_score_of_one_on_condescension_blocks_status(
        self,
        mock_bank_env: Path,
    ) -> None:
        traj = _trajectory()
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_mock_judge_call(
                [
                    _axis_payload(
                        condescension=1,
                        complexity_calibration=4,
                        helpfulness=4,
                    ),
                ]
            ),
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        assert out["status_proposal"] is False

    async def test_score_of_one_on_helpfulness_blocks_status(
        self,
        mock_bank_env: Path,
    ) -> None:
        traj = _trajectory()
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_mock_judge_call(
                [
                    _axis_payload(
                        condescension=4,
                        complexity_calibration=4,
                        helpfulness=1,
                    ),
                ]
            ),
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        assert out["status_proposal"] is False

    async def test_score_of_one_on_complexity_calibration_does_not_block(
        self,
        mock_bank_env: Path,
    ) -> None:
        # complexity_calibration is informational — a generic-level
        # response is not a hard fail, even if it scores 1. Catches
        # accidental gating-rule expansion.
        traj = _trajectory()
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_mock_judge_call(
                [
                    _axis_payload(
                        condescension=4,
                        complexity_calibration=1,
                        helpfulness=4,
                    ),
                ]
            ),
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        assert out["status_proposal"] is True

    async def test_all_high_scores_passes_status(self, mock_bank_env: Path) -> None:
        traj = _trajectory()
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_mock_judge_call(
                [
                    _axis_payload(
                        condescension=5,
                        complexity_calibration=5,
                        helpfulness=5,
                    ),
                ]
            ),
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        assert out["status_proposal"] is True


# ---------------------------------------------------------------------------
# Version mismatch / drift
# ---------------------------------------------------------------------------


class TestVersionMismatch:
    async def test_pinned_version_matches_current_no_mismatch(
        self,
        mock_bank_env: Path,
    ) -> None:
        traj = _trajectory(pinned_version="v0.6.0")
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_mock_judge_call(
                [
                    _axis_payload(
                        condescension=4,
                        complexity_calibration=4,
                        helpfulness=4,
                    ),
                ]
            ),
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        assert out["bank_version_mismatch"] is False

    async def test_pinned_version_differs_from_current_emits_mismatch_flag(
        self,
        mock_bank_env: Path,
    ) -> None:
        # Trajectory pinned to a fictional older bank_version.
        traj = _trajectory(pinned_version="v0.0.5")
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_mock_judge_call(
                [
                    _axis_payload(
                        condescension=4,
                        complexity_calibration=4,
                        helpfulness=4,
                    ),
                ]
            ),
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        assert out["bank_version_mismatch"] is True
        assert out["pinned_bank_version"] == "v0.0.5"
        assert out["bank_version"] == "v0.6.0"
        # Drift does NOT block status — it's surfaced as informational.
        assert out["status_proposal"] is True

    async def test_no_pinned_version_no_mismatch(self, mock_bank_env: Path) -> None:
        traj = _trajectory()
        # Strip the bank_version block from simulation_outcome.
        outcome = json.loads(traj["simulation_outcome"])
        outcome["provenance"]["bank_version"] = {}
        traj["simulation_outcome"] = json.dumps(outcome)
        with patch(
            "usersim.engine.evaluator.scorers.sov_ai_multilingual_parity.acall_llm",
            side_effect=_mock_judge_call(
                [
                    _axis_payload(
                        condescension=4,
                        complexity_calibration=4,
                        helpfulness=4,
                    ),
                ]
            ),
        ):
            out = await score_sov_ai_multilingual_parity_trajectory(
                traj,
                {"judge_a": object()},
            )
        assert out["bank_version_mismatch"] is False
        assert out["pinned_bank_version"] is None
