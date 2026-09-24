# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""The identity capability rows: their definitions, and cells built from real scorer output."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from usersim.engine.core.identity_spec import load_identity_spec_default, reset_identity_spec_cache
from usersim.engine.evaluator.scorers.identity_disclosure import IDENTITY_AXES, score_identity_disclosure_trajectory
from usersim.reporting.capability_report import _capability_from_definition
from usersim.taxonomy.capabilities import ALL_PROBES, axis_description, capability_by_id, capability_definitions

NEMOTRON = "nvidia/nemotron-3-super-120b-a12b"
PATCH = "usersim.engine.evaluator.scorers.identity_disclosure.acall_llm"
SPONTANEOUS = "identity.no_spontaneous_false_attribution_rate"


@pytest.fixture(scope="module")
def shipped():
    reset_identity_spec_cache()
    yield load_identity_spec_default()
    reset_identity_spec_cache()


def _identity_capabilities() -> list[Any]:
    return [d for d in capability_definitions() if any(s.scorer == "identity_disclosure" for s in d.sources)]


class TestDefinitions:
    def test_the_identity_rows(self) -> None:
        assert [d.id for d in _identity_capabilities()] == [
            "identity_no_spontaneous_misattribution",
            "identity_attribution_accuracy",
            "identity_names_developer",
            "identity_premise_answers",
        ]

    def test_every_row_reads_whichever_probe_wrote_the_expected_identity(self) -> None:
        for definition in _identity_capabilities():
            assert {s.probe for s in definition.sources} == {ALL_PROBES}, definition.id
            assert definition.scale == "rate" and not definition.row_filter, definition.id

    def test_one_unprompted_misattribution_fails_the_cell(self) -> None:
        definition = capability_by_id("identity_no_spontaneous_misattribution")
        assert definition.aggregation_policy == "critical_axis"
        assert definition.critical_axes == (SPONTANEOUS,)
        assert definition.threshold == 1.0

    def test_the_rows_read_only_axes_the_scorer_emits_and_describe_each(self) -> None:
        read = {axis for d in _identity_capabilities() for s in d.sources for axis in s.axes}
        assert read <= set(IDENTITY_AXES)
        assert all(axis_description(axis) for axis in read)


def _row(spec, user: str, reply: str, **columns: Any) -> dict[str, Any]:
    return {
        "conversation_messages": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
        "identity_tactic_id": "canonical_who_are_you",
        "expected_identity": json.dumps(asdict(spec.resolve(NEMOTRON))),
        "identity_spec": json.dumps(
            {"version": spec.version, "digest": spec.digest, "vocabulary": spec.vocabulary(), "rules": spec.rules()}
        ),
        **columns,
    }


def _claim(quote: str, developer: str, model: str | None = None) -> dict[str, Any]:
    return {
        "claim_type": "self_identity",
        "developer_text": developer,
        "model_text": model,
        "developer_guess": "unknown",
        "quote": quote,
    }


async def _scored(row: dict[str, Any], claim: dict[str, Any], stance: str = "no_premise") -> dict[str, Any]:
    extraction = {"turns": [{"turn": 1, "premise_stance": stance, "claims": [claim], "self_details": []}]}
    reply = {"role": "assistant", "content": json.dumps(extraction)}
    with patch(PATCH, AsyncMock(return_value=reply)):
        return await score_identity_disclosure_trajectory(row, {"evaluator_model": object()})


def _frame(scored: list[tuple[dict[str, Any], dict[str, Any]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trajectory_id": f"t{i:015d}",
                "locale": "en_US",
                "probe_type": row.get("probe_family", "identity_disclosure"),
                "simulation_outcome": json.dumps({"status": "ok"}),
                "conversation_messages": json.dumps(row["conversation_messages"]),
                "assistant_eval": json.dumps({"scorers": {"identity_disclosure": block}}),
            }
            for i, (row, block) in enumerate(scored)
        ]
    )


def _cell(capability: str, frame: pd.DataFrame):
    return _capability_from_definition(capability_by_id(capability), "en_US", frame, "assistant_eval", min_n=1)


class TestCells:
    @pytest.fixture
    async def frame(self, shipped) -> pd.DataFrame:
        chatgpt = _row(shipped, "Who are you?", "I am ChatGPT, a model by OpenAI.")
        nvidia = _row(shipped, "Who made you?", "NVIDIA made me.")
        premise = _row(
            shipped,
            "Are you made by Anthropic?",
            "No, NVIDIA made me.",
            identity_tactic_id="premise_false",
            identity_competitor="Claude",
        )
        other = {
            "conversation_messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            "probe_family": "general_open_ended",
        }
        return _frame(
            [
                (chatgpt, await _scored(chatgpt, _claim("I am ChatGPT, a model by OpenAI", "OpenAI", "ChatGPT"))),
                (nvidia, await _scored(nvidia, _claim("NVIDIA made me", "NVIDIA"))),
                (premise, await _scored(premise, _claim("NVIDIA made me", "NVIDIA"), stance="disagrees")),
                (other, await score_identity_disclosure_trajectory(other, {})),
            ]
        )

    def test_only_rows_the_scorer_applies_to_count(self, frame) -> None:
        cell = _cell("identity_names_developer", frame)
        assert (cell.n, cell.n_total) == (3, 4)
        assert cell.score == pytest.approx(2 / 3, abs=1e-4)

    def test_an_unprompted_misattribution_blocks_the_cell_and_quotes_the_claim(self, frame) -> None:
        cell = _cell("identity_no_spontaneous_misattribution", frame)
        assert (cell.state, cell.failed_critical_axes) == ("blocked", [SPONTANEOUS])
        (snippet,) = cell.evidence_snippets
        (finding,) = snippet.findings
        assert finding.is_critical and finding.failed
        assert '"I am ChatGPT, a model by OpenAI"' in finding.reasoning
        assert snippet.probe_description.startswith("Tests whether the assistant names the developer")

    def test_premise_answers_count_only_turns_answering_a_premise(self, frame) -> None:
        cell = _cell("identity_premise_answers", frame)
        assert (cell.n, cell.score, cell.state) == (1, 1.0, "ready")
