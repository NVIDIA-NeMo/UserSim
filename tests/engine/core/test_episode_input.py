# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public, model-free episode-input construction contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from usersim.cli._pipeline import known_probes
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.episode_input import construct_episode_preamble, construct_probe_episode
from usersim.engine.core.provenance import get_code_sha
from usersim.engine.external import ProbeEpisodeRuntime, materialize_episode_inputs


def test_materialization_resolves_theme_tools_and_restorable_config() -> None:
    """Rows are the stored dataset, so they must restore an episode on their own.

    Regeneration is deliberately not pinned: the rows are what gets stored and
    replayed, so making the sampler reproducible would only buy a property
    nothing depends on, at the cost of reseeding process-global generators.
    """
    first = materialize_episode_inputs(
        locale="en_US",
        num_rows=1,
        probe_mix={"tool_calling": 1.0},
        random_seed=73,
    )

    assert first[0]["probe_type"] == "tool_calling"
    assert first[0]["persona"]
    assert first[0]["theme"]
    assert first[0]["tools"]
    assert first[0]["toolset_category"]
    # Some shipped toolsets carry no name, so only the column is guaranteed.
    assert "toolset_name" in first[0]
    assert first[0]["trajectory_id"]
    provenance = json.loads(first[0]["usersim_provenance"])
    assert provenance["code_sha"] == get_code_sha()
    assert first[0]["usersim_config"]["random_seed"] == 73
    runtime = ProbeEpisodeRuntime.from_resolved_row(first[0], models={})
    assert runtime.preamble.trajectory_id == first[0]["trajectory_id"]
    assert runtime.preamble.provenance.to_dict() == provenance


@pytest.mark.parametrize("stored_as", ["json_text", "mapping"])
def test_a_stored_row_keeps_its_own_provenance(stored_as: str) -> None:
    """A replayed row keeps the provenance it was materialized with, stored as JSON text or as a mapping."""
    row = materialize_episode_inputs(locale="en_US", num_rows=1, probe_mix={"tool_calling": 1.0}, random_seed=73)[0]
    stored = {**json.loads(row["usersim_provenance"]), "code_sha": "0123456789ab"}
    row["usersim_provenance"] = json.dumps(stored) if stored_as == "json_text" else stored

    runtime = ProbeEpisodeRuntime.from_resolved_row(row, models={})

    assert runtime.preamble.provenance.code_sha == "0123456789ab"
    assert json.loads(runtime.preamble.to_row()["usersim_provenance"]) == stored


@pytest.mark.parametrize("probe_type", known_probes())
def test_materialization_supports_every_registered_probe(probe_type: str, tmp_path: Path) -> None:
    """Every probe materializes a complete row, and the row survives the parquet write of ``--materialize-inputs``."""
    rows = materialize_episode_inputs(
        locale="en_US",
        num_rows=1,
        probe_mix={probe_type: 1.0},
        random_seed=19,
    )

    assert len(rows) == 1
    assert rows[0]["probe_type"] == probe_type
    assert rows[0]["probe_family"]
    assert rows[0]["probe_variant"]
    assert rows[0]["persona_uuid"]
    assert rows[0]["trajectory_id"]
    path = tmp_path / "rows.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    assert pd.read_parquet(path)["trajectory_id"].tolist() == [rows[0]["trajectory_id"]]


# Trajectory identity is the resume and deduplication key for every stored row,
# so the formula is pinned per probe. Inputs are fixed (persona, seed, model
# names) and these values were cross-checked against `main` using its own
# `generator._probe_variant`. Seven probes discover their own variant while they
# run; that variant lands on the finished row and must NOT perturb the id here.
_PINNED_PERSONA = {
    "first_name": "Sarah",
    "last_name": "Chen",
    "age": 35,
    "sex": "Female",
    "city": "Seattle",
    "country": "United States",
    "education_level": "Bachelor",
    "occupation": "Software Engineer",
}
# Resolved from the pinned persona + probe metadata, never stored on the row.
_PERSONA_DERIVED_COLUMNS = (
    "behavioral_profile",
    "disclosure_style",
    "user_interaction_style",
    "persona_grounding",
    "persona_uuid",
    "probe_variant",
    "probe_family",
    "trajectory_id",
    "usersim_provenance",
    "usersim_config",
)
_PINNED_ASSISTANT_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
_PINNED_TRAJECTORY_IDS = {
    "financial_services": "e736ac84d4d9438a",
    "general_educational": "fef0f90231a81da0",
    "general_open_ended": "0fb84850f75a3f00",
    "health_decision_support_disclosure": "fd69ae7b7cfc94a9",
    "health_general_disclosure": "53a2c75f6cd26bce",
    "health_therapy_disclosure": "e34d48f7eac319a6",
    "health_triage_disclosure": "8c2c67f3cbd0f7b0",
    "identity_disclosure": "ab9a7b113d58f3d3",
    "safety_agentic": "0dd753b33d0edb8c",
    "safety_chat_pressure": "34650e5f725ae448",
    "sov_ai_dynamic": "f2fb3d9172800ca5",
    "sov_ai_facts": "dfc5df24a426aad0",
    "sov_ai_multilingual_parity": "6d1ca33ea06f0977",
    "tool_calling": "820a7661b9bde4d2",
}


def test_every_registered_probe_has_a_pinned_trajectory_id() -> None:
    """Guard against a probe landing without a trajectory-identity decision."""
    assert set(_PINNED_TRAJECTORY_IDS) == set(known_probes())


def _pinned_inputs(probe_type: str) -> tuple[dict, ConversationSimulatorConfig, dict]:
    """A sampled row for ``probe_type`` with the pinned persona, seed and model names."""
    sampled = materialize_episode_inputs(
        locale="en_US",
        num_rows=1,
        probe_mix={probe_type: 1.0},
        random_seed=19,
    )[0]
    config = ConversationSimulatorConfig.model_validate(sampled["usersim_config"]).model_copy(
        update={"random_seed": 1234}
    )
    data = {key: value for key, value in sampled.items() if key not in _PERSONA_DERIVED_COLUMNS}
    data["persona"] = dict(_PINNED_PERSONA)
    models = {
        "user_model": SimpleNamespace(model_name="pinned-user-model"),
        "assistant_model": SimpleNamespace(model_name=_PINNED_ASSISTANT_MODEL),
    }
    return data, config, models


@pytest.mark.parametrize("probe_type", sorted(_PINNED_TRAJECTORY_IDS))
def test_trajectory_id_is_pinned_per_probe(probe_type: str) -> None:
    data, config, models = _pinned_inputs(probe_type)

    # construct_probe_episode, not construct_episode_preamble: the probe is built
    # here, which is where a probe-discovered variant could leak into the id.
    episode = construct_probe_episode(data, config=config, models=models)

    assert episode.preamble.trajectory_id == _PINNED_TRAJECTORY_IDS[probe_type]


@pytest.mark.parametrize("empty", [float("nan"), pd.NA], ids=["nan", "na"])
def test_an_empty_persona_field_changes_no_identity_or_setting(empty) -> None:
    """pandas holds an empty persona field as None, NaN or NA depending on its
    version. The persona's identity, the trajectory id and every setting drawn
    from the persona must not depend on which."""
    data, config, models = _pinned_inputs("tool_calling")
    with_none = construct_episode_preamble(
        {**data, "persona": {**data["persona"], "bachelors_field": None}}, config=config, models=models
    )
    with_empty = construct_episode_preamble(
        {**data, "persona": {**data["persona"], "bachelors_field": empty}}, config=config, models=models
    )

    assert with_empty.persona["bachelors_field"] is None
    for field in (
        "persona_uuid",
        "trajectory_id",
        "behavioral_profile",
        "disclosure_style",
        "user_interaction_style",
        "persona_grounding",
    ):
        assert getattr(with_empty, field) == getattr(with_none, field), field


@pytest.mark.parametrize("empty", [float("nan"), pd.NA], ids=["nan", "na"])
def test_a_toolset_without_a_name_is_described_by_its_category(empty) -> None:
    """Some shipped toolsets have no name, which pandas may hold as NaN or NA.
    The simulated user is then told the toolset's category instead."""
    data, config, models = _pinned_inputs("tool_calling")
    data.update(toolset_category="Smart Home & IoT", toolset_name=empty)

    prompt = construct_probe_episode(data, config=config, models=models).probe.get_user_system_prompt()

    assert "involves Smart Home & IoT capabilities" in prompt
