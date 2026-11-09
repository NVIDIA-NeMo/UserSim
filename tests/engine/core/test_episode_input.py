# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public, model-free episode-input construction contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
from _probe_contract import built_in_labels, contract_file, probe_contract

from usersim.cli._pipeline import known_probes
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core import probes
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


@pytest.mark.parametrize(
    ("probe_type", "kind"), [("general_open_ended", "topics"), ("general_educational", "subjects")]
)
def test_a_general_probe_draws_its_theme_from_its_seeds(probe_type: str, kind: str) -> None:
    from usersim.engine.core.seeds import load_seeds

    (row,) = materialize_episode_inputs(locale="en_US", num_rows=1, probe_mix={probe_type: 1.0}, random_seed=5)

    assert json.loads(row["theme"]) in load_seeds("en_US", probe_type, kind)


def test_a_tool_calling_row_stores_no_drawn_theme() -> None:
    """A tool_calling row's theme is its toolset, so the run draws none and the row holds the placeholder."""
    (row,) = materialize_episode_inputs(locale="en_US", num_rows=1, probe_mix={"tool_calling": 1.0}, random_seed=5)

    assert row["theme"] == "__not_used__"
    assert row["toolset_category"]


def test_materialized_rows_keep_the_seed_table_to_the_probe_that_reads_it() -> None:
    """A host gets the toolset columns on the tool_calling rows only, as a simulate run stores them."""
    probe_mix = {"tool_calling": 0.5, "general_open_ended": 0.5}
    # No seed fixes which probe a row samples, and the test persona dataset
    # gives at most four rows a call, so draw until each probe has rows.
    rows: list[dict[str, object]] = []
    for _ in range(10):
        rows += materialize_episode_inputs(locale="en_US", num_rows=4, probe_mix=probe_mix)
        if {row["probe_type"] for row in rows} == set(probe_mix):
            break
    columns = ("toolset_source", "toolset_category", "toolset_name", "num_tools", "tools")
    by_probe = {probe: [row for row in rows if row["probe_type"] == probe] for probe in probe_mix}

    assert by_probe["tool_calling"] and by_probe["general_open_ended"]
    assert all(row["tools"] and row["toolset_category"] for row in by_probe["tool_calling"])
    assert all(row.get(column) is None for row in by_probe["general_open_ended"] for column in columns)


def _themes_from_its_folder(cls: type, locale: str) -> list[dict]:
    return json.loads((cls.assets_dir() / locale / "themes.json").read_text(encoding="utf-8"))


def test_a_probes_themes_reach_its_rows(register_test_probe, tmp_path: Path) -> None:
    """A probe's themes, read under its own asset folder as its banks are, are what its rows sample."""
    (tmp_path / "ext_themes" / "en_US").mkdir(parents=True)
    bread = {"type": "Baking", "description": "Pain de campagne à la maison"}
    (tmp_path / "ext_themes" / "en_US" / "themes.json").write_text(json.dumps([bread]), encoding="utf-8")
    register_test_probe("ext_themes", assets_root=tmp_path, themes=classmethod(_themes_from_its_folder))

    (row,) = materialize_episode_inputs(locale="en_US", num_rows=1, probe_mix={"ext_themes": 1.0}, random_seed=5)

    assert json.loads(row["theme"]) == bread
    assert "à la maison" in row["theme"]


def _bank_root(tmp_path: Path, label: str) -> Path:
    """An asset root whose ``<label>/`` folder holds one bank file, as the pre-flight requires of every probe."""
    (tmp_path / label).mkdir()
    (tmp_path / label / "bank.yaml").write_text("entries: []\n", encoding="utf-8")
    return tmp_path


def test_a_probe_that_reads_no_theme_gets_a_placeholder(register_test_probe, tmp_path) -> None:
    register_test_probe("ext_no_themes", assets_root=_bank_root(tmp_path, "ext_no_themes"))

    (row,) = materialize_episode_inputs(locale="en_US", num_rows=1, probe_mix={"ext_no_themes": 1.0}, random_seed=5)

    assert row["theme"] == probes.NO_THEME


def test_a_probe_with_no_themes_for_the_locale_is_refused(register_test_probe, tmp_path) -> None:
    """An empty list leaves the sampler nothing to draw for the probe's rows."""
    from usersim.engine.errors import ConfigError

    register_test_probe(
        "ext_no_themes",
        assets_root=_bank_root(tmp_path, "ext_no_themes"),
        themes=classmethod(lambda cls, locale: []),
    )

    with pytest.raises(ConfigError, match="ext_no_themes has no themes for en_US"):
        materialize_episode_inputs(locale="en_US", num_rows=1, probe_mix={"ext_no_themes": 1.0}, random_seed=5)


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


@pytest.mark.parametrize("probe_type", built_in_labels())
def test_trajectory_id_is_pinned_per_probe(probe_type: str) -> None:
    """Each built-in's contract pins the id of one episode, so a change to its identity scheme fails."""
    data, config, models = _pinned_inputs(probe_type)

    # construct_probe_episode, not construct_episode_preamble: the probe is built
    # here, which is where a probe-discovered variant could leak into the id.
    episode = construct_probe_episode(data, config=config, models=models)

    produced = episode.preamble.trajectory_id
    assert produced == probe_contract(probe_type).trajectory_id, (
        f"{probe_type} produces trajectory_id {produced!r} for the pinned episode. For a new probe, "
        f"or a deliberate change to its identity, set it as trajectory_id in {contract_file(probe_type)}."
    )


def test_two_probes_in_one_module_record_their_own_metadata(
    monkeypatch: pytest.MonkeyPatch, register_test_probe
) -> None:
    """A probe's label, variant and prompt version come from its own class, so
    two probes defined in one module each record their own."""
    data, config, models = _pinned_inputs("general_open_ended")
    module = ModuleType("_two_probes_module")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for label, version, variant in (("_first_probe", "v1.0", "alpha"), ("_second_probe", "v2.0", "beta")):
        register_test_probe(label, prompt_version=version, variants=(variant,), __module__=module.__name__)

    first = construct_episode_preamble({**data, "probe_type": "_first_probe"}, config=config, models=models)
    second = construct_episode_preamble({**data, "probe_type": "_second_probe"}, config=config, models=models)

    assert (first.probe_family, first.probe_variant, first.provenance.scenario_prompt_version) == (
        "_first_probe",
        "alpha",
        "v1.0",
    )
    assert (second.probe_family, second.probe_variant, second.provenance.scenario_prompt_version) == (
        "_second_probe",
        "beta",
        "v2.0",
    )


def test_a_probe_that_reads_no_seed_table_sees_the_run_s_seed_columns_empty() -> None:
    """Data Designer gives every row of a run a row of its seed table; only the probe that reads it sees it."""
    data, config, models = _pinned_inputs("tool_calling")
    config = config.model_copy(update={"seed_dataset_columns": ["toolset_name", "num_tools"]})
    seeded = {**data, "toolset_name": "Weather", "num_tools": 3}

    tool = construct_episode_preamble(seeded, config=config, models=models)
    general = construct_episode_preamble(
        {**seeded, "probe_type": "general_open_ended", "theme": '{"type": "cooking"}'}, config=config, models=models
    )

    assert (tool.data["toolset_name"], tool.data["num_tools"]) == ("Weather", 3)
    assert (general.data["toolset_name"], general.data["num_tools"]) == (None, None)


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
