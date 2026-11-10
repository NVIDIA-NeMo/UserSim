# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A probe's run settings: declared in its ``Options``, given per run as ``probe_options``, read as ``self.options``.

A run's settings are checked against each probe's ``Options`` before any model
call, so an unknown probe, an unknown setting or a bad value stops the run
rather than failing every row of that probe.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from usersim import cli
from usersim.cli._pipeline import simulator_config, verify_probe_options
from usersim.cli.simulate import _parse_probe_options
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.manifest import _sim_config_from_dict, build_run_manifest
from usersim.engine.core.probes import BaseProbe, probe_options_problems
from usersim.engine.errors import ConfigError
from usersim.engine.models_file import default_models_path, load_models_config
from usersim.testing import probe_conformance_problems


class _BookingOptions(BaseProbe.Options):
    party_size: int = 2
    tone: str = "plain"


@pytest.fixture
def options_probe(register_test_probe):
    return register_test_probe("ext_options", Options=_BookingOptions)


def _probe(cls: type[BaseProbe], probe_options: dict) -> BaseProbe:
    config = ConversationSimulatorConfig(name="conversation_messages", probe_options=probe_options)
    return cls(persona={}, locale="en_US", language="English", models={}, cfg=config)


def test_a_probe_reads_its_settings_validated(options_probe) -> None:
    assert _probe(options_probe, {"ext_options": {"party_size": "4"}}).options == _BookingOptions(party_size=4)


def test_a_probe_left_out_of_the_settings_runs_on_its_defaults(options_probe) -> None:
    assert _probe(options_probe, {"general_open_ended": {}}).options == _BookingOptions()


def test_each_problem_with_a_run_s_settings_is_named(options_probe) -> None:
    problems = probe_options_problems({"ext_options": {"party_size": "many", "colour": "red"}, "no_such_probe": {}})

    assert problems[0] == "ext_options has no setting 'colour'; its settings: party_size, tone"
    assert problems[1].startswith("ext_options.party_size: Input should be a valid integer")
    assert problems[2] == "'no_such_probe' is not a registered probe"
    assert len(problems) == 3


def test_the_pre_flight_refuses_settings_a_probe_does_not_take(options_probe) -> None:
    verify_probe_options({"ext_options": {"tone": "warm"}})
    with pytest.raises(ConfigError, match="ext_options has no setting 'colour'"):
        verify_probe_options({"ext_options": {"colour": "red"}})


def _never(*args, **kwargs):
    raise AssertionError("the run got past its pre-flight")


@pytest.mark.parametrize(
    ("mode", "out"),
    [([], "out"), (["--dry-run"], "out"), (["--materialize-inputs"], "rows.jsonl")],
    ids=["run", "dry_run", "materialize"],
)
@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("general_open_ended.tone=warm", "general_open_ended has no setting 'tone'; its settings: none"),
        ("tool_calling.max_tools=0", "tool_calling.max_tools: Input should be greater than or equal to 1"),
    ],
    ids=["unknown_setting", "tools_out_of_range"],
)
def test_a_run_with_a_bad_setting_stops_before_any_model_call(
    tmp_path, caplog, monkeypatch, mode, out, option, message
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "not-used")
    monkeypatch.setattr("usersim.cli._pipeline.build_simulator_config_builder", _never)
    monkeypatch.setattr("usersim.cli._pipeline.materialize_episode_inputs", _never)
    argv = ["simulate", "--locale", "en_US", "--num-rows", "1", *mode, "--out", str(tmp_path / out)]
    with caplog.at_level(logging.ERROR):
        assert cli.main([*argv, "--probe-option", option]) == 2
    assert message in caplog.text


def test_the_library_refuses_bad_settings_before_materializing(monkeypatch) -> None:
    from usersim.engine.external import materialize_episode_inputs

    monkeypatch.setattr("usersim.cli._pipeline.materialize_episode_inputs", _never)
    with pytest.raises(ValueError, match="tool_calling.max_tools: Input should be greater than or equal to 1"):
        materialize_episode_inputs(
            locale="en_US",
            num_rows=1,
            probe_mix={"tool_calling": 1.0},
            probe_options={"tool_calling": {"max_tools": 0}},
        )


@pytest.mark.parametrize(
    ("specs", "expected"),
    [
        (["ext_options.party_size=4"], {"ext_options": {"party_size": 4}}),
        (["ext_options.tone=warm", "ext_options.party_size=2"], {"ext_options": {"tone": "warm", "party_size": 2}}),
        (['ext_options.tone="4"'], {"ext_options": {"tone": "4"}}),
    ],
)
def test_probe_option_flags_read_a_json_value_else_text(specs, expected) -> None:
    assert _parse_probe_options(specs) == expected


@pytest.mark.parametrize("spec", ["ext_options=4", ".party_size=4", "ext_options.=4", "ext_options.party_size"])
def test_a_probe_option_flag_names_a_probe_a_setting_and_a_value(spec) -> None:
    with pytest.raises(SystemExit, match=r"PROBE\.SETTING=VALUE"):
        _parse_probe_options([spec])


@pytest.mark.parametrize(
    ("labels", "problem"),
    [
        (lambda cls: ["ext_labels"], "must return a tuple of probe labels"),
        (lambda cls: (), "must return a tuple of probe labels"),
        (lambda cls: ("other", "ext_labels"), "must start with its own label"),
    ],
)
def test_asset_labels_that_do_not_start_with_the_probe_s_own_are_reported(register_test_probe, labels, problem) -> None:
    probe = register_test_probe("ext_labels", asset_labels=classmethod(labels))

    assert any(problem in found for found in probe_conformance_problems(probe))


def test_options_without_a_default_are_reported(register_test_probe) -> None:
    class _Required(BaseProbe.Options):
        party_size: int

    probe = register_test_probe("ext_required_options", Options=_Required)

    assert any("Options needs a default for every setting" in problem for problem in probe_conformance_problems(probe))


def test_the_manifest_records_a_run_s_settings_and_reads_them_back() -> None:
    settings = {"ext_options": {"party_size": 4}, "general_open_ended": {}}
    manifest = build_run_manifest(
        run_id="1700000001",
        started_at_epoch=1700000001,
        finished_at_epoch=1700000002,
        models_config=SimpleNamespace(providers=(), models=()),
        sim_config=simulator_config(
            locale="en_US", models=load_models_config(default_models_path()), max_turns=2, probe_options=settings
        ),
        locales_requested=["en_US"],
        probe_mix={"general_open_ended": 1.0},
        trajectory_df=None,
    )

    assert manifest.simulation_config.probe_options == {"ext_options": {"party_size": 4}}
    assert _sim_config_from_dict(asdict(manifest.simulation_config)) == manifest.simulation_config
    assert _sim_config_from_dict({}).probe_options == {}
