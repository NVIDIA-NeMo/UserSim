# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``replay_reasoning`` in the models file, from the TOML row to the simulator config.

The invariants: replay is off unless the models file turns it on for
``assistant_model``; a setup that cannot replay (the switch on another alias,
reasoning not stored, a provider whose requests drop reasoning) is refused
before any model call; and a run that replays never shares trajectory ids with
one that does not, while ids without replay stay what they were.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from usersim.cli._errors import ConfigError
from usersim.cli._models import (
    ModelsConfig,
    ProviderSpec,
    apply_model_overrides,
    assistant_replays_reasoning,
    default_models_path,
    load_models_config,
)
from usersim.cli._pipeline import build_simulator_config_builder
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.episode_input import construct_episode_preamble
from usersim.engine.core.manifest import SimulationConfigSnapshot, build_run_manifest


def _models(providers=(), **by_alias):
    """The default models, with per-alias changes and extra providers."""
    base = load_models_config(default_models_path())
    specs = tuple(replace(spec, **by_alias.get(spec.alias, {})) for spec in base.models)
    return ModelsConfig(providers=base.providers + tuple(providers), models=specs)


def _assistant(config):
    return next(spec for spec in config.models if spec.alias == "assistant_model")


_REPLAY = {"assistant_model": {"replay_reasoning": True}}


def test_replay_is_off_unless_the_models_file_turns_it_on(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[[models]]\nalias = "assistant_model"\nmodel = "x"\nprovider = "nvidia"\nreplay_reasoning = true\n'
        '[[models]]\nalias = "user_model"\nmodel = "x"\nprovider = "nvidia"\n'
    )
    config = load_models_config(path)

    assert _assistant(load_models_config(default_models_path())).replay_reasoning is False
    assert [spec.replay_reasoning for spec in config.models] == [True, False]


def test_only_the_assistant_takes_the_switch():
    with pytest.raises(ConfigError, match="assistant_model"):
        assistant_replays_reasoning(_models(user_model={"replay_reasoning": True}), store_reasoning=True)


def test_replay_needs_stored_reasoning():
    assert assistant_replays_reasoning(_models(**_REPLAY), store_reasoning=True) is True
    with pytest.raises(ConfigError, match="store"):
        assistant_replays_reasoning(_models(**_REPLAY), store_reasoning=False)


def test_replay_needs_a_provider_that_sends_reasoning():
    """Data Designer's Anthropic adapter drops reasoning from requests."""
    claude = ProviderSpec(name="claude", endpoint="https://example.invalid", provider_type="anthropic", api_key="K")
    config = _models([claude], assistant_model={"replay_reasoning": True, "provider": "claude"})

    with pytest.raises(ConfigError, match="anthropic"):
        assistant_replays_reasoning(config, store_reasoning=True)


def test_the_run_config_carries_the_switch(tmp_path):
    def column(models, **kwargs) -> ConversationSimulatorConfig:
        _, builder = build_simulator_config_builder(
            locale="en_US",
            models=models,
            assets_dir=packaged_assets_dir(),
            probe_mix={"general_open_ended": 1.0},
            **kwargs,
        )
        return builder.get_column_config("conversation_messages")

    assert column(_models()).replay_assistant_reasoning is False
    assert column(_models(**_REPLAY)).replay_assistant_reasoning is True
    with pytest.raises(ConfigError, match="store"):
        column(_models(**_REPLAY), store_reasoning=False)


def test_swapping_the_assistant_model_turns_replay_off_unless_the_override_says_so():
    """Replay is a property of the model: a different model has not been trained on it."""
    config = _models(**_REPLAY)
    model = "openai/gpt-oss-20b"  # one the default provider serves

    assert _assistant(apply_model_overrides(config, {"assistant_model": model})).replay_reasoning is False
    kept = apply_model_overrides(config, {"assistant_model": {"model": model, "replay_reasoning": True}})
    assert _assistant(kept).replay_reasoning is True


def _trajectory_id(**changes) -> str:
    config = ConversationSimulatorConfig(name="conversation_messages", locale="en_US", random_seed=1, **changes)
    data = {"persona": {"first_name": "Ana", "age": 30}, "probe_type": "general_open_ended", "theme": "cooking"}
    return construct_episode_preamble(data, config=config, models={}).trajectory_id


def test_ids_change_only_when_replay_is_on():
    assert _trajectory_id() == _trajectory_id(replay_assistant_reasoning=False)
    assert _trajectory_id(replay_assistant_reasoning=True) != _trajectory_id()


def test_the_manifest_records_the_switch():
    config = ConversationSimulatorConfig(name="conversation_messages", replay_assistant_reasoning=True)
    manifest = build_run_manifest(
        run_id="1700000000",
        started_at_epoch=1700000000,
        models_config=load_models_config(default_models_path()),
        sim_config=config,
        locales_requested=["en_US"],
        probe_mix={"general_open_ended": 1.0},
    )

    assert manifest.simulation_config.replay_assistant_reasoning is True
    assert SimulationConfigSnapshot.__dataclass_fields__["replay_assistant_reasoning"].default is False


def test_simulate_refuses_replay_without_stored_reasoning_before_running(tmp_path, caplog):
    import logging

    from usersim import cli

    models = tmp_path / "models.toml"
    models.write_text(
        Path(default_models_path())
        .read_text()
        .replace('alias = "assistant_model",', 'alias = "assistant_model", replay_reasoning = true,', 1)
    )
    argv = ["simulate", "--locale", "en_US", "--num-rows", "1", "--out", str(tmp_path / "o"), "--models", str(models)]

    with caplog.at_level(logging.ERROR):
        assert cli.main([*argv, "--no-store-reasoning", "--dry-run"]) == 2
    assert "replay_reasoning" in caplog.text
    assert cli.main([*argv, "--dry-run"]) == 0


def test_a_manifest_keeps_the_switch_when_read_back():
    from usersim.engine.core.manifest import RunManifest

    config = ConversationSimulatorConfig(name="conversation_messages", replay_assistant_reasoning=True)
    manifest = build_run_manifest(
        run_id="1700000000",
        started_at_epoch=1700000000,
        models_config=load_models_config(default_models_path()),
        sim_config=config,
        locales_requested=["en_US"],
        probe_mix={"general_open_ended": 1.0},
    )

    assert RunManifest.from_dict(manifest.to_dict()).simulation_config.replay_assistant_reasoning is True


def test_a_plain_run_records_the_switch_in_its_manifest(tmp_path):
    from unittest.mock import patch

    import pandas as pd

    from usersim.cli.simulate import _simulate_to_parquet
    from usersim.engine.core.manifest import load_run_manifest
    from usersim.engine.core.storage import latest_run_id

    rows = pd.DataFrame(
        {
            "trajectory_id": ["t1"],
            "locale": ["en_US"],
            "probe_family": ["general"],
            "probe_type": ["general_open_ended"],
        }
    )

    class _Designer:
        def create(self, builder, num_records=None):
            return type("Result", (), {"load_dataset": lambda self: rows})()

    with patch("usersim.cli._pipeline.build_simulator_config_builder", lambda **kwargs: (_Designer(), object())):
        _simulate_to_parquet(
            models=_models(**_REPLAY),
            models_path=default_models_path(),
            panel=None,
            locales=["en_US"],
            num_rows=1,
            out=tmp_path,
            assets_dir=packaged_assets_dir(),
            max_turns=1,
            probe_mix={"general_open_ended": 1.0},
            random_seed=1,
            skip_existing=False,
        )

    manifest = load_run_manifest(latest_run_id(tmp_path), tmp_path)
    assert manifest.simulation_config.replay_assistant_reasoning is True


def test_materialized_rows_carry_the_switch_to_a_host(monkeypatch):
    """A host learns the setting from the row's ``usersim_config``."""
    from usersim.cli._pipeline import materialize_episode_inputs

    monkeypatch.setenv("NVIDIA_API_KEY", "not-used")
    (row,) = materialize_episode_inputs(
        locale="en_US",
        num_rows=1,
        models=_models(**_REPLAY),
        assets_dir=packaged_assets_dir(),
        probe_mix={"general_open_ended": 1.0},
    )

    assert row["usersim_config"]["replay_assistant_reasoning"] is True


def test_the_config_itself_refuses_replay_without_stored_reasoning():
    with pytest.raises(ValueError, match="store_reasoning"):
        ConversationSimulatorConfig(
            name="conversation_messages", replay_assistant_reasoning=True, store_reasoning=False
        )


def test_the_switch_must_be_a_real_boolean(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[[models]]\nalias = "assistant_model"\nmodel = "x"\nprovider = "nvidia"\nreplay_reasoning = "false"\n'
    )

    with pytest.raises(ConfigError, match="true or false"):
        load_models_config(path)


def test_the_provider_type_is_matched_whatever_its_case():
    """Data Designer normalizes the case before choosing an adapter."""
    hub = ProviderSpec(name="hub", endpoint="https://example.invalid", provider_type="OpenAI", api_key="K")
    config = _models([hub], assistant_model={"replay_reasoning": True, "provider": "hub"})

    assert assistant_replays_reasoning(config, store_reasoning=True) is True


def test_an_override_that_keeps_the_model_keeps_replay(caplog):
    import logging

    # Models the default provider serves, so an override can route them.
    config = _models(assistant_model={"replay_reasoning": True, "model": "openai/gpt-oss-20b"})

    kept = apply_model_overrides(config, {"assistant_model": {"model": "openai/gpt-oss-20b", "temperature": 0.2}})
    assert _assistant(kept).replay_reasoning is True
    with caplog.at_level(logging.WARNING):
        swapped = apply_model_overrides(config, {"assistant_model": "openai/gpt-oss-120b"})
    assert _assistant(swapped).replay_reasoning is False
    assert "replay_reasoning is off" in caplog.text


def test_the_simulate_notebook_records_the_switch_in_its_manifest():
    import json

    notebook = Path(__file__).resolve().parents[1] / "notebooks" / "01_simulate.ipynb"
    source = "\n".join("".join(cell["source"]) for cell in json.loads(notebook.read_text())["cells"])

    assert "replay_assistant_reasoning=assistant_replays_reasoning(MODELS" in source
