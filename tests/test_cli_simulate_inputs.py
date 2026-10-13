# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim simulate --inputs``: stored rows run exactly as stored, one run per setup.

The invariants:

- two setups run over the same stored rows meet the same simulated users, in
  the same scenario, and pair on ``input_id``;
- each setup gets its own run and its own ``trajectory_id`` per row, so a
  resume never mixes two setups and never skips a row it did not run;
- everything that can change a conversation is part of the setup, and
  nothing that cannot is.

These run the real Data Designer pipeline. Every simulator-side model call is
stubbed, health checks are skipped, and the provider points at a closed local
port, so a call that slips past the stubs fails at once instead of going out.
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import shutil
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from test_simulator_pipeline_e2e import _mock_call_llm_side_effect

from usersim import cli
from usersim.cli import _inputs
from usersim.cli._errors import ConfigError
from usersim.cli._inputs import (
    INPUT_ID_COLUMN,
    load_episode_inputs,
    prepare_episode_inputs,
    setup_fingerprint,
    simulate_inputs_to_dataset,
)
from usersim.cli._models import ModelsConfig, ProviderSpec, default_models_path, load_models_config
from usersim.cli._models import to_model_configs as _real_to_model_configs
from usersim.cli._pipeline import materialize_episode_inputs
from usersim.engine import generator as generator_module
from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.manifest import load_run_manifest
from usersim.engine.core.missing import none_for_missing
from usersim.engine.core.storage import list_runs, read_partitioned_dataset

#: Every simulator-side module that holds its own reference to ``acall_llm``,
#: plus the module itself for the code that imports it at call time.
_LLM_MODULES = (
    "usersim.engine.core.llm",
    "usersim.engine.core.simulation",
    "usersim.engine.core.judges",
    "usersim.engine.core.context",
    "usersim.engine.core.translation",
    "usersim.engine.core.realized_audit",
    "usersim.engine.probes.health_disclosure.move_runtime",
    "usersim.engine.probes.safety_agentic.generator",
    "usersim.engine.probes.safety_chat_pressure.classifier",
    "usersim.engine.probes.tool_calling.generator",
)
_LLM_SITES = tuple(m for m in _LLM_MODULES if hasattr(importlib.import_module(m), "acall_llm"))


def _probe_variant_pairs() -> list[tuple[str, str]]:
    """Every registered probe paired with every variant it declares."""
    from usersim.engine.core.probes import known_probes, resolve_probe

    pairs = []
    for probe in sorted(known_probes()):
        module = sys.modules[resolve_probe(probe).__module__]
        pairs.extend((probe, str(v)) for v in module.PROBE_VARIANTS)
    return pairs


#: A closed port: anything that reaches a provider fails at once.
_UNREACHABLE = "http://127.0.0.1:9/v1"


def _models(provider_api_key: str = "NVIDIA_API_KEY", **by_alias: dict[str, Any]) -> ModelsConfig:
    """The default models, served from an unreachable provider, with per-alias changes."""
    base = load_models_config(default_models_path())
    provider = ProviderSpec(name="nvidia", endpoint=_UNREACHABLE, provider_type="openai", api_key=provider_api_key)
    specs = tuple(replace(spec, **by_alias.get(spec.alias, {})) for spec in base.models)
    return ModelsConfig(providers=(provider,), models=specs)


@contextmanager
def _offline(side_effect=_mock_call_llm_side_effect):
    """Stub every simulator-side model call and skip Data Designer's health checks."""

    def no_health_checks(models):
        return [m.model_copy(update={"skip_health_check": True}) for m in _real_to_model_configs(models)]

    with ExitStack() as stack:
        stack.enter_context(patch("usersim.cli._pipeline.to_model_configs", no_health_checks))
        for site in _LLM_SITES:
            stack.enter_context(patch(f"{site}.acall_llm", side_effect=side_effect))
        yield


@pytest.fixture(autouse=True)
def _environment(monkeypatch, tmp_path):
    # A Data Designer preview, which materializing rows runs, leaves an
    # artifacts folder in the working directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "not-used")
    # A value left set in the shell would change what the health variants run.
    monkeypatch.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)


def _materialize(probe_mix: dict[str, float], num_rows: int, seed: int = 1, max_turns: int = 1) -> list[dict[str, Any]]:
    return materialize_episode_inputs(
        locale="en_US",
        num_rows=num_rows,
        models=_models(),
        assets_dir=packaged_assets_dir(),
        probe_mix=probe_mix,
        random_seed=seed,
        max_turns=max_turns,
    )


@pytest.fixture(scope="module")
def _stored_rows(tmp_path_factory) -> list[dict[str, Any]]:
    """Four stored rows from one materialization, with both probes among them, made once per module.

    The probe of each row is drawn at random, and the offline persona sample
    holds only four people, so draw again until both probes appear.
    """
    with pytest.MonkeyPatch.context() as env:
        # What ``_environment`` sets for each test, for a fixture that outlives one.
        env.chdir(tmp_path_factory.mktemp("materialize"))
        env.setenv("NVIDIA_API_KEY", "not-used")
        env.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)
        for _ in range(20):
            stored = _materialize({"general_open_ended": 0.5, "tool_calling": 0.5}, num_rows=4)
            if {row["probe_type"] for row in stored} == {"general_open_ended", "tool_calling"}:
                return stored
    raise AssertionError("twenty draws never mixed the two probes")


@pytest.fixture
def rows(_stored_rows) -> list[dict[str, Any]]:
    """The module's stored rows, copied, so a test can change its own."""
    return copy.deepcopy(_stored_rows)


def _run(rows, out, models=None, *, assets_dir=None, **kwargs):
    with _offline():
        inputs = prepare_episode_inputs(rows, assets_dir=assets_dir)
        return simulate_inputs_to_dataset(inputs=inputs, models=models or _models(), out=out, **kwargs)


def _saved(out: Path, run_id: str):
    return read_partitioned_dataset(out, run=run_id)


def _fingerprint(rows, models=None, *, assets_dir=None) -> str:
    return setup_fingerprint(models or _models(), prepare_episode_inputs(rows, assets_dir=assets_dir))[0]


# ── two setups, one set of stored rows ──────────────────────────────────


def test_two_setups_get_their_own_runs_and_pair_on_input_id(rows, tmp_path):
    first = _run(rows, tmp_path)
    second = _run(rows, tmp_path, _models(assistant_model={"extra_body": {"reasoning_effort": "low"}}))

    assert first.run_id != second.run_id
    assert (first.written, second.written) == (len(rows), len(rows))
    x, y = _saved(tmp_path, first.run_id), _saved(tmp_path, second.run_id)
    assert set(x["trajectory_id"]).isdisjoint(y["trajectory_id"])
    pairs = x.merge(y, on=INPUT_ID_COLUMN, suffixes=("_x", "_y"))
    assert sorted(pairs[INPUT_ID_COLUMN]) == sorted(row["trajectory_id"] for row in rows)


def test_both_setups_offer_each_tool_calling_row_the_same_tools(rows, tmp_path):
    """The conversation runs with the stored id, from which ``tool_calling`` draws its tools."""
    first = _run(rows, tmp_path)
    second = _run(rows, tmp_path, _models(assistant_model={"temperature": 0.1}))

    pairs = _saved(tmp_path, first.run_id).merge(_saved(tmp_path, second.run_id), on=INPUT_ID_COLUMN)
    tools = pairs[pairs["probe_type_x"] == "tool_calling"]
    assert len(tools)
    assert tools["tool_subset_x"].astype(str).tolist() == tools["tool_subset_y"].astype(str).tolist()


def test_a_resume_reopens_the_run_with_the_same_setup(rows, tmp_path):
    first = _run(rows, tmp_path)
    _run(rows, tmp_path, _models(assistant_model={"temperature": 0.1}))

    again = _run(rows, tmp_path)

    assert (again.run_id, again.resumed, again.written, again.skipped) == (first.run_id, True, 0, len(rows))


def test_without_skip_existing_each_call_starts_a_run(rows, tmp_path):
    first = _run(rows, tmp_path, skip_existing=False)
    second = _run(rows, tmp_path, skip_existing=False)

    assert first.run_id != second.run_id
    assert len(list_runs(tmp_path)) == 2


# ── what is part of a setup ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "models",
    [
        pytest.param(lambda: _models(provider_api_key="ANOTHER_KEY_VARIABLE"), id="api_key"),
        pytest.param(lambda: _models(assistant_model={"max_parallel_requests": 32}), id="max_parallel_requests"),
    ],
)
def test_a_setting_that_cannot_change_a_conversation_keeps_the_setup(rows, models):
    assert _fingerprint(rows, models()) == _fingerprint(rows)


def test_debug_logging_keeps_the_setup(rows, monkeypatch):
    before = _fingerprint(rows)
    monkeypatch.setenv("USERSIM_DEBUG_LOG", "1")

    assert _fingerprint(rows) == before


@pytest.mark.parametrize(
    "models",
    [
        pytest.param(lambda: _models(assistant_model={"timeout": 5.0}), id="timeout"),
        pytest.param(lambda: _models(assistant_model={"temperature": 0.1}), id="temperature"),
        pytest.param(lambda: _models(judge_model={"extra_body": {"reasoning_effort": "low"}}), id="judge"),
        pytest.param(lambda: _models(assistant_model={"model": "another/model"}), id="model"),
        pytest.param(lambda: _models(assistant_model={"replay_reasoning": True}), id="replay"),
    ],
)
def test_a_model_setting_that_can_change_a_conversation_is_a_new_setup(rows, models):
    assert _fingerprint(rows, models()) != _fingerprint(rows)


def test_the_runs_models_file_decides_replay_whatever_the_rows_were_stored_with(rows, tmp_path):
    """The assistant under test comes from the run's models file, and so does replaying its reasoning."""
    from usersim.engine.core.llm import replaying_assistant_reasoning

    seen: list[bool] = []

    def recording(models, alias, msgs, **kwargs):
        if alias == "assistant_model":
            seen.append(replaying_assistant_reasoning())
        return _mock_call_llm_side_effect(models, alias, msgs, **kwargs)

    mixed = _edited(rows, lambda r: r[0]["usersim_config"].update(replay_assistant_reasoning=True))
    for stored, models, replayed in [
        (mixed, _models(), False),
        (rows, _models(assistant_model={"replay_reasoning": True}), True),
    ]:
        seen.clear()
        out = tmp_path / f"replay-{replayed}"
        with _offline(recording):
            result = simulate_inputs_to_dataset(inputs=prepare_episode_inputs(stored), models=models, out=out)
        assert seen and set(seen) == {replayed}
        assert load_run_manifest(result.run_id, out).simulation_config.replay_assistant_reasoning is replayed
    assert _fingerprint(mixed) == _fingerprint(rows)


def test_a_models_file_that_cannot_replay_is_refused_before_any_model_call(rows, tmp_path):
    unstored = _edited(rows, lambda r: [row["usersim_config"].update(store_reasoning=False) for row in r])
    replay = _models(assistant_model={"replay_reasoning": True})

    with patch("usersim.cli._pipeline.build_inputs_config_builder", side_effect=AssertionError("ran")):
        with pytest.raises(ConfigError, match="store"):
            _run(unstored, tmp_path / "out", replay)
    assert list_runs(tmp_path / "out") == []


def test_a_providers_own_request_settings_are_part_of_the_setup(rows, monkeypatch):
    """Data Designer merges a provider's extra_body and extra_headers into every request it serves."""
    from data_designer.config.models import ModelProvider

    models = ModelsConfig(providers=(), models=load_models_config(default_models_path()).models)

    def providers(**extra):
        provider = ModelProvider(name="nvidia", endpoint=_UNREACHABLE, provider_type="openai", api_key="K", **extra)
        monkeypatch.setattr("data_designer.config.default_model_settings.get_default_providers", lambda: [provider])

    providers()
    plain = setup_fingerprint(models, prepare_episode_inputs(rows))[0]
    providers(extra_body={"reasoning_effort": "low"})
    with_body = setup_fingerprint(models, prepare_episode_inputs(rows))[0]
    providers(extra_headers={"Authorization": "Bearer secret-value"})
    with_header, record = setup_fingerprint(models, prepare_episode_inputs(rows))

    assert len({plain, with_body, with_header}) == 3
    assert "secret-value" not in json.dumps(record)


def test_changed_assets_are_a_new_setup_and_moved_ones_are_not(rows, tmp_path):
    first, second = tmp_path / "assets", tmp_path / "elsewhere"
    shutil.copytree(packaged_assets_dir(), first)
    shutil.copytree(packaged_assets_dir(), second)
    moved = _fingerprint(rows, assets_dir=first)
    readme = next(first.rglob("README.md"))
    readme.write_text(readme.read_text() + "\nedited\n")

    assert _fingerprint(rows, assets_dir=second) == moved
    assert _fingerprint(rows, assets_dir=first) != moved


def test_a_change_in_a_lower_priority_asset_root_is_a_new_setup(monkeypatch, tmp_path):
    """A spec that ``extends`` its own probe reads the copy below it, so every root counts."""
    top, below = tmp_path / "top", tmp_path / "below"
    shutil.copytree(packaged_assets_dir(), top)
    shutil.copytree(packaged_assets_dir(), below)
    monkeypatch.setenv("USERSIM_ASSET_PATH", str(below))
    before = _inputs._asset_roots_digest(top)
    shadowed = next(below.rglob("spec.yaml"))
    shadowed.write_text(shadowed.read_text() + "\n# edited\n")

    assert _inputs._asset_roots_digest(top) != before


def test_a_data_designer_release_is_part_of_the_setup(rows, monkeypatch):
    before = _fingerprint(rows)
    monkeypatch.setattr(_inputs, "_data_designer_version", lambda: "999.0")

    assert _fingerprint(rows) != before


def test_the_setup_is_known_on_a_machine_where_data_designer_never_ran(rows, monkeypatch):
    """Without its providers file, Data Designer serves the built-in providers it would write there."""

    def missing():
        raise FileNotFoundError("model_providers.yaml")

    models = ModelsConfig(providers=(), models=load_models_config(default_models_path()).models)
    monkeypatch.setattr("data_designer.config.default_model_settings.get_default_providers", missing)

    fingerprint, record = setup_fingerprint(models, prepare_episode_inputs(rows))
    assert record["models"]["assistant_model"]["provider"]["endpoint"]


def test_a_prompt_version_is_part_of_the_setup(rows, monkeypatch):
    from usersim.engine.core.probes import resolve_probe

    before = _fingerprint(rows)
    module = sys.modules[resolve_probe("general_open_ended").__module__]
    monkeypatch.setattr(module, "PROMPT_VERSION", "v999", raising=False)

    assert _fingerprint(rows) != before


def test_a_behaviour_switch_in_the_environment_is_a_new_setup(rows, monkeypatch):
    before = _fingerprint(rows)
    monkeypatch.setenv("USERSIM_DISCLOSURE_STANDIN", "1")

    assert _fingerprint(rows) != before


def test_a_bank_override_counts_by_its_contents(rows, monkeypatch, tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text('{"items": [1]}')
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text('{"items": [1]}')
    monkeypatch.setenv("USERSIM_SAFETY_CHAT_PRESSURE_BANK", str(bank))
    original = _fingerprint(rows)
    monkeypatch.setenv("USERSIM_SAFETY_CHAT_PRESSURE_BANK", str(elsewhere))
    moved = _fingerprint(rows)
    elsewhere.write_text('{"items": [2]}')

    assert moved == original
    assert _fingerprint(rows) != original


def test_an_extension_release_is_part_of_the_setup(rows, monkeypatch):
    # The suite runs with discovery off; this test is about what it finds.
    monkeypatch.delenv("USERSIM_DISABLE_EXTENSIONS", raising=False)

    def installed(version: str):
        entry = SimpleNamespace(
            name="partner_probe",
            value="partner.probes:PartnerProbe",
            dist=SimpleNamespace(name="partner", version=version),
        )
        monkeypatch.setattr(
            "importlib.metadata.entry_points", lambda group: [entry] if group == "usersim.probes" else []
        )

    installed("1.0")
    first = _fingerprint(rows)
    installed("2.0")

    assert _fingerprint(rows) != first


# ── changed rows, concurrent writers, failures ──────────────────────────


def test_a_changed_stored_row_is_refused_before_any_model_call(rows, tmp_path):
    first = _run(rows, tmp_path)
    changed = copy.deepcopy(rows)
    changed[0]["theme"] = "a theme edited after the run"

    with (
        patch("usersim.cli._pipeline.build_inputs_config_builder", side_effect=AssertionError("simulated")),
        pytest.raises(ValueError, match="different contents"),
    ):
        _run(changed, tmp_path)
    assert len(_saved(tmp_path, first.run_id)) == len(rows)


def _stale_plan(monkeypatch):
    """Make a run plan as if it had started before anything was saved."""
    plan = _inputs.plan_inputs_run

    def before_anything_was_saved(inputs, models, out, *, skip_existing=True):
        fresh = plan(inputs, models, out, skip_existing=False)
        return replace(fresh, matching_run=None, pending={row["trajectory_id"] for row in inputs.rows}, saved=0)

    monkeypatch.setattr(_inputs, "plan_inputs_run", before_anything_was_saved)


def test_rows_another_writer_saved_meanwhile_are_not_saved_twice(rows, tmp_path, monkeypatch):
    first = _run(rows, tmp_path)
    _stale_plan(monkeypatch)

    second = _run(rows, tmp_path)

    assert (second.run_id, second.written, second.skipped) == (first.run_id, 0, len(rows))
    assert list_runs(tmp_path) == [first.run_id]
    assert len(_saved(tmp_path, first.run_id)) == len(rows)


def test_a_conflict_saved_meanwhile_is_refused_at_the_write(rows, tmp_path, monkeypatch):
    first = _run(rows, tmp_path)
    changed = copy.deepcopy(rows)
    changed[0]["theme"] = "a theme edited after the run"
    _stale_plan(monkeypatch)

    with pytest.raises(ValueError, match="different contents"):
        _run(changed, tmp_path)
    assert len(_saved(tmp_path, first.run_id)) == len(rows)


def test_a_failure_before_any_save_leaves_no_run(rows, tmp_path):
    with (
        patch("usersim.cli._pipeline.build_inputs_config_builder", side_effect=RuntimeError("simulated")),
        pytest.raises(RuntimeError),
    ):
        _run(rows, tmp_path)

    assert list_runs(tmp_path) == []
    assert not list(tmp_path.glob(".claim-*"))


# ── the stored rows reach the probe, and the saved rows look like a plain run's ──


@contextmanager
def _built_rows():
    """Collect the rows the probe is built from."""
    built: list[dict[str, Any]] = []
    construct = generator_module.construct_probe_episode

    def spy(data, **kwargs):
        built.append(dict(data))
        return construct(data, **kwargs)

    with patch.object(generator_module, "construct_probe_episode", spy):
        yield built


@contextmanager
def _undeclared_column_warnings():
    seen: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING and "side_effect_columns" in record.getMessage():
                seen.append(record.getMessage())

    handler, engine_logger = _Capture(), logging.getLogger("usersim.engine")
    engine_logger.addHandler(handler)
    try:
        yield seen
    finally:
        engine_logger.removeHandler(handler)


@pytest.mark.parametrize(("probe_type", "variant"), _probe_variant_pairs())
def test_every_probe_is_built_from_its_stored_row_exactly(probe_type, variant, tmp_path):
    (stored,) = _materialize({probe_type: 1.0}, num_rows=1)
    stored["probe_variant"] = variant

    with _built_rows() as built, _undeclared_column_warnings() as warnings:
        result = _run([stored], tmp_path)

    (row,) = built
    expected = none_for_missing({key: value for key, value in stored.items() if key != "usersim_config"})
    # The probe gets plain Python values, while a stored row may hold numpy
    # scalars (a pandas 2 preview leaves `theme` as one).
    for key, value in expected.items():
        want = value.item() if isinstance(value, np.generic) else value
        assert row[key] == want and type(row[key]) is type(want), key
    assert warnings == []
    (saved_id,) = _saved(tmp_path, result.run_id)[INPUT_ID_COLUMN]
    assert saved_id == stored["trajectory_id"]


def test_values_keep_their_types_across_rows(tmp_path):
    """A seed table would turn this int into a float, because another row holds a float."""
    stored = _materialize({"general_open_ended": 1.0}, num_rows=2)
    stored[0]["extension_score"], stored[1]["extension_score"] = 9007199254740993, 0.5
    stored[0]["extension_memory"] = {"sessions": [{"week": 1, "notes": ["slept badly"]}], "flag": True}
    stored[1]["extension_memory"] = {"sessions": [], "flag": False}

    with _built_rows() as built:
        _run(stored, tmp_path)

    by_id = {row["trajectory_id"]: row for row in built}
    for row in stored:
        for key in ("extension_score", "extension_memory"):
            got = by_id[row["trajectory_id"]][key]
            assert got == row[key] and type(got) is type(row[key])


def test_an_override_column_on_a_stored_row_reaches_the_probe(tmp_path):
    """A probe's override column, set on a stored row, picks what the probe runs.

    ``usersim simulate --panel`` reads only a panel's locales, so this is the
    way to pin an override: materialize rows, set the column, run them.
    """
    import yaml

    (stored,) = _materialize({"safety_chat_pressure": 1.0}, num_rows=1)
    first = _run([stored], tmp_path / "derived")
    (derived,) = _saved(tmp_path / "derived", first.run_id)["strategy_id"]
    bank = yaml.safe_load((packaged_assets_dir() / "safety_chat_pressure" / "sample.yaml").read_text(encoding="utf-8"))
    chosen = next(strategy["id"] for strategy in bank["strategies"] if strategy["id"] != derived)

    stored["pressure_strategy_id"] = chosen
    second = _run([stored], tmp_path / "chosen")

    assert list(_saved(tmp_path / "chosen", second.run_id)["strategy_id"]) == [chosen]


@pytest.mark.parametrize("probe_type", ["general_open_ended", "tool_calling"])
def test_saved_rows_have_a_plain_runs_columns_plus_input_id(probe_type, tmp_path):
    """The columns of a plain run, and the stored values in them as stored.

    The ``persona_*`` columns, and ``tool_calling``'s toolset columns, are
    added back from the stored rows, so they must keep the stored text and
    numbers rather than come back encoded.
    """
    from usersim.cli.simulate import _simulate_to_parquet

    plain_out, inputs_out = tmp_path / "plain", tmp_path / "inputs"
    with _offline():
        _simulate_to_parquet(
            models=_models(),
            models_path=None,
            panel=None,
            locales=["en_US"],
            num_rows=2,
            out=plain_out,
            assets_dir=packaged_assets_dir(),
            max_turns=1,
            probe_mix={probe_type: 1.0},
            random_seed=1,
            skip_existing=True,
        )
    stored = _materialize({probe_type: 1.0}, num_rows=2)
    result = _run(stored, inputs_out)

    plain = read_partitioned_dataset(plain_out)
    saved = _saved(inputs_out, result.run_id)
    assert set(saved.columns) == set(plain.columns) | {INPUT_ID_COLUMN}
    by_input = saved.set_index(INPUT_ID_COLUMN)
    for row in stored:
        shared = sorted(set(row) & set(plain.columns) - {"trajectory_id"})
        values = none_for_missing({key: by_input.loc[row["trajectory_id"], key] for key in shared})
        for key in shared:
            got, want = (value.item() if isinstance(value, np.generic) else value for value in (values[key], row[key]))
            assert got == want and type(got) is type(want), key


@pytest.mark.parametrize(
    ("probe_type", "variant"),
    [("general_open_ended", "default"), ("tool_calling", "default"), ("health_therapy_disclosure", "guarded")],
)
def test_only_input_id_holds_the_stored_id(probe_type, variant, tmp_path):
    """No other saved value contains a stored id, JSON columns included.

    The conversation runs under the stored id, and two of these probes read it
    mid-run: ``tool_calling`` draws its tools from it, and a guarded health
    variant stamps it on every move it reports to a move hook, which therefore
    sees the stored id, the saved row's ``input_id``. Moves start on the second
    turn.
    """
    from usersim.engine.probes.health_disclosure.mixin import GuardedMoveMixin

    stored = _materialize({probe_type: 1.0}, num_rows=2, max_turns=3)
    for row in stored:
        row["probe_variant"] = variant
    reported: list[dict[str, Any]] = []
    previous = GuardedMoveMixin.register_move_hook(reported.append)
    try:
        result = _run(stored, tmp_path)
    finally:
        GuardedMoveMixin.register_move_hook(previous)

    saved = _saved(tmp_path, result.run_id)
    stored_ids = {row["trajectory_id"] for row in stored}
    assert set(saved[INPUT_ID_COLUMN]) == stored_ids
    if variant == "guarded":
        assert {payload["trajectory_id"] for payload in reported} == stored_ids
    for column in saved.columns.drop(INPUT_ID_COLUMN):
        text = saved[column].astype(str)
        for row in stored:
            assert not text.str.contains(row["trajectory_id"], regex=False).any(), column


def test_the_manifest_describes_the_run(rows, tmp_path, monkeypatch):
    source = tmp_path / "rows.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    # The invocation started long before its first rows were saved.
    monkeypatch.setattr(_inputs, "time", SimpleNamespace(time=lambda: 1_700_000_000.4))
    with _offline():
        inputs = prepare_episode_inputs(load_episode_inputs(source), source=source)
        result = simulate_inputs_to_dataset(inputs=inputs, models=_models(), out=tmp_path / "out")

    manifest = load_run_manifest(result.run_id, tmp_path / "out")
    assert int(result.run_id) > 1_700_000_000
    assert manifest.started_at_epoch == 1_700_000_000
    assert manifest.scope.panel_path == str(source.resolve())
    assert manifest.scope.panel_content_hash
    assert manifest.scope.trajectories_requested == {"en_US": len(rows)}
    assert manifest.simulation_config.max_turns == rows[0]["usersim_config"]["max_turns"]
    assert manifest.source.kind == "cli" and manifest.source.invocation


def test_a_plain_run_records_when_it_started_and_leaves_no_artifacts(tmp_path, monkeypatch):
    """Data Designer names artifacts by the second a batch starts; each batch gets its own folder."""
    from usersim.cli import simulate

    monkeypatch.setattr(simulate, "time", SimpleNamespace(time=lambda: 1_700_000_000.4))
    with _offline():
        simulate._simulate_to_parquet(
            models=_models(),
            models_path=None,
            panel=None,
            locales=["en_US"],
            num_rows=2,
            out=tmp_path / "plain",
            assets_dir=packaged_assets_dir(),
            max_turns=1,
            probe_mix={"general_open_ended": 1.0},
            random_seed=1,
            skip_existing=True,
        )

    (run_id,) = list_runs(tmp_path / "plain")
    assert load_run_manifest(run_id, tmp_path / "plain").started_at_epoch == 1_700_000_000
    assert not (tmp_path / "artifacts").exists()


def test_an_inputs_run_leaves_no_artifacts(rows, tmp_path):
    """Its Data Designer batches write into temporary folders, not the working directory."""
    _run(rows, tmp_path / "out")

    assert not (tmp_path / "artifacts").exists()


def test_saved_columns_keep_their_stored_values(tmp_path):
    """A column of one plain type is saved as it is; anything Arrow would change is saved as JSON text."""
    stored = _materialize({"general_open_ended": 1.0}, num_rows=2)
    stored[0]["extension_count"], stored[1]["extension_count"] = 9007199254740993, None
    stored[0]["extension_score"], stored[1]["extension_score"] = 9007199254740993, 0.5
    stored[0]["extension_note"], stored[1]["extension_note"] = {"mood": "low"}, "plain text"

    result = _run(stored, tmp_path)

    saved = _saved(tmp_path, result.run_id).set_index(INPUT_ID_COLUMN)
    first, second = (saved.loc[row["trajectory_id"]] for row in stored)
    assert int(first["extension_count"]) == 9007199254740993
    assert [json.loads(first["extension_score"]), json.loads(second["extension_score"])] == [9007199254740993, 0.5]
    assert [json.loads(first["extension_note"]), json.loads(second["extension_note"])] == [
        {"mood": "low"},
        "plain text",
    ]


def _two_rows_with(first: Any, second: Any) -> list[dict[str, Any]]:
    stored = _materialize({"general_open_ended": 1.0}, num_rows=2)
    stored[0]["extension_value"], stored[1]["extension_value"] = first, second
    return stored


def test_a_resume_saves_each_column_as_the_first_invocation_did(tmp_path):
    """Whether a column is saved as JSON text is chosen from all the stored rows, not the ones left to run."""
    stored = _two_rows_with(9007199254740993, 0.5)
    with _offline():
        inputs = prepare_episode_inputs(stored)
        plan = _inputs.plan_inputs_run(inputs, _models(), tmp_path)
        # The first invocation stops after saving the first row.
        first = simulate_inputs_to_dataset(
            inputs=inputs, models=_models(), out=tmp_path, plan=replace(plan, pending={stored[0]["trajectory_id"]})
        )
    second = _run(stored, tmp_path)

    assert second.run_id == first.run_id and (first.written, second.written) == (1, 1)
    saved = _saved(tmp_path, first.run_id).set_index(INPUT_ID_COLUMN)["extension_value"]
    assert [json.loads(saved[row["trajectory_id"]]) for row in stored] == [9007199254740993, 0.5]


def test_batches_whose_fields_keep_their_type_read_back_exactly(tmp_path):
    """Rows run in two calls into one run, a field holding one type in both."""
    stored = _two_rows_with(9007199254740993, 7)
    first = _run(stored[:1], tmp_path)
    second = _run(stored[1:], tmp_path)

    assert second.run_id == first.run_id
    saved = _saved(tmp_path, first.run_id).set_index(INPUT_ID_COLUMN)["extension_value"]
    assert [int(saved[row["trajectory_id"]]) for row in stored] == [9007199254740993, 7]


def test_concurrent_invocations_save_each_row_once(rows, tmp_path):
    import threading

    results, errors = [], []

    def invoke():
        try:
            inputs = prepare_episode_inputs(rows)
            results.append(simulate_inputs_to_dataset(inputs=inputs, models=_models(), out=tmp_path / "out"))
        except Exception as error:  # surfaced below
            errors.append(error)

    with _offline():
        threads = [threading.Thread(target=invoke) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)

    assert errors == []
    (run_id,) = list_runs(tmp_path / "out")
    saved = _saved(tmp_path / "out", run_id)
    assert sorted(saved[INPUT_ID_COLUMN]) == sorted(row["trajectory_id"] for row in rows)
    assert sum(result.written for result in results) == len(rows)


def test_rows_stay_in_the_run_the_plan_found(rows, tmp_path, monkeypatch):
    """A newer run with the same setup, made meanwhile, does not split the rows across two runs."""
    found = _run(rows[:2], tmp_path)
    newer = _run(rows[:2], tmp_path, skip_existing=False)
    matching = _inputs._matching_run
    calls = []

    def first_the_found_run_then_the_newer_one(out, fingerprint):
        calls.append(fingerprint)
        return found.run_id if len(calls) == 1 else matching(out, fingerprint)

    monkeypatch.setattr(_inputs, "_matching_run", first_the_found_run_then_the_newer_one)
    result = _run(rows, tmp_path)

    assert (result.run_id, result.written, result.skipped) == (found.run_id, len(rows) - 2, 2)
    assert len(_saved(tmp_path, found.run_id)) == len(rows)
    assert len(_saved(tmp_path, newer.run_id)) == 2


def test_a_conflict_found_when_saving_exits_with_a_message(rows, tmp_path, monkeypatch):
    out = tmp_path / "o"
    argv = ["simulate", "--inputs", str(_write(rows, tmp_path / "rows.jsonl")), "--out", str(out)]
    with _offline():
        assert cli.main(argv) == 0
        changed = _edited(rows, lambda r: r[0].update(theme="a theme edited after the run"))
        argv[2] = str(_write(changed, tmp_path / "changed.jsonl"))
        _stale_plan(monkeypatch)
        with pytest.raises(SystemExit, match="different contents"):
            cli.main(argv)


# ── checks before any model call ────────────────────────────────────────


def _edited(rows, edit):
    changed = copy.deepcopy(rows)
    edit(changed)
    return changed


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: r[1]["usersim_config"].update(max_turns=9), "different settings"),
        (lambda r: r[1].update(trajectory_id=r[0]["trajectory_id"]), "more than one row"),
        (lambda r: r[0].update(input_id="x"), "input_id"),
        (lambda r: r[0].update(usersim_episode_input="{}"), "usersim_episode_input"),
        (lambda r: r[0].pop("usersim_config"), "usersim_config"),
        (lambda r: r[0].update(probe_type="no_such_probe"), "no_such_probe"),
        (lambda r: r[0].update(locale="pt_BR"), "locale"),
        (lambda r: r[0].update(persona=None), "persona"),
        (lambda r: r[0].update(extension_blob=b"bytes"), "JSON"),
        (lambda r: r[0].update(trajectory_id=""), "trajectory_id"),
    ],
)
def test_rows_that_cannot_run_are_refused(rows, edit, message):
    with pytest.raises(ValueError, match=message):
        prepare_episode_inputs(_edited(rows, edit))


def test_a_missing_stored_asset_folder_is_an_error_until_one_is_given(rows):
    moved = _edited(rows, lambda r: [row["usersim_config"].update(assets_dir="/no/such/assets") for row in r])

    with pytest.raises(ValueError, match="--assets-dir"):
        prepare_episode_inputs(moved)
    assert prepare_episode_inputs(moved, assets_dir=packaged_assets_dir()).assets_dir == packaged_assets_dir()


def _write(rows, path: Path) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max-turns", "9"),
        ("--probe-mix", "general_open_ended=1"),
        ("--random-seed", "4"),
        ("--no-store-reasoning", None),
    ],
)
def test_a_flag_the_stored_rows_settle_is_refused(rows, tmp_path, flag, value):
    argv = ["simulate", "--inputs", str(_write(rows, tmp_path / "rows.jsonl")), "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match=flag):
        cli.main([*argv, flag, *([value] if value else []), "--dry-run"])


def test_a_dry_run_shows_the_plan_without_running(rows, tmp_path, capsys):
    argv = [
        "simulate",
        "--inputs",
        str(_write(rows, tmp_path / "rows.jsonl")),
        "--out",
        str(tmp_path / "o"),
        "--dry-run",
    ]
    with patch("usersim.cli._pipeline.build_inputs_config_builder", side_effect=AssertionError("ran")):
        assert cli.main(argv) == 0

    out = capsys.readouterr().out
    assert "dry run" in out and "new, assigned when the first rows are saved" in out
    assert f"max_turns={rows[0]['usersim_config']['max_turns']!r}" in out
    assert list_runs(tmp_path / "o") == []


def test_a_root_without_file_locks_still_gets_its_run_with_one_warning(rows, tmp_path, monkeypatch, caplog):
    """Shared filesystems such as Lustre are often mounted without ``flock``; stored rows run there anyway."""
    fcntl = pytest.importorskip("fcntl")
    import errno

    def unsupported(*args, **kwargs):
        raise OSError(errno.ENOSYS, "flock is not supported here")

    monkeypatch.setattr(fcntl, "flock", unsupported)
    with caplog.at_level(logging.WARNING, logger="usersim.engine"):
        first = _run(rows[:2], tmp_path)
        second = _run(rows, tmp_path)

    assert second.run_id == first.run_id and second.resumed
    assert set(_saved(tmp_path, first.run_id)[INPUT_ID_COLUMN]) == {row["trajectory_id"] for row in rows}
    assert sum("not supported" in record.getMessage() for record in caplog.records) == 1


def test_rows_stored_as_parquet_load_as_they_were_written(rows, tmp_path):
    """``--materialize-inputs`` writes parquet through pandas; reading it back must not re-type it."""
    import pandas as pd

    path = tmp_path / "rows.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    loaded = load_episode_inputs(path)
    inputs = prepare_episode_inputs(loaded)

    assert [row["trajectory_id"] for row in inputs.rows] == [row["trajectory_id"] for row in rows]
    assert [row["usersim_config"]["max_turns"] for row in loaded] == [
        row["usersim_config"]["max_turns"] for row in rows
    ]
    assert all(isinstance(row["persona"], dict) for row in loaded)
