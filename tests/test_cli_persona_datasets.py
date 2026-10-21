# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona datasets are checked before a run's first model call.

The suite's ``synthetic_persona_dataset`` fixture provides ``en_US`` only, so
every other locale here is one whose dataset has not been downloaded.
"""

from __future__ import annotations

import logging

import pytest
from data_designer.config.utils.constants import LOCALES_WITH_MANAGED_DATASETS

from usersim import cli
from usersim.cli._models import bundled_models_path
from usersim.cli._persona_language import persona_dataset_report, persona_datasets_dir


def _cli(*args: str) -> list[str]:
    return [*args, "--models", str(bundled_models_path())]


def test_the_check_reads_the_folder_data_designer_samples_from(synthetic_persona_dataset):
    assert persona_datasets_dir() == synthetic_persona_dataset / "datasets"
    assert persona_dataset_report(["en_US"]) is None


def test_a_missing_dataset_names_its_file_and_the_download_command(synthetic_persona_dataset):
    report = persona_dataset_report(["en_US", "ja_JP"])

    expected = synthetic_persona_dataset / "datasets" / "ja_JP.parquet"
    assert f"  - ja_JP: not downloaded; expected at {expected}" in report
    assert "  data-designer download personas --locale ja_JP\n" in report
    assert "en_US" not in report


def test_india_language_variants_share_one_en_IN_download():
    report = persona_dataset_report(["ta_Taml_IN", "bn_Beng_IN"])

    assert "  - en_IN (for ta_Taml_IN, bn_Beng_IN): not downloaded" in report
    assert report.count("--locale en_IN") == 1


@pytest.mark.parametrize("locale", ["en_GB", "de_DE"])
def test_a_locale_without_a_data_designer_dataset_lists_the_ones_with_one(locale):
    report = persona_dataset_report([locale])

    assert f"  - {locale}: Data Designer provides no persona dataset for this locale" in report
    assert "download personas" not in report
    assert f"Data Designer provides persona datasets for: {', '.join(LOCALES_WITH_MANAGED_DATASETS)}." in report


def test_simulate_stops_before_building_the_run(tmp_path, monkeypatch, caplog):
    def build(**kwargs):
        pytest.fail("the run was built although its persona dataset is missing")

    monkeypatch.setattr("usersim.cli._pipeline.build_simulator_config_builder", build)
    with caplog.at_level(logging.ERROR):
        rc = cli.main(_cli("simulate", "--locale", "ja_JP", "--num-rows", "1", "--out", str(tmp_path / "out")))

    assert rc == 2
    assert "data-designer download personas --locale ja_JP" in caplog.text


def test_materializing_inputs_stops_before_sampling(tmp_path, monkeypatch, caplog):
    def materialize(**kwargs):
        pytest.fail("personas were sampled although their dataset is missing")

    monkeypatch.setattr("usersim.cli._pipeline.materialize_episode_inputs", materialize)
    argv = _cli(
        "simulate",
        "--locale",
        "ja_JP",
        "--num-rows",
        "1",
        "--materialize-inputs",
        "--out",
        str(tmp_path / "rows.jsonl"),
    )
    with caplog.at_level(logging.ERROR):
        rc = cli.main(argv)

    assert rc == 2
    assert "data-designer download personas --locale ja_JP" in caplog.text
    assert not (tmp_path / "rows.jsonl").exists()


def test_panel_stops_before_creating_a_designer(tmp_path, monkeypatch, caplog):
    def designer(*args, **kwargs):
        pytest.fail("Data Designer was created although the persona dataset is missing")

    monkeypatch.setattr("data_designer.interface.DataDesigner", designer)
    with caplog.at_level(logging.ERROR):
        rc = cli.main(
            _cli("panel", "--locale", "ja_JP", "--num-personas", "1", "--out", str(tmp_path / "panel.parquet"))
        )

    assert rc == 2
    assert "data-designer download personas --locale ja_JP" in caplog.text


def _dry_run(command: str, locale: str, tmp_path) -> list[str]:
    size = ["--num-rows", "1", "--out", str(tmp_path / "out")]
    if command == "panel":
        size = ["--num-personas", "1", "--out", str(tmp_path / "panel.parquet")]
    return _cli(command, "--dry-run", "--locale", locale, *size)


@pytest.mark.parametrize("command", ["simulate", "panel"])
def test_a_dry_run_reports_a_missing_dataset_and_still_succeeds(command, tmp_path, capsys):
    assert cli.main(_dry_run(command, "ja_JP", tmp_path)) == 0

    out = capsys.readouterr().out
    assert "  personas:      not available, so a live run stops before" in out
    assert "data-designer download personas --locale ja_JP" in out


@pytest.mark.parametrize("command", ["simulate", "panel"])
def test_a_dry_run_says_when_personas_are_ready(command, tmp_path, capsys):
    assert cli.main(_dry_run(command, "en_US", tmp_path)) == 0

    out = capsys.readouterr().out
    assert "  personas:      ready\n" in out
    assert "download personas" not in out


def test_stored_rows_need_no_persona_dataset(tmp_path, monkeypatch):
    """``--inputs`` replays the personas stored on each row, so the dataset check never runs."""
    # A Data Designer preview, which materializing rows runs, writes artifacts to the working directory.
    monkeypatch.chdir(tmp_path)
    rows = tmp_path / "rows.jsonl"
    materialize = ("simulate", "--locale", "en_US", "--num-rows", "1", "--probe-mix", "general_open_ended=1.0")
    assert cli.main(_cli(*materialize, "--max-turns", "1", "--materialize-inputs", "--out", str(rows))) == 0

    def check(*args, **kwargs):
        pytest.fail("the persona dataset check ran for stored rows")

    monkeypatch.setattr("usersim.cli.simulate.persona_dataset_report", check)
    assert cli.main(_cli("simulate", "--inputs", str(rows), "--dry-run", "--out", str(tmp_path / "out"))) == 0
