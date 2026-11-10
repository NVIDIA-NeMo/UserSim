# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services from the CLI: its asset commands find the shipped bank, and a dry run refuses a locale whose bank does not load."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from usersim import cli


def test_the_default_folder_is_the_probes_shipped_bank():
    from usersim.cli._bank_dir import default_bank_dir

    default = default_bank_dir("financial_services", "en_US", flag="--dir")
    assert default.is_dir(), f"the default bank folder is missing: {default}"
    assert (default / "region_meta.yaml").exists() or any(default.iterdir())


def test_validate_assets_finds_the_shipped_bank(capsys):
    rc = cli.main(["validate-assets", "--domain", "financial_services", "--locale", "en_US"])
    out = capsys.readouterr().out
    assert "bank dir not found" not in out
    assert rc == 0


def _finance_only(assets: Path, *args: str) -> list[str]:
    return ["simulate", *args, "--probe-mix", "financial_services=1", "--assets-dir", str(assets)]


def test_a_dry_run_refuses_a_locale_whose_bank_does_not_load(tmp_path, finance_assets_without_fr_tasks, caplog):
    argv = _finance_only(finance_assets_without_fr_tasks, "--dry-run", "--locale", "fr_FR", "--num-rows", "1")
    with caplog.at_level(logging.ERROR):
        assert cli.main([*argv, "--out", str(tmp_path / "out")]) == 2
    assert "financial_services can't run fr_FR: its bank does not load" in caplog.text


def test_a_panel_dry_run_checks_the_panel_locales(tmp_path, finance_assets_without_fr_tasks, caplog):
    panel = tmp_path / "panel.parquet"
    pd.DataFrame({"locale": ["en_US", "fr_FR"], "persona": [{"first_name": "A"}, {"first_name": "B"}]}).to_parquet(
        panel
    )
    argv = _finance_only(finance_assets_without_fr_tasks, "--dry-run", "--panel", str(panel))
    with caplog.at_level(logging.ERROR):
        assert cli.main([*argv, "--out", str(tmp_path / "out")]) == 2
    assert "can't run fr_FR" in caplog.text


def test_materializing_refuses_a_locale_whose_bank_does_not_load(tmp_path, finance_assets_without_fr_tasks, caplog):
    out = tmp_path / "rows.jsonl"
    argv = _finance_only(finance_assets_without_fr_tasks, "--locale", "fr_FR", "--num-rows", "1")
    with caplog.at_level(logging.ERROR):
        assert cli.main([*argv, "--materialize-inputs", "--out", str(out)]) == 2
    assert "can't run fr_FR" in caplog.text
    assert not out.exists()
