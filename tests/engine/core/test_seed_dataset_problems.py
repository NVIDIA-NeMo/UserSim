# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A probe's ``seed_dataset`` names exactly one table in its asset folder.

``seed_dataset_problems`` says when it does not: a name that is not a bare file
name, or a folder holding neither ``<name>.parquet`` nor ``<name>.jsonl``, or
both. ``assert_probe_conforms`` and ``usersim smoke`` report what it finds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from usersim.engine.core.probes import seed_dataset_problems


def _asset_root(tmp_path: Path, label: str, *tables: str) -> Path:
    """An asset root whose ``<label>/`` folder holds the given files."""
    (tmp_path / label).mkdir()
    for name in tables:
        (tmp_path / label / name).write_bytes(b"")
    return tmp_path


def test_a_probe_without_a_seed_table_has_no_problem(register_test_probe) -> None:
    assert seed_dataset_problems(register_test_probe("no_seed_probe")) == []


@pytest.mark.parametrize("table", ["rows.parquet", "rows.jsonl"])
def test_one_table_in_the_asset_folder_has_no_problem(register_test_probe, tmp_path, table: str) -> None:
    cls = register_test_probe(
        "seeded_probe", seed_dataset="rows", assets_root=_asset_root(tmp_path, "seeded_probe", table)
    )

    assert seed_dataset_problems(cls) == []


@pytest.mark.parametrize("name", ["rows.parquet", "tables/rows", "", 7])
def test_a_name_that_is_not_a_bare_file_name_is_refused(register_test_probe, tmp_path, name: object) -> None:
    cls = register_test_probe("seeded_probe", seed_dataset=name, assets_root=_asset_root(tmp_path, "seeded_probe"))

    (problem,) = seed_dataset_problems(cls)
    assert "must be None or a file name without its suffix" in problem
    assert repr(name) in problem


@pytest.mark.parametrize(("tables", "held"), [((), "neither"), (("rows.parquet", "rows.jsonl"), "both")])
def test_a_folder_without_exactly_one_table_is_refused(
    register_test_probe, tmp_path, tables: tuple[str, ...], held: str
) -> None:
    cls = register_test_probe(
        "seeded_probe", seed_dataset="rows", assets_root=_asset_root(tmp_path, "seeded_probe", *tables)
    )

    (problem,) = seed_dataset_problems(cls)
    assert f"holds {held} of rows.parquet, rows.jsonl; it must hold one" in problem
