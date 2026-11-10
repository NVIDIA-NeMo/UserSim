# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_chat_pressure's strategy override column, set on a stored row, picks the strategy the probe runs."""

from __future__ import annotations

from test_cli_simulate_inputs import (  # noqa: F401  (_environment: the module's autouse fixture)
    _environment,
    _materialize,
    _run,
    _saved,
)

from usersim.engine.core._assets import probe_assets_dir


def test_an_override_column_on_a_stored_row_reaches_the_probe(tmp_path):
    """A probe's override column, set on a stored row, picks what the probe runs.

    ``usersim simulate --panel`` reads only a panel's locales, so this is the
    way to pin an override: materialize rows, set the column, run them.
    """
    import yaml

    (stored,) = _materialize({"safety_chat_pressure": 1.0}, num_rows=1)
    first = _run([stored], tmp_path / "derived")
    (derived,) = _saved(tmp_path / "derived", first.run_id)["strategy_id"]
    bank = yaml.safe_load((probe_assets_dir("safety_chat_pressure") / "sample.yaml").read_text(encoding="utf-8"))
    chosen = next(strategy["id"] for strategy in bank["strategies"] if strategy["id"] != derived)

    stored["pressure_strategy_id"] = chosen
    second = _run([stored], tmp_path / "chosen")

    assert list(_saved(tmp_path / "chosen", second.run_id)["strategy_id"]) == [chosen]
