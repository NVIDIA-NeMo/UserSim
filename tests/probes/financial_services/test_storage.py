# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A financial_services row that failed partway is stored, with its task id and retrieved documents."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from test_cli_simulate_run_partition import _synthetic_context_failure

from usersim.engine.core.storage import read_partitioned_dataset


def test_context_failure_row_survives_cli_storage(tmp_path: Path) -> None:
    """A structured partial failure is a row, not a Data Designer omission."""
    from usersim.cli.simulate import _simulate_to_parquet
    from usersim.engine.models_file import default_models_path, load_models_config

    out = tmp_path / "trajs"
    models = load_models_config(default_models_path())

    class _FakeResult:
        def load_dataset(self):
            return _synthetic_context_failure()

    class _FakeDD:
        def create(self, builder, num_records=None, artifact_path=None):
            assert num_records == 1
            return _FakeResult()

    with patch(
        "usersim.cli._pipeline.build_simulator_config_builder",
        lambda **kwargs: (_FakeDD(), object()),
    ):
        rc = _simulate_to_parquet(
            models=models,
            models_path=default_models_path(),
            panel=None,
            locales=["en_US"],
            num_rows=1,
            out=out,
            assets_dir=None,
            max_turns=1,
            probe_mix={"financial_services": 1.0},
            random_seed=42,
            skip_existing=False,
        )

    stored = read_partitioned_dataset(out)
    assert rc == 0
    assert len(stored) == 1
    row = stored.iloc[0]
    outcome = json.loads(row["simulation_outcome"])
    assert outcome["failure_attribution"] == "assistant_model"
    assert row["finance_task_id"] == "TASK-CTX"
    assert json.loads(row["retrieved_document_ids"]) == ["d1"]
    assert "evidence" in row["conversation_messages"]
