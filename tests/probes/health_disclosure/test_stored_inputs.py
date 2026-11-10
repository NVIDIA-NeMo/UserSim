# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A guarded health row re-run from stored inputs reports the stored id, and saves it only as ``input_id``."""

from __future__ import annotations

from typing import Any

from test_cli_simulate_inputs import (  # noqa: F401  (_environment: the module's autouse fixture)
    _environment,
    _materialize,
    _run,
    _saved,
    assert_only_input_id_holds_the_stored_ids,
)

from usersim.engine.probes.health_disclosure.mixin import GuardedMoveMixin


def test_a_guarded_row_reports_and_saves_the_stored_id(tmp_path):
    """The conversation runs under the stored id, and a guarded variant stamps
    it on every move it reports to a move hook, which therefore sees the stored
    id. Moves start on the second turn."""
    stored = _materialize({"health_therapy_disclosure": 1.0}, num_rows=2, max_turns=3)
    for row in stored:
        row["probe_variant"] = "guarded"
    reported: list[dict[str, Any]] = []
    previous = GuardedMoveMixin.register_move_hook(reported.append)
    try:
        result = _run(stored, tmp_path)
    finally:
        GuardedMoveMixin.register_move_hook(previous)

    assert {payload["trajectory_id"] for payload in reported} == {row["trajectory_id"] for row in stored}
    assert_only_input_id_holds_the_stored_ids(_saved(tmp_path, result.run_id), stored)
