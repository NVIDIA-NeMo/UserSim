# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Public, model-free episode-input construction contracts."""

from __future__ import annotations

import pytest

from usersim.cli._pipeline import known_probes
from usersim.engine.external import ProbeEpisodeRuntime, materialize_episode_inputs


def test_materialization_is_deterministic_and_resolves_theme_and_tools() -> None:
    kwargs = {
        "locale": "en_US",
        "num_rows": 1,
        "probe_mix": {"tool_calling": 1.0},
        "random_seed": 73,
    }

    first = materialize_episode_inputs(**kwargs)
    second = materialize_episode_inputs(**kwargs)

    assert first == second
    assert first[0]["probe_type"] == "tool_calling"
    assert first[0]["persona"]
    assert first[0]["theme"]
    assert first[0]["tools"]
    assert first[0]["toolset_name"]
    assert first[0]["trajectory_id"]
    assert first[0]["usersim_provenance"]["code_sha"]
    assert first[0]["usersim_config"]["random_seed"] == 73
    runtime = ProbeEpisodeRuntime.from_resolved_row(first[0], models={})
    assert runtime.preamble.trajectory_id == first[0]["trajectory_id"]
    assert runtime.preamble.provenance.to_dict() == first[0]["usersim_provenance"]


@pytest.mark.parametrize("probe_type", known_probes())
def test_materialization_supports_every_registered_probe(probe_type: str) -> None:
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
