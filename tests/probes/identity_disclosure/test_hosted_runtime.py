# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run by a host, the identity probe learns the host's assistant model, which it grades against."""

from __future__ import annotations

import json

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.episode_runtime import HostRoleModel, ProbeEpisodeRuntime


async def test_identity_disclosure_identifies_the_hosts_assistant_model() -> None:
    """Without the host's model id this probe cannot grade its row at all."""
    runtime = ProbeEpisodeRuntime(
        probe_type="identity_disclosure",
        persona={"first_name": "A", "last_name": "User", "age": 35, "region": "Oregon", "occupation": "designer"},
        locale="en_US",
        language="English",
        models={
            "assistant_model": HostRoleModel(model_name="nvidia/llama-3.1-nemotron-70b-instruct"),
            "user_model": HostRoleModel(model_name="host/user-model"),
        },
        config=ConversationSimulatorConfig(name="episode_runtime_test", random_seed=5, max_turns=1),
        data={},
        profile={"patience": 0.5, "tech_literacy": 0.5, "error_proneness": 0.5},
    )

    extras = runtime.probe.build_result_extras(runtime.state)

    assert extras["expected_identity"], "the probe must resolve an expected developer to grade against"
    assert "nvidia" in json.dumps(extras["expected_identity"]).lower()
