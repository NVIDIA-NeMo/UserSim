# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Models a probe names in ``model_aliases`` reach its rows.

A probe that calls a model beyond the conversation's own names its alias on
the class. Every row gets each declared model the registry has, and one the
registry lacks is left out for the probe to handle.
"""

from __future__ import annotations

from typing import Any

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.probes import declared_model_aliases
from usersim.testing import ScriptedModel, run_episode

PERSONA = {"first_name": "Sarah", "last_name": "Johnson", "age": 35, "city": "Seattle"}


def test_a_run_resolves_the_models_of_the_probes_in_its_mix_only(register_test_probe) -> None:
    register_test_probe("ext_embedder", model_aliases=classmethod(lambda cls, cfg: ("ext_embedding_model",)))

    def run_of(*probes: str) -> ConversationSimulatorConfig:
        return ConversationSimulatorConfig(name="conversation_messages", probes=list(probes))

    assert declared_model_aliases(run_of("general_open_ended")) == ()
    assert declared_model_aliases(run_of("ext_embedder", "general_open_ended")) == ("ext_embedding_model",)


async def test_a_row_gets_the_declared_models_the_registry_has(register_test_probe) -> None:
    seen: dict[str, Any] = {}

    def user_prompt(self) -> str:
        seen.update(self._models)
        return "You are a home cook."

    register_test_probe(
        "ext_critic",
        model_aliases=classmethod(lambda cls, cfg: ("ext_critic_model", "ext_absent_model")),
        get_user_system_prompt=user_prompt,
        get_assistant_system_prompt=lambda self: "",
    )
    config = ConversationSimulatorConfig(name="conversation_messages", max_turns=1)
    critic = ScriptedModel("critic")

    run = await run_episode("ext_critic", config=config, persona=PERSONA, models={"ext_critic_model": critic})

    assert run.row["conversation_status"] is True
    assert seen["ext_critic_model"] is critic
    assert "ext_absent_model" not in seen
