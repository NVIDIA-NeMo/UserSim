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
from usersim.engine.generator import ConversationSimulatorGenerator
from usersim.testing import ScriptedModel, _episode_runtime, _scripted_models

PERSONA = {"first_name": "Sarah", "last_name": "Johnson", "age": 35, "city": "Seattle"}


def test_the_built_in_probes_declare_the_embedding_model() -> None:
    assert declared_model_aliases(ConversationSimulatorConfig(name="conversation_messages")) == ("embedding_model",)


def test_a_declared_alias_follows_the_setting_that_names_it() -> None:
    config = ConversationSimulatorConfig(name="conversation_messages", finance_embedding_model_alias="my_embedder")

    assert declared_model_aliases(config) == ("my_embedder",)


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
    models = {**_scripted_models(ScriptedModel, []), "ext_critic_model": critic}
    runtime = _episode_runtime(
        "ext_critic", models, config=config, persona=PERSONA, data=None, profile=None, language="English"
    )

    row = await ConversationSimulatorGenerator.with_models(config, models).agenerate(runtime.preamble.to_row())

    assert row["conversation_status"] is True
    assert seen["ext_critic_model"] is critic
    assert "ext_absent_model" not in seen
