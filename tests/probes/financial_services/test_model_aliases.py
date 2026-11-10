# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services declares the embedding model it retrieves with, under the alias its setting names."""

from __future__ import annotations

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.probes import declared_model_aliases


def test_a_run_of_the_probe_resolves_the_embedding_model() -> None:
    config = ConversationSimulatorConfig(name="conversation_messages", probes=["financial_services"])

    assert declared_model_aliases(config) == ("embedding_model",)


def test_a_declared_alias_follows_the_setting_that_names_it() -> None:
    config = ConversationSimulatorConfig(
        name="conversation_messages",
        probes=["financial_services"],
        probe_options={"financial_services": {"embedding_model_alias": "my_embedder"}},
    )

    assert declared_model_aliases(config) == ("my_embedder",)
