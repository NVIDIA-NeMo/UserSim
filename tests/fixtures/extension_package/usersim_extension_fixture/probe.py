# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from usersim.engine.core.probes import BaseProbe, register_probe


@register_probe(family="fixture", prompt_version="v1.0", variants=("default",))
class FixtureProbe(BaseProbe):
    label = "fixture_probe"

    def get_user_system_prompt(self) -> str:
        return "You are a fixture user."

    def get_assistant_system_prompt(self) -> str:
        return "You are a fixture assistant."
