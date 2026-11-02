# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from usersim.engine.core.probes import BaseProbe, register_probe


@register_probe(family="fixture", prompt_version="v1.0", variants=("default",))
class FixtureProbe(BaseProbe):
    label = "fixture_probe"
    assets_root = Path(__file__).parent / "assets"
    output_columns = ("fixture_probe_marker",)

    def get_user_system_prompt(self) -> str:
        return "You are a fixture user."

    def get_assistant_system_prompt(self) -> str:
        return "You are a fixture assistant."

    def build_result_extras(self, state) -> dict:
        return {**super().build_result_extras(state), "fixture_probe_marker": True}
