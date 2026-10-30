# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from usersim.engine.core.probes import BaseProbe, register_probe


def _check_locale_bank(locale_dir: Path) -> None:
    if not (locale_dir / "sample.yaml").is_file():
        raise FileNotFoundError(f"no sample.yaml in {locale_dir}")


@register_probe(family="fixture", prompt_version="v1.0", variants=("default",))
class FixtureProbe(BaseProbe):
    label = "fixture_probe"
    assets_root = Path(__file__).parent / "assets"
    output_columns = ("fixture_probe_marker",)
    default_scorers = ("fixture_scorer",)
    locale_bank_check = staticmethod(_check_locale_bank)
    authoring_guide = "https://example.com/fixture-probe/locales"

    @classmethod
    def model_aliases(cls, cfg) -> tuple[str, ...]:
        return ("fixture_critic_model",)

    @classmethod
    def themes(cls, locale: str) -> list[dict]:
        return yaml.safe_load((cls.assets_dir() / locale / "themes.yaml").read_text(encoding="utf-8"))

    def get_user_system_prompt(self) -> str:
        return "You are a fixture user."

    def get_assistant_system_prompt(self) -> str:
        return "You are a fixture assistant."

    def build_result_extras(self, state) -> dict:
        return {**super().build_result_extras(state), "fixture_probe_marker": True}
