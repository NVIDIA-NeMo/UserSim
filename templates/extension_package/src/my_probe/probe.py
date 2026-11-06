# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import yaml

from usersim.extension import (
    BaseProbe,
    ToolExecutionMixin,
    format_persona_for_prompt,
    get_prompt,
    language_instruction,
    load_prompt_file,
    register_probe,
    render_prompt,
)

#: A new locale is a folder under assets/my_probe/ holding a themes.yaml in
#: its language.

#: The tool the assistant under test is offered. Replace it with yours, or drop
#: it, ToolExecutionMixin and execute_tool_call if the probe offers none.
DO_TASK = {
    "type": "function",
    "function": {
        "name": "do_task",
        "description": "Replace with what the tool does.",
        "parameters": {
            "type": "object",
            "properties": {"argument": {"type": "string", "description": "Replace with what the tool takes."}},
            "required": ["argument"],
        },
    },
}

#: The instructions used when prompts.yaml has none; asset_problems reports that case.
DEFAULT_USER_PROMPT = "{persona}\n\nYou want an assistant's help with this: {area}"


def read_themes(locale_dir: Path) -> list[dict]:
    """The areas in one locale's ``themes.yaml``, each with a ``description``."""
    themes = yaml.safe_load((locale_dir / "themes.yaml").read_text(encoding="utf-8"))
    if not themes or not all("description" in theme for theme in themes):
        raise ValueError(f"{locale_dir / 'themes.yaml'} needs at least one theme, each with a description")
    return themes


@register_probe
class MyProbe(ToolExecutionMixin, BaseProbe):
    label = "my_probe"
    description = "Replace with what the probe tests, as the per-row preview and the capability report show it."
    assets_root = Path(__file__).parent / "assets"
    default_scorers = ("my_probe",)
    output_columns = ("my_tool_calls",)

    @classmethod
    def check_locale(cls, locale_dir: Path) -> None:
        read_themes(locale_dir)

    @classmethod
    def themes(cls, locale: str) -> list[dict]:
        return read_themes(cls.assets_dir() / locale)

    @classmethod
    def asset_problems(cls) -> list[str]:
        # The default runs check_locale on each locale folder the package ships.
        problems = super().asset_problems()
        if "user_system_prompt" not in load_prompt_file(cls.label):
            problems.append("prompts.yaml has no user_system_prompt")
        return problems

    def get_user_system_prompt(self) -> str:
        prompt = render_prompt(
            get_prompt(self.label, "user_system_prompt", DEFAULT_USER_PROMPT),
            self._data,
            persona=format_persona_for_prompt(self._persona),
            area=self.theme()["description"],
        )
        return f"{prompt}\n\n{language_instruction(self._language, self._locale)}"

    def get_assistant_system_prompt(self) -> str:
        return ""

    def get_tools_for_assistant(self) -> list:
        return [DO_TASK]

    async def execute_tool_call(self, name, args, tc, state, models, *, turn_idx, call_idx) -> str:
        state.metadata.setdefault("my_tool_calls", []).append({"name": name, "arguments": args})
        return json.dumps({"status": "done"})

    def build_result_extras(self, state) -> dict:
        calls = state.metadata.get("my_tool_calls", [])
        return {**super().build_result_extras(state), "my_tool_calls": json.dumps(calls, ensure_ascii=False)}
