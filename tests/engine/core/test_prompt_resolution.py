# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for call-time prompt resolution and ``render_prompt``.

Scope: ``core/prompt_loader.py`` plus the per-probe accessors in
``probes/<probe>/prompts.py``. No LLM calls; prompts are written into a
``tmp_path`` asset root and read back.

The failure mode here is silent, which is why these exist. If an accessor
regressed to import-time binding, every other test in the suite would still
pass — none of them override the asset root — and the only symptom would be
``--assets-dir`` quietly failing to change a prompt in a real run.

Cache note: ``_load_prompt_file`` is ``lru_cache``d on
``(assets_root, probe)``. Each test therefore builds its own ``tmp_path``
root, so a cached load from one test cannot satisfy another.
"""

from __future__ import annotations

import json
from pathlib import Path

#: repo root: tests/core -> tests -> conversation-plugin -> repo

import pytest
import yaml

from usersim.engine.core._assets import set_runtime_assets_dir
from usersim.engine.core.prompt_loader import render_prompt
from usersim.engine.core._assets import packaged_assets_dir

#: The theme-driven probes, mapped to the name each one gives the
#: import-time constant behind ``user_agent_system_prompt()``. Spelled out
#: rather than derived: the names do not follow from the probe name
#: (``general_open_ended`` -> ``OPEN_ENDED_...``), and a test that guesses
#: would pass vacuously if a constant were renamed.
PROBE_CONSTANTS = {
    "general_open_ended": "OPEN_ENDED_USER_AGENT_SYSTEM_PROMPT",
    "general_educational": "EDUCATIONAL_USER_AGENT_SYSTEM_PROMPT",
    "tool_calling": "USER_AGENT_SYSTEM_PROMPT",
}
THEME_PROBES = tuple(PROBE_CONSTANTS)


def _assets_with_prompt(tmp_path: Path, probe: str, **keys: str) -> Path:
    """An asset root whose ``<probe>/prompts.yaml`` holds ``keys``."""
    (tmp_path / probe).mkdir(parents=True, exist_ok=True)
    (tmp_path / probe / "prompts.yaml").write_text(yaml.safe_dump(keys))
    return tmp_path


@pytest.fixture(autouse=True)
def _restore_assets_root():
    """Leave the process-wide asset root as we found it.

    Every test here repoints it; without this a later test in the same
    session would resolve prompts against a deleted tmp dir.
    """
    yield
    set_runtime_assets_dir(None)


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


class TestRenderPrompt:
    def test_fills_from_the_row(self):
        """A prompt asset can reference any column on the generated row.

        This is the point of the helper: `{uuid}` works without the probe
        being changed to pass it.
        """
        out = render_prompt("id={uuid}", {"uuid": "row-1"})
        assert out == "id=row-1"

    def test_fills_from_explicit_values(self):
        out = render_prompt("t={topic}", {}, topic="birdwatching")
        assert out == "t=birdwatching"

    def test_explicit_beats_row_on_a_name_collision(self):
        """The real case: the row carries `persona` as a dict while the probe
        passes the formatted text block under the same name."""
        out = render_prompt(
            "p={persona}",
            {"persona": {"first_name": "A"}},
            persona="Name: A B",
        )
        assert out == "p=Name: A B"

    def test_unknown_placeholder_names_itself(self):
        """A typo in a prompt asset should be diagnosable, not a bare
        ``KeyError: 'topc'`` out of ``str.format``."""
        with pytest.raises(KeyError) as e:
            render_prompt("{topc}", {"topic": "x"}, persona="p")
        msg = str(e.value)
        assert "topc" in msg
        assert "topic" in msg and "persona" in msg

    def test_row_may_be_empty(self):
        assert render_prompt("x={a}", {}, a="1") == "x=1"


# ---------------------------------------------------------------------------
# Call-time resolution
# ---------------------------------------------------------------------------


class TestAccessorsResolveAtCallTime:
    """Constants bind at import; probes import at plugin bootstrap, before a
    run's ``assets_dir`` is known. Only re-resolving per call lets
    ``--assets-dir`` / ``USERSIM_ASSETS_DIR`` govern prompts.
    """

    @pytest.mark.parametrize("probe", THEME_PROBES)
    def test_accessor_follows_the_active_asset_root(self, probe, tmp_path):
        mod = __import__(
            f"usersim.engine.probes.{probe}.prompts",
            fromlist=["prompts"],
        )
        before = mod.user_agent_system_prompt()
        set_runtime_assets_dir(
            _assets_with_prompt(
                tmp_path,
                probe,
                user_agent_system_prompt=f"OVERRIDE-{probe}",
            )
        )
        assert mod.user_agent_system_prompt() == f"OVERRIDE-{probe}"
        assert before != f"OVERRIDE-{probe}", "fixture did not actually change it"

    @pytest.mark.parametrize("probe,const", PROBE_CONSTANTS.items())
    def test_module_constant_stays_at_its_import_time_value(
        self,
        probe,
        const,
        tmp_path,
    ):
        """The constants are kept for direct importers and must NOT move."""
        mod = __import__(
            f"usersim.engine.probes.{probe}.prompts",
            fromlist=["prompts"],
        )
        pinned = getattr(mod, const)
        set_runtime_assets_dir(_assets_with_prompt(tmp_path, probe, user_agent_system_prompt="OVERRIDE"))
        assert getattr(mod, const) == pinned
        assert mod.user_agent_system_prompt() == "OVERRIDE"

    def test_missing_key_falls_back_to_the_compiled_default(self, tmp_path):
        """A prompts.yaml that omits a key must not blank the prompt."""
        from usersim.engine.probes.general_open_ended import prompts as oe

        set_runtime_assets_dir(_assets_with_prompt(tmp_path, "general_open_ended", trajectory_rubric="R"))
        assert oe.user_agent_system_prompt().strip()
        assert oe.trajectory_rubric() == "R"


# ---------------------------------------------------------------------------
# Per-row wiring
# ---------------------------------------------------------------------------


class _Cfg:
    persona_column = "persona"
    probe_type_column = "probe_type"
    theme_column = "theme"
    max_turns = 2
    max_query_attempts = 2
    context_compression = False
    compression_window = 1
    tools_column = None
    max_tools = 5
    max_steps = 10
    incremental_disclosure_ratio = 0.6
    persona_grounding_ratio = 1.0
    random_seed = None
    verbosity = 0


def _open_ended_probe(data: dict):
    """A real ``OpenEndedProbe`` for ``data``, with no LLM anywhere near it."""
    from usersim.engine.core.behavioral import compute_behavioral_profile
    from usersim.engine.probes.general_open_ended.generator import OpenEndedProbe

    persona = {"first_name": "A", "last_name": "B", "age": 30, "sex": "female"}
    return OpenEndedProbe(
        persona=persona,
        locale="en_US",
        language="English",
        models={},
        cfg=_Cfg(),
        provenance=None,
        profile=compute_behavioral_profile(persona),
        outcome_builder=None,
        data=data,
    )


class TestEveryPromptHookRenders:
    """All three hooks resolve and render at call time, not just the user one.

    They were not uniform before: the gate prompt used plain ``.format`` (so a
    rubric could not reference a row column) and two probes hardcoded
    ``return ""`` for the assistant prompt.
    """

    def test_gate_prompt_can_reference_a_row_column(self, tmp_path):
        """The asset-controlled part of the gate prompt is the RUBRIC, which
        is substituted into a compiled-in turn template. A row column used
        there now resolves, where plain ``.format`` would have raised."""
        set_runtime_assets_dir(
            _assets_with_prompt(
                tmp_path,
                "general_open_ended",
                user_judge_rubric="Reject unless it suits row {uuid}.",
            )
        )
        probe = _open_ended_probe({"uuid": "row-9", "theme": "{}"})
        out = probe.format_gate_prompt("hello?", "H")
        assert "suits row row-9" in out
        assert "hello?" in out and "H" in out

    def test_gate_prompt_typo_names_itself(self, tmp_path):
        """Same diagnosable failure as the user prompt, not a bare KeyError."""
        set_runtime_assets_dir(
            _assets_with_prompt(
                tmp_path,
                "general_open_ended",
                user_judge_rubric="{nosuchkey}",
            )
        )
        probe = _open_ended_probe({"uuid": "row-9", "theme": "{}"})
        with pytest.raises(KeyError) as e:
            probe.format_gate_prompt("q", "h")
        assert "nosuchkey" in str(e.value)

    @pytest.mark.parametrize("probe", THEME_PROBES)
    def test_assistant_prompt_is_empty_as_shipped(self, probe):
        """The pure-capability invariant: the model under test is primed with
        nothing. All three resolve the key; none ships a value."""
        mod = __import__(
            f"usersim.engine.probes.{probe}.prompts",
            fromlist=["prompts"],
        )
        assert mod.assistant_system_prompt() == ""

    def test_assistant_prompt_follows_the_asset_root(self, tmp_path):
        """It is asset-controlled, which is why the invariant is worth a test:
        a non-empty value here primes the model under test."""
        set_runtime_assets_dir(
            _assets_with_prompt(
                tmp_path,
                "general_open_ended",
                assistant_system_prompt="PRIMED",
            )
        )
        probe = _open_ended_probe({"uuid": "u", "theme": "{}"})
        assert probe.get_assistant_system_prompt() == "PRIMED"


class TestPromptIsRebuiltPerRow:
    def test_each_row_renders_its_own_values(self, tmp_path):
        """The dispatcher constructs a probe per row, so a row column
        referenced from prompts.yaml must reflect THAT row."""
        from usersim.engine.core.behavioral import compute_behavioral_profile
        from usersim.engine.probes.general_open_ended.generator import (
            OpenEndedProbe,
        )

        set_runtime_assets_dir(
            _assets_with_prompt(
                tmp_path,
                "general_open_ended",
                user_agent_system_prompt="uuid={uuid} topic={topic}",
            )
        )
        persona = {"first_name": "A", "last_name": "B", "age": 30, "sex": "female"}
        profile = compute_behavioral_profile(persona)

        rendered = []
        for uuid in ("row-0001", "row-0002"):
            probe = OpenEndedProbe(
                persona=persona,
                locale="en_US",
                language="English",
                models={},
                cfg=_Cfg(),
                provenance=None,
                profile=profile,
                outcome_builder=None,
                data={
                    "uuid": uuid,
                    "theme": json.dumps({"type": "t", "description": f"top-{uuid}"}),
                },
            )
            rendered.append(probe.get_user_system_prompt())

        assert rendered[0] == "uuid=row-0001 topic=top-row-0001"
        assert rendered[1] == "uuid=row-0002 topic=top-row-0002"

    def test_prompt_is_not_frozen_at_construction(self, tmp_path):
        """Rendering moved out of ``__init__`` into the getter, so the hook is
        what resolves -- not a string captured when the probe was built."""
        set_runtime_assets_dir(
            _assets_with_prompt(
                tmp_path,
                "general_open_ended",
                user_agent_system_prompt="FIRST",
            )
        )
        probe = _open_ended_probe({"uuid": "u", "theme": "{}"})
        assert probe.get_user_system_prompt() == "FIRST"

        second = tmp_path / "second"
        set_runtime_assets_dir(
            _assets_with_prompt(
                second,
                "general_open_ended",
                user_agent_system_prompt="SECOND",
            )
        )
        assert probe.get_user_system_prompt() == "SECOND"


# ---------------------------------------------------------------------------
# The loop-owned assistant judge prompt
# ---------------------------------------------------------------------------


class TestAssistantJudgePrompt:
    """`BaseProbe.format_assistant_judge_prompt` — a loop-owned prompt that a
    probe may override for itself.

    It resolves like every other prompt, so a probe can influence it and an
    asset root can reach it.
    """

    def test_default_comes_from_core_prompts(self, tmp_path):
        """No asset key present -> the compiled default, interpolated."""
        from usersim.engine.core.prompts import ASSISTANT_JUDGE_PROMPT

        set_runtime_assets_dir(_assets_with_prompt(tmp_path, "general_open_ended", trajectory_rubric="x"))
        probe = _open_ended_probe({"uuid": "u", "theme": "{}"})
        assert probe.format_assistant_judge_prompt("R", "H") == (
            ASSISTANT_JUDGE_PROMPT.format(assistant_response="R", conversation_history="H")
        )

    def test_probe_asset_overrides_the_default(self, tmp_path):
        """A probe sharpens the rubric for its own shape without the loop
        growing a per-probe branch."""
        set_runtime_assets_dir(
            _assets_with_prompt(
                tmp_path,
                "general_open_ended",
                assistant_judge_prompt="resp={assistant_response} hist={conversation_history}",
            )
        )
        probe = _open_ended_probe({"uuid": "u", "theme": "{}"})
        assert probe.format_assistant_judge_prompt("R", "H") == "resp=R hist=H"

    def test_override_is_scoped_to_one_probe(self, tmp_path):
        """Keyed on the probe label, so overriding it for one probe leaves
        every other probe on the default."""
        from usersim.engine.core.prompts import ASSISTANT_JUDGE_PROMPT
        from usersim.engine.probes.general_educational.generator import (
            EducationalProbe,
        )
        from usersim.engine.core.behavioral import compute_behavioral_profile

        set_runtime_assets_dir(
            _assets_with_prompt(
                tmp_path,
                "general_open_ended",
                assistant_judge_prompt="OVERRIDDEN",
            )
        )
        assert _open_ended_probe({"uuid": "u", "theme": "{}"}).format_assistant_judge_prompt("R", "H") == "OVERRIDDEN"

        persona = {"first_name": "A", "last_name": "B", "age": 30, "sex": "female"}
        educational = EducationalProbe(
            persona=persona,
            locale="en_US",
            language="English",
            models={},
            cfg=_Cfg(),
            provenance=None,
            profile=compute_behavioral_profile(persona),
            outcome_builder=None,
            data={"uuid": "u", "theme": "{}"},
        )
        assert educational.format_assistant_judge_prompt("R", "H").startswith(ASSISTANT_JUDGE_PROMPT[:40])

    def test_can_reference_a_row_column(self, tmp_path):
        set_runtime_assets_dir(
            _assets_with_prompt(
                tmp_path,
                "general_open_ended",
                assistant_judge_prompt="row={uuid} resp={assistant_response}",
            )
        )
        probe = _open_ended_probe({"uuid": "row-42", "theme": "{}"})
        assert probe.format_assistant_judge_prompt("R", "H") == "row=row-42 resp=R"

    def test_typo_names_itself(self, tmp_path):
        set_runtime_assets_dir(
            _assets_with_prompt(
                tmp_path,
                "general_open_ended",
                assistant_judge_prompt="{nosuchkey}",
            )
        )
        probe = _open_ended_probe({"uuid": "u", "theme": "{}"})
        with pytest.raises(KeyError) as e:
            probe.format_assistant_judge_prompt("R", "H")
        assert "nosuchkey" in str(e.value)

    @pytest.mark.parametrize("probe", THEME_PROBES)
    def test_shipped_asset_matches_the_compiled_default(self, probe):
        """The key ships in all three prompts.yaml so it is discoverable.

        That duplicates the text, and nothing else keeps the copies in step
        -- editing `core/prompts.py` alone would be silently overridden. This
        is the guard against that drift.
        """
        import yaml

        from usersim.engine.core.prompts import ASSISTANT_JUDGE_PROMPT

        shipped = yaml.safe_load((packaged_assets_dir() / probe / "prompts.yaml").read_text())
        assert shipped["assistant_judge_prompt"] == ASSISTANT_JUDGE_PROMPT
