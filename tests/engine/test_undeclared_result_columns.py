# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A probe's extra result columns reach the stored row only when they are declared.

``usersim simulate`` keeps a row's input columns plus
``ConversationSimulatorConfig.side_effect_columns`` and drops everything else. A
column a probe returns without declaring it would vanish with no error, and the
scorer that reads it would find nothing, so the generator warns about it. Per-row
data a probe cannot declare (an extension cannot edit the core config) belongs in
the probe's metadata, which is stored with the row as ``conversation_metadata``.
"""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import pytest

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.probes import _PROBE_REGISTRY, clear_registry, load_builtin_probes, register_probe
from usersim.engine.generator import ConversationSimulatorGenerator
from usersim.engine.probes.general_open_ended.generator import OpenEndedProbe

_PROBE = "_extra_column_probe"
_PERSONA = {"first_name": "Sarah", "last_name": "Johnson", "age": 35, "city": "Seattle", "state": "WA"}

#: The simulator-side modules that bind ``acall_llm`` at import, plus its own module.
_LLM_SITES = (
    "usersim.engine.core.llm",
    "usersim.engine.core.simulation",
    "usersim.engine.core.judges",
    "usersim.engine.core.context",
    "usersim.engine.core.translation",
)


async def _stub_llm(models, alias, msgs, **kwargs):
    if alias == "judge_model":
        return {"role": "assistant", "content": "<explanation>fine</explanation>\n<rating>success</rating>"}
    if alias == "summary_model":
        return {"role": "assistant", "content": "no"}
    return {"role": "assistant", "content": "Thanks, that is helpful."}


class _ExtraColumnProbe(OpenEndedProbe):
    """Returns an undeclared column, and keeps per-turn data in its metadata as the guide says."""

    label = _PROBE

    async def after_assistant_turn(self, models: dict, state: Any, assistant_response: dict, cfg: Any) -> str:
        content = await super().after_assistant_turn(models, state, assistant_response, cfg)
        state.metadata.setdefault("checks", []).append({"turn": len(state.messages), "passed": True})
        return content

    def build_result_extras(self, state: Any) -> dict:
        return {**super().build_result_extras(state), "extra_score": 0.5}


class _InPlaceInputProbe(OpenEndedProbe):
    """Changes an input column in place in its own dispatch."""

    label = "_in_place_input_probe"

    async def run_dispatch(self, *, models: dict, data: dict, cfg: Any, **kwargs: Any) -> dict:
        data["input_score"] = 0.5
        return await super().run_dispatch(models=models, data=data, cfg=cfg, **kwargs)


class _EchoingProbe(OpenEndedProbe):
    """Returns its whole row, inputs unchanged, alongside the result."""

    label = "_echoing_probe"

    async def run_dispatch(self, *, models: dict, data: dict, cfg: Any, **kwargs: Any) -> dict:
        result = await super().run_dispatch(models=models, data=data, cfg=cfg, **kwargs)
        return {**data, **result}


@pytest.fixture
def extra_column_probe():
    load_builtin_probes()
    snapshot = dict(_PROBE_REGISTRY)
    for probe in (_ExtraColumnProbe, _InPlaceInputProbe, _EchoingProbe):
        register_probe(family="general", prompt_version="v1.0", variants=("default",))(probe)
    yield _PROBE
    clear_registry()
    _PROBE_REGISTRY.update(snapshot)


def _config() -> ConversationSimulatorConfig:
    return ConversationSimulatorConfig(name="conversation_messages", locale="en_US", max_turns=1, random_seed=1)


def _row(probe_type: str, **extra: Any) -> dict[str, Any]:
    return {"persona": dict(_PERSONA), "probe_type": probe_type, "theme": "cooking at home", **extra}


class _Unused:
    async def acompletion(self, *args, **kwargs):
        raise AssertionError("acall_llm is patched")


async def _simulate(probe_type: str, rows: int = 1, **extra: Any) -> list[dict[str, Any]]:
    aliases = ("user_model", "assistant_model", "judge_model", "summary_model", "api_response_model")
    generator = ConversationSimulatorGenerator.with_models(_config(), {alias: _Unused() for alias in aliases})
    with ExitStack() as stack:
        for site in _LLM_SITES:
            stack.enter_context(patch(f"{site}.acall_llm", side_effect=_stub_llm))
        return [await generator.agenerate(_row(probe_type, **extra)) for _ in range(rows)]


@pytest.fixture
def undeclared_warnings():
    """The warnings the generator logs, captured on its own logger.

    A Data Designer run replaces the root logger's handlers, so pytest's
    ``caplog`` would miss what a pipeline run logs.
    """
    seen: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING and "side_effect_columns" in record.getMessage():
                seen.append(record.getMessage())

    handler, engine_logger = _Capture(), logging.getLogger("usersim.engine")
    engine_logger.addHandler(handler)
    yield seen
    engine_logger.removeHandler(handler)


async def test_a_column_the_probe_did_not_declare_is_reported_once(extra_column_probe, undeclared_warnings):
    await _simulate(extra_column_probe, rows=2)

    (message,) = undeclared_warnings
    assert _PROBE in message and "extra_score" in message
    # The warning points at where the data can go instead.
    assert "conversation_metadata" in message


async def test_a_value_set_on_an_undeclared_input_column_is_reported(extra_column_probe, undeclared_warnings):
    """A run keeps an input column's original value, so the probe's value is lost too."""
    await _simulate(extra_column_probe, extra_score=-1.0)

    (message,) = undeclared_warnings
    assert "extra_score" in message


async def test_a_change_made_in_place_to_an_input_column_is_reported(extra_column_probe, undeclared_warnings):
    """A run keeps the input's value, so a change made directly in the row is lost too."""
    await _simulate(_InPlaceInputProbe.label, input_score=-1.0)

    (message,) = undeclared_warnings
    assert "input_score" in message


@pytest.mark.parametrize("input_score", [-1.0, float("nan")])
async def test_returning_unchanged_inputs_reports_nothing(extra_column_probe, undeclared_warnings, input_score):
    """An input handed back as it came is kept by the run, so nothing is lost (NaN included)."""
    await _simulate(_EchoingProbe.label, input_score=input_score)

    assert undeclared_warnings == []


async def test_each_run_reports_for_itself(extra_column_probe, undeclared_warnings):
    """A standalone run does not use up the warning a later run should give."""
    await _simulate(extra_column_probe)
    await _simulate(extra_column_probe)

    assert len(undeclared_warnings) == 2


async def test_a_built_in_probe_reports_nothing(extra_column_probe, undeclared_warnings):
    await _simulate("general_open_ended")

    assert undeclared_warnings == []


def _run_pipeline(rows: list[dict[str, Any]], monkeypatch, tmp_path):
    """Run the rows through the real Data Designer pipeline, models stubbed."""
    import data_designer.config as dd
    import pandas as pd
    from data_designer.interface import DataDesigner

    from usersim.cli._models import bundled_models_path, load_models_config, to_data_designer_kwargs, to_model_configs

    monkeypatch.setenv("NVIDIA_API_KEY", "not-used")
    models = load_models_config(bundled_models_path())
    designer = DataDesigner(**to_data_designer_kwargs(models), artifact_path=tmp_path)
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[config.model_copy(update={"skip_health_check": True}) for config in to_model_configs(models)]
    )
    seed = pd.DataFrame([{**row, "persona": json.dumps(_PERSONA)} for row in rows])
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed))
    builder.add_column(_config())
    with ExitStack() as stack:
        for site in _LLM_SITES:
            stack.enter_context(patch(f"{site}.acall_llm", side_effect=_stub_llm))
        return designer.create(builder, num_records=len(rows)).load_dataset()


def test_a_run_drops_the_undeclared_column_keeps_the_metadata_and_warns_once(
    extra_column_probe, undeclared_warnings, monkeypatch, tmp_path
):
    """The premise of the warning, through the real Data Designer pipeline."""
    df = _run_pipeline([_row(extra_column_probe), _row(extra_column_probe)], monkeypatch, tmp_path)

    assert "extra_score" not in df.columns
    assert all(json.loads(metadata)["checks"] for metadata in df["conversation_metadata"])
    assert len(undeclared_warnings) == 1


def test_a_run_keeps_the_input_value_of_an_undeclared_column(extra_column_probe, monkeypatch, tmp_path):
    df = _run_pipeline([_row(extra_column_probe, extra_score=-1.0)], monkeypatch, tmp_path)

    assert df["extra_score"].tolist() == [-1.0]
