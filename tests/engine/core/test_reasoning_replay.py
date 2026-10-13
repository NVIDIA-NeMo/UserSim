# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in replay of the assistant's own earlier reasoning.

The invariants: with ``replay_assistant_reasoning`` on, the assistant's
requests carry the reasoning of its own earlier messages, on a hosted run as on
a local one; with it off, requests are unchanged; and no other model, the
simulated user above all, ever receives the assistant's reasoning.

These run whole episodes through the real model-call path, against scripted
models that record every request; every scripted reply carries reasoning
(``"assistant reasoning"`` for the assistant).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from test_simulator_pipeline_e2e import _cli_built_simulator_config, _synthetic_row_data

import usersim
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core.probes import known_probes, resolve_probe
from usersim.engine.generator import ConversationSimulatorGenerator
from usersim.testing import ScriptedModel, hosted_parity_report

_ALIASES = {
    "api_response_model": "api",
    "assistant_model": "assistant",
    "judge_model": "judge",
    "summary_model": "summary",
    "user_model": "user",
}
_TRACE = "assistant reasoning"


@pytest.fixture(autouse=True)
def _no_moves_override(monkeypatch):
    monkeypatch.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)


class _Continuing(ScriptedModel):
    """Scripted models whose summary model answers the early-stop check with "no", so turn 2 happens."""

    CONTENT = {**ScriptedModel.CONTENT, "summary": "no"}


def _config(probe_type: str, **changes: Any) -> ConversationSimulatorConfig:
    return _cli_built_simulator_config(probe_type, "en_US").model_copy(update={"max_turns": 2, **changes})


async def _requests(cfg: ConversationSimulatorConfig, probe_type: str, model=_Continuing, variant=None):
    """Run one episode and return every request, in order, as ``(role, messages)``."""
    prompts: list[dict[str, Any]] = []
    models = {alias: model(role, prompts) for alias, role in _ALIASES.items()}
    data = _synthetic_row_data(probe_type, "en_US", cfg)
    if variant is not None:
        data["probe_variant"] = variant
    await ConversationSimulatorGenerator.with_models(cfg, models).agenerate(data)
    return [(prompt["role"], prompt["messages"]) for prompt in prompts]


def _replayed(messages: list[dict[str, Any]]) -> list[str]:
    return [m.get("reasoning_content") for m in messages if m.get("role") == "assistant" and m.get("reasoning_content")]


async def test_the_assistant_receives_its_earlier_reasoning_when_replay_is_on():
    requests = await _requests(_config("general_open_ended", replay_assistant_reasoning=True), "general_open_ended")

    later = [messages for role, messages in requests if role == "assistant" and _replayed(messages)]
    assert later, "the assistant's second-turn request should carry its first-turn reasoning"
    assert all(trace == _TRACE for messages in later for trace in _replayed(messages))


async def test_requests_are_unchanged_when_replay_is_off():
    requests = await _requests(_config("general_open_ended"), "general_open_ended")

    assistant = [messages for role, messages in requests if role == "assistant"]
    assert len(assistant) >= 2, "the assistant needs a second turn for this to check anything"
    assert not any(_replayed(messages) for _, messages in requests)


@pytest.mark.parametrize("replay", [False, True])
async def test_no_other_model_ever_receives_the_assistants_reasoning(replay):
    requests = await _requests(_config("general_open_ended", replay_assistant_reasoning=replay), "general_open_ended")

    others = [messages for role, messages in requests if role != "assistant"]
    assert others and _TRACE not in json.dumps(others)


async def test_a_tool_step_carries_the_reasoning_of_the_step_before_it():
    requests = await _requests(_config("tool_calling", replay_assistant_reasoning=True), "tool_calling")

    tool_steps = [
        messages
        for role, messages in requests
        if role == "assistant" and any(m.get("role") == "tool" for m in messages)
    ]
    assert tool_steps
    assert all(_replayed(messages) for messages in tool_steps)


class _LongAnswers(_Continuing):
    """Long enough answers that the simulator summarizes them for later requests."""

    CONTENT = {**_Continuing.CONTENT, "assistant": "A long scripted answer. " * 40}


@pytest.mark.parametrize(("replay", "compressed"), [(False, True), (True, False)])
async def test_the_assistants_history_is_compressed_only_without_replay(replay, compressed):
    """A provider such as DeepSeek rejects a tool request whose earlier turns lack their reasoning."""
    cfg = _config("general_open_ended", replay_assistant_reasoning=replay, context_compression=True, max_turns=3)

    requests = await _requests(cfg, "general_open_ended", model=_LongAnswers)

    last_assistant = [messages for role, messages in requests if role == "assistant"][-1]
    assert ("[Summary of previous response" in json.dumps(last_assistant)) is compressed


def _probe_variant_pairs() -> list[tuple[str, str]]:
    pairs = []
    for probe in sorted(known_probes()):
        module = sys.modules[resolve_probe(probe).__module__]
        pairs.extend((probe, str(v)) for v in module.PROBE_VARIANTS)
    return pairs


@pytest.mark.parametrize(("probe_type", "variant"), _probe_variant_pairs())
async def test_every_probe_runs_with_replay_and_keeps_the_reasoning_from_others(probe_type, variant):
    requests = await _requests(
        _config(probe_type, replay_assistant_reasoning=True),
        probe_type,
        variant=variant if variant != "default" else None,
    )

    assert requests
    others = [messages for role, messages in requests if role != "assistant"]
    assert _TRACE not in json.dumps(others)


@pytest.mark.parametrize("probe_type", ["general_open_ended", "tool_calling"])
async def test_a_hosted_episode_replays_as_a_local_one_does(probe_type):
    cfg = _config(probe_type, replay_assistant_reasoning=True)
    data = _synthetic_row_data(probe_type, "en_US", cfg)
    persona = data.pop(cfg.persona_column)

    report = await hosted_parity_report(probe_type, config=cfg, persona=persona, data=data, model_factory=_Continuing)

    assert report.differences == []
    hosted = [p["messages"] for p in report.hosted_prompts if p["role"] == "assistant"]
    assert any(_replayed(messages) for messages in hosted)


async def test_a_probe_shortcut_replays_as_the_generator_does():
    """The public ``simulate_<probe>`` functions promise the dispatcher path's behaviour."""
    from usersim.engine.core.episode_input import construct_episode_preamble
    from usersim.engine.probes.general_open_ended.generator import simulate_general_open_ended

    cfg = _config("general_open_ended", replay_assistant_reasoning=True)
    prompts: list[dict[str, Any]] = []
    models = {alias: _Continuing(role, prompts) for alias, role in _ALIASES.items()}
    preamble = construct_episode_preamble(
        _synthetic_row_data("general_open_ended", "en_US", cfg), config=cfg, models=models
    )

    await simulate_general_open_ended(
        models=models,
        data=preamble.data,
        persona=preamble.persona,
        profile=preamble.behavioral_profile,
        locale=preamble.locale,
        language=preamble.language,
        cfg=cfg,
    )

    assert any(_replayed(p["messages"]) for p in prompts if p["role"] == "assistant")


class _RunDispatchCalls(ast.NodeVisitor):
    """Each ``run_dispatch`` call, other than ``super().run_dispatch(...)``, by its enclosing function."""

    def __init__(self) -> None:
        self.functions: list[str] = []
        self.calls: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "run_dispatch":
            receiver = func.value
            via_super = (
                isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name) and receiver.func.id == "super"
            )
            if not via_super:
                self.calls.append(self.functions[-1] if self.functions else "<module>")
        self.generic_visit(node)


def test_every_episode_in_the_package_runs_through_dispatch_episode():
    """Every ``run_dispatch`` call in the package, other than ``super().run_dispatch(...)``, is the one in ``dispatch_episode``.

    Per-episode settings such as replay are installed there and nowhere else,
    so a path that called ``run_dispatch`` itself would run without them.
    """
    package = Path(usersim.__file__).parent
    callers = []
    for path in sorted(package.rglob("*.py")):
        visitor = _RunDispatchCalls()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        callers += [(path.relative_to(package).as_posix(), function) for function in visitor.calls]

    assert callers == [("engine/core/probes.py", "dispatch_episode")]


def test_compression_follows_the_replay_in_force_not_the_config_alone():
    """A path that runs without the replay switch compresses as usual, so it is never half on."""
    from types import SimpleNamespace

    from usersim.engine.core.context import prepare_assistant_history
    from usersim.engine.core.llm import replay_reasoning_scope

    cfg = SimpleNamespace(context_compression=True, compression_window=1, replay_assistant_reasoning=True)
    state = SimpleNamespace(
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a long answer", "reasoning_content": _TRACE},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "the latest answer", "reasoning_content": _TRACE},
            {"role": "user", "content": "q3"},
        ],
        conv_summaries={1: "short", 3: "latest, summarized"},
    )
    probe = SimpleNamespace(transform_assistant_history=None)

    assert "[Summary" in json.dumps(prepare_assistant_history(state, probe, cfg))
    with replay_reasoning_scope(cfg):
        assert "[Summary" not in json.dumps(prepare_assistant_history(state, probe, cfg))
