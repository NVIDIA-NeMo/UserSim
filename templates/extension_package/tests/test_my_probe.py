# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from importlib.metadata import entry_points

from my_probe.capabilities import AXIS_DESCRIPTIONS, CAPABILITIES
from my_probe.probe import MyProbe  # registers the probe; discovery is off under test
from my_probe.scorer import score

from usersim.extension import ConversationSimulatorConfig
from usersim.testing import (
    ScriptedModel,
    assert_hosted_parity,
    assert_probe_conforms,
    assert_scorer_conforms,
    run_episode,
)

PERSONA = {"first_name": "Ana", "last_name": "Souza", "age": 35, "city": "Recife"}
CONFIG = ConversationSimulatorConfig(name="conversation_messages", max_turns=2, random_seed=7, locale="pt_BR")
AREA = MyProbe.themes("pt_BR")[0]


def test_the_entry_points_name_the_probe_the_scorer_and_the_capabilities():
    assert {ep.name: ep.value for ep in entry_points(group="usersim.probes")}["my_probe"] == "my_probe.probe:MyProbe"
    assert {ep.name: ep.value for ep in entry_points(group="usersim.scorers")}["my_probe"] == "my_probe.scorer"
    capabilities = {ep.name: ep.value for ep in entry_points(group="usersim.capabilities")}
    assert capabilities["my_probe"] == "my_probe.capabilities"


def test_the_probe_conforms():
    assert_probe_conforms("my_probe")


async def test_the_scorer_conforms():
    await assert_scorer_conforms("my_probe")


async def test_the_user_explores_the_row_area_in_their_own_words():
    run = await run_episode("my_probe", config=CONFIG, persona=PERSONA, data={"theme": AREA}, language="Portuguese")

    assert run.row["conversation_status"] is True
    instructions = run.prompts[0]["messages"][0]["content"]
    assert AREA["description"] in instructions
    opening = json.loads(run.row["conversation_messages"])[0]
    assert opening == {"role": "user", "content": ScriptedModel.CONTENT["user"]}  # the user model's own words
    calls = json.loads(run.row["my_tool_calls"])
    assert [call["name"] for call in calls] == ["do_task"]  # the scripted assistant calls the tool it is offered


async def test_the_scorer_reads_the_tool_calls():
    used = await score({"my_tool_calls": json.dumps([{"name": "do_task", "arguments": {}}])}, models={})
    assert used["scores"]["my_probe.tool_used_rate"]["score"] == 1.0
    unused = await score({"my_tool_calls": "[]"}, models={})
    assert unused["scores"]["my_probe.tool_used_rate"]["score"] == 0.0


def test_each_capability_reads_a_rate_it_describes():
    read = {axis for capability in CAPABILITIES for source in capability.sources for axis in source.axes}
    assert read <= set(AXIS_DESCRIPTIONS)


async def test_a_host_reproduces_a_standalone_run():
    await assert_hosted_parity("my_probe", config=CONFIG, persona=PERSONA, data={"theme": AREA}, language="Portuguese")
