# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The general_open_ended probe, run end to end with scripted models and nothing patched.

``run_episode`` runs one episode as ``usersim simulate`` does. A ``ScriptedModel``
subclass scripts a Japanese conversation about the row's topic, and each test
reads the row, or the requests the models received.
"""

from __future__ import annotations

import json

from usersim.extension import ConversationSimulatorConfig, language_instruction
from usersim.testing import EpisodeRun, ScriptedModel, run_episode

PERSONA = {
    "first_name": "Haruto",
    "last_name": "Sato",
    "age": 41,
    "city": "Osaka",
    "education_level": "Bachelor",
    "occupation": "Teacher",
}
TOPIC = {"type": "travel", "description": "週末の京都旅行の計画"}


class KyotoTrip(ScriptedModel):
    """A user planning a weekend in Kyoto who keeps asking, and an assistant answering in Japanese."""

    CONTENT = {
        **ScriptedModel.CONTENT,
        "user": "週末に京都へ行くなら、どのお寺を見ればいいですか？",
        "assistant": "清水寺と伏見稲荷大社がおすすめです。",
        "summary": "no",
    }


class SatisfiedAfterOneAnswer(KyotoTrip):
    """The same user, satisfied by the first answer."""

    CONTENT = {**KyotoTrip.CONTENT, "summary": "yes"}


async def _run(model_factory: type[ScriptedModel] = KyotoTrip) -> EpisodeRun:
    return await run_episode(
        "general_open_ended",
        config=ConversationSimulatorConfig(name="conversation_messages", locale="ja_JP", max_turns=3),
        persona=PERSONA,
        data={"theme": TOPIC},
        language="Japanese",
        model_factory=model_factory,
    )


async def test_the_user_asks_about_the_rows_topic_in_the_rows_language() -> None:
    run = await _run()

    system = next(prompt for prompt in run.prompts if prompt["role"] == "user")["messages"][0]
    assert system["role"] == "system"
    assert TOPIC["description"] in system["content"]
    assert language_instruction("Japanese", "ja_JP") in system["content"]


async def test_the_assistant_under_test_gets_no_system_prompt() -> None:
    run = await _run()

    sent = [message for prompt in run.prompts if prompt["role"] == "assistant" for message in prompt["messages"]]
    assert sent
    assert all(message["role"] != "system" for message in sent)


async def test_a_user_who_keeps_asking_gets_every_turn() -> None:
    run = await _run()

    roles = [message["role"] for message in json.loads(run.row["conversation_messages"])]
    assert roles == ["user", "assistant"] * 3
    assert run.row["conversation_status"] is True


async def test_a_satisfied_user_ends_the_conversation() -> None:
    run = await _run(SatisfiedAfterOneAnswer)

    roles = [message["role"] for message in json.loads(run.row["conversation_messages"])]
    assert roles == ["user", "assistant", "user"]
    assert run.row["conversation_status"] is True
