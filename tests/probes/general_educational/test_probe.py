# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The general_educational probe, run end to end with scripted models and nothing patched.

``run_episode`` runs one episode as ``usersim simulate`` does. A ``ScriptedModel``
subclass scripts a learner in Recife asking, in Brazilian Portuguese, about the
row's subject, and each test reads the row, or the requests the models received.
"""

from __future__ import annotations

import json

from usersim.extension import ConversationSimulatorConfig, language_instruction
from usersim.testing import EpisodeRun, ScriptedModel, run_episode

PERSONA = {
    "first_name": "Ana",
    "last_name": "Souza",
    "age": 16,
    "city": "Recife",
    "education_level": "Ensino médio incompleto",
    "occupation": "Estudante",
}
SUBJECT = {"type": "math", "description": "frações e porcentagens"}


class Fractions(ScriptedModel):
    """A learner who keeps asking about fractions, and a tutor answering in Portuguese."""

    CONTENT = {
        **ScriptedModel.CONTENT,
        "user": "Como eu transformo uma fração em porcentagem?",
        "assistant": "Divida o numerador pelo denominador e multiplique o resultado por cem.",
        "summary": "no",
    }


async def _run() -> EpisodeRun:
    return await run_episode(
        "general_educational",
        config=ConversationSimulatorConfig(name="conversation_messages", locale="pt_BR", max_turns=3),
        persona=PERSONA,
        data={"theme": SUBJECT},
        language="Portuguese",
        model_factory=Fractions,
    )


async def test_the_learner_asks_about_the_rows_subject_in_the_rows_language() -> None:
    run = await _run()

    system = next(prompt for prompt in run.prompts if prompt["role"] == "user")["messages"][0]
    assert system["role"] == "system"
    assert SUBJECT["description"] in system["content"]
    assert language_instruction("Portuguese", "pt_BR") in system["content"]


async def test_each_learner_turn_is_judged_before_the_tutor_sees_it() -> None:
    run = await _run()

    after_user = [run.prompts[i + 1] for i, prompt in enumerate(run.prompts) if prompt["role"] == "user"]
    assert len(after_user) == 3
    for gate in after_user:
        assert gate["role"] == "judge"
        assert Fractions.CONTENT["user"] in gate["messages"][-1]["content"]


async def test_the_tutor_under_test_gets_no_system_prompt() -> None:
    run = await _run()

    sent = [message for prompt in run.prompts if prompt["role"] == "assistant" for message in prompt["messages"]]
    assert sent
    assert all(message["role"] != "system" for message in sent)


async def test_every_turn_is_recorded() -> None:
    run = await _run()

    messages = json.loads(run.row["conversation_messages"])
    assert [message["role"] for message in messages] == ["user", "assistant"] * 3
    assert {message["content"] for message in messages if message["role"] == "assistant"} == {
        Fractions.CONTENT["assistant"]
    }
    assert run.row["conversation_status"] is True
