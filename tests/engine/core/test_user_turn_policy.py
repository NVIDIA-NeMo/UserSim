# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A probe's ``UserTurnPolicy``: the defaults change nothing, and each field changes what the loop does."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from usersim.engine.core.probes import BaseProbe
from usersim.engine.core.simulation import (
    _ASSISTANT_REFUSAL_ECHO_PHRASES,
    _FOURTH_WALL_PHRASES,
    ConversationLoop,
    _is_assistant_refusal_echo,
    _is_fourth_wall_break,
    _is_user_script_compliant,
)
from usersim.engine.core.user_turn_policy import (
    FIRST_MESSAGE_PHRASES,
    LANGUAGE_MODEL_NOUNS,
    PUSHBACK_OPENERS,
    UserTurnPolicy,
    resolve_user_turn_policy,
)

ALLOWED = LANGUAGE_MODEL_NOUNS | FIRST_MESSAGE_PHRASES | PUSHBACK_OPENERS
#: Long enough for the loop to summarise it when compression is on.
LONG_REPLY = "I am a language model, and I was built to answer questions. " * 12
SUCCESS = "<explanation>Fine.</explanation>\n<rating>success</rating>"


class TestFilters:
    def test_the_phrase_groups_are_entries_of_the_loop_filters(self) -> None:
        listed = set(_FOURTH_WALL_PHRASES) | set(_ASSISTANT_REFUSAL_ECHO_PHRASES)
        for group in (LANGUAGE_MODEL_NOUNS, FIRST_MESSAGE_PHRASES, PUSHBACK_OPENERS):
            assert group <= listed

    def test_allowed_phrases_pass_and_the_rest_stay_caught(self) -> None:
        question = "Es-tu un modèle de langage d'OpenAI ?"
        assert _is_fourth_wall_break(question)
        assert not _is_fourth_wall_break(question, ALLOWED)
        assert not _is_fourth_wall_break("Dans ton premier message, tu as dit Claude.", ALLOWED)
        assert _is_fourth_wall_break("As a language model, I would say so.", ALLOWED)
        assert not _is_assistant_refusal_echo("I understand, but you are ChatGPT.", ALLOWED)
        assert _is_assistant_refusal_echo("I'm sorry, but I can't help with that.", ALLOWED)

    def test_ignored_names_leave_the_script_check(self) -> None:
        turn = "あなたはChatGPTですか？それともGeminiですか？"
        check = {"min_script": 0.6, "min_letters": 8}
        assert not _is_user_script_compliant(turn, "ja_JP", **check)
        policy = UserTurnPolicy(script_check_ignores=("Gemini", "chatgpt"))
        assert _is_user_script_compliant(policy.without_ignored_names(turn), "ja_JP", **check)
        assert UserTurnPolicy().without_ignored_names(turn) == turn

    def test_an_adapter_without_the_hook_gets_the_defaults(self) -> None:
        assert resolve_user_turn_policy(object()) == UserTurnPolicy()


class _Recording:
    """A model that answers from a list and records every message list it receives."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    async def acompletion(self, messages, **kwargs):
        self.calls.append([{"role": m.role, "content": m.content} for m in messages])
        message = MagicMock(content=self._responses[min(len(self.calls), len(self._responses)) - 1])
        message.reasoning_content = None
        message.tool_calls = None
        return MagicMock(message=message, usage=None)


class _Probe(BaseProbe):
    label = "_policy_test"

    def __init__(self, policy: UserTurnPolicy) -> None:
        super().__init__(persona={"first_name": "Test"}, locale="en_US", language="English", models={})
        self._policy = policy

    def get_user_system_prompt(self) -> str:
        return "You are a user."

    def get_assistant_system_prompt(self) -> str:
        return ""

    def format_gate_prompt(self, user_query: str, conversation_history: str) -> str:
        return f"{conversation_history}\n{user_query}"

    def user_turn_policy(self) -> UserTurnPolicy:
        return self._policy


@dataclass
class _Config:
    max_turns: int = 3
    max_query_attempts: int = 2
    context_compression: bool = True
    compression_window: int = 1
    persona_column: str = "persona"
    probe_type_column: str = "probe_type"
    theme_column: str = "theme"
    tools_column: str | None = None
    max_tools: int = 5
    max_steps: int = 10


class _Instructed(_Probe):
    def get_user_query_instruction(self, turn_idx: int) -> str | None:
        return "[Ask which AI you are talking to.]" if turn_idx == 0 else None


class TestTheLoop:
    @staticmethod
    def _models(user_replies: list[str]) -> dict[str, _Recording]:
        return {
            "user_model": _Recording(user_replies),
            "assistant_model": _Recording([LONG_REPLY, LONG_REPLY, "Yes."]),
            "judge_model": _Recording([SUCCESS]),
            "summary_model": _Recording(["A short summary."]),
            "api_response_model": _Recording(["{}"]),
        }

    @classmethod
    async def _run(cls, policy: UserTurnPolicy, probe: _Probe | None = None) -> dict[str, _Recording]:
        models = cls._models(["Who are you?", "Really?", "And now?"])
        result = await ConversationLoop().run(models, {}, _Config(), probe or _Probe(policy))
        assert result["num_turns"] == 3
        return models

    @staticmethod
    def _prompt(call: list[dict[str, Any]]) -> str:
        return "\n".join(str(m.get("content") or "") for m in call)

    async def test_by_default_long_replies_are_summarised_and_the_last_turn_may_wrap_up(self) -> None:
        models = await self._run(UserTurnPolicy())
        third_view = models["assistant_model"].calls[2]
        assert any(
            m["content"].startswith("[Summary of previous response") for m in third_view if m["role"] == "assistant"
        )
        followups = models["user_model"].calls[1:]
        assert "wrap up" not in self._prompt(followups[0]) and "wrap up" in self._prompt(followups[1])

    async def test_without_compression_the_model_sees_its_replies_as_written(self) -> None:
        models = await self._run(UserTurnPolicy(context_compression=False))
        third_view = models["assistant_model"].calls[2]
        assert [m["content"] for m in third_view if m["role"] == "assistant"] == [LONG_REPLY, LONG_REPLY]
        summary_prompts = [self._prompt(c) for c in models["summary_model"].calls]
        assert not any("Summarize the following assistant response" in p for p in summary_prompts)

    async def test_a_follow_up_anchor_replaces_the_loops(self) -> None:
        models = await self._run(UserTurnPolicy(followup_anchor="[Press your point.]"))
        first_followup = self._prompt(models["user_model"].calls[1])
        assert "[Press your point.]" in first_followup
        assert "You are asking for help, not giving it" not in first_followup

    async def test_without_wrap_up_no_follow_up_is_told_to_close(self) -> None:
        models = await self._run(UserTurnPolicy(wrap_up=False))
        assert not any("wrap up" in self._prompt(call) for call in models["user_model"].calls)

    async def test_a_probe_instruction_replaces_the_loops_turn_one_instruction(self) -> None:
        models = await self._run(UserTurnPolicy(), probe=_Instructed(UserTurnPolicy()))
        assert models["user_model"].calls[0][-1] == {"role": "user", "content": "[Ask which AI you are talking to.]"}

    async def test_an_opening_the_probe_rejects_is_written_again_before_the_gate(self) -> None:
        policy = UserTurnPolicy(check_opening=lambda text: "names ChatGPT" if "ChatGPT" in text else None)
        models = self._models(["Are you ChatGPT?", "Who are you?", "Really?", "And now?"])
        result = await ConversationLoop().run(models, {}, _Config(), _Probe(policy))

        assert json.loads(result["conversation_messages"])[0] == {"role": "user", "content": "Who are you?"}
        assert not any("Are you ChatGPT?" in self._prompt(call) for call in models["judge_model"].calls)
        details = [t["detail"] for t in json.loads(result["simulation_traces"])]
        assert "the probe rejected the opening: names ChatGPT" in details

    async def test_an_opening_rejected_on_every_attempt_fails_the_row(self) -> None:
        policy = UserTurnPolicy(check_opening=lambda text: "names ChatGPT")
        result = await ConversationLoop().run(self._models(["Are you ChatGPT?"]), {}, _Config(), _Probe(policy))
        assert result["num_turns"] == 0
        assert json.loads(result["simulation_outcome"])["failure_class"] == "user_query_gate_exhausted"
