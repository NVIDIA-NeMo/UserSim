# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/simulation.py shared utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from usersim.engine.core.outcomes import (
    OutcomeBuilder,
    OutcomeStatus,
    TraceKind,
)
from usersim.engine.core.simulation import (
    ConversationLoop,
    ConversationState,
    ROMANIZED_LOCALES,
    _FOURTH_WALL_PHRASES,
    _build_user_followup_messages,
    _is_assistant_refusal_echo,
    _is_fourth_wall_break,
    _is_user_script_compliant,
    _language_judge_clause,
    _script_directive,
    language_instruction,
    make_failed,
    make_result,
)


class TestLanguageInstruction:
    def test_english_returns_empty(self):
        assert language_instruction("English") == ""

    def test_non_english_returns_instruction(self):
        result = language_instruction("Portuguese")
        assert "Portuguese" in result
        assert "MUST" in result

    def test_japanese(self):
        result = language_instruction("Japanese")
        assert "Japanese" in result

    def test_locale_adds_script_directive_for_devanagari(self):
        result = language_instruction("Hindi Devanagari", "hi_Deva_IN")
        assert "Devanagari" in result
        assert "do NOT romanize" in result

    def test_locale_no_directive_for_latin_locale(self):
        # en_IN is Latin-script; no romanization clause should appear.
        result = language_instruction("Indian English", "en_IN")
        assert "romanize" not in result

    def test_english_stays_empty_even_with_locale(self):
        assert language_instruction("English", "en_US") == ""


class TestMakeResult:
    def test_produces_expected_keys(self):
        msgs = [{"role": "user", "content": "hi"}]
        meta = {"info": "test"}
        result = make_result(msgs, meta, True)
        assert "conversation_messages" in result
        assert "conversation_metadata" in result
        assert "conversation_status" in result

    def test_status_preserved(self):
        result = make_result([], {}, False)
        assert result["conversation_status"] is False

    def test_messages_json_serialized(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = make_result(msgs, {}, True)
        parsed = json.loads(result["conversation_messages"])
        assert parsed[0]["content"] == "hello"

    def test_reasoning_content_survives_the_round_trip(self):
        """Storing the trace is only worth anything if it reads back.

        ``make_result`` serialises with ``ensure_ascii=False`` and
        ``default=str``, so the trace goes through the same escaping as
        visible content -- non-Latin scripts and embedded quotes /
        newlines included, which is what a real trace from an Indic-locale
        run looks like.
        """
        trace = 'thinking: भारत / 日本 — "quoted", line\nbreak, back\\slash'
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "reply", "reasoning_content": trace},
        ]
        parsed = json.loads(make_result(msgs, {}, True)["conversation_messages"])

        assert parsed[1]["reasoning_content"] == trace
        assert parsed[1]["content"] == "reply"
        # Absent on turns that never had one, not coerced to null.
        assert "reasoning_content" not in parsed[0]


class TestMakeFailed:
    def test_produces_expected_keys(self):
        result = make_failed("test reason")
        expected_keys = [
            "user_query", "conversation_messages", "conversation_metadata",
            "conversation_status", "simulation_outcome", "simulation_traces",
            "num_turns", "num_tool_calls", "tool_subset", "disclosure_style",
            "user_interaction_style",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_status_is_false(self):
        result = make_failed("test")
        assert result["conversation_status"] is False

    def test_metadata_has_reason(self):
        result = make_failed("something broke")
        meta = json.loads(result["conversation_metadata"])
        assert meta["reason"] == "something broke"
        assert meta["skipped"] is True

    def test_tool_subset_is_none(self):
        result = make_failed("no tools")
        assert result["tool_subset"] is None

    def test_sim2real_columns_present(self):
        result = make_failed("test")
        assert result["disclosure_style"] is None
        assert result["user_interaction_style"] is None


# ---------------------------------------------------------------------------
# Fourth-wall pre-filter
# ---------------------------------------------------------------------------


class TestFourthWallEnglish:
    """English baseline — kept for regression."""

    @pytest.mark.parametrize("text", [
        "Sure! Here is the opening line for our chat.",
        "Let me start the conversation by saying hi.",
        "As an AI, I can help you with that.",
        "As a language model, I have to point out that...",
        "In this simulation, you are the user.",
    ])
    def test_meta_talk_is_caught(self, text: str) -> None:
        assert _is_fourth_wall_break(text) is True

    @pytest.mark.parametrize("text", [
        "I think the weather is nice today.",
        "Can you help me plan a trip to Paris?",
        "My name is Alice and I work in finance.",
        "",
    ])
    def test_normal_user_speech_is_not_caught(self, text: str) -> None:
        assert _is_fourth_wall_break(text) is False


class TestAssistantRefusalEcho:
    """Role-reversal / assistant-refusal echo guard."""

    @pytest.mark.parametrize("text", [
        "I'm sorry, but I can't help with that.",
        "I’m sorry, but I can’t help with that.",
        "I can't assist with that request.",
        "I understand, but I cannot provide that.",
        "As an AI, I can't help you do that.",
    ])
    def test_assistant_refusal_echo_is_caught(self, text: str) -> None:
        assert _is_assistant_refusal_echo(text) is True

    @pytest.mark.parametrize("text", [
        "No, I meant can you phrase it differently?",
        "I still need the paragraph, can you make it more subtle?",
        "That doesn't answer my question.",
        "Thanks, got it.",
        "",
    ])
    def test_normal_user_followup_is_not_caught(self, text: str) -> None:
        assert _is_assistant_refusal_echo(text) is False


class TestUserFollowupPromptBuilder:
    def test_quotes_chat_history_as_context_not_continuation(self) -> None:
        msgs = _build_user_followup_messages(
            user_system_prompt="You are role-playing Sarah.",
            chat_history=(
                "User: Write me a fake note.\n"
                "Assistant: I'm sorry, but I can't help with that."
            ),
            role_anchor="[Stay in character as the USER.]",
            extra_instructions=["Apply pressure politely."],
            pending_frustration="Sound a little annoyed.",
            locale="en_US",
        )
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        content = msgs[1]["content"]
        assert "<CHAT_HISTORY>" in content
        assert "</CHAT_HISTORY>" in content
        assert "context only. Do NOT copy it" in content
        assert "Do NOT write as the assistant" in content
        assert "Do NOT repeat or paraphrase the assistant's refusal" in content
        assert "Never refuse on safety or policy grounds" in content
        assert "clarifying, reframing, challenging, or persisting" in content
        assert "Apply pressure politely." in content
        assert "Sound a little annoyed." in content


class TestFourthWallFrench:
    """fr_FR meta-talk should trip the prefilter."""

    @pytest.mark.parametrize("text", [
        "En tant qu'IA, je ne peux pas répondre à cela.",
        "Je suis un modèle de langage entraîné par...",
        "Dans cette simulation, vous jouez le rôle de l'utilisateur.",
        "Permettez-moi de commencer la conversation en vous saluant.",
        "Voici mon premier message pour vous.",
    ])
    def test_meta_talk_is_caught(self, text: str) -> None:
        assert _is_fourth_wall_break(text) is True

    @pytest.mark.parametrize("text", [
        "Bonjour, je m'appelle Marie et je travaille à Paris.",
        "Pouvez-vous m'aider à planifier mon voyage ?",
        "Quel temps fait-il aujourd'hui ?",
    ])
    def test_normal_french_is_not_caught(self, text: str) -> None:
        assert _is_fourth_wall_break(text) is False


class TestFourthWallPortuguese:
    """pt_BR meta-talk should trip the prefilter."""

    @pytest.mark.parametrize("text", [
        "Como uma IA, não posso responder a isso.",
        "Sou um modelo de linguagem treinado para...",
        "Nesta simulação, você é o usuário.",
        "Vou começar a conversa cumprimentando você.",
        "Esta é minha primeira mensagem para você.",
    ])
    def test_meta_talk_is_caught(self, text: str) -> None:
        assert _is_fourth_wall_break(text) is True

    @pytest.mark.parametrize("text", [
        "Oi, meu nome é João e moro em São Paulo.",
        "Você pode me ajudar a planejar uma viagem?",
        "Que tempo está fazendo hoje?",
    ])
    def test_normal_portuguese_is_not_caught(self, text: str) -> None:
        assert _is_fourth_wall_break(text) is False


class TestFourthWallJapanese:
    """ja_JP meta-talk should trip the prefilter."""

    @pytest.mark.parametrize("text", [
        "AIとして、それにはお答えできません。",
        "私は言語モデルとして訓練されています。",
        "このシミュレーションでは、あなたがユーザーです。",
        "では、会話を開始しましょう。",
        "これが最初のメッセージです。",
    ])
    def test_meta_talk_is_caught(self, text: str) -> None:
        assert _is_fourth_wall_break(text) is True

    @pytest.mark.parametrize("text", [
        "こんにちは、私の名前はゆきです。",
        "東京の天気はどうですか？",
        "旅行の計画を手伝ってくれますか？",
    ])
    def test_normal_japanese_is_not_caught(self, text: str) -> None:
        assert _is_fourth_wall_break(text) is False


class TestFourthWallHindi:
    """hi_Deva_IN meta-talk should trip the prefilter."""

    @pytest.mark.parametrize("text", [
        "एआई के रूप में, मैं इसका उत्तर नहीं दे सकता।",
        "मैं एक भाषा मॉडल हूँ।",
        "इस सिमुलेशन में आप उपयोगकर्ता हैं।",
        "मेरा पहला संदेश यह है।",
        "चलिए बातचीत शुरू करते हैं।",
    ])
    def test_meta_talk_is_caught(self, text: str) -> None:
        assert _is_fourth_wall_break(text) is True

    @pytest.mark.parametrize("text", [
        "नमस्ते, मेरा नाम राज है।",
        "आज मौसम कैसा है?",
        "क्या आप मेरी यात्रा की योजना बनाने में मदद कर सकते हैं?",
    ])
    def test_normal_hindi_is_not_caught(self, text: str) -> None:
        assert _is_fourth_wall_break(text) is False


class TestFourthWallPhraseInventory:
    """Asserts every shipped non-English locale contributed phrases.

    Guards against accidental regression that would silently re-collapse
    the multi-locale prefilter to English-only (the pre-Bucket-B state).
    """

    def test_french_phrases_present(self) -> None:
        assert any("modèle" in p for p in _FOURTH_WALL_PHRASES)
        assert any("simulation" in p and "cette" in p for p in _FOURTH_WALL_PHRASES)

    def test_portuguese_phrases_present(self) -> None:
        assert any("simulação" in p for p in _FOURTH_WALL_PHRASES)

    def test_japanese_phrases_present(self) -> None:
        assert any("言語モデル" in p for p in _FOURTH_WALL_PHRASES)
        assert any("シミュレーション" in p for p in _FOURTH_WALL_PHRASES)

    def test_hindi_phrases_present(self) -> None:
        assert any("भाषा मॉडल" in p for p in _FOURTH_WALL_PHRASES)
        assert any("सिमुलेशन" in p for p in _FOURTH_WALL_PHRASES)


# ---------------------------------------------------------------------------
# Verbatim-turn-1 injection helper
# ---------------------------------------------------------------------------


@dataclass
class _StubAdapter:
    """Minimal ProbeAdapter for unit-testing _inject_verbatim_first_turn."""

    label: str = "stub_probe"
    user_system: str = "You are testing the system."
    asst_system: str = "You are an assistant under test."

    def get_user_system_prompt(self) -> str:
        return self.user_system

    def get_assistant_system_prompt(self) -> str:
        return self.asst_system

    def get_tools_for_assistant(self):
        return None

    def format_gate_prompt(self, user_query: str, conversation_history: str) -> str:
        return ""

    def after_assistant_turn(self, models, state, response, cfg):
        return ""

    def build_result_extras(self, state) -> dict:
        return {}


class TestInjectVerbatimFirstTurn:
    def _make_state(self) -> ConversationState:
        return ConversationState(outcome=OutcomeBuilder())

    def test_messages_seeded_with_assistant_system_then_user_verbatim(self) -> None:
        loop = ConversationLoop()
        adapter = _StubAdapter(asst_system="ASST_SYS")
        state = self._make_state()
        loop._inject_verbatim_first_turn(adapter, state, "verbatim Q1")
        assert state.messages == [
            {"role": "system", "content": "ASST_SYS"},
            {"role": "user", "content": "verbatim Q1"},
        ]

    def test_skips_system_message_when_assistant_system_empty(self) -> None:
        loop = ConversationLoop()
        adapter = _StubAdapter(asst_system="")
        state = self._make_state()
        loop._inject_verbatim_first_turn(adapter, state, "verbatim Q1")
        assert state.messages == [{"role": "user", "content": "verbatim Q1"}]

    def test_user_history_mirrors_what_user_agent_would_have_said(self) -> None:
        loop = ConversationLoop()
        adapter = _StubAdapter(user_system="USER_SYS")
        state = self._make_state()
        loop._inject_verbatim_first_turn(adapter, state, "verbatim Q1")
        assert len(state.user_history) == 3
        assert state.user_history[0] == {"role": "system", "content": "USER_SYS"}
        assert state.user_history[1]["role"] == "user"
        assert "Generate" in state.user_history[1]["content"] or len(state.user_history[1]["content"]) > 0
        assert state.user_history[2] == {
            "role": "assistant", "content": "verbatim Q1"
        }

    def test_user_history_uses_grounded_instruction_when_flagged(self) -> None:
        loop = ConversationLoop()
        adapter = _StubAdapter()
        state = self._make_state()
        loop._inject_verbatim_first_turn(
            adapter, state, "verbatim Q1", use_grounded=True,
        )
        from usersim.engine.core.behavioral import (
            USER_QUERY_INSTRUCTION,
            USER_QUERY_INSTRUCTION_GROUNDED,
        )
        assert state.user_history[1]["content"] == USER_QUERY_INSTRUCTION_GROUNDED
        assert state.user_history[1]["content"] != USER_QUERY_INSTRUCTION

    def test_metadata_seeded_with_bypassed_judge_rating(self) -> None:
        loop = ConversationLoop()
        state = self._make_state()
        loop._inject_verbatim_first_turn(_StubAdapter(), state, "verbatim Q1")
        ratings = state.metadata["user_judge_ratings"]
        assert len(ratings) == 1
        assert ratings[0]["turn_idx"] == 0
        assert ratings[0]["rating"] == "bypassed"
        assert ratings[0]["success"] is True
        assert "frustration_events" in state.metadata

    def test_emits_user_query_gate_trace_with_bypassed_rating(self) -> None:
        loop = ConversationLoop()
        state = self._make_state()
        loop._inject_verbatim_first_turn(_StubAdapter(), state, "verbatim Q1")
        traces = state.outcome.traces()
        gate_traces = [t for t in traces if t.kind == TraceKind.USER_QUERY_GATE]
        assert len(gate_traces) == 1
        assert gate_traces[0].rating == "bypassed"
        assert gate_traces[0].turn_idx == 0
        assert gate_traces[0].call_idx == 0

    def test_does_not_increment_user_query_attempts(self) -> None:
        """No attempt was made — the budget is preserved for legitimate retries."""
        loop = ConversationLoop()
        state = self._make_state()
        loop._inject_verbatim_first_turn(_StubAdapter(), state, "verbatim Q1")
        out = state.outcome.finalize(OutcomeStatus.OK)
        assert out.n_user_query_attempts == 0


# ---------------------------------------------------------------------------
# Role-violation telemetry alive-ness
# ---------------------------------------------------------------------------


class TestRoleViolationCounterPromotesStatus:
    """The OutcomeBuilder.finalize() promote-on-role-violation path is fed
    from the loop's gate-failure handler. The unit-level coverage of
    inc_user_role_violations lives in test_outcomes.py; we pin the
    integration here: once the counter is non-zero, the otherwise-
    OK trajectory must finalize as COMPLETED_WITH_WARNINGS.
    """

    def test_zero_violations_stays_ok(self) -> None:
        b = OutcomeBuilder()
        out = b.finalize(OutcomeStatus.OK)
        assert out.status == OutcomeStatus.OK

    def test_one_violation_promotes_to_warnings(self) -> None:
        b = OutcomeBuilder()
        b.inc_user_role_violations()
        out = b.finalize(OutcomeStatus.OK)
        assert out.status == OutcomeStatus.COMPLETED_WITH_WARNINGS
        assert out.n_user_role_violations == 1

    def test_violation_does_not_promote_failed_status(self) -> None:
        """Only OK gets promoted; an outright FAILED outcome stays FAILED."""
        b = OutcomeBuilder()
        b.inc_user_role_violations()
        out = b.finalize(OutcomeStatus.FAILED)
        assert out.status == OutcomeStatus.FAILED


# ---------------------------------------------------------------------------
# User-model script-compliance gate (Bucket 1)
# ---------------------------------------------------------------------------

import types  # noqa: E402

from usersim.engine.core.outcomes import FailureClass  # noqa: E402


# A natural Devanagari user turn (well above the loanword threshold).
_DEVA_TURN = "मुझे अपनी पेंशन के नियमों के बारे में जानकारी चाहिए"
# The same request fully romanized into Latin — the failure mode the gate
# exists to catch.
_ROMANIZED_TURN = "mujhe apni pension ke niyamon ke baare mein jaankari chahiye"


class TestScriptDirective:
    def test_devanagari_locale_gets_directive(self) -> None:
        d = _script_directive("hi_Deva_IN")
        assert "Devanagari" in d and "romanize" in d

    def test_latin_locale_gets_nothing(self) -> None:
        assert _script_directive("en_US") == ""
        assert _script_directive("pt_BR") == ""

    def test_none_locale_gets_nothing(self) -> None:
        assert _script_directive(None) == ""

    def test_romanized_locale_gets_romanization_directive(self) -> None:
        d = _script_directive("hi_Latn_IN")
        # Steers the user-model to Latin/romanized Hindi, NOT Devanagari/English.
        assert "romanized" in d.lower()
        assert "Latin" in d or "Roman" in d
        assert "Devanagari" in d


class TestRomanizedLanguageJudgeClause:
    """The judge-folded language clause is the enforcement path for
    romanized locales (lingua/script checks can't verify them)."""

    def test_romanized_locales_are_hi_latn_plus_india_variants(self) -> None:
        # hi_Latn_IN ships first-class; the India language-variant layer adds
        # a romanized (_Latn_IN) twin per language. All are enforced via the
        # judge-folded language clause (lingua/script can't verify them).
        from usersim.engine.core.locale import INDIA_VARIANT_LOCALES
        expected = {"hi_Latn_IN"} | {
            loc for loc, v in INDIA_VARIANT_LOCALES.items() if v.romanized
        }
        assert ROMANIZED_LOCALES == frozenset(expected)
        # Every romanized locale has a non-empty judge clause.
        for loc in ROMANIZED_LOCALES:
            assert _language_judge_clause(loc)

    def test_clause_present_for_hi_latn_in(self) -> None:
        clause = _language_judge_clause("hi_Latn_IN")
        assert clause  # non-empty
        # Must instruct the judge to fail wrong-language turns with a token
        # the loop can parse, and reference the existing rating field.
        assert "[LANGUAGE]" in clause
        assert "failure" in clause.lower()

    def test_no_clause_for_english_or_non_latin_or_other_latin(self) -> None:
        # English, non-Latin (handled by deterministic script check), and
        # fr/pt (real lingua coverage) are deliberately left untouched.
        for loc in ("en_US", "en_IN", "hi_Deva_IN", "ja_JP", "ko_KR", "fr_FR", "pt_BR"):
            assert _language_judge_clause(loc) == ""
            assert loc not in ROMANIZED_LOCALES


class TestIsUserScriptCompliant:
    def test_devanagari_turn_passes_for_devanagari_locale(self) -> None:
        assert _is_user_script_compliant(
            _DEVA_TURN, "hi_Deva_IN", min_script=0.6, min_letters=8
        ) is True

    def test_romanized_turn_fails_for_devanagari_locale(self) -> None:
        assert _is_user_script_compliant(
            _ROMANIZED_TURN, "hi_Deva_IN", min_script=0.6, min_letters=8
        ) is False

    def test_latin_loanwords_tolerated_inside_devanagari(self) -> None:
        # A mostly-Devanagari turn with an English loanword stays above 0.6.
        mixed = "मेरा computer ठीक से काम नहीं कर रहा है"
        assert _is_user_script_compliant(
            mixed, "hi_Deva_IN", min_script=0.6, min_letters=8
        ) is True

    def test_english_passes_for_latin_locale(self) -> None:
        assert _is_user_script_compliant(
            "Hello, can you help me with my taxes?",
            "en_US", min_script=0.6, min_letters=8,
        ) is True

    def test_japanese_rejects_latin_only_turn(self) -> None:
        assert _is_user_script_compliant(
            "kon'nichiwa onegaishimasu", "ja_JP",
            min_script=0.6, min_letters=8,
        ) is False

    def test_short_turn_is_skipped(self) -> None:
        # Below min_letters: too little signal, so we don't enforce.
        assert _is_user_script_compliant(
            "ok", "hi_Deva_IN", min_script=0.6, min_letters=8
        ) is True

    def test_unknown_locale_is_noop(self) -> None:
        assert _is_user_script_compliant(
            _ROMANIZED_TURN, "xx_YY", min_script=0.6, min_letters=8
        ) is True


def _gate_cfg(**overrides):
    base = dict(
        max_query_attempts=3,
        enforce_user_language=True,
        user_language_min_script_compliance=0.6,
        user_language_min_letters=8,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class TestTurn1LanguageGate:
    """Integration over ``_generate_and_gate_first_turn`` with stubbed LLMs."""

    def _run(self, monkeypatch, *, user_text, cfg, judge_ok=True, locale="hi_Deva_IN"):
        import usersim.engine.core.simulation as sim

        monkeypatch.setattr(
            sim, "call_llm",
            lambda models, alias, msgs, **kw: {"content": user_text},
        )
        # The judge is only reached if the language + fourth-wall prefilters
        # pass; stub it so the positive path can succeed deterministically.
        monkeypatch.setattr(
            sim, "run_inline_judge",
            lambda models, alias, prompt: ("looks good", "success", judge_ok),
        )
        loop = ConversationLoop()
        state = ConversationState(outcome=OutcomeBuilder())
        return loop._generate_and_gate_first_turn(
            models={},
            state=state,
            probe=_StubAdapter(),
            cfg=cfg,
            query_instruction="Ask your question.",
            locale=locale,
            _t_loop_start=0.0,
        ), state

    def test_romanized_turn_exhausts_with_language_fail_kind(self, monkeypatch):
        (uq, ok, rating, expl, fail_kind), state = self._run(
            monkeypatch, user_text=_ROMANIZED_TURN, cfg=_gate_cfg(),
        )
        assert ok is False
        assert fail_kind == "language"
        out = state.outcome.finalize(OutcomeStatus.FAILED)
        assert out.n_user_language_violations == 3

    def test_language_prefilter_traces_emitted(self, monkeypatch):
        (_, ok, *_), state = self._run(
            monkeypatch, user_text=_ROMANIZED_TURN, cfg=_gate_cfg(),
        )
        lang_traces = [
            t for t in state.outcome.traces()
            if t.kind == TraceKind.LANGUAGE_PREFILTER
        ]
        assert len(lang_traces) == 3
        assert ok is False

    def test_devanagari_turn_passes_gate(self, monkeypatch):
        (uq, ok, rating, expl, fail_kind), state = self._run(
            monkeypatch, user_text=_DEVA_TURN, cfg=_gate_cfg(),
        )
        assert ok is True
        assert uq == _DEVA_TURN
        out = state.outcome.finalize(OutcomeStatus.OK)
        assert out.n_user_language_violations == 0

    def test_enforcement_disabled_lets_romanized_through(self, monkeypatch):
        (_, ok, *_), state = self._run(
            monkeypatch, user_text=_ROMANIZED_TURN,
            cfg=_gate_cfg(enforce_user_language=False),
        )
        # With enforcement off, the romanized turn reaches (and passes)
        # the stubbed judge instead of being rejected on script.
        assert ok is True
        out = state.outcome.finalize(OutcomeStatus.OK)
        assert out.n_user_language_violations == 0

    def test_latin_locale_is_unaffected(self, monkeypatch):
        (_, ok, *_), state = self._run(
            monkeypatch, user_text="Hello, I need help with my visa.",
            cfg=_gate_cfg(), locale="en_US",
        )
        assert ok is True
        out = state.outcome.finalize(OutcomeStatus.OK)
        assert out.n_user_language_violations == 0


class TestUserLanguageGateExhaustedFailureClass:
    """The dedicated failure class is wired and distinct from the gate one."""

    def test_failure_class_exists(self) -> None:
        assert FailureClass.USER_LANGUAGE_GATE_EXHAUSTED.value == (
            "user_language_gate_exhausted"
        )

    def test_language_violation_promotes_ok_to_warnings(self) -> None:
        b = OutcomeBuilder()
        b.inc_user_language_violations()
        out = b.finalize(OutcomeStatus.OK)
        assert out.status == OutcomeStatus.COMPLETED_WITH_WARNINGS
        assert out.n_user_language_violations == 1
