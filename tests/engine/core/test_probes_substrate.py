# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for core/probes.py — the probe-authoring substrate.

Covers:

- ``BaseProbe`` defaults are sane (every optional hook returns its
  documented default value).
- ``BankBackedProbe`` substrate handles bank load + task derivation +
  placeholder warning emission + bank-version pinning.
- The four mixins each correctly preconfigure their target hooks.
- ``LocalePromptPack`` validation fails fast on missing locales.
- ``@register_probe`` populates module constants + writes the registry,
  and ``known_probes`` / ``resolve_probe`` / ``clear_registry`` work.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from usersim.engine.core.outcomes import (
    OutcomeBuilder,
    Provenance,
    WarningKind,
)
from usersim.engine.core.probes import (
    _PROBE_REGISTRY,
    AgenticMixin,
    BankBackedProbe,
    BankLoadError,
    BankReframingMixin,
    BankVerbatimMixin,
    BaseProbe,
    CustomTurn1InstructionMixin,
    LocalePromptPack,
    LocalePromptPackError,
    ProbeAdapter,
    ProbeRegistrationError,
    ToolCallingMixin,
    ToolExecutionMixin,
    assistant_message,
    clear_registry,
    known_probes,
    register_probe,
    resolve_probe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _StubBank:
    bank_id: str = "test_bank"
    bank_version: str = "v1.0"


@dataclass
class _StubTask:
    id: str = "T-001"
    placeholder: bool = False
    verbatim_text: str = "Hello, can you help me?"
    domain: str = "general"


@dataclass
class _StubState:
    """Mimics ConversationState shape just enough for the substrate tests."""

    messages: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BaseProbe defaults
# ---------------------------------------------------------------------------


class _MinimalProbe(BaseProbe):
    label = "minimal"

    def get_user_system_prompt(self) -> str:
        return "USER_SYS"

    def get_assistant_system_prompt(self) -> str:
        return "ASST_SYS"


class TestBaseProbeDefaults:
    def _build(self) -> _MinimalProbe:
        return _MinimalProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
        )

    def test_implements_protocol(self) -> None:
        """BaseProbe satisfies the runtime-checkable ProbeAdapter Protocol."""
        probe = self._build()
        assert isinstance(probe, ProbeAdapter)

    def test_get_tools_for_assistant_default_none(self) -> None:
        assert self._build().get_tools_for_assistant() is None

    def test_format_gate_prompt_substitutes(self) -> None:
        probe = self._build()
        prompt = probe.format_gate_prompt("Hello?", "(no history)")
        assert "Hello?" in prompt
        assert "(no history)" in prompt

    def test_after_assistant_turn_appends_and_returns_content(self) -> None:
        probe = self._build()
        state = _StubState()
        out = probe.after_assistant_turn(
            models={},
            state=state,
            assistant_response={"content": "Sure, here is..."},
            cfg=None,
        )
        assert out == "Sure, here is..."
        assert state.messages == [{"role": "assistant", "content": "Sure, here is..."}]

    def test_build_result_extras_counts_user_turns_and_tools(self) -> None:
        probe = self._build()
        state = _StubState(
            messages=[
                {"role": "system"},
                {"role": "user"},
                {"role": "assistant"},
                {"role": "user"},
            ],
            metadata={"tools_called": ["fetch_weather", "send_email"]},
        )
        extras = probe.build_result_extras(state)
        assert extras["num_turns"] == 2
        assert extras["num_tool_calls"] == 2

    def test_optional_hooks_return_documented_defaults(self) -> None:
        probe = self._build()
        state = _StubState()
        assert probe.should_succeed(state) is True
        assert probe.get_verbatim_first_user_turn(state) is None
        assert probe.get_user_query_instruction(0) is None
        assert probe.get_user_query_instruction(5) is None
        assert probe.format_followup_user_instructions(1, state) == []
        assert probe.on_followup_failure(state, "gate_exhausted") == "skip"
        assert probe.should_inline_judge_user_turn(0, state) is True
        assert probe.should_inline_judge_assistant_turn(0, state) is True
        assert probe.should_continue_after_turn(state) is True
        assert probe.allow_early_stop_at_turn(0, state) is True

    def test_required_hooks_raise_when_unimplemented(self) -> None:
        class HalfBaked(BaseProbe):
            label = "half_baked"

        probe = HalfBaked(
            persona={},
            locale="en_US",
            language="English",
            models={},
        )
        with pytest.raises(NotImplementedError):
            probe.get_user_system_prompt()
        with pytest.raises(NotImplementedError):
            probe.get_assistant_system_prompt()


# ---------------------------------------------------------------------------
# BankBackedProbe
# ---------------------------------------------------------------------------


def _stub_bank_loader_locale_aware(locale: str) -> _StubBank:
    return _StubBank(bank_id=f"bank_{locale}", bank_version=f"v1.{locale}")


def _stub_bank_loader_locale_independent() -> _StubBank:
    return _StubBank(bank_id="safety_bank", bank_version="v2.0")


class _SubclassedBankProbe(BankBackedProbe):
    label = "bank_test"
    bank_loader = staticmethod(_stub_bank_loader_locale_aware)
    placeholder_warning_kind = WarningKind.USED_PLACEHOLDER_FACT
    bank_version_key = None  # locale-keyed

    def get_user_system_prompt(self) -> str:
        return ""

    def get_assistant_system_prompt(self) -> str:
        return ""

    def derive_task(self, persona, bank, *, cfg) -> _StubTask:
        return _StubTask(id=f"task_for_{bank.bank_id}", placeholder=False)


class _PlaceholderBankProbe(_SubclassedBankProbe):
    label = "bank_test_placeholder"

    def derive_task(self, persona, bank, *, cfg) -> _StubTask:
        return _StubTask(id="placeholder_task", placeholder=True)


class _LocaleIndependentBankProbe(BankBackedProbe):
    label = "bank_test_locale_independent"
    bank_loader = staticmethod(_stub_bank_loader_locale_independent)
    placeholder_warning_kind = WarningKind.USED_PLACEHOLDER_TARGET
    bank_version_key = "safety"

    def get_user_system_prompt(self) -> str:
        return ""

    def get_assistant_system_prompt(self) -> str:
        return ""

    def derive_task(self, persona, bank, *, cfg) -> _StubTask:
        return _StubTask(id="task_locale_independent")


class TestBankBackedProbe:
    def _build_outcome(self) -> tuple[OutcomeBuilder, Provenance]:
        prov = Provenance()
        builder = OutcomeBuilder(provenance=prov)
        return builder, prov

    def test_loads_bank_and_derives_task(self) -> None:
        builder, prov = self._build_outcome()
        probe = _SubclassedBankProbe(
            persona={"name": "x"},
            locale="ja_JP",
            language="Japanese",
            models={},
            outcome_builder=builder,
            provenance=prov,
        )
        assert probe._bank.bank_id == "bank_ja_JP"
        assert probe._bank.bank_version == "v1.ja_JP"
        assert probe._task.id == "task_for_bank_ja_JP"

    def test_pins_bank_version_under_locale_key_by_default(self) -> None:
        builder, prov = self._build_outcome()
        _SubclassedBankProbe(
            persona={},
            locale="ja_JP",
            language="Japanese",
            models={},
            outcome_builder=builder,
            provenance=prov,
        )
        assert prov.bank_version == {"ja_JP": "v1.ja_JP"}

    def test_pins_bank_version_under_fixed_key_when_set(self) -> None:
        builder, prov = self._build_outcome()
        _LocaleIndependentBankProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
            outcome_builder=builder,
            provenance=prov,
        )
        assert prov.bank_version == {"safety": "v2.0"}

    def test_emits_placeholder_warning_when_task_is_placeholder(self) -> None:
        builder, prov = self._build_outcome()
        _PlaceholderBankProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
            outcome_builder=builder,
            provenance=prov,
        )
        outcome = builder.finalize(
            __import__(
                "usersim.engine.core.outcomes",
                fromlist=["OutcomeStatus"],
            ).OutcomeStatus.OK
        )
        kinds = [w.kind for w in outcome.warnings]
        assert WarningKind.USED_PLACEHOLDER_FACT in kinds

    def test_no_placeholder_warning_when_task_is_clean(self) -> None:
        builder, prov = self._build_outcome()
        _SubclassedBankProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
            outcome_builder=builder,
            provenance=prov,
        )
        outcome = builder.finalize(
            __import__(
                "usersim.engine.core.outcomes",
                fromlist=["OutcomeStatus"],
            ).OutcomeStatus.OK
        )
        assert outcome.warnings == []

    def test_bank_load_failure_raises_bank_load_error(self) -> None:
        def angry_loader(_locale: str):
            raise FileNotFoundError("no bank on disk")

        class _FailingProbe(_SubclassedBankProbe):
            label = "failing_bank"
            bank_loader = staticmethod(angry_loader)

        builder, prov = self._build_outcome()
        with pytest.raises(BankLoadError) as exc_info:
            _FailingProbe(
                persona={},
                locale="en_US",
                language="English",
                models={},
                outcome_builder=builder,
                provenance=prov,
            )
        assert "no bank on disk" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------


class _BankVerbatimProbe(BankVerbatimMixin, _SubclassedBankProbe):
    """Mixin BEFORE base class so its overrides win MRO. Convention pinned in core/probes.py docstrings."""

    label = "bank_verbatim_test"
    verbatim_field = "verbatim_text"


class TestBankVerbatimMixin:
    def test_returns_verbatim_text_from_task(self) -> None:
        builder = OutcomeBuilder()
        probe = _BankVerbatimProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
            outcome_builder=builder,
            provenance=Provenance(),
        )
        assert probe.get_verbatim_first_user_turn(_StubState()) == "Hello, can you help me?"

    def test_skip_on_followup_failure(self) -> None:
        builder = OutcomeBuilder()
        probe = _BankVerbatimProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
            outcome_builder=builder,
            provenance=Provenance(),
        )
        assert probe.on_followup_failure(_StubState(), "any") == "skip"

    def test_returns_none_when_field_empty(self) -> None:
        class _EmptyTaskProbe(BankVerbatimMixin, _SubclassedBankProbe):
            label = "empty_task_test"
            verbatim_field = "verbatim_text"

            def derive_task(self, persona, bank, *, cfg) -> _StubTask:
                return _StubTask(verbatim_text="")

        builder = OutcomeBuilder()
        probe = _EmptyTaskProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
            outcome_builder=builder,
            provenance=Provenance(),
        )
        assert probe.get_verbatim_first_user_turn(_StubState()) is None


class _TurnOneInstructionProbe(CustomTurn1InstructionMixin, _SubclassedBankProbe):
    label = "custom_t1_test"
    turn1_instruction_template = "Ask about {domain} as if you have just discovered it."


class TestCustomTurn1InstructionMixin:
    def _build(self) -> _TurnOneInstructionProbe:
        builder = OutcomeBuilder()
        return _TurnOneInstructionProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
            outcome_builder=builder,
            provenance=Provenance(),
        )

    def test_returns_template_with_task_substitution_for_turn_0(self) -> None:
        probe = self._build()
        instr = probe.get_user_query_instruction(0)
        assert instr == "Ask about general as if you have just discovered it."

    def test_returns_none_for_subsequent_turns(self) -> None:
        probe = self._build()
        assert probe.get_user_query_instruction(1) is None
        assert probe.get_user_query_instruction(5) is None

    def test_skip_on_followup_failure(self) -> None:
        probe = self._build()
        assert probe.on_followup_failure(_StubState(), "any") == "skip"


@dataclass
class _ReframingTask(_StubTask):
    reframings: tuple = ()

    def reframing_at(self, turn_idx: int) -> str:
        if 0 < turn_idx <= len(self.reframings):
            return self.reframings[turn_idx - 1]
        return ""


class _ReframingProbe(BankReframingMixin, _SubclassedBankProbe):
    label = "reframing_test"

    def derive_task(self, persona, bank, *, cfg) -> _ReframingTask:
        return _ReframingTask(
            verbatim_text="Initial pressure question?",
            reframings=("First reframe.", "Second reframe.", "Third reframe."),
        )


class TestBankReframingMixin:
    def _build(self) -> _ReframingProbe:
        return _ReframingProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
            outcome_builder=OutcomeBuilder(),
            provenance=Provenance(),
        )

    def test_returns_reframing_for_each_turn(self) -> None:
        probe = self._build()
        state = _StubState()
        assert probe.format_followup_user_instructions(1, state) == ["First reframe."]
        assert probe.format_followup_user_instructions(2, state) == ["Second reframe."]
        assert probe.format_followup_user_instructions(3, state) == ["Third reframe."]

    def test_returns_empty_list_past_reframing_count(self) -> None:
        probe = self._build()
        assert probe.format_followup_user_instructions(99, _StubState()) == []

    def test_abort_on_followup_failure(self) -> None:
        probe = self._build()
        assert probe.on_followup_failure(_StubState(), "any") == "abort"

    def test_inherits_bank_verbatim_first_turn(self) -> None:
        probe = self._build()
        assert probe.get_verbatim_first_user_turn(_StubState()) == "Initial pressure question?"


class _AgenticProbe(AgenticMixin, _SubclassedBankProbe):
    label = "agentic_test"


class TestAgenticMixin:
    def _build(self) -> _AgenticProbe:
        return _AgenticProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
            outcome_builder=OutcomeBuilder(),
            provenance=Provenance(),
        )

    def test_continues_when_assistant_emitted_tool_calls(self) -> None:
        probe = self._build()
        state = _StubState(
            messages=[
                {"role": "user"},
                {"role": "assistant", "tool_calls": [{"id": "t1"}]},
            ]
        )
        assert probe.should_continue_after_turn(state) is True

    def test_stops_when_assistant_emitted_no_tool_calls(self) -> None:
        probe = self._build()
        state = _StubState(
            messages=[
                {"role": "user"},
                {"role": "assistant", "content": "I refuse."},
            ]
        )
        assert probe.should_continue_after_turn(state) is False

    def test_stops_on_empty_message_history(self) -> None:
        probe = self._build()
        assert probe.should_continue_after_turn(_StubState()) is False

    def test_skips_user_judge_for_turn_zero(self) -> None:
        probe = self._build()
        state = _StubState()
        assert probe.should_inline_judge_user_turn(0, state) is False
        assert probe.should_inline_judge_user_turn(1, state) is True


class _ToolCallingMixinProbe(ToolCallingMixin, _MinimalProbe):
    pass


class TestToolCallingMixin:
    def _build(self) -> _ToolCallingMixinProbe:
        return _ToolCallingMixinProbe(
            persona={},
            locale="en_US",
            language="English",
            models={},
        )

    def test_blocks_early_stop_when_no_tools_called_yet(self) -> None:
        probe = self._build()
        state = _StubState(metadata={"tools_called": []})
        assert probe.allow_early_stop_at_turn(0, state) is False

    def test_allows_early_stop_after_tool_use(self) -> None:
        probe = self._build()
        state = _StubState(metadata={"tools_called": ["fetch_weather"]})
        assert probe.allow_early_stop_at_turn(2, state) is True


# ---------------------------------------------------------------------------
# ToolExecutionMixin — the shared inner assistant<->tool loop
# ---------------------------------------------------------------------------


class _ToolExecProbe(ToolExecutionMixin, _MinimalProbe):
    """Records each executed call; returns a trivial tool payload."""

    def __init__(self, mode: str = "single", cap: int = 8) -> None:
        super().__init__(persona={}, locale="en_US", language="English", models={})
        self.tool_loop_mode = mode
        self.tool_max_calls_per_turn = cap
        self.executed: list[str] = []
        self.rounds: list[int] = []

    def get_tools_for_assistant(self):
        return [{"type": "function", "function": {"name": "t", "parameters": {}}}]

    def execute_tool_call(self, name, args, tc, state, models, *, turn_idx, call_idx):
        self.executed.append(name)
        return '{"ok": true}'

    def on_tool_round_complete(self, state, n_calls: int) -> None:
        self.rounds.append(n_calls)


def _tc(name: str = "t", args: str = "{}", cid: str = "c1") -> dict:
    return {"id": cid, "function": {"name": name, "arguments": args}}


class TestToolExecutionMixin:
    def test_parse_tool_call_openai_shape(self) -> None:
        name, args = ToolExecutionMixin.parse_tool_call({"function": {"name": "x", "arguments": '{"a": 1}'}})
        assert name == "x" and args == {"a": 1}

    def test_no_tool_calls_returns_content_without_llm(self) -> None:
        probe = _ToolExecProbe(mode="single")
        state = _StubState()
        with patch("usersim.engine.core.llm.call_llm") as m:
            out = probe.after_assistant_turn(
                {},
                state,
                {"content": "just talking", "tool_calls": None},
                object(),
            )
        assert out == "just talking"
        assert probe.executed == [] and probe.rounds == [0]
        m.assert_not_called()

    def test_single_mode_executes_one_round_then_synthesizes(self) -> None:
        probe = _ToolExecProbe(mode="single")
        state = _StubState()
        with patch(
            "usersim.engine.core.llm.call_llm",
            return_value={"content": "here is the answer", "tool_calls": None},
        ) as m:
            out = probe.after_assistant_turn(
                {},
                state,
                {"content": "", "tool_calls": [_tc()]},
                object(),
            )
        assert probe.executed == ["t"]  # one round of tools
        assert out == "here is the answer"  # single synthesis pass
        assert m.call_count == 1  # exactly one re-call (the synthesis)
        assert m.call_args.args[2] == state.messages[:-1]
        assert m.call_args.args[2][-1] == {
            "role": "tool",
            "content": '{"ok": true}',
            "tool_call_id": "c1",
        }
        assert state.messages[-1]["role"] == "assistant"

    def test_multi_mode_re_calls_until_no_tool_calls(self) -> None:
        probe = _ToolExecProbe(mode="multi")
        state = _StubState()
        # First re-call asks for another tool; second stops.
        responses = iter(
            [
                {"content": "", "tool_calls": [_tc(cid="c2")]},
                {"content": "done", "tool_calls": None},
            ]
        )
        with patch(
            "usersim.engine.core.llm.call_llm",
            side_effect=lambda *a, **k: next(responses),
        ):
            out = probe.after_assistant_turn(
                {},
                state,
                {"content": "", "tool_calls": [_tc()]},
                object(),
            )
        assert probe.executed == ["t", "t"]  # two rounds
        assert out == "done"

    def test_multi_mode_respects_per_turn_cap(self) -> None:
        probe = _ToolExecProbe(mode="multi", cap=1)
        state = _StubState()
        with patch(
            "usersim.engine.core.llm.call_llm",
            return_value={"content": "capped synth", "tool_calls": None},
        ):
            out = probe.after_assistant_turn(
                {},
                state,
                {"content": "", "tool_calls": [_tc(cid="c1"), _tc(cid="c2")]},
                object(),
            )
        assert probe.executed == ["t"]  # cap = 1: only the first call ran
        assert out == "capped synth"  # forced no-tools synthesis after the cap


# ---------------------------------------------------------------------------
# LocalePromptPack
# ---------------------------------------------------------------------------


class TestLocalePromptPack:
    def test_construction_succeeds_when_all_locales_present(self) -> None:
        pack = LocalePromptPack(
            label="test_pack",
            prompts={"en_US": "hi", "pt_BR": "olá", "ja_JP": "こんにちは"},
            shipped_locales=("en_US", "pt_BR", "ja_JP"),
        )
        assert pack.available_locales() == ("en_US", "ja_JP", "pt_BR")
        assert pack.shipped_locales() == ("en_US", "pt_BR", "ja_JP")

    def test_construction_fails_when_any_shipped_locale_is_missing(self) -> None:
        with pytest.raises(LocalePromptPackError) as exc_info:
            LocalePromptPack(
                label="test_pack_incomplete",
                prompts={"en_US": "hi"},
                shipped_locales=("en_US", "pt_BR", "ja_JP"),
            )
        msg = str(exc_info.value)
        assert "test_pack_incomplete" in msg
        assert "ja_JP" in msg
        assert "pt_BR" in msg

    def test_extra_prompts_beyond_shipped_locales_are_kept(self) -> None:
        pack = LocalePromptPack(
            label="extras_pack",
            prompts={"en_US": "hi", "pt_BR": "olá", "future_locale": "future"},
            shipped_locales=("en_US", "pt_BR"),
        )
        assert "future_locale" in pack.available_locales()

    def test_get_returns_template_when_no_kwargs(self) -> None:
        pack = LocalePromptPack(
            label="t",
            prompts={"en_US": "hello"},
            shipped_locales=("en_US",),
        )
        assert pack.get("en_US") == "hello"

    def test_get_substitutes_format_kwargs(self) -> None:
        pack = LocalePromptPack(
            label="t",
            prompts={"en_US": "Hello {name}, you are {age} years old."},
            shipped_locales=("en_US",),
        )
        assert pack.get("en_US", name="Alice", age=30) == "Hello Alice, you are 30 years old."

    def test_get_raises_on_unknown_locale(self) -> None:
        pack = LocalePromptPack(
            label="t",
            prompts={"en_US": "hi"},
            shipped_locales=("en_US",),
        )
        with pytest.raises(LocalePromptPackError, match="ru_RU"):
            pack.get("ru_RU")

    def test_get_raises_when_template_placeholder_missing_from_kwargs(self) -> None:
        pack = LocalePromptPack(
            label="t",
            prompts={"en_US": "Hello {name}!"},
            shipped_locales=("en_US",),
        )
        with pytest.raises(LocalePromptPackError, match="placeholder"):
            pack.get("en_US", other_kwarg="x")


# ---------------------------------------------------------------------------
# @register_probe
# ---------------------------------------------------------------------------


class TestRegisterProbe:
    def setup_method(self) -> None:
        self._snapshot = dict(_PROBE_REGISTRY)
        clear_registry()

    def teardown_method(self) -> None:
        clear_registry()
        _PROBE_REGISTRY.update(self._snapshot)

    def _make_module(self, name: str = "_probe_test_module") -> types.ModuleType:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    def test_registers_class_under_label_and_sets_module_constants(self) -> None:
        mod = self._make_module()

        @register_probe(family="general", prompt_version="v1.2", variants=("default", "alt"))
        class _Decorated(BaseProbe):
            label = "_decorated_probe"

            def get_user_system_prompt(self) -> str:
                return ""

            def get_assistant_system_prompt(self) -> str:
                return ""

        _Decorated.__module__ = mod.__name__
        # Re-apply the decorator so the module-attr step runs against
        # the module we just constructed (the original decoration ran
        # before we reassigned __module__).
        _Decorated = register_probe(
            family="general",
            prompt_version="v1.2",
            variants=("default", "alt"),
        )(_Decorated)

        assert "_decorated_probe" in known_probes()
        assert resolve_probe("_decorated_probe") is _Decorated
        assert mod.PROBE_FAMILY == "general"
        assert mod.PROMPT_VERSION == "v1.2"
        assert mod.PROBE_VARIANTS == ("default", "alt")

    def test_rejects_non_baseprobe_class(self) -> None:
        class _NotAProbe:
            label = "not_a_probe"

        with pytest.raises(ProbeRegistrationError, match="not a subclass"):
            register_probe(
                family="general",
                prompt_version="v1.0",
                variants=("default",),
            )(_NotAProbe)

    def test_rejects_class_without_label_set(self) -> None:
        class _NoLabel(BaseProbe):
            pass

        with pytest.raises(ProbeRegistrationError, match="label"):
            register_probe(
                family="general",
                prompt_version="v1.0",
                variants=("default",),
            )(_NoLabel)

    def test_resolve_probe_raises_keyerror_for_unknown(self) -> None:
        with pytest.raises(KeyError, match="bogus_probe"):
            resolve_probe("bogus_probe")

    def test_known_probes_returns_sorted_tuple(self) -> None:
        @register_probe(family="general", prompt_version="v1.0", variants=("default",))
        class _Z(BaseProbe):
            label = "z_probe"

            def get_user_system_prompt(self) -> str:
                return ""

            def get_assistant_system_prompt(self) -> str:
                return ""

        @register_probe(family="general", prompt_version="v1.0", variants=("default",))
        class _A(BaseProbe):
            label = "a_probe"

            def get_user_system_prompt(self) -> str:
                return ""

            def get_assistant_system_prompt(self) -> str:
                return ""

        assert known_probes() == ("a_probe", "z_probe")


# ---------------------------------------------------------------------------
# assistant_message() — the _UNSET sentinel's byte-identical claim
# ---------------------------------------------------------------------------


class TestAssistantMessageShape:
    """``_UNSET`` distinguishes "caller omitted tool_calls" from "caller
    passed None" so the emitted dict matches what each call site built by
    hand before the helper existed.

    The three literals below are copied from the pre-helper call sites in
    ``core/probes.py`` — the plain append, the tool-calling append, and
    the synthesis append. If the helper drifts from them, a trajectory's
    message shape changes silently, which is a schema change to the
    parquet nobody asked for.
    """

    NO_TRACE = {"content": "hi", "reasoning_content": None}

    def test_omitted_tool_calls_omits_the_key(self) -> None:
        """Plain probes built ``{"role", "content"}`` with no tool_calls."""
        assert assistant_message(self.NO_TRACE, "hi") == {
            "role": "assistant",
            "content": "hi",
        }

    def test_explicit_none_keeps_the_key(self) -> None:
        """Tool probes wrote an explicit ``"tool_calls": None``."""
        assert assistant_message(self.NO_TRACE, "hi", tool_calls=None) == {
            "role": "assistant",
            "content": "hi",
            "tool_calls": None,
        }

    def test_tool_calls_are_passed_through(self) -> None:
        calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        assert assistant_message(self.NO_TRACE, "", tool_calls=calls) == {
            "role": "assistant",
            "content": "",
            "tool_calls": calls,
        }

    def test_falsy_tool_calls_normalise_to_none(self) -> None:
        """Matches the old ``tool_calls if tool_calls else None``."""
        assert assistant_message(self.NO_TRACE, "hi", tool_calls=[]) == {
            "role": "assistant",
            "content": "hi",
            "tool_calls": None,
        }

    def test_absent_trace_adds_no_key(self) -> None:
        """A non-reasoning model must produce exactly the old dict — the
        key is absent, not present-and-empty, so consumers written
        against the old schema keep working."""
        for response in (
            {},
            {"reasoning_content": None},
            {"reasoning_content": ""},
            {"reasoning_content": "   \n "},
            None,
            "not-a-dict",
        ):
            assert assistant_message(response, "hi") == {
                "role": "assistant",
                "content": "hi",
            }

    def test_none_content_normalises_to_empty_string(self) -> None:
        assert assistant_message(self.NO_TRACE, None)["content"] == ""


class TestAssistantMessageStoreReasoning:
    """``cfg.store_reasoning`` gates capture, not serialization."""

    TRACED = {"content": "hi", "reasoning_content": "the thinking"}

    def test_stored_by_default(self) -> None:
        assert assistant_message(self.TRACED, "hi")["reasoning_content"] == ("the thinking")

    def test_disabled_omits_the_key_entirely(self) -> None:
        """Not empty-string, not null -- the same dict a non-reasoning
        model produces, so consumers cannot tell the two apart."""
        assert assistant_message(self.TRACED, "hi", store_reasoning=False) == {
            "role": "assistant",
            "content": "hi",
        }

    def test_disabled_preserves_the_rest_of_the_shape(self) -> None:
        calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        assert assistant_message(self.TRACED, "hi", tool_calls=calls, store_reasoning=False) == {
            "role": "assistant",
            "content": "hi",
            "tool_calls": calls,
        }

    def test_config_default_is_on(self) -> None:
        """The default behaviour should be to store reasoning content.
        If a probe wants to opt out, it can do so by setting store_reasoning to False."""
        from usersim.engine.config import ConversationSimulatorConfig

        assert ConversationSimulatorConfig.model_fields["store_reasoning"].default is True
