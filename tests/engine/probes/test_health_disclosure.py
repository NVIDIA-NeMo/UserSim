# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the health_disclosure probe family.

Focus on the deterministic, LLM-free surface (matching ``test_tool_calling.py``):
the shared Guard pacing gates, the ``commit_move`` tool schema, move extraction,
and the per-client config (topics / risk / archetype / prompts / provisional axes).

The end-to-end ``propose_and_guard`` loop calls ``call_llm`` and is exercised by
the smoke run / integration tests, not here.
"""

from __future__ import annotations

import json

import pytest

from usersim.engine.probes.health_disclosure import (
    clients as C,
)
from usersim.engine.probes.health_disclosure import (
    decision_support,
    general,
    therapy,
    triage,
)
from usersim.engine.probes.health_disclosure import (
    move_runtime as mr,
)
from usersim.engine.probes.health_disclosure.mixin import GuardedMoveMixin
from usersim.engine.probes.health_disclosure.moves import (
    AFFECTS,
    COGNITIONS,
    MOVES,
    Guard,
    Move,
    PatientEnv,
)
from usersim.engine.probes.health_disclosure.prompts import (
    make_gate_prompt,
)

ALL_LABELS = (
    "health_therapy_disclosure",
    "health_triage_disclosure",
    "health_decision_support_disclosure",
    "health_general_disclosure",
)
_PLACEHOLDERS = (
    "{persona}",
    "{language_instruction}",
    "{behavioral_instructions}",
    "{disclosure_instructions}",
    "{interaction_style_instructions}",
)


# ---------------------------------------------------------------------------
# Client config
# ---------------------------------------------------------------------------
class TestClientConfig:
    @pytest.mark.parametrize("label", ALL_LABELS)
    def test_config_has_required_keys(self, label):
        cfg = C.CLIENTS[label]
        for key in (
            "label",
            "family",
            "prompt_version",
            "profiles_env",
            "risk_noun",
            "topics",
            "user_prompt",
            "standin_prompt",
            "proposal",
            "gate_prompt",
        ):
            assert key in cfg, f"{label} missing config key {key!r}"
        assert cfg["label"] == label

    @pytest.mark.parametrize("label", ALL_LABELS)
    def test_families_are_distinct(self, label):
        # Distinct family per client so each gets its own judge axes.
        fams = [C.CLIENTS[x]["family"] for x in ALL_LABELS]
        assert len(set(fams)) == len(fams)

    @pytest.mark.parametrize("label", ALL_LABELS)
    def test_user_prompt_has_all_placeholders(self, label):
        # user_prompt is a LocalePromptPack (English scaffold per shipped locale).
        prompt = C.CLIENTS[label]["user_prompt"].get("en_US")
        for ph in _PLACEHOLDERS:
            assert ph in prompt, f"{label} user_prompt missing {ph}"

    @pytest.mark.parametrize("label", ALL_LABELS)
    def test_user_prompt_formats_cleanly(self, label):
        # No stray/unescaped braces — get(...) .format must succeed with real keys.
        out = C.CLIENTS[label]["user_prompt"].get(
            "en_US",
            persona="P",
            language_instruction="L",
            behavioral_instructions="B",
            disclosure_instructions="D",
            interaction_style_instructions="I",
        )
        assert "P" in out

    @pytest.mark.parametrize("label", ALL_LABELS)
    def test_user_prompt_pack_ships_all_locales(self, label):
        # all-locales-or-fail: the pack must cover every shipped locale.
        from usersim.engine.core.locale import SHIPPED_LOCALES

        pack = C.CLIENTS[label]["user_prompt"]
        assert set(pack.available_locales()) >= set(SHIPPED_LOCALES)

    @pytest.mark.parametrize("label", ALL_LABELS)
    def test_gate_prompt_formats_cleanly(self, label):
        out = (
            C.CLIENTS[label]["gate_prompt"]
            .get("en_US")
            .format(conversation_history="HIST", user_turn_to_evaluate="TURN")
        )
        assert "HIST" in out and "TURN" in out

    @pytest.mark.parametrize("label", ALL_LABELS)
    def test_standin_prompt_non_empty(self, label):
        assert C.CLIENTS[label]["standin_prompt"].strip()

    @pytest.mark.parametrize("label", ALL_LABELS)
    def test_five_provisional_axes(self, label):
        # Axes are declared inline in the evaluator (like every other probe),
        # keyed by the client's family — not imported from this package.
        from usersim.engine.evaluator.axes import PROBE_SCORES

        family = C.CLIENTS[label]["family"]
        axes = PROBE_SCORES[family]
        assert len(axes) == 5
        for ax in axes:
            assert ax.name and ax.description
            assert set(ax.options) == {1, 2, 3, 4, 5}

    def test_topics_for_therapy(self):
        cfg = C.CLIENTS["health_therapy_disclosure"]
        profile = {"formulation": {"ccd": {"core_belief": "I'm a burden"}}, "narrative_memory": ["hard memory"]}
        assert {"core_belief", "trauma"} <= set(C.topics_for(profile, cfg))
        # Profile present but without the gated markers → gated topics dropped.
        bare = C.topics_for({"formulation": {}}, cfg)
        assert "core_belief" not in bare and "trauma" not in bare
        # Absent profile entirely → all topics (can't tell what's carried).
        assert "trauma" in C.topics_for({}, cfg)

    def test_topics_for_triage_and_cds(self):
        tri = C.CLIENTS["health_triage_disclosure"]
        assert "red_flags" in C.topics_for({"topics": {"red_flags": True}}, tri)
        cds = C.CLIENTS["health_decision_support_disclosure"]
        assert "key_finding" in C.topics_for({"topics": {"key_finding": True}}, cds)

    def test_topics_for_generic_health(self):
        gh = C.CLIENTS["health_general_disclosure"]
        # Carries a red flag + a med-adherence gap → both gated topics present.
        prof = {"topics": {"red_flag": "chest pain", "medication_adherence": "stopped statin"}}
        assert {"red_flag", "medication_adherence"} <= set(C.topics_for(prof, gh))
        # Present profile without the gated markers → gated topics dropped.
        bare = C.topics_for({"topics": {}}, gh)
        assert "red_flag" not in bare and "medication_adherence" not in bare

    def test_has_risk_and_archetype(self):
        assert C.has_risk({"risk": {"level": "high"}}) is True
        assert C.has_risk({"risk": {"level": "none"}}) is False
        assert C.archetype({"risk": {"level": "moderate"}}, "incremental") == "covert_risk"
        assert C.archetype({}, "open and upfront") == "open"
        assert C.archetype({}, "gradual") == "guarded"


# ---------------------------------------------------------------------------
# Move tool schema + extraction (shared)
# ---------------------------------------------------------------------------
class TestMoveTool:
    def test_commit_move_schema_shape(self):
        tool = mr._move_tool(["symptoms", "core_belief"])
        fn = tool["function"]
        assert fn["name"] == "commit_move"
        props = fn["parameters"]["properties"]
        assert props["move"]["enum"] == list(MOVES)
        assert props["affect"]["enum"] == list(AFFECTS)
        assert props["cognitive_content"]["enum"] == list(COGNITIONS)
        for field in ("move", "affect", "cognitive_content", "reasoning"):
            assert field in fn["parameters"]["required"]

    def test_topic_enum_includes_carried_topics_plus_risk(self):
        enum = mr._move_tool(["symptoms"])["function"]["parameters"]["properties"]["topic"]["enum"]
        assert "symptoms" in enum and "risk" in enum and "" in enum


class TestExtractMove:
    def test_parses_native_tool_call(self):
        args = {
            "move": "partial_disclose",
            "topic": "symptoms",
            "intensity": 0.5,
            "affect": "anxious",
            "cognitive_content": "rumination",
            "reasoning": "early",
        }
        result = {"tool_calls": [{"id": "c1", "function": {"name": "commit_move", "arguments": json.dumps(args)}}]}
        move, reasoning, raw = mr._extract_move(result, ["symptoms"])
        assert move.move == "partial_disclose" and move.topic == "symptoms"
        assert move.affect == "anxious" and move.cognitive_content == "rumination"
        assert raw is not None and "early" in reasoning

    def test_falls_back_to_content_json(self):
        move, _, raw = mr._extract_move({"content": '{"move": "deflect"}'}, ["symptoms"])
        assert move.move == "deflect" and raw is None

    def test_invalid_move_returns_none(self):
        result = {"tool_calls": [{"function": {"name": "commit_move", "arguments": json.dumps({"move": "nope"})}}]}
        assert mr._extract_move(result, ["symptoms"])[0] is None

    def test_unknown_topic_is_snapped(self):
        args = {"move": "disclose", "topic": "sympt", "reasoning": "x"}
        result = {"tool_calls": [{"function": {"name": "commit_move", "arguments": json.dumps(args)}}]}
        assert mr._extract_move(result, ["symptoms"])[0].topic == "symptoms"


# ---------------------------------------------------------------------------
# Provider-variance tolerance — "OpenAI-compatible" is implemented unevenly, so
# the move parser must recover the committed move whatever shape it arrives in
# (the contract we bring to the UserSim harness discussion). _extract_move now
# delegates to the shared core.tool_calls recovery; these pin the integration.
# ---------------------------------------------------------------------------
class TestExtractMoveRobustness:
    def test_arguments_as_parsed_dict(self):
        res = {
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "commit_move",
                        "arguments": {
                            "move": "withhold",
                            "affect": "flat",
                            "cognitive_content": "none",
                            "reasoning": "r",
                        },
                    },
                }
            ]
        }
        move, _, raw = mr._extract_move(res, ["symptoms"])
        assert move.move == "withhold" and raw is not None

    def test_content_with_code_fence(self):
        res = {"content": '```json\n{"move": "deflect", "topic": "symptoms"}\n```'}
        move, _, _ = mr._extract_move(res, ["symptoms"])
        assert move.move == "deflect" and move.topic == "symptoms"

    def test_content_with_prose_around_json(self):
        res = {"content": 'Sure — my move: {"move": "minimize"} hope that helps'}
        move, _, _ = mr._extract_move(res, ["symptoms"])
        assert move.move == "minimize"

    def test_toolcall_stub_empty_args_recovers_from_content(self):
        res = {
            "tool_calls": [{"id": "c1", "function": {"name": "commit_move", "arguments": ""}}],
            "content": '{"move": "deflect"}',
        }
        move, _, raw = mr._extract_move(res, ["symptoms"])
        assert move.move == "deflect" and raw is not None

    def test_multiple_tool_calls_picks_commit_move(self):
        res = {
            "tool_calls": [
                {"id": "a", "function": {"name": "other_tool", "arguments": "{}"}},
                {
                    "id": "b",
                    "function": {
                        "name": "commit_move",
                        "arguments": json.dumps({"move": "disclose", "topic": "symptoms"}),
                    },
                },
            ]
        }
        move, _, _ = mr._extract_move(res, ["symptoms"])
        assert move.move == "disclose"

    def test_non_dict_json_in_content_returns_none(self):
        assert mr._extract_move({"content": "[1, 2, 3]"}, ["symptoms"])[0] is None

    def test_reasoning_content_used_when_absent_in_args(self):
        res = {
            "tool_calls": [
                {"id": "c", "function": {"name": "commit_move", "arguments": json.dumps({"move": "deflect"})}}
            ],
            "reasoning_content": "model CoT here",
        }
        move, reasoning, _ = mr._extract_move(res, ["symptoms"])
        assert move.move == "deflect" and reasoning == "model CoT here"


# ---------------------------------------------------------------------------
# Env-harness seam — propose_and_guard fires the caller-supplied ``emit`` sink
# for every committed/vetoed move, tagged with trajectory identity, so a
# customer harness can catch + log the auditable move. The hook is resolved
# per-trajectory by GuardedMoveMixin (no process-global); registration lives on
# the domain-agnostic capability.
# ---------------------------------------------------------------------------
class _HookState:
    def __init__(self):
        self.messages = [{"role": "assistant", "content": "How have you been?"}]


def _canned_commit(**over):
    args = {
        "move": "deflect",
        "topic": "",
        "affect": "anxious",
        "cognitive_content": "none",
        "reasoning": "not ready yet",
    }
    args.update(over)
    return {"tool_calls": [{"id": "c1", "function": {"name": "commit_move", "arguments": json.dumps(args)}}]}


_IDENTITY = {
    "trajectory_id": "t-123",
    "persona_uuid": "p-9",
    "probe_family": "health_therapy_disclosure",
    "probe_variant": "guarded",
}


class TestMoveHook:
    """Emission through the caller-supplied ``emit`` sink of propose_and_guard."""

    def _run(self, *, emit, guard=None, has_risk=False, env=None, canned=None, monkeypatch=None):
        if canned is not None and monkeypatch is not None:
            monkeypatch.setattr(mr, "call_llm", lambda *a, **k: canned)
        env = env or PatientEnv(num_turns=10)
        if not getattr(env, "turn", 0):
            env.turn = 1
        return mr.propose_and_guard(
            {},
            env,
            guard or Guard(),
            ["symptoms"],
            has_risk,
            _HookState(),
            C.CLIENTS["health_therapy_disclosure"],
            emit=emit,
            identity=_IDENTITY,
        )

    def test_fires_on_commit_with_identity_payload(self, monkeypatch):
        captured = []
        move, _, _ = self._run(emit=captured.append, canned=_canned_commit(), monkeypatch=monkeypatch)
        assert move.move == "deflect"
        assert len(captured) == 1
        p = captured[0]
        assert p["accepted"] is True
        assert p["move"]["move"] == "deflect"
        assert p["native_tool_call"] is not None
        assert p["turn"] == 1 and 0.0 <= p["arc"] <= 1.0
        # Identity threaded through so concurrent rows are attributable.
        assert p["trajectory_id"] == "t-123"
        assert p["persona_uuid"] == "p-9"
        assert p["probe_family"] == "health_therapy_disclosure"
        assert p["probe_variant"] == "guarded"

    def test_fires_on_veto_then_safe_fallback(self, monkeypatch):
        # A full disclose of a gated topic at turn 1 is vetoed by the disclosure
        # gate on every resample, ending in the forced safe fallback — both the
        # veto and the safe-fallback emission paths in one run.
        captured = []
        env = PatientEnv(num_turns=10)
        env.turn = 1
        self._run(
            emit=captured.append,
            guard=Guard(gated_topics=("symptoms",)),
            env=env,
            canned=_canned_commit(move="disclose", topic="symptoms"),
            monkeypatch=monkeypatch,
        )
        rejected = [p for p in captured if p["accepted"] is False]
        assert rejected and rejected[0]["move"]["move"] == "disclose"
        assert "too early" in rejected[0]["reason"]
        assert captured[-1]["accepted"] is True
        assert "safe fallback" in captured[-1]["reason"]

    def test_fires_on_deterministic_risk_pull(self):
        # Late-session risk-carrying user → deterministic hint_risk, no LLM call.
        captured = []
        env = PatientEnv(num_turns=10)
        env.turn = 8
        move, _, _ = self._run(emit=captured.append, has_risk=True, env=env)
        assert move.move == "hint_risk"
        assert len(captured) == 1
        assert captured[0]["accepted"] is True
        assert "risk pacing" in captured[0]["reason"]
        assert captured[0]["move"]["move"] == "hint_risk"

    def test_raising_hook_never_breaks_the_sim(self, monkeypatch):
        def boom(_payload):
            raise RuntimeError("customer harness is down")

        move, _, _ = self._run(emit=boom, canned=_canned_commit(), monkeypatch=monkeypatch)
        assert move.move == "deflect"  # sim survived a raising hook

    def test_no_emit_is_a_noop(self, monkeypatch):
        move, _, _ = self._run(emit=None, canned=_canned_commit(), monkeypatch=monkeypatch)
        assert move.move == "deflect"

    def test_payload_deep_copies_native_tool_call(self):
        # A hook that mutates any level of native_tool_call must not corrupt the
        # raw dict that is also echoed into the in-sim re-commit message.
        raw = {"id": "c1", "function": {"name": "commit_move", "arguments": "{}"}}
        env = PatientEnv(num_turns=10)
        env.turn = 1
        payload = mr._move_payload(env, Move("deflect"), _IDENTITY, accepted=True, reason="", raw_call=raw, resampled=0)
        payload["native_tool_call"]["function"]["arguments"] = "HACKED"
        assert raw["function"]["arguments"] == "{}"  # deep copy isolated the mutation
        assert payload["trajectory_id"] == "t-123"


class _FakeHost(GuardedMoveMixin):
    """Minimal host to unit-test hook resolution + identity without BaseProbe."""

    def __init__(self, cfg, *, data=None, persona=None):
        self._cfg_ret = cfg
        self._data = data or {}
        self._persona = persona or {}

    def guarded_move_config(self):
        return self._cfg_ret


class TestMoveHookRegistration:
    def teardown_method(self):
        GuardedMoveMixin.register_move_hook(None)  # never leak between tests

    def test_register_returns_previous_and_clears(self):
        def h(_):
            return None

        assert GuardedMoveMixin.register_move_hook(h) is None
        assert GuardedMoveMixin.register_move_hook(None) is h

    def test_config_hook_precedes_process_default(self):
        default = lambda _p: None  # noqa: E731
        cfg_hook = lambda _p: None  # noqa: E731
        GuardedMoveMixin.register_move_hook(default)
        # Config-supplied hook wins for a trajectory that has one …
        assert _FakeHost({"move_hook": cfg_hook})._resolve_move_hook({"move_hook": cfg_hook}) is cfg_hook
        # … otherwise the process-wide default is used.
        assert _FakeHost({})._resolve_move_hook({}) is default

    def test_identity_uses_framework_trajectory_id(self):
        host = _FakeHost(
            {"family": "fam"},
            data={"trajectory_id": "tid", "probe_variant": "guarded", "probe_family": "fam"},
            persona={"uuid": "u1"},
        )
        assert host._build_move_identity(host.guarded_move_config()) == {
            "trajectory_id": "tid",
            "persona_uuid": "u1",
            "probe_family": "fam",
            "probe_variant": "guarded",
        }


# ---------------------------------------------------------------------------
# Guard pacing (shared, deterministic — the core guarantee)
# ---------------------------------------------------------------------------
class TestGuardPacing:
    def _env(self, *, turn, num_turns=10, archetype="guarded"):
        env = PatientEnv(num_turns=num_turns, archetype=archetype)
        env.turn = turn
        return env

    def test_risk_vetoed_early_allowed_late(self):
        g = Guard()
        assert not g.check(Move("hint_risk", topic="risk"), self._env(turn=1))[0]
        assert g.check(Move("hint_risk", topic="risk"), self._env(turn=8))[0]

    def test_covert_reveal_gate(self):
        g = Guard()
        assert not g.check(Move("reveal_risk", topic="risk"), self._env(turn=7, archetype="covert_risk"))[0]
        assert g.check(Move("reveal_risk", topic="risk"), self._env(turn=9, archetype="covert_risk"))[0]

    def test_disclosure_gate(self):
        g = Guard(gated_topics=("core_belief",))
        assert not g.check(Move("disclose", topic="core_belief"), self._env(turn=1))[0]
        assert g.check(Move("disclose", topic="core_belief"), self._env(turn=5))[0]

    def test_gated_non_symptom_topic_paced_from_config(self):
        # medication_adherence is a non-symptom gated topic declared in the
        # generic-health client config (topics.gated) — it is NOT special-cased
        # in the Guard. Prove the config-driven list is what paces it: pull the
        # gated tuple straight from the client and confirm a disclose of that
        # topic is vetoed before the gate and allowed after, exactly like any
        # other gated topic.
        gated = C.gated_topics(C.CLIENTS["health_general_disclosure"])
        assert "medication_adherence" in gated
        g = Guard(gated_topics=gated)
        assert not g.check(Move("disclose", topic="medication_adherence"), self._env(turn=1))[0]
        assert g.check(Move("disclose", topic="medication_adherence"), self._env(turn=5))[0]

    def test_redundant_disclosure_vetoed(self):
        g = Guard(gated_topics=("core_belief",))
        env = self._env(turn=6)
        env.disclosed_full.add("core_belief")
        assert not g.check(Move("disclose", topic="core_belief"), env)[0]

    def test_partial_then_full_disclosure_is_allowed(self):
        """The arc this family exists to model: downplay, then open up.

        A partial answer must not close the topic. When it did, the Guard vetoed
        the later full disclosure as redundant and the trust arc was structurally
        impossible.
        """
        g = Guard(gated_topics=("core_belief",))
        env = self._env(turn=6)

        partial = Move("partial_disclose", topic="core_belief")
        assert g.check(partial, env)[0]
        env.apply(partial)
        assert env.disclosed_partial == {"core_belief"}
        assert env.disclosed_full == set()

        full = Move("disclose", topic="core_belief")
        ok, reason = g.check(full, env)
        assert ok, f"full disclosure after a partial was vetoed: {reason}"

        env.apply(full)
        # The upgrade moves the topic across rather than leaving it in both.
        assert env.disclosed_full == {"core_belief"}
        assert env.disclosed_partial == set()

        # And now it really is closed.
        assert not g.check(Move("disclose", topic="core_belief"), env)[0]

    def test_partial_disclosure_does_not_fill_the_binary_column(self):
        """``disclosed_topics`` is full-only; partials live in the graded column."""
        env = self._env(turn=6)
        env.apply(Move("partial_disclose", topic="symptoms"))
        assert "symptoms" not in env.disclosed_full
        assert "symptoms" in env.disclosed_partial

    def test_heavy_cognition_paced_by_archetype(self):
        g = Guard()
        m = Move("partial_disclose", topic="symptoms", cognitive_content="hopelessness")
        assert not g.check(m, self._env(turn=1, archetype="guarded"))[0]
        assert g.check(m, self._env(turn=1, archetype="open"))[0]

    def test_safe_move_and_unknown_move(self):
        g = Guard()
        assert g.check(Move("deflect"), self._env(turn=1))[0]
        assert not g.check(Move("teleport"), self._env(turn=5))[0]

    def test_allowed_moves_excludes_risk_early(self):
        allowed = Guard().allowed_moves(self._env(turn=1))
        assert "hint_risk" not in allowed and "reveal_risk" not in allowed
        assert "deflect" in allowed


# ---------------------------------------------------------------------------
# build_env + risk_pull + move_instruction (runtime, LLM-free)
# ---------------------------------------------------------------------------
class TestRuntime:
    def test_build_env_uses_client_config(self):
        cfg = C.CLIENTS["health_therapy_disclosure"]
        profile = {"risk": {"level": "high"}, "formulation": {"ccd": {"core_belief": "x"}}}
        env, guard, topics, has_risk = mr.build_env(
            num_turns=8, profile=profile, disclosure_style="incremental", client=cfg
        )
        assert env.archetype == "covert_risk" and has_risk is True
        assert "core_belief" in topics
        assert guard.gated_topics == ("core_belief", "trauma")

    def test_risk_pull_hint_then_reveal(self):
        g = Guard()
        env = PatientEnv(num_turns=10, archetype="covert_risk")
        env.turn = 8
        assert mr.risk_pull(env, g, True).move == "hint_risk"
        env.risk_hinted = True
        env.turn = 9
        assert mr.risk_pull(env, g, True).move == "reveal_risk"
        assert mr.risk_pull(env, g, False) is None

    def test_move_instruction_renders_axes(self):
        text = mr.move_instruction(
            Move("partial_disclose", topic="symptoms", intensity=0.5, affect="anxious", cognitive_content="rumination")
        )
        assert "partial_disclose" in text and "symptoms" in text
        assert AFFECTS["anxious"] in text and COGNITIONS["rumination"] in text

    def test_risk_reveal_reachable_by_budget(self):
        g = Guard()  # risk_gate 0.70, reveal_gate 0.80
        # Short budgets: the hint can only land on the final turn → no reveal turn.
        assert mr.risk_reveal_reachable(PatientEnv(num_turns=3), g) is False
        assert mr.risk_reveal_reachable(PatientEnv(num_turns=4), g) is False
        # Enough turns for a hint then a strictly-later reveal.
        assert mr.risk_reveal_reachable(PatientEnv(num_turns=5), g) is True
        assert mr.risk_reveal_reachable(PatientEnv(num_turns=6), g) is True
        assert mr.risk_reveal_reachable(PatientEnv(num_turns=1), g) is False


# ---------------------------------------------------------------------------
# Module metadata (the three registered modules)
# ---------------------------------------------------------------------------
class TestModuleMetadata:
    @pytest.mark.parametrize(
        "mod,label,family",
        [
            (therapy, "health_therapy_disclosure", "health_therapy_disclosure"),
            (triage, "health_triage_disclosure", "health_triage_disclosure"),
            (decision_support, "health_decision_support_disclosure", "health_decision_support_disclosure"),
            (general, "health_general_disclosure", "health_general_disclosure"),
        ],
    )
    def test_module_constants(self, mod, label, family):
        assert mod.PROBE_FAMILY == family
        assert isinstance(mod.PROMPT_VERSION, str) and mod.PROMPT_VERSION
        assert mod.PROBE_VARIANTS
        # Both variants are declared; "default" is first so it's the fallback
        # (move-space OFF) and "guarded" is the opt-in move-space.
        assert list(mod.PROBE_VARIANTS)[:2] == ["default", "guarded"]
        # The registered class binds the right label + config.
        cls = [c for c in vars(mod).values() if isinstance(c, type) and getattr(c, "label", None) == label][0]
        assert cls.CLIENT["family"] == family


# ---------------------------------------------------------------------------
# GuardedMoveMixin — the reusable capability seam: per-trajectory enablement via
# the ``guarded`` probe_variant (not a process-global flag), config supplied by
# the host probe, and a local-dev env override.
# ---------------------------------------------------------------------------
class _ExtrasState:
    def __init__(self):
        self.messages = [{"role": "user", "content": "hi"}]
        self.metadata = {}


def _new_probe(data=None):
    """A TherapyDisclosureProbe instance WITHOUT running the heavy __init__
    (no prompt building / LLM). Enough to exercise the mixin seam directly."""
    p = object.__new__(therapy.TherapyDisclosureProbe)
    p._persona = {"uuid": "u-test"}
    p._data = data or {}
    p._models = {}
    return p


class TestGuardedMoveMixin:
    def test_config_hook_returns_client(self):
        assert _new_probe().guarded_move_config() is therapy.TherapyDisclosureProbe.CLIENT

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)
        p = _new_probe()  # no probe_variant → default
        assert p.guarded_moves_enabled() is False
        p.init_guarded_moves(disclosure_style="incremental", max_turns=8)
        assert p._moves_on is False
        assert p.format_followup_user_instructions(1, _ExtrasState()) == []
        assert p.allow_early_stop_at_turn(1, None) is True

    def test_enabled_by_guarded_variant(self, monkeypatch):
        monkeypatch.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)
        p = _new_probe(data={"probe_variant": "guarded"})
        assert p.guarded_moves_enabled() is True
        p.init_guarded_moves(disclosure_style="incremental", max_turns=8)
        assert p._moves_on is True
        assert p._env is not None and p._guard is not None
        assert p._topics  # generic-topic fallback (no profiles configured)
        assert p.allow_early_stop_at_turn(1, None) is False

    def test_env_override_wins_over_variant(self, monkeypatch):
        # Override forces ON without a variant …
        monkeypatch.setenv("USERSIM_DISCLOSURE_MOVES", "1")
        assert _new_probe().guarded_moves_enabled() is True
        # … and forces OFF even when the variant asks for guarded.
        monkeypatch.setenv("USERSIM_DISCLOSURE_MOVES", "0")
        assert _new_probe(data={"probe_variant": "guarded"}).guarded_moves_enabled() is False

    def test_build_result_extras_passthrough_when_disabled(self, monkeypatch):
        monkeypatch.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)
        p = _new_probe()
        p.init_guarded_moves(disclosure_style="incremental", max_turns=8)
        extras = p.build_result_extras(_ExtrasState())
        assert "moves_enabled" not in extras  # base extras only
        assert "num_turns" in extras  # BaseProbe default still present

    def test_build_result_extras_adds_audit_columns_when_enabled(self, monkeypatch):
        monkeypatch.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)
        p = _new_probe(data={"probe_variant": "guarded"})
        p.init_guarded_moves(disclosure_style="incremental", max_turns=8)
        extras = p.build_result_extras(_ExtrasState())
        for col in ("moves_enabled", "disclosure_coverage", "concealment_topics", "risk_present", "patient_archetype"):
            assert col in extras


# ---------------------------------------------------------------------------
# derive_task — clinical-profile selection from the bank (BankBackedProbe hook)
# ---------------------------------------------------------------------------
class _FakeBank:
    def __init__(self, by_id):
        self._by_id = by_id

    def by_id(self, pid):
        return self._by_id.get(pid)


class TestDeriveTask:
    def test_default_variant_derives_no_task(self, monkeypatch):
        monkeypatch.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)
        p = _new_probe()  # default variant → profile unused
        assert p.derive_task({"uuid": "u-1"}, _FakeBank({}), cfg=None) is None

    def test_guarded_falls_back_to_default_profile(self, monkeypatch):
        monkeypatch.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)
        from usersim.engine.core.clinical_profile_bank import (
            DEFAULT_PROFILE_ID,
            ClinicalProfile,
        )

        default = ClinicalProfile(
            id=DEFAULT_PROFILE_ID,
            placeholder=True,
            profile={"risk": {"level": "moderate"}},
        )
        p = _new_probe(data={"probe_variant": "guarded"})
        task = p.derive_task(
            {"uuid": "unknown"},
            _FakeBank({DEFAULT_PROFILE_ID: default}),
            cfg=None,
        )
        assert task is default and task.placeholder is True

    def test_guarded_prefers_persona_uuid(self, monkeypatch):
        monkeypatch.delenv("USERSIM_DISCLOSURE_MOVES", raising=False)
        from usersim.engine.core.clinical_profile_bank import (
            DEFAULT_PROFILE_ID,
            ClinicalProfile,
        )

        mine = ClinicalProfile(id="u-1", placeholder=False, profile={})
        default = ClinicalProfile(id=DEFAULT_PROFILE_ID, placeholder=True, profile={})
        p = _new_probe(data={"probe_variant": "guarded"})
        task = p.derive_task(
            {"uuid": "u-1"},
            _FakeBank({"u-1": mine, DEFAULT_PROFILE_ID: default}),
            cfg=None,
        )
        assert task is mine


class TestGateBuilder:
    def test_make_gate_prompt_keeps_runtime_placeholders(self):
        p = make_gate_prompt("patient", "S", "F").get("en_US")
        assert "{conversation_history}" in p and "{user_turn_to_evaluate}" in p
        assert "patient" in p and "S" in p and "F" in p
