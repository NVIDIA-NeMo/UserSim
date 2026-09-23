# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for ``core.realized_audit`` — reconciling realized behavior vs intent.

These live in ``tests/core`` because the audit moved out of the probe: it is
consumed by the evaluator, and the two layers may not import each other, so the
shared definition sits in core.
"""

from __future__ import annotations

import json

from usersim.engine.core import realized_audit as ra


def _v(full=None, partial=None, risk=False):
    return {"fully_disclosed": list(full or []), "partially_disclosed": list(partial or []), "risk_revealed": risk}


# ---------------------------------------------------------------------------
# reconcile_realized — realized-behavior vs committed-intent (pure)
#   Moves align to verdicts BY TURN NUMBER; disclosure is graded full/partial/none.
# ---------------------------------------------------------------------------
class TestReconcileRealized:
    def _moves(self):
        return [
            {"turn": 2, "move": "partial_disclose", "topic": "symptoms"},
            {"turn": 3, "move": "withhold", "topic": "trauma"},
            {"turn": 4, "move": "reveal_risk", "topic": "risk"},
        ]

    def test_consistent_behavior_no_mismatch(self):
        # partial_disclose landing as PARTIAL is not a mismatch (it did its job).
        verdicts = {
            2: _v(partial=["symptoms"]),
            3: _v(),
            4: _v(risk=True),
        }
        out = ra.reconcile_realized(self._moves(), verdicts)
        assert out["move_realized_mismatches"] == 0
        assert out["moves_verified"] == 3
        assert out["realized_disclosure_levels"] == {"symptoms": "partial"}
        assert out["realized_partial_topics"] == ["symptoms"]
        assert out["realized_disclosed_topics"] == []
        assert out["realized_risk_revealed"] is True

    def test_leaked_concealment_and_missing_disclosure_are_mismatches(self):
        verdicts = {
            2: _v(),  # partial_disclose(symptoms) → nothing → mismatch
            3: _v(full=["trauma"]),  # withhold(trauma) → fully leaked → mismatch
            4: _v(risk=False),  # reveal_risk → no risk → mismatch
        }
        out = ra.reconcile_realized(self._moves(), verdicts)
        assert out["move_realized_mismatches"] == 3
        assert out["realized_disclosed_topics"] == ["trauma"]
        assert out["realized_risk_revealed"] is False

    def test_partial_disclose_landing_partial_is_not_mismatch(self):
        # The core #4 fix: partial elicitation graded as partial, not punished.
        out = ra.reconcile_realized(
            [{"turn": 2, "move": "partial_disclose", "topic": "symptoms"}], {2: _v(partial=["symptoms"])}
        )
        assert out["move_realized_mismatches"] == 0
        assert out["realized_disclosure_levels"] == {"symptoms": "partial"}

    def test_full_disclose_that_only_partly_lands_is_not_a_hard_mismatch(self):
        out = ra.reconcile_realized(
            [{"turn": 2, "move": "disclose", "topic": "symptoms"}], {2: _v(partial=["symptoms"])}
        )
        assert out["move_realized_mismatches"] == 0
        assert out["realized_disclosure_levels"] == {"symptoms": "partial"}

    def test_missing_verdict_falls_back_to_intent(self):
        # No verdicts at all → aggregates derived from intent, no mismatches.
        out = ra.reconcile_realized(self._moves(), {})
        assert out["moves_verified"] == 0
        assert out["move_realized_mismatches"] == 0
        # partial_disclose intent → partial; reveal_risk intent → risk revealed.
        assert out["realized_disclosure_levels"] == {"symptoms": "partial"}
        assert out["realized_risk_revealed"] is True

    def test_alignment_is_by_turn_not_position(self):
        # Only turn 3 audited; turn 2 & 4 fall back to intent (no mismatch).
        out = ra.reconcile_realized(self._moves(), {3: _v(full=["trauma"])})
        assert out["moves_verified"] == 1
        assert out["move_realized_mismatches"] == 1  # withhold(trauma) leaked
        assert out["realized_disclosure_levels"]["trauma"] == "full"
        assert out["realized_disclosure_levels"]["symptoms"] == "partial"  # intent fallback

    def test_conceal_risk_leak_is_mismatch(self):
        out = ra.reconcile_realized([{"turn": 2, "move": "conceal_risk", "topic": "risk"}], {2: _v(risk=True)})
        assert out["move_realized_mismatches"] == 1
        assert out["realized_risk_revealed"] is True


# ---------------------------------------------------------------------------
# verify_realized_transcript — the REAL function, with only the LLM boundary faked.
#
#   These exist because of a regression that shipped: the audit was moved to this
#   module without its imports, so every call raised NameError, a bare
#   ``except Exception`` turned that into ``{}``, and the scorer silently graded
#   committed intent forever. The whole suite stayed green, because the tests
#   monkeypatched ``verify_realized_transcript`` itself and never ran a line of it.
#
#   The rule these encode: fake the transport (``acall_llm``), never the function
#   under test.
# ---------------------------------------------------------------------------
class TestVerifyRealizedTranscriptRealPath:
    ITEMS = [
        {"turn": 2, "text": "sleep's been rough, but honestly it's fine"},
        {"turn": 3, "text": "ok — some nights I don't see the point of any of it"},
    ]
    TOPICS = ["symptoms", "core_belief"]

    def _stub(self, audits):
        """A stand-in for the awaited auditor call, returning canned verdicts."""

        async def _call(*args, **kwargs):
            return self._completion(audits)

        return _call

    def _completion(self, audits):
        import json

        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a1", "function": {"name": "record_audit", "arguments": json.dumps({"audits": audits})}}
            ],
        }

    async def test_parses_a_native_tool_call_into_verdicts(self, monkeypatch):
        monkeypatch.setattr(
            ra,
            "acall_llm",
            self._stub(
                [
                    {"turn": 2, "fully_disclosed": [], "partially_disclosed": ["symptoms"], "risk_revealed": False},
                    {"turn": 3, "fully_disclosed": ["core_belief"], "partially_disclosed": [], "risk_revealed": True},
                ]
            ),
        )
        out = await ra.verify_realized_transcript({"judge_model": object()}, self.ITEMS, self.TOPICS, "self-harm")
        assert set(out) == {2, 3}
        assert out[2]["partially_disclosed"] == ["symptoms"]
        assert out[3]["fully_disclosed"] == ["core_belief"] and out[3]["risk_revealed"] is True

    async def test_prompt_names_the_concrete_risk_not_the_generic_word(self, monkeypatch):
        # risk_noun is interpolated into the auditor's system prompt. If the
        # producer stops persisting it the audit still "works" while asking about
        # a meaningless generic "risk", so pin the interpolation.
        seen = {}

        async def fake(models, alias, msgs, **kw):
            seen["system"] = msgs[0]["content"]
            seen["user"] = msgs[1]["content"]
            return await self._completion([])

        monkeypatch.setattr(ra, "acall_llm", fake)
        await ra.verify_realized_transcript(
            {"judge_model": object()}, self.ITEMS, self.TOPICS, "an urgent red-flag symptom"
        )
        assert "an urgent red-flag symptom" in seen["system"]
        # And every graded turn reaches the auditor, keyed by turn number.
        assert "[turn 2]" in seen["user"] and "[turn 3]" in seen["user"]

    async def test_topics_outside_the_carried_set_are_dropped(self, monkeypatch):
        monkeypatch.setattr(
            ra,
            "acall_llm",
            self._stub(
                [
                    {
                        "turn": 2,
                        "fully_disclosed": ["symptoms", "not_a_carried_topic"],
                        "partially_disclosed": [],
                        "risk_revealed": False,
                    }
                ]
            ),
        )
        out = await ra.verify_realized_transcript({"judge_model": object()}, self.ITEMS, self.TOPICS, "self-harm")
        assert out[2]["fully_disclosed"] == ["symptoms"]

    async def test_entries_for_turns_that_were_not_graded_are_ignored(self, monkeypatch):
        monkeypatch.setattr(
            ra,
            "acall_llm",
            self._stub(
                [{"turn": 99, "fully_disclosed": ["symptoms"], "partially_disclosed": [], "risk_revealed": False}]
            ),
        )
        out = await ra.verify_realized_transcript({"judge_model": object()}, self.ITEMS, self.TOPICS, "self-harm")
        assert out == {}

    async def test_transport_failure_falls_back_quietly(self, monkeypatch):
        async def boom(*a, **k):
            raise RuntimeError("auditor endpoint down")

        monkeypatch.setattr(ra, "acall_llm", boom)
        out = await ra.verify_realized_transcript({"judge_model": object()}, self.ITEMS, self.TOPICS, "self-harm")
        assert out == {}

    async def test_no_models_short_circuits_without_calling_out(self, monkeypatch):
        async def boom(*a, **k):
            raise AssertionError("must not reach the model with no models wired")

        monkeypatch.setattr(ra, "acall_llm", boom)
        assert await ra.verify_realized_transcript({}, self.ITEMS, self.TOPICS, "self-harm") == {}

    async def test_empty_turns_are_not_sent_for_grading(self, monkeypatch):
        seen = {}

        async def fake(models, alias, msgs, **kw):
            seen["user"] = msgs[1]["content"]
            return await self._completion([])

        monkeypatch.setattr(ra, "acall_llm", fake)
        await ra.verify_realized_transcript(
            {"judge_model": object()},
            [{"turn": 2, "text": "  "}, {"turn": 3, "text": "real content"}],
            self.TOPICS,
            "self-harm",
        )
        assert "[turn 3]" in seen["user"] and "[turn 2]" not in seen["user"]


class TestVerifyPromptPack:
    def test_resolves_for_every_shipped_locale(self):
        from usersim.engine.core.locale import SHIPPED_LOCALES

        for loc in SHIPPED_LOCALES:
            assert "disclosure auditor" in ra.VERIFY_SYSTEM_PACK.get(loc, risk_noun="self-harm")


class TestAuditorAliasResolution:
    """The audit runs in the EVALUATOR, so it must resolve against the aliases
    the evaluator actually holds.

    This shipped broken: the function kept defaulting to ``user_model``, a
    simulation-time alias, after the audit moved layers. A real eval wires
    ``evaluator_model`` and no ``user_model``, so every trajectory raised
    "Model alias 'user_model' not found in models dict" deep inside acall_llm and
    silently fell back to committed intent. The offline tests missed it because
    they all passed a models dict containing ``judge_model``.
    """

    async def test_evaluator_shaped_models_resolve(self):
        # What `usersim eval` actually passes: no judge_model, no user_model.
        assert ra.resolve_audit_model({"evaluator_model": object()}) == "evaluator_model"

    async def test_judge_wins_when_present(self):
        assert ra.resolve_audit_model({"judge_model": object(), "evaluator_model": object()}) == "judge_model"

    async def test_simulation_shaped_models_still_resolve(self):
        assert ra.resolve_audit_model({"user_model": object()}) == "user_model"

    async def test_no_usable_alias_returns_empty_rather_than_guessing(self):
        assert ra.resolve_audit_model({"assistant_model": object()}) == ""
        assert ra.resolve_audit_model({}) == ""

    async def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("USERSIM_AUDIT_MODEL", "my_model")
        assert ra.resolve_audit_model({"judge_model": object()}) == "my_model"

    async def test_audit_runs_with_an_evaluator_shaped_models_dict(self, monkeypatch):
        """The production case, end to end: evaluator_model only, and the audit
        must actually call out rather than raising on a missing alias."""
        used = {}

        async def fake_call_llm(models, alias, msgs, **kw):
            used["alias"] = alias
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "a1",
                        "function": {
                            "name": "record_audit",
                            "arguments": json.dumps(
                                {
                                    "audits": [
                                        {
                                            "turn": 2,
                                            "fully_disclosed": ["symptoms"],
                                            "partially_disclosed": [],
                                            "risk_revealed": False,
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ],
            }

        monkeypatch.setattr(ra, "acall_llm", fake_call_llm)
        out = await ra.verify_realized_transcript(
            {"evaluator_model": object()}, [{"turn": 2, "text": "sleep has been rough"}], ["symptoms"], "self-harm"
        )
        assert used["alias"] == "evaluator_model"
        assert out[2]["fully_disclosed"] == ["symptoms"]

    async def test_unusable_models_dict_degrades_without_calling_out(self, monkeypatch):
        async def boom(*a, **k):
            raise AssertionError("must not attempt a call with no usable alias")

        monkeypatch.setattr(ra, "acall_llm", boom)
        assert (
            await ra.verify_realized_transcript(
                {"assistant_model": object()}, [{"turn": 2, "text": "text"}], ["symptoms"], "self-harm"
            )
            == {}
        )


class TestAuditReasoningEffort:
    """The auditor fills in a form; it does not need to deliberate.

    The catalog runs the 120b judge at reasoning_effort=high with an 8192-token
    budget. On a seven-turn transcript a live evaluation came back with no tool
    call and empty content — the budget went on thinking. Extraction tasks want
    the tokens spent on the answer.
    """

    async def test_defaults_to_low_effort(self, monkeypatch):
        monkeypatch.delenv("USERSIM_AUDIT_REASONING_EFFORT", raising=False)
        assert ra._audit_reasoning_kwargs() == {"reasoning_effort": "low"}

    async def test_can_be_turned_off_entirely(self, monkeypatch):
        monkeypatch.setenv("USERSIM_AUDIT_REASONING_EFFORT", "off")
        assert ra._audit_reasoning_kwargs() == {"chat_template_kwargs": {"enable_thinking": False}}

    async def test_explicit_levels_pass_through(self, monkeypatch):
        monkeypatch.setenv("USERSIM_AUDIT_REASONING_EFFORT", "high")
        assert ra._audit_reasoning_kwargs() == {"reasoning_effort": "high"}

    async def test_unrecognised_value_falls_back_to_catalog_defaults(self, monkeypatch):
        monkeypatch.setenv("USERSIM_AUDIT_REASONING_EFFORT", "banana")
        assert ra._audit_reasoning_kwargs() == {}

    async def test_the_override_actually_reaches_the_model_call(self, monkeypatch):
        # The knob is worthless if it is computed and then not forwarded.
        seen = {}

        async def fake(models, alias, msgs, **kw):
            seen.update(kw)
            return {"role": "assistant", "content": "", "tool_calls": []}

        monkeypatch.delenv("USERSIM_AUDIT_REASONING_EFFORT", raising=False)
        monkeypatch.setattr(ra, "acall_llm", fake)
        await ra.verify_realized_transcript(
            {"judge_model": object()}, [{"turn": 2, "text": "text"}], ["symptoms"], "self-harm"
        )
        assert seen.get("reasoning_effort") == "low"


class TestPlainJsonOutputFormat:
    """The audit asks for JSON in the message body, not a tool call.

    Measured, not stylistic. Across the models on an OpenAI-compatible hub,
    asking for a tool call was the worst of four output formats: two open models
    returned nothing usable for any of seven turns, including the model that
    emits the probe's own flat commit_move calls all day. Requesting a tool call
    actively suppressed the answer -- the model reasoned through the whole task
    and then emitted neither a call nor content. Plain JSON was the only format
    that worked everywhere, at the lowest token cost, in one round trip.
    """

    _ITEMS = [{"turn": 2, "text": "sleep has been rough"}, {"turn": 3, "text": "I'd rather not talk about home"}]
    _PAYLOAD = {
        "audits": [
            {"turn": 2, "fully_disclosed": ["symptoms"], "partially_disclosed": [], "risk_revealed": False},
            {"turn": 3, "fully_disclosed": [], "partially_disclosed": ["trauma"], "risk_revealed": False},
        ]
    }

    async def _run(self, monkeypatch, content):
        seen = {}

        async def fake(models, alias, msgs, **kw):
            seen.update(kw)
            seen["system"], seen["user"] = msgs[0]["content"], msgs[1]["content"]
            return {"role": "assistant", "content": content}

        monkeypatch.setattr(ra, "acall_llm", fake)
        out = await ra.verify_realized_transcript(
            {"judge_model": object()}, self._ITEMS, ["symptoms", "trauma"], "self-harm"
        )
        return out, seen

    async def test_no_tools_are_sent(self, monkeypatch):
        _, seen = await self._run(monkeypatch, json.dumps(self._PAYLOAD))
        assert "tools" not in seen, "sending tools is what suppressed the answer"
        assert "tool_choice" not in seen

    async def test_the_prompt_asks_for_json_not_a_tool_call(self, monkeypatch):
        _, seen = await self._run(monkeypatch, json.dumps(self._PAYLOAD))
        assert "JSON" in seen["system"] or "JSON" in seen["user"]
        assert "audits" in seen["user"]
        # The old instruction would send a model looking for a tool that is no
        # longer offered.
        assert "record_audit tool" not in seen["system"]

    async def test_bare_json_is_parsed(self, monkeypatch):
        out, _ = await self._run(monkeypatch, json.dumps(self._PAYLOAD))
        assert out[2]["fully_disclosed"] == ["symptoms"]
        assert out[3]["partially_disclosed"] == ["trauma"]

    async def test_fenced_json_is_parsed(self, monkeypatch):
        out, _ = await self._run(monkeypatch, "```json\n" + json.dumps(self._PAYLOAD) + "\n```")
        assert set(out) == {2, 3}

    async def test_json_wrapped_in_prose_is_parsed(self, monkeypatch):
        out, _ = await self._run(monkeypatch, "Here you go:\n" + json.dumps(self._PAYLOAD) + "\nLet me know!")
        assert set(out) == {2, 3}

    async def test_a_native_tool_call_would_still_be_read(self, monkeypatch):
        """No tools are offered, but a provider that volunteers one is still
        parsed -- recover_tool_call handles both shapes, so nothing is lost."""

        async def fake(models, alias, msgs, **kw):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "1", "function": {"name": "record_audit", "arguments": json.dumps(self._PAYLOAD)}}
                ],
            }

        monkeypatch.setattr(ra, "acall_llm", fake)
        out = await ra.verify_realized_transcript(
            {"judge_model": object()}, self._ITEMS, ["symptoms", "trauma"], "self-harm"
        )
        assert set(out) == {2, 3}

    async def test_prose_with_no_json_degrades_to_intent(self, monkeypatch):
        out, _ = await self._run(monkeypatch, "Turn 2 mentions sleep problems.")
        assert out == {}
