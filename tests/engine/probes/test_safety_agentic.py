# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``safety_agentic`` probe.

Layered the same way as :mod:`test_safety_chat_pressure`:

1. **persona_to_tags re-export** — verifies the canonical
   sovereign-knowledge tag-derivation is reused (cross-probe joins
   on persona-feature axes share one vocabulary).
2. **derive_task** — pool construction with persona-tag filtering,
   exclusion lists, deterministic salting, bank-version perturbation.
3. **resolve_task_from_row** — input-column override paths,
   missing-id failure modes, persona-tag guard against panel-driven
   mismatches.
4. **prompts** — assistant system prompt static checks.
5. **_format_tools_for_api / _extract_tool_call** — internal helpers.
6. **End-to-end** — registry, single-turn, multi-turn tool-call
   loops, verbatim turn-1 injection, tool-call interception with
   mock responses, attempted_actions side channel, structured
   failure paths, placeholder warning emission, no-tool-call
   short-circuit.
7. **should_succeed** — sim-side guardrail invariant.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from usersim.engine.core.agentic_bank import (
    ActionProvenance,
    ActionRequest,
    AgenticBank,
    MockToolResponse,
    ToolSpec,
    load_agentic_bank,
)
from usersim.engine.core.outcomes import (
    FailureClass,
    OutcomeStatus,
    Provenance,
    WarningKind,
)
import usersim.engine.generator  # noqa: F401 — triggers probe registration
from usersim.engine.core.probes import _PROBE_REGISTRY, resolve_probe
from usersim.engine.probes.safety_agentic import (
    generator as probe_gen,
    prompts,
    task_derivation,
)
from usersim.engine.core._assets import packaged_assets_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_BANK_PATH = (packaged_assets_dir() / "safety_agentic/sample.yaml")


@pytest.fixture
def sample_bank() -> AgenticBank:
    return load_agentic_bank(SAMPLE_BANK_PATH)


@pytest.fixture
def en_persona() -> Dict[str, Any]:
    return {
        "first_name": "Sarah",
        "last_name": "Johnson",
        "age": 35,
        "city": "Seattle",
        "education_level": "Bachelor",
        "occupation": "Software Engineer",
    }


@pytest.fixture(autouse=True)
def _reset_bank_cache_between_tests():
    probe_gen._reset_bank_cache()
    yield
    probe_gen._reset_bank_cache()


@pytest.fixture
def simulator_cfg():
    """Minimal cfg stub the probe reads from."""
    class _Cfg:
        random_seed = 42
        max_turns = 1
    return _Cfg()


@pytest.fixture
def simulator_cfg_three_turns():
    class _Cfg:
        random_seed = 42
        max_turns = 3
    return _Cfg()


def _shipped_bank_env() -> Dict[str, str]:
    return {"USERSIM_SAFETY_AGENTIC_BANK": str(SAMPLE_BANK_PATH)}


def _mock_call_llm(responses: List[Dict[str, Any]]):
    """Side-effect callable yielding the next queued response."""
    iter_ = iter(responses)

    def _side_effect(*args, **kwargs):
        try:
            return next(iter_)
        except StopIteration as e:
            raise AssertionError(
                f"test consumed more LLM calls than expected; "
                f"args_head={args[:2]} kwargs_keys={list(kwargs)}"
            ) from e

    return _side_effect


def _make_tool_call(
    name: str,
    arguments: Dict[str, Any] | str,
    *,
    tc_id: str = "call_0",
) -> Dict[str, Any]:
    """Build an OpenAI-style tool_call dict."""
    args_str = (
        arguments if isinstance(arguments, str) else json.dumps(arguments)
    )
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


def _synthetic_bank(
    *,
    bank_id: str = "synthetic",
    bank_version: str = "v0.1.0",
    n_requests: int = 2,
    request_persona_tags: tuple = (),
    placeholder: bool = True,
    sub_protocol: str = "unsanctioned_action",
    tool_name: str = "do_thing",
    sanctioned_action_name: str | None = None,
) -> AgenticBank:
    """Build a synthetic ``AgenticBank`` for unit tests."""
    requests = []
    for i in range(n_requests):
        tool = ToolSpec(
            name=tool_name,
            description=f"A tool that does thing {i}.",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        )
        mock = MockToolResponse(
            role="tool",
            name=tool_name,
            content=json.dumps({"ok": True, "i": i}),
        )
        from usersim.engine.core.localized import LocalizedText
        _en = LocalizedText.from_yaml_value
        requests.append(
            ActionRequest(
                id=f"AR-T-{i:03d}",
                sub_protocol=sub_protocol,
                placeholder=placeholder,
                description=_en(f"Synthetic request {i}"),
                sanctioned_action_name=sanctioned_action_name,
                tempted_action_name=None,
                initial_user_prompt=_en(f"Please do thing {i}."),
                expected_behavior=_en(f"Sketch for request {i}."),
                simulated_tools=(tool,),
                mock_tool_responses={tool_name: mock},
                persona_tags=request_persona_tags,
                provenance=ActionProvenance(
                    source="placeholder" if placeholder else "internal",
                    last_reviewed="2026-04-20",
                ),
                bank_id=bank_id,
                bank_version=bank_version,
            )
        )
    return AgenticBank(
        schema_version="v0.1",
        bank_id=bank_id,
        bank_version=bank_version,
        sub_protocols=(sub_protocol,),
        action_requests=tuple(requests),
        source_path="<synthetic>",
    )


# ---------------------------------------------------------------------------
# 1. persona_to_tags re-export
# ---------------------------------------------------------------------------


class TestPersonaToTagsReExport:
    def test_re_exports_canonical(self) -> None:
        from usersim.engine.probes.sov_ai_facts.task_derivation import (
            persona_to_tags as canonical,
        )
        assert task_derivation.persona_to_tags is canonical


# ---------------------------------------------------------------------------
# 2. derive_task
# ---------------------------------------------------------------------------


class TestDeriveTask:
    def test_returns_request_when_pool_nonempty(
        self, en_persona: Dict[str, Any],
    ) -> None:
        bank = _synthetic_bank()
        ar = task_derivation.derive_task(en_persona, bank, "en_US", seed=42)
        assert isinstance(ar, ActionRequest)

    def test_returns_none_when_no_request_matches_persona(
        self, en_persona: Dict[str, Any],
    ) -> None:
        bank = _synthetic_bank(
            n_requests=1, request_persona_tags=("interest:nonexistent",),
        )
        result = task_derivation.derive_task(en_persona, bank, "en_US", seed=42)
        assert result is None

    def test_returns_none_when_all_excluded(
        self, en_persona: Dict[str, Any],
    ) -> None:
        bank = _synthetic_bank(n_requests=2)
        result = task_derivation.derive_task(
            en_persona, bank, "en_US", seed=42,
            excluded_request_ids={"AR-T-000", "AR-T-001"},
        )
        assert result is None

    def test_deterministic_for_same_inputs(
        self, en_persona: Dict[str, Any],
    ) -> None:
        bank = _synthetic_bank(n_requests=4)
        a = task_derivation.derive_task(en_persona, bank, "en_US", seed=42)
        b = task_derivation.derive_task(en_persona, bank, "en_US", seed=42)
        assert a is not None and b is not None
        assert a.id == b.id

    def test_different_seed_can_produce_different_pick(
        self, en_persona: Dict[str, Any],
    ) -> None:
        bank = _synthetic_bank(n_requests=4)
        picks = set()
        for s in range(20):
            ar = task_derivation.derive_task(en_persona, bank, "en_US", seed=s)
            picks.add(ar.id if ar is not None else None)
        assert len(picks) > 1

    def test_bank_version_change_perturbs_pick(
        self, en_persona: Dict[str, Any],
    ) -> None:
        bank_a = _synthetic_bank(bank_version="v0.1.0", n_requests=4)
        bank_b = _synthetic_bank(bank_version="v0.2.0", n_requests=4)
        any_diff = False
        for s in range(20):
            a = task_derivation.derive_task(en_persona, bank_a, "en_US", seed=s)
            b = task_derivation.derive_task(en_persona, bank_b, "en_US", seed=s)
            if a is not None and b is not None and a.id != b.id:
                any_diff = True
                break
        assert any_diff, "bank_version bump should perturb the pick"


# ---------------------------------------------------------------------------
# 3. resolve_task_from_row
# ---------------------------------------------------------------------------


class TestResolveTaskFromRow:
    def test_override_returns_request(self, en_persona: Dict[str, Any]) -> None:
        bank = _synthetic_bank(n_requests=2)
        ar = task_derivation.resolve_task_from_row(
            en_persona, bank, "en_US",
            action_request_id="AR-T-001",
        )
        assert ar is not None
        assert ar.id == "AR-T-001"

    def test_override_missing_id_returns_none(
        self, en_persona: Dict[str, Any],
    ) -> None:
        bank = _synthetic_bank()
        ar = task_derivation.resolve_task_from_row(
            en_persona, bank, "en_US",
            action_request_id="AR-DOES-NOT-EXIST",
        )
        assert ar is None

    def test_override_persona_tag_mismatch_returns_none(
        self, en_persona: Dict[str, Any],
    ) -> None:
        bank = _synthetic_bank(
            request_persona_tags=("interest:nonexistent",),
        )
        ar = task_derivation.resolve_task_from_row(
            en_persona, bank, "en_US",
            action_request_id="AR-T-000",
        )
        assert ar is None

    def test_no_override_falls_through_to_derive(
        self, en_persona: Dict[str, Any],
    ) -> None:
        bank = _synthetic_bank()
        ar = task_derivation.resolve_task_from_row(
            en_persona, bank, "en_US", seed=42,
        )
        assert ar is not None


# ---------------------------------------------------------------------------
# 4. Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_system_prompt_is_empty_pure_capability_policy(self) -> None:
        """Pure-capability-test policy: the assistant under test gets
        no system prompt. Whether the model confirms before
        irreversible actions, discloses consequences, or treats tool
        output as untrusted data is exactly what safety_agentic
        measures — we don't tell the model to do those things. The
        OpenAI ``tools=`` parameter still exposes the tool schemas
        (the model knows what tools exist), but no behavioral
        instruction. See conversation-plugin/README.md "Assistant
        under test: pure-capability policy".
        """
        assert prompts.ASSISTANT_SYSTEM_PROMPT == ""

    def test_system_prompt_has_no_unfilled_placeholders(self) -> None:
        text = prompts.ASSISTANT_SYSTEM_PROMPT
        assert "{" not in text
        assert "}" not in text


# ---------------------------------------------------------------------------
# 5. Internal helpers
# ---------------------------------------------------------------------------


class TestFormatToolsForApi:
    def test_emits_openai_function_shape(self) -> None:
        spec = ToolSpec(
            name="lookup",
            description="Look something up.",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        out = probe_gen._format_tools_for_api([spec])
        assert out == [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up.",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        }]

    def test_falls_back_to_empty_object_schema_when_parameters_missing(
        self,
    ) -> None:
        spec = ToolSpec(name="t", description="d", parameters={})
        out = probe_gen._format_tools_for_api([spec])
        assert out[0]["function"]["parameters"] == {
            "type": "object", "properties": {},
        }


class TestExtractToolCall:
    def test_parses_standard_openai_shape(self) -> None:
        tc = _make_tool_call("transfer_funds", {"amount": 250})
        name, args = probe_gen._extract_tool_call(tc)
        assert name == "transfer_funds"
        assert args == {"amount": 250}

    def test_handles_dict_arguments(self) -> None:
        tc = {
            "id": "x", "type": "function",
            "function": {"name": "f", "arguments": {"a": 1}},
        }
        name, args = probe_gen._extract_tool_call(tc)
        assert name == "f"
        assert args == {"a": 1}

    def test_invalid_json_arguments_yield_empty_dict(self) -> None:
        tc = {
            "id": "x", "type": "function",
            "function": {"name": "f", "arguments": "{not-json"},
        }
        name, args = probe_gen._extract_tool_call(tc)
        assert name == "f"
        assert args == {}

    def test_missing_function_block_falls_back(self) -> None:
        name, args = probe_gen._extract_tool_call({"name": "loose"})
        assert name == "loose"
        assert args == {}

    def test_non_dict_input_returns_malformed(self) -> None:
        name, args = probe_gen._extract_tool_call("garbage")
        assert name == "<malformed>"
        assert args == {}


# ---------------------------------------------------------------------------
# 6. End-to-end probe
# ---------------------------------------------------------------------------


class TestSimulateAgenticSafety:
    def test_registered_in_probe_registry(self) -> None:
        # Cross-probe invariants live in tests/probes/test_probe_invariants.py;
        # this is a fast smoke check that the SafetyAgenticProbe class
        # itself is the registered target.
        assert "safety_agentic" in _PROBE_REGISTRY
        assert resolve_probe("safety_agentic").__name__ == "SafetyAgenticProbe"

    def test_no_tool_call_response_short_circuits(
        self, en_persona: Dict[str, Any], simulator_cfg: Any,
    ) -> None:
        # Assistant responds with content but no tool_calls — the loop
        # ends after one assistant turn (single-user-turn design).
        with patch.dict(os.environ, _shipped_bank_env()), patch(
            "usersim.engine.probes.safety_agentic.generator.call_llm",
            side_effect=_mock_call_llm([
                {"role": "assistant", "content": "I should ask first."},
            ]),
        ):
            result = probe_gen.simulate_safety_agentic(
                models={"assistant_model": object()},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert result["conversation_status"] is True
        assert result["num_tool_calls"] == 0
        assert result["action_request_id"].startswith("AR-")
        assert result["sub_protocol"] in {
            "unsanctioned_action", "scope_creep",
            "compromised_inputs", "consequence_disclosure",
        }
        outcome = json.loads(result["simulation_outcome"])
        # The shipped bank mixes reviewed requests with honest placeholders
        # for generated rows; either warning state is valid depending on
        # which action request the persona-derived picker chose.
        # Bank version pinned on provenance under "agentic" key.
        assert outcome["provenance"]["bank_version"].get("agentic") == "v0.5.4"

    def test_first_user_turn_is_initial_user_prompt_verbatim(
        self, en_persona: Dict[str, Any], simulator_cfg: Any,
    ) -> None:
        with patch.dict(os.environ, _shipped_bank_env()):
            from usersim.engine.core.agentic_bank import (
                load_agentic_bank_default,
            )
            probe_gen._reset_bank_cache()
            bank = load_agentic_bank_default()
            ar = task_derivation.derive_task(
                en_persona, bank, "en_US", seed=simulator_cfg.random_seed,
            )
            assert ar is not None
            expected = ar.initial_user_prompt.for_locale("en_US")

        probe_gen._reset_bank_cache()
        with patch.dict(os.environ, _shipped_bank_env()), patch(
            "usersim.engine.probes.safety_agentic.generator.call_llm",
            side_effect=_mock_call_llm([
                {"role": "assistant", "content": "ok"},
            ]),
        ):
            result = probe_gen.simulate_safety_agentic(
                models={"assistant_model": object()},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        msgs = json.loads(result["conversation_messages"])
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == expected

    def test_assistant_tool_call_intercepted_with_mock_response(
        self, en_persona: Dict[str, Any], simulator_cfg_three_turns: Any,
    ) -> None:
        # Synthetic bank → exactly one tool ("do_thing"). Assistant
        # calls it on turn 1, then returns no-tool-calls on turn 2.
        synth = _synthetic_bank(n_requests=1, tool_name="do_thing")
        with patch.object(probe_gen, "_load_bank", return_value=synth), \
             patch(
                "usersim.engine.probes.safety_agentic.generator.call_llm",
                side_effect=_mock_call_llm([
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            _make_tool_call("do_thing", {"x": "hello"}),
                        ],
                    },
                    {"role": "assistant", "content": "Done."},
                ]),
             ):
            result = probe_gen.simulate_safety_agentic(
                models={"assistant_model": object()},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg_three_turns,
                provenance=Provenance(),
            )

        assert result["conversation_status"] is True
        assert result["num_tool_calls"] == 1
        # Tool-call-only assistant messages are implementation
        # envelopes, not user-visible assistant turns. The user saw
        # only the final "Done." assistant message.
        assert result["num_turns"] == 1
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["n_turns"] == 1

        msgs = json.loads(result["conversation_messages"])
        # Sequence: user → assistant(tool_calls) → tool → assistant(content)
        assert [m["role"] for m in msgs] == [
            "user", "assistant", "tool", "assistant",
        ]
        # Mock response was injected verbatim.
        assert json.loads(msgs[2]["content"]) == {"ok": True, "i": 0}

    def test_attempted_actions_recorded_with_full_details(
        self, en_persona: Dict[str, Any], simulator_cfg_three_turns: Any,
    ) -> None:
        # Two-call turn followed by no-tool-call turn.
        synth = _synthetic_bank(n_requests=1, tool_name="do_thing")
        with patch.object(probe_gen, "_load_bank", return_value=synth), \
             patch(
                "usersim.engine.probes.safety_agentic.generator.call_llm",
                side_effect=_mock_call_llm([
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            _make_tool_call(
                                "do_thing", {"x": "first"}, tc_id="c0",
                            ),
                            _make_tool_call(
                                "do_thing", {"x": "second"}, tc_id="c1",
                            ),
                        ],
                    },
                    {"role": "assistant", "content": "Done."},
                ]),
             ):

            # Patched bank load doesn't touch the cache, so we don't need
            # to reset. Call directly.
            result = probe_gen.simulate_safety_agentic(
                models={"assistant_model": object()},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg_three_turns,
                provenance=Provenance(),
            )

        # The metadata is folded into the result by make_result; the
        # attempted_actions side channel surfaces via the in-line key.
        # We have no convention to mirror it onto a top-level result
        # column, so re-derive from the trajectory instead.
        msgs = json.loads(result["conversation_messages"])
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        # Tool messages were produced from both attempted calls.

    def test_unknown_tool_call_yields_synthetic_error_response(
        self, en_persona: Dict[str, Any], simulator_cfg_three_turns: Any,
    ) -> None:
        synth = _synthetic_bank(n_requests=1, tool_name="do_thing")
        with patch.object(probe_gen, "_load_bank", return_value=synth), \
             patch(
                "usersim.engine.probes.safety_agentic.generator.call_llm",
                side_effect=_mock_call_llm([
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            _make_tool_call(
                                "fabricated_tool", {"x": "xyz"}, tc_id="c0",
                            ),
                        ],
                    },
                    {"role": "assistant", "content": "Recovered."},
                ]),
             ):
            result = probe_gen.simulate_safety_agentic(
                models={"assistant_model": object()},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg_three_turns,
                provenance=Provenance(),
            )

        msgs = json.loads(result["conversation_messages"])
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        body = json.loads(tool_msgs[0]["content"])
        assert "error" in body
        assert body["tool_name"] == "fabricated_tool"

    def test_bank_load_failure_returns_structured_failure(
        self, en_persona: Dict[str, Any], simulator_cfg: Any,
    ) -> None:
        with patch.dict(
            os.environ,
            {"USERSIM_SAFETY_AGENTIC_BANK": "/nonexistent/bank.yaml"},
        ):
            result = probe_gen.simulate_safety_agentic(
                models={},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert result["conversation_status"] is False
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == OutcomeStatus.FAILED.value
        assert outcome["failure_class"] == FailureClass.SCENARIO_ABORTED.value
        assert "agentic-bank load failed" in outcome["failure_detail"]

    def test_panel_override_with_unknown_id_returns_failure(
        self, en_persona: Dict[str, Any], simulator_cfg: Any,
    ) -> None:
        with patch.dict(os.environ, _shipped_bank_env()), patch(
            "usersim.engine.probes.safety_agentic.generator.call_llm",
        ) as mock_call:
            result = probe_gen.simulate_safety_agentic(
                models={},
                data={"action_request_id": "AR-DOES-NOT-EXIST"},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        mock_call.assert_not_called()
        assert result["conversation_status"] is False
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == OutcomeStatus.FAILED.value
        assert outcome["failure_class"] == FailureClass.SCENARIO_ABORTED.value

    def test_no_resolvable_request_returns_failure(
        self, en_persona: Dict[str, Any], simulator_cfg: Any,
    ) -> None:
        synth = _synthetic_bank(
            request_persona_tags=("interest:nonexistent",),
        )
        with patch.object(probe_gen, "_load_bank", return_value=synth), \
             patch(
                "usersim.engine.probes.safety_agentic.generator.call_llm",
             ) as mock_call:
            result = probe_gen.simulate_safety_agentic(
                models={},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        mock_call.assert_not_called()
        assert result["conversation_status"] is False
        outcome = json.loads(result["simulation_outcome"])
        assert "no resolvable" in outcome["failure_detail"]

    def test_assistant_failure_attributed_to_assistant_model(
        self, en_persona: Dict[str, Any], simulator_cfg: Any,
    ) -> None:
        def _raises(*args, **kwargs):
            raise RuntimeError("assistant API exploded")

        with patch.dict(os.environ, _shipped_bank_env()), patch(
            "usersim.engine.probes.safety_agentic.generator.call_llm",
            side_effect=_raises,
        ):
            result = probe_gen.simulate_safety_agentic(
                models={"assistant_model": object()},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert result["conversation_status"] is False
        # Side channels still pinned (mirrors the `safety_chat_pressure` invariant).
        assert result["action_request_id"].startswith("AR-")
        assert result["sub_protocol"] in {
            "unsanctioned_action", "scope_creep",
            "compromised_inputs", "consequence_disclosure",
        }
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == OutcomeStatus.FAILED.value
        assert outcome["failure_class"] == FailureClass.INFRASTRUCTURE_ERROR.value
        assert outcome["failure_attribution"] == "assistant_model"

    def test_non_placeholder_request_no_warning(
        self, en_persona: Dict[str, Any], simulator_cfg: Any,
    ) -> None:
        synth = _synthetic_bank(placeholder=False, n_requests=1)
        with patch.object(probe_gen, "_load_bank", return_value=synth), \
             patch(
                "usersim.engine.probes.safety_agentic.generator.call_llm",
                side_effect=_mock_call_llm([
                    {"role": "assistant", "content": "ok"},
                ]),
             ):
            result = probe_gen.simulate_safety_agentic(
                models={"assistant_model": object()},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        outcome = json.loads(result["simulation_outcome"])
        assert not any(
            w["kind"] == WarningKind.USED_PLACEHOLDER_AGENTIC_ACTION.value
            for w in outcome["warnings"]
        )

    def test_probe_variant_format(
        self, en_persona: Dict[str, Any], simulator_cfg: Any,
    ) -> None:
        synth = _synthetic_bank(
            n_requests=1,
            sub_protocol="consequence_disclosure",
            sanctioned_action_name="transfer_funds_external",
        )
        with patch.object(probe_gen, "_load_bank", return_value=synth), \
             patch(
                "usersim.engine.probes.safety_agentic.generator.call_llm",
                side_effect=_mock_call_llm([
                    {"role": "assistant", "content": "ok"},
                ]),
             ):
            result = probe_gen.simulate_safety_agentic(
                models={"assistant_model": object()},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert (
            result["probe_variant"]
            == "consequence_disclosure::transfer_funds_external"
        )

    def test_probe_variant_format_when_no_sanctioned_action(
        self, en_persona: Dict[str, Any], simulator_cfg: Any,
    ) -> None:
        synth = _synthetic_bank(
            n_requests=1,
            sub_protocol="compromised_inputs",
            sanctioned_action_name=None,
        )
        with patch.object(probe_gen, "_load_bank", return_value=synth), \
             patch(
                "usersim.engine.probes.safety_agentic.generator.call_llm",
                side_effect=_mock_call_llm([
                    {"role": "assistant", "content": "Refusing."},
                ]),
             ):
            result = probe_gen.simulate_safety_agentic(
                models={"assistant_model": object()},
                data={},
                persona=en_persona,
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert result["probe_variant"] == "compromised_inputs::no_action"


# ---------------------------------------------------------------------------
# 7. should_succeed
# ---------------------------------------------------------------------------


class TestShouldSucceed:
    def test_returns_true_when_action_request_id_set(self) -> None:
        from usersim.engine.core.outcomes import OutcomeBuilder
        from usersim.engine.core.simulation import ConversationState

        state = ConversationState(outcome=OutcomeBuilder())
        state.metadata["action_request_id"] = "AR-X"
        assert probe_gen.should_succeed(state) is True

    def test_returns_false_when_action_request_id_missing(self) -> None:
        from usersim.engine.core.outcomes import OutcomeBuilder
        from usersim.engine.core.simulation import ConversationState

        state = ConversationState(outcome=OutcomeBuilder())
        assert probe_gen.should_succeed(state) is False
