# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator/generator.py — TrajectoryEvaluatorGenerator.

Focus: the parts that don't require a live DD model registry —
JSON field decoding, envelope construction, partial-re-run skip,
short-circuit on missing-assistant-content, scorer-error trapping.

Full end-to-end with a real DD pipeline lands alongside the
synthetic-training-extraction example.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from usersim.engine.evaluator import scorers as scorers_module
from usersim.engine.evaluator.config import (
    JudgeSpecConfig,
    TrajectoryEvaluatorConfig,
)
from usersim.engine.evaluator.generator import (
    EVALUATOR_VERSION,
    TrajectoryEvaluatorGenerator,
    _decode_json_field,
)

# ── Pure helpers ────────────────────────────────────────────────────


class TestDecodeJsonField:
    def test_none_returns_default(self) -> None:
        assert _decode_json_field(None, default=[]) == []
        assert _decode_json_field(None, default={"k": "v"}) == {"k": "v"}

    def test_dict_passthrough(self) -> None:
        d = {"a": 1}
        assert _decode_json_field(d, default={}) is d

    def test_list_passthrough(self) -> None:
        lst = [{"role": "user"}]
        assert _decode_json_field(lst, default=[]) is lst

    def test_json_string(self) -> None:
        assert _decode_json_field('{"a": 1}', default={}) == {"a": 1}
        assert _decode_json_field("[1,2,3]", default=[]) == [1, 2, 3]

    def test_invalid_json_returns_default(self) -> None:
        assert _decode_json_field("not json", default={"fallback": True}) == {"fallback": True}

    def test_empty_string_returns_default(self) -> None:
        assert _decode_json_field("", default=[]) == []
        assert _decode_json_field("   ", default=[]) == []


# ── Generator construction helper ──────────────────────────────────


def _build_generator(
    cfg: TrajectoryEvaluatorConfig,
    judge_responses: dict[str, dict[str, Any]] | None = None,
) -> TrajectoryEvaluatorGenerator:
    """Construct a generator without DD's full ResourceProvider machinery.

    We bypass __init__ (which requires a live ResourceProvider) and
    install just the bits the generator's `generate()` actually reads:
    `_config`, plus stubbed `get_model` / `get_models` / `_call_judge`.
    """
    gen = TrajectoryEvaluatorGenerator.__new__(TrajectoryEvaluatorGenerator)
    gen._config = cfg
    gen._resource_provider = MagicMock()

    # Stub get_model: returns a sentinel facade per alias.
    facades = {j.alias: MagicMock(name=f"facade_{j.alias}") for j in cfg.judges}
    gen.get_model = lambda alias: facades.get(alias)  # type: ignore[method-assign]
    gen.get_models = lambda: facades  # type: ignore[method-assign]

    # Stub _call_judge with canned responses keyed by alias.
    responses = judge_responses or {}

    def _stub_call_judge(models, judge_alias, prompt, schema_model, max_tokens):
        return responses.get(judge_alias, {})

    gen._call_judge = _stub_call_judge  # type: ignore[method-assign]
    return gen


def _ok_judges() -> list[JudgeSpecConfig]:
    return [
        JudgeSpecConfig(alias="judge_a", family="openai"),
        JudgeSpecConfig(alias="judge_b", family="nvidia_nemotron"),
    ]


# ── Behavioral tests ───────────────────────────────────────────────


class TestNoAssistantMessages:
    def test_short_circuits_with_skipped_record(self) -> None:
        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=_ok_judges(),
            axes=["helpfulness"],
        )
        gen = _build_generator(cfg)
        # Trajectory with only user messages (the simulator's
        # "infrastructure failure" case).
        data = {
            "conversation_messages": json.dumps([{"role": "user", "content": "hi"}]),
            "persona": json.dumps({"first_name": "A", "last_name": "B"}),
            "probe_family": "general_open_ended",
        }
        out = gen.generate(data)
        assert "eval_v2" in out
        parsed = json.loads(out["eval_v2"])
        assert parsed["skipped"] is True
        assert parsed["skipped_reason"] == "no_assistant_messages"
        assert parsed["axes"] == {}
        assert parsed["envelope"]["evaluator_version"] == EVALUATOR_VERSION


class TestFailedSimulation:
    def test_partial_failed_trajectory_is_not_quality_scored(self) -> None:
        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=_ok_judges(),
            axes=["helpfulness"],
        )
        gen = _build_generator(cfg)
        gen._call_judge = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError("failed trajectory must not be judged")
        )
        data = {
            "conversation_messages": json.dumps(
                [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "c1", "function": {"name": "lookup"}}],
                    },
                    {"role": "tool", "content": '{"partial":true}'},
                ]
            ),
            "simulation_outcome": json.dumps({"status": "failed"}),
            "persona": json.dumps({"first_name": "A", "last_name": "B"}),
            "probe_family": "tool_calling",
        }

        parsed = json.loads(gen.generate(data)["eval_v2"])

        assert parsed["skipped"] is True
        assert parsed["skipped_reason"] == "simulation_failed"
        assert parsed["axes"] == {}
        gen._call_judge.assert_not_called()


class TestPartialReRunSkip:
    def _trajectory_with_assistant(self) -> dict:
        return {
            "conversation_messages": json.dumps(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            ),
            "persona": json.dumps({"first_name": "A", "last_name": "B"}),
            "probe_family": "general_open_ended",
            "locale": "en_US",
            "conversation_language": "English",
        }

    def test_matching_envelope_skips_judges(self) -> None:
        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=_ok_judges(),
            axes=["helpfulness"],
            prompt_version="v1.0",
        )
        gen = _build_generator(
            cfg,
            judge_responses={
                # If the skip works, these should NOT be consumed.
                "judge_a": {"helpfulness": {"score": 5, "reasoning": "fresh"}},
                "judge_b": {"helpfulness": {"score": 4, "reasoning": "fresh"}},
            },
        )
        # Pre-populate the row with an "existing" cell that matches the
        # current envelope.
        existing = {
            "envelope": {
                "judge_aliases": ["judge_a", "judge_b"],
                "judge_families": ["openai", "nvidia_nemotron"],
                "axes": ["helpfulness"],
                "scorers": [],
                "prompt_version": "v1.0",
                "evaluator_version": EVALUATOR_VERSION,
            },
            "axes": {"helpfulness": {"judge_a": {"score": 3, "reasoning": "old"}}},
            "scorers": {},
            "skipped": False,
            "skipped_reason": None,
        }
        data = self._trajectory_with_assistant()
        data["eval_v2"] = json.dumps(existing)
        out = gen.generate(data)
        parsed = json.loads(out["eval_v2"])
        # The old reasoning survives — judges were not re-run.
        assert parsed["axes"]["helpfulness"]["judge_a"]["reasoning"] == "old"
        # Skipped marker is set.
        assert parsed["skipped"] is True
        assert parsed["skipped_reason"] == "envelope_match"

    def test_changed_prompt_version_invalidates(self) -> None:
        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=_ok_judges(),
            axes=["helpfulness"],
            prompt_version="v1.1",  # bumped
        )
        gen = _build_generator(
            cfg,
            judge_responses={
                "judge_a": {"helpfulness": {"score": 5, "reasoning": "fresh"}},
                "judge_b": {"helpfulness": {"score": 4, "reasoning": "fresh"}},
            },
        )
        existing = {
            "envelope": {
                "judge_aliases": ["judge_a", "judge_b"],
                "judge_families": ["openai", "nvidia_nemotron"],
                "axes": ["helpfulness"],
                "scorers": [],
                "prompt_version": "v1.0",  # old
                "evaluator_version": EVALUATOR_VERSION,
            },
            "axes": {"helpfulness": {"judge_a": {"score": 3, "reasoning": "old"}}},
            "scorers": {},
            "skipped": False,
            "skipped_reason": None,
        }
        data = self._trajectory_with_assistant()
        data["eval_v2"] = json.dumps(existing)
        out = gen.generate(data)
        parsed = json.loads(out["eval_v2"])
        # Fresh judging happened.
        assert parsed["skipped"] is False
        assert parsed["axes"]["helpfulness"]["judge_a"]["reasoning"] == "fresh"
        assert parsed["envelope"]["prompt_version"] == "v1.1"

    def test_skip_if_existing_false_forces_rerun(self) -> None:
        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=_ok_judges(),
            axes=["helpfulness"],
            skip_if_existing=False,  # force
        )
        gen = _build_generator(
            cfg,
            judge_responses={
                "judge_a": {"helpfulness": {"score": 5, "reasoning": "fresh"}},
                "judge_b": {"helpfulness": {"score": 4, "reasoning": "fresh"}},
            },
        )
        existing = {
            "envelope": {
                "judge_aliases": ["judge_a", "judge_b"],
                "judge_families": ["openai", "nvidia_nemotron"],
                "axes": ["helpfulness"],
                "scorers": [],
                "prompt_version": "v1.0",
                "evaluator_version": EVALUATOR_VERSION,
            },
            "axes": {"helpfulness": {"judge_a": {"score": 3, "reasoning": "old"}}},
            "scorers": {},
            "skipped": False,
            "skipped_reason": None,
        }
        data = self._trajectory_with_assistant()
        data["eval_v2"] = json.dumps(existing)
        out = gen.generate(data)
        parsed = json.loads(out["eval_v2"])
        # Fresh judges ran despite envelope match.
        assert parsed["skipped"] is False
        assert parsed["axes"]["helpfulness"]["judge_a"]["reasoning"] == "fresh"


class TestScorerDispatch:
    # Snapshot and restore rather than leaving the registry empty: it is
    # process-global, so a bare clear() in teardown makes every later test in
    # the same worker see zero scorers. Under `pytest -n auto` that surfaces
    # only when the affected files land on the same worker, which is why it
    # read as a flake. Mirrors TestRegisterProbe in test_probes_substrate.py.
    def setup_method(self) -> None:
        self._snapshot = dict(scorers_module._REGISTRY)
        scorers_module.clear_registry()

    def teardown_method(self) -> None:
        scorers_module.clear_registry()
        scorers_module._REGISTRY.update(self._snapshot)

    def test_unknown_scorer_recorded_as_error_not_raised(self) -> None:
        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=_ok_judges(),
            axes=["helpfulness"],
            scorers=["does_not_exist"],
        )
        gen = _build_generator(
            cfg,
            judge_responses={
                "judge_a": {"helpfulness": {"score": 4, "reasoning": "ok"}},
                "judge_b": {"helpfulness": {"score": 4, "reasoning": "ok"}},
            },
        )
        data = {
            "conversation_messages": json.dumps(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            ),
            "persona": json.dumps({"first_name": "A", "last_name": "B"}),
            "probe_family": "general_open_ended",
            "locale": "en_US",
            "conversation_language": "English",
        }
        out = gen.generate(data)
        parsed = json.loads(out["eval_v2"])
        assert "does_not_exist" in parsed["scorers"]
        assert "error" in parsed["scorers"]["does_not_exist"]

    def test_registered_scorer_runs(self) -> None:
        captured: list[dict] = []

        def my_scorer(traj: dict, models: dict) -> dict:
            captured.append(traj)
            return {"my_metric": 0.42}

        scorers_module.register_scorer("my_scorer", my_scorer)

        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=_ok_judges(),
            axes=["helpfulness"],
            scorers=["my_scorer"],
        )
        gen = _build_generator(
            cfg,
            judge_responses={
                "judge_a": {"helpfulness": {"score": 4, "reasoning": "ok"}},
                "judge_b": {"helpfulness": {"score": 4, "reasoning": "ok"}},
            },
        )
        data = {
            "conversation_messages": json.dumps(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            ),
            "persona": json.dumps({"first_name": "A", "last_name": "B"}),
            "probe_family": "general_open_ended",
            "locale": "en_US",
            "conversation_language": "English",
        }
        out = gen.generate(data)
        parsed = json.loads(out["eval_v2"])
        assert parsed["scorers"]["my_scorer"] == {"my_metric": 0.42}
        # The scorer received a deserialized trajectory row.
        assert len(captured) == 1
        assert captured[0]["probe_family"] == "general_open_ended"
        assert isinstance(captured[0]["conversation_messages"], list)


class TestScorerColumnPassthrough:
    """Pin the evaluator → scorer column-passthrough contract.

    Production scorers in evaluator/scorers/ read probe-specific
    top-level keys directly off the trajectory dict
    (``sovereign_facts_probed`` / ``query_id`` /
    ``target_request_id`` / ``strategy_id`` / ``reframings_used`` /
    ``action_request_id`` / ``sub_protocol`` / ``attempted_actions`` /
    ``probing_categories_explored`` / ``probing_subtopic_hints_used`` /
    ``tool_subset``). For years the generator built ``traj_row`` from a
    literal 7-key projection that silently dropped these — every
    probe-specific scorer returned its no-op skip envelope in
    production. The current projection is a pass-through merge
    ``traj_row = {**data, **traj_row}``; this test class guards
    against regression to the literal-keys form.
    """

    PROBE_SPECIFIC_KEYS = (
        "sovereign_facts_probed",
        "query_id",
        "target_request_id",
        "strategy_id",
        "reframings_used",
        "action_request_id",
        "sub_protocol",
        "attempted_actions",
        "probing_categories_explored",
        "probing_subtopic_hints_used",
        "tool_subset",
    )

    def setup_method(self) -> None:
        self._snapshot = dict(scorers_module._REGISTRY)
        scorers_module.clear_registry()

    def teardown_method(self) -> None:
        scorers_module.clear_registry()
        scorers_module._REGISTRY.update(self._snapshot)

    def _make_data_with_probe_columns(self) -> dict:
        data = {
            "conversation_messages": json.dumps(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            ),
            "persona": json.dumps({"first_name": "A", "last_name": "B"}),
            "probe_family": "inclusion",
            "probe_variant": "default",
            "simulation_outcome": json.dumps({"status": "ok", "metadata": {}}),
            "locale": "en_US",
            "conversation_language": "English",
            "trajectory_id": "TID-PASSTHROUGH-001",
        }
        for key in self.PROBE_SPECIFIC_KEYS:
            data[key] = f"<probe-specific:{key}>"
        return data

    def test_all_probe_specific_columns_reach_scorer(self) -> None:
        seen: dict[str, dict] = {}

        def spy(traj: dict, models: dict) -> dict:
            seen["traj_row"] = traj
            return {"spy": True}

        scorers_module.register_scorer("__spy__", spy)

        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=_ok_judges(),
            axes=["helpfulness"],
            scorers=["__spy__"],
        )
        gen = _build_generator(
            cfg,
            judge_responses={
                "judge_a": {"helpfulness": {"score": 4, "reasoning": "ok"}},
                "judge_b": {"helpfulness": {"score": 4, "reasoning": "ok"}},
            },
        )
        gen.generate(self._make_data_with_probe_columns())

        traj_row = seen["traj_row"]
        for key in self.PROBE_SPECIFIC_KEYS:
            assert key in traj_row, (
                f"probe-specific key {key!r} missing from traj_row — "
                "the evaluator/scorer pass-through is broken — "
                "TrajectoryEvaluatorGenerator must merge the full "
                "row dict into traj_row, not project a fixed set of keys"
            )
            assert traj_row[key] == f"<probe-specific:{key}>"

    def test_explicit_projection_wins_over_passthrough(self) -> None:
        """Decoded ``conversation_messages`` / ``persona`` / etc. take precedence.

        The pass-through merge spreads ``data`` first, then re-applies
        the explicit projection — so the scorer sees ``messages`` as a
        deserialized list (not the raw JSON string) and ``persona`` as
        a deserialized dict. Same for ``simulation_outcome`` (decoded)
        and ``locale`` / ``language`` (defaults backfilled).
        """
        seen: dict[str, dict] = {}

        def spy(traj: dict, models: dict) -> dict:
            seen["traj_row"] = traj
            return {"spy": True}

        scorers_module.register_scorer("__spy__", spy)

        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=_ok_judges(),
            axes=["helpfulness"],
            scorers=["__spy__"],
        )
        gen = _build_generator(
            cfg,
            judge_responses={
                "judge_a": {"helpfulness": {"score": 4, "reasoning": "ok"}},
                "judge_b": {"helpfulness": {"score": 4, "reasoning": "ok"}},
            },
        )
        gen.generate(self._make_data_with_probe_columns())

        traj_row = seen["traj_row"]
        # Decoded forms, not raw JSON strings.
        assert isinstance(traj_row["conversation_messages"], list)
        assert isinstance(traj_row["persona"], dict)
        assert isinstance(traj_row["simulation_outcome"], dict)
        # Resolved defaults.
        assert traj_row["language"] == "English"
        assert traj_row["locale"] == "en_US"

    def test_unknown_keys_passed_through_unchanged(self) -> None:
        """A future probe's new side-channel column reaches scorers without code change.

        The whole point of pass-through merge over a static allow-list:
        new probes add new keys, the scorer reads them, no evaluator
        plumbing required.
        """
        seen: dict[str, dict] = {}

        def spy(traj: dict, models: dict) -> dict:
            seen["traj_row"] = traj
            return {"spy": True}

        scorers_module.register_scorer("__spy__", spy)

        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=_ok_judges(),
            axes=["helpfulness"],
            scorers=["__spy__"],
        )
        gen = _build_generator(
            cfg,
            judge_responses={
                "judge_a": {"helpfulness": {"score": 4, "reasoning": "ok"}},
                "judge_b": {"helpfulness": {"score": 4, "reasoning": "ok"}},
            },
        )

        data = self._make_data_with_probe_columns()
        data["future_probe_side_channel"] = ["a", "b", "c"]
        data["another_brand_new_column"] = {"nested": True}

        gen.generate(data)

        traj_row = seen["traj_row"]
        assert traj_row["future_probe_side_channel"] == ["a", "b", "c"]
        assert traj_row["another_brand_new_column"] == {"nested": True}


class TestEnvelope:
    def test_envelope_includes_resolved_families(self) -> None:
        cfg = TrajectoryEvaluatorConfig(
            name="eval_v2",
            judges=[
                JudgeSpecConfig(alias="judge_a", family="openai"),
                JudgeSpecConfig(alias="judge_b", family="nvidia_nemotron"),
            ],
            axes=["helpfulness"],
        )
        gen = _build_generator(
            cfg,
            judge_responses={
                "judge_a": {"helpfulness": {"score": 5, "reasoning": "x"}},
                "judge_b": {"helpfulness": {"score": 4, "reasoning": "y"}},
            },
        )
        data = {
            "conversation_messages": json.dumps(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            ),
            "persona": json.dumps({"first_name": "A", "last_name": "B"}),
            "probe_family": "general_open_ended",
            "locale": "en_US",
            "conversation_language": "English",
        }
        out = gen.generate(data)
        parsed = json.loads(out["eval_v2"])
        env = parsed["envelope"]
        assert env["judge_aliases"] == ["judge_a", "judge_b"]
        assert env["judge_families"] == ["openai", "nvidia_nemotron"]
        assert env["axes"] == ["helpfulness"]
        assert env["evaluator_version"] == EVALUATOR_VERSION
