# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/identity.py — content-hashed persona/trajectory identity.

Acceptance points covered here:
- ``persona_uuid`` is stable across runs given identical content.
- ``persona_uuid`` is invariant to insignificant changes (key order;
  whitespace; OCEAN-as-string vs OCEAN-as-dict serialization quirks
  across locales).
- ``persona_uuid`` *changes* when a hash-relevant field changes.
- Adding a brand-new persona attribute that is not in the canonical
  key list does NOT change a persona's UUID — protecting future
  Nemotron-Personas releases from retroactively altering identity.
- ``trajectory_id`` is deterministic given identical inputs.
- ``trajectory_id`` discriminates on every input it claims to depend
  on (persona, probe, variant, seed, models, prompt version).
- ``resolve_model_name`` falls back to alias when no model identity
  is exposed; honours the documented attribute paths when present.
"""

from __future__ import annotations

import json

import pytest

from usersim.engine.core.identity import (
    canonical_persona_json,
    persona_uuid,
    resolve_model_name,
    trajectory_id,
)


# ── Persona UUID ────────────────────────────────────────────────────────


class TestPersonaUuid:
    def test_stable_across_calls(self, full_persona: dict) -> None:
        a = persona_uuid(full_persona)
        b = persona_uuid(full_persona)
        assert a == b
        assert isinstance(a, str)
        assert len(a) == 16

    def test_invariant_to_key_order(self, full_persona: dict) -> None:
        # dict literal order should never affect the canonical hash.
        reordered = {k: full_persona[k] for k in reversed(list(full_persona.keys()))}
        assert persona_uuid(full_persona) == persona_uuid(reordered)

    def test_invariant_to_irrelevant_extra_keys(self, full_persona: dict) -> None:
        # Adding a key that is NOT in the canonical key set must not
        # change the UUID. This is the guarantee that future
        # Nemotron-Personas extensions can land without invalidating
        # every prior trajectory's identity.
        with_extra = dict(full_persona)
        with_extra["__internal_dd_id"] = "transient-1234"
        with_extra["nemotron_release_tag"] = "2026-04"
        assert persona_uuid(full_persona) == persona_uuid(with_extra)

    def test_changes_when_relevant_field_changes(self, full_persona: dict) -> None:
        edited = dict(full_persona)
        edited["age"] = full_persona["age"] + 1
        assert persona_uuid(full_persona) != persona_uuid(edited)

    @pytest.mark.parametrize(
        "field,first,second",
        [
            ("religion", "Hindu", "Muslim"),
            ("first_language", "Hindi", "Malayalam"),
            ("second_language", "English", "Hindi"),
            ("third_language", "-", "English"),
            (
                "religious_background",
                "Observes major Hindu festivals",
                "Observes major Muslim festivals",
            ),
            (
                "linguistic_background",
                "Hindi at home, English at work",
                "Malayalam at home, English at work",
            ),
        ],
    )
    def test_prompt_visible_india_fields_change_identity(
        self,
        field: str,
        first: str,
        second: str,
    ) -> None:
        """Every field rendered into the user-agent prompt is identity-bearing."""
        base = {"first_name": "Ravi", "last_name": "Kumar", field: first}
        edited = {**base, field: second}
        assert persona_uuid(base) != persona_uuid(edited)

    def test_handles_ocean_as_dict_or_string(self, full_persona: dict) -> None:
        # In en_US the OCEAN trait values arrive as floats; in other
        # locales they arrive as JSON-encoded strings of dicts. The
        # canonical form normalizes JSON-string look-alikes so the
        # same logical persona hashes the same regardless of
        # representation. Float vs dict legitimately differ — only
        # equivalent JSON-string vs dict should match.
        as_dict = dict(full_persona)
        as_dict["openness"] = {"label": "high", "t_score": 60}
        as_str = dict(full_persona)
        as_str["openness"] = json.dumps({"label": "high", "t_score": 60})
        # Both representations should hash to the same UUID.
        assert persona_uuid(as_dict) == persona_uuid(as_str)
        # And different from the original (which has a float, not a
        # dict — these are intentionally different representations).
        assert persona_uuid(as_dict) != persona_uuid(full_persona)

    def test_two_distinct_personas_collide_only_on_match(self, full_persona: dict, minimal_persona: dict) -> None:
        assert persona_uuid(full_persona) != persona_uuid(minimal_persona)

    def test_canonical_json_is_sorted_and_stable(self, full_persona: dict) -> None:
        # The canonical JSON itself should be deterministic, not just
        # the hash. Useful for debugging UUID divergence.
        a = canonical_persona_json(full_persona)
        b = canonical_persona_json({k: full_persona[k] for k in reversed(list(full_persona))})
        assert a == b
        assert json.loads(a)  # valid JSON


# ── Trajectory ID ───────────────────────────────────────────────────────


class TestTrajectoryId:
    BASE = dict(
        persona_uuid="abcdef1234567890",
        probe_family="general_open_ended",
        probe_variant="default",
        scenario_seed=None,
        user_model="user_model:openai/gpt-oss-120b",
        assistant_model="assistant_model:nvidia/nemotron-3-super-120b-a12b",
        prompt_version="v1.0",
    )

    def test_stable_across_calls(self) -> None:
        a = trajectory_id(**self.BASE)
        b = trajectory_id(**self.BASE)
        assert a == b
        assert isinstance(a, str)
        assert len(a) == 16

    @pytest.mark.parametrize(
        "field,new_value",
        [
            ("persona_uuid", "different-persona"),
            ("probe_family", "tool_calling"),
            ("probe_variant", "another_variant"),
            ("scenario_seed", 42),
            ("user_model", "user_model:other-model"),
            ("assistant_model", "assistant_model:other-model"),
            ("prompt_version", "v1.1"),
        ],
    )
    def test_changes_when_any_input_changes(self, field: str, new_value) -> None:
        baseline = trajectory_id(**self.BASE)
        diff = dict(self.BASE)
        diff[field] = new_value
        assert trajectory_id(**diff) != baseline, f"trajectory_id failed to discriminate on {field}"

    def test_extra_keys_discriminate(self) -> None:
        baseline = trajectory_id(**self.BASE)
        with_extra = trajectory_id(
            **self.BASE,
            extra_keys=[("pressure_strategy_seed", "topic_drift")],
        )
        assert baseline != with_extra
        # Same extra_keys -> same id.
        repeat = trajectory_id(
            **self.BASE,
            extra_keys=[("pressure_strategy_seed", "topic_drift")],
        )
        assert with_extra == repeat


# ── Model name resolution ──────────────────────────────────────────────


class _FacadeWithModelConfig:
    class _Cfg:
        model = "openai/gpt-oss-120b"

    model_config = _Cfg()


class _FacadeWithModelName:
    model_name = "anthropic/claude-3.5"


class _FacadeWithConfigName:
    class _Cfg:
        name = "qwen/qwen3-72b"

    config = _Cfg()


class _OpaqueFacade:
    pass


class TestResolveModelName:
    def test_facade_with_model_config_dot_model(self) -> None:
        assert resolve_model_name(_FacadeWithModelConfig(), "user_model") == ("openai/gpt-oss-120b")

    def test_facade_with_model_name(self) -> None:
        assert resolve_model_name(_FacadeWithModelName(), "judge_model") == ("anthropic/claude-3.5")

    def test_facade_with_config_dot_name(self) -> None:
        assert resolve_model_name(_FacadeWithConfigName(), "judge_model") == ("qwen/qwen3-72b")

    def test_opaque_facade_falls_back_to_alias(self) -> None:
        assert resolve_model_name(_OpaqueFacade(), "user_model") == "user_model"

    def test_none_facade_falls_back_to_alias(self) -> None:
        assert resolve_model_name(None, "assistant_model") == "assistant_model"
