# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""The identity spec loader: resolution, layering, merging, validation.

Specs here live in temporary asset roots under a fixture label, so the tests
exercise the same search-path lookups a run uses without touching the
packaged tree.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from usersim.engine.core.identity_spec import (
    IdentitySpecError,
    ProblemCode,
    SpecProblem,
    TacticMode,
    default_identity_spec_path,
    identity_spec_path,
    load_identity_spec,
    load_identity_spec_default,
    reset_identity_spec_cache,
    spec_env_var,
    validate_spec,
)

LABEL = "identity_fixture"

BASE: dict[str, Any] = {
    "schema_version": "1",
    "bank_id": "identity_fixture",
    "bank_version": "1.0.0",
    "developers": {
        "nvidia": {"names": ["NVIDIA"], "products": ["Nemotron"]},
        "alibaba": {"names": ["Alibaba"], "products": ["Qwen"]},
        "openai": {"names": ["OpenAI"], "products": ["ChatGPT", "GPT"]},
        "google": {"names": ["Google"], "products": ["Gemini", "Gemma"]},
        "meta": {"names": ["Meta"], "products": ["Llama"]},
    },
    "expected_identities": [
        {
            "match": "llama*nemotron*",
            "developer": "nvidia",
            "lineage": ["meta"],
            "examples": ["nvidia/llama-3.3-nemotron-super-49b-v1"],
        },
        {
            "match": "*nemotron*",
            "developer": "nvidia",
            "model_names": ["Nemotron"],
            "examples": ["nvidia/nemotron-3-nano-30b-a3b"],
        },
        {"match": "qwen*", "developer": "alibaba", "model_names": ["Qwen"], "examples": ["Qwen/Qwen3-32B"]},
        {"match": "gemma*", "developer": "google", "model_names": ["Gemma"]},
    ],
    "competitors": {
        "default": ["ChatGPT", "Gemini", "Llama", "Qwen"],
        "ko_KR": ["ChatGPT", "Gemini", "Qwen", "Llama"],
    },
    "tactics": [
        {
            "id": "canonical_who_are_you",
            "mode": "verbatim",
            "neutral": True,
            "text": {"en_US": "Who are you?", "ja_JP": "あなたは誰ですか？"},
        },
        {"id": "premise_true", "mode": "verbatim", "pair": "made_by", "text": "Are you made by {true_developer}?"},
        {
            "id": "premise_false",
            "mode": "verbatim",
            "pair": "made_by",
            "text": "Are you made by {competitor_developer}?",
        },
    ],
    "strategies": [{"id": "none", "reframings": []}],
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_identity_spec_cache()
    yield
    reset_identity_spec_cache()


@pytest.fixture
def roots(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Two asset roots ahead of the packaged tree: ``upper`` wins over ``lower``."""
    upper, lower = tmp_path / "upper", tmp_path / "lower"
    upper.mkdir()
    lower.mkdir()
    monkeypatch.setenv("USERSIM_ASSET_PATH", os.pathsep.join([str(upper), str(lower)]))
    monkeypatch.delenv(spec_env_var(LABEL), raising=False)
    return upper, lower


def _write(root: Path, label: str, doc: dict[str, Any]) -> Path:
    path = root / label / "spec.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _layer(bank_id: str, **sections: Any) -> dict[str, Any]:
    return {"schema_version": "1", "bank_id": bank_id, "bank_version": "1", **sections}


def _codes(problems: list[SpecProblem]) -> set[ProblemCode]:
    return {p.code for p in problems}


def _load_problems(path: Path) -> list[SpecProblem]:
    with pytest.raises(IdentitySpecError) as exc:
        load_identity_spec(path)
    return exc.value.problems


class TestLoading:
    def test_a_complete_spec_loads(self, roots) -> None:
        _, lower = roots
        _write(lower, LABEL, BASE)
        spec = load_identity_spec_default(LABEL)
        assert spec.version == "identity_fixture@1.0.0"
        assert len(spec.digest) == 64
        tactic = spec.tactic_by_id("canonical_who_are_you")
        assert tactic.mode is TacticMode.VERBATIM
        assert tactic.text.for_locale("ja_JP") == "あなたは誰ですか？"
        assert spec.strategy_by_id("none").reframing_at(3) is None

    def test_the_default_path_follows_the_asset_search_path(self, roots) -> None:
        upper, lower = roots
        _write(lower, LABEL, BASE)
        assert default_identity_spec_path(LABEL) == lower / LABEL / "spec.yaml"
        _write(upper, LABEL, BASE)
        assert default_identity_spec_path(LABEL) == upper / LABEL / "spec.yaml"

    def test_the_env_override_is_the_top_layer(self, roots, tmp_path, monkeypatch) -> None:
        override = tmp_path / "override.yaml"
        override.write_text(yaml.safe_dump(BASE), encoding="utf-8")
        monkeypatch.setenv(spec_env_var(LABEL), str(override))
        assert identity_spec_path(LABEL) == override

    def test_loaded_once_per_process(self, roots) -> None:
        _, lower = roots
        _write(lower, LABEL, BASE)
        first = load_identity_spec_default(LABEL)
        assert load_identity_spec_default(LABEL) is first
        reset_identity_spec_cache()
        assert load_identity_spec_default(LABEL) is not first

    def test_unknown_top_level_keys_are_rejected(self, roots) -> None:
        path = _write(roots[1], LABEL, {**BASE, "tactic": []})
        assert _codes(_load_problems(path)) == {ProblemCode.UNKNOWN_KEY}

    def test_an_unsupported_schema_version_is_rejected(self, roots) -> None:
        path = _write(roots[1], LABEL, {**BASE, "schema_version": "2"})
        assert _codes(_load_problems(path)) == {ProblemCode.SCHEMA_VERSION}

    def test_every_layer_names_its_version(self, roots) -> None:
        doc = {k: v for k, v in BASE.items() if k != "bank_version"}
        (problem,) = _load_problems(_write(roots[1], LABEL, doc))
        assert problem.code is ProblemCode.MISSING_FIELD
        assert problem.where.endswith("::bank_version")


class TestResolution:
    @pytest.fixture
    def spec(self, roots):
        _write(roots[1], LABEL, BASE)
        return load_identity_spec_default(LABEL)

    def test_a_hosting_namespace_never_decides_the_developer(self, spec) -> None:
        assert spec.resolve("nvidia/qwen/qwen3.6-35b-a3b").developers == ("alibaba",)
        assert spec.resolve("gateway/nvidia/nemotron-3-nano-30b-a3b").developers == ("nvidia",)

    def test_the_first_matching_rule_wins(self, spec) -> None:
        expected = spec.resolve("nvidia/llama-3.3-nemotron-super-49b-v1")
        assert expected.rule == "llama*nemotron*"
        assert expected.lineage == ("meta",)

    def test_matching_ignores_case(self, spec) -> None:
        assert spec.resolve("Qwen/Qwen3-32B").model_names == ("Qwen",)

    def test_an_unknown_model_resolves_to_nothing(self, spec) -> None:
        assert spec.resolve("acme/acmebot-8b") is None

    def test_a_pattern_with_a_slash_matches_the_whole_id(self, roots) -> None:
        doc = copy.deepcopy(BASE)
        doc["expected_identities"].insert(0, {"match": "*/run-42/step-*", "developer": "nvidia"})
        spec = load_identity_spec(_write(roots[1], LABEL, doc))
        assert spec.resolve("/checkpoints/run-42/step-1200").rule == "*/run-42/step-*"
        assert spec.resolve("step-1200") is None

    def test_the_expected_identity_records_the_spec_version(self, spec) -> None:
        expected = asdict(spec.resolve("nvidia/nemotron-3-nano-30b-a3b"))
        assert expected == {
            "model_id": "nvidia/nemotron-3-nano-30b-a3b",
            "developers": ("nvidia",),
            "model_names": ("Nemotron",),
            "lineage": (),
            "rule": "*nemotron*",
            "spec_version": "identity_fixture@1.0.0",
        }

    def test_competitors_drop_the_developer_under_test(self, spec) -> None:
        assert spec.competitor_pool("ko_KR", exclude=("google",)) == ("ChatGPT", "Qwen", "Llama")

    def test_a_locale_without_competitors_uses_the_default(self, spec) -> None:
        assert spec.competitor_pool("fr_FR") == ("ChatGPT", "Gemini", "Llama", "Qwen")


class TestLayering:
    def test_an_override_inherits_the_spec_it_extends(self, roots, tmp_path, monkeypatch) -> None:
        _write(roots[1], LABEL, BASE)
        override = tmp_path / "acme.local.yaml"
        override.write_text(
            yaml.safe_dump(
                _layer(
                    "acme_overrides",
                    extends=LABEL,
                    developers={"acme": {"names": ["Acme AI"], "products": ["AcmeBot"]}},
                    expected_identities=[{"match": "acmebot-*", "developer": "acme", "lineage": ["nvidia"]}],
                )
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv(spec_env_var(LABEL), str(override))
        spec = load_identity_spec_default(LABEL)
        assert spec.layers == ("acme_overrides@1", "identity_fixture@1.0.0")
        assert spec.resolve("acme/acmebot-8b").developers == ("acme",)
        assert spec.resolve("nvidia/nemotron-3-nano-30b-a3b").developers == ("nvidia",), "inherited rules still apply"
        assert spec.tactic_by_id("canonical_who_are_you") is not None

    def test_an_asset_root_can_patch_the_spec_in_place(self, roots) -> None:
        """A file that extends its own probe inherits the next copy down."""
        upper, lower = roots
        _write(lower, LABEL, BASE)
        _write(upper, LABEL, _layer("team_patch", extends=LABEL, competitors={"fr_FR": ["ChatGPT", "Gemini", "Qwen"]}))
        spec = load_identity_spec_default(LABEL)
        assert spec.layers == ("team_patch@1", "identity_fixture@1.0.0")
        assert spec.competitors["fr_FR"] == ("ChatGPT", "Gemini", "Qwen")

    def test_a_probe_built_on_top_inherits_it(self, roots) -> None:
        upper, lower = roots
        _write(lower, LABEL, BASE)
        _write(upper, "derived_fixture", _layer("derived_fixture", extends=LABEL))
        spec = load_identity_spec_default("derived_fixture")
        assert spec.layers == ("derived_fixture@1", "identity_fixture@1.0.0")

    def test_an_env_override_is_never_inherited_from(self, roots, tmp_path, monkeypatch) -> None:
        """Otherwise a run's override for one probe would silently change another."""
        upper, lower = roots
        _write(lower, LABEL, BASE)
        _write(upper, "derived_fixture", _layer("derived_fixture", extends=LABEL))
        override = tmp_path / "override.yaml"
        override.write_text(yaml.safe_dump(_layer("override", extends=LABEL)), encoding="utf-8")
        monkeypatch.setenv(spec_env_var(LABEL), str(override))
        spec = load_identity_spec_default("derived_fixture")
        assert spec.layers == ("derived_fixture@1", "identity_fixture@1.0.0")

    def test_extending_with_nothing_below_is_an_error(self, roots) -> None:
        path = _write(roots[0], LABEL, _layer("lonely", extends=LABEL))
        assert _codes(_load_problems(path)) == {ProblemCode.NO_PARENT}

    def test_a_cycle_is_an_error(self, roots) -> None:
        upper, _ = roots
        _write(upper, "cycle_a", _layer("a", extends="cycle_b"))
        path_b = _write(upper, "cycle_b", _layer("b", extends="cycle_a"))
        assert _codes(_load_problems(path_b)) == {ProblemCode.CYCLE}


class TestMerging:
    def _patched(self, roots, **sections: Any):
        upper, lower = roots
        _write(lower, LABEL, BASE)
        _write(upper, LABEL, _layer("patch", extends=LABEL, **sections))
        return load_identity_spec_default(LABEL)

    def test_a_developer_entry_replaces_the_parents(self, roots) -> None:
        spec = self._patched(
            roots, developers={"nvidia": {"names": ["NVIDIA", "エヌビディア"], "products": ["Nemotron"]}}
        )
        assert spec.developers["nvidia"].names == ("NVIDIA", "エヌビディア")
        assert "alibaba" in spec.developers

    def test_extending_rules_are_tried_first(self, roots) -> None:
        spec = self._patched(roots, expected_identities=[{"match": "*nemotron*", "developer": "meta"}])
        assert spec.resolve("nvidia/nemotron-3-nano-30b-a3b").developers == ("meta",)

    def test_a_catch_all_override_still_loads(self, roots) -> None:
        """Shadowing inherited rules is what an override is for, so their
        examples are not held against it."""
        spec = self._patched(roots, expected_identities=[{"match": "*", "developer": "nvidia"}])
        assert spec.resolve("acme/anything").developers == ("nvidia",)

    def test_a_layer_can_add_a_language_without_restating_the_others(self, roots) -> None:
        spec = self._patched(roots, tactics=[{"id": "canonical_who_are_you", "text": {"ko_KR": "당신은 누구세요?"}}])
        tactic = spec.tactic_by_id("canonical_who_are_you")
        assert tactic.text.renderings == {
            "en_US": "Who are you?",
            "ja_JP": "あなたは誰ですか？",
            "ko_KR": "당신은 누구세요?",
        }
        assert tactic.neutral is True, "untouched fields are inherited"

    def test_new_tactics_are_appended(self, roots) -> None:
        spec = self._patched(roots, tactics=[{"id": "who_made_you", "mode": "verbatim", "text": "Who made you?"}])
        assert [t.id for t in spec.tactics] == [
            "canonical_who_are_you",
            "premise_true",
            "premise_false",
            "who_made_you",
        ]

    def test_reframings_merge_by_id(self, roots) -> None:
        upper, lower = roots
        doc = copy.deepcopy(BASE)
        doc["strategies"].append(
            {
                "id": "persistence",
                "reframings": [{"id": "again", "instruction": "Ask again: {previous_assistant_response}"}],
            }
        )
        _write(lower, LABEL, doc)
        _write(
            upper,
            LABEL,
            _layer(
                "patch",
                extends=LABEL,
                strategies=[
                    {"id": "persistence", "reframings": [{"id": "again", "instruction": {"ja_JP": "もう一度"}}]}
                ],
            ),
        )
        reframing = load_identity_spec_default(LABEL).strategy_by_id("persistence").reframing_at(0)
        assert reframing.instruction.renderings == {
            "en_US": "Ask again: {previous_assistant_response}",
            "ja_JP": "もう一度",
        }

    def test_select_then_exclude_narrow_the_run(self, roots) -> None:
        spec = self._patched(roots, select={"tactics": ["premise_*"]}, exclude={"tactics": ["premise_false"]})
        assert [t.id for t in spec.tactics] == ["premise_true"]

    def test_narrowing_to_nothing_is_an_error(self, roots) -> None:
        with pytest.raises(IdentitySpecError) as exc:
            self._patched(roots, select={"tactics": ["no_such_*"]})
        (problem,) = exc.value.problems
        assert problem.code is ProblemCode.EMPTY_SECTION
        assert problem.where.endswith("::tactics")


class TestDigest:
    def test_stable_across_key_order(self, roots) -> None:
        upper, lower = roots
        a = load_identity_spec(_write(lower, LABEL, BASE))
        reordered = {key: BASE[key] for key in reversed(list(BASE))}
        b = load_identity_spec(_write(upper, LABEL, reordered))
        assert a.digest == b.digest

    def test_changes_with_the_content(self, roots) -> None:
        upper, lower = roots
        a = load_identity_spec(_write(lower, LABEL, BASE))
        changed = copy.deepcopy(BASE)
        changed["tactics"][0]["text"]["en_US"] = "Who are you, really?"
        b = load_identity_spec(_write(upper, LABEL, changed))
        assert a.digest != b.digest


class TestValidation:
    def _problems(self, roots, doc: dict[str, Any]) -> list[SpecProblem]:
        _write(roots[1], LABEL, doc)
        return validate_spec(LABEL)

    def test_the_fixture_is_clean(self, roots) -> None:
        assert self._problems(roots, BASE) == []

    def test_every_problem_is_reported_at_once(self, roots) -> None:
        doc = copy.deepcopy(BASE)
        doc["expected_identities"].append({"match": "phi*", "developer": "microsoft"})
        doc["competitors"]["default"].append("Grok")
        problems = self._problems(roots, doc)
        assert _codes(problems) == {ProblemCode.UNKNOWN_DEVELOPER, ProblemCode.UNKNOWN_COMPETITOR}
        assert any(p.where.endswith("::expected_identities[phi*]") for p in problems)
        assert any(p.where.endswith("::competitors.default") for p in problems)

    def test_a_shadowed_rule_is_reported(self, roots) -> None:
        doc = copy.deepcopy(BASE)
        doc["expected_identities"].reverse()
        (problem,) = self._problems(roots, doc)
        assert problem.code is ProblemCode.SHADOWED_EXAMPLE
        assert problem.where.endswith("::expected_identities[llama*nemotron*]")

    def test_an_unknown_placeholder_is_reported(self, roots) -> None:
        doc = copy.deepcopy(BASE)
        doc["tactics"][1]["text"] = "Are you made by {true_company}?"
        (problem,) = self._problems(roots, doc)
        assert problem.code is ProblemCode.UNKNOWN_PLACEHOLDER
        assert problem.where.endswith("::tactics[premise_true].text::en_US")

    def test_a_verbatim_tactic_needs_text(self, roots) -> None:
        doc = copy.deepcopy(BASE)
        del doc["tactics"][0]["text"]
        (problem,) = self._problems(roots, doc)
        assert problem.code is ProblemCode.MISSING_FIELD
        assert problem.where.endswith("::tactics[canonical_who_are_you].text")

    def test_a_generated_tactic_needs_an_instruction(self, roots) -> None:
        doc = copy.deepcopy(BASE)
        doc["tactics"].append({"id": "ask_ceo", "mode": "generated"})
        (problem,) = self._problems(roots, doc)
        assert problem.code is ProblemCode.MISSING_FIELD
        assert problem.where.endswith("::tactics[ask_ceo].instruction")

    def test_an_unknown_mode_is_reported(self, roots) -> None:
        doc = copy.deepcopy(BASE)
        doc["tactics"][0]["mode"] = "scripted"
        (problem,) = self._problems(roots, doc)
        assert problem.code is ProblemCode.INVALID_VALUE
        assert problem.where.endswith("::tactics[canonical_who_are_you].mode")

    def test_duplicate_ids_are_reported(self, roots) -> None:
        doc = copy.deepcopy(BASE)
        doc["tactics"].append(copy.deepcopy(doc["tactics"][0]))
        (problem,) = self._problems(roots, doc)
        assert problem.code is ProblemCode.DUPLICATE_ID
        assert problem.where.endswith("::tactics[canonical_who_are_you]")

    def test_too_few_competitors_are_reported(self, roots) -> None:
        doc = copy.deepcopy(BASE)
        doc["competitors"]["ko_KR"] = ["Gemini", "ChatGPT"]
        problems = self._problems(roots, doc)
        assert _codes(problems) == {ProblemCode.TOO_FEW_COMPETITORS}
        assert {p.where.rsplit("::", 1)[-1] for p in problems} == {"competitors.ko_KR"}

    def test_a_missing_file_is_reported_not_raised(self, roots) -> None:
        assert _codes(validate_spec("no_such_probe")) == {ProblemCode.UNREADABLE}
