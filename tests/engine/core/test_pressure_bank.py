# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for ``core/pressure_bank.py`` — pressure-strategy bank loader.

Same three-layer test pattern as ``test_query_bank.py`` /
``test_fact_bank.py`` / ``test_probing_taxonomy.py``:

1. **Schema validation** — every required field, every cross-product
   invariant (placeholder discipline, template placeholders), every
   error path.
2. **Lookup helpers** — `strategy_by_id`, `target_by_id`,
   `targets_by_harm_category`, `targets_for_persona`,
   `Strategy.reframing_at` (with wrap-around).
3. **Cache + env override + shipped sample** — process-local cache
   behaves correctly, env override picks up alternate path, shipped
   sample loads cleanly (catches drift between schema doc and content).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.pressure_bank import (
    PressureBank,
    PressureBankError,
    default_pressure_bank_path,
    load_pressure_bank,
    load_pressure_bank_default,
    reset_pressure_bank_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_bank_yaml(
    *,
    bank_id: str = "test_bank",
    bank_version: str = "v0.1.0",
    extra_strategies: str = "",
    extra_targets: str = "",
    n_reframings: int = 1,
) -> str:
    """Build a minimal valid bank as a YAML string. Direct construction
    rather than textwrap.dedent so f-string interpolations don't fight
    indentation."""
    lines: list[str] = []
    lines.append('schema_version: "v0.1"')
    lines.append(f'bank_id: "{bank_id}"')
    lines.append(f'bank_version: "{bank_version}"')
    lines.append("harm_categories:")
    lines.append("  - safety_bypass")
    lines.append("strategies:")
    lines.append("  - id: persistence")
    lines.append("    description: |")
    lines.append("      Strategy description.")
    lines.append("    system_prompt_template: |")
    lines.append("      Persona: {persona}; Target: {target_description}; Opening: {initial_prompt}")
    lines.append("    reframings:")
    for i in range(n_reframings):
        lines.append(f"      - id: r{i}")
        lines.append(f"        description: Reframing {i}.")
        lines.append("        instruction: |")
        lines.append(f"          Tactic {i}: react to {{previous_assistant_response}}")
    if extra_strategies:
        lines.append(extra_strategies.rstrip("\n"))
    lines.append("target_requests:")
    lines.append("  - id: TR-TEST-001")
    lines.append("    description: Test target.")
    lines.append("    harm_category: safety_bypass")
    lines.append("    placeholder: true")
    lines.append("    initial_prompt: |")
    lines.append("      Initial prompt body.")
    lines.append("    expected_refusal: |")
    lines.append("      Expected refusal sketch.")
    lines.append("    provenance:")
    lines.append('      source: "placeholder"')
    lines.append('      last_reviewed: "2026-04-20"')
    if extra_targets:
        lines.append(extra_targets.rstrip("\n"))
    return "\n".join(lines) + "\n"


def _write_bank(tmp_path: Path, yaml_text: str, name: str = "bank.yaml") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    p.write_text(yaml_text, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    reset_pressure_bank_cache()
    yield
    reset_pressure_bank_cache()


# ---------------------------------------------------------------------------
# Happy path: minimal valid bank
# ---------------------------------------------------------------------------


class TestMinimalBankLoad:
    def test_loads_minimal_bank(self, tmp_path: Path) -> None:
        bank = load_pressure_bank(_write_bank(tmp_path, _minimal_bank_yaml()))
        assert isinstance(bank, PressureBank)
        assert bank.bank_id == "test_bank"
        assert bank.bank_version == "v0.1.0"
        assert bank.harm_categories == ("safety_bypass",)
        assert len(bank.strategies) == 1
        assert len(bank.target_requests) == 1

    def test_strategy_has_reframings(self, tmp_path: Path) -> None:
        bank = load_pressure_bank(_write_bank(tmp_path, _minimal_bank_yaml(n_reframings=3)))
        strat = bank.strategies[0]
        assert len(strat.reframings) == 3
        # Every reframing carries its parent strategy id.
        for r in strat.reframings:
            assert r.strategy_id == "persistence"

    def test_target_carries_bank_provenance(self, tmp_path: Path) -> None:
        bank = load_pressure_bank(_write_bank(tmp_path, _minimal_bank_yaml()))
        t = bank.target_requests[0]
        assert t.bank_id == "test_bank"
        assert t.bank_version == "v0.1.0"

    def test_provenance_summary(self, tmp_path: Path) -> None:
        bank = load_pressure_bank(_write_bank(tmp_path, _minimal_bank_yaml()))
        s = bank.provenance_summary()
        assert s["bank_id"] == "test_bank"
        assert s["n_strategies"] == 1
        assert s["n_targets"] == 1
        assert s["n_placeholder_targets"] == 1


# ---------------------------------------------------------------------------
# Top-level + entry validation
# ---------------------------------------------------------------------------


class TestTopLevelValidation:
    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace('"v0.1"', '"v9.9"')
        with pytest.raises(PressureBankError, match="not supported"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_missing_bank_id_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace('bank_id: "test_bank"\n', "")
        with pytest.raises(PressureBankError, match="bank_id"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_empty_harm_categories_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "harm_categories:\n  - safety_bypass",
            "harm_categories: []",
        )
        with pytest.raises(PressureBankError, match="harm_categories"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_empty_strategies_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml()
        # Replace whole strategies block with empty.
        before, _, _ = text.partition("strategies:")
        text = (
            before
            + "strategies: []\n"
            + "target_requests:\n"
            + "  - id: TR-TEST-001\n"
            + "    description: Test.\n"
            + "    harm_category: safety_bypass\n"
            + "    placeholder: true\n"
            + "    initial_prompt: |\n      x\n"
            + "    expected_refusal: |\n      x\n"
            + "    provenance:\n"
            + '      source: "placeholder"\n'
            + '      last_reviewed: "2026-04-20"\n'
        )
        with pytest.raises(PressureBankError, match="strategies"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_empty_targets_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml()
        before, _, _ = text.partition("target_requests:")
        text = before + "target_requests: []\n"
        with pytest.raises(PressureBankError, match="target_requests"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PressureBankError, match="not found"):
            load_pressure_bank(tmp_path / "absent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("not: valid: yaml:\n  - [\n", encoding="utf-8")
        with pytest.raises(PressureBankError, match="YAML parse failure"):
            load_pressure_bank(path)


# ---------------------------------------------------------------------------
# Strategy + reframing template-placeholder validation
# ---------------------------------------------------------------------------


class TestTemplateValidation:
    def test_strategy_template_missing_persona_placeholder_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        text = _minimal_bank_yaml().replace(
            "Persona: {persona}; Target: {target_description}; Opening: {initial_prompt}",
            "Target: {target_description}; Opening: {initial_prompt}",
        )
        with pytest.raises(PressureBankError, match=r"\{persona\}"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_strategy_template_missing_target_description_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        text = _minimal_bank_yaml().replace(
            "Persona: {persona}; Target: {target_description}; Opening: {initial_prompt}",
            "Persona: {persona}; Opening: {initial_prompt}",
        )
        with pytest.raises(PressureBankError, match=r"\{target_description\}"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_strategy_template_missing_initial_prompt_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        text = _minimal_bank_yaml().replace(
            "Persona: {persona}; Target: {target_description}; Opening: {initial_prompt}",
            "Persona: {persona}; Target: {target_description}",
        )
        with pytest.raises(PressureBankError, match=r"\{initial_prompt\}"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_reframing_instruction_missing_placeholder_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        text = _minimal_bank_yaml().replace(
            "Tactic 0: react to {previous_assistant_response}",
            "Tactic 0: just react somehow",
        )
        with pytest.raises(PressureBankError, match=r"\{previous_assistant_response\}"):
            load_pressure_bank(_write_bank(tmp_path, text))


# ---------------------------------------------------------------------------
# Entry-level validation
# ---------------------------------------------------------------------------


class TestEntryValidation:
    def test_duplicate_strategy_id_rejected(self, tmp_path: Path) -> None:
        extra = "\n".join(
            [
                "  - id: persistence",
                "    description: Duplicate.",
                "    system_prompt_template: |",
                "      {persona} {target_description} {initial_prompt}",
                "    reframings:",
                "      - id: r0",
                "        description: dup.",
                "        instruction: |",
                "          {previous_assistant_response}",
            ]
        )
        text = _minimal_bank_yaml(extra_strategies=extra)
        with pytest.raises(PressureBankError, match="duplicate strategy id"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_duplicate_reframing_id_within_strategy_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        # Add a second reframing with the same id as the first.
        text = _minimal_bank_yaml().replace(
            "      - id: r0\n"
            "        description: Reframing 0.\n"
            "        instruction: |\n"
            "          Tactic 0: react to {previous_assistant_response}\n",
            "      - id: r0\n"
            "        description: Reframing 0.\n"
            "        instruction: |\n"
            "          Tactic 0: react to {previous_assistant_response}\n"
            "      - id: r0\n"
            "        description: Duplicate.\n"
            "        instruction: |\n"
            "          Tactic 0bis: react to {previous_assistant_response}\n",
        )
        with pytest.raises(PressureBankError, match="duplicate reframing id"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_duplicate_target_id_rejected(self, tmp_path: Path) -> None:
        extra = "\n".join(
            [
                "  - id: TR-TEST-001",
                "    description: Duplicate.",
                "    harm_category: safety_bypass",
                "    placeholder: true",
                "    initial_prompt: |",
                "      x",
                "    expected_refusal: |",
                "      x",
                "    provenance:",
                '      source: "placeholder"',
                '      last_reviewed: "2026-04-20"',
            ]
        )
        text = _minimal_bank_yaml(extra_targets=extra)
        with pytest.raises(PressureBankError, match="duplicate target id"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_unknown_harm_category_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "harm_category: safety_bypass",
            "harm_category: not_in_bank",
        )
        with pytest.raises(PressureBankError, match="not in the bank's declared harm_categories"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_strategy_with_no_reframings_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "    reframings:\n"
            "      - id: r0\n"
            "        description: Reframing 0.\n"
            "        instruction: |\n"
            "          Tactic 0: react to {previous_assistant_response}\n",
            "    reframings: []\n",
        )
        with pytest.raises(PressureBankError, match="reframings"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_placeholder_must_be_bool(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "    placeholder: true",
            "    placeholder: maybe",
        )
        with pytest.raises(PressureBankError, match="placeholder"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_placeholder_source_with_non_placeholder_entry_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        text = _minimal_bank_yaml().replace(
            "    placeholder: true",
            "    placeholder: false",
        )
        with pytest.raises(PressureBankError, match="'placeholder' is only allowed"):
            load_pressure_bank(_write_bank(tmp_path, text))

    def test_non_iso_date_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            'last_reviewed: "2026-04-20"',
            'last_reviewed: "20-04-2026"',
        )
        with pytest.raises(PressureBankError, match="last_reviewed"):
            load_pressure_bank(_write_bank(tmp_path, text))


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


class TestLookups:
    @pytest.fixture
    def bank(self, tmp_path: Path) -> PressureBank:
        return load_pressure_bank(_write_bank(tmp_path, _minimal_bank_yaml(n_reframings=4)))

    def test_strategy_by_id(self, bank: PressureBank) -> None:
        assert bank.strategy_by_id("persistence") is not None
        assert bank.strategy_by_id("missing") is None

    def test_target_by_id(self, bank: PressureBank) -> None:
        assert bank.target_by_id("TR-TEST-001") is not None
        assert bank.target_by_id("missing") is None

    def test_targets_by_harm_category(self, bank: PressureBank) -> None:
        assert len(bank.targets_by_harm_category("safety_bypass")) == 1
        assert len(bank.targets_by_harm_category("manipulation")) == 0

    def test_strategy_ids_target_ids(self, bank: PressureBank) -> None:
        assert bank.strategy_ids() == ("persistence",)
        assert bank.target_ids() == ("TR-TEST-001",)

    def test_reframing_at_wraps_around(self, bank: PressureBank) -> None:
        s = bank.strategy_by_id("persistence")
        assert s is not None
        # 4 reframings; index 0..7 → r0, r1, r2, r3, r0, r1, r2, r3.
        ids = [s.reframing_at(i).id for i in range(8)]
        assert ids == ["r0", "r1", "r2", "r3", "r0", "r1", "r2", "r3"]

    def test_reframing_by_id(self, bank: PressureBank) -> None:
        s = bank.strategy_by_id("persistence")
        assert s is not None
        assert s.reframing_by_id("r0") is not None
        assert s.reframing_by_id("missing") is None


# ---------------------------------------------------------------------------
# Persona-tag matching
# ---------------------------------------------------------------------------


class TestPersonaMatching:
    def test_target_with_no_tags_matches_any_persona(self, tmp_path: Path) -> None:
        bank = load_pressure_bank(_write_bank(tmp_path, _minimal_bank_yaml()))
        t = bank.target_requests[0]
        assert t.matches_persona([]) is True
        assert t.matches_persona(["interest:health", "age:25-44"]) is True

    def test_target_with_tags_intersects_persona(self, tmp_path: Path) -> None:
        # Add a target with persona_tags constraint.
        extra = "\n".join(
            [
                "  - id: TR-TEST-002",
                "    description: Targeted.",
                "    harm_category: safety_bypass",
                "    placeholder: true",
                "    initial_prompt: |",
                "      x",
                "    expected_refusal: |",
                "      x",
                '    persona_tags: ["interest:tech"]',
                "    provenance:",
                '      source: "placeholder"',
                '      last_reviewed: "2026-04-20"',
            ]
        )
        text = _minimal_bank_yaml(extra_targets=extra)
        bank = load_pressure_bank(_write_bank(tmp_path, text))
        t = bank.target_by_id("TR-TEST-002")
        assert t is not None
        assert t.matches_persona(["interest:tech"]) is True
        assert t.matches_persona(["interest:health"]) is False

    def test_targets_for_persona_returns_applicable(self, tmp_path: Path) -> None:
        extra = "\n".join(
            [
                "  - id: TR-TEST-002",
                "    description: Tech-only target.",
                "    harm_category: safety_bypass",
                "    placeholder: true",
                "    initial_prompt: |",
                "      x",
                "    expected_refusal: |",
                "      x",
                '    persona_tags: ["interest:tech"]',
                "    provenance:",
                '      source: "placeholder"',
                '      last_reviewed: "2026-04-20"',
            ]
        )
        bank = load_pressure_bank(_write_bank(tmp_path, _minimal_bank_yaml(extra_targets=extra)))
        # Persona without interest:tech: only the unrestricted target.
        match_a = bank.targets_for_persona(["interest:health"])
        assert {t.id for t in match_a} == {"TR-TEST-001"}
        # Persona with interest:tech: both.
        match_b = bank.targets_for_persona(["interest:tech"])
        assert {t.id for t in match_b} == {"TR-TEST-001", "TR-TEST-002"}


# ---------------------------------------------------------------------------
# Cache + env override
# ---------------------------------------------------------------------------


class TestCacheAndDefault:
    def test_load_once_cached(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        with patch.dict(os.environ, {"USERSIM_SAFETY_CHAT_PRESSURE_BANK": str(path)}):
            b1 = load_pressure_bank_default()
            b2 = load_pressure_bank_default()
        assert b1 is b2

    def test_env_override_picks_up_alternate_path(self, tmp_path: Path) -> None:
        path = _write_bank(
            tmp_path,
            _minimal_bank_yaml(bank_id="alt_bank"),
            name="alt.yaml",
        )
        with patch.dict(os.environ, {"USERSIM_SAFETY_CHAT_PRESSURE_BANK": str(path)}):
            bank = load_pressure_bank_default()
        assert bank.bank_id == "alt_bank"

    def test_reset_drops_cache(self, tmp_path: Path) -> None:
        a = _write_bank(tmp_path / "a", _minimal_bank_yaml(bank_id="a"))
        b = _write_bank(
            tmp_path / "b",
            _minimal_bank_yaml(bank_id="b"),
            name="bank.yaml",
        )
        with patch.dict(os.environ, {"USERSIM_SAFETY_CHAT_PRESSURE_BANK": str(a)}):
            assert load_pressure_bank_default().bank_id == "a"
        reset_pressure_bank_cache()
        with patch.dict(os.environ, {"USERSIM_SAFETY_CHAT_PRESSURE_BANK": str(b)}):
            assert load_pressure_bank_default().bank_id == "b"

    def test_default_path_resolves_to_assets(self) -> None:
        p = default_pressure_bank_path()
        assert p.parts[-3:] == ("assets", "safety_chat_pressure", "sample.yaml")


# ---------------------------------------------------------------------------
# Shipped sample bank — end-to-end check that schema + content stay in sync
# ---------------------------------------------------------------------------


class TestShippedSampleBank:
    @pytest.fixture
    def sample_bank(self) -> PressureBank:
        return load_pressure_bank((packaged_assets_dir() / "safety_chat_pressure/sample.yaml"))

    def test_loads_clean(self, sample_bank: PressureBank) -> None:
        assert sample_bank.bank_id == "sample_v1"
        assert sample_bank.bank_version == "v0.5.4"

    def test_six_strategies_with_ten_reframings_each(
        self,
        sample_bank: PressureBank,
    ) -> None:
        # Beta bank: 6 strategies × 10 reframings = 60 turn-tactics.
        assert len(sample_bank.strategies) == 6
        for s in sample_bank.strategies:
            assert len(s.reframings) == 10, (
                f"strategy {s.id} has {len(s.reframings)} reframings, expected 10 (per the beta bank target)"
            )

    def test_beta_strategy_ids(self, sample_bank: PressureBank) -> None:
        assert set(sample_bank.strategy_ids()) == {
            "persistence",
            "hypothetical_framing",
            "authority_impersonation",
            "emotional_dependency",
            "reward_penalty_framing",
            "partial_compliance",
        }

    def test_sixty_four_target_requests_across_eight_harm_categories(
        self,
        sample_bank: PressureBank,
    ) -> None:
        assert len(sample_bank.target_requests) == 64
        observed = {t.harm_category for t in sample_bank.target_requests}
        assert observed == set(sample_bank.harm_categories)
        for cat in sample_bank.harm_categories:
            assert len(sample_bank.targets_by_harm_category(cat)) == 8

    def test_reviewed_targets_are_not_generated_placeholders(
        self,
        sample_bank: PressureBank,
    ) -> None:
        # Ground truth for "this entry is AI-authored, not human-reviewed" is
        # the provenance.source field, not the ID format. After the asset
        # rewrite normalised entry IDs, all entries follow the uniform
        # TR-<HARM>-B<NNN> scheme, so we cannot distinguish reviewed from
        # AI-authored via ID alone.
        for t in sample_bank.target_requests:
            source = (t.provenance.source or "").lower() if t.provenance else ""
            is_ai_authored = "ai-authored" in source
            if is_ai_authored:
                assert t.placeholder is True, f"{t.id}: AI-authored entry must be placeholder=True"
            else:
                assert t.placeholder is False, f"{t.id}: human-reviewed entry must be placeholder=False"

    def test_every_target_has_substantive_expected_refusal(
        self,
        sample_bank: PressureBank,
    ) -> None:
        # The expected_refusal field is the load-bearing reviewer-facing
        # field — the scorer reads it to ground judgement. Validate each
        # locale rendering individually (an empty pt_BR rendering would
        # produce a confounded judge prompt for Portuguese trajectories).
        for t in sample_bank.target_requests:
            for loc, text in t.expected_refusal.renderings.items():
                assert len(text) >= 80, f"{t.id}::expected_refusal[{loc}] too short ({len(text)} chars)"

    def test_every_strategy_template_includes_required_placeholders(
        self,
        sample_bank: PressureBank,
    ) -> None:
        for s in sample_bank.strategies:
            for loc, text in s.system_prompt_template.renderings.items():
                for ph in ("{persona}", "{target_description}", "{initial_prompt}"):
                    assert ph in text, f"strategy {s.id}::system_prompt_template[{loc}] missing {ph}"

    def test_every_reframing_includes_previous_assistant_placeholder(
        self,
        sample_bank: PressureBank,
    ) -> None:
        for s in sample_bank.strategies:
            for r in s.reframings:
                for loc, text in r.instruction.renderings.items():
                    assert "{previous_assistant_response}" in text, (
                        f"reframing {s.id}::{r.id}::instruction[{loc}] missing {{previous_assistant_response}}"
                    )

    def test_provenance_summary_smoke(self, sample_bank: PressureBank) -> None:
        s = sample_bank.provenance_summary()
        assert s["n_strategies"] == 6
        assert s["n_targets"] == 64
        assert s["n_placeholder_targets"] == 59
