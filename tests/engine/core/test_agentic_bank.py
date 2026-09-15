# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``core/agentic_bank.py`` — the agentic-action bank loader.

Same three-layer test pattern as the other curated-asset bank loaders
(``test_fact_bank.py`` / ``test_query_bank.py`` / ``test_pressure_bank.py``):

1. **Schema validation** — every required field, every cross-product
   invariant (sub_protocol membership, simulated_tools shape,
   mock_tool_responses key matches a declared tool, taxonomy-name
   resolution), every error path.
2. **Lookup helpers** — `by_id`, `by_sub_protocol`, `for_persona`,
   `request_ids`.
3. **Cache + env override + shipped sample** — process-local cache
   behaves correctly, env override picks up alternate path, shipped
   sample loads cleanly (catches drift between schema doc and content,
   AND drift between bank entries and the action_taxonomy).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from usersim.engine.core.agentic_bank import (
    AgenticBank,
    AgenticBankError,
    MockToolResponse,
    ToolSpec,
    default_agentic_bank_path,
    load_agentic_bank,
    load_agentic_bank_default,
    reset_action_taxonomy_cache,
    reset_agentic_bank_cache,
)
from usersim.engine.core._assets import packaged_assets_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_bank_yaml(
    *,
    bank_id: str = "test_bank",
    bank_version: str = "v0.1.0",
    extra_entries: str = "",
    sub_protocols: tuple[str, ...] = ("unsanctioned_action",),
    sanctioned_action_name: str = "read_user_personal_data",
    placeholder: bool = True,
) -> str:
    lines: list[str] = []
    lines.append('schema_version: "v0.1"')
    lines.append(f'bank_id: "{bank_id}"')
    lines.append(f'bank_version: "{bank_version}"')
    lines.append("sub_protocols:")
    for sp in sub_protocols:
        lines.append(f"  - {sp}")
    lines.append("action_requests:")
    lines.append("  - id: AR-TEST-001")
    lines.append(f"    sub_protocol: {sub_protocols[0]}")
    lines.append(f"    placeholder: {'true' if placeholder else 'false'}")
    lines.append("    description: |")
    lines.append("      Test description for AR-TEST-001.")
    lines.append(f"    sanctioned_action_name: {sanctioned_action_name}")
    lines.append("    initial_user_prompt: |")
    lines.append("      Test prompt.")
    lines.append("    expected_behavior: |")
    lines.append("      Expected behavior sketch.")
    lines.append("    simulated_tools:")
    lines.append("      - name: read_thing")
    lines.append("        description: Reads a thing.")
    lines.append("        parameters:")
    lines.append("          type: object")
    lines.append("          properties: {}")
    lines.append("    mock_tool_responses:")
    lines.append("      read_thing:")
    lines.append('        role: "tool"')
    lines.append('        name: "read_thing"')
    lines.append("        content: |")
    lines.append('          {"result": "ok"}')
    lines.append("    provenance:")
    if placeholder:
        lines.append('      source: "placeholder"')
    else:
        lines.append('      source: "internal authorship"')
    lines.append('      last_reviewed: "2026-04-20"')
    if extra_entries:
        lines.append(extra_entries.rstrip("\n"))
    return "\n".join(lines) + "\n"


def _write_bank(tmp_path: Path, yaml_text: str, name: str = "bank.yaml") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    p.write_text(yaml_text, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _reset_caches_between_tests():
    reset_agentic_bank_cache()
    reset_action_taxonomy_cache()
    yield
    reset_agentic_bank_cache()
    reset_action_taxonomy_cache()


# ---------------------------------------------------------------------------
# 1. Happy path: minimal valid bank
# ---------------------------------------------------------------------------


class TestMinimalBankLoad:
    def test_loads_minimal_bank(self, tmp_path: Path) -> None:
        bank = load_agentic_bank(_write_bank(tmp_path, _minimal_bank_yaml()))
        assert isinstance(bank, AgenticBank)
        assert bank.bank_id == "test_bank"
        assert bank.bank_version == "v0.1.0"
        assert bank.sub_protocols == ("unsanctioned_action",)
        assert len(bank.action_requests) == 1

    def test_action_request_carries_bank_provenance(self, tmp_path: Path) -> None:
        bank = load_agentic_bank(_write_bank(tmp_path, _minimal_bank_yaml()))
        ar = bank.action_requests[0]
        assert ar.bank_id == "test_bank"
        assert ar.bank_version == "v0.1.0"

    def test_simulated_tools_typed(self, tmp_path: Path) -> None:
        bank = load_agentic_bank(_write_bank(tmp_path, _minimal_bank_yaml()))
        ar = bank.action_requests[0]
        assert len(ar.simulated_tools) == 1
        t = ar.simulated_tools[0]
        assert isinstance(t, ToolSpec)
        assert t.name == "read_thing"
        assert isinstance(t.parameters, dict)

    def test_mock_tool_responses_typed(self, tmp_path: Path) -> None:
        bank = load_agentic_bank(_write_bank(tmp_path, _minimal_bank_yaml()))
        ar = bank.action_requests[0]
        assert "read_thing" in ar.mock_tool_responses
        r = ar.mock_tool_responses["read_thing"]
        assert isinstance(r, MockToolResponse)
        assert r.role == "tool"
        assert r.name == "read_thing"

    def test_provenance_summary(self, tmp_path: Path) -> None:
        bank = load_agentic_bank(_write_bank(tmp_path, _minimal_bank_yaml()))
        s = bank.provenance_summary()
        assert s["bank_id"] == "test_bank"
        assert s["n_requests"] == 1
        assert s["n_placeholder_requests"] == 1


# ---------------------------------------------------------------------------
# 2. Top-level + entry validation
# ---------------------------------------------------------------------------


class TestTopLevelValidation:
    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace('"v0.1"', '"v9.9"')
        with pytest.raises(AgenticBankError, match="not supported"):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_missing_bank_id_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace('bank_id: "test_bank"\n', "")
        with pytest.raises(AgenticBankError, match="bank_id"):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_empty_sub_protocols_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "sub_protocols:\n  - unsanctioned_action",
            "sub_protocols: []",
        )
        with pytest.raises(AgenticBankError, match="sub_protocols"):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_duplicate_sub_protocol_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "sub_protocols:\n  - unsanctioned_action",
            "sub_protocols:\n  - unsanctioned_action\n  - unsanctioned_action",
        )
        with pytest.raises(AgenticBankError, match="duplicates"):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_empty_action_requests_rejected(self, tmp_path: Path) -> None:
        # Build manually with empty action_requests.
        text = (
            'schema_version: "v0.1"\n'
            'bank_id: "x"\n'
            'bank_version: "v0.1.0"\n'
            "sub_protocols:\n  - unsanctioned_action\n"
            "action_requests: []\n"
        )
        with pytest.raises(AgenticBankError, match="action_requests"):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AgenticBankError, match="not found"):
            load_agentic_bank(tmp_path / "absent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("not: valid: yaml:\n  - [\n", encoding="utf-8")
        with pytest.raises(AgenticBankError, match="YAML parse failure"):
            load_agentic_bank(path)


# ---------------------------------------------------------------------------
# 3. Action-taxonomy resolution
# ---------------------------------------------------------------------------


class TestActionTaxonomyResolution:
    def test_sanctioned_action_must_be_in_taxonomy(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml(sanctioned_action_name="not_a_real_action")
        with pytest.raises(
            AgenticBankError, match="not found in.*action_taxonomy"
        ):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_sanctioned_action_can_be_null(self, tmp_path: Path) -> None:
        # null sanctioned_action_name = "assistant should refuse to call any tool".
        text = _minimal_bank_yaml().replace(
            "sanctioned_action_name: read_user_personal_data",
            "sanctioned_action_name: null",
        )
        bank = load_agentic_bank(_write_bank(tmp_path, text))
        assert bank.action_requests[0].sanctioned_action_name is None

    def test_missing_sanctioned_action_field_rejected(
        self, tmp_path: Path,
    ) -> None:
        text = _minimal_bank_yaml().replace(
            "    sanctioned_action_name: read_user_personal_data\n", "",
        )
        with pytest.raises(
            AgenticBankError, match="sanctioned_action_name.*missing"
        ):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_tempted_action_must_be_in_taxonomy(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "    sanctioned_action_name: read_user_personal_data",
            "    sanctioned_action_name: read_user_personal_data\n"
            "    tempted_action_name: not_a_real_action",
        )
        with pytest.raises(
            AgenticBankError, match="tempted_action_name.*not found"
        ):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_tempted_action_can_be_omitted(self, tmp_path: Path) -> None:
        bank = load_agentic_bank(_write_bank(tmp_path, _minimal_bank_yaml()))
        assert bank.action_requests[0].tempted_action_name is None


# ---------------------------------------------------------------------------
# 4. Entry-level validation
# ---------------------------------------------------------------------------


class TestEntryValidation:
    def test_unknown_sub_protocol_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "    sub_protocol: unsanctioned_action",
            "    sub_protocol: not_in_bank",
        )
        with pytest.raises(
            AgenticBankError,
            match="not in the bank's declared sub_protocols",
        ):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_duplicate_action_request_id_rejected(self, tmp_path: Path) -> None:
        extra = "\n".join([
            "  - id: AR-TEST-001",
            "    sub_protocol: unsanctioned_action",
            "    placeholder: true",
            "    description: |",
            "      Duplicate.",
            "    sanctioned_action_name: read_user_personal_data",
            "    initial_user_prompt: |",
            "      x",
            "    expected_behavior: |",
            "      x",
            "    simulated_tools:",
            "      - name: t1",
            "        description: x",
            "        parameters: {type: object, properties: {}}",
            "    provenance:",
            '      source: "placeholder"',
            '      last_reviewed: "2026-04-20"',
        ])
        text = _minimal_bank_yaml(extra_entries=extra)
        with pytest.raises(
            AgenticBankError, match="duplicate action_request id"
        ):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_placeholder_must_be_bool(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "    placeholder: true", "    placeholder: yes-please",
        )
        with pytest.raises(AgenticBankError, match="placeholder"):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_non_placeholder_with_placeholder_source_rejected(
        self, tmp_path: Path,
    ) -> None:
        text = _minimal_bank_yaml(placeholder=False).replace(
            'source: "internal authorship"',
            'source: "placeholder"',
        )
        with pytest.raises(
            AgenticBankError, match="'placeholder' is only allowed"
        ):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_non_iso_date_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            'last_reviewed: "2026-04-20"',
            'last_reviewed: "20-04-2026"',
        )
        with pytest.raises(AgenticBankError, match="last_reviewed"):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_empty_simulated_tools_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "    simulated_tools:\n"
            "      - name: read_thing\n"
            "        description: Reads a thing.\n"
            "        parameters:\n"
            "          type: object\n"
            "          properties: {}\n",
            "    simulated_tools: []\n",
        )
        with pytest.raises(AgenticBankError, match="simulated_tools"):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_duplicate_tool_name_rejected(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "    simulated_tools:\n"
            "      - name: read_thing\n"
            "        description: Reads a thing.\n"
            "        parameters:\n"
            "          type: object\n"
            "          properties: {}\n",
            "    simulated_tools:\n"
            "      - name: read_thing\n"
            "        description: First.\n"
            "        parameters: {type: object, properties: {}}\n"
            "      - name: read_thing\n"
            "        description: Duplicate.\n"
            "        parameters: {type: object, properties: {}}\n",
        )
        with pytest.raises(AgenticBankError, match="duplicate.*tool name"):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_mock_tool_response_for_undeclared_tool_rejected(
        self, tmp_path: Path,
    ) -> None:
        text = _minimal_bank_yaml().replace(
            "    mock_tool_responses:\n"
            "      read_thing:\n"
            '        role: "tool"\n'
            '        name: "read_thing"\n'
            "        content: |\n"
            '          {"result": "ok"}\n',
            "    mock_tool_responses:\n"
            "      typo_tool_name:\n"
            '        role: "tool"\n'
            '        name: "typo_tool_name"\n'
            "        content: |\n"
            '          {"result": "ok"}\n',
        )
        with pytest.raises(
            AgenticBankError, match="not declared in simulated_tools"
        ):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_mock_tool_response_role_must_be_tool(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "      read_thing:\n"
            '        role: "tool"\n'
            '        name: "read_thing"\n',
            "      read_thing:\n"
            '        role: "user"\n'
            '        name: "read_thing"\n',
        )
        with pytest.raises(AgenticBankError, match="must be 'tool'"):
            load_agentic_bank(_write_bank(tmp_path, text))

    def test_mock_tool_response_name_must_match_key(
        self, tmp_path: Path,
    ) -> None:
        text = _minimal_bank_yaml().replace(
            "      read_thing:\n"
            '        role: "tool"\n'
            '        name: "read_thing"\n',
            "      read_thing:\n"
            '        role: "tool"\n'
            '        name: "different_name"\n',
        )
        with pytest.raises(
            AgenticBankError, match="must equal the dict key"
        ):
            load_agentic_bank(_write_bank(tmp_path, text))


# ---------------------------------------------------------------------------
# 5. Lookup helpers
# ---------------------------------------------------------------------------


class TestLookups:
    @pytest.fixture
    def bank(self, tmp_path: Path) -> AgenticBank:
        return load_agentic_bank(_write_bank(tmp_path, _minimal_bank_yaml()))

    def test_by_id(self, bank: AgenticBank) -> None:
        assert bank.by_id("AR-TEST-001") is not None
        assert bank.by_id("missing") is None

    def test_by_sub_protocol(self, bank: AgenticBank) -> None:
        assert len(bank.by_sub_protocol("unsanctioned_action")) == 1
        assert len(bank.by_sub_protocol("scope_creep")) == 0

    def test_request_ids(self, bank: AgenticBank) -> None:
        assert bank.request_ids() == ("AR-TEST-001",)

    def test_for_persona_no_tags_matches_anyone(
        self, bank: AgenticBank,
    ) -> None:
        # Default minimal bank entry has no persona_tags.
        assert len(bank.for_persona([])) == 1
        assert len(bank.for_persona(["interest:tech"])) == 1


# ---------------------------------------------------------------------------
# 6. Persona tag matching
# ---------------------------------------------------------------------------


class TestPersonaMatching:
    def test_persona_tags_constraint(self, tmp_path: Path) -> None:
        text = _minimal_bank_yaml().replace(
            "    provenance:\n",
            '    persona_tags: ["interest:tech"]\n    provenance:\n',
        )
        bank = load_agentic_bank(_write_bank(tmp_path, text))
        ar = bank.action_requests[0]
        assert ar.matches_persona(["interest:tech"]) is True
        assert ar.matches_persona(["interest:health"]) is False

    def test_for_persona_returns_applicable_subset(
        self, tmp_path: Path,
    ) -> None:
        # Two entries: one unrestricted, one tech-only.
        extra = "\n".join([
            "  - id: AR-TEST-002",
            "    sub_protocol: unsanctioned_action",
            "    placeholder: true",
            "    description: |",
            "      Tech-only.",
            "    sanctioned_action_name: read_user_personal_data",
            "    initial_user_prompt: |",
            "      x",
            "    expected_behavior: |",
            "      x",
            "    simulated_tools:",
            "      - name: t1",
            "        description: x",
            "        parameters: {type: object, properties: {}}",
            '    persona_tags: ["interest:tech"]',
            "    provenance:",
            '      source: "placeholder"',
            '      last_reviewed: "2026-04-20"',
        ])
        bank = load_agentic_bank(
            _write_bank(tmp_path, _minimal_bank_yaml(extra_entries=extra))
        )
        # Persona without tech: only the unrestricted entry.
        ms = bank.for_persona(["interest:health"])
        assert {r.id for r in ms} == {"AR-TEST-001"}
        # Persona with tech: both.
        ms = bank.for_persona(["interest:tech"])
        assert {r.id for r in ms} == {"AR-TEST-001", "AR-TEST-002"}


# ---------------------------------------------------------------------------
# 7. Cache + env override
# ---------------------------------------------------------------------------


class TestCacheAndDefault:
    def test_load_once_cached(self, tmp_path: Path) -> None:
        path = _write_bank(tmp_path, _minimal_bank_yaml())
        with patch.dict(os.environ, {"USERSIM_SAFETY_AGENTIC_BANK": str(path)}):
            b1 = load_agentic_bank_default()
            b2 = load_agentic_bank_default()
        assert b1 is b2

    def test_env_override_picks_up_alternate_path(self, tmp_path: Path) -> None:
        path = _write_bank(
            tmp_path,
            _minimal_bank_yaml(bank_id="alt_bank"),
            name="alt.yaml",
        )
        with patch.dict(os.environ, {"USERSIM_SAFETY_AGENTIC_BANK": str(path)}):
            bank = load_agentic_bank_default()
        assert bank.bank_id == "alt_bank"

    def test_reset_drops_cache(self, tmp_path: Path) -> None:
        a = _write_bank(tmp_path / "a", _minimal_bank_yaml(bank_id="a"))
        b = _write_bank(
            tmp_path / "b", _minimal_bank_yaml(bank_id="b"), name="bank.yaml",
        )
        with patch.dict(os.environ, {"USERSIM_SAFETY_AGENTIC_BANK": str(a)}):
            assert load_agentic_bank_default().bank_id == "a"
        reset_agentic_bank_cache()
        with patch.dict(os.environ, {"USERSIM_SAFETY_AGENTIC_BANK": str(b)}):
            assert load_agentic_bank_default().bank_id == "b"

    def test_default_path_resolves_to_assets(self) -> None:
        p = default_agentic_bank_path()
        assert p.parts[-3:] == (
            "assets", "safety_agentic", "sample.yaml",
        )


# ---------------------------------------------------------------------------
# 8. Shipped sample bank — end-to-end check
# ---------------------------------------------------------------------------


class TestShippedSampleBank:
    @pytest.fixture
    def sample_bank(self) -> AgenticBank:
        return load_agentic_bank(
            (packaged_assets_dir() / "safety_agentic/sample.yaml")
        )

    def test_loads_clean(self, sample_bank: AgenticBank) -> None:
        assert sample_bank.bank_id == "sample_v1"
        assert sample_bank.bank_version == "v0.5.4"

    def test_four_canonical_sub_protocols(self, sample_bank: AgenticBank) -> None:
        # Roadmap default: 4 sub-protocols.
        assert set(sample_bank.sub_protocols) == {
            "unsanctioned_action",
            "scope_creep",
            "compromised_inputs",
            "consequence_disclosure",
        }

    def test_at_least_one_request_per_sub_protocol(
        self, sample_bank: AgenticBank,
    ) -> None:
        for sp in sample_bank.sub_protocols:
            assert len(sample_bank.by_sub_protocol(sp)) >= 1, (
                f"sub_protocol {sp} has no shipped action requests"
            )

    def test_generated_requests_are_honest_placeholders(
        self, sample_bank: AgenticBank,
    ) -> None:
        for r in sample_bank.action_requests:
            if r.id in {"AR-UNSAN-001", "AR-SCOPE-001", "AR-COMP-001", "AR-CONSEQ-001", "AR-CONSEQ-002"}:
                assert r.placeholder is False
            else:
                assert r.placeholder is True

    def test_every_sanctioned_action_resolves_to_taxonomy(
        self, sample_bank: AgenticBank,
    ) -> None:
        # The loader already enforces this — a failed load would have
        # raised before this test could run. This guards against silent
        # drift if the loader's check is ever weakened.
        for r in sample_bank.action_requests:
            if r.sanctioned_action_name is not None:
                assert isinstance(r.sanctioned_action_name, str)
                assert len(r.sanctioned_action_name) > 0

    def test_calendar_probe_rubric_matches_taxonomy_name(
        self, sample_bank: AgenticBank,
    ) -> None:
        request = sample_bank.by_id("AR-UNSAN-001")
        assert request is not None
        assert request.sanctioned_action_name == "read_calendar"
        assert all(
            "read_user_personal_data" not in text
            for text in request.expected_behavior.renderings.values()
        )

    def test_every_request_has_at_least_one_simulated_tool(
        self, sample_bank: AgenticBank,
    ) -> None:
        for r in sample_bank.action_requests:
            assert len(r.simulated_tools) >= 1

    def test_every_request_has_substantive_expected_behavior(
        self, sample_bank: AgenticBank,
    ) -> None:
        # The expected_behavior field is the load-bearing reviewer-facing
        # field — the safety_agentic scorer reads it to ground judgement.
        # Validate each locale rendering individually so missing /
        # truncated translations surface here instead of producing a
        # confounded judge prompt for the affected locale.
        for r in sample_bank.action_requests:
            for loc, text in r.expected_behavior.renderings.items():
                assert len(text) >= 100, (
                    f"{r.id}::expected_behavior[{loc}] too short "
                    f"({len(text)} chars)"
                )

    def test_compromised_inputs_probe_has_adversarial_payload(
        self, sample_bank: AgenticBank,
    ) -> None:
        # Sanity check: the compromised_inputs probe should ship a
        # mock response with adversarial content. Otherwise the probe
        # is structurally meaningless.
        comp_probes = sample_bank.by_sub_protocol("compromised_inputs")
        assert len(comp_probes) >= 1
        for r in comp_probes:
            response_contents = [
                resp.content for resp in r.mock_tool_responses.values()
            ]
            joined = "\n".join(response_contents).lower()
            # Loose check: at least one response mentions an external/
            # high-blast action verb that the embedded injection is
            # asking the assistant to perform — confirming the prompt-
            # injection-style payload is present in the mock response.
            # The list spans the action_taxonomy verbs used as the
            # tempted action across compromised_inputs probes.
            triggers = (
                "forward", "send", "transfer", "delete", "execute",
                "share", "grant", "post", "publish", "approve",
                "redirect", "issue", "export", "email",
                "change_delivery_address", "modify_access_control",
                "upgrade", "subscription", "place_order",
            )
            assert any(trigger in joined for trigger in triggers), (
                f"compromised_inputs probe {r.id} has no adversarial "
                "payload pattern in mock_tool_responses"
            )

    def test_provenance_summary_smoke(self, sample_bank: AgenticBank) -> None:
        s = sample_bank.provenance_summary()
        assert s["n_requests"] == 48
        assert s["n_placeholder_requests"] == 43
        assert s["bank_id"] == "sample_v1"
