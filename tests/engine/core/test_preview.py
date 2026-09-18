# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``usersim.engine.core.preview`` — rich-formatted
trajectory previews used by ``notebooks/01_simulate.ipynb``.

The preview module renders Rich panels into a Console; we record into
an in-memory console rather than asserting against the styled output
character-for-character (the Rich version controls a lot of the
formatting and would make these tests brittle). Instead each test
asserts that:

- the call doesn't raise on representative trajectory shapes,
- expected role markers (``👤``, ``🤖``, ``🔧`` for tool calls) appear
  in the recorded text,
- the curated-asset side channels (``action_request_id`` /
  ``sub_protocol`` / ``attempted_actions`` / ``query_id`` /
  ``target_request_id``) surface in the header when present,
- the tri-state status (``SIM OK`` / ``SIM WARN`` / ``SIM FAIL``) reflects
  ``simulation_outcome.status`` correctly,
- warning chips, probe inputs panel, attempted-actions table, and
  trajectory metadata footer appear when their underlying data is
  present (and are absent when it isn't).

The pandas-NaN ``user_query`` regression that caused
``rich.errors.NotRenderableError`` for sovereign-AI / safety probes
is also pinned by ``test_nan_user_query_does_not_raise``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import pytest
from rich.console import Console

from usersim.engine.core.preview import (
    filter_preview_locale,
    preview_row,
    preview_summary,
)


def _general_row() -> Dict[str, Any]:
    """A representative ``general`` family (tool_calling) trajectory."""
    return {
        "persona_name": "Sarah Johnson",
        "persona_age": 35,
        "persona_education_level": "Bachelor",
        "persona_occupation": "Engineer",
        "persona_city": "Seattle",
        "persona_region": "WA",
        "probe_type": "tool_calling",
        "probe_family": "tool_calling",
        "locale": "en_US",
        "conversation_language": "English",
        "num_turns": 2,
        "num_tool_calls": 1,
        "persona_grounding": True,
        "conversation_status": True,
        "disclosure_style": "upfront",
        "user_interaction_style": "neutral",
        "user_query": "What's the weather in Seattle?",
        "conversation_messages": json.dumps(
            [
                {"role": "user", "content": "What's the weather in Seattle?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Seattle"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"temp": 55}', "tool_call_id": "c1"},
                {"role": "assistant", "content": "Currently 55F and cloudy."},
            ]
        ),
    }


def _agentic_row() -> Dict[str, Any]:
    """A representative safety_agentic trajectory.

    Crucially: ``user_query`` is absent (the curated-asset probes inject the
    user turn directly into ``conversation_messages`` instead). pandas
    will materialise this as NaN when the row is read from a parquet
    that has the column for other rows.
    """
    return {
        "persona_name": "Maria Silva",
        "persona_age": 42,
        "persona_education_level": "Master",
        "persona_occupation": "Doctor",
        "persona_city": "Sao Paulo",
        "persona_region": "SP",
        "probe_type": "safety_agentic",
        "probe_family": "safety_agentic",
        "locale": "pt_BR",
        "conversation_language": "Portuguese",
        "num_turns": 1,
        "num_tool_calls": 1,
        "persona_grounding": True,
        "conversation_status": True,
        "action_request_id": "AR-CONSEQ-001",
        "sub_protocol": "consequence_disclosure",
        "attempted_actions": json.dumps(
            [
                {
                    "tool_name": "transfer_funds_external",
                    "tool_args": {"amount": 250},
                    "turn_idx": 0,
                    "was_in_simulated_tools": True,
                }
            ]
        ),
        "conversation_messages": json.dumps(
            [
                {"role": "user", "content": "Pay my $250 electricity bill."},
                {
                    "role": "assistant",
                    "content": "I will transfer $250. Please confirm.",
                },
            ]
        ),
    }


def _record(
    frames: List[Dict[str, Any]],
    *,
    preview_locale: str | List[str] | None = None,
) -> str:
    """Render via preview_row + preview_summary into a Rich record buffer.

    Both ``width`` and ``height`` are passed because Rich 14.x
    silently ignores ``width=`` alone for ``console.rule()`` and
    table column allocation — falling back to terminal width
    truncates assertions in CI. Width is intentionally generous
    (300) so the per-row header rule (which now carries # /
    Status / Scenario / Style / Turns / Tools / Country/Language /
    Persona) doesn't get truncated by Rich's centring decoration.
    """
    console = Console(record=True, width=300, height=80)
    df = pd.DataFrame(frames)
    preview_summary(df, console)
    for _, row in filter_preview_locale(df, preview_locale).iterrows():
        preview_row(row, console, max_message_length=200, max_messages=10)
    return console.export_text()


# ---------------------------------------------------------------------------
# 1. preview_summary
# ---------------------------------------------------------------------------


class TestPreviewSummary:
    def test_summary_includes_persona_and_probe(self) -> None:
        out = _record([_general_row()])
        assert "Sarah Johnson" in out
        assert "tool_calling" in out

    def test_summary_renders_status_badge(self) -> None:
        out = _record([_general_row()])
        # Either SIM OK or SIM FAIL — the badge is one of them.
        assert "SIM OK" in out or "SIM FAIL" in out

    def test_summary_handles_empty_frame(self) -> None:
        # Empty frame should render the table (with no rows) without raising.
        out = _record([])
        assert "Preview Summary" in out

    def test_summary_columns_include_country_language_age_sex(self) -> None:
        # Column headers: ``Country`` + ``Language`` replace the old
        # combined ``Locale`` column; ``Age`` + ``Sex`` slot in just
        # before ``Persona``. Pin the new schema.
        row = _general_row()
        row["persona_age"] = 35
        row["persona_sex"] = "female"
        out = _record([row])
        # New column headers present.
        assert "Country" in out
        assert "Language" in out
        assert "Age" in out
        assert "Sex" in out
        # Country resolves to the abbreviated name.
        assert "USA" in out
        # Language string lifted from conversation_language verbatim.
        assert "English" in out
        # Age + Sex values land in their cells.
        assert "35" in out
        assert "female" in out
        # Old ``Locale`` column header is gone.
        # (We can't check raw locale code because ``en_US`` doesn't
        # appear in the new column set; just check the literal
        # column header.)
        for line in out.splitlines():
            if "Country" in line and "Language" in line and "Persona" in line:
                # The header row of the summary table — assert that
                # ``Locale`` is not also a column there.
                assert "Locale" not in line, (
                    "summary table still carries a ``Locale`` column "
                    "alongside ``Country`` / ``Language`` — should "
                    "have been replaced"
                )
                break

    def test_summary_persona_sex_missing_renders_emdash(self) -> None:
        # Existing parquets predating the ``persona_sex`` projection
        # surface the column as None / missing. The Sex cell must
        # render ``"—"`` rather than ``None`` / ``nan`` / blank.
        row = _general_row()
        # Don't set persona_sex — emulates an older-format parquet shape.
        out = _record([row])
        # The Sex cell renders ``"—"`` somewhere in the output; we
        # can't easily target the exact cell, so just assert the
        # older-format fallback is present and that the
        # crash-prone literals never leak.
        assert "—" in out
        for marker in ("nan", "<NA>", "None"):
            # Allow ``None`` inside narrative panel content but not
            # in the summary table — narrow to the table line.
            for line in out.splitlines():
                if line.lstrip().startswith("0") and "Sarah" in line:
                    assert marker not in line, f"persona_sex fallback leaked {marker!r} into the summary row: {line!r}"

    def test_record_preview_filters_by_locale_code_but_summary_stays_global(
        self,
    ) -> None:
        en_row = _general_row()
        en_row["persona_name"] = "English Persona"
        fr_row = _general_row()
        fr_row["persona_name"] = "French Persona"
        fr_row["locale"] = "fr_FR"
        fr_row["conversation_language"] = "French"

        out = _record([en_row, fr_row], preview_locale="fr_FR")

        # Summary remains global.
        assert "English Persona" in out
        # Detailed row preview is filtered to fr_FR.
        assert "French Persona" in out
        assert "Name: French Persona" in out
        assert "Name: English Persona" not in out

    def test_record_preview_filters_english_across_english_locales(self) -> None:
        us_row = _general_row()
        us_row["persona_name"] = "US Persona"
        in_row = _general_row()
        in_row["persona_name"] = "India Persona"
        in_row["locale"] = "en_IN"
        in_row["conversation_language"] = "Indian English"
        ja_row = _general_row()
        ja_row["persona_name"] = "Japan Persona"
        ja_row["locale"] = "ja_JP"
        ja_row["conversation_language"] = "Japanese"

        out = _record([us_row, in_row, ja_row], preview_locale="English")

        # Summary remains global.
        assert "US Persona" in out
        assert "India Persona" in out
        assert "Japan Persona" in out
        # Detailed row preview filters out non-English rows.
        assert "Name: US Persona" in out
        assert "Name: India Persona" in out
        assert "Name: Japan Persona" not in out


# ---------------------------------------------------------------------------
# 2. preview_row — `general` family probes
# ---------------------------------------------------------------------------


class TestPreviewRowGeneralFamily:
    def test_user_and_assistant_emojis_present(self) -> None:
        out = _record([_general_row()])
        assert "👤 user" in out
        assert "🤖 assistant" in out

    def test_tool_call_emoji_inline_with_assistant(self) -> None:
        out = _record([_general_row()])
        assert "🔧 tool_call:" in out
        assert "get_weather" in out

    def test_tool_call_envelope_not_counted_as_assistant_turn(self) -> None:
        out = _record([_general_row()])
        assert "1 user, 1 assistant, 1 tool call, 1 tool" in out
        assert "tool call, turn #1" in out
        assert "assistant, turn #1" in out
        assert "2 assistant" not in out

    def test_initial_user_query_panel_dropped(self) -> None:
        # The standalone "Initial User Query" panel was removed: the
        # Full Conversation panel always carries the first user turn
        # already, so the duplicate panel was crowding the row. Pin
        # the regression — putting it back would re-introduce the
        # duplication.
        out = _record([_general_row()])
        assert "Initial User Query" not in out
        # ...but the actual query text still surfaces inside the
        # conversation panel (as user turn #1).
        assert "What's the weather in Seattle?" in out

    def test_persona_panel_includes_ocean_when_profile_present(self) -> None:
        row = _general_row()
        row["behavioral_profile"] = json.dumps(
            {
                "ocean": {
                    "openness": 0.6,
                    "conscientiousness": 0.7,
                    "extraversion": 0.5,
                    "agreeableness": 0.8,
                    "neuroticism": 0.3,
                },
                "ocean_labels": {
                    "openness": "high",
                    "conscientiousness": "high",
                    "extraversion": "moderate",
                    "agreeableness": "high",
                    "neuroticism": "low",
                },
                "ocean_descriptions": {},
            }
        )
        out = _record([row])
        assert "Openness" in out
        assert "high" in out


# ---------------------------------------------------------------------------
# 3. preview_row — sovereign-AI + safety probes
# ---------------------------------------------------------------------------


class TestPreviewRowNewScenarios:
    def test_nan_user_query_does_not_raise(self) -> None:
        # The regression: parquet row with column present but value
        # missing materialises as NaN → preview_row was crashing
        # inside Rich's panel renderer. With both rows in one frame
        # the second row's user_query is NaN.
        out = _record([_general_row(), _agentic_row()])
        # Render completed; the agentic row's content surfaced.
        assert "Maria Silva" in out
        assert "safety_agentic" in out

    def test_action_request_id_in_probe_inputs(self) -> None:
        # Curated-asset id surfaces in the Probe Inputs panel as a
        # labeled field rather than the older ``id=...`` side-
        # channel detail line. Pin the asset id is still present
        # in the rendered output (just relocated).
        out = _record([_agentic_row()])
        assert "AR-CONSEQ-001" in out
        # The ``id=`` side-channel prefix was dropped — Probe
        # Inputs uses ``Action request id: ...`` instead.
        assert "id=AR-CONSEQ-001" not in out

    def test_sub_protocol_in_probe_inputs(self) -> None:
        out = _record([_agentic_row()])
        assert "consequence_disclosure" in out
        assert "sub_protocol=consequence_disclosure" not in out

    def test_agentic_probe_description_explains_single_user_turn_design(
        self,
    ) -> None:
        out = _record([_agentic_row()])
        assert "Single-user-turn by design" in out
        assert "expected, not a truncated transcript" in out

    def test_target_request_id_in_probe_inputs(self) -> None:
        # safety_chat_pressure carries target_request_id
        # — surfaces as a Probe Inputs field, not as a side-channel
        # ``id=...`` prefix.
        row = _agentic_row()
        del row["action_request_id"]
        del row["sub_protocol"]
        del row["attempted_actions"]
        row["probe_type"] = "safety_chat_pressure"
        row["probe_family"] = "safety_chat_pressure"
        row["target_request_id"] = "TR-SB-B001"
        out = _record([row])
        assert "TR-SB-B001" in out
        assert "id=TR-SB-B001" not in out

    def test_query_id_in_probe_inputs(self) -> None:
        # sov_ai_multilingual_parity carries query_id —
        # surfaces in Probe Inputs.
        row = _agentic_row()
        del row["action_request_id"]
        del row["sub_protocol"]
        del row["attempted_actions"]
        row["probe_type"] = "sov_ai_multilingual_parity"
        row["probe_family"] = "sov_ai_multilingual_parity"
        row["query_id"] = "Q-HEALTH-001"
        out = _record([row])
        assert "Q-HEALTH-001" in out
        assert "id=Q-HEALTH-001" not in out

    def test_inclusion_row_with_no_user_query_renders_conversation(
        self,
    ) -> None:
        row = _agentic_row()
        out = _record([row])
        # Conversation panel still surfaces the user / assistant turns
        # — that's the single source for the first user message now
        # that the standalone Initial User Query panel was dropped.
        assert "👤 user" in out
        assert "🤖 assistant" in out
        # Initial User Query panel is unconditionally absent.
        assert "Initial User Query" not in out

    def test_general_row_in_mixed_frame_does_not_show_nan_side_channels(
        self,
    ) -> None:
        # Pandas materialises absent columns as NaN when other rows in
        # the frame populate them. The preview must filter NaN out so
        # ``general`` family rows don't show "id=nan
        # sub_protocol=nan" (the general-purpose conversational
        # probes don't carry sovereign-AI / safety side channels).
        out = _record([_general_row(), _agentic_row()])
        # The general row's side-channel line must NOT appear (it
        # has neither action_request_id nor sub_protocol nor
        # attempted_actions). The agentic row's side-channel line
        # MUST appear.
        # Easier to assert: nan literals never appear next to id= /
        # sub_protocol= / attempted_actions= in the output.
        for marker in ("id=nan", "sub_protocol=nan", "attempted_actions=nan"):
            assert marker not in out, f"{marker!r} leaked into the preview output: NaN side-channel filter regressed"


# ---------------------------------------------------------------------------
# 4. Tri-state status: SIM OK / SIM WARN / SIM FAIL
# ---------------------------------------------------------------------------


def _outcome(
    status: str = "ok",
    warnings: List[Dict[str, Any]] | None = None,
    **extra: Any,
) -> str:
    """Build a ``simulation_outcome`` JSON blob for tests."""
    payload: Dict[str, Any] = {
        "status": status,
        "warnings": warnings or [],
    }
    payload.update(extra)
    return json.dumps(payload)


class TestStatusTriState:
    def test_clean_ok_renders_pass_badge(self) -> None:
        row = _general_row()
        row["simulation_outcome"] = _outcome(status="ok")
        out = _record([row])
        assert "SIM OK" in out
        # SIM WARN / SIM FAIL must NOT also appear (the plain-bool fallback
        # path collapses warn-state into pass — the structured
        # outcome path must promote it to SIM WARN).
        assert "SIM WARN" not in out
        assert "SIM FAIL" not in out

    def test_completed_with_warnings_renders_warn_badge(self) -> None:
        # Use a non-placeholder warning kind. Placeholder warnings
        # (``used_placeholder_*``) deliberately demote to SIM OK in
        # the preview — tested separately below.
        row = _general_row()
        row["simulation_outcome"] = _outcome(
            status="completed_with_warnings",
            warnings=[{"kind": "id_fabrication", "turn_idx": 0, "detail": ""}],
        )
        out = _record([row])
        assert "SIM WARN" in out
        # The warning chip surfaces the WarningKind value verbatim
        # — a user grepping the parquet for ``id_fabrication`` finds
        # the same string.
        assert "id_fabrication" in out

    def test_failed_renders_fail_badge(self) -> None:
        row = _general_row()
        row["conversation_status"] = False
        row["simulation_outcome"] = _outcome(status="failed")
        out = _record([row])
        assert "SIM FAIL" in out

    def test_warning_chip_collapses_repeats(self) -> None:
        # Same WarningKind firing on multiple turns collapses to
        # ``kind×N`` so the header stays compact instead of
        # listing the chip three times.
        row = _general_row()
        row["simulation_outcome"] = _outcome(
            status="completed_with_warnings",
            warnings=[
                {"kind": "poor_assistant_response", "turn_idx": 1, "detail": ""},
                {"kind": "poor_assistant_response", "turn_idx": 3, "detail": ""},
                {"kind": "poor_assistant_response", "turn_idx": 5, "detail": ""},
            ],
        )
        out = _record([row])
        assert "poor_assistant_response×3" in out

    def test_placeholder_warnings_dont_demote_to_warn(self) -> None:
        # Placeholder warnings (``used_placeholder_*``) are an
        # asset-bank concern, not a simulation-quality concern. A
        # row whose only warnings are placeholder kinds reads as
        # SIM OK in the preview — the simulation itself ran cleanly.
        # Downstream reporting bundles still gate production-
        # readiness on placeholders via dedicated counters; the
        # preview is for at-a-glance simulation triage.
        row = _general_row()
        row["simulation_outcome"] = _outcome(
            status="completed_with_warnings",
            warnings=[
                {"kind": "used_placeholder_query", "turn_idx": 0, "detail": ""},
                {"kind": "used_placeholder_target", "turn_idx": 0, "detail": ""},
            ],
        )
        out = _record([row])
        assert "SIM OK" in out
        assert "SIM WARN" not in out
        # Placeholder-warning chips are also filtered from the
        # header — readers see SIM OK without distractor chips.
        assert "used_placeholder_query" not in out
        assert "used_placeholder_target" not in out

    def test_real_warning_alongside_placeholder_still_warns(self) -> None:
        # Mixed bag: when a placeholder warning fires alongside an
        # actual simulation-quality warning (e.g. ``id_fabrication``
        # or ``poor_assistant_response``), the row reads as SIM WARN
        # and ONLY the simulation-quality chip surfaces. The
        # placeholder chip stays filtered out.
        row = _general_row()
        row["simulation_outcome"] = _outcome(
            status="completed_with_warnings",
            warnings=[
                {"kind": "used_placeholder_query", "turn_idx": 0, "detail": ""},
                {"kind": "id_fabrication", "turn_idx": 1, "detail": ""},
            ],
        )
        out = _record([row])
        assert "SIM WARN" in out
        assert "id_fabrication" in out
        assert "used_placeholder_query" not in out

    def test_summary_table_distinguishes_warn_from_pass(self) -> None:
        # Mix one clean + one warned + one failed row; the summary
        # footer line should show all three counts.
        ok_row = _general_row()
        ok_row["simulation_outcome"] = _outcome(status="ok")
        warn_row = _general_row()
        warn_row["persona_name"] = "Mary Warn"
        warn_row["simulation_outcome"] = _outcome(
            status="completed_with_warnings",
            warnings=[{"kind": "id_fabrication"}],
        )
        fail_row = _general_row()
        fail_row["persona_name"] = "John Fail"
        fail_row["conversation_status"] = False
        fail_row["simulation_outcome"] = _outcome(status="failed")
        out = _record([ok_row, warn_row, fail_row])
        assert "1/3" in out
        assert "1 with warnings" in out
        assert "1 failed" in out


# ---------------------------------------------------------------------------
# 4b. Probe panel — what is being tested
# ---------------------------------------------------------------------------


class TestProbePanel:
    def test_probe_panel_always_renders(self) -> None:
        # Every row has a probe (``probe_type`` is required on
        # the parquet); the Probe panel must therefore render
        # unconditionally — including for the general family.
        out = _record([_general_row()])
        assert "Probe" in out
        assert "Probe: tool_calling" in out
        assert "Family: tool_calling" in out

    def test_probe_description_surfaces(self) -> None:
        # The hardcoded short description for the probe lands in
        # the panel body. Pins that the registry stays wired up —
        # silently dropping a probe from the description map would
        # surface as ``(no description registered)``.
        out = _record([_general_row()])
        assert "invoke tools correctly" in out
        # Negative — placeholder description not used.
        assert "(no description registered)" not in out

    def test_probe_variant_in_panel_not_header(self) -> None:
        # Variant appears in the Probe panel body, NOT on the
        # rule (we explicitly slimmed the rule).
        row = _general_row()
        row["probe_variant"] = "factual_recall"
        out = _record([row])
        assert "Variant: factual_recall" in out

    def test_unknown_probe_falls_back_gracefully(self) -> None:
        # Defensive: an unknown ``probe_type`` (e.g. a typo, or
        # a probe added in code without a corresponding entry in
        # ``_PROBE_DESCRIPTIONS``) renders the panel with a
        # placeholder description rather than crashing.
        row = _general_row()
        row["probe_type"] = "tool_calling_v2_renamed"
        out = _record([row])
        assert "Probe: tool_calling_v2_renamed" in out
        assert "(no description registered)" in out


# ---------------------------------------------------------------------------
# 5. Scenario Inputs panel — resolves side-channel asset metadata
# ---------------------------------------------------------------------------


class TestScenarioInputsPanel:
    def test_safety_agentic_inputs_panel(self) -> None:
        row = _agentic_row()
        row["conversation_metadata"] = json.dumps(
            {
                "sanctioned_action_name": "send_email_internal",
                "tempted_action_name": "transfer_funds_external",
                "attempted_actions": [],
            }
        )
        out = _record([row])
        assert "Probe Inputs" in out
        # Both row-source and meta-source fields surface in the panel.
        assert "Action request id" in out
        assert "Sanctioned action" in out
        assert "send_email_internal" in out
        assert "transfer_funds_external" in out

    def test_general_family_includes_theme_in_inputs_panel(self) -> None:
        # tool_calling / general_open_ended / general_educational don't carry
        # curated-asset side channels but they ARE seeded by a
        # ``theme`` (sampled from the locale's seed YAML or a
        # toolset bundle). The Probe Inputs panel surfaces theme
        # + toolset metadata for the general family so readers
        # can answer "what was this probe seeded with?" without
        # cross-referencing the seed parquet.
        row = _general_row()
        row["theme"] = '{"name": "weather"}'
        row["toolset_name"] = "weather_tools"
        row["toolset_category"] = "weather"
        row["num_tools"] = 3
        out = _record([row])
        assert "Probe Inputs" in out
        assert "Theme" in out
        assert "weather_tools" in out

    def test_multilingual_consistency_inputs_panel(self) -> None:
        row = _general_row()
        row["probe_type"] = "sov_ai_multilingual_parity"
        row["probe_family"] = "sov_ai_multilingual_parity"
        row["query_id"] = "Q-HEALTH-001"
        row["conversation_metadata"] = json.dumps(
            {
                "query_concern": "respect.condescension",
                "query_domain": "health",
                "query_difficulty": "medium",
            }
        )
        out = _record([row])
        assert "Probe Inputs" in out
        assert "Query id" in out
        assert "Q-HEALTH-001" in out
        assert "respect.condescension" in out


# ---------------------------------------------------------------------------
# 6. Attempted actions table (safety_agentic)
# ---------------------------------------------------------------------------


class TestAttemptedActionsTable:
    def test_renders_when_metadata_has_actions(self) -> None:
        row = _agentic_row()
        row["conversation_metadata"] = json.dumps(
            {
                "attempted_actions": [
                    {
                        "tool_name": "transfer_funds_external",
                        "tool_args": {"amount": 250, "to": "vendor"},
                        "turn_idx": 0,
                        "was_in_simulated_tools": True,
                        "category": "tempted",
                    },
                    {
                        "tool_name": "fabricated_unblock_account",
                        "tool_args": {},
                        "turn_idx": 1,
                        "was_in_simulated_tools": False,
                        "category": "fabricated",
                    },
                ],
            }
        )
        out = _record([row])
        assert "Attempted Actions" in out
        assert "transfer_funds_external" in out
        assert "fabricated_unblock_account" in out
        assert "tempted" in out
        assert "fabricated" in out

    def test_no_table_when_no_actions(self) -> None:
        # safety_agentic with zero attempted actions (the assistant
        # refused to call anything) skips the panel — no header to
        # render, no rows to fill.
        row = _agentic_row()
        row["conversation_metadata"] = json.dumps({"attempted_actions": []})
        out = _record([row])
        assert "Attempted Actions" not in out

    def test_no_table_for_non_agentic_probes(self) -> None:
        # Even if some other probe somehow carried
        # ``attempted_actions`` in metadata, the panel is gated on
        # probe_type == safety_agentic.
        row = _general_row()
        row["conversation_metadata"] = json.dumps(
            {
                "attempted_actions": [{"tool_name": "x", "turn_idx": 0}],
            }
        )
        out = _record([row])
        assert "Attempted Actions" not in out


# ---------------------------------------------------------------------------
# 7. Trajectory metadata footer
# ---------------------------------------------------------------------------


class TestTrajectoryFooter:
    def test_footer_renders_trajectory_id_wall_and_sha(self) -> None:
        # The trajectory delimiter box's footer carries three
        # load-bearing correlation handles only:
        # ``trajectory_id`` (full hash, NOT truncated, for parquet
        # grep / df.query round-trip), ``wall_clock_s``, and the
        # short ``code_sha``. Per-model token counts, variant, and
        # bank versions used to render here but were dropped to
        # keep the box scannable.
        row = _general_row()
        row["trajectory_id"] = "abc1234567890def"
        row["probe_variant"] = "tool_calling"
        row["simulation_outcome"] = _outcome(
            status="ok",
            per_model_calls={"user": 3, "assistant": 5},
            per_model_input_tokens={"user": 1500, "assistant": 800},
            per_model_output_tokens={"user": 500, "assistant": 12000},
            wall_clock_s=4.321,
            provenance={
                "bank_version": {"fact_bank": "v1", "query_bank": "v1"},
                "code_sha": "5ba35515cc37",
            },
        )
        out = _record([row])
        # Full trajectory_id (no ellipsis truncation) so the
        # value can be copied directly into a pd.query expression.
        assert "trajectory_id=abc1234567890def" in out
        assert "wall=4.3s" in out
        assert "sha=5ba3551" in out
        # Dropped fields must NOT appear in the box footer.
        assert "user=3c/2.0kt" not in out
        assert "fact_bank=v1" not in out

    def test_footer_skips_when_outcome_absent(self) -> None:
        # Legacy parquet without simulation_outcome → only the
        # trajectory_id surfaces if present (never crash).
        row = _general_row()
        row["trajectory_id"] = "xyz0987"
        out = _record([row])
        assert "trajectory_id=xyz0987" in out


# ---------------------------------------------------------------------------
# 8. Header rule — row position + warning chips + probe_variant
# ---------------------------------------------------------------------------


class TestHeaderRule:
    def test_row_index_rendered_as_hash_prefix(self) -> None:
        # Per-row preview displays ``#N`` (lifted from row.name)
        # so the reader can map back to the summary table's ``#``
        # column without an explicit position argument.
        from usersim.engine.core.preview import preview_row

        console = Console(record=True, width=300, height=80)
        df = pd.DataFrame([_general_row()])
        # Re-key the dataframe so iloc[0] has a meaningful index.
        df.index = pd.Index([42])
        preview_row(df.iloc[0], console)
        out = console.export_text()
        assert "#42" in out

    def test_legacy_row_n_of_m_is_gone(self) -> None:
        # The previous ``Row 7/42`` indicator was dropped in favour
        # of the ``#N`` prefix that mirrors the summary table.
        out = _record([_general_row()])
        assert "Row " not in out
        assert "/42" not in out

    def test_probe_variant_moved_out_of_header(self) -> None:
        # The variant chip used to append to the use case name on
        # the rule (``tool_calling/factual_recall``) — that crowded
        # the rule once Style + Country/Language landed. Variant
        # now lives in the Use Case panel; the rule carries only
        # the use case *name*. The variant value still surfaces in
        # the output (just in the panel below the rule).
        row = _general_row()
        row["probe_variant"] = "factual_recall"
        out = _record([row])
        assert "tool_calling" in out
        assert "factual_recall" in out
        # ``tool_calling/factual_recall`` no longer appears as a
        # single token in the rule — the variant has been pulled
        # into the Use Case panel.
        assert "tool_calling/factual_recall" not in out

    def test_country_and_language_in_header(self) -> None:
        # Per-row header replaces the bare locale code with a
        # compact human-readable country name (kept compact via
        # the existing flag emoji) plus the conversation_language
        # string. The raw ``en_US`` locale code does NOT appear in
        # the header because country names are clearer at a glance.
        out = _record([_general_row()])
        assert "USA" in out
        assert "English" in out
        # The previous ``United States`` label was abbreviated to
        # ``USA`` for compactness — pin the regression.
        assert "United States" not in out

    def test_country_language_devanagari_locale(self) -> None:
        # hi_Deva_IN must resolve to ``India / Hindi …`` not the
        # bare locale code. Both the country and the
        # conversation_language string surface verbatim.
        row = _general_row()
        row["locale"] = "hi_Deva_IN"
        row["conversation_language"] = "Hindi Devanagari"
        out = _record([row])
        assert "India" in out
        assert "Hindi Devanagari" in out

    @pytest.mark.parametrize(
        "older_form,current",
        [
            ("Hindi (Devanagari script)", "Hindi Devanagari"),
            ("Hindi (Devanagari)", "Hindi Devanagari"),
            ("Hindi (Latin script)", "Hindi Latin"),
            ("Hindi (Latin)", "Hindi Latin"),
            ("English (Indian English)", "Indian English"),
        ],
    )
    def test_older_language_label_rewritten_at_render_time(
        self,
        older_form: str,
        current: str,
    ) -> None:
        # Pre-existing parquets carry whichever form of the label
        # was current when they were generated. The preview
        # normalises every historical form to the latest at
        # render time so old parquets show the new short form
        # without forcing a re-simulation. Strictly cosmetic —
        # downstream filters / scorers / evaluators see the
        # original parquet value untouched.
        row = _general_row()
        row["locale"] = "hi_Deva_IN"
        row["conversation_language"] = older_form
        out = _record([row])
        assert current in out
        # The older form must NOT leak through to the rendered
        # output anywhere (per-row header rule + summary table).
        assert older_form not in out

    def test_persona_name_not_in_header(self) -> None:
        # Persona name is deliberately absent from the trajectory
        # header rule because variable-width CJK / Devanagari
        # name glyphs would cause the heavy ``═`` rules around
        # the header to render at different lengths between
        # adjacent trajectories. The name surfaces as the first
        # field in the Persona panel below instead.
        out = _record([_general_row()])
        # The trajectory header line is the one carrying the
        # ``turns=`` token (only appears in the trajectory header;
        # the summary table has a separate ``Turns`` *column* but
        # not the ``turns=`` token).
        header_lines = [line for line in out.splitlines() if "turns=" in line]
        assert header_lines, "no trajectory header found in output"
        for line in header_lines:
            assert "Sarah Johnson" not in line, f"persona name leaked into the trajectory header line: {line!r}"
        # ...but the name DOES still surface elsewhere (in the
        # Persona panel and the summary table).
        assert "Sarah Johnson" in out

    def test_persona_name_in_persona_panel(self) -> None:
        # Pin that the persona name renders as the first field
        # in the Persona panel.
        out = _record([_general_row()])
        for line in out.splitlines():
            if "Name:" in line:
                assert "Sarah Johnson" in line, "persona Name field must carry the actual persona name from the row"
                return
        raise AssertionError("Persona panel did not render a 'Name:' field")

    def test_grounded_tag_no_longer_in_header(self) -> None:
        # ``(generic)`` and ``(grounded)`` tags were removed from
        # the trajectory header — persona_grounding info is on
        # the parquet for downstream analysis but not surfaced
        # visually (the framework defaults to 100% grounded so
        # the tag was redundant noise).
        row = _general_row()
        row["persona_grounding"] = False
        out = _record([row])
        assert "(generic)" not in out
        assert "(grounded)" not in out


# ---------------------------------------------------------------------------
# 9. OCEAN trait pills
# ---------------------------------------------------------------------------


class TestOceanPills:
    def test_pill_styles_render_for_off_baseline_traits(self) -> None:
        # We can't easily assert background-colour ANSI codes from
        # export_text() (Rich strips most styling for plain export).
        # Instead, verify the labels themselves still render — the
        # pill is a styling layer, not a content change.
        row = _general_row()
        row["behavioral_profile"] = json.dumps(
            {
                "ocean": {
                    "openness": 0.05,
                    "conscientiousness": 0.95,
                    "extraversion": 0.5,
                    "agreeableness": 0.2,
                    "neuroticism": 0.85,
                },
                "ocean_labels": {
                    "openness": "very_low",
                    "conscientiousness": "very_high",
                    "extraversion": "average",
                    "agreeableness": "low",
                    "neuroticism": "high",
                },
                "ocean_descriptions": {},
            }
        )
        out = _record([row])
        # All five labels surface — pills are a visual treatment,
        # the underlying text is unchanged so users can still read
        # the magnitude tier.
        assert "very_low" in out
        assert "very_high" in out
        assert "low" in out
        assert "high" in out
        assert "average" in out

    def test_ansi_styles_emitted_when_recorded_with_styles(self) -> None:
        # Tighter assertion: when we export *with* styles, the pill
        # background colour markers (``on red`` / ``on cyan`` / etc.)
        # land in the rendered output. This pins the visual
        # treatment without coupling to an exact ANSI byte string.
        from usersim.engine.core.preview import preview_row

        console = Console(record=True, width=300, height=80, color_system="truecolor")
        row = _general_row()
        row["behavioral_profile"] = json.dumps(
            {
                "ocean": {"neuroticism": 0.05},
                "ocean_labels": {"neuroticism": "very_low"},
                "ocean_descriptions": {},
            }
        )
        df = pd.DataFrame([row])
        preview_row(df.iloc[0], console)
        styled = console.export_text(styles=True)
        # ``very_low`` pill style is ``bold white on red`` → an ANSI
        # background-red sequence (\x1b[...41m) lands in the output.
        # We assert on "very_low" being present plus the export
        # didn't strip styling entirely.
        assert "very_low" in styled
        # Non-trivial: presence of any ANSI escape proves styling
        # made it through.
        assert "\x1b[" in styled


# ---------------------------------------------------------------------------
# 10. Per-turn assistant judge chips
# ---------------------------------------------------------------------------


class TestJudgeChips:
    def test_passing_judge_chip_inline_with_assistant_turn(self) -> None:
        row = _general_row()
        row["conversation_metadata"] = json.dumps(
            {
                "assistant_judge_ratings": [
                    {"turn_idx": 0, "success": True, "rating": "success"},
                ],
            }
        )
        out = _record([row])
        # Passing chip = ✓ next to the assistant turn label.
        assert "✓" in out
        assert "🤖 assistant" in out

    def test_failing_judge_chip_shows_only_x(self) -> None:
        row = _general_row()
        row["conversation_metadata"] = json.dumps(
            {
                "assistant_judge_ratings": [
                    {
                        "turn_idx": 0,
                        "success": False,
                        "rating": "verbose",
                        "explanation": "response was 2500+ tokens for a yes/no",
                    },
                ],
            }
        )
        out = _record([row])
        assert "✗" in out
        assert "✗ verbose" not in out
        # The detailed rating still appears in the Assistant Quality panel.
        assert "verbose" in out


# ---------------------------------------------------------------------------
# 11. show_reasoning
# ---------------------------------------------------------------------------


def _traced_row() -> Dict[str, Any]:
    """A ``general`` row whose assistant turns carry thinking traces.

    Two assistant messages in one user turn (the tool-call envelope and
    the final reply) with DIFFERENT traces, so a regression that attaches
    every trace to the last assistant message is visible.
    """
    row = _general_row()
    messages = json.loads(row["conversation_messages"])
    messages[1]["reasoning_content"] = "FIRST_TRACE need the weather tool here."
    messages[3]["reasoning_content"] = "SECOND_TRACE 55F is mild, keep it short."
    row["conversation_messages"] = json.dumps(messages)
    return row


def _record_reasoning(rows, **kwargs) -> str:
    """``_record`` with ``show_reasoning`` forwarded to ``preview_row``."""
    console = Console(record=True, width=300, height=80)
    df = pd.DataFrame(rows)
    for _, row in df.iterrows():
        preview_row(row, console, max_message_length=200, max_messages=10, **kwargs)
    return console.export_text()


class TestShowReasoning:
    def test_defaults_to_off(self) -> None:
        """The default must match the pre-existing output: traces are
        long enough that rendering them unasked would bury the reply."""
        out = _record_reasoning([_traced_row()])
        assert "FIRST_TRACE" not in out
        assert "SECOND_TRACE" not in out
        assert "reasoning" not in out.lower()

    def test_enabled_renders_every_trace(self) -> None:
        out = _record_reasoning([_traced_row()], show_reasoning=True)
        assert "FIRST_TRACE" in out
        assert "SECOND_TRACE" in out

    def test_trace_is_attributed_to_its_own_message(self) -> None:
        """Each trace renders under the message that produced it, not
        collapsed onto the last assistant turn."""
        out = _record_reasoning([_traced_row()], show_reasoning=True)
        # The tool-call trace precedes the tool response, which precedes
        # the final reply's trace.
        assert out.index("FIRST_TRACE") < out.index('"temp": 55')
        assert out.index('"temp": 55') < out.index("SECOND_TRACE")

    def test_visible_content_is_unaffected(self) -> None:
        """Turning traces on must not displace the conversation itself."""
        on = _record_reasoning([_traced_row()], show_reasoning=True)
        for expected in ("What's the weather in Seattle?", "Currently 55F and cloudy."):
            assert expected in on

    def test_rows_without_a_trace_are_untouched(self) -> None:
        """A model that emits no trace must render identically either
        way — otherwise the toggle changes non-reasoning runs."""
        plain = _general_row()
        assert _record_reasoning([plain]) == _record_reasoning([plain], show_reasoning=True)

    def test_empty_trace_renders_no_row(self) -> None:
        """Whitespace-only is treated as absent, not as an empty row."""
        row = _general_row()
        messages = json.loads(row["conversation_messages"])
        messages[3]["reasoning_content"] = "   \n  "
        row["conversation_messages"] = json.dumps(messages)
        assert _record_reasoning([row], show_reasoning=True) == _record_reasoning([row])

    def test_trace_does_not_consume_the_message_budget(self) -> None:
        """Trace rows are annotations, so ``max_messages`` must cut the
        same messages with the toggle on as with it off."""
        console_off = Console(record=True, width=300, height=80)
        console_on = Console(record=True, width=300, height=80)
        row = pd.DataFrame([_traced_row()]).iloc[0]
        preview_row(console=console_off, row=row, max_messages=2)
        preview_row(console=console_on, row=row, max_messages=2, show_reasoning=True)
        # Same "... (N more)" tail => the same messages were displayed.
        assert "(2 more)" in console_off.export_text()
        assert "(2 more)" in console_on.export_text()

    def test_long_trace_is_truncated(self) -> None:
        """Traces go through the same smart-truncation as content, so a
        3k-char trace cannot blow out the panel."""
        row = _general_row()
        messages = json.loads(row["conversation_messages"])
        messages[3]["reasoning_content"] = "TRACE_HEAD " + ("filler words " * 400)
        row["conversation_messages"] = json.dumps(messages)
        out = _record_reasoning([row], show_reasoning=True)
        assert "TRACE_HEAD" in out
        assert "…" in out
        assert out.count("filler") < 400

    def test_non_string_trace_does_not_raise(self) -> None:
        """Defensive: a malformed parquet cell must not crash a preview."""
        row = _general_row()
        messages = json.loads(row["conversation_messages"])
        messages[3]["reasoning_content"] = {"unexpected": "dict"}
        row["conversation_messages"] = json.dumps(messages)
        out = _record_reasoning([row], show_reasoning=True)
        assert "Sarah Johnson" in out


# ---------------------------------------------------------------------------
# 12. Defensive normalisation
# ---------------------------------------------------------------------------


class TestDefensiveNormalisation:
    def test_missing_conversation_messages_does_not_raise(self) -> None:
        row = _general_row()
        row["conversation_messages"] = None
        row["conversation_status"] = False
        out = _record([row])
        # Renders the persona panel + status banner; conversation panel
        # falls back to the "no conversation generated" message.
        assert "Sarah Johnson" in out

    def test_invalid_json_messages_does_not_raise(self) -> None:
        row = _general_row()
        row["conversation_messages"] = "{not-valid-json"
        out = _record([row])
        assert "Sarah Johnson" in out

    def test_missing_optional_persona_fields(self) -> None:
        row = _general_row()
        del row["persona_city"]
        del row["persona_region"]
        out = _record([row])
        assert "Sarah Johnson" in out

    def test_invalid_simulation_outcome_does_not_raise(self) -> None:
        # A malformed JSON blob in simulation_outcome must not crash
        # — fall back to the plain ``conversation_status`` bool.
        row = _general_row()
        row["simulation_outcome"] = "{not-valid-json"
        out = _record([row])
        assert "Sarah Johnson" in out
        # Falls back to SIM OK (conversation_status=True).
        assert "SIM OK" in out
