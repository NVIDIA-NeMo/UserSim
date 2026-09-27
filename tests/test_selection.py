# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for the trajectory-selection layer (``selection`` package).

Two fixture sources:

- the shared ``trajectory_df`` / ``evaluator_df`` from ``tests/conftest.py``
  (same frames ``test_extraction_example.py`` uses) for the universal-gate
  and health-gate cases, and
- bespoke synthetic rows built here for cases the shared fixtures cannot
  exercise (registry-unknown axes, scorer-error recording, no_eval, holdout),
  since those fixtures only span general_open_ended + tool_calling with the
  helpfulness/accuracy judge axes.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from usersim.selection import (
    MANIFEST_NAME,
    load_run,  # noqa: F401  (imported to assert it is exported)
    select_trajectories,
    write_curated_dataset,
)
from usersim.selection.io import parse_repo_id
from usersim.selection.profiles import (
    ALL_FLOORS,
    GENEROUS,
    SelectionProfile,
    applicable_gates,
)
from usersim.selection.schemas import (
    LOGICAL_GROUPS,
    OPT_IN_ONLY,
    project,
    resolve_columns,
)

# Shared-fixture trajectory ids (see tests/conftest.py).
T_OK = "t000000000000001"  # general_open_ended, ok, scored
T_FAILED = "t000000000000002"  # failed + skipped (no_assistant_messages)
T_WARN_ROLE = "t000000000000003"  # completed_with_warnings, n_user_role_violations=1
T_TOOL_OK = "t000000000000004"  # tool_calling ok, tool_use scorer ok
T_SKIPPED = "t000000000000005"  # ok status but eval skipped (envelope_match)
T_TOOL_ERR = "t000000000000006"  # tool_calling ok, tool_use scorer errored


def _ids(df) -> set[str]:
    return {str(x) for x in df["trajectory_id"].tolist()}


# ---------------------------------------------------------------------------
# Universal gates on the shared fixtures
# ---------------------------------------------------------------------------


class TestUniversalGates:
    def test_generous_keeps_clean_drops_skipped_and_failed(self, trajectory_df, evaluator_df):
        res = select_trajectories(trajectory_df, evaluator_df, profile=GENEROUS)
        assert res.summary.total == 6
        # Passing: the OK/warned rows that have a real (non-skipped) eval.
        assert _ids(res.curated) == {T_OK, T_WARN_ROLE, T_TOOL_OK, T_TOOL_ERR}
        # Dropped: the failed (also skipped) row and the envelope-match skip.
        assert res.summary.dropped == 2
        assert set(res.summary.drop_reasons) == {"skipped"}

    def test_funnel_counts_are_consistent(self, trajectory_df, evaluator_df):
        res = select_trajectories(trajectory_df, evaluator_df, profile=GENEROUS)
        assert res.summary.passed + res.summary.dropped == res.summary.total
        assert sum(res.summary.drop_reasons.values()) == res.summary.dropped
        assert sum(res.summary.kept_by_group.values()) == res.summary.passed

    def test_status_gate_excludes_failed(self, trajectory_df, evaluator_df):
        res = select_trajectories(trajectory_df, evaluator_df, profile=GENEROUS)
        assert T_FAILED not in _ids(res.curated)

    def test_require_eval_present_drops_missing_eval(self, trajectory_df, evaluator_df):
        # Drop eval rows for the two tool_calling trajectories -> they become no_eval.
        partial = evaluator_df[~evaluator_df["trajectory_id"].isin([T_TOOL_OK, T_TOOL_ERR])]
        res = select_trajectories(trajectory_df, partial, profile=GENEROUS)
        assert res.summary.drop_reasons.get("no_eval") == 2
        assert {T_TOOL_OK, T_TOOL_ERR}.isdisjoint(_ids(res.curated))


# ---------------------------------------------------------------------------
# Simulator-health gate (opt-in)
# ---------------------------------------------------------------------------


class TestHealthGate:
    def test_role_violation_drops_only_when_capped(self, trajectory_df, evaluator_df):
        # Default GENEROUS keeps the warned row...
        assert T_WARN_ROLE in _ids(select_trajectories(trajectory_df, evaluator_df, profile=GENEROUS).curated)
        # ...but capping user-role violations at 0 drops it.
        strict_health = dataclasses.replace(GENEROUS, max_user_role_violations=0)
        res = select_trajectories(trajectory_df, evaluator_df, profile=strict_health)
        assert T_WARN_ROLE not in _ids(res.curated)
        assert res.summary.drop_reasons.get("health:n_user_role_violations") == 1


# ---------------------------------------------------------------------------
# Quality score
# ---------------------------------------------------------------------------


class TestQualityScore:
    def test_quality_is_normalized_mean_of_1_5_axes(self, trajectory_df, evaluator_df):
        res = select_trajectories(trajectory_df, evaluator_df, profile=GENEROUS)
        row = res.curated[res.curated["trajectory_id"] == T_OK].iloc[0]
        # helpfulness (5+4)/2=4.5 -> 0.9 ; accuracy (5+5)/2=5.0 -> 1.0 ; mean 0.95
        assert row["selection_quality_score"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Bespoke synthetic rows for cases the shared fixtures cannot cover
# ---------------------------------------------------------------------------


def _outcome(status="ok", n_turns=2, **counters):
    base = {
        "status": status,
        "n_turns": n_turns,
        "n_user_role_violations": 0,
        "n_user_language_violations": 0,
        "n_fourth_wall_triggers": 0,
        "n_assistant_inline_failures": 0,
        "provenance": {"code_sha": "deadbeef", "scenario_prompt_version": "v1.0"},
    }
    base.update(counters)
    return json.dumps(base)


def _cell(axes=None, scorers=None, skipped=False):
    return json.dumps(
        {
            "envelope": {
                "judge_aliases": ["j"],
                "judge_families": ["openai"],
                "axes": list((axes or {}).keys()),
                "scorers": list((scorers or {}).keys()),
                "prompt_version": "v1.0",
                "evaluator_version": "v1.0",
            },
            "axes": {a: {"j": {"score": s}} for a, s in (axes or {}).items()},
            "scorers": scorers or {},
            "skipped": skipped,
            "skipped_reason": None,
        }
    )


def _row(
    tid,
    *,
    probe_family="general_open_ended",
    persona="p",
    n_turns=2,
    axes=None,
    scorers=None,
    skipped=False,
    status="ok",
    **counters,
):
    return {
        "trajectory_id": tid,
        "persona_uuid": persona,
        "probe_family": probe_family,
        "locale": "hi_Deva_IN",
        "num_turns": n_turns,
        "conversation_messages": json.dumps([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]),
        "simulation_outcome": _outcome(status=status, n_turns=n_turns, **counters),
        "assistant_eval": _cell(axes=axes, scorers=scorers, skipped=skipped),
    }


def _frame(rows):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(rows)


class TestBespokeGates:
    def test_min_turns_is_opt_in(self):
        df = _frame([_row("single", n_turns=1, axes={"accuracy": 5})])
        # Default min_turns=1: a one-turn conversation survives.
        assert _ids(select_trajectories(df, df, profile=GENEROUS).curated) == {"single"}
        # min_turns=2 drops it.
        prof = dataclasses.replace(GENEROUS, min_turns=2)
        res = select_trajectories(df, df, profile=prof)
        assert not _ids(res.curated)
        assert res.summary.drop_reasons.get("min_turns") == 1

    def test_registry_unknown_axis_still_gates(self):
        # role_adherence is produced by the evaluator but absent from the
        # capability registry; the floor must still fire via cell presence.
        rows = [
            _row("low", axes={"role_adherence": 1, "accuracy": 5}),
            _row("high", axes={"role_adherence": 5, "accuracy": 5}),
        ]
        df = _frame(rows)
        res = select_trajectories(df, df, profile=GENEROUS)
        assert _ids(res.curated) == {"high"}
        assert res.summary.drop_reasons.get("role_adherence") == 1

    def test_not_applicable_scorer_axis_passes(self):
        # A profile floors a tool_use axis, but the row's probe ran no such
        # scorer -> not applicable -> the row passes (generous).
        prof = dataclasses.replace(GENEROUS, axis_floors={**GENEROUS.axis_floors, "overall": 4.0})
        df = _frame([_row("noscore", axes={"accuracy": 5})])
        assert _ids(select_trajectories(df, df, profile=prof).curated) == {"noscore"}

    def test_scorer_error_recorded_not_dropped(self):
        prof = dataclasses.replace(GENEROUS, axis_floors={"overall": 4.0}, critical_axes=(ALL_FLOORS,))
        rows = [
            _row(
                "ok_scorer",
                probe_family="tool_calling",
                scorers={"tool_use": {"scores": {"overall": {"score": 5}}, "status_proposal": True}},
            ),
            _row(
                "err_scorer",
                probe_family="tool_calling",
                scorers={"tool_use": {"scores": {}, "status_proposal": True, "error": "RuntimeError: boom"}},
            ),
        ]
        df = _frame(rows)
        res = select_trajectories(df, df, profile=prof)
        # The errored scorer's row is NOT dropped...
        assert _ids(res.curated) == {"ok_scorer", "err_scorer"}
        # ...but the scorer error is recorded for visibility.
        err_row = res.all[res.all["trajectory_id"] == "err_scorer"].iloc[0]
        assert "overall:scorer_error" in json.loads(err_row["selection_failed_gates"])

    def test_critical_vs_soft_floor(self):
        # A non-critical floor records but does not drop; critical drops.
        soft = SelectionProfile(name="soft", axis_floors={"helpfulness": 4.0}, critical_axes=())
        df = _frame([_row("weak", axes={"helpfulness": 2})])
        res = select_trajectories(df, df, profile=soft)
        assert _ids(res.curated) == {"weak"}  # soft floor: kept
        row = res.all[res.all["trajectory_id"] == "weak"].iloc[0]
        assert "helpfulness" in json.loads(row["selection_failed_gates"])

        hard = dataclasses.replace(soft, critical_axes=(ALL_FLOORS,))
        assert not _ids(select_trajectories(df, df, profile=hard).curated)


class TestHoldout:
    def test_holdout_deterministic_and_labelled(self):
        rows = [_row(f"t{i}", persona=f"persona-{i}", axes={"accuracy": 5}) for i in range(40)]
        df = _frame(rows)
        prof = dataclasses.replace(GENEROUS, holdout_fraction=0.5)
        a = select_trajectories(df, df, profile=prof)
        b = select_trajectories(df, df, profile=prof)
        # Both splits present and the assignment is stable across runs.
        assert set(a.summary.holdout) == {"train", "holdout"}
        assert a.summary.holdout == b.summary.holdout
        split_a = dict(zip(a.curated["trajectory_id"], a.curated["selection_split"]))
        split_b = dict(zip(b.curated["trajectory_id"], b.curated["selection_split"]))
        assert split_a == split_b


class TestStability:
    def test_added_columns_and_profile_label(self):
        df = _frame([_row("x", axes={"accuracy": 5})])
        res = select_trajectories(df, df, profile=GENEROUS)
        for col in (
            "selection_passed",
            "selection_quality_score",
            "selection_axis_scores",
            "selection_failed_gates",
            "selection_drop_reason",
            "selection_profile",
            "selection_split",
        ):
            assert col in res.curated.columns
        assert res.curated.iloc[0]["selection_profile"] == "generous:v1.0"

    def test_idempotent_across_runs(self, trajectory_df, evaluator_df):
        a = select_trajectories(trajectory_df, evaluator_df, profile=GENEROUS)
        b = select_trajectories(trajectory_df, evaluator_df, profile=GENEROUS)
        assert _ids(a.curated) == _ids(b.curated)
        sa = dict(zip(a.curated["trajectory_id"], a.curated["selection_axis_scores"]))
        sb = dict(zip(b.curated["trajectory_id"], b.curated["selection_axis_scores"]))
        assert sa == sb


# ---------------------------------------------------------------------------
# Applicable-gate resolution
# ---------------------------------------------------------------------------


class TestApplicableGates:
    def test_gate_only_for_present_axes(self):
        from usersim.taxonomy.eval_cell import decode_cell

        cell = decode_cell(_cell(axes={"accuracy": 5}))  # no safety axis present
        gates = {g.axis for g in applicable_gates(GENEROUS, cell)}
        assert "accuracy" in gates
        assert "safety" not in gates  # floored, but absent from this cell


class TestMaxRecords:
    def test_cap_keeps_highest_quality(self):
        # 5 passing rows with distinct (ungated) helpfulness -> quality; cap at 2.
        rows = [_row(f"t{i}", persona=f"p{i}", axes={"helpfulness": i, "accuracy": 5}) for i in range(1, 6)]
        df = _frame(rows)
        traj, ev = df.drop(columns=["assistant_eval"]), df[["trajectory_id", "assistant_eval"]]
        res = select_trajectories(traj, ev, profile=GENEROUS, max_records=2)
        assert res.summary.passed == 5
        assert res.summary.selected == 2
        assert res.summary.max_records == 2
        # Highest accuracy (5, 4) kept.
        assert _ids(res.curated) == {"t5", "t4"}

    def test_no_cap_keeps_all(self):
        rows = [_row(f"t{i}", persona=f"p{i}", axes={"accuracy": 5}) for i in range(3)]
        df = _frame(rows)
        traj, ev = df.drop(columns=["assistant_eval"]), df[["trajectory_id", "assistant_eval"]]
        res = select_trajectories(traj, ev, profile=GENEROUS, max_records=None)
        assert res.summary.selected == res.summary.passed == 3


class TestStratifiedCap:
    def test_balances_across_probes(self):
        from usersim.selection.engine import select_trajectories as st

        # Probe A: 10 high-quality rows; Probe B: 10 lower-quality rows.
        rows = []
        for i in range(10):
            rows.append(
                _row(f"a{i}", persona=f"a{i}", probe_family="tool_calling", axes={"helpfulness": 5, "accuracy": 5})
            )
            rows.append(
                _row(
                    f"b{i}",
                    persona=f"b{i}",
                    probe_family="general_open_ended",
                    axes={"helpfulness": 2 + (i % 3), "accuracy": 5},
                )
            )
        df = _frame(rows)
        traj, ev = df.drop(columns=["assistant_eval"]), df[["trajectory_id", "assistant_eval"]]

        # Global top-K would take all 10 high-quality A rows; stratified balances.
        glob = st(traj, ev, profile=GENEROUS, max_records=8, stratify_by=None)
        by_probe = glob.curated["probe_family"].value_counts().to_dict()
        assert by_probe.get("tool_calling", 0) == 8  # A dominates globally

        strat = st(traj, ev, profile=GENEROUS, max_records=8, stratify_by=["probe_family"])
        bp = strat.curated["probe_family"].value_counts().to_dict()
        assert bp == {"tool_calling": 4, "general_open_ended": 4}  # balanced
        assert strat.summary.stratify_by == ("probe_family",)

    def test_redistributes_when_a_stratum_is_small(self):
        from usersim.selection.engine import _allocate_quota

        # 2 strata, one tiny: tiny gets all it has, the rest goes to the big one.
        assert _allocate_quota({"a": 2, "b": 100}, 10) == {"a": 2, "b": 8}
        # even split when both have capacity
        assert _allocate_quota({"a": 100, "b": 100}, 10) == {"a": 5, "b": 5}
        # fewer slots than strata: one each, in order, until exhausted
        alloc = _allocate_quota({"a": 5, "b": 5, "c": 5}, 2)
        assert sum(alloc.values()) == 2 and all(v <= 1 for v in alloc.values())
        # capped at total availability
        assert sum(_allocate_quota({"a": 3, "b": 4}, 100).values()) == 7

    def test_within_stratum_keeps_highest_quality(self):
        from usersim.selection.engine import select_trajectories as st

        rows = [
            _row(f"b{i}", persona=f"b{i}", probe_family="general_open_ended", axes={"helpfulness": i, "accuracy": 5})
            for i in range(1, 6)
        ]
        df = _frame(rows)
        traj, ev = df.drop(columns=["assistant_eval"]), df[["trajectory_id", "assistant_eval"]]
        res = st(traj, ev, profile=GENEROUS, max_records=2, stratify_by=["probe_family"])
        # Highest helpfulness (5, 4) kept within the single probe.
        assert _ids(res.curated) == {"b5", "b4"}


class TestMaxPerLocale:
    @staticmethod
    def _locale_rows(locales, per, hq=True):
        rows = []
        for loc in locales:
            for i in range(per):
                r = _row(f"{loc}{i}", persona=f"{loc}{i}", axes={"helpfulness": 5, "accuracy": 5})
                r["locale"] = loc
                rows.append(r)
        return rows

    def _traj_ev(self, rows):
        df = _frame(rows)
        return df.drop(columns=["assistant_eval"]), df[["trajectory_id", "assistant_eval"]]

    def test_caps_each_locale_independently(self):
        from usersim.selection.engine import select_trajectories as st

        traj, ev = self._traj_ev(self._locale_rows(["ta_Taml_IN", "bn_Beng_IN"], 6))
        res = st(traj, ev, profile=GENEROUS, max_per_locale=3, stratify_by=None)
        assert res.curated["locale"].value_counts().to_dict() == {
            "ta_Taml_IN": 3,
            "bn_Beng_IN": 3,
        }
        assert res.summary.max_per_locale == 3
        assert len(res.curated) == 6

    def test_within_locale_keeps_highest_quality(self):
        from usersim.selection.engine import select_trajectories as st

        rows = []
        for i in range(1, 6):
            r = _row(f"t{i}", persona=f"t{i}", axes={"helpfulness": i, "accuracy": 5})
            r["locale"] = "ta_Taml_IN"
            rows.append(r)
        traj, ev = self._traj_ev(rows)
        res = st(traj, ev, profile=GENEROUS, max_per_locale=2, stratify_by=None)
        assert _ids(res.curated) == {"t5", "t4"}  # top-2 by quality within the locale

    def test_composes_with_global_max_records(self):
        from usersim.selection.engine import select_trajectories as st

        traj, ev = self._traj_ev(self._locale_rows(["ta_Taml_IN", "bn_Beng_IN", "te_Telu_IN"], 6))
        # per-locale cap 4 -> 12 kept, then global cap 5 -> 5 total.
        res = st(traj, ev, profile=GENEROUS, max_per_locale=4, max_records=5, stratify_by=None)
        assert len(res.curated) == 5
        assert all(v <= 4 for v in res.curated["locale"].value_counts().to_dict().values())

    def test_none_is_noop(self):
        from usersim.selection.engine import select_trajectories as st

        traj, ev = self._traj_ev(self._locale_rows(["ta_Taml_IN"], 4))
        res = st(traj, ev, profile=GENEROUS, max_per_locale=None)
        assert len(res.curated) == 4
        assert res.summary.max_per_locale is None


class TestGenerationModelProvenance:
    """The dataset card's model must come from the run, not from a human.

    A hand-entered value silently goes stale the first time someone switches
    provider and forgets to update it, and the card is published.
    """

    @staticmethod
    def _write_manifest(root, run_id: str, assistant: str) -> None:
        from usersim.engine.core.manifest import manifest_path

        p = manifest_path(root, run_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "models": {
                        "assistant_model": {"alias": "assistant_model", "model": assistant},
                    },
                }
            )
        )

    def test_reads_the_model_from_the_run_manifest(self, tmp_path):
        from usersim.selection import resolve_generation_model

        self._write_manifest(tmp_path, "111", "gpt-5.6-luna")
        assert resolve_generation_model(tmp_path, "111") == "gpt-5.6-luna"

    def test_concatenated_runs_name_every_distinct_model(self, tmp_path):
        """A card claiming one model for a mixed dataset would be wrong."""
        from usersim.selection import resolve_generation_model

        self._write_manifest(tmp_path, "111", "gpt-5.6-luna")
        self._write_manifest(tmp_path, "222", "gpt-5.6-luna")  # duplicate
        self._write_manifest(tmp_path, "333", "google/gemma-4-31b-it")
        got = resolve_generation_model(tmp_path, "111+222+333")
        assert got == "gpt-5.6-luna, google/gemma-4-31b-it"

    def test_explicit_value_wins(self, tmp_path):
        """For runs whose manifest is absent or came from elsewhere."""
        from usersim.selection import resolve_generation_model

        self._write_manifest(tmp_path, "111", "gpt-5.6-luna")
        assert resolve_generation_model(tmp_path, "111", explicit="other") == "other"

    def test_missing_manifest_is_not_fatal(self, tmp_path):
        """Curation must not fail because a run predates the manifest."""
        from usersim.selection import resolve_generation_model

        assert resolve_generation_model(tmp_path, "nope") is None
        assert resolve_generation_model(tmp_path, "") is None


class TestSchema:
    def test_full_drops_run_keeps_rest(self):
        cols = resolve_columns("full", ["run", "locale", "probe_family", "trajectory_id", "x"])
        assert "run" not in cols
        assert set(cols) == {"locale", "probe_family", "trajectory_id", "x"}

    @pytest.mark.parametrize("schema", [None, "standard", "compact"])
    def test_default_export_omits_opt_in_columns(self, schema):
        """Nothing about a person leaves by default that nobody asked for.

        The verbatim fields reproduce rows of the source persona dataset, and
        ``persona_religion_language_context`` carries protected attributes.
        Both stay reachable, but only when named.
        """
        available = [
            "run",
            "locale",
            "probe_family",
            "trajectory_id",
            "persona_uuid",
            "conversation_messages",
            "persona_age_bin",
            "user_interaction_style",
            *OPT_IN_ONLY,
        ]
        cols = resolve_columns(schema, available)
        assert not (set(cols) & OPT_IN_ONLY), f"{schema} leaked {set(cols) & OPT_IN_ONLY}"
        # The derived columns are the point of the export, so they stay.
        assert "persona_age_bin" in cols
        assert "user_interaction_style" in cols

    @pytest.mark.parametrize(
        "schema,expected",
        [
            ("full", OPT_IN_ONLY),
            ("persona_verbatim", set(LOGICAL_GROUPS["persona_verbatim"])),
            ("persona_protected", set(LOGICAL_GROUPS["persona_protected"])),
            ("persona_name", {"persona_name"}),
        ],
    )
    def test_opt_in_columns_are_reachable_when_named(self, schema, expected):
        available = ["run", "locale", "probe_family", *OPT_IN_ONLY]
        assert set(resolve_columns(schema, available)) & OPT_IN_ONLY == expected

    def test_asking_for_verbatim_does_not_pull_in_protected(self):
        """The two groups are separate so one cannot smuggle in the other."""
        available = ["locale", "probe_family", *OPT_IN_ONLY]
        cols = resolve_columns("persona_verbatim", available)
        assert "persona_religion_language_context" not in cols

    def test_compact_expands_groups_and_skips_missing(self):
        available = [
            "run",
            "locale",
            "probe_family",
            "conversation_messages",
            "conversation_metadata",
            "conversation_status",
            "trajectory_id",
            "persona_uuid",
            "probe_type",
            "probe_variant",
            "theme",
            "persona_age_bin",
            "user_interaction_style",
            "tools",
            "num_tools",
            "persona_name",
            "persona_age",  # verbatim; explicit opt-in
            # num_turns intentionally absent -> "if available" skip
            "simulation_traces",
            "assistant_eval",  # should be excluded
            "persona_religion_language_context",  # protected; explicit opt-in
        ]
        cols = resolve_columns("compact", available)
        assert "conversation_messages" in cols  # group expansion
        assert "persona_age_bin" in cols  # group expansion
        assert "run" in cols  # compact keeps run
        assert "num_turns" not in cols  # absent -> skipped
        assert "simulation_traces" not in cols and "assistant_eval" not in cols
        assert "persona_religion_language_context" not in cols
        assert "persona_name" not in cols and "persona_age" not in cols

    def test_compact_includes_finance_metadata_when_present(self):
        # financial_services gold-metadata is first-class in compact exports so
        # the training-data flywheel keeps it; absent on non-finance rows.
        available = [
            "run",
            "locale",
            "probe_family",
            "trajectory_id",
            "finance_task_id",
            "task_tier",
            "task_contract_version",
            "institution_id",
            "institution_type",
            "domain",
            "gold_document_ids",
            "gold_tool_sequence",
            "expected_state_deltas",
            "retrieved_document_ids",
            "attempted_tool_names",
            "dynamic_category_id",
            "dynamic_subtopic_hint",
            "taxonomy_version",
        ]
        cols = resolve_columns("compact", available)
        for c in (
            "finance_task_id",
            "task_tier",
            "institution_id",
            "domain",
            "gold_tool_sequence",
            "expected_state_deltas",
            "dynamic_category_id",
        ):
            assert c in cols, c
        # Non-finance frame: finance columns simply absent (no error).
        non_finance = resolve_columns(
            "compact",
            ["run", "locale", "probe_family", "trajectory_id"],
        )
        assert "finance_task_id" not in non_finance

    def test_finance_group_matches_probe_contract(self):
        # Schema-parity guard across the package boundary: the selection
        # `finance` group must match the probe's canonical column contract, so
        # a rename/add on either side can't silently drop export columns.
        from usersim.engine.probes.financial_services.generator import (
            FINANCE_TRAJECTORY_COLUMNS,
        )
        from usersim.selection.schemas import LOGICAL_GROUPS

        assert set(LOGICAL_GROUPS["finance"]) == set(FINANCE_TRAJECTORY_COLUMNS)

    def test_explicit_list_and_partition_forced(self):
        cols = resolve_columns(
            ["trajectory_id", "conversation"], ["trajectory_id", "conversation_messages", "locale", "probe_family"]
        )
        # group expanded, and partition columns force-appended for the writer
        assert "conversation_messages" in cols
        assert "locale" in cols and "probe_family" in cols

    def test_project_returns_resolved_columns(self):
        df = _frame([_row("x", axes={"accuracy": 5})])
        # _row frames have no 'run'; add one to mimic load_run
        df["run"] = "1700000000"
        proj = project(df, "compact")
        assert "run" in proj.columns
        assert "simulation_outcome" not in proj.columns


class TestLoadRuns:
    def test_concatenates_and_labels_multiple_runs(self, monkeypatch):
        import usersim.selection.io as sio

        frames = {
            "r1": _frame([_row("a", persona="pa"), _row("b", persona="pb")]),
            "r2": _frame([_row("c", persona="pc")]),
        }

        def fake_load_run(traj, ev, *, run):
            return frames[run].copy(), None, run

        monkeypatch.setattr(sio, "load_run", fake_load_run)
        traj, ev, label = sio.load_runs("traj", "eval", runs=["r1", "r2"])
        assert label == "r1+r2"
        assert _ids(traj) == {"a", "b", "c"}

    def test_dedup_by_trajectory_id(self, monkeypatch):
        import usersim.selection.io as sio

        def fake_load_run(traj, ev, *, run):
            # Both runs surface the same trajectory id "dup".
            return _frame([_row("dup", persona="p"), _row(run, persona=run)]), None, run

        monkeypatch.setattr(sio, "load_run", fake_load_run)
        traj, ev, label = sio.load_runs("traj", "eval", runs=["x", "y"])
        assert (traj["trajectory_id"] == "dup").sum() == 1  # deduped
        assert _ids(traj) == {"dup", "x", "y"}

    def test_single_run_delegates(self, monkeypatch):
        import usersim.selection.io as sio

        monkeypatch.setattr(sio, "load_run", lambda *a, **k: ("T", "E", k["run"]))
        assert sio.load_runs("t", "e", runs=["only"]) == ("T", "E", "only")
        assert sio.load_runs("t", "e", runs="latest") == ("T", "E", "latest")


class TestRepoId:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://huggingface.co/datasets/nvidia/usersim-x", "nvidia/usersim-x"),
            ("hf.co/datasets/org/name", "org/name"),
            ("org/name", "org/name"),
            ("org/name/", "org/name"),
        ],
    )
    def test_parse_repo_id(self, raw, expected):
        assert parse_repo_id(raw) == expected


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


class TestWrite:
    def test_roundtrip_and_manifest(self, tmp_path, trajectory_df, evaluator_df):
        from usersim.engine.core.storage import read_partitioned_dataset

        res = select_trajectories(trajectory_df, evaluator_df, profile=GENEROUS)
        root = write_curated_dataset(res.curated, tmp_path, "1700000000", GENEROUS, summary=res.summary)
        assert root.name == "profile=generous"
        manifest = json.loads((root / MANIFEST_NAME).read_text())
        assert manifest["funnel"]["passed"] == res.summary.passed
        assert manifest["provenance"]["code_sha"]  # captured from outcome
        assert manifest["provenance"]["judge_models"] == ["vendor/judge-a", "vendor/judge-b"]
        back = read_partitioned_dataset(root)
        assert len(back) == res.summary.passed
        assert bool(back["selection_passed"].all())

    def test_profile_namespacing(self, tmp_path, trajectory_df, evaluator_df):
        res = select_trajectories(trajectory_df, evaluator_df, profile=GENEROUS)
        strict_like = dataclasses.replace(GENEROUS, name="other")
        r1 = write_curated_dataset(res.curated, tmp_path, "1700000000", GENEROUS, summary=res.summary)
        r2 = write_curated_dataset(res.curated, tmp_path, "1700000000", strict_like, summary=res.summary)
        assert r1 != r2 and r1.exists() and r2.exists()

    def test_empty_result_writes_manifest_only(self, tmp_path, trajectory_df, evaluator_df):
        drop_all = dataclasses.replace(GENEROUS, allowed_statuses=())
        res = select_trajectories(trajectory_df, evaluator_df, profile=drop_all)
        assert res.summary.passed == 0
        root = write_curated_dataset(res.curated, tmp_path, "1700000000", drop_all, summary=res.summary)
        assert (root / MANIFEST_NAME).exists()
        assert not list(root.rglob("*.parquet"))
