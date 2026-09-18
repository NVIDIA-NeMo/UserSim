# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``reporting/eval_manifest.py``.

Pivotal invariants:

- Schema v2 round-trips losslessly through JSON.
- Legacy v1 (single-object manifests on disk from before this module
  existed) is read transparently as a one-pass v2 manifest.
- ``record_eval_sample_pass`` APPENDS by default (preserves the audit
  trail across locale-targeted re-evals); ``fresh=True`` resets the
  history for whole-run rebuilds.
- ``consensus_mode`` returns the shared mode when all passes agree,
  ``"mixed"`` when they don't, and ``"unknown"`` for an empty manifest.
- A run-id mismatch on the existing manifest refuses to merge -- the
  caller likely has the wrong path, and silent cross-run contamination
  would be far worse than a loud error.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from usersim.reporting.eval_manifest import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    EvalSampleManifest,
    EvalSamplePass,
    read_eval_sample_manifest,
    record_eval_sample_pass,
)


# ─── Schema round-trip ──────────────────────────────────────────────


class TestSchemaRoundTrip:
    """JSON serialisation + parsing must be lossless. Lock the on-disk
    contract so a future field addition doesn't silently drop bytes."""

    def _build(self) -> EvalSampleManifest:
        return EvalSampleManifest(
            run_id="1778723989",
            passes=[
                EvalSamplePass(
                    wrote_at_epoch=1778723990,
                    mode="full",
                    n=None,
                    random_seed=42,
                    eval_locales=None,
                    n_selected=1024,
                    n_total=1024,
                    trajectory_ids=["T1", "T2"],
                ),
                EvalSamplePass(
                    wrote_at_epoch=1778800000,
                    mode="full",
                    n=None,
                    random_seed=42,
                    eval_locales=["en_US"],
                    n_selected=128,
                    n_total=1024,
                    trajectory_ids=["T1"],
                ),
            ],
        )

    def test_to_dict_and_back(self) -> None:
        m = self._build()
        roundtripped = EvalSampleManifest.from_dict(m.to_dict())
        assert roundtripped.run_id == m.run_id
        assert len(roundtripped.passes) == len(m.passes)
        for a, b in zip(roundtripped.passes, m.passes):
            assert a == b

    def test_serialisation_includes_schema_version(self) -> None:
        m = self._build()
        d = m.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION

    def test_serialisation_preserves_pass_order(self) -> None:
        m = self._build()
        d = m.to_dict()
        # Order matters -- the dashboard's "latest pass" semantics
        # depend on the final element being the most-recent write.
        assert d["passes"][0]["wrote_at_epoch"] == 1778723990
        assert d["passes"][1]["wrote_at_epoch"] == 1778800000

    def test_eval_locales_none_vs_list_preserved(self) -> None:
        """``eval_locales=None`` (whole-run write) and ``eval_locales=[]``
        (programming error -- should never be written, but defensive)
        are semantically distinct. Round-trip must not coerce one to
        the other."""
        m = EvalSampleManifest(
            run_id="r",
            passes=[
                EvalSamplePass(
                    wrote_at_epoch=0,
                    mode="full",
                    n=None,
                    random_seed=0,
                    eval_locales=None,
                    n_selected=0,
                    n_total=0,
                    trajectory_ids=[],
                ),
                EvalSamplePass(
                    wrote_at_epoch=0,
                    mode="full",
                    n=None,
                    random_seed=0,
                    eval_locales=[],
                    n_selected=0,
                    n_total=0,
                    trajectory_ids=[],
                ),
            ],
        )
        roundtripped = EvalSampleManifest.from_dict(m.to_dict())
        assert roundtripped.passes[0].eval_locales is None
        assert roundtripped.passes[1].eval_locales == []


# ─── Legacy v1 read shim ────────────────────────────────────────────


class TestLegacyV1Read:
    """Single-object manifests on disk (from before this module
    existed) must be readable as a one-pass v2 manifest. The bug we're
    fixing landed because we used to OVERWRITE this format on every
    sampling pass; existing artefacts on disk must keep loading
    without manual migration."""

    def test_legacy_format_promotes_to_single_pass(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / MANIFEST_FILENAME
        # The exact shape the user pasted in the bug report.
        path.write_text(
            json.dumps(
                {
                    "run_id": "1778723989",
                    "mode": "full",
                    "n": None,
                    "random_seed": 42,
                    "eval_locales": ["en_US"],
                    "n_selected": 128,
                    "n_total": 1024,
                    "trajectory_ids": ["T1", "T2"],
                },
                indent=2,
            )
        )
        m = read_eval_sample_manifest(path)
        assert m is not None
        assert m.run_id == "1778723989"
        assert len(m.passes) == 1
        p = m.passes[0]
        assert p.mode == "full"
        assert p.eval_locales == ["en_US"]
        assert p.n_selected == 128
        assert p.n_total == 1024
        assert p.trajectory_ids == ["T1", "T2"]
        # Legacy manifests have no wrote_at_epoch -> 0 (the "unknown"
        # convention).
        assert p.wrote_at_epoch == 0

    def test_legacy_without_random_seed_defaults_to_negative_one(
        self,
        tmp_path: Path,
    ) -> None:
        # Some very-early manifests may have omitted random_seed.
        path = tmp_path / MANIFEST_FILENAME
        path.write_text(
            json.dumps(
                {
                    "run_id": "r",
                    "mode": "full",
                    "n_selected": 10,
                    "n_total": 10,
                    "trajectory_ids": [],
                }
            )
        )
        m = read_eval_sample_manifest(path)
        assert m is not None
        assert m.passes[0].random_seed == -1


# ─── record_eval_sample_pass ────────────────────────────────────────


class TestRecordEvalSamplePass:
    def test_first_pass_creates_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / MANIFEST_FILENAME
        m = record_eval_sample_pass(
            path,
            run_id="r1",
            mode="full",
            n=None,
            random_seed=42,
            eval_locales=None,
            n_selected=1024,
            n_total=1024,
            trajectory_ids=["T1", "T2"],
            wrote_at_epoch=1000,
        )
        assert m.run_id == "r1"
        assert len(m.passes) == 1
        # And the file on disk matches.
        on_disk = json.loads(path.read_text())
        assert on_disk["schema_version"] == SCHEMA_VERSION
        assert on_disk["run_id"] == "r1"
        assert len(on_disk["passes"]) == 1

    def test_second_pass_appends_by_default(self, tmp_path: Path) -> None:
        """The headline bug-fix: re-eval pass APPENDS rather than
        overwriting. The cumulative audit trail is preserved."""
        path = tmp_path / MANIFEST_FILENAME
        record_eval_sample_pass(
            path,
            run_id="r1",
            mode="full",
            n=None,
            random_seed=42,
            eval_locales=None,
            n_selected=1024,
            n_total=1024,
            trajectory_ids=["T1", "T2"],
            wrote_at_epoch=1000,
        )
        # Locale-targeted re-eval: en_US only.
        m = record_eval_sample_pass(
            path,
            run_id="r1",
            mode="full",
            n=None,
            random_seed=42,
            eval_locales=["en_US"],
            n_selected=128,
            n_total=1024,
            trajectory_ids=["T1"],
            wrote_at_epoch=2000,
        )
        assert len(m.passes) == 2
        assert m.passes[0].eval_locales is None
        assert m.passes[0].wrote_at_epoch == 1000
        assert m.passes[1].eval_locales == ["en_US"]
        assert m.passes[1].wrote_at_epoch == 2000
        # to_dict preserves both passes.
        d = json.loads(path.read_text())
        assert len(d["passes"]) == 2

    def test_fresh_drops_existing_passes(self, tmp_path: Path) -> None:
        """Whole-run rewrites (``OVERWRITE_EVAL_RUN=True`` with no
        ``EVAL_LOCALES``) reset the audit trail because the eval
        parquet itself is being rmtree'd. Preserving stale pass
        records when the underlying data is gone would be its own
        kind of lie."""
        path = tmp_path / MANIFEST_FILENAME
        record_eval_sample_pass(
            path,
            run_id="r1",
            mode="full",
            n=None,
            random_seed=42,
            eval_locales=None,
            n_selected=1024,
            n_total=1024,
            trajectory_ids=["T1"],
        )
        # Whole-run rewrite -> fresh=True -> history dropped.
        m = record_eval_sample_pass(
            path,
            run_id="r1",
            mode="per_locale",
            n=10,
            random_seed=42,
            eval_locales=None,
            n_selected=80,
            n_total=1024,
            trajectory_ids=["T2"],
            fresh=True,
        )
        assert len(m.passes) == 1
        assert m.passes[0].mode == "per_locale"

    def test_run_id_mismatch_raises(self, tmp_path: Path) -> None:
        """Defensive: refuse to merge a new pass into a manifest that
        carries a different run_id. The caller has the wrong path;
        silent cross-run merging would corrupt the audit trail of
        whichever run actually owns the manifest."""
        path = tmp_path / MANIFEST_FILENAME
        record_eval_sample_pass(
            path,
            run_id="r1",
            mode="full",
            n=None,
            random_seed=42,
            eval_locales=None,
            n_selected=10,
            n_total=10,
            trajectory_ids=[],
        )
        with pytest.raises(ValueError, match=r"run_id='r1'"):
            record_eval_sample_pass(
                path,
                run_id="r2",  # different
                mode="full",
                n=None,
                random_seed=42,
                eval_locales=None,
                n_selected=10,
                n_total=10,
                trajectory_ids=[],
            )

    def test_fresh_bypasses_run_id_check(self, tmp_path: Path) -> None:
        """``fresh=True`` means "I am consciously starting over".
        Skip the run-id check because the caller is about to drop the
        old history anyway -- useful for a notebook that wants to
        reuse a manifest path across different run ids without
        triggering false-positive errors."""
        path = tmp_path / MANIFEST_FILENAME
        record_eval_sample_pass(
            path,
            run_id="r1",
            mode="full",
            n=None,
            random_seed=42,
            eval_locales=None,
            n_selected=10,
            n_total=10,
            trajectory_ids=[],
        )
        # Different run_id but fresh -> fine.
        m = record_eval_sample_pass(
            path,
            run_id="r2",
            mode="full",
            n=None,
            random_seed=42,
            eval_locales=None,
            n_selected=10,
            n_total=10,
            trajectory_ids=[],
            fresh=True,
        )
        assert m.run_id == "r2"

    def test_wrote_at_epoch_defaults_to_now(self, tmp_path: Path) -> None:
        path = tmp_path / MANIFEST_FILENAME
        before = int(time.time())
        m = record_eval_sample_pass(
            path,
            run_id="r",
            mode="full",
            n=None,
            random_seed=0,
            eval_locales=None,
            n_selected=0,
            n_total=0,
            trajectory_ids=[],
        )
        after = int(time.time())
        assert before <= m.passes[0].wrote_at_epoch <= after

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deeply" / "nested" / MANIFEST_FILENAME
        record_eval_sample_pass(
            path,
            run_id="r",
            mode="full",
            n=None,
            random_seed=0,
            eval_locales=None,
            n_selected=0,
            n_total=0,
            trajectory_ids=[],
        )
        assert path.exists()


# ─── consensus_mode ─────────────────────────────────────────────────


class TestConsensusMode:
    """The label the dashboard shows on the "Sample" pill. Must reflect
    the cumulative eval state honestly: identical modes across passes
    report that mode; differing modes report "mixed"; no passes report
    "unknown"."""

    def test_all_same_mode_returns_that_mode(self) -> None:
        m = EvalSampleManifest(
            run_id="r",
            passes=[
                EvalSamplePass(
                    wrote_at_epoch=0,
                    mode="full",
                    n=None,
                    random_seed=0,
                    eval_locales=None,
                    n_selected=0,
                    n_total=0,
                    trajectory_ids=[],
                ),
                EvalSamplePass(
                    wrote_at_epoch=0,
                    mode="full",
                    n=None,
                    random_seed=0,
                    eval_locales=["en_US"],
                    n_selected=0,
                    n_total=0,
                    trajectory_ids=[],
                ),
            ],
        )
        assert m.consensus_mode == "full"

    def test_mixed_modes_returns_mixed(self) -> None:
        m = EvalSampleManifest(
            run_id="r",
            passes=[
                EvalSamplePass(
                    wrote_at_epoch=0,
                    mode="full",
                    n=None,
                    random_seed=0,
                    eval_locales=None,
                    n_selected=0,
                    n_total=0,
                    trajectory_ids=[],
                ),
                EvalSamplePass(
                    wrote_at_epoch=0,
                    mode="per_locale",
                    n=10,
                    random_seed=0,
                    eval_locales=["en_US"],
                    n_selected=0,
                    n_total=0,
                    trajectory_ids=[],
                ),
            ],
        )
        assert m.consensus_mode == "mixed"

    def test_empty_passes_returns_unknown(self) -> None:
        m = EvalSampleManifest(run_id="r", passes=[])
        assert m.consensus_mode == "unknown"


class TestLocaleTargetedDetection:
    def test_no_targeted_pass(self) -> None:
        m = EvalSampleManifest(
            run_id="r",
            passes=[
                EvalSamplePass(
                    wrote_at_epoch=0,
                    mode="full",
                    n=None,
                    random_seed=0,
                    eval_locales=None,
                    n_selected=0,
                    n_total=0,
                    trajectory_ids=[],
                ),
            ],
        )
        assert m.has_locale_targeted_pass is False

    def test_with_targeted_pass(self) -> None:
        m = EvalSampleManifest(
            run_id="r",
            passes=[
                EvalSamplePass(
                    wrote_at_epoch=0,
                    mode="full",
                    n=None,
                    random_seed=0,
                    eval_locales=None,
                    n_selected=0,
                    n_total=0,
                    trajectory_ids=[],
                ),
                EvalSamplePass(
                    wrote_at_epoch=0,
                    mode="full",
                    n=None,
                    random_seed=0,
                    eval_locales=["en_US"],
                    n_selected=0,
                    n_total=0,
                    trajectory_ids=[],
                ),
            ],
        )
        assert m.has_locale_targeted_pass is True


# ─── read_eval_sample_manifest edge cases ───────────────────────────


class TestReadManifest:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        m = read_eval_sample_manifest(tmp_path / "doesntexist.json")
        assert m is None

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / MANIFEST_FILENAME
        path.write_text("not valid json {")
        with pytest.raises(ValueError, match="not valid JSON"):
            read_eval_sample_manifest(path)

    def test_non_object_top_level_raises(self, tmp_path: Path) -> None:
        path = tmp_path / MANIFEST_FILENAME
        path.write_text(json.dumps(["not", "an", "object"]))
        with pytest.raises(ValueError, match="must be a JSON object"):
            read_eval_sample_manifest(path)


# ─── End-to-end bug reproduction lock ───────────────────────────────


class TestBugReportScenario:
    """Reproduce and lock the exact scenario from the bug report:

    1. Full 1024-row eval across 8 locales (single pass).
    2. Later, locale-targeted re-eval of en_US (128 rows).

    The user observed: the manifest after step 2 lied about the
    cumulative state -- it claimed n_selected=128 / n_total=1024 with
    only the en_US trajectory_ids, while the eval parquet had 1024
    cumulative rows.

    Fix invariant: both passes survive in the manifest; the second
    pass's metadata is recoverable AND the first pass's metadata is
    recoverable. Dashboard's consensus_mode is "full" (both passes
    used mode="full"), but a reviewer reading the manifest can see
    that a locale-targeted re-eval happened on en_US.
    """

    def test_full_then_locale_targeted_preserves_both(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / MANIFEST_FILENAME
        run_id = "1778723989"

        # Step 1: original full pass.
        full_ids = [f"T{i:04d}" for i in range(1024)]
        record_eval_sample_pass(
            path,
            run_id=run_id,
            mode="full",
            n=None,
            random_seed=42,
            eval_locales=None,
            n_selected=1024,
            n_total=1024,
            trajectory_ids=full_ids,
            wrote_at_epoch=1778723990,
        )

        # Step 2: locale-targeted re-eval (the bug-triggering pass).
        en_us_ids = [f"T{i:04d}" for i in range(128)]
        record_eval_sample_pass(
            path,
            run_id=run_id,
            mode="full",
            n=None,
            random_seed=42,
            eval_locales=["en_US"],
            n_selected=128,
            n_total=1024,
            trajectory_ids=en_us_ids,
            wrote_at_epoch=1778800000,
        )

        m = read_eval_sample_manifest(path)
        assert m is not None
        assert m.run_id == run_id
        # The headline invariant: TWO passes, not one.
        assert len(m.passes) == 2

        # Pass 1 is the original full-run write.
        assert m.passes[0].eval_locales is None
        assert m.passes[0].n_selected == 1024
        assert len(m.passes[0].trajectory_ids) == 1024
        assert m.passes[0].wrote_at_epoch == 1778723990

        # Pass 2 is the en_US re-eval.
        assert m.passes[1].eval_locales == ["en_US"]
        assert m.passes[1].n_selected == 128
        assert len(m.passes[1].trajectory_ids) == 128
        assert m.passes[1].wrote_at_epoch == 1778800000

        # Consensus mode is "full" since both passes used mode="full".
        # Dashboard's "Sample full" label is honest.
        assert m.consensus_mode == "full"

        # And the manifest knows about the locale-targeted pass for
        # future "rebuilt-piecewise" UX.
        assert m.has_locale_targeted_pass is True
