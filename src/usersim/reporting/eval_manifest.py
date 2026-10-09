# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit-trail manifest for evaluator-pass history.

The eval store under ``output/evaluations/run=<id>/locale=*/probe_family=*/``
is the source of truth for "what got evaluated". The manifest at
``output/report/run=<id>/eval_sample_manifest.json`` is the audit trail
of HOW that eval state was built: which sampling mode each pass used,
what locales it touched, which trajectory_ids it covered.

Why it matters:

- ``write_run_locales_partition`` lets a notebook do a locale-targeted
  re-eval (re-evaluate en_US after an infra issue; add a newly-simulated
  locale to a previously-evaluated run). The cumulative eval state on
  disk is the union of all passes.
- The old single-object manifest format (schema v1) was overwritten on
  every sampling pass, so a re-eval would silently erase the original
  pass's metadata: the manifest would claim ``n_selected: 128`` while
  the eval parquet had 1024 cumulative rows. Reviewers reading the
  dashboard would see "Sample full" while the audit trail was a lie.

Schema v2 (this module's canonical format):

    {
      "schema_version": 2,
      "run_id": "<unix-epoch>",
      "passes": [
        {
          "wrote_at_epoch": <int>,
          "mode": "full" | "per_locale" | "per_locale_probe" | "matched_pair",
          "n": <int|null>,
          "random_seed": <int>,
          "eval_locales": <list[str]|null>,
          "n_selected": <int>,
          "n_total": <int>,
          "trajectory_ids": [...]
        },
        ...
      ]
    }

The list grows append-only. Whole-run rewrites
(``OVERWRITE_EVAL_RUN=True``, no ``EVAL_LOCALES``) reset the history
because the eval parquet is being rmtree'd from scratch -- preserving
old passes would be its own kind of lie.

Schema v1 (legacy, read-only): the single-object format we used to
write. ``read_eval_sample_manifest`` promotes it to a single-pass v2
manifest transparently so the dashboard's ``sample_mode`` derivation
works against historical artefacts without manual migration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

MANIFEST_FILENAME = "eval_sample_manifest.json"
#: Where ``usersim eval`` records its pass: beside the evaluation partitions,
#: underscore-prefixed so the parquet dataset readers skip it.
EVAL_STORE_MANIFEST_FILENAME = "_eval_sample_manifest.json"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class EvalSamplePass:
    """Metadata for one evaluator invocation.

    Captured once per sampling+eval pass (full-run write OR locale-
    targeted re-eval). The ``eval_locales`` field disambiguates: ``None``
    means "every locale on the trajectory parquet was evaluated this
    pass"; a list means "only these locales were re-evaluated, others
    were untouched". ``trajectory_ids`` is the verbatim list this pass
    sent to the evaluator -- useful for "did this specific trajectory
    get re-scored?" audits after a partial re-eval.
    """

    wrote_at_epoch: int
    mode: str
    n: int | None
    random_seed: int
    eval_locales: list[str] | None
    n_selected: int
    n_total: int
    trajectory_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "wrote_at_epoch": int(self.wrote_at_epoch),
            "mode": self.mode,
            "n": self.n,
            "random_seed": int(self.random_seed),
            "eval_locales": (list(self.eval_locales) if self.eval_locales is not None else None),
            "n_selected": int(self.n_selected),
            "n_total": int(self.n_total),
            "trajectory_ids": list(self.trajectory_ids),
        }


@dataclass
class EvalSampleManifest:
    """Append-only history of sampling+eval passes for one run id."""

    run_id: str
    passes: list[EvalSamplePass] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvalSampleManifest":
        """Parse either the v2 canonical format or the legacy v1
        single-object format (no ``passes`` key)."""
        run_id = str(raw.get("run_id", ""))
        # v2: passes list present (regardless of schema_version field --
        # presence of `passes` is the discriminator).
        if isinstance(raw.get("passes"), list):
            return cls(
                run_id=run_id,
                passes=[_pass_from_dict(p) for p in raw["passes"]],
            )
        # v1 legacy: single-pass at the top level. Promote to a one-
        # element passes list.
        return cls(
            run_id=run_id,
            passes=[
                EvalSamplePass(
                    # Legacy manifests don't carry a timestamp. Zero is
                    # the convention for "unknown" -- distinguishable
                    # from a real epoch.
                    wrote_at_epoch=0,
                    mode=str(raw.get("mode", "unknown")),
                    n=raw.get("n"),
                    random_seed=int(raw.get("random_seed", -1)),
                    eval_locales=raw.get("eval_locales"),
                    n_selected=int(raw.get("n_selected", 0)),
                    n_total=int(raw.get("n_total", 0)),
                    trajectory_ids=list(raw.get("trajectory_ids", [])),
                ),
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "passes": [p.to_dict() for p in self.passes],
        }

    @property
    def latest_pass(self) -> EvalSamplePass | None:
        """Most-recent pass, or ``None`` if the manifest has no passes
        recorded yet (shouldn't happen after the first write)."""
        return self.passes[-1] if self.passes else None

    @property
    def consensus_mode(self) -> str:
        """Sampling-mode label for the dashboard.

        - All passes agree -> the shared mode (``"full"`` typically).
        - Different modes across passes -> ``"mixed"``. Surfaces honestly
          on the dashboard that the eval store on disk wasn't built by
          a single uniform pass.
        - No passes (empty manifest) -> ``"unknown"``.
        """
        modes = {p.mode for p in self.passes}
        if not modes:
            return "unknown"
        if len(modes) == 1:
            return next(iter(modes))
        return "mixed"

    @property
    def has_locale_targeted_pass(self) -> bool:
        """True if any recorded pass was a locale-targeted re-eval
        (i.e. ``eval_locales`` was set to a non-empty list rather than
        ``None``). Lets the dashboard show a "rebuilt-piecewise" badge
        in future work."""
        return any(p.eval_locales is not None and len(p.eval_locales) > 0 for p in self.passes)


def _pass_from_dict(raw: dict[str, Any]) -> EvalSamplePass:
    """Decode one pass record. Defaults for missing fields keep older
    manifests parseable when the schema gains optional fields."""
    return EvalSamplePass(
        wrote_at_epoch=int(raw.get("wrote_at_epoch", 0)),
        mode=str(raw.get("mode", "unknown")),
        n=raw.get("n"),
        random_seed=int(raw.get("random_seed", -1)),
        eval_locales=raw.get("eval_locales"),
        n_selected=int(raw.get("n_selected", 0)),
        n_total=int(raw.get("n_total", 0)),
        trajectory_ids=list(raw.get("trajectory_ids", [])),
    )


def read_eval_sample_manifest(
    path: Path | str,
) -> EvalSampleManifest | None:
    """Read the manifest at ``path``, transparently promoting legacy
    v1 single-object manifests to v2. Returns ``None`` when the file
    doesn't exist (caller decides whether that's an error)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"eval_sample_manifest at {p} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"eval_sample_manifest at {p} must be a JSON object, got {type(raw).__name__}")
    return EvalSampleManifest.from_dict(raw)


def eval_sample_mode(*paths: Path | str) -> str:
    """The sampling mode recorded at the first of ``paths`` holding a manifest.

    ``"unknown"`` when none does: an evaluation that recorded nothing about
    how it sampled cannot claim to be full.
    """
    for path in paths:
        manifest = read_eval_sample_manifest(path)
        if manifest is not None:
            return manifest.consensus_mode
    return "unknown"


def record_eval_sample_pass(
    path: Path | str,
    *,
    run_id: str,
    mode: str,
    n: int | None,
    random_seed: int,
    eval_locales: Iterable[str] | None,
    n_selected: int,
    n_total: int,
    trajectory_ids: Iterable[str],
    fresh: bool = False,
    wrote_at_epoch: int | None = None,
) -> EvalSampleManifest:
    """Append a new pass record to the manifest at ``path``.

    Use cases:

    - **Initial pass** (no manifest on disk yet): writes a one-pass v2
      manifest. ``fresh`` is moot.
    - **Whole-run rewrite** (the eval parquet is being rmtree'd for a
      clean rebuild): pass ``fresh=True`` so the manifest history is
      reset to match. Preserving old passes when the underlying data
      is gone would be its own kind of lie.
    - **Locale-targeted re-eval** (only some locales are being
      rewritten; others stay on disk): pass ``fresh=False`` (the
      default) so the new pass APPENDS. The cumulative eval state on
      disk is now ``passes[0] + passes[1] (locale-subset)``, and the
      manifest reflects that history.

    Raises ``ValueError`` if an existing manifest's ``run_id`` doesn't
    match the caller's ``run_id`` -- a strong signal the caller has the
    wrong path. Refusing to merge protects against silent
    cross-contamination.
    """
    p = Path(path)
    existing = None if fresh else read_eval_sample_manifest(p)
    if existing is not None and existing.run_id and existing.run_id != run_id:
        raise ValueError(
            f"eval_sample_manifest at {p} carries run_id="
            f"{existing.run_id!r}, but record_eval_sample_pass was "
            f"called with run_id={run_id!r}. Refusing to merge across "
            f"runs -- caller likely has the wrong path."
        )
    manifest = existing or EvalSampleManifest(run_id=run_id)
    pass_record = EvalSamplePass(
        wrote_at_epoch=(int(wrote_at_epoch) if wrote_at_epoch is not None else int(time.time())),
        mode=mode,
        n=n,
        random_seed=random_seed,
        eval_locales=(list(eval_locales) if eval_locales is not None else None),
        n_selected=n_selected,
        n_total=n_total,
        trajectory_ids=list(trajectory_ids),
    )
    manifest.passes.append(pass_record)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )
    return manifest
