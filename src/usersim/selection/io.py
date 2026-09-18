# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""I/O for the selection layer: load a run's join inputs, write the curated subset.

Thin wrappers over :mod:`usersim.engine.core.storage` so the CLI and the
notebook share one read/write path. Output is namespaced by profile:

    <out>/run=<id>/profile=<name>/locale=<l>/probe_family=<f>/part-*.parquet
    <out>/run=<id>/profile=<name>/selection_manifest.json

so re-running with a different profile never clobbers a prior curation.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from usersim.taxonomy.eval_cell import decode_cell
from usersim.selection.engine import SelectionSummary
from usersim.selection.profiles import SelectionProfile
from usersim.selection.schemas import Schema, project

# Manifest filename. Underscore prefix => ignored by the pyarrow dataset
# reader (mirrors the trajectory dataset's ``_manifest.json``).
MANIFEST_NAME = "_selection_manifest.json"


def load_run(
    trajectories: str | Path,
    evaluations: Optional[str | Path] = None,
    *,
    run: Optional[str] = None,
) -> tuple[Any, Any, str]:
    """Read a run's trajectory + evaluation frames and resolve its run id.

    Goes through ``read_partitioned_dataset`` (NOT raw ``pd.read_parquet``):
    the trajectory tree carries ``list<string>`` columns that the pandas
    parquet reader rejects.
    """
    from usersim.engine.core.storage import (
        read_partitioned_dataset,
        resolve_run,
    )

    traj_path = Path(trajectories).resolve()
    run_id = resolve_run(traj_path, run)
    if run_id is None:
        raise FileNotFoundError(
            f"no runs found under {traj_path} — run `usersim simulate` first"
        )
    traj_df = read_partitioned_dataset(traj_path, run=run_id)
    # Reading at the run= level means ``run`` is not reconstructed as a column;
    # re-attach it so downstream schema projection (and a flattened HF export)
    # can carry the source run id.
    if "run" not in traj_df.columns:
        traj_df["run"] = str(run_id)

    eval_df = None
    if evaluations is not None:
        eval_path = Path(evaluations).resolve()
        if eval_path.exists():
            eval_df = read_partitioned_dataset(eval_path, run=run_id)
    return traj_df, eval_df, run_id


def load_runs(
    trajectories: str | Path,
    evaluations: Optional[str | Path] = None,
    *,
    runs: Optional[Union[str, Sequence[str]]] = None,
) -> tuple[Any, Any, str]:
    """Load and concatenate one or more runs' trajectory + evaluation frames.

    ``runs`` may be ``None``/``"latest"``/a single id (delegates to
    :func:`load_run`) or a list of run ids. Multiple runs are concatenated and
    deduplicated by ``trajectory_id`` (keeping the first occurrence — defensive
    against an accidental double-load; distinct runs normally share no ids).
    The returned run label is the ids joined by ``+`` so the curated output is
    namespaced by the exact run set it was drawn from.
    """
    import pandas as pd

    if runs is None or isinstance(runs, str):
        return load_run(trajectories, evaluations, run=runs)

    run_list = [str(r) for r in runs]
    if not run_list:
        raise ValueError("runs is empty")
    if len(run_list) == 1:
        return load_run(trajectories, evaluations, run=run_list[0])

    traj_parts: list[Any] = []
    eval_parts: list[Any] = []
    for run in run_list:
        traj_df, eval_df, _ = load_run(trajectories, evaluations, run=run)
        traj_parts.append(traj_df)
        if eval_df is not None:
            eval_parts.append(eval_df)

    traj_all = pd.concat(traj_parts, ignore_index=True).drop_duplicates(
        "trajectory_id", keep="first"
    )
    eval_all = (
        pd.concat(eval_parts, ignore_index=True).drop_duplicates(
            "trajectory_id", keep="first"
        )
        if eval_parts
        else None
    )
    return traj_all, eval_all, "+".join(run_list)


def _provenance(df: Any, eval_column: str) -> dict[str, Any]:
    """Best-effort provenance pulled from the first decodable row."""
    out: dict[str, Any] = {
        "code_sha": None,
        "scenario_prompt_version": None,
        "judge_aliases": None,
        "judge_families": None,
        "evaluator_prompt_version": None,
    }
    if "simulation_outcome" in getattr(df, "columns", []):
        for raw in df["simulation_outcome"]:
            prov = (decode_cell(raw).get("provenance")) or {}
            if prov:
                out["code_sha"] = prov.get("code_sha")
                out["scenario_prompt_version"] = prov.get("scenario_prompt_version")
                break
    if eval_column in getattr(df, "columns", []):
        for raw in df[eval_column]:
            env = (decode_cell(raw).get("envelope")) or {}
            if env:
                out["judge_aliases"] = env.get("judge_aliases")
                out["judge_families"] = env.get("judge_families")
                out["evaluator_prompt_version"] = env.get("prompt_version")
                break
    return out


def resolve_generation_model(
    trajectories: Any,
    run_label: str,
    *,
    explicit: Optional[str] = None,
) -> Optional[str]:
    """Return the model under test for ``run_label``, read from its manifest.

    The assistant model is not a trajectory column, but every run records it
    in ``_manifest.json``. Reading it there keeps the dataset card's
    provenance correct by construction: a hand-entered value silently goes
    stale the first time someone switches provider and forgets to update it,
    and the card is a published artifact.

    ``run_label`` may be several ids joined by ``+`` (what ``load_runs``
    returns when concatenating). Every distinct model is reported rather than
    the first, because a card claiming one model for a mixed dataset is worse
    than one naming both.

    ``explicit`` wins when given, for runs whose manifest predates this field
    or was produced elsewhere. Returns ``None`` when nothing is recoverable,
    which the card renders as "unspecified".
    """
    if explicit:
        return explicit
    # Deferred: keeps the selection layer importable without the engine, and
    # matches how reporting reaches the same function.
    from usersim.engine.core.manifest import load_run_manifest

    found: list[str] = []
    for run_id in str(run_label).split("+"):
        run_id = run_id.strip()
        if not run_id:
            continue
        try:
            manifest = load_run_manifest(run_id, trajectories)
        except Exception:  # a missing or unreadable manifest is not fatal
            continue
        identity = (manifest.models or {}).get("assistant_model") if manifest else None
        if identity is not None and identity.model and identity.model not in found:
            found.append(identity.model)
    return ", ".join(found) or None


def _profile_to_dict(profile: SelectionProfile) -> dict[str, Any]:
    d = dataclasses.asdict(profile)
    # axis_floors is a Mapping; ensure plain dict for JSON.
    d["axis_floors"] = dict(profile.axis_floors)
    return d


def build_manifest(
    *,
    profile: SelectionProfile,
    summary: SelectionSummary,
    source_run: str,
    eval_column: str,
    provenance: dict[str, Any],
    generation_model: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "kind": "selection",
        "source_run": source_run,
        "eval_column": eval_column,
        "created_at": int(time.time()),
        "generation_model": generation_model,
        "profile": _profile_to_dict(profile),
        "funnel": summary.to_dict(),
        "provenance": provenance,
    }


def _records_by_probe(funnel: dict) -> dict[str, int]:
    """Aggregate the funnel's per-(locale/probe_family) kept counts by probe."""
    by_probe: dict[str, int] = {}
    for key, count in (funnel.get("kept_by_group") or {}).items():
        probe = key.split("/", 1)[-1] if "/" in key else key
        by_probe[probe] = by_probe.get(probe, 0) + int(count)
    return by_probe


def write_curated_dataset(
    df: Any,
    out_root: str | Path,
    run_id: str,
    profile: SelectionProfile,
    *,
    summary: SelectionSummary,
    eval_column: str = "assistant_eval",
    columns: Schema = None,
) -> Path:
    """Write the curated frame under ``out_root/run=<id>/profile=<name>/``.

    ``columns`` selects the output schema (a preset name, comma-separated
    string, or explicit list); ``None`` writes every column except ``run``.
    The profile subroot is removed first (mirrors ``usersim eval`` / ``report``
    overwrite semantics). When ``df`` is empty, the manifest is still written
    and the partition write is skipped (``write_partitioned_dataset`` cannot
    group an empty frame).
    """
    from usersim.engine.core.storage import (
        run_subroot,
        write_partitioned_dataset,
    )

    out_root = Path(out_root).resolve()
    profile_root = run_subroot(out_root, run_id) / f"profile={profile.name}"
    if profile_root.exists():
        shutil.rmtree(profile_root)
    profile_root.mkdir(parents=True, exist_ok=True)

    # Project to the requested schema (``None`` => all columns minus ``run``,
    # which is encoded in the path).
    payload = project(df, columns)
    if len(payload) > 0:
        write_partitioned_dataset(payload, profile_root)

    manifest = build_manifest(
        profile=profile,
        summary=summary,
        source_run=str(run_id),
        eval_column=eval_column,
        provenance=_provenance(df, eval_column),
    )
    manifest["schema"] = list(payload.columns)
    # Underscore prefix so pyarrow's dataset reader ignores it (it skips
    # files prefixed with '_' / '.'); a bare ``selection_manifest.json``
    # inside the dataset root would be misread as a parquet fragment.
    (profile_root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return profile_root


def parse_repo_id(repo: str) -> str:
    """Normalize a HF dataset URL or ``org/name`` string to a repo id."""
    repo = repo.strip().rstrip("/")
    for prefix in (
        "https://huggingface.co/datasets/",
        "http://huggingface.co/datasets/",
        "huggingface.co/datasets/",
        "hf.co/datasets/",
    ):
        if repo.startswith(prefix):
            repo = repo[len(prefix):]
            break
    return repo


def _dataset_card(repo_id: str, profile: SelectionProfile, manifest: dict) -> str:
    funnel = manifest.get("funnel", {})
    prov = manifest.get("provenance", {})
    gen_model = manifest.get("generation_model")

    by_probe = _records_by_probe(funnel)
    probe_table = "| probe | records |\n|---|---|\n" + "".join(
        f"| {probe} | {count} |\n"
        for probe, count in sorted(by_probe.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    if by_probe:
        probe_table += f"| **total** | **{sum(by_probe.values())}** |\n"

    return (
        "---\n"
        "license: other\n"
        "tags:\n  - usersim\n  - synthetic\n  - user-simulation\n"
        "---\n\n"
        f"# {repo_id}\n\n"
        "Curated trajectories produced by [**NeMo UserSim**]"
        "(https://github.com/NVIDIA-NeMo/UserSim) "
        "(`selection` layer over simulated + evaluated multi-turn conversations).\n\n"
        "## Generation\n\n"
        f"- generation model (assistant under test): "
        f"`{gen_model if gen_model else 'unspecified'}`\n"
        f"- evaluator / judges: {prov.get('judge_aliases')} "
        f"(families: {prov.get('judge_families')})\n"
        f"- source run(s): `{manifest.get('source_run')}` · code_sha: "
        f"`{prov.get('code_sha')}`\n\n"
        "## Selection\n\n"
        f"- profile: `{manifest.get('profile', {}).get('name')}:"
        f"{manifest.get('profile', {}).get('version')}`\n"
        f"- selected / passed / total: "
        f"{funnel.get('selected')} / {funnel.get('passed')} / {funnel.get('total')}\n\n"
        "## Records by probe\n\n"
        f"{probe_table}\n"
        "## Columns\n\n"
        f"{', '.join('`%s`' % c for c in manifest.get('schema', []))}\n\n"
        "## Provenance\n\n"
        "The simulated users are sampled from "
        "[Nemotron-Personas](https://huggingface.co/collections/nvidia/nemotron-personas), "
        "which carries its own terms separate from NeMo UserSim's. Set the "
        "`license` field above to whatever governs this dataset; `other` is "
        "the placeholder, not an answer.\n"
    )


def push_to_hf(
    df: Any,
    repo: str,
    *,
    private: bool = True,
    token: Optional[str] = None,
    manifest: Optional[dict] = None,
    profile: Optional[SelectionProfile] = None,
    columns: Schema = None,
) -> str:
    """Push a curated frame to a (private by default) HF dataset repo.

    Uploads the projected frame as a single parquet split plus a dataset card
    and the selection manifest. Uses ``huggingface_hub`` directly (no
    ``datasets`` dependency). Returns the dataset URL.
    """
    import os
    import tempfile

    from huggingface_hub import HfApi

    repo_id = parse_repo_id(repo)
    payload = project(df, columns)
    if manifest is not None:
        manifest = {**manifest, "schema": list(payload.columns)}

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(
        repo_id, repo_type="dataset", private=private, exist_ok=True
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        parquet_path = tmp_path / "train-00000-of-00001.parquet"
        payload.to_parquet(parquet_path, index=False)
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo="data/train-00000-of-00001.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )
        if manifest is not None:
            (tmp_path / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            api.upload_file(
                path_or_fileobj=str(tmp_path / "manifest.json"),
                path_in_repo="_selection_manifest.json",
                repo_id=repo_id,
                repo_type="dataset",
            )
            if profile is not None:
                (tmp_path / "README.md").write_text(
                    _dataset_card(repo_id, profile, manifest), encoding="utf-8"
                )
                api.upload_file(
                    path_or_fileobj=str(tmp_path / "README.md"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

    return f"https://huggingface.co/datasets/{repo_id}"
