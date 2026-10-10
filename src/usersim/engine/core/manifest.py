# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-run manifest writer + reader for the simulator.

A manifest captures the run-level configuration that produced a trajectory
dataset — models, sim params, scope, asset versions, outcome summary — as a
single JSON artifact at ``output/trajectories/run=<id>/_manifest.json``.

The manifest is the canonical source of truth for "what was tested in this
run". The dashboard, the eval report, and (downstream) training-data
extraction read it instead of trying to reconstruct run-level identity from
whatever model config happens to be loaded in the consumer's process.

Why per-run, not per-row
------------------------
Per-row provenance (``simulation_outcome.provenance``) is the audit unit for
content lineage — every trajectory row carries enough to be filtered /
joined / replayed independently. The manifest is the run-level companion
that cheap consumers (a dashboard headline, a "did this run succeed?"
check) read once instead of scanning every parquet row. The two are
disjoint by design: the manifest does NOT duplicate per-row provenance, it
describes the run as a whole.

Schema version
--------------
``manifest_schema_version`` is stamped on every write. The replay fields
(``replayed_from_run_id``, ``matched_pair_group_id``, ``panel_content_hash``)
are reserved in v1 even when always null — keeps the future replay
workstream additive (adding a non-null value to an existing field is a
non-breaking change; bumping to v2 because we forgot to reserve a field is).

Atomic write
------------
``write_run_manifest`` is safe under concurrent calls (a scheduler's
distributed-worker case): it stages the JSON to a temp file in the same
directory and uses ``os.link`` to install the final name. Hard-link create
fails atomically if the target already exists, so the first writer wins
and subsequent workers noop without clobbering. POSIX-only — same
constraint as the rest of the storage layer.

Public API
----------
- :class:`RunManifest` — typed dataclass mirror of the on-disk schema.
- :func:`build_run_manifest` — assemble a manifest from the inputs the
  simulator has at run-end (models config, sim config, scope, optional
  trajectory frame for outcome aggregation).
- :func:`write_run_manifest` — atomic write to
  ``<root>/run=<run_id>/_manifest.json``.
- :func:`load_run_manifest` — canonical reader; returns ``None`` if no
  manifest is present (back-compat for runs predating this work).

Consumers MUST go through :func:`load_run_manifest` rather than rolling
their own JSON loader, so future schema migrations have one chokepoint.

Training-data lineage contract
------------------------------
Downstream training-data extraction (SFT / DPO / RL pipelines built on
top of the simulator's parquet output) keys all run-level filtering off
this manifest. Every trajectory row's ``run_id`` (the partition key under
``output/trajectories/run=<id>/``) maps to exactly one
:class:`RunManifest`. The contract:

================================================================  =================================================================
Filter / partition need                                           Manifest field
================================================================  =================================================================
"Which assistant model produced this conversation?"               ``manifest.models["assistant_model"].model``
"Were any rows produced with ``reasoning_effort=low``?"           ``manifest.models["assistant_model"].extra_body``
"Which sampling temperature was active?"                          ``manifest.models["assistant_model"].temperature``
"Which provider / endpoint was hit?"                              ``manifest.models["assistant_model"].provider`` / ``.endpoint``
"Which env var supplied the credential?"                          ``manifest.models["assistant_model"].api_key_env_var`` (env var NAME, never the resolved key)
"Drop rows where the in-sim judge was model X" (judge-bias)       ``manifest.models["judge_model"].model``
"Which model scored this run?"                                    each evaluation row's ``envelope.judge_models`` (an evaluator alias here is the config at simulation time)
"Pin scenario prompts to a specific version per probe"            ``manifest.asset_versions.prompt_versions[probe_family]``
"Which curated banks were active for sov_ai_facts:en_US?"         ``manifest.asset_versions.bank_versions``
"Was context-compression on?"                                     ``manifest.simulation_config.context_compression``
"Filter out runs from a dirty checkout / specific code SHA"       ``manifest.code.code_sha``
"Reproduce: which TOML / panel produced this?"                    ``manifest.models_config_path`` / ``manifest.scope.panel_path`` + ``.panel_content_hash``
"Was this a panel replay of run Y?" (reserved, future)            ``manifest.replay.replayed_from_run_id``
================================================================  =================================================================

The contract intentionally does NOT include opaque hash decoding —
``trajectory_id`` is content-derived but not human-readable. The
manifest is the queryable, named source of truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger("usersim.engine")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION: str = "v1"
MANIFEST_FILENAME: str = "_manifest.json"


# ---------------------------------------------------------------------------
# Dataclasses (typed mirror of the on-disk JSON schema)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelIdentity:
    """One model alias as it ran for this run.

    ``api_key_env_var`` is the *name* of the env var the provider reads
    its credential from (e.g. ``"NVIDIA_API_KEY"``). The resolved key
    value is NEVER stored: credentials must not leak into artifacts.
    """

    alias: str
    model: str
    provider: str | None = None
    endpoint: str | None = None
    api_key_env_var: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_parallel_requests: int | None = None
    timeout: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceInfo:
    """How this run was launched."""

    kind: str  # "cli" | "notebook" | "slurm" | "unknown"
    invocation: str
    hostname: str
    platform: str
    python_version: str


@dataclass(frozen=True)
class CodeInfo:
    """Code identity at run time."""

    code_sha: str | None = None
    conversation_plugin_version: str | None = None
    data_designer_version: str | None = None


@dataclass(frozen=True)
class SimulationConfigSnapshot:
    """Snapshot of every knob in ``ConversationSimulatorConfig`` at run start."""

    max_turns: int
    max_steps: int
    max_tools: int
    max_query_attempts: int
    max_assistant_attempts: int
    incremental_disclosure_ratio: float
    persona_grounding_ratio: float
    context_compression: bool
    compression_window: int
    random_seed: int | None = None
    store_reasoning: bool = True
    replay_assistant_reasoning: bool = False


@dataclass(frozen=True)
class ScopeInfo:
    """What this run sampled / replayed."""

    locales: list[str]
    probes: list[str]
    probe_mix: dict[str, float]
    tool_calling_themes: list[str] = field(default_factory=list)
    toolset_seed_path: str | None = None
    toolset_content_hash: str | None = None
    panel_path: str | None = None
    panel_content_hash: str | None = None
    trajectories_requested: dict[str, int] = field(default_factory=dict)
    trajectories_completed: dict[str, int] = field(default_factory=dict)

    #: Per-locale persona constraint AS APPLIED, e.g.
    #: ``{"ta_Taml_IN": {"first_language": ["Tamil"]}}``. A locale is
    #: absent when its population was not narrowed -- whether because
    #: matching was not requested, the corpus has no language column, or
    #: the language did not resolve. Recorded as the applied filter
    #: rather than as the requested flag because the two disagree:
    #: ``--match-persona-language --locale en_US`` requests matching and
    #: applies none, and nominally-Hindi locales apply different values
    #: (``hi_Deva_IN`` filters on "\u0939\u093f\u0902\u0926\u0940",
    #: ``hi_Latn_IN`` on "Hindi"). Which people were eligible is the
    #: thing that decides whether two runs are comparable.
    persona_language_filters: dict[str, dict[str, list[str]]] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetVersionsInfo:
    """Aggregated asset / prompt / variant lineage across the run."""

    assets_dir: str | None = None
    nemotron_personas_version: str | None = None
    bank_versions: dict[str, str] = field(default_factory=dict)
    prompt_versions: dict[str, str] = field(default_factory=dict)
    probe_variants_exercised: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class OutcomeSummary:
    """Cached run-level outcome stats so consumers don't have to scan parquet."""

    trajectories_total: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    failure_class_histogram: dict[str, int] = field(default_factory=dict)
    coverage_per_locale_probe: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayInfo:
    """Reserved-in-v1 replay fields. Always null today; kept so the future
    replay workstream is an additive change rather than a v2 bump.
    """

    replayed_from_run_id: str | None = None
    matched_pair_group_id: str | None = None


@dataclass(frozen=True)
class RunManifest:
    """v1 run manifest — typed mirror of the on-disk JSON.

    Stamped with :data:`MANIFEST_SCHEMA_VERSION` so consumers can reject
    or migrate incompatible future versions without guessing.
    """

    manifest_schema_version: str
    run_id: str
    started_at_epoch: int
    finished_at_epoch: int | None
    total_runtime_s: float | None
    source: SourceInfo
    code: CodeInfo
    models_config_path: str | None
    models: dict[str, ModelIdentity]
    simulation_config: SimulationConfigSnapshot
    scope: ScopeInfo
    asset_versions: AssetVersionsInfo
    outcome_summary: OutcomeSummary
    replay: ReplayInfo

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict of the manifest."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunManifest":
        """Construct a ``RunManifest`` from a dict (e.g. parsed JSON).

        Tolerant of missing optional sub-blocks — fills with defaults so a
        partial / older manifest still loads. ``manifest_schema_version``
        is preserved as-is; consumers that care about compatibility check
        the field themselves.
        """
        models_raw = d.get("models", {}) or {}
        return cls(
            manifest_schema_version=str(d.get("manifest_schema_version") or MANIFEST_SCHEMA_VERSION),
            run_id=str(d["run_id"]),
            started_at_epoch=int(d.get("started_at_epoch") or 0),
            finished_at_epoch=(int(d["finished_at_epoch"]) if d.get("finished_at_epoch") is not None else None),
            total_runtime_s=(float(d["total_runtime_s"]) if d.get("total_runtime_s") is not None else None),
            source=_source_from_dict(d.get("source") or {}),
            code=_code_from_dict(d.get("code") or {}),
            models_config_path=d.get("models_config_path"),
            models={alias: _model_identity_from_dict(alias, body or {}) for alias, body in models_raw.items()},
            simulation_config=_sim_config_from_dict(d.get("simulation_config") or {}),
            scope=_scope_from_dict(d.get("scope") or {}),
            asset_versions=_asset_versions_from_dict(d.get("asset_versions") or {}),
            outcome_summary=_outcome_summary_from_dict(d.get("outcome_summary") or {}),
            replay=_replay_from_dict(d.get("replay") or {}),
        )


# ── from_dict helpers ──────────────────────────────────────────────────────


def _model_identity_from_dict(alias: str, d: Mapping[str, Any]) -> ModelIdentity:
    return ModelIdentity(
        alias=str(d.get("alias", alias)),
        model=str(d.get("model", "")),
        provider=d.get("provider"),
        endpoint=d.get("endpoint"),
        api_key_env_var=d.get("api_key_env_var"),
        temperature=d.get("temperature"),
        top_p=d.get("top_p"),
        max_tokens=d.get("max_tokens"),
        max_parallel_requests=d.get("max_parallel_requests"),
        timeout=d.get("timeout"),
        extra_body=dict(d.get("extra_body") or {}),
    )


def _source_from_dict(d: Mapping[str, Any]) -> SourceInfo:
    return SourceInfo(
        kind=str(d.get("kind", "unknown")),
        invocation=str(d.get("invocation", "")),
        hostname=str(d.get("hostname", "")),
        platform=str(d.get("platform", "")),
        python_version=str(d.get("python_version", "")),
    )


def _code_from_dict(d: Mapping[str, Any]) -> CodeInfo:
    return CodeInfo(
        code_sha=d.get("code_sha"),
        conversation_plugin_version=d.get("conversation_plugin_version"),
        data_designer_version=d.get("data_designer_version"),
    )


def _sim_config_from_dict(d: Mapping[str, Any]) -> SimulationConfigSnapshot:
    return SimulationConfigSnapshot(
        max_turns=int(d.get("max_turns", 5)),
        max_steps=int(d.get("max_steps", 10)),
        max_tools=int(d.get("max_tools", 5)),
        max_query_attempts=int(d.get("max_query_attempts", 3)),
        max_assistant_attempts=int(d.get("max_assistant_attempts", 1)),
        store_reasoning=bool(d.get("store_reasoning", True)),
        replay_assistant_reasoning=bool(d.get("replay_assistant_reasoning", False)),
        incremental_disclosure_ratio=float(d.get("incremental_disclosure_ratio", 0.6)),
        persona_grounding_ratio=float(d.get("persona_grounding_ratio", 1.0)),
        context_compression=bool(d.get("context_compression", True)),
        compression_window=int(d.get("compression_window", 1)),
        random_seed=d.get("random_seed"),
    )


def _scope_from_dict(d: Mapping[str, Any]) -> ScopeInfo:
    return ScopeInfo(
        locales=list(d.get("locales") or []),
        probes=list(d.get("probes") or []),
        probe_mix=dict(d.get("probe_mix") or {}),
        tool_calling_themes=list(d.get("tool_calling_themes") or []),
        toolset_seed_path=d.get("toolset_seed_path"),
        toolset_content_hash=d.get("toolset_content_hash"),
        panel_path=d.get("panel_path"),
        panel_content_hash=d.get("panel_content_hash"),
        trajectories_requested=dict(d.get("trajectories_requested") or {}),
        trajectories_completed=dict(d.get("trajectories_completed") or {}),
        persona_language_filters=dict(d.get("persona_language_filters") or {}),
    )


def _asset_versions_from_dict(d: Mapping[str, Any]) -> AssetVersionsInfo:
    return AssetVersionsInfo(
        assets_dir=d.get("assets_dir"),
        nemotron_personas_version=d.get("nemotron_personas_version"),
        bank_versions=dict(d.get("bank_versions") or {}),
        prompt_versions=dict(d.get("prompt_versions") or {}),
        probe_variants_exercised={k: list(v or []) for k, v in (d.get("probe_variants_exercised") or {}).items()},
    )


def _outcome_summary_from_dict(d: Mapping[str, Any]) -> OutcomeSummary:
    return OutcomeSummary(
        trajectories_total=int(d.get("trajectories_total", 0)),
        status_counts=dict(d.get("status_counts") or {}),
        failure_class_histogram=dict(d.get("failure_class_histogram") or {}),
        coverage_per_locale_probe=dict(d.get("coverage_per_locale_probe") or {}),
    )


def _replay_from_dict(d: Mapping[str, Any]) -> ReplayInfo:
    return ReplayInfo(
        replayed_from_run_id=d.get("replayed_from_run_id"),
        matched_pair_group_id=d.get("matched_pair_group_id"),
    )


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------


def _detect_source(
    *,
    kind_override: str | None = None,
    invocation_override: str | None = None,
) -> SourceInfo:
    """Best-effort detection of how this run was launched.

    Detection order:

    1. ``kind_override`` if the caller passed one (manual override wins).
    2. SLURM env var → ``slurm``.
    3. IPython kernel detected → ``notebook``.
    4. Otherwise → ``cli`` (sys.argv).
    """
    if kind_override:
        kind = kind_override
        invocation = invocation_override or ""
    else:
        slurm_job = os.environ.get("SLURM_JOB_ID")
        if slurm_job:
            kind = "slurm"
            invocation = f"slurm_job_id={slurm_job}"
        elif _is_running_in_notebook():
            kind = "notebook"
            invocation = (
                os.environ.get("JPY_SESSION_NAME") or os.environ.get("JUPYTER_SERVER_ROOT") or "<unknown notebook>"
            )
        else:
            kind = "cli"
            invocation = " ".join(sys.argv) if sys.argv else "<unknown cli>"

    if invocation_override is not None:
        invocation = invocation_override

    return SourceInfo(
        kind=kind,
        invocation=invocation,
        hostname=_safe_hostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
    )


def _is_running_in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    return type(ip).__name__ in ("ZMQInteractiveShell", "TerminalInteractiveShell")


def _safe_hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "<unknown host>"


# ---------------------------------------------------------------------------
# Code / package version detection
# ---------------------------------------------------------------------------


def _detect_code_info() -> CodeInfo:
    """Capture code SHA + relevant package versions.

    ``code_sha`` reuses the cached helper from ``core/provenance.py``.
    Package versions are best-effort (``importlib.metadata.version``) —
    development checkouts may resolve to ``None`` if the package isn't
    installed; that's fine, the manifest records the gap honestly.
    """
    from usersim.engine.core.provenance import get_code_sha

    return CodeInfo(
        code_sha=get_code_sha(),
        conversation_plugin_version=_own_version(),
        data_designer_version=_pkg_version("data_designer"),
    )


def _own_version() -> str | None:
    """Version of the distribution providing this package.

    Resolved through ``packages_distributions`` rather than a hardcoded
    distribution name, so renaming the distribution cannot silently start
    writing a null version into every manifest.
    """
    try:
        from importlib.metadata import packages_distributions

        dists = packages_distributions().get("usersim", [])
    except ImportError:  # pragma: no cover
        return None
    return _pkg_version(dists[0]) if dists else None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:  # pragma: no cover — Python <3.8 only
        return None


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


def _file_content_hash(path: Path | None) -> str | None:
    """SHA-256 of a file's contents, or ``None`` if path is missing."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError as e:
        logger.debug("manifest: file hash failed for %s: %s", p, e)
        return None
    return f"sha256:{h.hexdigest()}"


# ---------------------------------------------------------------------------
# Trajectory aggregation
# ---------------------------------------------------------------------------


def _aggregate_from_trajectories(
    traj_df: Any,
) -> dict[str, Any]:
    """Aggregate per-row data into the manifest's ``asset_versions`` and
    ``outcome_summary`` dicts.

    Returns a dict with keys ``bank_versions``, ``prompt_versions``,
    ``probe_variants_exercised``, ``status_counts``,
    ``failure_class_histogram``, ``coverage_per_locale_probe``,
    ``trajectories_completed`` (per locale), ``probes`` (sorted distinct
    list), ``locales`` (sorted distinct list), ``trajectories_total``.

    Defensive: if the frame is missing expected columns or rows, returns
    sensible defaults rather than raising.
    """
    if traj_df is None or len(traj_df) == 0:
        return {
            "bank_versions": {},
            "prompt_versions": {},
            "probe_variants_exercised": {},
            "status_counts": {},
            "failure_class_histogram": {},
            "coverage_per_locale_probe": {},
            "trajectories_completed": {},
            "probes": [],
            "locales": [],
            "trajectories_total": 0,
        }

    bank_versions: dict[str, str] = {}
    prompt_versions: dict[str, str] = {}
    probe_variants: dict[str, set] = {}
    status_counts: dict[str, int] = {}
    failure_classes: dict[str, int] = {}
    coverage: dict[str, int] = {}
    completed_per_locale: dict[str, int] = {}
    probes_seen: set = set()
    locales_seen: set = set()

    for _, row in traj_df.iterrows():
        probe_family = row.get("probe_family") if hasattr(row, "get") else None
        probe_variant = row.get("probe_variant") if hasattr(row, "get") else None
        locale = row.get("locale") if hasattr(row, "get") else None
        if isinstance(probe_family, str):
            probes_seen.add(probe_family)
            if isinstance(probe_variant, str) and probe_variant:
                probe_variants.setdefault(probe_family, set()).add(probe_variant)
        if isinstance(locale, str):
            locales_seen.add(locale)
            completed_per_locale[locale] = completed_per_locale.get(locale, 0) + 1
        if isinstance(probe_family, str) and isinstance(locale, str):
            cell_key = f"{locale}:{probe_family}"
            coverage[cell_key] = coverage.get(cell_key, 0) + 1

        outcome_raw = row.get("simulation_outcome") if hasattr(row, "get") else None
        outcome = _decode_json_field(outcome_raw)
        if not isinstance(outcome, dict):
            continue

        status = outcome.get("status")
        if isinstance(status, str):
            status_counts[status] = status_counts.get(status, 0) + 1
        fc = outcome.get("failure_class")
        if isinstance(fc, str):
            failure_classes[fc] = failure_classes.get(fc, 0) + 1

        prov = outcome.get("provenance") or {}
        if isinstance(prov, dict):
            scenario_pv = prov.get("scenario_prompt_version")
            if isinstance(scenario_pv, str) and isinstance(probe_family, str):
                # Per-probe rollup. Last write wins; warn if a probe family
                # ever has more than one prompt_version in a single run
                # (that would suggest a code change mid-run).
                prior = prompt_versions.get(probe_family)
                if prior is None or prior == scenario_pv:
                    prompt_versions[probe_family] = scenario_pv
                else:
                    logger.warning(
                        "manifest: probe %r has multiple prompt_versions in run (%r vs %r); keeping the first",
                        probe_family,
                        prior,
                        scenario_pv,
                    )
            bv = prov.get("bank_version")
            if isinstance(bv, dict):
                for key, value in bv.items():
                    if isinstance(value, str):
                        bank_versions.setdefault(key, value)

    return {
        "bank_versions": dict(sorted(bank_versions.items())),
        "prompt_versions": dict(sorted(prompt_versions.items())),
        "probe_variants_exercised": {k: sorted(v) for k, v in sorted(probe_variants.items())},
        "status_counts": dict(sorted(status_counts.items())),
        "failure_class_histogram": dict(sorted(failure_classes.items())),
        "coverage_per_locale_probe": dict(sorted(coverage.items())),
        "trajectories_completed": dict(sorted(completed_per_locale.items())),
        "probes": sorted(probes_seen),
        "locales": sorted(locales_seen),
        "trajectories_total": int(len(traj_df)),
    }


def _decode_json_field(raw: Any) -> Any:
    """Best-effort JSON decode of fields that may arrive as str or dict."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Models -> ModelIdentity resolution
# ---------------------------------------------------------------------------


# Mirrors cli/_models.py::_BUILTIN_PROVIDER_API_KEY_ENV +
# _BUILTIN_PROVIDER_ENDPOINTS. Duplicated here because usersim.engine
# core must not depend on cli/. Keep in sync.
_BUILTIN_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_BUILTIN_PROVIDER_ENDPOINTS: dict[str, str] = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _resolve_models(
    *,
    model_specs: Sequence[Any],
    provider_specs: Sequence[Any],
) -> dict[str, ModelIdentity]:
    """Build the alias → :class:`ModelIdentity` map from the simulator's
    typed config (``ModelsConfig.models`` and ``.providers`` from
    ``cli/_models.py``).

    ``provider_specs`` carries custom-provider endpoint + api_key env-var
    mappings; built-in providers (``nvidia`` / ``openai`` / ``openrouter``)
    fall back to the static dicts at module top so we never store the
    resolved key value.
    """
    custom_by_name: dict[str, Any] = {p.name: p for p in provider_specs}
    out: dict[str, ModelIdentity] = {}

    for spec in model_specs:
        endpoint: str | None = None
        env_var: str | None = None
        if spec.provider in custom_by_name:
            cp = custom_by_name[spec.provider]
            endpoint = cp.endpoint
            env_var = cp.api_key
        else:
            endpoint = _BUILTIN_PROVIDER_ENDPOINTS.get(spec.provider)
            env_var = _BUILTIN_PROVIDER_API_KEY_ENV.get(spec.provider)

        out[spec.alias] = ModelIdentity(
            alias=spec.alias,
            model=spec.model,
            provider=spec.provider,
            endpoint=endpoint,
            api_key_env_var=env_var,
            temperature=spec.temperature,
            top_p=spec.top_p,
            max_tokens=spec.max_tokens,
            max_parallel_requests=spec.max_parallel_requests,
            timeout=spec.timeout,
            extra_body=dict(spec.extra_body or {}),
        )
    return out


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_run_manifest(
    *,
    run_id: str,
    started_at_epoch: int,
    finished_at_epoch: int | None = None,
    models_config: Any,
    models_config_path: Path | None = None,
    sim_config: Any,
    locales_requested: Sequence[str],
    probe_mix: Mapping[str, float],
    tool_calling_themes: Sequence[str] = (),
    toolset_seed_path: Path | None = None,
    panel_path: Path | None = None,
    assets_dir: Path | None = None,
    trajectories_requested: Mapping[str, int] | None = None,
    persona_language_filters: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    trajectory_df: Any = None,
    source_kind: str | None = None,
    source_invocation: str | None = None,
    replayed_from_run_id: str | None = None,
    matched_pair_group_id: str | None = None,
) -> RunManifest:
    """Assemble a :class:`RunManifest` from the inputs the simulator has
    at run-end.

    ``models_config`` is the typed ``ModelsConfig`` from ``cli/_models.py``
    (carries both ``models`` and ``providers``). ``sim_config`` is the
    ``ConversationSimulatorConfig`` for any locale (the snapshot is
    locale-independent — only the ``locale`` field varies and it does not
    appear in the snapshot).

    ``trajectory_df`` is optional. When provided, it drives the manifest's
    ``asset_versions`` aggregation (bank / prompt / variant rollups) and
    the ``outcome_summary`` block. When absent, those blocks render as
    empty dicts — useful for dry-run / panel-build paths that produce no
    rows.

    ``trajectories_requested`` is a per-locale-keyed dict of intended row
    counts; used by the report to compute "asked for N, got M".
    """
    finished = finished_at_epoch if finished_at_epoch is not None else int(time.time())
    runtime_s = max(0.0, float(finished - started_at_epoch))

    models = _resolve_models(
        model_specs=getattr(models_config, "models", ()) or (),
        provider_specs=getattr(models_config, "providers", ()) or (),
    )

    aggregates = _aggregate_from_trajectories(trajectory_df)

    sim_snapshot = SimulationConfigSnapshot(
        max_turns=int(getattr(sim_config, "max_turns", 5)),
        max_steps=int(getattr(sim_config, "max_steps", 10)),
        max_tools=int(getattr(sim_config, "max_tools", 5)),
        max_query_attempts=int(getattr(sim_config, "max_query_attempts", 3)),
        max_assistant_attempts=int(getattr(sim_config, "max_assistant_attempts", 1)),
        store_reasoning=bool(getattr(sim_config, "store_reasoning", True)),
        replay_assistant_reasoning=bool(getattr(sim_config, "replay_assistant_reasoning", False)),
        incremental_disclosure_ratio=float(getattr(sim_config, "incremental_disclosure_ratio", 0.6)),
        persona_grounding_ratio=float(getattr(sim_config, "persona_grounding_ratio", 1.0)),
        context_compression=bool(getattr(sim_config, "context_compression", True)),
        compression_window=int(getattr(sim_config, "compression_window", 1)),
        random_seed=getattr(sim_config, "random_seed", None),
    )

    # ``probes`` from aggregates is what actually ran; ``probe_mix`` is
    # the requested distribution. We surface both so a consumer can see
    # "asked for 30% safety_chat_pressure, ended up with 5%".
    locales_run = aggregates["locales"] or list(locales_requested)
    requested_per_locale = {l: int(c) for l, c in (trajectories_requested or {}).items()}

    scope = ScopeInfo(
        locales=sorted(set(list(locales_run) + list(locales_requested))),
        probes=list(aggregates["probes"]) or sorted(probe_mix.keys()),
        probe_mix=dict(probe_mix),
        tool_calling_themes=list(tool_calling_themes),
        toolset_seed_path=str(toolset_seed_path) if toolset_seed_path else None,
        toolset_content_hash=_file_content_hash(toolset_seed_path) if toolset_seed_path else None,
        panel_path=str(panel_path) if panel_path else None,
        panel_content_hash=_file_content_hash(panel_path) if panel_path else None,
        trajectories_requested=requested_per_locale,
        trajectories_completed=aggregates["trajectories_completed"],
        persona_language_filters=dict(persona_language_filters or {}),
    )

    asset_versions = AssetVersionsInfo(
        assets_dir=str(assets_dir) if assets_dir is not None else None,
        nemotron_personas_version=_nemotron_version(),
        bank_versions=aggregates["bank_versions"],
        prompt_versions=aggregates["prompt_versions"],
        probe_variants_exercised=aggregates["probe_variants_exercised"],
    )

    outcome = OutcomeSummary(
        trajectories_total=aggregates["trajectories_total"],
        status_counts=aggregates["status_counts"],
        failure_class_histogram=aggregates["failure_class_histogram"],
        coverage_per_locale_probe=aggregates["coverage_per_locale_probe"],
    )

    return RunManifest(
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=str(run_id),
        started_at_epoch=int(started_at_epoch),
        finished_at_epoch=int(finished),
        total_runtime_s=round(runtime_s, 3),
        source=_detect_source(kind_override=source_kind, invocation_override=source_invocation),
        code=_detect_code_info(),
        models_config_path=str(models_config_path) if models_config_path else None,
        models=models,
        simulation_config=sim_snapshot,
        scope=scope,
        asset_versions=asset_versions,
        outcome_summary=outcome,
        replay=ReplayInfo(
            replayed_from_run_id=replayed_from_run_id,
            matched_pair_group_id=matched_pair_group_id,
        ),
    )


def _nemotron_version() -> str | None:
    """Re-export of the Nemotron-Personas version helper (env-var-backed)."""
    from usersim.engine.core.provenance import get_nemotron_personas_version

    return get_nemotron_personas_version()


# ---------------------------------------------------------------------------
# Writer + reader
# ---------------------------------------------------------------------------


def manifest_path(root: Path | str, run_id: str) -> Path:
    """Canonical on-disk path for a run's manifest.

    ``<root>/run=<run_id>/_manifest.json``.
    """
    return Path(root) / f"run={run_id}" / MANIFEST_FILENAME


def write_run_manifest(
    manifest: RunManifest,
    root: Path | str,
) -> Path:
    """Atomic write of ``manifest`` to ``<root>/run=<run_id>/_manifest.json``.

    Concurrency contract:

    - First-writer-wins. The writer stages the JSON to a temp file in the
      same directory and uses ``os.link`` to install the final name. Hard-
      link-create fails atomically if the target already exists, so a
      second concurrent writer noops without clobbering.
    - The function always returns the canonical manifest path (whether we
      wrote it or someone beat us to it).

    POSIX-only — same constraint as the rest of the storage layer.
    """
    out_path = manifest_path(root, manifest.run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        # Fast path: manifest already on disk (re-run, partial resume,
        # or another worker beat us). Noop. We do NOT validate content
        # match here — a divergent re-write would still noop on the link
        # step below, and surfacing a hard error here would break legit
        # resume flows.
        logger.debug("manifest: noop, %s already exists", out_path)
        return out_path

    fd, tmp_str = tempfile.mkstemp(
        prefix=f".{MANIFEST_FILENAME}.tmp.",
        dir=str(out_path.parent),
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(str(tmp), str(out_path))
        except FileExistsError:
            logger.debug("manifest: lost link race for %s; another writer won", out_path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return out_path


def write_simulator_manifest(
    *,
    run_id: str,
    started_at_epoch: int,
    trajectory_root: Path | str,
    models_config: Any,
    models_config_path: Path | None = None,
    sim_config: Any,
    locales_requested: Sequence[str],
    probe_mix: Mapping[str, float],
    tool_calling_themes: Sequence[str] = (),
    toolset_seed_path: Path | None = None,
    panel_path: Path | None = None,
    assets_dir: Path | None = None,
    trajectories_requested: Mapping[str, int] | None = None,
    persona_language_filters: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    source_kind: str | None = None,
    source_invocation: str | None = None,
    replayed_from_run_id: str | None = None,
    matched_pair_group_id: str | None = None,
) -> Path:
    """Build + write the simulator manifest in a single call.

    Re-reads the trajectory partitions for ``run_id`` under
    ``trajectory_root`` to compute the manifest's ``asset_versions`` and
    ``outcome_summary`` aggregates. This is the canonical entry point for
    every simulator caller (CLI, notebook, scheduler worker) — using it
    keeps all three paths producing identical manifest content.

    Returns the on-disk path of the manifest. Atomic: a concurrent caller
    that loses the race noops without clobbering the first writer's file.
    """
    from usersim.engine.core.storage import read_partitioned_dataset

    try:
        traj_df = read_partitioned_dataset(trajectory_root, run=run_id)
    except (FileNotFoundError, ValueError) as e:
        logger.debug("manifest: traj read failed (%s); aggregates will be empty", e)
        traj_df = None

    manifest = build_run_manifest(
        run_id=run_id,
        started_at_epoch=started_at_epoch,
        models_config=models_config,
        models_config_path=models_config_path,
        sim_config=sim_config,
        locales_requested=locales_requested,
        probe_mix=probe_mix,
        tool_calling_themes=tool_calling_themes,
        toolset_seed_path=toolset_seed_path,
        panel_path=panel_path,
        assets_dir=assets_dir,
        trajectories_requested=trajectories_requested,
        persona_language_filters=persona_language_filters,
        trajectory_df=traj_df,
        source_kind=source_kind,
        source_invocation=source_invocation,
        replayed_from_run_id=replayed_from_run_id,
        matched_pair_group_id=matched_pair_group_id,
    )
    return write_run_manifest(manifest, trajectory_root)


def load_run_manifest(
    run_id: str,
    root: Path | str,
) -> RunManifest | None:
    """Canonical reader. Returns ``None`` for runs without a manifest.

    Consumers MUST go through this function rather than rolling their
    own ``json.load()`` against the path — keeps schema-migration
    handling in one chokepoint.
    """
    p = manifest_path(root, run_id)
    if not p.exists() or not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("manifest: failed to load %s: %s", p, e)
        return None
    if not isinstance(payload, dict):
        logger.warning("manifest: %s is not a JSON object", p)
        return None
    try:
        return RunManifest.from_dict(payload)
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("manifest: malformed %s: %s", p, e)
        return None
