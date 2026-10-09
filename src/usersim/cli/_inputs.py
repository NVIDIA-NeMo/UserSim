# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running stored episode rows: ``usersim simulate --inputs``.

Rows written by ``usersim simulate --materialize-inputs`` run exactly as they
were stored, so two setups can be compared on the same simulated users.

- **Identity.** Each run of a row gets its own ``trajectory_id``: a hash of the
  stored row and of everything about the setup that can change the
  conversation (see :func:`setup_fingerprint`). The stored row's own id is
  saved as ``input_id``, the column that pairs two runs of the same rows.
- **The conversation.** It runs with the stored id as ``trajectory_id``, as
  the hosted runtime does, so whatever a probe derives from its id (the tools
  ``tool_calling`` offers, say) is the same for every setup. The new id is
  written only when the rows are saved.
- **Runs.** A run holds one setup. A resume reopens the latest run of stored
  rows whose setup fingerprint matches; otherwise the first saved rows claim
  a new run. Every save happens under the root's write lock, after checking
  the run's saved rows again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("usersim.cli.simulate")

#: The saved row's copy of the stored row's ``trajectory_id``.
INPUT_ID_COLUMN = "input_id"

#: Names the id formula, so a later formula never collides with this one.
_ID_SCHEME = "usersim-inputs-v1"

#: ``USERSIM_*`` variables that cannot change a conversation: debug logging,
#: where the models file is (the models themselves are fingerprinted),
#: provenance labels, and the evaluator's audit settings. Every other
#: ``USERSIM_*`` variable is part of the setup, including ones added later.
_OPERATIONAL_ENV = frozenset(
    {
        "USERSIM_DEBUG_LOG",
        "USERSIM_MODELS_CONFIG",
        "USERSIM_CODE_SHA",
        "USERSIM_NEMOTRON_PERSONAS_VERSION",
        "USERSIM_AUDIT_MODEL",
        "USERSIM_AUDIT_REASONING_EFFORT",
    }
)

#: Settings that may differ between the rows of one invocation.
_PER_ROW_SETTINGS = ("locale", "assets_dir")


@dataclass(frozen=True)
class EpisodeInputs:
    """Stored rows checked and ready to run."""

    rows: list[dict[str, Any]]
    #: The ``usersim_config`` every row shares, apart from ``_PER_ROW_SETTINGS``.
    config: dict[str, Any]
    #: The asset root the rows run against.
    assets_dir: Path
    #: The file the rows came from, when they came from one.
    source: Path | None = None


@dataclass(frozen=True)
class InputsRun:
    """What one call to :func:`simulate_inputs_to_dataset` did."""

    run_id: str | None
    resumed: bool
    written: int
    skipped: int


# ── loading and checking ────────────────────────────────────────────────


def load_episode_inputs(path: str | Path) -> list[dict[str, Any]]:
    """Read stored rows from a ``.jsonl`` or ``.parquet`` file, values as stored.

    Parquet is read through pyarrow, not pandas, so no value is re-typed on
    the way in.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    raise ValueError(f"expected a .jsonl or .parquet file, got {path}")


def _plain(value: Any) -> Any:
    """Plain Python for a row built in memory, which may hold numpy values."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, str):
        return str(value)
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def prepare_episode_inputs(
    rows: list[Mapping[str, Any]],
    *,
    assets_dir: str | Path | None = None,
    source: str | Path | None = None,
) -> EpisodeInputs:
    """Check stored rows before any model call, and settle what they run against.

    Raises ``ValueError`` naming the row and the problem. The rows must share
    one setup: their ``usersim_config`` may differ only in locale and asset
    root, so that one run holds one setup.
    """
    from pydantic import ValidationError

    from usersim.engine.config import ConversationSimulatorConfig
    from usersim.engine.core.episode_input import EPISODE_INPUT_COLUMN
    from usersim.engine.core.probes import resolve_probe

    if not rows:
        raise ValueError("no rows to run")
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"row {i} is not a mapping")
        row = _plain(dict(raw))
        stored_id = row.get("trajectory_id")
        if not isinstance(stored_id, str) or not stored_id:
            raise ValueError(f"row {i} has no trajectory_id; each stored row needs its own")
        where = f"row {i} ({stored_id})"
        if stored_id in seen:
            raise ValueError(f"{where}: trajectory_id appears in more than one row; each stored row needs its own")
        seen.add(stored_id)
        for reserved in (INPUT_ID_COLUMN, EPISODE_INPUT_COLUMN):
            if reserved in row:
                raise ValueError(f"{where} already has {reserved!r}; that is a run's output, not a stored row")
        if not isinstance(row.get("usersim_config"), Mapping):
            raise ValueError(f"{where} has no usersim_config; run rows written by --materialize-inputs")
        try:
            config = ConversationSimulatorConfig.model_validate(dict(row["usersim_config"]))
        except ValidationError as error:
            raise ValueError(f"{where}: usersim_config does not load: {error}") from None
        if not row.get(config.persona_column):
            raise ValueError(f"{where} has no {config.persona_column}")
        try:
            resolve_probe(str(row.get(config.probe_type_column)))
        except KeyError as error:
            raise ValueError(f"{where}: {error.args[0]}") from None
        if "locale" in row and row["locale"] != config.locale:
            raise ValueError(f"{where}: locale {row['locale']!r} differs from its config's {config.locale!r}")
        try:
            json.dumps(row, allow_nan=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{where} holds a value JSON cannot represent: {error}") from None
        checked.append(row)
    return EpisodeInputs(
        rows=checked,
        config=_shared_config(checked),
        assets_dir=_resolve_assets_dir(checked, assets_dir),
        source=Path(source).resolve() if source is not None else None,
    )


def _shared_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The setup every row shares; ``ValueError`` naming the settings that differ."""

    def setup(row: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in row["usersim_config"].items() if key not in _PER_ROW_SETTINGS}

    first = setup(rows[0])
    for row in rows[1:]:
        other = setup(row)
        differing = sorted(key for key in first.keys() | other.keys() if first.get(key) != other.get(key))
        if differing:
            shown = ", ".join(f"{key}: {first.get(key)!r} vs {other.get(key)!r}" for key in differing)
            raise ValueError(
                f"the rows were stored with different settings ({shown}); one run holds one setup, "
                "so run them separately"
            )
    return dict(rows[0]["usersim_config"])


def _resolve_assets_dir(rows: list[dict[str, Any]], given: str | Path | None) -> Path:
    """``--assets-dir`` when given, else the one asset root the rows were stored with."""
    from usersim.engine.core._assets import packaged_assets_dir

    if given is not None:
        path = Path(given).resolve()
        if not path.is_dir():
            raise ValueError(f"--assets-dir {path} is not a directory")
        return path
    stored = {row["usersim_config"].get("assets_dir") for row in rows}
    hint = f"pass --assets-dir (the assets that ship with User Sim are at {packaged_assets_dir()})"
    if len(stored) != 1 or None in stored:
        raise ValueError(f"the rows do not name one asset folder; {hint}")
    path = Path(stored.pop())
    if not path.is_dir():
        raise ValueError(f"the asset folder the rows were stored with, {path}, does not exist here; {hint}")
    return path


# ── identity ────────────────────────────────────────────────────────────


def _digest(value: Any) -> str:
    """SHA-256 of ``value``'s canonical JSON form."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _contents_digest(path: Path) -> str:
    """SHA-256 over a file's bytes, or over every file under a folder by relative path.

    Hidden files and folders (``.DS_Store``, ``.ipynb_checkpoints``) are left
    out: no loader reads them, and they change without the assets changing.
    """
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    entries = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and not any(part.startswith(".") for part in item.relative_to(path).parts)
    )
    return _digest(
        [[item.relative_to(path).as_posix(), hashlib.sha256(item.read_bytes()).hexdigest()] for item in entries]
    )


def _model_settings(models: Any, aliases: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Each alias's settings as Data Designer sends them: readable, and as hashed.

    The readable form names a provider's extra headers without their values,
    which may be credentials; the hashed form carries a digest of them.
    """
    from usersim.cli._models import resolved_model_providers, to_model_configs

    configs = {config.alias: config for config in to_model_configs(models)}
    providers = {provider.name: provider for provider in resolved_model_providers(models)}
    readable: dict[str, Any] = {}
    hashed: dict[str, Any] = {}
    for alias in sorted(set(aliases)):
        config = configs.get(alias)
        if config is None:
            readable[alias] = hashed[alias] = None
            continue
        params = config.inference_parameters
        settings = {"model": config.model, "request": dict(params.generate_kwargs), "timeout": params.timeout}
        provider = providers.get(config.provider)
        if provider is None:
            readable[alias] = hashed[alias] = {**settings, "provider": None}
            continue
        served = {
            "endpoint": provider.endpoint,
            "provider_type": provider.provider_type,
            "extra_body": provider.extra_body or {},
        }
        headers = provider.extra_headers or {}
        readable[alias] = {**settings, "provider": {**served, "extra_headers": sorted(headers)}}
        hashed[alias] = {**settings, "provider": {**served, "extra_headers": _digest(headers)}}
    return readable, hashed


def _asset_roots_digest(assets_dir: Path) -> list[str]:
    """The contents of every asset root a run searches, in priority order.

    Every root counts, not just the first copy of each file: an identity spec
    that ``extends`` its own probe reads the copy below it on the path. Where a
    root is does not count, so the same assets in another folder are the same
    setup.
    """
    from usersim.engine.core._assets import asset_search_path, reset_runtime_assets_dir, set_runtime_assets_dir

    token = set_runtime_assets_dir(assets_dir)
    try:
        roots = asset_search_path()
    finally:
        reset_runtime_assets_dir(token)
    return [_contents_digest(root) if root.exists() else "missing" for root in roots]


def _environment_settings() -> dict[str, list[str]]:
    """Every ``USERSIM_*`` variable that can change a conversation.

    A part of a value (split on ``os.pathsep``) that names an existing file or
    folder counts by its contents, so the same bank in another place is the
    same setup, and an edited bank is a different one.
    """
    settings: dict[str, list[str]] = {}
    for name in sorted(os.environ):
        if not name.startswith("USERSIM_") or name in _OPERATIONAL_ENV:
            continue
        parts = os.environ[name].split(os.pathsep)
        settings[name] = [_contents_digest(Path(part)) if part and Path(part).exists() else part for part in parts]
    return settings


def _prompt_versions() -> dict[str, str]:
    """``PROMPT_VERSION`` of every registered probe, read as the episode preamble reads it."""
    from usersim.engine.core.probes import known_probes, resolve_probe

    return {
        label: str(getattr(sys.modules.get(resolve_probe(label).__module__), "PROMPT_VERSION", "v1.0"))
        for label in known_probes()
    }


def _extension_identities() -> list[dict[str, Any]]:
    """Every entry point User Sim discovers, with the package release behind it."""
    from importlib.metadata import entry_points

    from usersim.engine.core._extensions import ENTRY_POINT_GROUPS, extensions_disabled

    if extensions_disabled():
        return []
    identities: list[dict[str, Any]] = []
    for group in ENTRY_POINT_GROUPS:
        for entry in sorted(entry_points(group=group), key=lambda e: (e.name, e.value)):
            dist = getattr(entry, "dist", None)
            identities.append(
                {
                    "group": group,
                    "name": entry.name,
                    "value": entry.value,
                    "distribution": dist.name if dist else None,
                    "version": dist.version if dist else None,
                }
            )
    return identities


def _data_designer_version() -> str | None:
    """The installed Data Designer release, which shapes every model request."""
    from usersim.engine.core.manifest import _pkg_version

    return _pkg_version("data_designer")


def setup_fingerprint(models: Any, inputs: EpisodeInputs) -> tuple[str, dict[str, Any]]:
    """The fingerprint of everything about a setup that can change a conversation.

    Returns the fingerprint and the run record that names it. Covered:

    - every simulator model alias as Data Designer sends it, provider settings
      included (``api_key``, a provider's name and ``max_parallel_requests``
      are left out: they don't change what a model says);
    - the shared ``usersim_config`` as loaded, defaults applied;
    - the contents of every asset root, and every non-operational
      ``USERSIM_*`` variable;
    - every probe's ``PROMPT_VERSION``, every extension's package release, and
      the User Sim and Data Designer releases.

    A code change that bumps none of these versions is not seen.
    """
    from usersim.engine.config import MODEL_ALIASES, ConversationSimulatorConfig
    from usersim.engine.core.manifest import _own_version
    from usersim.engine.core.storage import RUN_KIND_STORED

    simulation = ConversationSimulatorConfig.model_validate(dict(inputs.config)).model_dump(mode="json")
    for key in _PER_ROW_SETTINGS:
        simulation.pop(key, None)
    aliases = [*MODEL_ALIASES, simulation.get("finance_embedding_model_alias") or ""]
    readable, hashed = _model_settings(models, [alias for alias in aliases if alias])
    fingerprint = _digest(
        {
            "models": hashed,
            "simulation": simulation,
            "assets": _asset_roots_digest(inputs.assets_dir),
            "environment": _environment_settings(),
            "prompts": _prompt_versions(),
            "extensions": _extension_identities(),
            "usersim": _own_version(),
            "data_designer": _data_designer_version(),
        }
    )
    record = {"kind": RUN_KIND_STORED, "fingerprint": fingerprint, "models": readable, "simulation": simulation}
    return fingerprint, record


def replay_trajectory_id(row: Mapping[str, Any], fingerprint: str) -> str:
    """The ``trajectory_id`` of ``row`` run under the setup ``fingerprint`` names.

    The stored row is hashed whole, apart from its machine-specific asset
    path, so a row whose contents changed under the same stored id is a
    different conversation.
    """
    stored = dict(row)
    stored["usersim_config"] = {key: value for key, value in row["usersim_config"].items() if key != "assets_dir"}
    composite = f"{_ID_SCHEME}|{_digest(stored)}|{fingerprint}"
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]


# ── running ─────────────────────────────────────────────────────────────


def _matching_run(out: Path, fingerprint: str) -> str | None:
    """The latest run of stored rows made with this setup.

    A run whose record does not say how it saved its columns, one made before
    runs recorded that, is never resumed: rows appended to it could change how
    its columns read back. A new run is started instead.
    """
    from usersim.engine.core.storage import RUN_KIND_STORED, latest_run_where

    def matches(run_id: str, record: dict[str, Any] | None) -> bool:
        if not record or record.get("kind") != RUN_KIND_STORED or record.get("fingerprint") != fingerprint:
            return False
        if "saved_columns" not in record:
            logger.warning(
                "run=%s holds stored rows with this setup but does not record how it saved their columns; "
                "starting a new run instead",
                run_id,
            )
            return False
        return True

    return latest_run_where(out, matches)


def _saved_inputs(out: Path, run_id: str) -> dict[str, set[str]]:
    """Each ``input_id`` saved in ``run_id``, with the trajectory ids saved under it."""
    from usersim.engine.core.storage import read_partitioned_dataset

    saved = read_partitioned_dataset(out, run=run_id, columns=["trajectory_id", INPUT_ID_COLUMN])
    pairs: dict[str, set[str]] = {}
    if not saved.empty:
        for input_id, trajectory_id in zip(saved[INPUT_ID_COLUMN].astype(str), saved["trajectory_id"].astype(str)):
            pairs.setdefault(input_id, set()).add(trajectory_id)
    return pairs


def _unsaved(planned: list[tuple[str, str]], saved: dict[str, set[str]], run_id: str) -> tuple[set[str], int]:
    """The ``(input_id, trajectory_id)`` pairs not yet saved in ``run_id``, and how many were.

    Raises ``ValueError`` for an input saved there under a different id: its
    stored contents changed, and one run must not hold two versions of it.
    """
    todo: set[str] = set()
    done = 0
    for input_id, trajectory_id in planned:
        prior = saved.get(input_id)
        if not prior:
            todo.add(input_id)
        elif trajectory_id in prior:
            done += 1
        else:
            raise ValueError(
                f"stored row {input_id!r} already ran in run {run_id} with different contents; give the "
                "changed row its own trajectory_id, or write to another --out"
            )
    return todo, done


@dataclass(frozen=True)
class InputsPlan:
    """What a run of stored rows would do, worked out before any model call."""

    fingerprint: str
    #: The record a new run with this setup is published with.
    record: dict[str, Any]
    #: The run of stored rows with this setup that a resume reopens, if any.
    matching_run: str | None
    #: Stored ids not yet saved in that run.
    pending: set[str]
    #: Rows already saved there.
    saved: int


def plan_inputs_run(inputs: EpisodeInputs, models: Any, out: str | Path, *, skip_existing: bool = True) -> InputsPlan:
    """Fingerprint the setup and find what the matching run already holds.

    Raises ``ValueError`` if that run holds one of the stored rows with
    different contents, or saved one of their columns in a way these rows'
    values cannot keep.
    """
    fingerprint, record = setup_fingerprint(models, inputs)
    stored_ids = {row["trajectory_id"] for row in inputs.rows}
    matching = _matching_run(Path(out).resolve(), fingerprint) if skip_existing else None
    if matching is None:
        return InputsPlan(fingerprint, record, matching_run=None, pending=stored_ids, saved=0)
    planned = [(row["trajectory_id"], replay_trajectory_id(row, fingerprint)) for row in inputs.rows]
    pending, saved = _unsaved(planned, _saved_inputs(Path(out).resolve(), matching), matching)
    _settle_encodings(_column_encodings(inputs.rows), _recorded_encodings(Path(out).resolve(), matching), matching)
    return InputsPlan(fingerprint, record, matching_run=matching, pending=pending, saved=saved)


def _column_config(inputs: EpisodeInputs, locale: str):
    """The simulator column for ``locale``'s rows, on the run's asset root."""
    from usersim.engine.config import ConversationSimulatorConfig

    return ConversationSimulatorConfig.model_validate(
        {**inputs.config, "locale": locale, "assets_dir": str(inputs.assets_dir)}
    )


#: How a run saves a stored column it adds back: as JSON text, or as one plain type.
_JSON = "json"
_PLAIN_ENCODINGS = {str: "text", bool: "bool", int: "int", float: "float"}


def _column_encodings(rows: list[dict[str, Any]]) -> dict[str, str]:
    """How these rows' stored columns would be saved, by column.

    A column keeps its values as they are (``text``, ``bool``, ``int`` or
    ``float``) when every value present is of that one plain type, an ``int``
    within 64 bits. Anything else, a nested value or a mix such as an ``int``
    beside a ``float``, would be converted or refused by Arrow, so it is saved
    as JSON text (``json``), which keeps every value exactly. A column with no
    value present has no encoding yet.
    """
    kinds: dict[str, set[type]] = {}
    oversized: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if value is None:
                continue
            kinds.setdefault(key, set()).add(type(value))
            if type(value) is int and not -(2**63) <= value < 2**63:
                oversized.add(key)
    encodings: dict[str, str] = {}
    for key, types in kinds.items():
        (only,) = types if len(types) == 1 else (None,)
        plain = _PLAIN_ENCODINGS.get(only)
        encodings[key] = _JSON if key in oversized or plain is None else plain
    return encodings


def _settle_encodings(proposed: dict[str, str], recorded: Mapping[str, str], run_id: str | None) -> dict[str, str]:
    """The encodings a save into ``run_id`` uses: the run's recorded ones, then these rows' own.

    A run saves each column one way across invocations, so its partitions read
    back as one type. JSON text holds any value; a plain encoding holds only its
    own type, so rows it cannot hold exactly are refused rather than saved as a
    different type that reading the run would convert.
    """
    settled = dict(proposed)
    for column, encoding in recorded.items():
        wanted = proposed.get(column)
        if wanted is not None and wanted != encoding and encoding != _JSON:
            raise ValueError(
                f"stored column {column!r} was saved as {encoding} in run {run_id}, and these rows hold values "
                f"saved as {wanted}, which that column cannot keep exactly; write them to another --out"
            )
        settled[column] = encoding
    return settled


def _recorded_encodings(out: Path, run_id: str | None) -> dict[str, str]:
    """The encodings ``run_id`` has saved its added-back columns with."""
    from usersim.engine.core.storage import read_run_record

    if run_id is None:
        return {}
    return dict((read_run_record(out, run_id) or {}).get("saved_columns") or {})


def _record_encodings(out: Path, run_id: str, columns: dict[str, str]) -> None:
    """Add ``columns`` to the encodings ``run_id``'s record keeps.

    Call while holding the write lock, before writing the rows, so a column is
    never saved without its encoding on record. The record is replaced
    atomically, so a reader sees the old one or the new one.
    """
    from usersim.engine.core.storage import RUN_RECORD_FILENAME, read_run_record, run_subroot

    record = read_run_record(out, run_id) or {}
    saved = {**(record.get("saved_columns") or {}), **columns}
    if saved == record.get("saved_columns"):
        return
    path = run_subroot(out, run_id) / RUN_RECORD_FILENAME
    partial = path.with_name(f".{RUN_RECORD_FILENAME}.partial")
    partial.write_text(json.dumps({**record, "saved_columns": saved}, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)


def _as_saved_rows(df, stored: dict[str, tuple[dict[str, Any], str]], encodings: dict[str, str]):
    """The rows a run saves, and the stored columns it added back.

    The rows get the new ids, the stored id as ``input_id``, and a plain run's
    columns. ``df`` comes back from Data Designer with each stored row's own
    id as ``trajectory_id``. The carrier column and the full persona are
    dropped, as a plain run drops the persona, and a stored column the run did
    not keep is added back from the stored rows, as JSON text where
    ``encodings`` says.
    """
    import pandas as pd

    from usersim.engine.core.episode_input import EPISODE_INPUT_COLUMN

    df = df.copy()
    input_ids = df["trajectory_id"].astype(str).tolist()
    df[INPUT_ID_COLUMN] = input_ids
    df["trajectory_id"] = [stored[input_id][1] for input_id in input_ids]
    df = df.drop(columns=[EPISODE_INPUT_COLUMN, "persona"], errors="ignore")
    missing = sorted(
        {key for row, _ in stored.values() for key in row} - set(df.columns) - {"usersim_config", "persona"}
    )
    for key in missing:
        values = [stored[input_id][0].get(key) for input_id in input_ids]
        if encodings.get(key) == _JSON:
            values = [None if value is None else json.dumps(value, ensure_ascii=False) for value in values]
        df[key] = pd.Series(values, index=df.index, dtype=object)
    return df, missing


def simulate_inputs_to_dataset(
    *,
    inputs: EpisodeInputs,
    models: Any,
    out: str | Path,
    models_path: str | Path | None = None,
    skip_existing: bool = True,
    plan: InputsPlan | None = None,
) -> InputsRun:
    """Run stored rows with ``models`` and save them as one run under ``out``.

    Each locale's rows run as one Data Designer batch, through the simulator
    column ``usersim simulate`` uses. With ``skip_existing`` (the default), the
    rows go into the latest run with this setup, skipping rows already saved
    there; otherwise they start a new run. ``plan`` is
    :func:`plan_inputs_run`'s result for the same arguments, when the caller
    already has it.

    Raises ``ValueError`` when the run already holds a stored row with
    different contents, or saved a stored column in a way these rows' values
    cannot keep: before any model call when it is found up front, and when the
    rows are saved when another writer saved them meanwhile.
    """
    from usersim.cli._pipeline import build_inputs_config_builder
    from usersim.engine.core.manifest import write_simulator_manifest
    from usersim.engine.core.storage import (
        publish_new_run,
        run_started_at,
        run_write_lock,
        write_locale_partition,
    )

    started_at = time.time()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A first look at the matching run, to skip saved rows before any model
    # call and refuse changed ones early. Only the check under the lock counts.
    plan = plan or plan_inputs_run(inputs, models, out, skip_existing=skip_existing)
    fingerprint, matching, pending, skipped = plan.fingerprint, plan.matching_run, plan.pending, plan.saved
    record = {**plan.record, "started_at": started_at}
    stored = {row["trajectory_id"]: (row, replay_trajectory_id(row, fingerprint)) for row in inputs.rows}
    logger.info("setup %s: %d stored rows, %d already saved", fingerprint[:12], len(stored), skipped)

    by_locale: dict[str, dict[str, tuple[dict[str, Any], str]]] = {}
    for input_id in sorted(pending):
        row, new_id = stored[input_id]
        by_locale.setdefault(row["usersim_config"]["locale"], {})[input_id] = (row, new_id)

    # The rows go into the run the plan found, even if a newer run with this
    # setup appears meanwhile, so they never split across two runs.
    run_id: str | None = matching
    resumed = matching is not None
    encodings = _column_encodings(inputs.rows)
    written = 0
    requested: Counter[str] = Counter()
    for locale, group in by_locale.items():
        requested[locale] += len(group)
        data_designer, builder = build_inputs_config_builder(
            models=models, config=_column_config(inputs, locale), rows=[row for row, _ in group.values()]
        )
        # Data Designer names a batch's artifacts by the second it started, so
        # two invocations in one directory could share them; each batch gets
        # its own folder, removed once its rows are loaded.
        with tempfile.TemporaryDirectory(prefix="usersim_inputs_") as artifacts:
            result = data_designer.create(builder, num_records=len(group), artifact_path=artifacts)
            generated = result.load_dataset()
        with run_write_lock(out):
            target = run_id or (_matching_run(out, fingerprint) if skip_existing else None)
            settled = _settle_encodings(encodings, _recorded_encodings(out, target), target)
            df, added = _as_saved_rows(generated, group, settled)
            columns = {column: settled[column] for column in added if column in settled}
            if target is not None:
                planned = list(zip(df[INPUT_ID_COLUMN].astype(str), df["trajectory_id"].astype(str)))
                todo, done = _unsaved(planned, _saved_inputs(out, target), target)
                df = df[df[INPUT_ID_COLUMN].astype(str).isin(todo)]
                skipped += done
                # This invocation keeps the run it found, even if every row
                # was saved there meanwhile.
                resumed = resumed or run_id is None
                run_id = target
            if df.empty:
                continue
            if target is None:
                run_id = publish_new_run(df, out, locale=locale, record={**record, "saved_columns": columns})
            else:
                _record_encodings(out, target, columns)
                write_locale_partition(df, out, locale=locale, run_id=target)
        written += len(df)
        logger.info("locale=%s: wrote %d trajectories from stored rows into run=%s", locale, len(df), run_id)

    if written and run_id is not None:
        probes = Counter(str(row.get("probe_type")) for row, _ in stored.values())
        try:
            write_simulator_manifest(
                run_id=run_id,
                started_at_epoch=run_started_at(out, run_id) or int(started_at),
                trajectory_root=out,
                models_config=models,
                models_config_path=Path(models_path) if models_path else None,
                sim_config=_column_config(inputs, next(iter(requested))),
                locales_requested=sorted(requested),
                probe_mix={probe: count / len(stored) for probe, count in sorted(probes.items())},
                panel_path=inputs.source,
                assets_dir=inputs.assets_dir,
                trajectories_requested=dict(requested),
            )
        except Exception as error:  # pragma: no cover - never fail a finished run on manifest issues
            logger.warning("failed to write run manifest: %s", error)
    return InputsRun(run_id=run_id, resumed=resumed, written=written, skipped=skipped)
