# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hive-partitioned trajectory + evaluation parquet storage.

Trajectories and evaluator outputs are written as Hive-partitioned
datasets keyed by ``(run, locale, probe_family)``: e.g.,
``output/trajectories/run=1714074853/locale=en_US/probe_family=safety/part-0.parquet``.

Three partition levels:

1. ``run=<unix-epoch-seconds>`` — top-level isolation per simulator
   invocation. Every fresh ``cli simulate`` (or notebook
   ``RESUME=False``) mints a new run id; ``RESUME=True`` resumes
   into the latest existing run. Eval and report stages inherit the
   trajectory run's id so a single run is the unit of "X's
   trajectories + X's evaluations + X's reports."
2. ``locale=<locale-code>`` — one directory per locale, written
   atomically per-locale so a crash mid-run leaves earlier locales'
   partitions intact and resumable.
3. ``probe_family=<family>`` — one directory per probe family
   inside each locale, so selective reads (``filters=[("probe_family",
   "=", "tool_calling")]``) skip the irrelevant partitions.

The reader defaults to "latest run" so notebooks + downstream
tooling see the most recent results without explicit configuration.
Pass ``run="all"`` to load every run for cross-run comparison or
``run="<epoch-seconds>"`` to pin a specific historical run.

Legacy fallback: a directory that contains bare ``locale=*/`` /
``probe_family=*/`` subdirectories with NO ``run=*`` parent (the
older layout, before run-partitioning) is read transparently — it
surfaces as a synthetic single run with id ``"legacy"``. Existing
on-disk data keeps working without manual migration; new writes
always use the run-partitioned layout.

Schema unification: different probes write different side-channel
columns (``query_id`` for sov_ai_multilingual_parity,
``attempted_actions`` for safety_agentic, etc.) and historical-drift
recovery ensures that cross-fragment ``string`` vs ``double`` (from
all-NaN columns in older fragments) promotes cleanly to ``string``.
See ``_unify_fragment_schemas_robust`` for the recovery logic.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("usersim.engine")


DEFAULT_PARTITION_COLS: tuple[str, ...] = ("locale", "probe_family")

# Top-level partition column for run isolation. Inserted ABOVE
# DEFAULT_PARTITION_COLS in the on-disk layout
# (``<root>/run=<id>/locale=<l>/probe_family=<f>/...``) but kept
# distinct from DEFAULT_PARTITION_COLS so callers can construct
# the per-run subroot via ``root / f"run={run_id}"`` and then call
# the existing ``write_partitioned_dataset`` against that subroot.
RUN_PARTITION_COL: str = "run"

# Sentinel run id used by the reader when it walks a bare
# ``locale=*/probe_family=*/`` tree (the older flat layout, before
# run-partitioning). Lets legacy data co-exist with the new
# ``run=<epoch>`` partitions without manual migration.
LEGACY_RUN_ID: str = "legacy"


def new_run_id(now: float | None = None) -> str:
    """Generate a fresh run id for a simulator invocation.

    The id is the integer Unix epoch second at the moment of the
    call — sortable as a string (Python's ``str.sort()`` is
    lexicographic but for the same-length integer strings this
    matches numeric order until 2286, well past any realistic
    horizon), unambiguous across timezones, and trivially
    convertible back to a human-readable timestamp via
    ``time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(run_id)))``.

    Tests pass ``now=<fixed-value>`` for reproducible run ids.
    """
    epoch = int(now) if now is not None else int(time.time())
    return str(epoch)


def list_runs(root: str | Path) -> list[str]:
    """Return the list of run ids present under ``root``, oldest first.

    A "run" is a top-level subdirectory whose name starts with
    ``run=`` (i.e. the standard Hive partitioning encoding). Returns
    an empty list if ``root`` doesn't exist or contains no
    ``run=*`` subdirectories. Legacy bare-layout trees (no
    ``run=*`` partitions but bare ``locale=*/`` subdirectories) are
    surfaced as a single ``"legacy"`` run id.
    """
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []
    runs: list[str] = []
    has_legacy = False
    for entry in sorted(root_path.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("run="):
            runs.append(entry.name[len("run=") :])
        elif entry.name.startswith("locale="):
            has_legacy = True
    if not runs and has_legacy:
        return [LEGACY_RUN_ID]
    return runs


def latest_run_id(root: str | Path) -> str | None:
    """Return the most-recent run id under ``root``, or ``None`` if
    no runs exist. ``"legacy"`` is treated as the oldest possible
    run (i.e. it is returned only when no real runs exist)."""
    runs = list_runs(root)
    if not runs:
        return None
    real = [r for r in runs if r != LEGACY_RUN_ID]
    if real:
        return max(real)
    return LEGACY_RUN_ID


def resolve_run(root: str | Path, run: str | None) -> str | None:
    """Resolve a caller-facing ``run`` argument into a concrete id.

    - ``None`` or ``"latest"`` → the most-recent run under ``root``,
      or ``None`` when the root is empty (so callers can short-circuit
      with an empty result rather than raising on first-run setups).
    - ``"all"`` → returned as-is; the reader interprets this as
      "concatenate every run".
    - any other string → treated as a literal run id; the reader
      validates that ``root/run=<id>/`` exists and raises otherwise.
    """
    if run is None or run == "latest":
        return latest_run_id(root)
    return run


def resolve_run_or_raise(
    root: str | Path,
    run: str | int | None,
    *,
    label: str = "run",
) -> str:
    """``resolve_run`` plus a friendly raise on failure.

    - ``None``/``"latest"`` → most-recent run under ``root``; raises
      ``FileNotFoundError`` listing the (empty) run set when no runs exist.
    - ``"all"`` → returned as-is (no existence check; the reader handles it).
    - any other string or int → must resolve to an existing
      ``<root>/run=<id>/`` directory; raises ``FileNotFoundError``
      listing the available runs otherwise.

    Integers are coerced to ``str`` so a user-typed
    ``RUN_ID = 1778530815`` (no quotes) works just like
    ``RUN_ID = "1778530815"``; the on-disk partition key is a string
    either way.

    ``label`` names the caller-side knob in the error message
    (e.g. ``label="SIM_RUN_ID"``) so notebook users see which knob to fix.
    """
    # Coerce int-typed run ids gracefully -- the most common notebook
    # foot-gun is typing the epoch as a bare integer literal rather than
    # a string.
    if run is not None and run not in ("latest", "all"):
        run = str(run)
    if run == "all":
        return "all"
    available = list_runs(root)
    resolved = resolve_run(root, run)
    if resolved is None:
        raise FileNotFoundError(f"{label}: no runs under {root}" + (f" (available: {available})" if available else ""))
    if run not in (None, "latest", LEGACY_RUN_ID) and resolved not in available:
        raise FileNotFoundError(f"{label}={run!r} not found under {root} (available: {available})")
    return resolved


def write_run_partition(
    df,
    root: str | Path,
    run_id: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``df`` under ``<root>/run=<run_id>/`` as a Hive-partitioned
    dataset (the standard ``locale=*/probe_family=*/`` layout below the
    run partition). Returns the run subroot path.

    With ``overwrite=True``, blasts the existing ``run=<run_id>/`` subtree
    first and emits an info-level log line — avoids silent destruction
    of a previous run when the caller is re-running a partial pipeline.
    """
    subroot = run_subroot(root, run_id)
    if overwrite and subroot.exists():
        logger.info("Overwriting existing run partition at %s", subroot)
        shutil.rmtree(subroot)
    subroot.mkdir(parents=True, exist_ok=True)
    write_partitioned_dataset(df, subroot)
    return subroot


def write_run_locales_partition(
    df,
    root: str | Path,
    run_id: str,
    *,
    locales: Sequence[str],
    overwrite: bool = True,
) -> Path:
    """Write ``df`` under ``<root>/run=<run_id>/`` for ONLY the given
    locales, leaving any other locales' partitions in the same run
    untouched.

    Use this for **locale-targeted re-evaluation** -- e.g., re-running
    the evaluator on en_US after an infra issue, or adding a newly
    simulated locale to an existing eval run, without nuking the eval
    data for the other 7 locales.

    Behaviour:

    - ``df`` MUST contain only rows whose ``locale`` value is in
      ``locales``. The check is a hard pre-condition; mixing locales
      that aren't in the set silently writes them, which would defeat
      the targeted-overwrite semantics. Raises ``ValueError`` on
      mismatch.
    - With ``overwrite=True`` (the default for this helper -- different
      from ``write_run_partition``'s default), each target locale's
      subtree (``<run>/locale=<l>/``) is ``rmtree``'d before the write.
      That gives clean overwrite semantics for the targeted locales
      (no leftover part-files from earlier eval runs) while pyarrow's
      ``existing_data_behavior="overwrite_or_ignore"`` keeps the write
      atomic-ish for the rest of the run.
    - With ``overwrite=False``, the existing partitions for the target
      locales are preserved -- the new write APPENDS additional part
      files. The basename token is suffixed with the writer pid so
      filenames don't collide. Useful for additive workflows (initial
      eval was partial, fill in the missing locales).

    Returns the run subroot path.
    """
    if "locale" not in df.columns:
        raise ValueError("write_run_locales_partition: dataframe is missing 'locale' column")
    locales_seq = list(locales)
    if not locales_seq:
        raise ValueError(
            "write_run_locales_partition: locales must be a non-empty "
            "sequence; pass write_run_partition() if you want a "
            "whole-run write."
        )
    target = {str(loc) for loc in locales_seq}
    df_locales = {str(v) for v in df["locale"].unique() if v is not None}
    extra = df_locales - target
    if extra:
        raise ValueError(
            f"write_run_locales_partition: df contains locales {sorted(extra)} "
            f"not in target locales {sorted(target)}. Filter the dataframe "
            f"to the target subset before calling this helper -- mixing "
            f"locales would silently write rows that defeat the "
            f"targeted-overwrite contract."
        )
    if df.empty:
        # Nothing to write. Don't rmtree empty target locales -- the
        # caller may have legitimately filtered to zero rows (e.g., an
        # empty trajectory subset) and expects existing partitions to
        # stay intact.
        return run_subroot(root, run_id)

    subroot = run_subroot(root, run_id)
    if overwrite:
        for loc in locales_seq:
            locale_dir = subroot / f"locale={_format_partition_value(loc)}"
            if locale_dir.exists():
                logger.info("Overwriting existing locale partition at %s", locale_dir)
                shutil.rmtree(locale_dir)
    subroot.mkdir(parents=True, exist_ok=True)
    if overwrite:
        write_partitioned_dataset(df, subroot)
    else:
        # Append: include a writer-id token so concurrent / repeated
        # appends to the same (locale, probe_family) cell don't collide.
        import os

        writer_id = f"pid{os.getpid()}"
        write_partitioned_dataset(
            df,
            subroot,
            basename_template=f"part-{{i}}-{writer_id}.parquet",
        )
    return subroot


def run_subroot(root: str | Path, run_id: str) -> Path:
    """Return the on-disk path for a single run's data:
    ``<root>/run=<run_id>/`` for real runs, or ``<root>/`` itself
    for the legacy sentinel (where the bare layout lives directly
    under root).
    """
    root_path = Path(root)
    if run_id == LEGACY_RUN_ID:
        return root_path
    return root_path / f"run={run_id}"


def write_partitioned_dataset(
    df,
    root: str | Path,
    *,
    partition_cols: Sequence[str] = DEFAULT_PARTITION_COLS,
    basename_template: str | None = None,
    existing_data_behavior: str = "overwrite_or_ignore",
) -> Path:
    """Write ``df`` to a Hive-partitioned parquet dataset rooted at ``root``.

    Wraps ``pyarrow.parquet.write_to_dataset`` with the conventions the
    framework standardizes on:

    - Partition columns become directory levels:
      ``root/locale=en_US/probe_family=safety/part-N.parquet``.
    - ``existing_data_behavior="overwrite_or_ignore"`` makes per-locale
      atomic writes safe: writing a new locale's partition does not
      touch any other locale's partition. Pre-existing files in the
      same partition are silently overwritten — desired since per-locale
      writes are designed to be idempotent.
    - A ``basename_template`` argument lets the caller include a
      caller-unique token (e.g., ``"part-{i}-locale-en_US.parquet"``)
      so multiple writers can append to the same partition without
      filename collisions; the default (``None``) defers to pyarrow's
      ``part-N.parquet`` naming.

    Returns the resolved root path.

    Raises ``ValueError`` if ``df`` is missing any of ``partition_cols``;
    silently wrong partitioning would break downstream reads.
    """
    import os
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq

    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    missing_cols = [c for c in partition_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"write_partitioned_dataset: dataframe missing partition "
            f"column(s) {missing_cols}; have columns "
            f"{sorted(df.columns)}"
        )

    # PyArrow's ``write_to_dataset`` / ``dataset.write_dataset`` keep
    # the partition columns inside each parquet file's row groups by
    # default — so the values live in BOTH the directory names AND the
    # row groups. We do the partitioning manually to avoid that
    # duplication: group by the partition columns, drop them from each
    # group's row payload, and write each group to its computed
    # ``key=value/`` directory. The dataset reader reconstructs the
    # partition columns from the directory paths on read, so the
    # round-trip is lossless and on-disk bytes shrink (especially for
    # low-cardinality columns like ``locale`` that pay a per-row
    # dictionary-index cost in the row group).
    partition_cols_list = list(partition_cols)
    non_partition_cols = [c for c in df.columns if c not in partition_cols_list]

    if basename_template is not None:
        # ``{i}`` placeholder lets multiple writers compose without
        # filename collisions in the same partition directory.
        if "{i}" in basename_template:
            filename = basename_template.format(i=uuid.uuid4().hex[:12])
        else:
            filename = basename_template
    else:
        filename = f"part-{os.getpid()}-{uuid.uuid4().hex[:8]}.parquet"

    # ``df.groupby`` on multiple columns yields a tuple of values per
    # group when len > 1, a single scalar when len == 1. Normalize to
    # always-tuple for path-building.
    grouped = df.groupby(partition_cols_list, sort=False, dropna=False)
    for group_keys, group_df in grouped:
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        path_parts = [f"{col}={_format_partition_value(val)}" for col, val in zip(partition_cols_list, group_keys)]
        out_dir = root_path.joinpath(*path_parts)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        if existing_data_behavior == "overwrite_or_ignore" and out_path.exists():
            # Same writer, same partition: overwrite (idempotent rerun).
            pass
        elif existing_data_behavior == "error" and out_path.exists():
            raise FileExistsError(out_path)
        # Subset to non-partition columns + write.
        sub_df = group_df[non_partition_cols]
        table = pa.Table.from_pandas(sub_df, preserve_index=False)
        pq.write_table(table, out_path)

    return root_path


def _format_partition_value(value) -> str:
    """Render a partition value as a directory-name-safe string.

    Mirrors pyarrow's Hive partitioning encoding (which itself mirrors
    Apache Hive's): values are stringified and minimally escaped.
    Slashes / equals signs would break path parsing, but in practice
    our partition columns (``locale``, ``probe_family``) carry only
    snake_case identifiers, so we keep the encoding simple.
    """
    s = str(value)
    if "/" in s:
        s = s.replace("/", "%2F")
    if "=" in s:
        s = s.replace("=", "%3D")
    return s


def read_partitioned_dataset(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    run: str | None = None,
):
    """Read a parquet path, transparently handling single files and dataset dirs.

    - If ``path`` is a regular ``.parquet`` file, reads it via
      ``pyarrow.parquet.read_table`` (single-file legacy support);
      the ``run`` argument is ignored.
    - If ``path`` is a directory, treats it as a Hive-partitioned
      dataset and reads via ``pyarrow.dataset``. Partition columns
      (``run`` / ``locale`` / ``probe_family``) are auto-promoted
      into the returned DataFrame as regular columns.
    - Schema-union across partitions is enabled. Different probes
      that wrote different side-channel columns into different
      partitions all surface as a single wide DataFrame with
      ``None``-filled cells where the column did not exist in the
      source partition. Cross-fragment type drift (the
      ``string`` vs ``double``-from-NaN problem) is recovered via
      a per-name promotion priority — see
      ``_unify_fragment_schemas_robust``.

    Run scoping:

    - ``run=None`` (default) or ``run="latest"``: read only the
      most-recent run under ``path``. Returns an empty DataFrame
      if no runs exist.
    - ``run="all"``: read every run under ``path``; the resulting
      DataFrame includes a ``run`` column that callers can group
      / aggregate on for cross-run comparison.
    - ``run="<epoch-seconds>"``: pin to a specific historical run.
      Raises ``FileNotFoundError`` if that run does not exist.
    - Bare legacy layout (no ``run=*`` partitions, just direct
      ``locale=*/`` subdirs): treated as a single synthetic run with
      id ``"legacy"``; ``run="latest"`` reads it when no real runs
      are present.

    Returns a pandas DataFrame with ``dtype_backend="pyarrow"``.

    Raises ``FileNotFoundError`` if ``path`` does not exist (or if a
    pinned ``run`` id does not exist under ``path``).
    """
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.is_file():
        table = pq.read_table(p, columns=columns)
        return table.to_pandas(types_mapper=_pyarrow_type_mapper())

    # Run scoping. Resolve to either a single run subroot (default)
    # or to the parent root with run="all" semantics applied.
    if run == "all":
        return _read_all_runs(p, columns=columns)

    resolved = resolve_run(p, run)
    if resolved is None:
        # Empty root — no runs and no legacy bare layout. Return an
        # empty DataFrame so notebooks short-circuit cleanly on a
        # first-run setup.
        import pandas as pd

        return pd.DataFrame()

    subroot = run_subroot(p, resolved)
    if not subroot.exists():
        raise FileNotFoundError(f"run={resolved!r} not found under {p} (available: {list_runs(p)})")
    p = subroot

    # Directory — read as a partitioned dataset, with schema
    # unification across partitions. Different probes write different
    # side-channel columns; the default ``pads.dataset(...)`` infers
    # the schema from one fragment and silently drops columns that
    # only exist in others. Walking every fragment's schema and
    # passing the union back as the dataset schema is the supported
    # workaround (pyarrow does not have an auto-union read mode for
    # parquet datasets).
    initial = pads.dataset(str(p), format="parquet", partitioning="hive")
    fragment_schemas = [frag.physical_schema for frag in initial.get_fragments()]
    if fragment_schemas:
        unified = _unify_fragment_schemas_robust(fragment_schemas)
        # Re-add the partition fields so the dataset reader still
        # promotes them. Hive partitioning derives them from the
        # directory names; the unified schema we computed above only
        # carries the per-file (non-partition) columns.
        for partition_field in initial.partitioning.schema:
            if partition_field.name not in unified.names:
                unified = unified.append(partition_field)
        # Read each fragment individually and cast it to the unified
        # schema. ``pads.dataset(...).to_table()`` honours the
        # `schema=` arg only for column projection, not type
        # coercion — when a fragment's column is e.g. ``double`` but
        # the unified schema says ``string``, the dataset read raises
        # ``ArrowInvalid``. Per-fragment cast is the only way to
        # reconcile post-hoc schema drift.
        try:
            return _read_fragments_with_cast(
                initial,
                unified,
                columns=columns,
            )
        except Exception:
            # Fallback to the older non-casting path. Lets a user
            # whose dataset doesn't actually have type drift get the
            # cheaper read path; the cast path is slower per-fragment.
            dataset = pads.dataset(
                str(p),
                format="parquet",
                partitioning="hive",
                schema=unified,
            )
            table = dataset.to_table(columns=columns)
            return table.to_pandas(types_mapper=_pyarrow_type_mapper())
    else:
        dataset = initial
    table = dataset.to_table(columns=columns)
    return table.to_pandas(types_mapper=_pyarrow_type_mapper())


# Type promotion order for cross-fragment unification. The "richest"
# representation wins for a given field name. ``string`` beats
# ``double`` / ``int*`` / ``null`` because side-channel columns that
# pandas writes as ``object`` (string) but PyArrow infers as ``double``
# when all-NaN are the most common drift source — see the storage
# regression that landed when broken-run residuals (failed-probe
# trajectories with side-channel columns left None) co-existed with
# healthy-run output (same columns, populated as strings).
_PROMOTION_PRIORITY: tuple = (
    "large_list",
    "list",
    "large_string",
    "string",
    "binary",
    "timestamp",
    "duration",
    "double",
    "float",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint64",
    "uint32",
    "uint16",
    "uint8",
    "bool",
    "null",
)


def _type_priority(t) -> int:
    """Return a small integer rank where lower = higher promotion
    priority. Unknown types get the lowest priority so they never
    silently override the typed columns we know how to compare."""
    s = str(t)
    for i, prefix in enumerate(_PROMOTION_PRIORITY):
        if s.startswith(prefix):
            return i
    return len(_PROMOTION_PRIORITY)


def _unify_fragment_schemas_robust(fragment_schemas):
    """Unify a list of ``pyarrow.Schema`` objects, falling back to
    name-by-name promotion when ``pa.unify_schemas`` rejects
    incompatible types (the historical-drift case).

    The first attempt uses ``pa.unify_schemas(...)``. When that fails
    (typically with ``Unable to merge: Field X has incompatible
    types: double vs string`` after a partial-run residual + healthy
    re-run), we walk every (name → type) tuple across all fragment
    schemas and pick the highest-priority type per
    :data:`_PROMOTION_PRIORITY` (string > numeric > null). Returns a
    ``pa.Schema`` callers can pass back to the dataset read as the
    target schema for cast-on-read.
    """
    import pyarrow as pa

    try:
        return pa.unify_schemas(fragment_schemas, promote_options="default")
    except pa.lib.ArrowTypeError:
        pass

    # Per-name promotion fallback.
    chosen: dict = {}
    for sch in fragment_schemas:
        for field in sch:
            current = chosen.get(field.name)
            if current is None or _type_priority(field.type) < _type_priority(current.type):
                chosen[field.name] = field
    # Preserve first-fragment field order (mirrors ``unify_schemas``
    # ordering semantics).
    seen = set()
    ordered_fields = []
    for sch in fragment_schemas:
        for name in sch.names:
            if name not in seen:
                ordered_fields.append(chosen[name])
                seen.add(name)
    return pa.schema(ordered_fields)


def _read_fragments_with_cast(initial, unified, *, columns):
    """Read every fragment individually, cast it to the unified
    schema, and return the concatenated pandas DataFrame.

    Per-fragment cast is the only way to reconcile a fragment whose
    on-disk column type differs from the unified schema's type
    (the dataset reader's ``schema=`` arg only does projection +
    re-ordering, not coercion). Cells that need coercion go through
    ``pa.compute.cast(..., safe=False)`` so a numeric all-NaN column
    promotes to a string-typed all-null column without raising.

    Partition columns (from Hive-style directory paths like
    ``locale=en_US/probe_family=tool_calling/...``) are re-injected
    per-fragment because per-fragment reads via ``frag.to_table()``
    only return the on-disk data — the partition-promotion logic
    that ``Dataset.to_table()`` runs is bypassed.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc

    target_by_name = {f.name: f.type for f in unified}
    partition_fields = list(initial.partitioning.schema)
    partition_field_names = {f.name for f in partition_fields}

    # If the caller projected away partition columns, drop them from
    # the re-injection list so we don't add them back.
    if columns is not None:
        wanted = set(columns)
        partition_fields = [f for f in partition_fields if f.name in wanted]
        partition_field_names = {f.name for f in partition_fields}

    parts: list = []
    for frag in initial.get_fragments():
        # Project only non-partition columns at read time — partition
        # values come from the path, not from the file body.
        if columns is None:
            file_columns = None
        else:
            file_columns = [c for c in columns if c not in partition_field_names]
        tab = frag.to_table(columns=file_columns)

        # Cast misaligned columns toward the unified schema.
        new_arrays = []
        new_fields = []
        for col_name in tab.schema.names:
            arr = tab.column(col_name)
            target_type = target_by_name.get(col_name)
            if target_type is None or arr.type == target_type:
                new_arrays.append(arr)
                new_fields.append(pa.field(col_name, arr.type))
                continue
            try:
                cast_arr = pc.cast(arr, target_type, safe=False)
            except (pa.lib.ArrowInvalid, pa.lib.ArrowNotImplementedError):
                cast_arr = pc.cast(arr, pa.string(), safe=False)
            new_arrays.append(cast_arr)
            new_fields.append(pa.field(col_name, cast_arr.type))

        # Re-inject partition columns from the fragment path.
        partition_values = _partition_values_from_fragment(
            frag,
            partition_fields,
        )
        for field in partition_fields:
            value = partition_values.get(field.name)
            arr = pa.array([value] * tab.num_rows, type=field.type)
            new_arrays.append(arr)
            new_fields.append(field)

        parts.append(pa.Table.from_arrays(new_arrays, schema=pa.schema(new_fields)))

    if not parts:
        return pd.DataFrame()

    # Align every part to the unified schema (column-superset),
    # filling missing columns with nulls of the right type.
    # If the caller asked for a column projection, the unified schema
    # is filtered down to exactly those columns so missing-column
    # backfill doesn't widen the projection.
    if columns is None:
        target_fields = list(unified)
    else:
        wanted = set(columns)
        target_fields = [f for f in unified if f.name in wanted]

    aligned: list = []
    for tab in parts:
        present = {n: tab.column(n) for n in tab.schema.names}
        cols = []
        fields = []
        for f in target_fields:
            if f.name in present:
                col = present[f.name]
                if col.type != f.type:
                    col = pc.cast(col, f.type, safe=False)
                cols.append(col)
            else:
                cols.append(pa.nulls(tab.num_rows, type=f.type))
            fields.append(pa.field(f.name, f.type))
        aligned.append(pa.Table.from_arrays(cols, schema=pa.schema(fields)))

    combined = pa.concat_tables(aligned, promote_options="default")
    return combined.to_pandas(types_mapper=_pyarrow_type_mapper())


def _read_all_runs(root: Path, *, columns: list[str] | None = None):
    """Read every run under ``root`` and concatenate, tagging each row
    with its source ``run`` id.

    Reads each run via ``read_partitioned_dataset(..., run=<id>)`` so
    the per-run schema-drift recovery still applies. Adds a ``run``
    column to every resulting frame so downstream code can group
    on it.
    """
    import pandas as pd

    runs = list_runs(root)
    if not runs:
        return pd.DataFrame()

    parts = []
    for run_id in runs:
        # If the caller projected `columns`, ensure the synthetic
        # ``run`` column we add is preserved.
        sub_cols = list(set(columns) | {"run"}) if columns else None
        sub_df = read_partitioned_dataset(root, columns=sub_cols, run=run_id)
        if "run" not in sub_df.columns:
            sub_df["run"] = run_id
        parts.append(sub_df)
    return pd.concat(parts, ignore_index=True, sort=False)


def _partition_values_from_fragment(frag, partition_fields) -> dict:
    """Parse Hive-style partition values from the fragment's file
    path: ``.../locale=en_US/probe_family=tool_calling/part-*.parquet``
    → ``{"locale": "en_US", "probe_family": "tool_calling"}``.

    PyArrow fragments expose ``frag.partition_expression``, but it is
    a structured Expression that's awkward to walk; the path-string
    parse is straightforward and matches the Hive partitioning
    contract exactly.
    """
    field_names = {f.name for f in partition_fields}
    parts: dict = {}
    for segment in Path(frag.path).parts:
        if "=" in segment:
            key, _, val = segment.partition("=")
            if key in field_names:
                parts[key] = val
    return parts


def existing_trajectory_ids(
    path: str | Path,
    *,
    run: str | None = None,
) -> set[str]:
    """Return the set of ``trajectory_id`` values present at ``path``.

    Polymorphic on path type (single-file or partitioned-directory).
    Returns an empty set if the path does not exist (the common case
    for a fresh run) or does not carry a ``trajectory_id`` column.

    Run scoping (directory paths only):

    - ``run=None`` (default) or ``run="latest"``: scan only the
      most-recent run under ``path``. Returns an empty set when no
      runs exist (so the simulator driver's "skip what's already
      done" check is a no-op on first invocation).
    - ``run="all"``: scan every run.
    - ``run="<epoch-seconds>"``: scan a specific run id.
    - Bare legacy layouts (no ``run=*`` partitions) are treated as
      a synthetic ``"legacy"`` run.

    Used by the simulator driver to filter the planned batch down to
    only the trajectories that have not yet been written for the
    target run.
    """
    p = Path(path)
    if not p.exists():
        return set()
    try:
        import pyarrow.dataset as pads
        import pyarrow.parquet as pq
    except ImportError:
        logger.debug("  |-- existing_trajectory_ids: pyarrow unavailable; treating output as empty")
        return set()

    if p.is_file():
        scan_paths = [p]
    elif run == "all":
        scan_paths = [run_subroot(p, r) for r in list_runs(p)]
        if not scan_paths:
            return set()
    else:
        resolved = resolve_run(p, run)
        if resolved is None:
            return set()
        sub = run_subroot(p, resolved)
        if not sub.exists():
            return set()
        scan_paths = [sub]

    ids: set[str] = set()
    for sp in scan_paths:
        try:
            if sp.is_file():
                table = pq.read_table(sp, columns=["trajectory_id"])
            else:
                dataset = pads.dataset(
                    str(sp),
                    format="parquet",
                    partitioning="hive",
                )
                if "trajectory_id" not in dataset.schema.names:
                    continue
                table = dataset.to_table(columns=["trajectory_id"])
        except (KeyError, ValueError) as e:
            logger.debug(f"  |-- existing_trajectory_ids: column missing in {sp} ({e}); treating as empty")
            continue
        column = table.column("trajectory_id")
        for v in column:
            if v.is_valid:
                py = v.as_py()
                if py is not None and str(py).strip():
                    ids.add(py)
    return ids


def materialize_to_temp_file(path: str | Path) -> Path:
    """Read ``path`` (single-file or partitioned) and write to a temp parquet.

    Use this when handing a trajectory dataset off to a third-party
    consumer that expects a single ``.parquet`` file (e.g., DataDesigner's
    ``LocalFileSeedSource``, which does not understand partitioned
    datasets).

    Returns the path to a NamedTemporaryFile. The caller is responsible
    for deleting it when done. Uses ``delete=False`` and a stable suffix
    so the file survives until the caller decides to clean up.
    """
    import pyarrow.parquet as pq

    df = read_partitioned_dataset(path)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".parquet",
        delete=False,
        prefix="usersim_materialize_",
    )
    tmp.close()
    out = Path(tmp.name)
    import pyarrow as pa

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out)
    return out


def is_partitioned_directory(path: str | Path) -> bool:
    """Return True iff ``path`` exists and is a directory (not a file).

    Convenience predicate for callers that want to decide between
    single-file and partitioned-dataset code paths without having to
    import ``pathlib`` themselves.
    """
    p = Path(path)
    return p.exists() and p.is_dir()


def _pyarrow_type_mapper():
    """Return the type-mapper for pandas ``dtype_backend="pyarrow"`` semantics.

    Imported lazily so this module does not require pandas at import
    time (the writer can run without pandas — pyarrow's
    ``Table.from_pandas`` is the only pandas dependency).
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover — pandas always present in our env
        return None
    return pd.ArrowDtype


def write_locale_partition(
    df,
    root: str | Path,
    *,
    locale: str,
    run_id: str,
    partition_cols: Sequence[str] = DEFAULT_PARTITION_COLS,
    writer_id: str | None = None,
) -> Path:
    """Atomically write one locale's partition into a specific run.

    Filters ``df`` to ``locale`` first (defensive — the caller may
    pass a pre-filtered frame, but mixing locales would silently
    cross-pollinate partitions). Adds a ``writer_id`` token to the
    output basename so concurrent writers can append into the same
    ``(locale, probe_family)`` partition without filename collisions.

    The ``run_id`` argument scopes the write to a specific run
    partition under ``root`` — the file lands at
    ``<root>/run=<run_id>/locale=<locale>/probe_family=<f>/part-*.parquet``.
    Callers obtain the ``run_id`` from :func:`new_run_id` (fresh
    invocation) or :func:`latest_run_id` / :func:`resolve_run`
    (resuming into an existing run).

    Used by the simulator's per-locale loop: after each locale's
    batch of trajectories returns from the LLM, this writes that
    locale's partition immediately. A crash in the next locale's
    batch leaves the prior locales' partitions on disk and resumable
    (within the same run id).
    """
    if "locale" not in df.columns:
        raise ValueError("write_locale_partition: dataframe is missing 'locale' column")
    locale_mask = df["locale"].astype(str) == str(locale)
    locale_df = df[locale_mask]
    if locale_df.empty:
        return Path(root).resolve()

    import os

    if writer_id is None:
        writer_id = f"pid{os.getpid()}"
    basename_template = f"part-{{i}}-{locale}-{writer_id}.parquet"

    subroot = run_subroot(root, run_id)
    return write_partitioned_dataset(
        locale_df,
        subroot,
        partition_cols=partition_cols,
        basename_template=basename_template,
    )


def list_partition_keys(
    root: str | Path,
    *,
    partition_col: str = "locale",
) -> list[str]:
    """List the distinct values of ``partition_col`` present in ``root``.

    Returns an empty list if ``root`` does not exist. Useful for the
    smoke check ("did the simulator write partitions for the locales
    I expected?") and for resume logic ("which locales' partitions
    are already on disk?").
    """
    p = Path(root)
    if not p.exists() or not p.is_dir():
        return []
    try:
        import pyarrow.dataset as pads
    except ImportError:
        return []
    dataset = pads.dataset(str(p), format="parquet", partitioning="hive")
    column = dataset.to_table(columns=[partition_col]).column(partition_col)
    return sorted({v.as_py() for v in column if v.is_valid and v.as_py() is not None})
