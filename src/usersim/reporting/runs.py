# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-disk run discovery + pretty-print.

Walks the three run roots that the notebook + CLI care about
(``output/trajectories/``, ``output/evaluations/``, ``output/report/``)
and renders a ``rich.Table`` overview of every trajectory run with its
eval / report status and the actor models captured in the run manifest.

Used by ``notebooks/02_evaluate_simulation.ipynb``'s "Available Runs"
cell. Lives here rather than in the notebook so the cell stays a single
import + call, and so future CLI / dashboard surfaces can re-use the
same discovery shape.

``rich`` is imported lazily inside the function so the rest of the
``reporting`` package doesn't gain a hard dependency on it (only this
one rendering helper needs it).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Lead with assistant_model in the manifest-derived "models" block;
# everything else follows in the order users care about (the simulator
# substrate first, then the post-hoc judge / summary aliases).
_MODEL_ALIASES_ORDERED = (
    "assistant_model",
    "user_model",
    "api_response_model",
    "judge_model",
    "summary_model",
)


def _display_path(p: Path | str) -> str:
    """Render ``p`` as a short, repo-rooted display string.

    Walks up from ``p`` looking for ``pyproject.toml`` / ``.git`` markers;
    when found, returns ``<repo-name>/<rest>`` (e.g.
    ``usersim/output/trajectories``). Otherwise falls back to the
    last two path components so we never blast full ``/Users/<name>/...``
    paths into the rendered title.
    """
    p = Path(p).resolve()
    for ancestor in (p, *p.parents):
        if (ancestor / "pyproject.toml").exists() or (ancestor / ".git").exists():
            if ancestor == p:
                return ancestor.name
            return f"{ancestor.name}/{p.relative_to(ancestor)}"
    parts = p.parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else str(p)


def _format_started(run_id: str) -> str:
    if run_id == "legacy":
        return "(legacy)"
    try:
        return datetime.fromtimestamp(int(run_id), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return "?"


def _format_models_block(run_id: str, trajectory_root: Path) -> str:
    """Manifest → one-cell layout: assistant first (bold green), blank line,
    then user / api_response / judge / summary (dim). Alias labels are
    right-padded to the longest one's width so the colons + model names
    line up vertically -- easier to scan when the labels are different
    lengths (``user`` vs ``api_response`` etc.).
    """
    from usersim.engine.core.manifest import load_run_manifest

    if run_id == "legacy":
        return "—"
    manifest = load_run_manifest(run_id, trajectory_root)
    if manifest is None:
        return "—"

    labels = [alias.removesuffix("_model") for alias in _MODEL_ALIASES_ORDERED]
    width = max(len(label) for label in labels)

    lines: list[str] = []
    assistant = manifest.models.get("assistant_model")
    if assistant is not None:
        label = "assistant".ljust(width)
        lines.append(f"[green][bold]{label}[/bold]: {assistant.model}[/green]")
    for alias in _MODEL_ALIASES_ORDERED[1:]:
        ident = manifest.models.get(alias)
        if ident is None:
            continue
        label = alias.removesuffix("_model").ljust(width)
        lines.append(f"[dim]{label}: {ident.model}[/dim]")
    return "\n".join(lines) if lines else "—"


def print_available_runs(
    trajectory_root: Path | str,
    eval_root: Path | str,
    report_root: Path | str,
    *,
    console: Optional[object] = None,
) -> None:
    """Render a ``rich.Table`` of trajectory runs with their eval/report
    status + actor models (from each run's manifest, when present).

    Columns: ``run_id`` / ``started (UTC)`` / ``# traj`` / ``# locales`` /
    ``models`` / ``eval`` / ``report``. The models cell shows the
    assistant model first, then a blank line, then user / api / judge /
    summary aliases.

    Pass ``console`` (a ``rich.Console`` instance) to redirect output
    (tests use ``Console(file=stringio)``); defaults to a stdout console.
    """
    from rich.console import Console
    from rich.table import Table

    from usersim.engine.core.storage import (
        existing_trajectory_ids,
        list_runs,
        run_subroot,
    )

    trajectory_root = Path(trajectory_root)
    eval_root = Path(eval_root)
    report_root = Path(report_root)
    # Force a generous render width so Jupyter HTML output keeps the full
    # run_id / timestamp / model-name strings visible (the cell scrolls
    # horizontally rather than truncating with `…`). Tests pass an
    # explicit `console=` so this default doesn't affect them.
    console = console or Console(width=200)

    runs = list_runs(trajectory_root)
    display_root = _display_path(trajectory_root)
    if not runs:
        console.print(f"[yellow]No trajectory runs under {display_root}. Run `01_simulate.ipynb` first.[/yellow]")
        return

    # `min_width` keeps narrow numeric / status columns from collapsing
    # their headers to `trajecto…` / `loc…` / `e…` / `re…` when the body
    # data is shorter than the header label. `show_lines=True` draws a
    # horizontal rule between rows so consecutive runs don't visually
    # blur together.
    table = Table(
        title=f"Trajectory runs under {display_root}",
        title_style="bold cyan",
        header_style="bold",
        show_lines=True,
    )
    # `min_width` keeps each column wide enough for its longest expected
    # body value (epoch id = 10 chars, ISO-minute timestamp = 16 chars,
    # the longest shipped model name string with its padded label
    # prefix ~58 chars) so rich never truncates with `…` when the
    # cell's horizontal budget is tight.
    table.add_column("run_id", style="cyan", no_wrap=True, min_width=10)
    table.add_column("started (UTC)", style="dim", no_wrap=True, min_width=16)
    table.add_column("trajectories", justify="right", no_wrap=True, min_width=12)
    table.add_column("locales", justify="right", no_wrap=True, min_width=7)
    table.add_column("models", no_wrap=True, min_width=58)
    table.add_column("eval", justify="center", no_wrap=True, min_width=4)
    table.add_column("report", justify="center", no_wrap=True, min_width=6)

    for rid in sorted(runs, reverse=True):
        n_traj = len(existing_trajectory_ids(trajectory_root, run=rid))
        n_locales = sum(1 for p in run_subroot(trajectory_root, rid).glob("locale=*") if p.is_dir())
        eval_ok = run_subroot(eval_root, rid).exists()
        report_ok = (run_subroot(report_root, rid) / "index.html").exists()
        table.add_row(
            rid,
            _format_started(rid),
            str(n_traj),
            str(n_locales),
            _format_models_block(rid, trajectory_root),
            "[green]\u2713[/green]" if eval_ok else "[dim]\u2014[/dim]",
            "[green]\u2713[/green]" if report_ok else "[dim]\u2014[/dim]",
        )

    console.print(table)
    n_eval = sum(1 for r in runs if run_subroot(eval_root, r).exists())
    n_report = sum(1 for r in runs if (run_subroot(report_root, r) / "index.html").exists())
    console.print(
        f"[dim]{len(runs)} run(s) on disk \u00b7 {n_eval} with evals \u00b7 {n_report} with rendered report.[/dim]"
    )


def print_comparison_runs(
    paths: list[Path | str],
    *,
    console: Optional[object] = None,
) -> None:
    """Render a ``rich.Table`` of report manifests selected for comparison.

    Reads each manifest's top-level metadata (``model_id`` / ``n_trajectories``
    / ``sample_mode`` / ``eval_column``) so a reviewer can eyeball the
    apples-to-apples fit before paying the build cost. Mirrors the style
    of :func:`print_available_runs`; pass ``console`` to redirect output.

    Empty input renders a yellow hint; a single-run input prints a
    "comparison requires >=2 runs" warning below the table.
    """
    import json as _json

    from rich.console import Console
    from rich.table import Table

    paths = [Path(p) for p in paths]
    console = console or Console(width=200)

    if not paths:
        console.print(
            "[yellow]No reports selected for comparison. Build at least two "
            "via cell 14 (Capability Dashboard), or check COMPARE_RUN_IDS.[/yellow]"
        )
        return

    table = Table(
        title=f"Selected for comparison ({len(paths)} run{'s' if len(paths) != 1 else ''})",
        title_style="bold cyan",
        header_style="bold",
        show_lines=True,
    )
    table.add_column("run_id", style="cyan", no_wrap=True, min_width=10)
    table.add_column("started (UTC)", style="dim", no_wrap=True, min_width=16)
    table.add_column("assistant model", min_width=40)
    table.add_column("trajectories", justify="right", no_wrap=True, min_width=12)
    table.add_column("sample mode", justify="center", no_wrap=True, min_width=12)
    table.add_column("eval column", style="dim", no_wrap=True, min_width=14)

    for path in paths:
        try:
            manifest = _json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            run_id = path.parent.name.removeprefix("run=") if path.parent else "?"
            table.add_row(
                run_id,
                _format_started(run_id),
                f"[red]load error: {exc.__class__.__name__}[/red]",
                "—",
                "—",
                "—",
            )
            continue
        run_id = str(manifest.get("run_id") or path.parent.name.removeprefix("run="))
        table.add_row(
            run_id,
            _format_started(run_id),
            f"[green]{manifest.get('model_id') or '—'}[/green]",
            f"{manifest.get('n_trajectories', 0):,}",
            str(manifest.get("sample_mode") or "?"),
            str(manifest.get("eval_column") or "?"),
        )

    console.print(table)
    if len(paths) < 2:
        console.print(f"[yellow]\u26a0  Comparison requires \u22652 runs; found {len(paths)}.[/yellow]")
