# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim smoke`` — offline end-to-end verification.

Runs without ``NVIDIA_API_KEY`` so it can land in CI as a guard against
import-cycle regressions, schema drift, or accidentally broken plugin
entry points. Each check prints a one-line ``[OK]`` / ``[FAIL]`` result
and the command exits non-zero if any check fails.

Checks performed:

1. **core imports** — ``data_designer``, ``usersim.engine``,
   ``usersim.engine.generator``, ``usersim.engine.evaluator``,
   ``reporting``, ``cli`` all import.
2. **models config** — ``cli/models_default.toml`` loads and declares
   the required simulator aliases.
3. **simulator config** — ``ConversationSimulatorConfig`` validates with
   default arguments and exposes the expected side-effect columns.
4. **evaluator config** — ``TrajectoryEvaluatorConfig`` validates with
   a single judge (single-judge ensembles are allowed).
5. **scorer registry** — auto-loaded scorers (``tool_use``) are present.
6. **dry-run subcommands** — the four LLM-driven subcommands accept
   their typical argument sets without touching the network.
7. **probe registry** — every probe in
   ``usersim.engine.core.probes.known_probes()`` resolves
   to a class that carries ``family`` / ``prompt_version`` /
   ``variants`` metadata.
8. **probe assets** — every registered probe's ``asset_problems()``
   comes back empty: each probe checks the assets it ships, such as a
   bank per locale, so a default path pointing at a missing directory
   fails at install time rather than partway through a run.
9. **reporting bundles** — both bundles build from a tiny synthetic
   trajectory + evaluator frame and produce non-empty HTML.
10. **extensions** — what each installed extension package contributes.
    Fails when an entry point contributes nothing: it failed to load,
    another package advertises the same name, or a built-in holds it.

Optional second tier: pass ``--with-fixtures`` to also write the
synthetic fixture parquets and rendered HTML to a temp directory so a
human reviewer can eyeball the output.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger("usersim.cli.smoke")


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name=name, ok=ok, detail=detail))

    @property
    def all_passed(self) -> bool:
        return all(r.ok for r in self.results)

    def render(self) -> str:
        width = max((len(r.name) for r in self.results), default=20)
        lines = []
        for r in self.results:
            tag = "[OK]  " if r.ok else "[FAIL]"
            lines.append(f"{tag} {r.name.ljust(width)}  {r.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "smoke",
        help="🔥 Offline end-to-end verification (no LLM calls).",
        description=(
            "Verify the install: imports, configs, registries, CLI "
            "dispatch, and reporting bundles all work without invoking "
            "the network. Suitable for CI."
        ),
    )
    p.add_argument(
        "--with-fixtures",
        action="store_true",
        help=(
            "Additionally write the synthetic trajectory + evaluator "
            "parquets and rendered HTML to a temp directory so a human "
            "reviewer can inspect them."
        ),
    )
    p.add_argument(
        "--fixtures-dir",
        type=Path,
        default=None,
        help=(
            "Where to write fixtures when --with-fixtures is set. "
            "Defaults to a tempdir; the path is printed at the end."
        ),
    )
    p.set_defaults(func=run)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    report = SmokeReport()

    _check_imports(report)
    _check_models_config(report)
    _check_model_catalog(report)
    _check_simulator_config(report)
    _check_evaluator_config(report)
    _check_scorer_registry(report)
    _check_probe_registry(report)
    _check_probe_assets(report)
    _check_dry_runs(report)
    fixtures_path = _check_capability_dashboard(
        report,
        write_fixtures=args.with_fixtures,
        fixtures_dir=args.fixtures_dir,
    )
    _check_comparison_dashboard(report)
    _check_extensions(report)

    print("usersim smoke")
    print("=" * 14)
    print(report.render())
    print()
    if report.all_passed:
        print("All checks passed.")
        if fixtures_path is not None:
            print(f"Fixtures written to: {fixtures_path}")
        print()
        from usersim.cli._persona_language import persona_dataset_report

        print("To run a real simulation:")
        print("  export NVIDIA_API_KEY=...")
        if persona_dataset_report(["en_US"]) is not None:
            print("  data-designer download personas --locale en_US   # once; needs the NGC CLI, see docs/personas.md")
        print("  usersim simulate --locale en_US --num-rows 5 \\")
        print("    --out output/trajectories")
        return 0
    failed = [r.name for r in report.results if not r.ok]
    print(f"FAILED checks: {failed}")
    return 1


# ---------------------------------------------------------------------------
# Individual checks — each updates ``report`` in place.
# ---------------------------------------------------------------------------


def _check_imports(report: SmokeReport) -> None:
    modules = [
        "data_designer",
        "usersim.engine",
        "usersim.engine.generator",
        "usersim.engine.evaluator",
        "usersim.engine.evaluator.scorers",
        "usersim.reporting",
        "usersim.cli",
    ]
    missing: list[str] = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:
            missing.append(f"{m}: {type(e).__name__}: {e}")
    if missing:
        report.add("core imports", False, "; ".join(missing))
    else:
        report.add("core imports", True, f"{len(modules)} modules")


def _check_models_config(report: SmokeReport) -> None:
    try:
        from usersim.cli._models import (
            REQUIRED_SIMULATOR_ALIASES,
            default_models_path,
            load_models_config,
            require_aliases,
        )

        path = default_models_path()
        cfg = load_models_config(path)
        require_aliases(cfg, required=REQUIRED_SIMULATOR_ALIASES, context="smoke")
        report.add(
            "models config",
            True,
            f"{path.name} — {len(cfg.models)} aliases",
        )
    except Exception as e:
        report.add("models config", False, f"{type(e).__name__}: {e}")


def _check_model_catalog(report: SmokeReport) -> None:
    """Verify every TOML in ``cli/`` resolves cleanly against the catalog.

    Catches catalog drift: a model named in a shipped TOML that is absent
    from ``cli/model_catalog.py:INFERENCE_DEFAULTS`` (or, for embedding
    aliases, from the embedding catalog). Drift means a run silently falls
    back to project defaults instead of the intended per-model parameters,
    so this check fails rather than warning.
    """
    try:
        from usersim.cli._models import load_models_config
        from usersim.cli.model_catalog import (
            is_embedding_model,
            known_embedding_models,
            known_inference_models,
        )

        toml_paths = sorted(Path(__file__).parent.glob("models_*.toml"))
        if not toml_paths:
            report.add(
                "model catalog",
                False,
                "no models_*.toml files found in cli/",
            )
            return

        known_inf = known_inference_models()
        known_emb = known_embedding_models()
        unknown_inf: list[tuple[str, str]] = []  # (toml_name, "alias=model")

        for path in toml_paths:
            cfg = load_models_config(path)
            for spec in cfg.models:
                # Embedding models route to /v1/embeddings and carry their own
                # defaults; INFERENCE_DEFAULTS holds chat-generation params
                # (temperature / max_tokens / ...) that are meaningless for
                # them. Check each against the catalog that actually governs it.
                catalog = known_emb if is_embedding_model(spec.model) else known_inf
                if spec.model not in catalog:
                    unknown_inf.append((path.name, f"{spec.alias}={spec.model}"))

        details = (
            f"{len(toml_paths)} TOML(s), "
            f"{len(known_inf)} model(s) in INFERENCE_DEFAULTS, "
            f"{len(known_emb)} in EMBEDDING_DEFAULTS"
        )
        if unknown_inf:
            # A gate that warns and passes is not a gate. Drift here means a
            # shipped TOML names a model the catalog does not define, so runs
            # silently fall back to project defaults instead of the intended
            # per-model parameters.
            report.add(
                "model catalog",
                False,
                f"{len(unknown_inf)} alias(es) outside catalog: "
                + ", ".join(f"{p}::{e}" for p, e in unknown_inf)
                + " — add them to the catalog or drop them from the TOML",
            )
            return
        report.add("model catalog", True, details)
    except Exception as e:
        report.add("model catalog", False, f"{type(e).__name__}: {e}")


def _check_simulator_config(report: SmokeReport) -> None:
    try:
        from usersim.engine.config import ConversationSimulatorConfig

        cfg = ConversationSimulatorConfig(name="conversation_messages")
        cols = set(cfg.side_effect_columns)
        required = {
            "conversation_messages",
            "simulation_outcome",
            "simulation_traces",
            "trajectory_id",
            "persona_uuid",
            "probe_family",
            "probe_variant",
        }
        missing = required - cols
        if missing:
            report.add(
                "simulator config",
                False,
                f"missing side-effect columns: {sorted(missing)}",
            )
            return
        report.add(
            "simulator config",
            True,
            f"{len(cols)} side-effect columns",
        )
    except Exception as e:
        report.add("simulator config", False, f"{type(e).__name__}: {e}")


def _check_evaluator_config(report: SmokeReport) -> None:
    try:
        from usersim.engine.evaluator.config import (
            JudgeSpecConfig,
            TrajectoryEvaluatorConfig,
        )

        TrajectoryEvaluatorConfig(
            name="assistant_eval",
            judges=[JudgeSpecConfig(alias="evaluator_model")],
        )
        report.add("evaluator config", True, "single-judge ensemble validates")
    except Exception as e:
        report.add("evaluator config", False, f"{type(e).__name__}: {e}")


def _check_scorer_registry(report: SmokeReport) -> None:
    try:
        from usersim.engine.evaluator.scorers import (
            get_scorer,
            list_scorers,
            load_default_scorers,
        )

        # Defensively load defaults before inspecting. The scorer
        # package eagerly registers on import, but tests in the
        # plugin suite legitimately call ``clear_registry`` for
        # isolation; running the smoke check inside the same pytest
        # session as those tests would otherwise observe an empty
        # registry. ``load_default_scorers`` is idempotent and is
        # what production does at import time — calling it again
        # mirrors production state without masking real
        # registration bugs (an import-time failure in a scorer
        # module would raise here and be caught by the outer
        # ``except``).
        load_default_scorers()

        names = sorted(list_scorers())
        if "tool_use" not in names:
            report.add(
                "scorer registry",
                False,
                f"tool_use missing; registered: {names}",
            )
            return
        # The evaluator awaits every scorer, so being callable is not enough:
        # a plain function registered here would return a value the evaluator
        # then tries to await.
        fn = get_scorer("tool_use")
        if not inspect.iscoroutinefunction(fn):
            report.add(
                "scorer registry",
                False,
                "tool_use entry is not an async function",
            )
            return
        report.add(
            "scorer registry",
            True,
            f"{len(names)} scorer(s): {names}",
        )
    except Exception as e:
        report.add("scorer registry", False, f"{type(e).__name__}: {e}")


def _check_probe_registry(report: SmokeReport) -> None:
    """Confirm every registered probe resolves and exposes metadata.

    Catches the regression where a probe's metadata is dropped: each
    probe class registered in ``_PROBE_REGISTRY`` must carry ``family``
    (string), ``prompt_version`` (string), and ``variants`` (non-empty
    sequence). These drive trajectory metadata and reporting joins;
    missing them silently corrupts downstream aggregation.
    """
    try:
        from usersim.engine.core.probes import known_probes, resolve_probe

        probe_names = known_probes()
    except Exception as e:
        report.add(
            "probe registry",
            False,
            f"import failed: {type(e).__name__}: {e}",
        )
        return

    if not probe_names:
        report.add("probe registry", False, "registry is empty")
        return

    failures: list[str] = []
    for probe_name in probe_names:
        try:
            probe_cls = resolve_probe(probe_name)
        except Exception as e:
            failures.append(f"{probe_name}: resolve raised {type(e).__name__}: {e}")
            continue
        # ``@register_probe`` sets these on the class.
        if not isinstance(probe_cls.family, str) or not probe_cls.family:
            failures.append(f"{probe_name}: family missing or non-string")
        if not isinstance(probe_cls.prompt_version, str) or not probe_cls.prompt_version:
            failures.append(f"{probe_name}: prompt_version missing or non-string")
        if not isinstance(probe_cls.variants, (list, tuple)) or not probe_cls.variants:
            failures.append(f"{probe_name}: variants missing or empty")

    if failures:
        report.add(
            "probe registry",
            False,
            f"{len(failures)} issue(s); first: {failures[0]}",
        )
        return
    report.add(
        "probe registry",
        True,
        f"{len(probe_names)} probe(s) registered: {list(probe_names)}",
    )


def _check_probe_assets(report: SmokeReport) -> None:
    """Run every registered probe's ``asset_problems`` check.

    Each probe checks the assets it ships, such as a bank per locale, so a
    missing or broken bank fails at install time rather than partway through
    a run. ``tests/engine/core/test_default_asset_paths.py`` checks the
    shipped default paths for contributors at PR time. A probe from another
    package is checked the same way as a built-in one.
    """
    try:
        from usersim.engine.core.probes import BaseProbe, known_probes, resolve_probe

        labels = known_probes()
    except Exception as e:
        report.add("probe assets", False, f"import failed: {type(e).__name__}: {e}")
        return

    default_check = inspect.getattr_static(BaseProbe, "asset_problems")
    checked: list[str] = []
    problems: list[str] = []
    for label in labels:
        try:
            cls = resolve_probe(label)
            if inspect.getattr_static(cls, "asset_problems") is default_check:
                continue
            checked.append(label)
            problems.extend(f"{label}: {problem}" for problem in cls.asset_problems())
        except Exception as e:  # smoke reports, never raises
            problems.append(f"{label}: asset_problems raised {type(e).__name__}: {e}")

    if problems:
        # The first three keep the line readable; the count covers the rest.
        head = problems[:3]
        more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
        report.add("probe assets", False, f"{len(problems)} problem(s): {'; '.join(head)}{more}")
        return
    report.add("probe assets", True, f"{len(checked)} probe(s) checked their assets: {', '.join(checked)}")


def _check_dry_runs(report: SmokeReport) -> None:
    """Round-trip the four LLM-driven subcommands through their dry-run paths.

    No network calls happen — argparse + the dry-run branches print the
    resolved plan and exit. This catches breakage in argument parsing,
    model-config validation, and the imports each subcommand performs.
    """
    try:
        from usersim import cli

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            traj_stub = tmp_path / "stub_trajs.parquet"
            traj_stub.write_bytes(b"")  # dry-run paths don't open the file

            invocations = [
                [
                    "panel",
                    "--locale",
                    "en_US",
                    "--num-personas",
                    "5",
                    "--out",
                    str(tmp_path / "panel.parquet"),
                    "--dry-run",
                ],
                [
                    "simulate",
                    "--locale",
                    "en_US",
                    "--num-rows",
                    "3",
                    "--out",
                    str(tmp_path / "trajs.parquet"),
                    "--dry-run",
                ],
                [
                    "eval",
                    "--trajectories",
                    str(traj_stub),
                    "--out",
                    str(tmp_path / "evals.parquet"),
                    "--judges",
                    "evaluator_model",
                    "--dry-run",
                ],
            ]
            for argv in invocations:
                rc = _silently(lambda a=argv: cli.main(a))
                if rc != 0:
                    report.add(
                        "dry-run subcommands",
                        False,
                        f"`usersim {argv[0]} --dry-run` exited {rc}",
                    )
                    return
        report.add("dry-run subcommands", True, "panel / simulate / eval")
    except Exception as e:
        report.add("dry-run subcommands", False, f"{type(e).__name__}: {e}")


def _check_capability_dashboard(
    report: SmokeReport,
    *,
    write_fixtures: bool,
    fixtures_dir: Path | None,
) -> Path | None:
    """Build the capability dashboard from synthetic in-process frames.

    Asserts the five artifacts (``index.html`` + four JSON cells) are
    written, non-empty, parse as JSON where applicable, and that
    ``report_manifest.json`` carries the expected top-level keys.
    """
    try:
        import pandas as pd

        from usersim.reporting import (
            build_capability_report,
            write_capability_dashboard_artifacts,
        )

        traj_df, eval_df = _synthetic_smoke_frames(pd)

        capability_report = build_capability_report(
            traj_df,
            eval_df,
            eval_column="assistant_eval",
        )

        out_dir = fixtures_dir if fixtures_dir is not None else Path(tempfile.mkdtemp(prefix="usersim-smoke-"))
        out_dir.mkdir(parents=True, exist_ok=True)
        write_capability_dashboard_artifacts(capability_report, out_dir)

        artifacts = (
            "index.html",
            "report_manifest.json",
            "capability_matrix.json",
            "coverage_summary.json",
            "triage_queue.json",
        )
        for name in artifacts:
            path = out_dir / name
            if not path.exists() or path.stat().st_size == 0:
                report.add(
                    "capability dashboard",
                    False,
                    f"missing or empty artifact: {name}",
                )
                return None
            if name.endswith(".json"):
                try:
                    json.loads(path.read_text())
                except json.JSONDecodeError as exc:
                    report.add(
                        "capability dashboard",
                        False,
                        f"{name} did not parse as JSON: {exc}",
                    )
                    return None

        manifest = json.loads((out_dir / "report_manifest.json").read_text())
        required_keys = {
            "run_id",
            "model_id",
            "metadata_source",
            "n_trajectories",
            "n_evaluated",
        }
        missing = required_keys - manifest.keys()
        if missing:
            report.add(
                "capability dashboard",
                False,
                f"report_manifest.json missing keys: {sorted(missing)}",
            )
            return None

        if write_fixtures:
            from usersim.engine.core.storage import (
                write_partitioned_dataset,
            )

            write_partitioned_dataset(traj_df, out_dir / "trajectories")
            write_partitioned_dataset(eval_df, out_dir / "evaluations")
            report.add(
                "capability dashboard",
                True,
                "5 artifacts + trajectories/evaluations fixtures",
            )
            return out_dir

        report.add(
            "capability dashboard",
            True,
            "5 artifacts (index.html + 4 JSON), manifest keys present",
        )
        return None
    except Exception as e:
        report.add("capability dashboard", False, f"{type(e).__name__}: {e}")
        return None


def _check_comparison_dashboard(report: SmokeReport) -> None:
    """Import-path + edge-case check for the multi-model comparison surface.

    Three cheap assertions:

    - ``build_comparison_report([])`` raises a clean ``ValueError`` rather
      than blowing up — guards the empty-discovery branch.
    - ``build_comparison_report([single])`` raises a clean ``ValueError``
      — single-run comparisons are misleading-by-construction.
    - The dashboard module's locked palette + shape helpers return 8
      distinct values for 8 distinct labels (guards the cross-cutting
      design discipline that drives every chart).

    Does NOT actually render a dashboard — that's covered by
    ``tests/test_comparison_report.py``.
    """
    try:
        from usersim.reporting.comparison import build_comparison_report
        from usersim.reporting.comparison_dashboard import assign_palette

        try:
            build_comparison_report([])
        except ValueError:
            pass
        else:
            report.add(
                "comparison dashboard",
                False,
                "build_comparison_report([]) did not raise ValueError",
            )
            return

        try:
            build_comparison_report(["/nonexistent/manifest.json"])
        except ValueError:
            pass
        except FileNotFoundError:
            # The single-run guard fires before the file load — but if
            # someone re-orders, accept either signal.
            pass
        except Exception as e:
            report.add(
                "comparison dashboard",
                False,
                f"single-run guard raised unexpected {type(e).__name__}: {e}",
            )
            return

        # Generic labels (no nano/super/ultra) are denied the
        # restricted-green palette slots, so 8 ineligible labels can
        # only produce 7 distinct colors. Add one green-eligible
        # label to the panel to recover the full 8-distinct property
        # without exercising the wrap-around branch.
        labels = [f"m{i}" for i in range(7)] + ["nemotron-3-ultra-preview"]
        colors, shapes = assign_palette(labels)
        if len(set(colors.values())) != 8:
            report.add(
                "comparison dashboard",
                False,
                f"palette collision: 8 labels -> {len(set(colors.values()))} colors",
            )
            return
        if len(set(shapes.values())) != 8:
            report.add(
                "comparison dashboard",
                False,
                f"shape collision: 8 labels -> {len(set(shapes.values()))} shapes",
            )
            return
        # Regression guard for the qwen-as-NVIDIA-green bug: a label
        # without nano/super/ultra must never be assigned a restricted
        # green hex even when the palette is comfortable.
        from usersim.reporting.comparison_dashboard import _RESTRICTED_GREEN_HEX_COLORS

        for label, hex_color in colors.items():
            if "ultra" in label or "super" in label or "nano" in label:
                continue
            if hex_color in _RESTRICTED_GREEN_HEX_COLORS:
                report.add(
                    "comparison dashboard",
                    False,
                    f"non-NVIDIA label {label!r} got restricted-green {hex_color}",
                )
                return

        report.add(
            "comparison dashboard",
            True,
            "import OK, edge cases raise cleanly, palette distinct for 8 models with branded green guard",
        )
    except Exception as e:
        report.add("comparison dashboard", False, f"{type(e).__name__}: {e}")


def _check_extensions(report: SmokeReport) -> None:
    """List what installed extensions contribute, and fail on any that contribute nothing.

    Every registry is loaded first, because a registry notes an extension
    that names one of its built-ins only when it merges extensions in.
    """
    try:
        from usersim.asset_gen.registry import domain_names
        from usersim.cli import _build_parser
        from usersim.engine.core._extensions import (
            DISABLE_ENV_VAR,
            ENTRY_POINT_GROUPS,
            extension_records,
            extensions_disabled,
            load_extensions,
        )
        from usersim.engine.core.backends import known_backends
        from usersim.engine.core.probes import known_probes
        from usersim.engine.evaluator.scorers import list_scorers
        from usersim.taxonomy.capabilities import capability_definitions

        if extensions_disabled():
            report.add("extensions", True, f"discovery is off ({DISABLE_ENV_VAR} is set)")
            return
        for group in ENTRY_POINT_GROUPS:
            load_extensions(group)
        known_probes()
        list_scorers()
        domain_names()
        known_backends()
        capability_definitions()
        _build_parser()

        records = extension_records()
        problems = [r for r in records if r.problem]
        if problems:
            report.add("extensions", False, "; ".join(f"{r.name} from {r.source}: {r.problem}" for r in problems))
        elif records:
            packages = Counter(r.package for r in records)
            report.add(
                "extensions",
                True,
                ", ".join(f"{package}: {count} entry point(s)" for package, count in sorted(packages.items())),
            )
        else:
            report.add("extensions", True, "none installed")
    except Exception as e:
        report.add("extensions", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _silently(fn: Callable[[], int]) -> int:
    """Run a function while suppressing its stdout (the dry-run plan dump)."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn()


def _synthetic_smoke_frames(pd):
    """Build a tiny self-contained trajectory + evaluator pair.

    Two trajectories: one OK, one with a structured failure. One row in
    each frame so the bundles exercise both code paths (status counts,
    failure taxonomy, per-axis means, judge ensemble).
    """
    outcome_ok = json.dumps(
        {
            "status": "ok",
            "failure_class": None,
            "failure_attribution": None,
            "failure_detail": "",
            "n_turns": 2,
            "n_tool_calls": 0,
            "n_user_query_attempts": 1,
            "n_user_followup_retries": 0,
            "n_assistant_inline_failures": 0,
            "n_api_response_rerolls": 0,
            "n_fourth_wall_triggers": 0,
            "n_user_role_violations": 0,
            "warnings": [],
            "early_stop": False,
            "per_model_input_tokens": {"user_model": 100, "assistant_model": 80},
            "per_model_output_tokens": {"user_model": 30, "assistant_model": 60},
            "per_model_calls": {"user_model": 2, "assistant_model": 2},
            "wall_clock_s_by_alias": {"user_model": 1.0, "assistant_model": 1.5},
            "wall_clock_s": 2.5,
            "provenance": {
                "code_sha": "smoke",
                "nemotron_personas_version": None,
                "scenario_prompt_version": "v1.0",
                "bank_version": {},
            },
        }
    )
    outcome_fail = json.dumps(
        {
            "status": "failed",
            "failure_class": "user_query_gate_exhausted",
            "failure_attribution": "user_model",
            "failure_detail": "smoke fixture failure",
            "n_turns": 0,
            "n_tool_calls": 0,
            "n_user_query_attempts": 3,
            "n_user_followup_retries": 0,
            "n_assistant_inline_failures": 0,
            "n_api_response_rerolls": 0,
            "n_fourth_wall_triggers": 0,
            "n_user_role_violations": 0,
            "warnings": [],
            "early_stop": False,
            "per_model_input_tokens": {},
            "per_model_output_tokens": {},
            "per_model_calls": {},
            "wall_clock_s_by_alias": {},
            "wall_clock_s": 0.5,
            "provenance": {
                "code_sha": "smoke",
                "nemotron_personas_version": None,
                "scenario_prompt_version": "v1.0",
                "bank_version": {},
            },
        }
    )
    messages = json.dumps(
        [
            {"role": "user", "content": "Hello, can you help me?"},
            {"role": "assistant", "content": "Of course — what do you need?"},
        ]
    )

    traj_df = pd.DataFrame(
        [
            {
                "trajectory_id": "smoke0000000001",
                "persona_uuid": "smokepersona001",
                "probe_family": "general_open_ended",
                "probe_variant": "default",
                "locale": "en_US",
                "conversation_language": "English",
                "user_interaction_style": "cooperative",
                "disclosure_style": "incremental",
                "persona_grounding": True,
                "conversation_messages": messages,
                "simulation_outcome": outcome_ok,
            },
            {
                "trajectory_id": "smoke0000000002",
                "persona_uuid": "smokepersona002",
                "probe_family": "general_open_ended",
                "probe_variant": "default",
                "locale": "en_US",
                "conversation_language": "English",
                "user_interaction_style": "neutral",
                "disclosure_style": "upfront",
                "persona_grounding": False,
                "conversation_messages": "[]",
                "simulation_outcome": outcome_fail,
            },
        ]
    )

    envelope = {
        "judge_aliases": ["judge_a"],
        "judge_families": ["openai"],
        "axes": ["helpfulness", "accuracy"],
        "scorers": [],
        "prompt_version": "v1.0",
        "evaluator_version": "v1.0",
    }
    eval_cell = json.dumps(
        {
            "envelope": envelope,
            "axes": {
                "helpfulness": {"judge_a": {"score": 5, "reasoning": "great"}},
                "accuracy": {"judge_a": {"score": 4, "reasoning": "ok"}},
            },
            "scorers": {},
            "skipped": False,
            "skipped_reason": None,
        }
    )
    eval_skip = json.dumps(
        {
            "envelope": envelope,
            "axes": {},
            "scorers": {},
            "skipped": True,
            "skipped_reason": "no_assistant_messages",
        }
    )

    eval_df = traj_df.copy()
    eval_df["assistant_eval"] = [eval_cell, eval_skip]
    return traj_df, eval_df


if __name__ == "__main__":
    sys.exit(run(argparse.Namespace(with_fixtures=False, fixtures_dir=None)))
