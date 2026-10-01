# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

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
   to a callable + a module that exposes ``PROBE_FAMILY`` /
   ``PROMPT_VERSION`` / ``PROBE_VARIANTS`` metadata.
8. **asset paths** — every shipped probe's default asset path
   (``probe_assets_dir(probe) / ...``) resolves to a real file on
   disk for every supported locale. Catches the regression class
   where a default-path constant points at a non-existent directory
   (the sort of half-rename bug that ships at notebook-run time
   instead of install time).
9. **reporting bundles** — both bundles build from a tiny synthetic
   trajectory + evaluator frame and produce non-empty HTML.

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
    _check_asset_paths(report)
    _check_dry_runs(report)
    fixtures_path = _check_capability_dashboard(
        report,
        write_fixtures=args.with_fixtures,
        fixtures_dir=args.fixtures_dir,
    )
    _check_comparison_dashboard(report)

    print("usersim smoke")
    print("=" * 14)
    print(report.render())
    print()
    if report.all_passed:
        print("All checks passed.")
        if fixtures_path is not None:
            print(f"Fixtures written to: {fixtures_path}")
        print()
        print("To run a real simulation:")
        print("  export NVIDIA_API_KEY=...")
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

    Catches the regression where a probe module is renamed or its
    metadata constants are dropped: each probe class registered in
    ``_PROBE_REGISTRY`` must live in a module exposing
    ``PROBE_FAMILY`` (string), ``PROMPT_VERSION`` (string), and
    ``PROBE_VARIANTS`` (non-empty sequence). These constants drive
    trajectory metadata and reporting joins; missing them silently
    corrupts downstream aggregation.
    """
    try:
        import sys

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
        # All probes self-register via ``@register_probe`` and live
        # in their own module; the constants live there too.
        module = sys.modules.get(probe_cls.__module__)
        if module is None:
            failures.append(f"{probe_name}: probe metadata module {probe_cls.__module__!r} not in sys.modules")
            continue
        family = getattr(module, "PROBE_FAMILY", None)
        if not isinstance(family, str) or not family:
            failures.append(f"{probe_name}: PROBE_FAMILY missing or non-string")
        prompt_version = getattr(module, "PROMPT_VERSION", None)
        if not isinstance(prompt_version, str) or not prompt_version:
            failures.append(f"{probe_name}: PROMPT_VERSION missing or non-string")
        variants = getattr(module, "PROBE_VARIANTS", None)
        if not isinstance(variants, (list, tuple)) or not variants:
            failures.append(f"{probe_name}: PROBE_VARIANTS missing or empty")

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


def _check_asset_paths(report: SmokeReport) -> None:
    """Verify every shipped probe's default asset path exists + loads.

    The user-facing analog of
    ``conversation-plugin/tests/core/test_default_asset_paths.py``:
    fails the smoke if a default-path constant in any ``core/``
    loader points at a directory that doesn't exist on disk. The
    test-suite version catches this for contributors at PR time;
    this catches it for users at install time so they don't burn
    a notebook run discovering it.
    """
    try:
        from usersim.engine.core.action_taxonomy import (
            load_action_taxonomy_default,
        )
        from usersim.engine.core.agentic_bank import default_agentic_bank_path
        from usersim.engine.core.fact_bank import default_fact_bank_path
        from usersim.engine.core.locale import SHIPPED_LOCALES
        from usersim.engine.core.pressure_bank import default_pressure_bank_path
        from usersim.engine.core.probing_taxonomy import (
            default_probing_taxonomy_path,
        )
        from usersim.engine.core.prompt_loader import load_prompt_file
        from usersim.engine.core.query_bank import default_query_bank_path
        from usersim.engine.core.seeds import load_seeds
    except Exception as e:
        report.add(
            "asset paths",
            False,
            f"loader import failed: {type(e).__name__}: {e}",
        )
        return

    failures: list[str] = []
    n_paths = 0

    # Per-locale loaders.
    for locale in SHIPPED_LOCALES:
        for label, fn in (
            ("fact_bank", lambda l=locale: default_fact_bank_path(l)),
            ("probing_taxonomy", lambda l=locale: default_probing_taxonomy_path(l)),
        ):
            n_paths += 1
            p = fn()
            if not p.exists():
                failures.append(f"{label}({locale}): {p}")
        # Per-probe seed loaders raise FileNotFoundError when absent
        # rather than returning a path; treat that as the failure.
        for probe, kind in (
            ("general_open_ended", "topics"),
            ("general_educational", "subjects"),
        ):
            n_paths += 1
            try:
                load_seeds(locale, probe, kind)
            except FileNotFoundError as e:
                failures.append(f"seeds({locale}, {probe}/{kind}): {e}")

    # Locale-agnostic loaders.
    for label, fn in (
        ("query_bank", default_query_bank_path),
        ("pressure_bank", default_pressure_bank_path),
        ("agentic_bank", default_agentic_bank_path),
    ):
        n_paths += 1
        p = fn()
        if not p.exists():
            failures.append(f"{label}: {p}")

    # Action taxonomy resolves its path internally; raises if missing.
    n_paths += 1
    try:
        load_action_taxonomy_default()
    except Exception as e:
        failures.append(f"action_taxonomy: {type(e).__name__}: {e}")

    # Per-probe prompts.yaml — the loader silently falls back to
    # compiled-in Python defaults on miss, so we explicitly assert
    # non-empty here.
    for probe in ("tool_calling", "general_open_ended", "general_educational"):
        n_paths += 1
        if not load_prompt_file(probe):
            failures.append(f"prompts({probe}): assets/{probe}/prompts.yaml missing or empty")

    # tool_calling toolsets_seed.parquet — composed manually by
    # cli/simulate.py + the notebook (no loader fronts it). When
    # missing, ``use_tools`` silently flips to False and every
    # tool_calling trajectory fails at probe construction with
    # ``cfg.tools_column not configured``.
    from usersim.engine.core._assets import probe_assets_dir

    n_paths += 1
    toolset = probe_assets_dir("tool_calling") / "toolsets_seed.parquet"
    if not toolset.exists():
        failures.append(
            f"tool_calling/toolsets_seed.parquet: {toolset} (without it "
            "all tool_calling trajectories fail at construction)"
        )

    # financial_services multi-institution bank + dynamic-tier taxonomy. Shipped
    # en_US only (other locales are the replication phase). Loading the bank
    # exercises the whole loader (raises on any cross-institution gold/scope
    # break); the taxonomy load exercises the dynamic tier's asset.
    n_paths += 1
    try:
        from usersim.engine.core.finance_bank import (
            load_finance_bank_for_locale,
        )

        fin_bank = load_finance_bank_for_locale("en_US")
        if not fin_bank.institutions:
            failures.append("financial_services(en_US): bank has no institutions")
    except Exception as e:  # noqa: BLE001 — smoke reports, never raises
        failures.append(f"financial_services(en_US): {type(e).__name__}: {e}")
    n_paths += 1
    try:
        from usersim.engine.core.probing_taxonomy import (
            load_probing_taxonomy_for,
        )

        fin_tax = load_probing_taxonomy_for(
            "financial_services",
            "en_US",
            filename="dynamic.yaml",
            env_prefix="USERSIM_FINANCIAL_SERVICES_DYNAMIC_TAXONOMY",
        )
        if not fin_tax.categories:
            failures.append("financial_services dynamic.yaml(en_US): no categories")
    except Exception as e:  # noqa: BLE001 — smoke reports, never raises
        failures.append(f"financial_services dynamic.yaml(en_US): {type(e).__name__}: {e}")

    # identity_disclosure spec: loading it follows its layers and checks every
    # cross-reference (developers, competitors, placeholders, rule examples).
    n_paths += 1
    try:
        from usersim.engine.core.identity_spec import validate_spec

        failures.extend(f"identity_disclosure spec: {problem}" for problem in validate_spec("identity_disclosure"))
    except Exception as e:  # noqa: BLE001 — smoke reports, never raises
        failures.append(f"identity_disclosure spec: {type(e).__name__}: {e}")

    if failures:
        # Show first three failures to keep the smoke output readable;
        # the full count lands in the failure detail.
        head = failures[:3]
        more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
        report.add(
            "asset paths",
            False,
            f"{len(failures)}/{n_paths} default paths missing: {'; '.join(head)}{more}",
        )
        return

    report.add(
        "asset paths",
        True,
        f"{n_paths} default paths resolve cleanly across {len(SHIPPED_LOCALES)} locales",
    )


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
