# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""``usersim simulate`` — run the conversation simulator.

Drives the ``conversation-simulator`` Data Designer plugin over either
a panel parquet or an inline ``--locale`` + ``--num-rows`` sampler.

A panel fixes the *shape* of the run -- which locales, and how many rows
each -- and those are read from the parquet. The personas themselves are
re-sampled per run rather than taken from the panel, so two runs over the
same panel cover the same locales and counts with different people.
Writes a Hive-partitioned trajectory dataset rooted at ``--out``,
with one partition per ``(locale, probe_family)`` pair, that the
evaluator and reporting layers consume downstream.

Per-locale atomic writes: each locale's batch lands as its own set of
partition files immediately after that locale completes. A crash
mid-run leaves earlier locales' partitions intact.

Idempotent re-runs: if the output dataset already exists, rows whose
``trajectory_id`` is present are skipped (the simulator re-emits
identical bytes for them anyway, but skipping avoids the LLM cost).
"""

from __future__ import annotations

import argparse
import logging
from fractions import Fraction
from pathlib import Path

from usersim.cli._pipeline import resolve_toolset_seed_for_mix, verify_probe_assets

logger = logging.getLogger("usersim.cli.simulate")


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "simulate",
        help="🗣  Run the conversation simulator and write a trajectory parquet.",
        description=(
            "Run the simulator pipeline against a panel parquet (or an "
            "inline locale + sampler) and write trajectories. Resumable: "
            "rows whose trajectory_id already exists in the output "
            "parquet are skipped."
        ),
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--panel",
        type=Path,
        help=(
            "Panel parquet from `usersim panel`. Sets the run's locales "
            "and per-locale row counts; personas are re-sampled rather "
            "than read from the panel."
        ),
    )
    src.add_argument(
        "--locale",
        action="append",
        help=("Locale to sample inline (repeatable). Use with --num-rows. Mutually exclusive with --panel."),
    )

    p.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Rows per locale when sampling inline (required with --locale).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output trajectory dataset root. Becomes a Hive-partitioned "
            "directory tree: <out>/locale=X/probe_family=Y/part-N.parquet."
        ),
    )
    p.add_argument(
        "--models",
        type=Path,
        default=None,
        help="Path to models TOML. Defaults to cli/models_default.toml.",
    )
    p.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help=(
            "Path to an alternate assets/ tree (seeds + toolsets). Defaults to the banks shipped inside the package."
        ),
    )
    p.add_argument(
        "--match-persona-language",
        action="store_true",
        help=(
            "Sample only personas whose first language is the conversation "
            "language. Off by default. Note it shifts the demographic makeup "
            "of the sampled population, not just its language, so two "
            "language-matched locales differ by more than language and a "
            "difference measured between them may not be linguistic. For "
            "en_IN, omit this flag to run in English across the full India "
            "persona corpus; enabling it selects only records whose current "
            "first_language field is recorded as English."
        ),
    )
    p.add_argument(
        "--toolset-seed-path",
        type=Path,
        default=None,
        help=(
            "Toolset file for the tool_calling probe (.parquet or .jsonl), "
            "one row per toolset with columns toolset_source, "
            "toolset_category, toolset_name, num_tools and tools (JSON list "
            "of OpenAI-style function specs) -- see the Schema section of "
            "assets/tool_calling/README.md. "
            "Defaults to <assets-dir>/tool_calling/toolsets_seed.{parquet,jsonl}. "
            "Requires tool_calling in --probe-mix; a missing or unsupported "
            "explicit file is an error."
        ),
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum conversation turns per trajectory.",
    )
    p.add_argument(
        "--max-assistant-attempts",
        type=int,
        default=1,
        help=(
            "Assistant-turn resampling budget. 1 (default) keeps the "
            "per-turn assistant judge flag-only; >1 discards a "
            "judge-rejected assistant candidate and regenerates it "
            "within the budget (SDG-style runs). Probes with "
            "side-effectful after_assistant_turn are capped to 1."
        ),
    )
    p.add_argument(
        "--probe-mix",
        type=str,
        default="tool_calling=0.34,general_open_ended=0.33,general_educational=0.33",
        help=(
            "Comma-separated probe_type=weight pairs. Weights are "
            "renormalized to 1.0 if they don't sum to 1. Available "
            "probes: tool_calling, general_open_ended, general_educational, "
            "sov_ai_facts, sov_ai_dynamic, "
            "sov_ai_multilingual_parity, "
            "safety_chat_pressure, safety_agentic. "
            "financial_services is OPT-IN (asset-backed; not in the default "
            "mix) — add e.g. 'financial_services=1.0'."
        ),
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible tool sampling (per row).",
    )
    p.add_argument(
        "--finance-tier-mix",
        type=float,
        default=0.0,
        help=(
            "financial_services only: probability [0,1] a trajectory uses the "
            "DYNAMIC tier (generated open-ended, judge-scored) vs the default "
            "VERIFIABLE tier (param-locked, deterministically verified). "
            "0.0 (default) = verifiable-only."
        ),
    )
    p.add_argument(
        "--finance-retrieval-mode",
        choices=("golden", "dense"),
        default="dense",
        help=(
            "financial_services only: kb_search backend. 'dense' (default) "
            "ranks the institution's precomputed doc embeddings against the "
            "query vector (embedded via the 'embedding_model' alias) so BOTH "
            "tiers retrieve from the KB (tau-style); degrades to lexical when "
            "no embedding endpoint is reachable. 'golden' is a reasoning-"
            "isolation ABLATION that injects the verifiable task's gold "
            "documents (removes retrieval quality as a variable); the dynamic "
            "tier has no gold docs so golden degrades to lexical there. "
            "Retrieval quality is measured separately by the "
            "financial_retrieval_recall capability."
        ),
    )
    p.add_argument(
        "--no-store-reasoning",
        action="store_true",
        help=(
            "Drop the assistant's thinking trace instead of storing it on "
            "the message. Default is to store it when the model emits one "
            "(reasoning must also be enabled model-side -- see the models "
            "TOML). Traces are never replayed to a model either way; this "
            "only controls whether the trajectory carries them."
        ),
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Force re-simulation of rows whose trajectory_id is already in --out.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan and exit without invoking the LLM.",
    )
    p.add_argument(
        "--materialize-inputs",
        action="store_true",
        help=(
            "Write fully resolved UserSim episode-input rows to --out and exit "
            "without invoking any model. --out must end in .jsonl or .parquet."
        ),
    )
    # Choices come from the registry rather than a literal list, so an
    # installed backend is offered in --help without editing this file.
    from usersim.engine.core.backends import known_backends

    p.add_argument(
        "--backend",
        choices=known_backends(),
        default="local",
        help="Where the run executes (default: local, in this process).",
    )
    p.set_defaults(func=run)


def _parse_probe_mix(spec: str) -> dict:
    """Parse + validate ``--probe-mix``.

    Validates keys against the plugin's ``_PROBE_REGISTRY`` so an
    unknown probe fails at argument-parse time with a friendly
    error rather than as an opaque sampler error mid-run.
    """
    from usersim.cli._pipeline import known_probes

    out: dict[str, float] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"--probe-mix expects k=v pairs; got {pair!r} in {spec!r}")
        k, v = pair.split("=", 1)
        # ``Fraction`` takes "1/3" as well as "0.5" and "2", so an even split
        # can be written the way you would say it. Weights are normalised
        # below, so they need not sum to 1.
        try:
            out[k.strip()] = float(Fraction(v.strip()))
        except (ValueError, ZeroDivisionError):
            raise SystemExit(
                f"--probe-mix weight {v.strip()!r} for {k.strip()!r} is not a number; "
                f"use an integer (1), a decimal (0.5) or a fraction (1/3)"
            ) from None
    if not out:
        raise SystemExit("--probe-mix is empty")
    total = sum(out.values())
    if total <= 0:
        raise SystemExit(f"--probe-mix weights sum to {total}; expected > 0")

    known = set(known_probes())
    unknown = sorted(k for k in out if k not in known)
    if unknown:
        raise SystemExit(
            f"--probe-mix references unknown probe_type(s): {unknown}. Registered probes: {sorted(known)}."
        )

    return {k: v / total for k, v in out.items()}


def run(args: argparse.Namespace) -> int:
    from usersim.cli._models import (
        REQUIRED_SIMULATOR_ALIASES,
        default_models_path,
        load_models_config,
        require_aliases,
        warn_if_missing_api_key,
    )

    models_path = args.models or default_models_path()
    models = load_models_config(models_path)
    require_aliases(models, required=REQUIRED_SIMULATOR_ALIASES, context="`usersim simulate`")

    probe_mix = _parse_probe_mix(args.probe_mix)

    # Resolve --assets-dir once, here, so every downstream consumer sees a
    # real path rather than None.
    from usersim.engine.core._assets import default_assets_dir

    # Two different questions. ``pinned`` is whether the user named a
    # directory: if so, resolution stays there and a missing bank is an
    # error rather than a silent fall-through to the packaged copies.
    # ``args.assets_dir`` is the single concrete root recorded in the
    # generator config and the run manifest, which cannot hold a list.
    pinned_assets_dir = args.assets_dir
    if args.assets_dir is None:
        args.assets_dir = default_assets_dir()

    if args.locale and args.num_rows is None:
        raise SystemExit("--locale requires --num-rows")

    if args.materialize_inputs:
        return _materialize_inputs(args, models, probe_mix)

    if args.dry_run:
        # Resolved before anything is printed so a bad config fails cleanly
        # instead of interleaving with a half-written plan.
        verify_probe_assets(probe_mix, args.locale or [], pinned_assets_dir)
        seed = resolve_toolset_seed_for_mix(
            probe_mix,
            pinned_assets_dir,
            args.toolset_seed_path,
        )
        print("usersim simulate — dry run")
        if args.panel:
            print(f"  panel:         {args.panel}")
        else:
            print(f"  locales:       {args.locale}")
            print(f"  per-locale:    {args.num_rows}")
        print(f"  probe mix:     {probe_mix}")
        print(f"  max_turns:     {args.max_turns}")
        print(f"  max_assistant_attempts: {args.max_assistant_attempts}")
        print(f"  random_seed:   {args.random_seed}")
        print(f"  models config: {models_path}")
        if pinned_assets_dir is not None:
            print(f"  assets_dir:    {args.assets_dir}")
        else:
            from usersim.engine.core._assets import asset_search_path

            print(f"  asset path:    {', '.join(str(r) for r in asset_search_path())}")
        print(f"  match_persona_language: {args.match_persona_language}")
        print(f"  toolset_seed_path: {seed or '(not used)'}")
        print(f"  output:        {args.out}")
        print(f"  resumable:     {not args.no_skip_existing}")
        print(f"  store_reasoning: {not args.no_store_reasoning}")
        return 0

    if (msg := warn_if_missing_api_key(models)) is not None:
        logger.warning(msg)

    call_kwargs = dict(
        models=models,
        models_path=models_path,
        panel=args.panel,
        locales=args.locale or [],
        num_rows=args.num_rows,
        out=args.out,
        assets_dir=args.assets_dir,
        pinned_assets_dir=pinned_assets_dir,
        match_persona_language=args.match_persona_language,
        toolset_seed_path=args.toolset_seed_path,
        max_turns=args.max_turns,
        max_assistant_attempts=args.max_assistant_attempts,
        probe_mix=probe_mix,
        random_seed=args.random_seed,
        finance_tier_mix=args.finance_tier_mix,
        finance_retrieval_mode=args.finance_retrieval_mode,
        skip_existing=not args.no_skip_existing,
        store_reasoning=not args.no_store_reasoning,
    )
    # A live models object and Path do not cross a scheduler boundary, so the
    # spec also carries a plain-data description of the same run for a backend
    # that has to rebuild it elsewhere.
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in call_kwargs.items() if k != "models"}
    params["models_path"] = str(models_path)

    # Execution goes through the backend so the local path is the same code
    # a contributed backend replaces, rather than a special case it has to
    # reimplement. Everything above this point is validation and stays here:
    # a bad config must fail the same way regardless of where the run lands.
    from usersim.engine.core.backends import JobSpec, get_backend

    backend = get_backend(getattr(args, "backend", "local"))
    handle = backend.submit(
        JobSpec(
            name="simulate",
            params=params,
            callable_=lambda: _simulate_to_parquet(**call_kwargs),
        ),
    )
    status = backend.poll(handle)
    if status.error is not None:
        raise status.error
    if status.state == "failed" and status.exit_code is None:
        logger.error("simulate failed on backend %r: %s", backend.name, status.message)
        return 1
    return status.exit_code or 0


def _materialize_inputs(args, models, probe_mix: dict[str, float]) -> int:
    """CLI adapter for the public no-model-call materialization seam."""
    import json

    import pandas as pd

    from usersim.cli._pipeline import materialize_episode_inputs

    locales = _locales_from_panel(args.panel) if args.panel else list(args.locale or [])
    rows = []
    for locale in locales:
        count = _count_rows_for_locale(args.panel, locale) if args.panel else args.num_rows
        rows.extend(
            materialize_episode_inputs(
                locale=locale,
                num_rows=count,
                models=models,
                assets_dir=Path(args.assets_dir).resolve(),
                probe_mix=probe_mix,
                random_seed=args.random_seed,
                toolset_seed_path=args.toolset_seed_path,
                match_persona_language=args.match_persona_language,
                max_turns=args.max_turns,
                max_assistant_attempts=args.max_assistant_attempts,
                store_reasoning=not args.no_store_reasoning,
                finance_tier_mix=args.finance_tier_mix,
                finance_retrieval_mode=args.finance_retrieval_mode,
            )
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".jsonl":
        out.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows))
    elif out.suffix == ".parquet":
        pd.DataFrame(rows).to_parquet(out, index=False)
    else:
        raise SystemExit("--materialize-inputs requires --out ending in .jsonl or .parquet")
    print(f"wrote {len(rows)} resolved episode inputs to {out}")
    return 0


def _simulate_to_parquet(
    *,
    models,
    models_path,
    panel,
    locales,
    num_rows,
    out,
    assets_dir,
    pinned_assets_dir=None,
    max_turns,
    match_persona_language=False,
    toolset_seed_path=None,
    probe_mix,
    random_seed,
    finance_tier_mix=0.0,
    finance_retrieval_mode="hybrid",
    skip_existing,
    max_assistant_attempts=1,
    store_reasoning=True,
) -> int:
    from usersim.cli._persona_language import (
        DEFAULT_PERSONA_DATASETS_DIR,
        log_persona_language_population,
        persona_language_filter,
    )
    from usersim.cli._pipeline import (
        DEFAULT_TOOL_CALLING_THEMES,
        build_simulator_config_builder,
    )
    from usersim.engine.core.locale import persona_dataset_locale
    from usersim.engine.core.manifest import write_simulator_manifest
    from usersim.engine.core.storage import (
        existing_trajectory_ids,
        latest_run_id,
        new_run_id,
        write_locale_partition,
    )

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Resolve the target run id. ``--no-skip-existing`` means
    # "fresh start" → mint a new run id. Default (``--skip-existing``)
    # means "resume into the latest existing run". On a first
    # invocation with no existing runs, both paths land on a fresh
    # ``new_run_id()``.
    if skip_existing:
        run_id = latest_run_id(out) or new_run_id()
    else:
        run_id = new_run_id()
    logger.info("run_id=%s (%s)", run_id, "resume" if skip_existing else "fresh")
    print(f"  run_id:        {run_id}")

    # Skip-existing scopes to the resolved run only (not all runs).
    # Resuming into an existing run skips trajectory_ids already
    # written for THIS run; trajectory_ids in other runs are
    # independent (they had different model identities, prompts,
    # seeds — content-hashed trajectory_ids would still collide if
    # everything matched, but cross-run skipping is not the
    # behaviour we want).
    already_done = existing_trajectory_ids(out, run=run_id) if skip_existing else set()
    if already_done:
        logger.info(
            "run=%s already has %d trajectories — will skip",
            run_id,
            len(already_done),
        )

    # Resolve only for a run that includes tool_calling. The builder also
    # enforces this for library callers, while doing it here gives the manifest
    # the exact file used and keeps non-tool runs independent of toolset assets.
    toolset_seed_path = resolve_toolset_seed_for_mix(
        probe_mix,
        pinned_assets_dir,
        toolset_seed_path,
    )

    if panel:
        run_locales = _locales_from_panel(panel)
        seed_source = Path(panel)
    else:
        run_locales = list(locales)
        seed_source = None

    # Fail before any model call rather than from inside a loader mid-run.
    verify_probe_assets(probe_mix, run_locales, pinned_assets_dir)

    total_written = 0
    # What the persona sampler actually narrowed to, per locale. Recorded
    # rather than the flag: the two disagree wherever the corpus has no
    # language column, and the value differs between the Hindi corpora.
    applied_language_filters: dict = {}
    for locale in run_locales:
        logger.info("simulating locale=%s", locale)
        _persona_locale = persona_dataset_locale(
            locale,
            datasets_dir=DEFAULT_PERSONA_DATASETS_DIR,
        )
        _applied = persona_language_filter(
            _persona_locale,
            locale,
            match_language=match_persona_language,
        )
        if _applied:
            applied_language_filters[locale] = _applied
            log_persona_language_population(
                persona_locale=_persona_locale,
                conversation_locale=locale,
                constraint=_applied,
                datasets_dir=DEFAULT_PERSONA_DATASETS_DIR,
            )
        data_designer, builder = build_simulator_config_builder(
            locale=locale,
            models=models,
            assets_dir=Path(assets_dir).resolve(),
            max_turns=max_turns,
            max_assistant_attempts=max_assistant_attempts,
            store_reasoning=store_reasoning,
            probe_mix=probe_mix,
            match_persona_language=match_persona_language,
            toolset_seed_path=toolset_seed_path,
            random_seed=random_seed,
            finance_tier_mix=finance_tier_mix,
            finance_retrieval_mode=finance_retrieval_mode,
        )
        if seed_source is not None:
            n = _count_rows_for_locale(panel, locale)
        else:
            n = num_rows
        result = data_designer.create(builder, num_records=n)
        df = result.load_dataset()
        if already_done:
            df = df[~df["trajectory_id"].isin(already_done)]
        if df.empty:
            logger.info("locale=%s: no new trajectories", locale)
            continue
        # Per-locale atomic write: this locale's partition lands on
        # disk before the next locale starts. A crash mid-run leaves
        # earlier locales' partitions intact and resumable (within
        # the same run id).
        write_locale_partition(df, out, locale=locale, run_id=run_id)
        total_written += len(df)
        logger.info(
            "locale=%s: wrote %d new trajectories",
            locale,
            len(df),
        )

    if total_written == 0:
        print("no new trajectories generated")
        return 0

    print(f"wrote {total_written} new trajectories to {out}")
    print(
        f"\nNext step: score them\n  usersim eval --run {run_id} \\\n    --trajectories {out} --out output/evaluations"
    )

    # Write the run manifest. ``run_id`` is the unix-epoch start time
    # (see ``new_run_id``), so we pass it as ``started_at_epoch`` to keep
    # the manifest's started/runtime fields meaningful even on resume.
    # First-writer-wins: a resume into a run that already has a manifest
    # is a noop. ``trajectories_requested`` is the per-locale request
    # count from this invocation; the manifest's ``trajectories_completed``
    # comes from re-reading the parquet partitions for this run.
    requested_per_locale: dict[str, int] = {l: int(num_rows) for l in run_locales} if num_rows else {}
    sim_cfg_for_manifest = _build_sim_config_for_manifest(
        max_turns=max_turns,
        random_seed=random_seed,
        store_reasoning=store_reasoning,
    )
    try:
        manifest_path_written = write_simulator_manifest(
            run_id=run_id,
            started_at_epoch=int(run_id),
            trajectory_root=out,
            models_config=models,
            models_config_path=models_path,
            sim_config=sim_cfg_for_manifest,
            locales_requested=run_locales,
            probe_mix=probe_mix,
            tool_calling_themes=DEFAULT_TOOL_CALLING_THEMES,
            toolset_seed_path=toolset_seed_path,
            panel_path=Path(panel) if panel else None,
            assets_dir=Path(assets_dir).resolve(),
            trajectories_requested=requested_per_locale,
            persona_language_filters=applied_language_filters,
            source_kind="cli",
        )
        print(f"  manifest:      {manifest_path_written}")
    except Exception as e:  # pragma: no cover — defensive: never fail a run on manifest issues
        logger.warning("failed to write run manifest: %s", e)
    return 0


def _build_sim_config_for_manifest(
    *,
    max_turns: int,
    random_seed,
    store_reasoning: bool = True,
):
    """Snapshot the sim-config knobs the manifest captures.

    Mirrors what ``cli/_pipeline.build_simulator_config_builder`` passes
    to ``ConversationSimulatorConfig``. Default fields fall through from
    the Pydantic config class so that any future default change is
    picked up automatically by the manifest snapshot.
    """
    from usersim.engine.config import ConversationSimulatorConfig

    return ConversationSimulatorConfig(
        name="conversation_messages",
        locale="placeholder",
        max_turns=max_turns,
        context_compression=True,
        compression_window=1,
        random_seed=random_seed,
        store_reasoning=store_reasoning,
    )


def _locales_from_panel(panel_path: Path) -> list[str]:
    """Read the unique locales from a panel parquet (single-file).

    The panel parquet is intentionally NOT a partitioned dataset —
    it is a frozen persona seed and probe assignment happens at
    simulate time, not at panel-build time.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(panel_path, columns=["locale"])
    column = table.column("locale")
    return sorted({v.as_py() for v in column if v.is_valid and v.as_py() is not None})


def _count_rows_for_locale(panel_path: Path, locale: str) -> int:
    """Count rows in the panel parquet for a given locale."""
    import pyarrow.parquet as pq

    table = pq.read_table(panel_path, columns=["locale"])
    column = table.column("locale")
    return sum(1 for v in column if v.is_valid and str(v.as_py()) == str(locale))
