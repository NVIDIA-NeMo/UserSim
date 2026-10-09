# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer pipeline builders for simulator + evaluator.

Two pipelines are exposed:

- :func:`build_simulator_config_builder` — assembles a ``ConfigBuilder``
  that produces trajectory rows (``conversation_messages`` + side-effect
  columns including ``simulation_outcome``, ``simulation_traces``,
  ``persona_uuid``, ``trajectory_id``).
- :func:`build_evaluator_config_builder` — assembles a ``ConfigBuilder``
  that reads a trajectory parquet and emits a wide evaluator cell
  (per-axis × per-judge scores + scorer outputs) for each row.

Both builders take a ``ModelsConfig`` (loaded by :mod:`cli._models`)
and a small set of pipeline knobs. They are pure factory functions —
they do not call the network or write files; the caller invokes
``DataDesigner.create(builder, num_records=N).load_dataset()`` (or
``.to_parquet()``) to actually run the pipeline.

The notebooks under ``notebooks/`` import these same functions so the
notebook flow matches the CLI flow byte-for-byte.

DD imports are local so ``--help`` works without ``data_designer``
installed (e.g., on doc-build VMs).
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from usersim.cli._errors import ConfigError
from usersim.cli._models import ModelsConfig, to_data_designer_kwargs, to_model_configs
from usersim.cli._persona_language import (
    DEFAULT_PERSONA_DATASETS_DIR,
    persona_sampler_params,
)
from usersim.engine.evaluator.prompts import EVAL_PROMPT_VERSION

logger = logging.getLogger("usersim.cli.pipeline")

DEFAULT_TOOL_CALLING_THEMES: tuple[str, ...] = (
    '{"type": "Weather & Location Lookup", "description": "Look up current weather, forecasts, or location-based info.", "tool_expected": true}',
    '{"type": "Calendar & Scheduling", "description": "Create, modify, or check calendar events and availability.", "tool_expected": true}',
    '{"type": "Data Retrieval & Search", "description": "Search databases, look up records, or retrieve structured data.", "tool_expected": true}',
    '{"type": "Financial Operations", "description": "Check balances, make transactions, or calculate financial metrics.", "tool_expected": true}',
    '{"type": "Communication & Messaging", "description": "Send emails, messages, or notifications on behalf of the user.", "tool_expected": true}',
    '{"type": "Multi-Step Workflow", "description": "Complete a task requiring multiple sequential tool calls.", "tool_expected": true}',
    '{"type": "General Knowledge Question", "description": "Answer a factual question that requires no tools.", "tool_expected": false}',
    '{"type": "Opinion or Advice Request", "description": "Ask for subjective advice or opinions where tools are unnecessary.", "tool_expected": false}',
)

DEFAULT_PROBE_MIX: dict[str, float] = {
    "tool_calling": 1 / 3,
    "general_open_ended": 1 / 3,
    "general_educational": 1 / 3,
}

# Probes that consume the ``theme`` column. Other probes are
# persona-derived (they read curated assets like fact / query / pressure
# / agentic banks) and the ``theme`` column is irrelevant to them.
THEME_DRIVEN_PROBES: frozenset = frozenset(
    {
        "tool_calling",
        "general_open_ended",
        "general_educational",
    }
)

# Sentinel value used in the SubcategorySamplerParams entry for
# non-theme-driven probes. The probe never reads this; it
# exists so DD's SubcategorySampler has a value to draw when the
# probe_type sampled is one that doesn't use themes.
_NO_THEME_PLACEHOLDER: str = "__not_used__"


def seed_themes(
    locale: str,
    probe: str,
    kind: str,
    assets_dir: Path | str | None = None,
) -> list[str]:
    """Load a theme-driven probe's seeds and JSON-encode them for the sampler.

    Called only for probes actually present in the run's ``probe_mix``, so an
    assets root that ships seeds for just one general probe stays usable
    (``load_seeds`` raises ``FileNotFoundError`` when neither the locale nor
    the ``base`` seed file exists).

    ``ensure_ascii=False`` so non-Latin theme descriptions (Japanese / Hindi /
    Chinese / etc.) survive serialization as readable native script rather
    than ``\\uXXXX`` escapes. Downstream ``json.loads`` reads either form
    correctly; the parquet column is just easier on human eyeballs this way.
    """
    from usersim.engine.core.seeds import load_seeds

    return [json.dumps(t, ensure_ascii=False) for t in load_seeds(locale, probe, kind, assets_dir)]


#: Seed-file suffixes the ``tool_calling`` toolset seed may use. Both are
#: accepted by Data Designer's
#: ``LocalFileSeedSource`` (``VALID_DATASET_FILE_EXTENSIONS``); parquet is
#: what ships, jsonl is the hand-editable alternative.
TOOLSET_SEED_SUFFIXES: tuple[str, ...] = (".parquet", ".jsonl")


def resolve_toolset_seed_path(
    assets_dir: Path | str | None = None,
    toolset_seed_path: Path | str | None = None,
) -> Path | None:
    """The ``tool_calling`` toolset seed to use, or ``None`` if there isn't one.

    Two sources, at least one of which must be given:

    - ``toolset_seed_path`` — an explicit file, which wins when both are
      passed.
      It must exist: you named a path, and a run that silently ignores it
      is worse than one that stops.
    - ``assets_dir`` — the asset root to look under, accepting
      ``toolsets_seed.parquet`` or ``toolsets_seed.jsonl``
      (``TOOLSET_SEED_SUFFIXES``).

    Lives here rather than in the CLI because ``assets_dir`` is already a
    builder argument: any caller that names an asset root can have its
    toolset found for it, including the notebooks.
    Passing ``toolset_seed_path`` is an override, not a requirement.

    Returns ``None`` only when ``assets_dir`` holds neither suffix.
    Absence is a resolution result; whether it matters depends on the
    probe mix, which ``_wire_toolsets`` decides.

    Raises ``ConfigError`` (a ``ValueError``) — never ``SystemExit`` — so
    library callers can handle it and ``cli.main`` can present it as a
    message rather than a traceback, when an explicit path is unusable or an
    asset root holds BOTH suffixes. That last is a configuration error rather
    than a precedence question, since silently preferring one would make an
    edit to the other look like it had no effect. Passing neither source
    searches the configured asset path, which is how an unpinned run finds
    the shipped seed.

    Routes through ``probe_assets_dir`` so the path is composed against the
    same source-of-truth helper the asset loaders use, rather than a string
    literal that can drift from the real layout.
    """

    if toolset_seed_path is not None:
        resolved = Path(toolset_seed_path).resolve()
        if not resolved.is_file():
            raise ConfigError(f"toolset seed not found: {resolved}")
        if resolved.suffix not in TOOLSET_SEED_SUFFIXES:
            raise ConfigError(
                f"unsupported toolset seed suffix {resolved.suffix!r}: "
                f"{resolved}. Expected one of {TOOLSET_SEED_SUFFIXES}."
            )
        return resolved

    from usersim.engine.core._assets import probe_assets_dir

    # ``None`` means the caller did not pin a root, so the seed is looked
    # for along the same search path the loaders use.
    probe_dir = probe_assets_dir(
        "tool_calling",
        assets_root=Path(assets_dir).resolve() if assets_dir is not None else None,
    )
    candidates = [probe_dir / f"toolsets_seed{sfx}" for sfx in TOOLSET_SEED_SUFFIXES]
    present = [p for p in candidates if p.is_file()]
    if len(present) > 1:
        raise ConfigError(
            "ambiguous tool_calling toolset seed: "
            + ", ".join(p.name for p in present)
            + f" both exist in {probe_dir}. Keep exactly one."
        )
    return present[0] if present else None


#: Probe asset trees that are partitioned by locale rather than flat. For
#: these, having the probe directory is not enough -- the run's locale must
#: have its own subtree, or the loader fails once the run is already going.
_LOCALE_SCOPED_PROBES: frozenset[str] = frozenset(
    {
        "financial_services",
        "sov_ai_dynamic",
        "sov_ai_facts",
    }
)


def verify_probe_assets(
    probe_mix: dict[str, float],
    locales: "list[str] | tuple[str, ...]",
    assets_dir: Path | str | None = None,
) -> None:
    """Fail before any model call if a selected probe's assets are absent.

    Without this, a missing bank surfaces from deep inside a loader partway
    through a run -- after the other probes in the mix have already been
    billed for. ``--dry-run`` reports the plan and exits 0, which is worse
    than an unclear error: it says the run is fine.

    Raises ``ConfigError`` naming every missing path at once, so a caller
    fixes one thing rather than rediscovering the next on each attempt.
    """
    from usersim.engine.core._assets import probe_assets_dir

    root = Path(assets_dir) if assets_dir is not None else None
    problems: list[str] = []

    for probe in sorted(probe_mix):
        probe_dir = probe_assets_dir(probe, assets_root=root)
        if not probe_dir.is_dir():
            problems.append(f"{probe}: no asset directory at {probe_dir}")
            continue
        if not any(probe_dir.rglob("*.yaml")) and not any(probe_dir.rglob("*.json")):
            problems.append(f"{probe}: {probe_dir} contains no bank files")
            continue
        if probe in _LOCALE_SCOPED_PROBES:
            missing = [loc for loc in locales if not (probe_dir / loc).is_dir()]
            if missing:
                available = sorted(p.name for p in probe_dir.iterdir() if p.is_dir())
                problems.append(
                    f"{probe}: no assets for locale(s) {', '.join(missing)} "
                    f"(available: {', '.join(available) or 'none'})"
                )

    if problems:
        raise ConfigError(
            "probe assets are missing for this run:\n  - "
            + "\n  - ".join(problems)
            + "\n\nEither point --assets-dir at a tree that has them, "
            "generate the bank (`usersim gen-assets --domain <d>`), "
            "or drop the probe from --probe-mix."
        )


def resolve_toolset_seed_for_mix(
    probe_mix: dict[str, float],
    assets_dir: Path | str | None = None,
    toolset_seed_path: Path | str | None = None,
) -> Path | None:
    """Resolve a toolset only when the run actually includes ``tool_calling``.

    A non-tool run must not be blocked by an ambiguous or missing toolset in
    an otherwise valid asset tree, and its manifest must not claim it used a
    file that never entered the pipeline. A tool run must resolve a real seed.
    Conversely, an explicit override on a non-tool run is rejected rather than
    silently ignored.
    """
    if "tool_calling" not in probe_mix:
        if toolset_seed_path is not None:
            raise ConfigError(
                "toolset_seed_path was provided, but probe_mix does not "
                "include tool_calling. Add tool_calling or remove the override."
            )
        return None
    resolved = resolve_toolset_seed_path(assets_dir, toolset_seed_path)
    if resolved is None:
        raise ConfigError(
            "probe_mix includes tool_calling but no toolset seed is available. "
            "Pass toolset_seed_path=<path>, point assets_dir at a tree with "
            "tool_calling/toolsets_seed.{parquet,jsonl}, or drop tool_calling "
            "from the mix."
        )
    return resolved


def _wire_toolsets(
    config_builder: Any,
    dd: Any,
    toolset_seed_path: Path | None,
) -> dict[str, Any]:
    """Attach ``tool_calling``'s toolset seed and return its config keys.

    Both halves of the wiring — the builder mutation and the
    ``ConversationSimulatorConfig`` keys — are decided here together, so the
    two cannot drift apart: a run that attaches the seed dataset always gets
    the matching tool columns, and one that does not always leaves them
    unset.

    Called only from the ``tool_calling`` branch. The mix-aware resolver
    normally rejects a missing file first; the check here is a defensive
    invariant. Failing at config-build time rather than letting the run start
    is the point: with the columns unset, every ``tool_calling`` row aborts at
    probe construction, so the whole partition comes back failed and the
    cause is visible only by reading ``simulation_outcome`` afterwards.

    ``ConfigError`` matches this module's other user-fixable errors (see
    the ``probe_mix`` validation above) and guards every caller — the CLI,
    and the notebooks — rather than just the one that
    happens to resolve the path. ``cli.main`` catches it and prints the
    message alone, so this text is the entire error output a user sees.

    Takes an already-resolved path: the caller resolves only after confirming
    that ``tool_calling`` is in the mix, so this function needs no
    ``assets_dir``.
    """
    if toolset_seed_path is None or not Path(toolset_seed_path).is_file():
        raise ConfigError(
            "probe_mix includes tool_calling but no toolset seed is "
            f"available (resolved to {toolset_seed_path!r}). Pass "
            "toolset_seed_path=<path>, point assets_dir at a tree with "
            "tool_calling/toolsets_seed.{parquet,jsonl}, or drop "
            "tool_calling from the mix."
        )
    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(toolset_seed_path)),
        sampling_strategy=dd.SamplingStrategy.SHUFFLE,
    )
    return {"tools_column": "tools", "toolset_name_column": "toolset_name"}


def _set_run_config(data_designer, dd) -> None:
    """Configure the DataDesigner instance with the ``NATIVE`` Jinja2 engine.

    Required by :func:`_add_persona_expressions`'s ``persona_location``
    template, which needs ``select | unique | join`` to collapse the
    four-tuple of city/district/region/country into a single string.
    ``NATIVE`` mode disables DD's hardened sandbox; safe here because
    every Jinja template we render is authored by us (asset banks + this
    module's own expression strings) — no user-supplied template ever
    reaches DD.

    Trajectory concurrency is not set here. Trajectories are coroutines
    on the engine's event loop, and how many run at once is governed by
    the scheduler's own admission control. What bounds load on any one
    provider is ``max_parallel_requests``, set per alias in the models
    TOML.

    The metrics endpoint is off. Left on, every run opens a local HTTP
    listener on a fixed port that nothing here reads, and a second run
    sharing the machine cannot bind it -- which surfaces as a traceback
    mid-run. Set ``otel_metrics_port`` on the run config to turn it back
    on for a session that wants to scrape it.
    """
    run_config = dd.RunConfig(
        jinja_rendering_engine=dd.JinjaRenderingEngine.NATIVE,
        otel_metrics_port=None,
    )
    data_designer.set_run_config(run_config)


def known_probes() -> tuple[str, ...]:
    """Return the set of probe_type values the simulator dispatcher
    knows about. Source-of-truth is the plugin's ``_PROBE_REGISTRY``.

    Used by ``cli.simulate`` to validate ``--probe-mix`` keys before
    running.
    """
    from usersim.engine.core.probes import known_probes as _known

    return _known()


def build_simulator_config_builder(
    *,
    locale: str,
    models: ModelsConfig,
    assets_dir: Path,
    max_turns: int = 5,
    max_assistant_attempts: int = 1,
    store_reasoning: bool = True,
    probe_mix: dict[str, float] | None = None,
    tool_calling_themes: tuple[str, ...] | None = None,
    toolset_seed_path: Path | None = None,
    match_persona_language: bool = False,
    verbosity: int = 1,
    random_seed: int | None = None,
    finance_tier_mix: float = 0.0,
    finance_retrieval_mode: str = "hybrid",
    finance_embedding_model_alias: str = "embedding_model",
):
    """Build a Data Designer ConfigBuilder for the conversation simulator.

    Returns ``(data_designer, config_builder)`` so the caller can
    invoke ``data_designer.create(config_builder, num_records=N)``.
    ``random_seed`` is forwarded to
    ``ConversationSimulatorConfig.random_seed`` so tool sampling is
    reproducible per run.
    """
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    from usersim.engine.config import ConversationSimulatorConfig

    probe_mix = probe_mix or DEFAULT_PROBE_MIX

    dd_kwargs = to_data_designer_kwargs(models)
    data_designer = DataDesigner(**dd_kwargs)
    _set_run_config(data_designer, dd)
    config_builder = dd.DataDesignerConfigBuilder(model_configs=to_model_configs(models))
    toolset_kwargs = _add_episode_input_columns(
        config_builder,
        dd,
        locale=locale,
        assets_dir=assets_dir,
        probe_mix=probe_mix,
        tool_calling_themes=tool_calling_themes,
        toolset_seed_path=toolset_seed_path,
        match_persona_language=match_persona_language,
        drop_persona=True,
    )

    config_builder.add_column(
        ConversationSimulatorConfig(
            name="conversation_messages",
            persona_column="persona",
            probe_type_column="probe_type",
            theme_column="theme",
            **toolset_kwargs,
            locale=locale,
            assets_dir=str(Path(assets_dir).resolve()),
            max_turns=max_turns,
            max_assistant_attempts=max_assistant_attempts,
            store_reasoning=store_reasoning,
            context_compression=True,
            compression_window=1,
            verbosity=verbosity,
            random_seed=random_seed,
            finance_tier_mix=finance_tier_mix,
            finance_retrieval_mode=finance_retrieval_mode,
            finance_embedding_model_alias=finance_embedding_model_alias,
        )
    )

    return data_designer, config_builder


def _add_episode_input_columns(
    config_builder: Any,
    dd: Any,
    *,
    locale: str,
    assets_dir: Path,
    probe_mix: dict[str, float],
    tool_calling_themes: tuple[str, ...] | None,
    toolset_seed_path: Path | None,
    match_persona_language: bool,
    drop_persona: bool,
) -> dict[str, Any]:
    """Install the one canonical persona/probe/theme/toolset sampler graph."""
    from usersim.engine.core.locale import persona_dataset_locale

    persona_locale = persona_dataset_locale(locale, datasets_dir=DEFAULT_PERSONA_DATASETS_DIR)
    unknown = [key for key in probe_mix if key not in set(known_probes())]
    if unknown:
        raise ConfigError(f"Unknown probe_type(s) in probe_mix: {unknown}. Registered: {sorted(known_probes())}.")
    resolved_toolset_seed = resolve_toolset_seed_for_mix(probe_mix, assets_dir, toolset_seed_path)
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="persona",
            drop=drop_persona,
            sampler_type=dd.SamplerType.PERSON,
            params=persona_sampler_params(
                dd,
                persona_locale,
                locale,
                match_language=match_persona_language,
            ),
        )
    )
    _add_persona_expressions(config_builder, dd)
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="probe_type",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=list(probe_mix), weights=list(probe_mix.values())),
        )
    )
    themes: dict[str, list[str]] = {}
    toolset_kwargs: dict[str, Any] = {}
    for probe in probe_mix:
        if probe == "tool_calling":
            themes[probe] = list(tool_calling_themes or DEFAULT_TOOL_CALLING_THEMES)
            toolset_kwargs = _wire_toolsets(config_builder, dd, resolved_toolset_seed)
        elif probe == "general_open_ended":
            themes[probe] = seed_themes(locale, probe, "topics", assets_dir)
        elif probe == "general_educational":
            themes[probe] = seed_themes(locale, probe, "subjects", assets_dir)
        else:
            themes[probe] = [_NO_THEME_PLACEHOLDER]
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="theme",
            sampler_type=dd.SamplerType.SUBCATEGORY,
            params=dd.SubcategorySamplerParams(category="probe_type", values=themes),
        )
    )
    return toolset_kwargs


def materialize_episode_inputs(
    *,
    locale: str,
    num_rows: int,
    models: ModelsConfig,
    assets_dir: Path,
    probe_mix: dict[str, float],
    random_seed: int | None = None,
    tool_calling_themes: tuple[str, ...] | None = None,
    toolset_seed_path: Path | None = None,
    match_persona_language: bool = False,
    max_turns: int = 5,
    max_assistant_attempts: int = 1,
    store_reasoning: bool = True,
    finance_tier_mix: float = 0.0,
    finance_retrieval_mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """Materialize fully resolved simulator rows without model calls."""
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    from usersim.engine.config import ConversationSimulatorConfig
    from usersim.engine.core.episode_input import construct_probe_episode

    designer = DataDesigner(**to_data_designer_kwargs(models))
    _set_run_config(designer, dd)
    builder = dd.DataDesignerConfigBuilder()
    toolset_kwargs = _add_episode_input_columns(
        builder,
        dd,
        locale=locale,
        assets_dir=assets_dir,
        probe_mix=probe_mix,
        tool_calling_themes=tool_calling_themes,
        toolset_seed_path=toolset_seed_path,
        match_persona_language=match_persona_language,
        drop_persona=False,
    )
    config = ConversationSimulatorConfig(
        name="conversation_messages",
        locale=locale,
        assets_dir=str(assets_dir.resolve()),
        random_seed=random_seed,
        max_turns=max_turns,
        max_assistant_attempts=max_assistant_attempts,
        store_reasoning=store_reasoning,
        finance_tier_mix=finance_tier_mix,
        finance_retrieval_mode=finance_retrieval_mode,
        **toolset_kwargs,
    )
    model_identities = {spec.alias: SimpleNamespace(model_name=spec.model) for spec in models.models}
    sampled = designer.preview(builder, num_records=num_rows).dataset
    if sampled is None:
        return []
    # ``usersim_config`` rides on the materialized row only. It is what lets a
    # host restore the episode with ProbeEpisodeRuntime.from_resolved_row; the
    # simulator's own trajectory rows do not repeat it.
    config_snapshot = config.model_dump(mode="json")
    rows: list[dict[str, Any]] = []
    for row in sampled.to_dict(orient="records"):
        resolved = construct_probe_episode(row, config=config, models=model_identities).preamble.to_row()
        resolved["usersim_config"] = deepcopy(config_snapshot)
        rows.append(resolved)
    return rows


def build_inputs_config_builder(
    *,
    models: ModelsConfig,
    config: Any,
    rows: list[dict[str, Any]],
):
    """Build a Data Designer pipeline that runs stored episode rows as they are.

    Each row becomes one seed row: the columns the simulator column requires
    (``persona``, ``probe_type``, ``theme``), plus the whole stored row as JSON
    in ``EPISODE_INPUT_COLUMN``, which the generator unpacks before the probe
    is built, so every value keeps the type it was stored with. ``config`` is
    the simulator column, restored from the rows' ``usersim_config``.

    Returns ``(data_designer, config_builder)``, like
    :func:`build_simulator_config_builder`.
    """
    import data_designer.config as dd
    import pandas as pd
    from data_designer.interface import DataDesigner

    from usersim.engine.core.episode_input import EPISODE_INPUT_COLUMN

    def _text(value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    seed = pd.DataFrame(
        [
            {
                config.persona_column: _text(row[config.persona_column]),
                config.probe_type_column: str(row[config.probe_type_column]),
                config.theme_column: _text(row.get(config.theme_column, "")),
                EPISODE_INPUT_COLUMN: json.dumps(
                    {key: value for key, value in row.items() if key != "usersim_config"}, ensure_ascii=False
                ),
            }
            for row in rows
        ]
    )
    data_designer = DataDesigner(**to_data_designer_kwargs(models))
    _set_run_config(data_designer, dd)
    config_builder = dd.DataDesignerConfigBuilder(model_configs=to_model_configs(models))
    config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed))
    config_builder.add_column(config)
    return data_designer, config_builder


def build_evaluator_config_builder(
    *,
    models: ModelsConfig,
    trajectory_parquet: Path,
    judges: list[dict[str, Any]] | None = None,
    axes: list[str] | None = None,
    scorers: list[str] | None = None,
    skip_if_existing: bool = True,
    output_column: str = "assistant_eval",
    prompt_version: str = EVAL_PROMPT_VERSION,
):
    """Build a Data Designer ConfigBuilder for the trajectory evaluator.

    The trajectory parquet feeds in as the seed dataset; the evaluator
    plugin reads ``conversation_messages`` + ``simulation_outcome`` per
    row and emits a single wide JSON cell named ``output_column``.

    ``judges`` is a list of dicts shaped like::

        [{"alias": "evaluator_model"}, {"alias": "evaluator_model_secondary", "family": "OPENAI"}]

    matching ``JudgeSpecConfig``. Defaults to a single-judge ensemble
    keyed on the ``evaluator_model`` alias — which is what
    ``REQUIRED_EVALUATOR_ALIASES`` guarantees is present in the
    ``ModelsConfig``. Multi-judge ensembles for inter-judge κ work
    pass an explicit list. ``axes`` defaults to the universal set;
    ``scorers`` defaults to the empty list (opt in per probe).
    """
    if judges is None:
        judges = [{"alias": "evaluator_model"}]
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    from usersim.engine.evaluator.config import (
        JudgeSpecConfig,
        TrajectoryEvaluatorConfig,
    )

    dd_kwargs = to_data_designer_kwargs(models)
    data_designer = DataDesigner(**dd_kwargs)
    _set_run_config(data_designer, dd)
    config_builder = dd.DataDesignerConfigBuilder(model_configs=to_model_configs(models))

    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(Path(trajectory_parquet).resolve())),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )

    judge_specs = [JudgeSpecConfig(**j) for j in judges]
    _warn_on_undispatched_scorers(scorers or [])

    config_builder.add_column(
        TrajectoryEvaluatorConfig(
            name=output_column,
            judges=judge_specs,
            axes=axes,
            scorers=scorers or [],
            skip_if_existing=skip_if_existing,
            prompt_version=prompt_version,
        )
    )

    return data_designer, config_builder


def _warn_on_undispatched_scorers(scorers: list[str]) -> None:
    """Say so when the run cannot possibly score some capability.

    The evaluator dispatches exactly the scorers it is handed and keys off nothing
    else -- not the probe, not the row. A capability whose scorer is absent finds no
    evidence, which is indistinguishable from "the model was never tested": the cell
    reads Untested with n=0. That is how the 11 financial_services capabilities came
    back empty across a 13-locale run while the scorer sat registered and correct.
    Warn rather than raise, because scoring a subset on purpose is legitimate.
    """
    try:
        from usersim.taxonomy.capabilities import capabilities_missing_scorer
    except ImportError:  # reporting is not a dependency of the plugin distribution
        return
    missing = capabilities_missing_scorer(scorers)
    if not missing:
        return
    affected = sum(len(ids) for ids in missing.values())
    if not scorers:
        # Dispatching nothing is unambiguous, so say it once instead of once per
        # scorer -- a warning that fires ten times on a deliberate choice is noise
        # people learn to scroll past, which is how the real one gets missed.
        logger.warning(
            "no scorers dispatched, so all %d capabilit%s needing one will render as Untested (n=0).",
            affected,
            "y" if affected == 1 else "ies",
        )
        return
    for scorer, ids in missing.items():
        shown = ", ".join(ids[:5]) + (f", +{len(ids) - 5} more" if len(ids) > 5 else "")
        logger.warning(
            "scorer %r is not in this run's scorer list, so %d capabilit%s will "
            "render as Untested (n=0) rather than scored: %s. Add it to --scorers "
            "(notebook: SCORERS) to measure %s.",
            scorer,
            len(ids),
            "y" if len(ids) == 1 else "ies",
            shown,
            "it" if len(ids) == 1 else "them",
        )


def sample_trajectories(
    df,
    *,
    mode: str,
    n: int | None = None,
    seed: int = 42,
):
    """Coverage-aware sampler for the trajectory-evaluator's seed dataset.

    Modes:

    - ``"full"``             — pass-through; return the input df unchanged.
    - ``"per_locale"``       — sample ``min(n, |group|)`` rows per ``locale``.
    - ``"per_locale_probe"`` — sample ``min(n, |group|)`` rows per
      ``(locale, probe_family)`` pair.
    - ``"matched_pair"``     — restrict to the ``sov_ai_multilingual_parity``
      probe and keep the first ``n`` ``query_id`` values (every row sharing
      one of those query_ids comes through, preserving cross-locale matched
      pairs for the parity scorer).

    ``n=None`` (the default) means "no cap": ``per_locale`` /
    ``per_locale_probe`` return every row in each group (functionally
    equivalent to ``mode="full"``); ``matched_pair`` keeps every
    ``query_id`` in the parity probe.

    Stable across runs at fixed ``seed`` (the per-group ``DataFrame.sample``
    calls are seeded; the ``"matched_pair"`` mode is deterministic by
    sorted-query_id selection).
    """
    import pandas as pd

    def _take(group):
        return group if n is None else group.sample(n=min(n, len(group)), random_state=seed)

    if mode == "full":
        return df
    if mode == "per_locale":
        return pd.concat([_take(g) for _, g in df.groupby("locale", sort=True)]).sort_index()
    if mode == "per_locale_probe":
        return pd.concat([_take(g) for _, g in df.groupby(["locale", "probe_family"], sort=True)]).sort_index()
    if mode == "matched_pair":
        parity = df[df["probe_family"].astype(str) == "sov_ai_multilingual_parity"]
        all_ids = sorted(parity["query_id"].dropna().astype(str).unique())
        query_ids = all_ids if n is None else all_ids[:n]
        return parity[parity["query_id"].astype(str).isin(query_ids)].sort_index()
    raise ValueError(f"Unknown sample mode {mode!r}")


def evaluate_dataframe(
    df,
    *,
    models: ModelsConfig,
    scorers: list[str],
    judges: list[dict[str, Any]] | None = None,
    axes: list[str] | None = None,
    prompt_version: str = EVAL_PROMPT_VERSION,
    eval_column: str = "assistant_eval",
    skip_if_existing: bool = True,
):
    """Run the trajectory evaluator over ``df`` and return a clean DataFrame.

    Encapsulates the temp-parquet seed-source dance that DataDesigner's
    ``LocalFileSeedSource`` requires (it reads from a path, not a
    DataFrame). ``judges`` defaults to a single-judge ensemble keyed on
    the ``evaluator_model`` alias; pass an explicit list only for
    multi-judge inter-κ work. Three quirks preserved from the prior
    notebook flow:

    - Temp-parquet prefix ``usersim_eval_seed_`` (so log filtering and
      cleanup hooks that key on this prefix keep working).
    - ``result.load_dataset().reset_index(drop=True)`` — DataDesigner
      returns a non-default index that breaks downstream joins if not
      reset.
    - Backfills ``trajectory_id`` / ``locale`` / ``probe_family`` on rows
      where DataDesigner's output dropped seed columns (some scorer paths
      project away the seed columns; the dashboard's join needs them).
    """
    import tempfile
    from pathlib import Path as _Path

    import pandas as pd  # noqa: F401  -- imported for clarity at the call site

    if judges is None:
        judges = [{"alias": "evaluator_model"}]

    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, prefix="usersim_eval_seed_")
    tmp.close()
    seed_path = _Path(tmp.name)
    try:
        df.to_parquet(seed_path, index=False)
        data_designer, builder = build_evaluator_config_builder(
            models=models,
            trajectory_parquet=seed_path,
            judges=judges,
            axes=axes,
            scorers=scorers,
            skip_if_existing=skip_if_existing,
            output_column=eval_column,
            prompt_version=prompt_version,
        )
        result = data_designer.create(builder, num_records=len(df))
        eval_df = result.load_dataset().reset_index(drop=True)
    finally:
        if seed_path.exists():
            seed_path.unlink()

    key_cols = ["trajectory_id", "locale", "probe_family"]
    source = df.reset_index(drop=True)
    for col in key_cols:
        if col not in eval_df.columns and col in source.columns:
            eval_df[col] = source[col]
    return eval_df


def _add_persona_expressions(config_builder, dd) -> None:
    """Add the seven persona-* expression columns shared by the notebook."""
    config_builder.add_column(
        dd.ExpressionColumnConfig(name="persona_name", expr="{{ persona.first_name }} {{ persona.last_name }}")
    )
    config_builder.add_column(dd.ExpressionColumnConfig(name="persona_age", expr="{{ persona.age }}"))
    # ``persona_sex`` was added after the original 7-column projection
    # so existing parquets won't have it. The preview defends against
    # that with a None fallback. Coerced to lowercase string so
    # downstream filters / stratifications get a stable vocabulary
    # (Nemotron-Personas already ships ``"female"`` / ``"male"`` but
    # locale extensions may surface other values; lowercase + str()
    # keeps the column predictable). Trailing ``'-'`` fallback is
    # the same defence as ``persona_city`` / ``persona_region``: DD
    # rejects empty-string template renders, so a locale persona
    # missing the ``sex`` field would crash mid-batch otherwise.
    # The preview's ``_format_summary_sex`` treats ``"-"`` like any
    # other string value (no special "missing" rendering).
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="persona_sex",
            expr="{{ ((persona.sex or '-') | string | lower) }}",
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="persona_age_bin",
            expr=(
                "{% if persona.age < 25 %}18-24"
                "{% elif persona.age < 35 %}25-34"
                "{% elif persona.age < 45 %}35-44"
                "{% elif persona.age < 55 %}45-54"
                "{% elif persona.age < 65 %}55-64"
                "{% else %}65+{% endif %}"
            ),
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(name="persona_education_level", expr="{{ persona.education_level }}")
    )
    config_builder.add_column(dd.ExpressionColumnConfig(name="persona_occupation", expr="{{ persona.occupation }}"))
    # Location fields. The Nemotron-Personas dataset has a uniform
    # schema across every shipped locale — ``city`` / ``region`` /
    # ``district`` / ``country`` (verified by inspecting
    # ``~/.data-designer/managed-assets/datasets/<locale>.parquet``).
    # What VARIES is which fields are populated:
    #
    #   - ``en_US``: city + region (state code) + district (county)
    #     + country
    #   - ``en_SG``: district (subzone, e.g. "Bukit Panjang") +
    #     country only — city-state, no city / region
    #   - ``fr_FR``, ``ja_JP``, ``pt_BR``, ``en_IN``,
    #     ``hi_Deva_IN``: district (city-equivalent) + region
    #     (state-equivalent) + country — city always null
    #
    # ``persona_location`` projects the four fields in small→large
    # order, deduped + truthy-filtered, producing strings like:
    #
    #   - en_US:      "Aiea, Honolulu County, HI, USA"
    #   - en_SG:      "Bukit Panjang, Singapore"
    #   - fr_FR:      "Iffendic, Ille-et-Vilaine, France"
    #   - ja_JP:      "関東地方, 東京都, 日本"
    #   - en_IN:      "Raipur, Chhattisgarh, India"
    #
    # ``persona_country`` is a separate single-field projection for
    # downstream stratification (analysis.py / cohort reports) where
    # joining on the full Location string is too granular.
    #
    # The explicit branchy template avoids broader Jinja filters such
    # as ``select`` / ``unique`` / ``join`` so serialized configs can
    # run under a client-controlled Data Designer run config.
    # It still skips missing values and adjacent duplicate location
    # fields, then falls back to ``'-'`` to satisfy DD's no-empty-render
    # constraint.
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="persona_location",
            expr=(
                "{% if persona.city %}{{ persona.city }}{% endif %}"
                "{% if persona.district and persona.district != persona.city %}"
                "{% if persona.city %}, {% endif %}{{ persona.district }}{% endif %}"
                "{% if persona.region and persona.region != persona.city "
                "and persona.region != persona.district %}"
                "{% if persona.city or (persona.district and persona.district != persona.city) %}, {% endif %}"
                "{{ persona.region }}{% endif %}"
                "{% if persona.country and persona.country != persona.city "
                "and persona.country != persona.district and persona.country != persona.region %}"
                "{% if persona.city or (persona.district and persona.district != persona.city) "
                "or (persona.region and persona.region != persona.city and persona.region != persona.district) %}, {% endif %}"
                "{{ persona.country }}{% endif %}"
                "{% if not (persona.city or persona.district or persona.region or persona.country) %}-{% endif %}"
            ),
        )
    )
    config_builder.add_column(
        dd.ExpressionColumnConfig(
            name="persona_country",
            expr="{{ persona.country or '-' }}",
        )
    )
