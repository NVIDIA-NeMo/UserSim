# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported model-free episode materialization, hosted-runtime and configured-model API."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from usersim.engine.core.episode_input import (
    ConstructedEpisode,
    EpisodePreamble,
    construct_episode_preamble,
    construct_probe_episode,
)
from usersim.engine.core.episode_runtime import (
    ActivationRequest,
    ActivationResult,
    ActivationUsage,
    EpisodeContractError,
    EpisodeLifecycleComplete,
    ExecutedToolCall,
    HostRoleModel,
    ProbeEpisodeRuntime,
    ProbeRuntimeDescriptor,
)

logger = logging.getLogger("usersim.engine")

__all__ = [
    "ActivationRequest",
    "ActivationResult",
    "ActivationUsage",
    "ConstructedEpisode",
    "EpisodeContractError",
    "EpisodeLifecycleComplete",
    "EpisodePreamble",
    "ExecutedToolCall",
    "HostRoleModel",
    "ProbeEpisodeRuntime",
    "ProbeRuntimeDescriptor",
    "construct_episode_preamble",
    "construct_probe_episode",
    "materialize_episode_inputs",
    "model_facades",
    "replays_assistant_reasoning",
    "simulate_episode_inputs",
]


def materialize_episode_inputs(
    *,
    locale: str,
    num_rows: int,
    probe_mix: dict[str, float],
    random_seed: int | None = None,
    models_path: Path | None = None,
    assets_dir: Path | None = None,
    toolset_seed_path: Path | None = None,
    match_persona_language: bool = False,
    max_turns: int = 5,
    max_assistant_attempts: int = 1,
    store_reasoning: bool = True,
    finance_tier_mix: float = 0.0,
    finance_retrieval_mode: str = "hybrid",
) -> list[dict[str, object]]:
    """Materialize canonical, fully resolved episode rows without model calls.

    Model configuration is read only for stable trajectory identities and Data
    Designer configuration; this function never invokes a configured model.
    """
    from usersim.cli._models import default_models_path, load_models_config
    from usersim.cli._pipeline import materialize_episode_inputs as _materialize
    from usersim.engine.core._assets import default_assets_dir

    return _materialize(
        locale=locale,
        num_rows=num_rows,
        models=load_models_config(models_path or default_models_path()),
        assets_dir=(assets_dir or default_assets_dir()).resolve(),
        probe_mix=probe_mix,
        random_seed=random_seed,
        toolset_seed_path=toolset_seed_path,
        match_persona_language=match_persona_language,
        max_turns=max_turns,
        max_assistant_attempts=max_assistant_attempts,
        store_reasoning=store_reasoning,
        finance_tier_mix=finance_tier_mix,
        finance_retrieval_mode=finance_retrieval_mode,
    )


def simulate_episode_inputs(
    rows: list[dict[str, object]],
    *,
    out: Path,
    models_path: Path | None = None,
    assets_dir: Path | None = None,
    skip_existing: bool = True,
):
    """Run stored rows with the configured models and save them as a run under ``out``.

    The library form of ``usersim simulate --inputs``. ``rows`` are what
    :func:`materialize_episode_inputs` returns, or the same rows read back
    from storage. Each row runs exactly as stored, and is saved where
    ``usersim eval`` reads it, with a ``trajectory_id`` of its own and the
    stored id as ``input_id``.

    Each setup (models, settings, assets) gets its own run under ``out``.
    Calling again with the same setup resumes that run and skips the rows it
    already holds, so a caller can run a large set in batches (a stored field
    that two batches save differently does not read back exactly; see
    "Running stored rows" in ``docs/engine/EXTERNAL_PROBE_RUNTIME.md``). Raises
    ``ValueError`` for rows that cannot run, before any model call, and for a
    stored row that the run already holds with different contents: before any
    model call when it is found up front, and when the rows are saved when
    another writer saved it meanwhile, where the write lock is shared. Where
    ``out`` is on a filesystem without file locks, the rows are written
    without a lock shared between processes, with a warning. Two calls at
    once with one setup can then each start a run, and resuming one run from
    two places at once can save a row twice, or both versions of a changed
    row. Returns an ``InputsRun`` with the run id and the number of rows
    written and skipped.
    """
    from usersim.cli._inputs import prepare_episode_inputs, simulate_inputs_to_dataset
    from usersim.cli._models import REQUIRED_SIMULATOR_ALIASES, default_models_path, load_models_config, require_aliases

    path = models_path or default_models_path()
    models = load_models_config(path)
    require_aliases(models, required=REQUIRED_SIMULATOR_ALIASES, context="simulate_episode_inputs")
    return simulate_inputs_to_dataset(
        inputs=prepare_episode_inputs(rows, assets_dir=assets_dir),
        models=models,
        out=out,
        models_path=path,
        skip_existing=skip_existing,
    )


def model_facades(models_path: Path | None = None, *, aliases: Sequence[str] | None = None) -> dict[str, Any]:
    """The models a models file configures, as facades a probe can run on outside a Data Designer pipeline.

    For running episodes yourself: pass the result to
    ``ConversationSimulatorGenerator.with_models`` or ``ProbeEpisodeRuntime``.
    Providers, keys and each alias's inference settings (``max_tokens``,
    ``extra_body``, the catalog defaults) are resolved as ``usersim simulate``
    resolves them, so a request matches the one a run would send. An unset key's
    environment variable is logged as a warning for the whole file, as a run
    checks it, whichever ``aliases`` are built.
    The facades use async clients, as a run's pipeline does. Building them
    calls no model, checks no endpoint and writes nothing under the working
    directory, and the caller's logging configuration is left as it is. Data
    Designer itself seeds its default settings under ``DATA_DESIGNER_HOME``
    (``~/.data-designer``) the first time, as for any Data Designer run.

    ``aliases`` limits the result to those aliases; by default every alias in
    the file is included. Raises ``ConfigError`` for an alias the file does not
    declare.
    """
    import data_designer.config as dd
    from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
    from data_designer.interface import DataDesigner

    from usersim.cli._errors import ConfigError
    from usersim.cli._models import (
        default_models_path,
        load_models_config,
        to_data_designer_kwargs,
        to_model_configs,
        warn_if_missing_api_key,
    )

    models = load_models_config(models_path or default_models_path())
    declared = list(models.aliases())
    wanted = list(aliases) if aliases is not None else declared
    unknown = sorted(set(wanted) - set(declared))
    if unknown:
        raise ConfigError(f"models config declares no {unknown}; declared aliases: {sorted(declared)}")
    if (msg := warn_if_missing_api_key(models)) is not None:
        logger.warning(msg)
    builder = dd.DataDesignerConfigBuilder(model_configs=to_model_configs(models))
    # Data Designer's public ``get_models`` builds only its default models, with
    # sync clients; the resource provider is how a pipeline builds a configured
    # set. tests/engine/test_model_facades.py calls it on every build and checks
    # the clients are async, so a Data Designer change to either fails there.
    # The provider's artifact root and managed-assets path go to a temporary
    # folder: the models use neither, the first would otherwise be created as
    # ./artifacts, and the person reader built from the second raises when that
    # path does not exist. Data Designer is told to leave logging alone, so the
    # caller's handlers survive.
    with tempfile.TemporaryDirectory(prefix="usersim-facades-") as scratch:
        designer = DataDesigner(
            scratch, managed_assets_path=scratch, auto_configure_logging=False, **to_data_designer_kwargs(models)
        )
        provider = designer._create_resource_provider(
            "usersim-facades", builder, client_concurrency_mode=ClientConcurrencyMode.ASYNC
        )
        registry = provider.model_registry
        return {alias: registry.get_model(model_alias=alias) for alias in wanted}


def replays_assistant_reasoning(models_path: Path | None = None, *, store_reasoning: bool = True) -> bool:
    """Whether a models file turns on replay of the assistant's own reasoning.

    Set ``ConversationSimulatorConfig.replay_assistant_reasoning`` to this when
    running on :func:`model_facades` from the same file, so a run outside a
    pipeline replays as ``usersim simulate`` would. Raises ``ConfigError`` for
    a setup that cannot replay (the switch on another alias, reasoning not
    stored, or a provider that drops reasoning).
    """
    from usersim.cli._models import assistant_replays_reasoning, default_models_path, load_models_config

    models = load_models_config(models_path or default_models_path())
    return assistant_replays_reasoning(models, store_reasoning=store_reasoning)
