# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported model-free episode materialization and hosted-runtime API."""

from __future__ import annotations

from pathlib import Path

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
    already holds, so a caller can run a large set in batches. Raises
    ``ValueError`` for rows that cannot run, before any model call, and for a
    stored row that the run already holds with different contents: before any
    model call when it is found up front, and when the rows are saved when
    another writer saved it meanwhile. Returns an ``InputsRun`` with the run
    id and the number of rows written and skipped.
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
