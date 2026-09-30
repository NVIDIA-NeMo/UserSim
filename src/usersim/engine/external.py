# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

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
    EpisodeLifecycleComplete,
    ProbeEpisodeRuntime,
)

__all__ = [
    "ActivationRequest",
    "ActivationResult",
    "ConstructedEpisode",
    "EpisodeLifecycleComplete",
    "EpisodePreamble",
    "ProbeEpisodeRuntime",
    "construct_episode_preamble",
    "construct_probe_episode",
    "materialize_episode_inputs",
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
