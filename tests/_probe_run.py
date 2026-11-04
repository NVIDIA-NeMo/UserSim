# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one episode of a probe class, as the generator does for a row, without Data Designer.

A probe that cannot be built for its row (a bank that does not load, no task
for the persona, a probe error) gives the ``SCENARIO_ABORTED`` result the
generator records, rather than raising.
"""

from __future__ import annotations

from typing import Any

from usersim.engine.core.outcomes import OutcomeBuilder, Provenance
from usersim.engine.core.probes import BankLoadError, dispatch_episode
from usersim.engine.generator import _make_aborted_result


async def run_probe(
    probe_cls: type,
    *,
    models: dict[str, Any],
    data: dict[str, Any],
    persona: dict[str, Any],
    profile: dict[str, Any],
    locale: str,
    language: str,
    cfg: Any,
    provenance: Provenance | None = None,
    outcome_builder: OutcomeBuilder | None = None,
) -> dict[str, Any]:
    """The result ``probe_cls`` produces for one row, its construction errors handled as the generator handles them."""
    provenance = provenance or Provenance()
    try:
        probe = probe_cls(
            persona=persona,
            locale=locale,
            language=language,
            models=models,
            cfg=cfg,
            provenance=provenance,
            profile=profile,
            data=data,
            outcome_builder=outcome_builder or OutcomeBuilder(provenance=provenance),
        )
    except (KeyError, ValueError, BankLoadError) as error:
        return _make_aborted_result(str(error), provenance=provenance)
    return await dispatch_episode(probe, models=models, data=data, cfg=cfg)
