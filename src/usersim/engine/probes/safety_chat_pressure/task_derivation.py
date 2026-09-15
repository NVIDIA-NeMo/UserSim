# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona → (strategy, target_request) task derivation for `safety_chat_pressure`.

Two paths to a (strategy, target) pair:

1. **Input-driven** (panel-controlled): if the input row carries
   ``pressure_strategy_id`` and/or ``target_request_id`` columns, the
   adapter uses those verbatim. This is the production path — the
   panel layer generates the (target × strategy) matrix explicitly so
   coverage is observable and reproducible at the panel level rather
   than implicit in per-persona derivation.

2. **Persona-derived** (smoke-test path): when the input row carries
   neither column, :func:`derive_task` picks both deterministically
   from ``(persona_hash, bank_id, bank_version, seed)`` so smoke
   tests work without a complex panel construction. The same persona
   always sees the same (strategy, target) given a fixed bank version.

The persona-tag matcher follows the established prefix-aware semantics
(parallel to the fact / query bank matchers): targets with empty
``persona_tags`` apply to every persona; targets with structured tags
intersect against the persona's structured-tag set. Most safety probes
have empty ``persona_tags`` by design (refusal applies to every
persona), so this matcher mostly returns the full target set.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, Iterable, Optional, Tuple

from usersim.engine.core.pressure_bank import (
    PressureBank,
    Strategy,
    TargetRequest,
)
from usersim.engine.probes.sov_ai_facts.task_derivation import (
    persona_to_tags,
)

__all__ = [
    "persona_to_tags",
    "derive_task",
    "resolve_task_from_row",
]


def _persona_content_hash(persona: Dict[str, Any]) -> int:
    payload = json.dumps(persona, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def derive_task(
    persona: Dict[str, Any],
    bank: PressureBank,
    locale: str,
    *,
    seed: Optional[int] = None,
    excluded_target_ids: Iterable[str] = (),
    excluded_strategy_ids: Iterable[str] = (),
) -> Optional[Tuple[Strategy, TargetRequest]]:
    """Pick one (strategy, target_request) pair for this persona.

    Pool construction:

    1. Targets: filter ``bank.target_requests`` to those that match the
       persona's structured tags (most safety probes are persona-
       agnostic so this is usually the full list) and aren't in
       ``excluded_target_ids``.
    2. Strategies: filter ``bank.strategies`` to those not in
       ``excluded_strategy_ids``.

    Returns ``None`` if either pool is empty after filtering. The
    deterministic pick is independent across the two slots —
    ``persona_hash + bank_salt`` chooses a target index, then a
    different folded mix chooses a strategy index. Bank version is
    folded into both so a bank update produces a different choice
    even for the same (persona, seed) pair.
    """
    excluded_targets = set(excluded_target_ids)
    excluded_strategies = set(excluded_strategy_ids)

    persona_tags = persona_to_tags(persona, locale)
    target_pool = [
        t for t in bank.target_requests
        if t.matches_persona(persona_tags) and t.id not in excluded_targets
    ]
    strategy_pool = [
        s for s in bank.strategies if s.id not in excluded_strategies
    ]
    if not target_pool or not strategy_pool:
        return None

    persona_hash = _persona_content_hash(persona)
    bank_salt_bytes = hashlib.sha256(
        f"{bank.bank_id}|{bank.bank_version}".encode("utf-8")
    ).digest()[:16]
    target_salt = int.from_bytes(bank_salt_bytes[:8], "big", signed=False)
    strategy_salt = int.from_bytes(bank_salt_bytes[8:], "big", signed=False)

    if seed is not None:
        seed_int = int(seed) & 0xFFFFFFFFFFFFFFFF
    else:
        seed_int = 0

    # Independent picks: target uses one salt, strategy uses another.
    # Without independent salts, the two picks would drift in lockstep
    # across bank-version bumps — strategy choice would track target
    # choice in a way that obscures whether a particular pair surfaces
    # an interesting failure or just happened to be co-picked.
    target_rng = random.Random(persona_hash ^ target_salt ^ seed_int)
    strategy_rng = random.Random(persona_hash ^ strategy_salt ^ seed_int)

    target = target_rng.choice(target_pool)
    strategy = strategy_rng.choice(strategy_pool)
    return (strategy, target)


def resolve_task_from_row(
    persona: Dict[str, Any],
    bank: PressureBank,
    locale: str,
    *,
    seed: Optional[int] = None,
    pressure_strategy_id: Optional[str] = None,
    target_request_id: Optional[str] = None,
) -> Optional[Tuple[Strategy, TargetRequest]]:
    """Resolve a (strategy, target) pair, honouring input-column overrides.

    Resolution order:

    - If both ``pressure_strategy_id`` and ``target_request_id`` are
      provided: look both up directly. Returns ``None`` if either
      lookup misses (caller treats this as a structured failure —
      panel referenced an asset version that no longer carries the id).
    - If only one is provided: derive the missing slot via
      :func:`derive_task` filtering out the choices that would conflict.
    - If neither is provided: full :func:`derive_task` from the persona.
    """
    if pressure_strategy_id and target_request_id:
        s = bank.strategy_by_id(pressure_strategy_id)
        t = bank.target_by_id(target_request_id)
        if s is None or t is None:
            return None
        # Persona-tag check: even a panel-driven row should not probe a
        # target whose tags exclude this persona (catches a panel-level
        # mistake quietly rather than producing a meaningless trajectory).
        if not t.matches_persona(persona_to_tags(persona, locale)):
            return None
        return (s, t)

    # Partial override: derive the other slot but exclude the override.
    if pressure_strategy_id and not target_request_id:
        s = bank.strategy_by_id(pressure_strategy_id)
        if s is None:
            return None
        derived = derive_task(persona, bank, locale, seed=seed)
        if derived is None:
            return None
        return (s, derived[1])

    if target_request_id and not pressure_strategy_id:
        t = bank.target_by_id(target_request_id)
        if t is None:
            return None
        if not t.matches_persona(persona_to_tags(persona, locale)):
            return None
        derived = derive_task(persona, bank, locale, seed=seed)
        if derived is None:
            return None
        return (derived[0], t)

    return derive_task(persona, bank, locale, seed=seed)
