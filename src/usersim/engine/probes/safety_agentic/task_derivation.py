# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona → ActionRequest task derivation for the safety_agentic probe.

Two paths to an :class:`ActionRequest`, mirroring the
``safety_chat_pressure`` probe's design:

1. **Input-driven** (panel-controlled): if the input row carries an
   ``action_request_id`` column, the adapter looks it up directly.
   This is the production path — the panel layer constructs the
   (sub_protocol × action_taxonomy.category) matrix explicitly so
   coverage is observable at the panel level rather than implicit in
   per-persona derivation.

2. **Persona-derived** (smoke-test path): when the column is absent,
   :func:`derive_task` picks an ``ActionRequest`` deterministically
   from ``(persona_hash, bank_id, bank_version, seed)`` so smoke
   tests work without a complex panel construction.

The persona-tag matcher follows the established prefix-aware
semantics. Most agentic-safety probes have empty ``persona_tags`` by
design (refusal-to-overreach is persona-agnostic), so this matcher
mostly returns the full request set.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, Iterable, Optional

from usersim.engine.core.agentic_bank import ActionRequest, AgenticBank
from usersim.engine.probes.sov_ai_facts.task_derivation import (
    persona_to_tags,
)

__all__ = ["persona_to_tags", "derive_task", "resolve_task_from_row"]


def _persona_content_hash(persona: Dict[str, Any]) -> int:
    payload = json.dumps(persona, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def derive_task(
    persona: Dict[str, Any],
    bank: AgenticBank,
    locale: str,
    *,
    seed: Optional[int] = None,
    excluded_request_ids: Iterable[str] = (),
) -> Optional[ActionRequest]:
    """Pick one :class:`ActionRequest` for this persona.

    Pool: ``bank.for_persona(persona_to_tags(persona, locale))`` minus
    ``excluded_request_ids``. Returns ``None`` when the pool is empty
    (caller treats this as a structured failure).

    Determinism: ``(persona_hash, bank_id, bank_version, seed)`` fully
    determines the pick. Bank version is folded into the seed via
    SHA-256 so a bank update produces a different choice for the same
    persona — without this, upgrading a bank would silently re-pick
    the same request.
    """
    excluded = set(excluded_request_ids)
    persona_tags = persona_to_tags(persona, locale)
    pool = [r for r in bank.for_persona(persona_tags) if r.id not in excluded]
    if not pool:
        return None

    persona_hash = _persona_content_hash(persona)
    bank_salt_bytes = hashlib.sha256(f"{bank.bank_id}|{bank.bank_version}".encode("utf-8")).digest()[:8]
    bank_salt = int.from_bytes(bank_salt_bytes, "big", signed=False)
    mix = persona_hash ^ bank_salt
    if seed is not None:
        mix ^= int(seed) & 0xFFFFFFFFFFFFFFFF

    rng = random.Random(mix)
    return rng.choice(pool)


def resolve_task_from_row(
    persona: Dict[str, Any],
    bank: AgenticBank,
    locale: str,
    *,
    seed: Optional[int] = None,
    action_request_id: Optional[str] = None,
) -> Optional[ActionRequest]:
    """Resolve an action request, honouring the optional input override.

    - If ``action_request_id`` is provided: look it up directly. Returns
      ``None`` if the lookup misses (caller treats this as a structured
      failure — the panel referenced an asset version that no longer
      carries the id) OR if the request's persona_tags exclude this
      persona (catches a panel-level mistake quietly).
    - Otherwise: derive via :func:`derive_task`.
    """
    if action_request_id:
        ar = bank.by_id(action_request_id)
        if ar is None:
            return None
        if not ar.matches_persona(persona_to_tags(persona, locale)):
            return None
        return ar
    return derive_task(persona, bank, locale, seed=seed)
