# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona → fact-bank task derivation.

:func:`derive_task` picks, for a persona and a loaded bank, a single
:class:`Fact` whose ``persona_tags`` intersect the tags
:func:`~usersim.engine.core.persona_tags.persona_to_tags` gives the persona.
It is deterministic given ``(persona content, bank_version, seed)``.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Iterable

from usersim.engine.core.persona_tags import persona_to_tags
from usersim.engine.probes.sov_ai_facts.bank import Fact, FactBank


def _persona_content_hash(persona: dict[str, Any]) -> int:
    """Deterministic int seed derived from persona content.

    Uses the same canonicalization style as ``core.identity``: JSON with
    sorted keys so the hash is stable across Python sessions.
    """
    payload = json.dumps(persona, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def derive_task(
    persona: dict[str, Any],
    fact_bank: FactBank,
    *,
    seed: int | None = None,
    excluded_fact_ids: Iterable[str] = (),
) -> Fact | None:
    """Pick one fact from the bank for this persona.

    The pool is ``fact_bank.matching_persona_tags(persona_to_tags(...))``;
    ``excluded_fact_ids`` (e.g., facts already probed in a prior
    trajectory for the same persona) are removed from the pool. The
    choice is deterministic in ``(persona content, bank_version, seed)``:

    - If ``seed`` is provided, we mix it into the persona-derived
      content hash so the same persona can produce a different fact
      across runs (for canary / matched-pair cohorts) while remaining
      reproducible given the same seed.
    - If ``seed`` is ``None``, the fact is fully persona-determined —
      the same persona always gets the same fact.

    Returns ``None`` if no matching fact remains after exclusion
    (the caller should log + emit a structured failure).
    """
    excluded = set(excluded_fact_ids)

    persona_tags = persona_to_tags(persona, fact_bank.locale)
    candidates = [f for f in fact_bank.matching_persona_tags(persona_tags) if f.id not in excluded]
    if not candidates:
        return None

    persona_hash = _persona_content_hash(persona)
    # Fold the bank version into the seed so a bank update produces a
    # different choice even for the same (persona, seed) pair. Without
    # this, upgrading a bank would silently re-pick the same old fact
    # for the same persona — which is usually *not* what you want.
    # Using hashlib (content-based) keeps the salt stable across
    # Python sessions; built-in ``hash()`` is salted per-process.
    bank_salt_bytes = hashlib.sha256(f"{fact_bank.bank_id}|{fact_bank.bank_version}".encode("utf-8")).digest()[:8]
    bank_salt = int.from_bytes(bank_salt_bytes, "big", signed=False)
    mix = persona_hash ^ bank_salt
    if seed is not None:
        mix ^= int(seed) & 0xFFFFFFFFFFFFFFFF

    rng = random.Random(mix)
    return rng.choice(candidates)
