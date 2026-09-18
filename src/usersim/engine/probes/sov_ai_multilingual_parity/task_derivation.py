# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona → query-bank task derivation for sov_ai_multilingual_parity.

Two helpers, structurally parallel to
:mod:`usersim.engine.probes.sov_ai_facts.task_derivation`:

- :func:`persona_to_tags` — re-exported from the sov_ai_facts probe
  module since the structured-tag vocabulary is the same across all
  sovereign-AI probes. Keeping a single source of truth means
  cross-probe joins on persona-feature axes work directly.
- :func:`derive_task` — picks one :class:`Query` from a loaded bank
  for this persona, **filtered to queries that have a rendering for
  this locale** (queries with the locale in ``renderings_opted_out``
  are skipped). The choice is deterministic given
  ``(persona content, bank_version, seed)``.

The matched-pair design intent: when running cross-locale, two
demographically similar personas (e.g., both ``age:25-44``,
``education:tertiary``, ``interest:health``) in different locales will
land on the same pool of queries. They will not always pick the *same*
``query_id`` deterministically — that depends on each persona's full
content hash — but the matched-pair downstream analysis joins on
``(query_id, demographic_bucket)`` and gets the cross-locale pair from
the *intersection* across many trajectories. Forcing identical picks
across locales would couple the persona-derivation logic to the
matched-pair join semantics in a brittle way.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, Iterable, List, Optional

from usersim.engine.core.query_bank import Query, QueryBank
from usersim.engine.probes.sov_ai_facts.task_derivation import (
    persona_to_tags,
)

__all__ = ["persona_to_tags", "derive_task"]


def _persona_content_hash(persona: Dict[str, Any]) -> int:
    """Deterministic int seed derived from persona content."""
    payload = json.dumps(persona, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def derive_task(
    persona: Dict[str, Any],
    query_bank: QueryBank,
    locale: str,
    *,
    seed: Optional[int] = None,
    excluded_query_ids: Iterable[str] = (),
) -> Optional[Query]:
    """Pick one query from the bank for this persona in this locale.

    Pool construction:

    1. Take the persona's structured tag set via :func:`persona_to_tags`
       (which uses the bank's ``locale`` field for region resolution —
       here we pass the runtime ``locale`` so e.g. a pt_BR persona's
       Brazilian region is folded in).
    2. Intersect with ``query_bank.matching_persona_tags(...)`` to keep
       only queries whose ``persona_tags`` are satisfiable.
    3. Filter to queries that have a rendering for this locale. Queries
       with the locale in their ``renderings_opted_out`` are dropped
       at this step, NOT at the bank-load step, so a single query bank
       can serve every locale even when a few entries don't translate.
    4. Drop ``excluded_query_ids`` (e.g., queries already probed for
       this persona in a prior trajectory).

    Returns ``None`` when the pool is empty after filtering. The caller
    should log + emit a structured failure (the dispatcher does).

    Determinism: ``(persona_hash, bank_id, bank_version, seed)`` fully
    determines the pick. Bank version is folded into the seed via SHA-
    256 so a bank update produces a different choice for the same
    persona — without this, upgrading a bank would silently re-pick the
    same query.
    """
    excluded = set(excluded_query_ids)

    persona_tags = persona_to_tags(persona, locale)
    candidates: List[Query] = [
        q for q in query_bank.matching_persona_tags(persona_tags) if q.supports_locale(locale) and q.id not in excluded
    ]
    if not candidates:
        return None

    persona_hash = _persona_content_hash(persona)
    bank_salt_bytes = hashlib.sha256(f"{query_bank.bank_id}|{query_bank.bank_version}".encode("utf-8")).digest()[:8]
    bank_salt = int.from_bytes(bank_salt_bytes, "big", signed=False)
    mix = persona_hash ^ bank_salt
    if seed is not None:
        mix ^= int(seed) & 0xFFFFFFFFFFFFFFFF

    rng = random.Random(mix)
    return rng.choice(candidates)
