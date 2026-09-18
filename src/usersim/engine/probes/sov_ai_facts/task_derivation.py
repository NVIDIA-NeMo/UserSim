# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona → fact-bank task derivation.

Two helpers:

- :func:`persona_to_tags` — maps a Nemotron-Personas persona (region,
  age, occupation, education, topical interests inferred from
  occupation) to the structured tag vocabulary the fact bank uses
  (``<prefix>:<value>`` strings).
- :func:`derive_task` — given a persona and a loaded bank, picks a
  single :class:`Fact` whose ``persona_tags`` intersect the persona's.
  Deterministic given ``(persona content, bank_version, seed)``.

Each supported locale has its own mapping block (Brazilian regions,
Indian states, Japanese prefectures, …); unknown locales get a
locale-agnostic fallback that only produces interest-level tags.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Set

from usersim.engine.core.fact_bank import Fact, FactBank

# ---------------------------------------------------------------------------
# Brazilian region classifier
# ---------------------------------------------------------------------------

# Map each Brazilian state (uppercase 2-letter code OR full name, both
# lowercased for matching) to the IBGE macro-region. Reference:
# https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial.html
_BR_STATE_TO_REGION: Dict[str, str] = {
    # Norte
    "am": "north",
    "amazonas": "north",
    "pa": "north",
    "pará": "north",
    "para": "north",
    "ac": "north",
    "acre": "north",
    "ro": "north",
    "rondônia": "north",
    "rondonia": "north",
    "rr": "north",
    "roraima": "north",
    "ap": "north",
    "amapá": "north",
    "amapa": "north",
    "to": "north",
    "tocantins": "north",
    # Nordeste
    "ba": "northeast",
    "bahia": "northeast",
    "pe": "northeast",
    "pernambuco": "northeast",
    "ce": "northeast",
    "ceará": "northeast",
    "ceara": "northeast",
    "ma": "northeast",
    "maranhão": "northeast",
    "maranhao": "northeast",
    "pb": "northeast",
    "paraíba": "northeast",
    "paraiba": "northeast",
    "rn": "northeast",
    "rio grande do norte": "northeast",
    "al": "northeast",
    "alagoas": "northeast",
    "se": "northeast",
    "sergipe": "northeast",
    "pi": "northeast",
    "piauí": "northeast",
    "piaui": "northeast",
    # Centro-Oeste
    "go": "central-west",
    "goiás": "central-west",
    "goias": "central-west",
    "mt": "central-west",
    "mato grosso": "central-west",
    "ms": "central-west",
    "mato grosso do sul": "central-west",
    "df": "central-west",
    "distrito federal": "central-west",
    # Sudeste
    "sp": "southeast",
    "são paulo": "southeast",
    "sao paulo": "southeast",
    "rj": "southeast",
    "rio de janeiro": "southeast",
    "mg": "southeast",
    "minas gerais": "southeast",
    "es": "southeast",
    "espírito santo": "southeast",
    "espirito santo": "southeast",
    # Sul
    "rs": "south",
    "rio grande do sul": "south",
    "sc": "south",
    "santa catarina": "south",
    "pr": "south",
    "paraná": "south",
    "parana": "south",
}


def _br_region_from_persona(persona: Dict[str, Any]) -> Optional[str]:
    """Resolve a Brazilian macro-region tag from persona state fields."""
    # Nemotron-Personas-Brazil carries a ``state`` or ``state_abbrev``
    # field. Fall back to ``region`` if present.
    for key in ("state_abbrev", "state", "region"):
        raw = persona.get(key)
        if isinstance(raw, str) and raw.strip():
            code = raw.strip().lower()
            mapped = _BR_STATE_TO_REGION.get(code)
            if mapped:
                return mapped
    return None


# ---------------------------------------------------------------------------
# Age bin → tag
# ---------------------------------------------------------------------------

_AGE_BINS: List[tuple] = [
    (0, 17, "age:under-18"),
    (18, 24, "age:18-24"),
    (25, 44, "age:25-44"),
    (45, 64, "age:45-64"),
    (65, 200, "age:65+"),
]


def _age_tag(persona: Dict[str, Any]) -> Optional[str]:
    """Extract an age-bin tag from a persona's numeric age if possible."""
    raw = persona.get("age")
    try:
        age = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    for lo, hi, tag in _AGE_BINS:
        if lo <= age <= hi:
            return tag
    return None


# ---------------------------------------------------------------------------
# Education level → tag
# ---------------------------------------------------------------------------

_EDUCATION_MAP: Dict[str, str] = {
    # Map common Nemotron-Personas education_level values to our coarse
    # tag vocabulary. Keys are lowercased substring matches.
    "no schooling": "primary",
    "primary": "primary",
    "elementary": "primary",
    "ensino fundamental": "primary",
    "middle": "secondary",
    "high school": "secondary",
    "ensino médio": "secondary",
    "ensino medio": "secondary",
    "associate": "secondary",
    "bachelor": "tertiary",
    "bachelors": "tertiary",
    "ensino superior": "tertiary",
    "graduação": "tertiary",
    "graduacao": "tertiary",
    "master": "tertiary",
    "doctor": "tertiary",
    "phd": "tertiary",
    "postgraduate": "tertiary",
    "pós-graduação": "tertiary",
    "pos-graduacao": "tertiary",
}


def _education_tag(persona: Dict[str, Any]) -> Optional[str]:
    raw = persona.get("education_level")
    if not isinstance(raw, str):
        return None
    needle = raw.strip().lower()
    # Longest-match first so "masters degree" maps to tertiary via "master".
    for key in sorted(_EDUCATION_MAP, key=len, reverse=True):
        if key in needle:
            return f"education:{_EDUCATION_MAP[key]}"
    return None


# ---------------------------------------------------------------------------
# Occupation → occupation-family + interest inference
# ---------------------------------------------------------------------------

# Occupation phrases (lowercased substring match) → (family, interests).
# Very coarse on purpose: the point is to seed interest tags, not to build a
# complete occupation taxonomy. Reviewers can extend this block per locale.
_OCCUPATION_MAP: List[tuple] = [
    # (substring, occupation-family, extra interests)
    ("teacher", "teacher", ("interest:literature", "interest:history")),
    ("professor", "teacher", ("interest:literature",)),
    ("docente", "teacher", ("interest:literature",)),
    ("student", "student", ("interest:literature",)),
    ("estudante", "student", ("interest:literature",)),
    ("nurse", "healthcare", ("interest:health",)),
    ("enfermeir", "healthcare", ("interest:health",)),
    ("doctor", "healthcare", ("interest:health",)),
    ("médico", "healthcare", ("interest:health",)),
    ("medico", "healthcare", ("interest:health",)),
    ("entrepreneur", "small-business", ()),
    ("empresári", "small-business", ()),
    ("empreendedor", "small-business", ()),
    ("self-employed", "small-business", ()),
    ("shopkeeper", "small-business", ()),
    ("comerciante", "small-business", ()),
    ("engineer", "engineering", ("interest:geography",)),
    ("engenheir", "engineering", ("interest:geography",)),
    ("farmer", "agriculture", ("interest:geography",)),
    ("agricultor", "agriculture", ("interest:geography",)),
    ("journalist", "media", ("interest:history", "interest:literature")),
    ("jornalista", "media", ("interest:history", "interest:literature")),
    ("musician", "arts", ("interest:music",)),
    ("músic", "arts", ("interest:music",)),
    ("artist", "arts", ("interest:music",)),
    ("artista", "arts", ("interest:music",)),
    ("writer", "arts", ("interest:literature",)),
    ("escritor", "arts", ("interest:literature",)),
    ("retired", "retired", ()),
    ("aposentad", "retired", ()),
    ("homemaker", "homemaker", ()),
    ("housewife", "homemaker", ()),
    ("dona de casa", "homemaker", ()),
]


def _occupation_tags(persona: Dict[str, Any]) -> List[str]:
    """Best-effort occupation-family + derived-interest tags."""
    raw = persona.get("occupation")
    if not isinstance(raw, str):
        return []
    needle = raw.strip().lower()
    if not needle:
        return []
    # First-match wins (list is ordered roughly most-specific-first).
    for key, family, interests in _OCCUPATION_MAP:
        if key in needle:
            tags: List[str] = [f"occupation-family:{family}"]
            tags.extend(interests)
            return tags
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def persona_to_tags(persona: Dict[str, Any], locale: str) -> List[str]:
    """Build the persona-tag set the matcher intersects with fact tags.

    Always includes ``region:any``, ``age:any``, and ``interest:geography``
    / ``interest:history`` as broad fallbacks so every persona has *some*
    matching surface even when demographic fields are missing. The
    locale-specific block (e.g., Brazilian macro-regions for ``pt_BR``)
    adds precise tags on top.
    """
    tags: Set[str] = {
        "region:any",
        "age:any",
        # Broad interests everyone shares to some degree — ensures the
        # matcher never returns an empty set against a well-populated
        # bank just because the persona had no occupation.
        "interest:geography",
        "interest:history",
    }

    # Locale-specific region resolution.
    if locale == "pt_BR":
        region = _br_region_from_persona(persona)
        if region is not None:
            tags.add(f"region:{region}")

    # Age bin.
    age_tag = _age_tag(persona)
    if age_tag is not None:
        tags.add(age_tag)

    # Education level.
    edu_tag = _education_tag(persona)
    if edu_tag is not None:
        tags.add(edu_tag)

    # Occupation-family + derived interests.
    for t in _occupation_tags(persona):
        tags.add(t)

    return sorted(tags)


def _persona_content_hash(persona: Dict[str, Any]) -> int:
    """Deterministic int seed derived from persona content.

    Uses the same canonicalization style as ``core.identity``: JSON with
    sorted keys so the hash is stable across Python sessions.
    """
    payload = json.dumps(persona, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def derive_task(
    persona: Dict[str, Any],
    fact_bank: FactBank,
    *,
    seed: Optional[int] = None,
    excluded_fact_ids: Iterable[str] = (),
) -> Optional[Fact]:
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
