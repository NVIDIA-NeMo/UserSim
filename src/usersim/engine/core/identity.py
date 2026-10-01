# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable identity for personas and trajectories — the reproducibility spine.

Two derived IDs anchor every replay / matched-pair / canary / idempotent
re-run in this framework:

- ``persona_uuid`` is the SHA-256 (16 hex chars) of the persona's
  canonical JSON form. **It is observed, not assigned.** Two runs that
  happen to draw the same persona (whether via DD's live ``PersonSampler``
  or via reload from a prior trajectory parquet) get the same UUID.
- ``trajectory_id`` is the SHA-256 (16 hex chars) of all the inputs
  that determine what a row's content depends on: persona, probe
  family + variant, ``scenario_seed`` (the per-row seed input column),
  the resolved model identities for each role, and the probe's
  ``PROMPT_VERSION``. Deterministic given the same inputs — so the
  simulator can dedup on it cheaply.

See the design plan §2.6 ("Reproducibility spine") and the roadmap
load-bearing-decisions section for the rationale: dynamic persona
sampling is preserved (not replaced by a fixed panel); identity is
content-derived; replay is a special case of "load a trajectory parquet
as a seed."
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

# Persona attributes that participate in the canonical hash. We
# explicitly enumerate them rather than hashing the whole persona dict
# so that:
#   - Provider-assigned transient IDs (DD's internal `_id`, ephemeral
#     uuid columns, etc) never leak into the hash.
#   - Adding a new field to Nemotron-Personas does not silently change
#     every trajectory_id (which would defeat the replay guarantee).
#   - The same persona drawn at different times — same content, no
#     transient noise — yields the same UUID by construction.
#
# The list reflects the PGM-grounded fields shipped today across
# en_US / pt_BR / ja_JP / en_IN / hi_Deva_IN / fr_FR. Locale-specific
# region keys (state, prefecture, departement, region) are handled
# below by always projecting whichever ones are present.
_CANONICAL_PERSONA_KEYS: tuple[str, ...] = (
    "first_name",
    "middle_name",
    "last_name",
    "sex",
    "age",
    "marital_status",
    "education_level",
    "bachelors_field",
    "occupation",
    "country",
    "city",
    "commune",
    "municipality",
    "district",
    "state",
    "prefecture",
    "departement",
    "region",
    "postcode",
    # India-dataset fields now rendered into the user-agent prompt.
    # They are content-determining inputs, so omitting them would let two
    # materially different simulated users share one persona_uuid.
    "religion",
    "first_language",
    "second_language",
    "third_language",
    "religious_background",
    "linguistic_background",
    "cultural_background",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
    "persona",
    "detailed_persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


def _canonical_value(v: Any) -> Any:
    """Recursively normalize a persona attribute value for hashing.

    OCEAN traits arrive either as dicts (en_US) or as JSON strings
    (other locales) per ``core/behavioral.py``; we parse strings that
    look like JSON so the hash is locale-agnostic. Nested dicts and
    lists are normalized to sorted keys / preserved order.
    """
    if isinstance(v, str):
        s = v.strip()
        if s and s[0] in "{[" and s[-1] in "}]":
            try:
                return _canonical_value(json.loads(s))
            except (json.JSONDecodeError, TypeError):
                return v
        return v
    if isinstance(v, dict):
        return {k: _canonical_value(v[k]) for k in sorted(v.keys())}
    if isinstance(v, (list, tuple)):
        return [_canonical_value(x) for x in v]
    return v


def canonical_persona_json(persona: dict[str, Any]) -> str:
    """Build a canonical JSON string of the persona's hash-relevant attributes.

    Only fields in ``_CANONICAL_PERSONA_KEYS`` that are actually present
    on this persona contribute to the canonical form. Missing fields are
    omitted (not represented as ``null``) so that adding a new field to
    a future Nemotron-Personas release does not retroactively change the
    UUIDs of personas that lack the new field.
    """
    canonical: dict[str, Any] = {}
    for key in _CANONICAL_PERSONA_KEYS:
        if key in persona and persona[key] is not None:
            canonical[key] = _canonical_value(persona[key])
    return json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)


def persona_uuid(persona: dict[str, Any]) -> str:
    """Content-hashed 16-char persona identifier.

    Stable across runs given identical content. Replay-safe: a
    persona snapshot reloaded from a prior trajectory parquet
    produces the same UUID under unchanged content.
    """
    canonical = canonical_persona_json(persona)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def resolve_model_name(facade: Any, alias: str) -> str:
    """Best-effort resolution of a model facade to a stable identity string.

    Used as input to ``trajectory_id`` so model-version drift produces
    a different ``trajectory_id`` (which is exactly what we want — a
    replay against a different model is a different trajectory by
    definition). Falls back to the alias name if the facade does not
    expose its underlying model identity.

    Tries, in order: ``facade.model_config.model``, ``facade.model_name``,
    ``facade.config.name``, then the alias itself.
    """
    if facade is None:
        return alias
    for path in (
        ("model_config", "model"),
        ("model_name",),
        ("config", "name"),
        ("config", "model"),
    ):
        cur: Any = facade
        ok = True
        for attr in path:
            cur = getattr(cur, attr, None)
            if cur is None:
                ok = False
                break
        if ok and isinstance(cur, str) and cur:
            return cur
    return alias


def trajectory_id(
    *,
    persona_uuid: str,
    probe_family: str,
    probe_variant: str,
    scenario_seed: int | str | None,
    user_model: str,
    assistant_model: str,
    prompt_version: str,
    extra_keys: Iterable[tuple[str, str]] | None = None,
) -> str:
    """Deterministic 16-char trajectory identifier.

    ``trajectory_id = sha256(persona_uuid | probe_family | probe_variant
                             | scenario_seed | user_model | assistant_model
                             | prompt_version | *extra_keys)[:16]``

    ``extra_keys`` is an optional iterable of ``(name, value)`` pairs for
    probe-specific salting (e.g., for the ``safety_chat_pressure`` probe
    we may include the `pressure_strategy_seed` to discriminate cohorts
    that share persona + probe_family + variant).
    """
    parts: list[str] = [
        persona_uuid,
        probe_family,
        probe_variant,
        "" if scenario_seed is None else str(scenario_seed),
        user_model,
        assistant_model,
        prompt_version,
    ]
    if extra_keys:
        for k, v in extra_keys:
            parts.append(f"{k}={v}")
    composite = "|".join(parts)
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]
