# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probing-taxonomy loader + validator + lookup helpers for ``sov_ai_dynamic``.

A probing taxonomy is a curated, per-locale YAML file used by the
``sov_ai_dynamic`` probe to seed the user-agent's opening turn with a
category-bound, persona-driven probe. It carries **no ground truth** —
that's the whole point of dynamic probing — just per-category invitation
prose and sub-topic hints used for collapse avoidance via per-trajectory
rotation.

Schema specified in ``assets/sov_ai_dynamic/SCHEMA.md``. This module
exposes:

- :class:`Category` — typed view of a single category entry.
- :class:`ProbingTaxonomy` — typed view of the whole taxonomy + lookup
  helpers.
- :class:`ProbingTaxonomyError` — raised for schema violations.
- :func:`load_probing_taxonomy` — parse + validate a single file.
- :func:`load_probing_taxonomy_for_locale` / :func:`reset_probing_taxonomy_cache`
  — locale-keyed shared cache so the probe and the ``sov_ai_dynamic``
  scorer agree on ``taxonomy_version`` from one source of truth
  (same pattern as ``core.fact_bank``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

logger = logging.getLogger("usersim.engine")

SUPPORTED_SCHEMA_VERSIONS: FrozenSet[str] = frozenset({"v0.1"})
MIN_SUBTOPIC_HINTS: int = 2  # rotation requires at least two distinct hints


class ProbingTaxonomyError(ValueError):
    """User-facing schema or load problem.

    Message format: ``"<path>::<category id or field>: <what's wrong>"``
    so reviewers can grep the offending location.
    """


# ---------------------------------------------------------------------------
# Typed views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Category:
    id: str
    invitation: str
    subtopic_hints: Tuple[str, ...]
    # Optional entity-TYPE scoping (empty = applies to every type). Used by
    # GROUNDED dynamic-tier probes (e.g. financial_services scopes categories
    # to the institution type); ungrounded population probing (sov_ai_dynamic)
    # leaves it empty so every category is universal.
    applies_to_types: Tuple[str, ...] = ()
    # Optional persona-tag affinities for SOFT-WEIGHTING selection toward
    # ecologically-valid topics (e.g. ``age:55-64`` / ``financial-literacy:low``).
    # A soft prior only: consumers boost, never zero out, so coverage holds.
    # Empty (the sov_ai_dynamic case) => uniform selection.
    persona_affinities: Tuple[str, ...] = ()
    # Set by the loader from the taxonomy's top-level fields so
    # downstream consumers can carry provenance per-category.
    locale: str = ""
    taxonomy_id: str = ""
    taxonomy_version: str = ""

    def applies_to(self, entity_type: str) -> bool:
        return not self.applies_to_types or entity_type in self.applies_to_types


@dataclass(frozen=True)
class ProbingTaxonomy:
    schema_version: str
    locale: str
    taxonomy_id: str
    taxonomy_version: str
    placeholder: bool = False
    categories: Tuple[Category, ...] = field(default_factory=tuple)
    source_path: Optional[str] = None

    # ── Lookups ─────────────────────────────────────────────────────

    def by_id(self, category_id: str) -> Optional[Category]:
        for c in self.categories:
            if c.id == category_id:
                return c
        return None

    def category_ids(self) -> Tuple[str, ...]:
        return tuple(c.id for c in self.categories)

    def categories_for_type(self, entity_type: str) -> Tuple[Category, ...]:
        """Categories applicable to ``entity_type`` (the runtime scope filter).

        Categories with an empty ``applies_to_types`` are universal (the
        sov_ai_dynamic case), so this returns everything when no category is
        type-scoped.
        """
        return tuple(c for c in self.categories if c.applies_to(entity_type))

    def provenance_summary(self) -> Dict[str, Any]:
        """Surfaces that the downstream reporting layer reads."""
        return {
            "taxonomy_id": self.taxonomy_id,
            "taxonomy_version": self.taxonomy_version,
            "locale": self.locale,
            "placeholder": self.placeholder,
            "n_categories": len(self.categories),
            "category_ids": list(self.category_ids()),
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_probing_taxonomy(path: str | Path) -> ProbingTaxonomy:
    """Parse and validate a probing taxonomy from disk.

    Raises :class:`ProbingTaxonomyError` with a message that names the
    offending file, category id (or field), and the invariant that
    was violated.
    """
    import yaml  # lazy: kept off the plugin's top-level import graph

    src_path = Path(path)
    if not src_path.exists():
        raise ProbingTaxonomyError(f"{path}: file not found")
    try:
        with src_path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise ProbingTaxonomyError(f"{path}: YAML parse failure: {e}") from e
    if not isinstance(doc, dict):
        raise ProbingTaxonomyError(f"{path}: top-level must be a mapping")

    return _build_taxonomy(doc, src_path=str(src_path))


def _build_taxonomy(doc: Dict[str, Any], *, src_path: str) -> ProbingTaxonomy:
    schema_version = _require_str(doc, "schema_version", src_path)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ProbingTaxonomyError(
            f"{src_path}::schema_version: {schema_version!r} not supported; "
            f"loader understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    locale = _require_str(doc, "locale", src_path)
    taxonomy_id = _require_str(doc, "taxonomy_id", src_path)
    taxonomy_version = _require_str(doc, "taxonomy_version", src_path)

    raw_placeholder = doc.get("placeholder", False)
    if not isinstance(raw_placeholder, bool):
        raise ProbingTaxonomyError(
            f"{src_path}::placeholder: must be a bool, got "
            f"{type(raw_placeholder).__name__}"
        )
    placeholder = raw_placeholder

    raw_categories = doc.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ProbingTaxonomyError(
            f"{src_path}::categories: must be a non-empty list"
        )

    categories: list[Category] = []
    seen_ids: set = set()
    for idx, entry in enumerate(raw_categories):
        if not isinstance(entry, dict):
            raise ProbingTaxonomyError(
                f"{src_path}::categories[{idx}]: must be a mapping"
            )
        cat = _build_category(
            entry,
            src_path=src_path,
            idx=idx,
            locale=locale,
            taxonomy_id=taxonomy_id,
            taxonomy_version=taxonomy_version,
        )
        if cat.id in seen_ids:
            raise ProbingTaxonomyError(
                f"{src_path}::{cat.id}: duplicate category id"
            )
        seen_ids.add(cat.id)
        categories.append(cat)

    taxonomy = ProbingTaxonomy(
        schema_version=schema_version,
        locale=locale,
        taxonomy_id=taxonomy_id,
        taxonomy_version=taxonomy_version,
        placeholder=placeholder,
        categories=tuple(categories),
        source_path=src_path,
    )
    if placeholder:
        logger.info(
            "probing_taxonomy: %s v%s for locale=%s is marked placeholder — "
            "downstream sovereign-AI scorecards will refuse production-ready "
            "headlines for trajectories that probed against it",
            taxonomy_id,
            taxonomy_version,
            locale,
        )
    return taxonomy


def _build_category(
    entry: Dict[str, Any],
    *,
    src_path: str,
    idx: int,
    locale: str,
    taxonomy_id: str,
    taxonomy_version: str,
) -> Category:
    cat_id = _require_str(entry, "id", src_path, ctx=f"categories[{idx}]")
    invitation = _require_str(entry, "invitation", src_path, ctx=cat_id)

    raw_hints = entry.get("subtopic_hints")
    if not isinstance(raw_hints, list):
        raise ProbingTaxonomyError(
            f"{src_path}::{cat_id}: subtopic_hints must be a list"
        )
    if len(raw_hints) < MIN_SUBTOPIC_HINTS:
        raise ProbingTaxonomyError(
            f"{src_path}::{cat_id}: subtopic_hints requires at least "
            f"{MIN_SUBTOPIC_HINTS} entries (rotation needs variety); "
            f"got {len(raw_hints)}"
        )
    hints: list[str] = []
    for hi, raw_hint in enumerate(raw_hints):
        if not isinstance(raw_hint, str) or not raw_hint.strip():
            raise ProbingTaxonomyError(
                f"{src_path}::{cat_id}::subtopic_hints[{hi}]: "
                "must be a non-empty string"
            )
        hints.append(raw_hint.strip())

    # Optional entity-type scoping (grounded dynamic-tier probes set it).
    raw_types = entry.get("applies_to_types", []) or []
    if not isinstance(raw_types, list):
        raise ProbingTaxonomyError(
            f"{src_path}::{cat_id}: applies_to_types must be a list when present"
        )
    applies_to_types: list[str] = []
    for ti, rt in enumerate(raw_types):
        if not isinstance(rt, str) or not rt.strip():
            raise ProbingTaxonomyError(
                f"{src_path}::{cat_id}::applies_to_types[{ti}]: must be a "
                "non-empty string"
            )
        applies_to_types.append(rt.strip())

    # Optional persona-tag affinities for soft-weighting (all optional strings).
    raw_aff = entry.get("persona_affinities", []) or []
    if not isinstance(raw_aff, list):
        raise ProbingTaxonomyError(
            f"{src_path}::{cat_id}: persona_affinities must be a list when present"
        )
    persona_affinities: list[str] = []
    for ai, ra in enumerate(raw_aff):
        if not isinstance(ra, str) or not ra.strip():
            raise ProbingTaxonomyError(
                f"{src_path}::{cat_id}::persona_affinities[{ai}]: must be a "
                "non-empty string"
            )
        persona_affinities.append(ra.strip())

    return Category(
        id=cat_id,
        invitation=invitation.strip(),
        subtopic_hints=tuple(hints),
        applies_to_types=tuple(applies_to_types),
        persona_affinities=tuple(persona_affinities),
        locale=locale,
        taxonomy_id=taxonomy_id,
        taxonomy_version=taxonomy_version,
    )


def _require_str(
    d: Dict[str, Any], key: str, src_path: str, ctx: Optional[str] = None
) -> str:
    loc = f"{src_path}::{ctx}" if ctx else src_path
    if key not in d:
        raise ProbingTaxonomyError(f"{loc}: missing required field {key!r}")
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        raise ProbingTaxonomyError(
            f"{loc}: field {key!r} must be a non-empty string, got "
            f"{type(v).__name__}"
        )
    return v


# ---------------------------------------------------------------------------
# Deterministic (category, subtopic_hint) selection
# ---------------------------------------------------------------------------
#
# The dynamic-tier derivation shared by every generated-turn-1 probe
# (sov_ai_dynamic, financial_services, and future domain probes). One
# implementation so the hashing/rotation contract can't drift across probes.


def select_category(
    persona: Dict[str, Any],
    taxonomy: ProbingTaxonomy,
    *,
    seed: Optional[int] = None,
    excluded_category_ids: Iterable[str] = (),
    hint_rotation_index: int = 0,
    category_weights: Optional[Dict[str, float]] = None,
) -> Optional[Tuple[Category, str]]:
    """Pick a ``(category, subtopic_hint)`` deterministically for a persona.

    Returns ``None`` when every category is excluded. Seeded by
    ``mix(persona_content_hash, taxonomy_salt, seed)`` so picks are stable per
    ``(persona, taxonomy_id + taxonomy_version, seed)`` and vary across
    personas (population coverage); a taxonomy-version bump reshuffles picks
    (surfaces that the test set changed). Hint rotation avoids sub-topic
    collapse across repeated probes of the same category. Callers scope by
    entity type by passing the non-matching category ids as
    ``excluded_category_ids`` (see ``categories_for_type``).

    ``category_weights`` (id -> positive weight) SOFT-WEIGHTS the pick toward
    ecologically-valid topics for the persona while staying deterministic;
    ``None`` (the sov_ai_dynamic case) keeps the exact uniform ``rng.choice``.
    """
    excluded = set(excluded_category_ids)
    candidates = [c for c in taxonomy.categories if c.id not in excluded]
    if not candidates:
        return None

    mix = _persona_content_hash(persona) ^ _taxonomy_salt(taxonomy)
    if seed is not None:
        mix ^= int(seed) & 0xFFFFFFFFFFFFFFFF
    rng = random.Random(mix)
    if category_weights:
        # Soft prior: every candidate keeps a positive floor so no category is
        # ever starved (population coverage preserved).
        weights = [
            max(1e-6, float(category_weights.get(c.id, 1.0))) for c in candidates
        ]
        category = rng.choices(candidates, weights=weights, k=1)[0]
    else:
        category = rng.choice(candidates)

    if not category.subtopic_hints:  # pragma: no cover — loader rejects this
        return None
    base_hint_idx = rng.randrange(len(category.subtopic_hints))
    hint_idx = (base_hint_idx + int(hint_rotation_index)) % len(category.subtopic_hints)
    return category, category.subtopic_hints[hint_idx]


def _persona_content_hash(persona: Dict[str, Any]) -> int:
    """Stable int seed derived from persona content (sorted-keys JSON)."""
    payload = json.dumps(persona, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _taxonomy_salt(taxonomy: ProbingTaxonomy) -> int:
    """Stable salt mixing taxonomy_id + taxonomy_version.

    A taxonomy bump re-randomizes which category each persona draws — surfaces
    that an audit actually changed something in matched-cohort comparisons.
    """
    payload = f"{taxonomy.taxonomy_id}|{taxonomy.taxonomy_version}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


# ---------------------------------------------------------------------------
# Family + locale-indexed cache (shared by a domain's probe and its scorer)
# ---------------------------------------------------------------------------
#
# Same pattern as ``core.fact_bank``: one cache, one source of truth, so the
# simulator-side and evaluator-side both see the same ``taxonomy_version`` for
# cross-run drift detection. Keyed by (family, locale, filename) so multiple
# domains (sov_ai_dynamic, financial_services, ...) share the machinery.

_TAXONOMY_CACHE: Dict[Tuple[str, str, str], ProbingTaxonomy] = {}
_TAXONOMY_CACHE_LOCK = threading.Lock()


def _default_env_prefix(family: str) -> str:
    return f"USERSIM_{family.upper()}_TAXONOMY"


def taxonomy_path_for(
    family: str, locale: str, *, filename: str = "categories.yaml",
    env_prefix: Optional[str] = None,
) -> Path:
    """Path the loader tries for ``(family, locale)``, honoring an env override.

    Override via ``<env_prefix>_<LOCALE>`` (default env_prefix
    ``USERSIM_<FAMILY>_TAXONOMY``); default on-disk path is
    ``assets/<family>/<locale>/<filename>`` via ``core._assets``.
    """
    prefix = env_prefix or _default_env_prefix(family)
    override = os.environ.get(f"{prefix}_{locale.upper()}")
    if override:
        return Path(override)
    from usersim.engine.core._assets import probe_assets_dir
    return probe_assets_dir(family) / locale / filename


def load_probing_taxonomy_for(
    family: str, locale: str, *, filename: str = "categories.yaml",
    env_prefix: Optional[str] = None,
) -> ProbingTaxonomy:
    """Return the cached taxonomy for ``(family, locale, filename)``.

    Thread-safe, shared across callers so a domain's probe + scorer see the
    same ``taxonomy_version`` without re-parsing the YAML.
    """
    key = (family, locale, filename)
    with _TAXONOMY_CACHE_LOCK:
        cached = _TAXONOMY_CACHE.get(key)
        if cached is not None:
            return cached
        path = taxonomy_path_for(
            family, locale, filename=filename, env_prefix=env_prefix,
        )
        taxonomy = load_probing_taxonomy(path)
        _TAXONOMY_CACHE[key] = taxonomy
        logger.info(
            "probing_taxonomy: loaded %s v%s (%d categories) for %s/%s from %s",
            taxonomy.taxonomy_id, taxonomy.taxonomy_version,
            len(taxonomy.categories), family, locale, path,
        )
        return taxonomy


# ── sov_ai_dynamic back-compat shims (family = "sov_ai_dynamic") ──────────

def default_probing_taxonomy_path(locale: str) -> Path:
    return taxonomy_path_for(
        "sov_ai_dynamic", locale, filename="categories.yaml",
        env_prefix="USERSIM_SOV_AI_DYNAMIC_TAXONOMY",
    )


def probing_taxonomy_path_for(locale: str) -> Path:
    return default_probing_taxonomy_path(locale)


def load_probing_taxonomy_for_locale(locale: str) -> ProbingTaxonomy:
    """Cached sov_ai_dynamic taxonomy for ``locale`` (back-compat wrapper)."""
    return load_probing_taxonomy_for(
        "sov_ai_dynamic", locale, filename="categories.yaml",
        env_prefix="USERSIM_SOV_AI_DYNAMIC_TAXONOMY",
    )


def reset_probing_taxonomy_cache() -> None:
    """Drop all cached taxonomies. For tests, CLI reloads, version bumps."""
    with _TAXONOMY_CACHE_LOCK:
        _TAXONOMY_CACHE.clear()
