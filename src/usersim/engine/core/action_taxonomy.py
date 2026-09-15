# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed loader for the shipped ``assets/safety_agentic/action_taxonomy.yaml``.

The ``core.agentic_bank`` loader already reads this file at load time
for *name-only* validation (does every bank entry's
``sanctioned_action_name`` resolve to a taxonomy entry?). The
``safety_agentic`` scorer needs more: it joins every attempted action
against the taxonomy's risk metadata to compute
``modeled_blast_radius``, decide whether confirmation was required,
whether consequence disclosure was required, and to classify
boundary violations.

Design choices:

- **Separate cache from the name-only cache in ``agentic_bank``.** The
  two callers have different needs (names for bank validation, rich
  objects for scoring) and keeping them separate means changing
  either surface doesn't touch the other. Both honour the same
  ``USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY`` env override so they agree on
  *which* taxonomy is being read.
- **``blast_radius_rank``** maps the ordinal ``blast_radius_levels``
  vocabulary onto integers 0..5. Scorers and bundles aggregate on
  the rank, not the string, so "higher blast radius" is meaningful
  as an arithmetic operation. Rank is defined by position in the
  taxonomy's ``blast_radius_levels`` list (authoritative — not
  hardcoded here) so extending the vocabulary via a new entry at the
  right position is a schema-change-only operation.
- **``p_user_detect.value``** is lifted onto the entry directly for
  convenience; the ``rationale`` stays under a nested structure to
  keep the reviewer-authored text accessible.

Raises :class:`ActionTaxonomyError` with a message that names the
offending file + entry so drift is debuggable at load time.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("usersim.engine")

__all__ = [
    "ActionTaxonomyError",
    "ActionTaxonomyEntry",
    "ActionTaxonomyProvenance",
    "ActionTaxonomy",
    "load_action_taxonomy",
    "load_action_taxonomy_default",
    "reset_action_taxonomy_cache",
]


class ActionTaxonomyError(ValueError):
    """User-facing schema or load problem.

    Message format: ``"<path>::<entry name or field>: <what's wrong>"``
    so reviewers can grep the offending location.
    """


# ---------------------------------------------------------------------------
# Typed views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionTaxonomyProvenance:
    source: str
    last_reviewed: str


@dataclass(frozen=True)
class ActionTaxonomyEntry:
    name: str
    description: str
    category: str
    risk_tier: str
    reversible: bool
    requires_confirmation: bool
    consequence_disclosure_required: bool
    modeled_blast_radius: str
    blast_radius_rank: int
    p_user_detect_value: float
    p_user_detect_rationale: str
    tags: Tuple[str, ...] = field(default_factory=tuple)
    provenance: Optional[ActionTaxonomyProvenance] = None


@dataclass(frozen=True)
class ActionTaxonomy:
    schema_version: str
    risk_tiers: Tuple[str, ...]
    blast_radius_levels: Tuple[str, ...]
    categories: Tuple[str, ...]
    entries: Tuple[ActionTaxonomyEntry, ...]
    source_path: Optional[str] = None

    def by_name(self, name: str) -> Optional[ActionTaxonomyEntry]:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def names(self) -> Tuple[str, ...]:
        return tuple(e.name for e in self.entries)

    def blast_radius_rank(self, level: str) -> Optional[int]:
        """Return the rank (index) for a blast-radius level string.

        ``None`` when the level is not in the authoritative list.
        Callers should treat that as a structured error, not a 0.
        """
        try:
            return self.blast_radius_levels.index(level)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Cache (separate from the name-only cache in agentic_bank)
# ---------------------------------------------------------------------------


_ACTION_TAXONOMY_CACHE: Dict[str, ActionTaxonomy] = {}
_ACTION_TAXONOMY_CACHE_LOCK = threading.Lock()


def _default_action_taxonomy_path() -> Path:
    """Discover the shipped ``assets/safety_agentic/action_taxonomy.yaml``
    regardless of CWD. See ``core._assets.default_assets_dir``."""
    from usersim.engine.core._assets import probe_assets_dir
    return probe_assets_dir("safety_agentic") / "action_taxonomy.yaml"


def _resolve_path() -> Path:
    override = os.environ.get("USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY")
    return Path(override) if override else _default_action_taxonomy_path()


def load_action_taxonomy_default() -> ActionTaxonomy:
    """Load the shipped taxonomy (or the env-overridden path) with caching.

    Honours ``USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY``. Cache key is the
    resolved path so a test that points the env var at a different file
    gets a different cached object.
    """
    path = _resolve_path()
    cache_key = str(path)
    with _ACTION_TAXONOMY_CACHE_LOCK:
        cached = _ACTION_TAXONOMY_CACHE.get(cache_key)
        if cached is not None:
            return cached
        loaded = load_action_taxonomy(path)
        _ACTION_TAXONOMY_CACHE[cache_key] = loaded
        logger.info(
            "action_taxonomy: loaded %d rich entries from %s",
            len(loaded.entries), path,
        )
        return loaded


def reset_action_taxonomy_cache() -> None:
    """Drop the cached rich taxonomy. For tests / version bumps."""
    with _ACTION_TAXONOMY_CACHE_LOCK:
        _ACTION_TAXONOMY_CACHE.clear()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_action_taxonomy(path: str | Path) -> ActionTaxonomy:
    """Parse and validate the rich action taxonomy from disk.

    Same YAML file as the one the agentic-bank loader reads for
    name-only validation; this pass parses the full risk metadata.
    """
    import yaml

    src_path = Path(path)
    if not src_path.exists():
        raise ActionTaxonomyError(f"{src_path}: file does not exist")

    try:
        with src_path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise ActionTaxonomyError(f"{src_path}: YAML parse error: {e}") from e

    if not isinstance(doc, dict):
        raise ActionTaxonomyError(
            f"{src_path}: top-level must be a mapping"
        )

    schema_version = doc.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ActionTaxonomyError(
            f"{src_path}: missing or non-string `schema_version`"
        )

    risk_tiers = _require_str_list(doc, "risk_tiers", src_path)
    blast_radius_levels = _require_str_list(
        doc, "blast_radius_levels", src_path,
    )
    categories = _require_str_list(doc, "categories", src_path)

    raw_entries = doc.get("entries")
    if not isinstance(raw_entries, list):
        raise ActionTaxonomyError(
            f"{src_path}: missing top-level `entries` list"
        )

    entries: List[ActionTaxonomyEntry] = []
    seen_names: set[str] = set()

    for i, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ActionTaxonomyError(
                f"{src_path}::entries[{i}]: must be a mapping"
            )
        entry = _parse_entry(
            raw, i, src_path,
            risk_tiers=risk_tiers,
            blast_radius_levels=blast_radius_levels,
            categories=categories,
        )
        if entry.name in seen_names:
            raise ActionTaxonomyError(
                f"{src_path}::entries[{i}]::{entry.name}: duplicate name"
            )
        seen_names.add(entry.name)
        entries.append(entry)

    return ActionTaxonomy(
        schema_version=schema_version,
        risk_tiers=tuple(risk_tiers),
        blast_radius_levels=tuple(blast_radius_levels),
        categories=tuple(categories),
        entries=tuple(entries),
        source_path=str(src_path),
    )


def _require_str_list(doc: Dict[str, Any], key: str, path: Path) -> List[str]:
    raw = doc.get(key)
    if not isinstance(raw, list) or not all(
        isinstance(x, str) and x for x in raw
    ):
        raise ActionTaxonomyError(
            f"{path}: `{key}` must be a non-empty list of strings"
        )
    return list(raw)


def _parse_entry(
    raw: Dict[str, Any],
    idx: int,
    path: Path,
    *,
    risk_tiers: List[str],
    blast_radius_levels: List[str],
    categories: List[str],
) -> ActionTaxonomyEntry:
    where = f"{path}::entries[{idx}]"

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ActionTaxonomyError(
            f"{where}: missing or non-string `name`"
        )
    where = f"{path}::{name}"

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ActionTaxonomyError(
            f"{where}: missing or empty `description`"
        )

    category = raw.get("category")
    if category not in categories:
        raise ActionTaxonomyError(
            f"{where}: `category`={category!r} not in declared "
            f"categories {categories}"
        )

    risk_tier = raw.get("risk_tier")
    if risk_tier not in risk_tiers:
        raise ActionTaxonomyError(
            f"{where}: `risk_tier`={risk_tier!r} not in declared "
            f"risk_tiers {risk_tiers}"
        )

    reversible = raw.get("reversible")
    if not isinstance(reversible, bool):
        raise ActionTaxonomyError(
            f"{where}: `reversible` must be a boolean"
        )

    requires_confirmation = raw.get("requires_confirmation")
    if not isinstance(requires_confirmation, bool):
        raise ActionTaxonomyError(
            f"{where}: `requires_confirmation` must be a boolean"
        )

    consequence_disclosure_required = raw.get("consequence_disclosure_required")
    if not isinstance(consequence_disclosure_required, bool):
        raise ActionTaxonomyError(
            f"{where}: `consequence_disclosure_required` must be a boolean"
        )

    modeled_blast_radius = raw.get("modeled_blast_radius")
    if modeled_blast_radius not in blast_radius_levels:
        raise ActionTaxonomyError(
            f"{where}: `modeled_blast_radius`={modeled_blast_radius!r} not "
            f"in declared blast_radius_levels {blast_radius_levels}"
        )
    blast_rank = blast_radius_levels.index(modeled_blast_radius)

    p_user_detect_raw = raw.get("p_user_detect")
    if not isinstance(p_user_detect_raw, dict):
        raise ActionTaxonomyError(
            f"{where}: `p_user_detect` must be a mapping with `value` "
            "and `rationale`"
        )
    p_value = p_user_detect_raw.get("value")
    if not isinstance(p_value, (int, float)) or not 0.0 <= float(p_value) <= 1.0:
        raise ActionTaxonomyError(
            f"{where}: `p_user_detect.value` must be a number in [0, 1]; "
            f"got {p_value!r}"
        )
    p_rationale = p_user_detect_raw.get("rationale")
    if not isinstance(p_rationale, str) or not p_rationale.strip():
        raise ActionTaxonomyError(
            f"{where}: `p_user_detect.rationale` must be a non-empty string"
        )

    tags_raw = raw.get("tags", [])
    if not isinstance(tags_raw, list) or not all(
        isinstance(t, str) and t for t in tags_raw
    ):
        raise ActionTaxonomyError(
            f"{where}: `tags` must be a list of non-empty strings"
        )

    prov_raw = raw.get("provenance")
    prov: Optional[ActionTaxonomyProvenance] = None
    if prov_raw is not None:
        if not isinstance(prov_raw, dict):
            raise ActionTaxonomyError(
                f"{where}: `provenance` must be a mapping when present"
            )
        source = prov_raw.get("source")
        last_reviewed = prov_raw.get("last_reviewed")
        if not isinstance(source, str) or not source.strip():
            raise ActionTaxonomyError(
                f"{where}: `provenance.source` must be a non-empty string"
            )
        if not isinstance(last_reviewed, str) or not last_reviewed.strip():
            raise ActionTaxonomyError(
                f"{where}: `provenance.last_reviewed` must be a non-empty string"
            )
        prov = ActionTaxonomyProvenance(
            source=source, last_reviewed=last_reviewed,
        )

    return ActionTaxonomyEntry(
        name=name,
        description=description.strip(),
        category=category,
        risk_tier=risk_tier,
        reversible=reversible,
        requires_confirmation=requires_confirmation,
        consequence_disclosure_required=consequence_disclosure_required,
        modeled_blast_radius=modeled_blast_radius,
        blast_radius_rank=blast_rank,
        p_user_detect_value=float(p_value),
        p_user_detect_rationale=p_rationale.strip(),
        tags=tuple(tags_raw),
        provenance=prov,
    )
