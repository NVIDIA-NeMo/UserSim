# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Fact-bank loader + validator + lookup helpers.

A fact bank is a curated YAML file of authoritative, locale-specific
ground-truth facts used by the ``sov_ai_facts`` probe. The schema lives
in ``assets/sov_ai_facts/SCHEMA.md``.

The module exposes:

- :class:`Fact` — typed view of a single entry.
- :class:`FactBank` — typed view of a whole bank + lookup helpers.
- :class:`FactBankError` — raised for schema violations. The loader is
  strict; schema drift should surface at load time, not at simulation
  time.
- :func:`load_fact_bank` — parse YAML + validate + return ``FactBank``.

Contract:

- Loader refuses unknown ``schema_version``.
- Every fact is frozen and hashable once constructed; mutation of a
  loaded bank is impossible by design.
- Placeholder entries (``placeholder: true``) are loaded normally, but
  the loader emits a one-line INFO log summarizing their count so the
  calling driver never silently promotes a placeholder-heavy bank.
- Lookups are stable — ``by_category`` / ``matching_persona_tags`` /
  etc. always return entries in their file order so consumers get
  deterministic iteration.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("usersim.engine")

SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"v0.1"})
ALLOWED_QUESTION_TYPES: frozenset[str] = frozenset({"factual_recall", "completion"})
ALLOWED_DIFFICULTIES: frozenset[str] = frozenset({"easy", "medium", "hard"})

_PERSONA_TAG_RE = re.compile(r"^(region|age|occupation-family|education|interest):[a-zA-Z0-9_\-+]+$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class FactBankError(ValueError):
    """A user-facing fact-bank validation problem.

    Message format: ``"<path>::<entry_id or field>: <what's wrong>"``
    so reviewers can grep the offending location.
    """


# ---------------------------------------------------------------------------
# Typed views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactProvenance:
    source: str
    last_reviewed: str  # YYYY-MM-DD
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fact:
    id: str
    category: str
    question_type: str  # factual_recall | completion
    placeholder: bool
    question: str
    false_premises: tuple[str, ...]
    ground_truth: str
    graceful_unknown: str | None
    persona_tags: tuple[str, ...]
    difficulty: str  # easy | medium | hard
    provenance: FactProvenance
    # Set by the loader from the bank's top-level fields so downstream
    # consumers can carry provenance per-fact without holding the whole
    # bank.
    locale: str = ""
    bank_id: str = ""
    bank_version: str = ""


@dataclass(frozen=True)
class FactBank:
    schema_version: str
    locale: str
    bank_id: str
    bank_version: str
    categories: tuple[str, ...]
    tags: tuple[str, ...]
    facts: tuple[Fact, ...] = field(default_factory=tuple)
    source_path: str | None = None

    # ── Lookups ─────────────────────────────────────────────────────

    def by_id(self, fact_id: str) -> Fact | None:
        for f in self.facts:
            if f.id == fact_id:
                return f
        return None

    def by_category(self, category: str) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.category == category)

    def by_tag(self, tag: str) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if tag in f.persona_tags)

    def matching_persona_tags(self, tags: Iterable[str]) -> tuple[Fact, ...]:
        """Return facts whose persona_tags intersect ``tags``.

        ``*:any`` wildcards on the fact side match every persona tag
        with the same prefix. Example: a fact tagged ``region:any``
        matches a persona carrying ``region:northeast``.
        """
        wanted = set(tags)
        if not wanted:
            return ()

        wildcard_prefixes: set = set()
        concrete: set = set()
        for t in wanted:
            if ":" in t:
                prefix, value = t.split(":", 1)
                if value == "any":
                    wildcard_prefixes.add(prefix)
                    continue
            concrete.add(t)

        out: list[Fact] = []
        for f in self.facts:
            if concrete & set(f.persona_tags):
                out.append(f)
                continue
            # Persona carries *:any → matches any fact tag with that prefix.
            if wildcard_prefixes and any(t.split(":", 1)[0] in wildcard_prefixes for t in f.persona_tags if ":" in t):
                out.append(f)
                continue
            # Fact carries *:any → matches any persona tag with that prefix.
            fact_wildcards = {t.split(":", 1)[0] for t in f.persona_tags if t.endswith(":any")}
            if fact_wildcards and any(t.split(":", 1)[0] in fact_wildcards for t in wanted if ":" in t):
                out.append(f)
        return tuple(out)

    def placeholder_only(self) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.placeholder)

    def non_placeholder(self) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if not f.placeholder)

    def placeholder_fraction(self) -> float:
        if not self.facts:
            return 0.0
        return sum(1 for f in self.facts if f.placeholder) / len(self.facts)

    def provenance_summary(self) -> dict[str, Any]:
        """Surfaces that the downstream reporting layer reads."""
        return {
            "bank_id": self.bank_id,
            "bank_version": self.bank_version,
            "locale": self.locale,
            "n_facts": len(self.facts),
            "n_placeholder": sum(1 for f in self.facts if f.placeholder),
            "categories": list(self.categories),
            "distinct_sources": sorted({f.provenance.source for f in self.facts}),
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_fact_bank(path: str | Path) -> FactBank:
    """Parse and validate a fact bank from disk.

    Raises :class:`FactBankError` with a message that names the offending
    file, entry id (or field), and the invariant that was violated.
    """
    import yaml  # lazy: kept off the plugin's top-level import graph

    src_path = Path(path)
    if not src_path.exists():
        raise FactBankError(f"{path}: file not found")
    try:
        with src_path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise FactBankError(f"{path}: YAML parse failure: {e}") from e
    if not isinstance(doc, dict):
        raise FactBankError(f"{path}: top-level must be a mapping")

    return _build_fact_bank(doc, src_path=str(src_path))


# ---------------------------------------------------------------------------
# Locale-indexed cache (shared by the probe and the scorer)
# ---------------------------------------------------------------------------
#
# Both the simulator-side probe (``sov_ai_facts``) and
# the out-of-sim scorer need the same bank instance per locale. Caching
# here — rather than duplicating in each caller — guarantees they agree
# on bank_version and entry contents, which is what keeps the scorer's
# ground truth in lockstep with the simulator's probe.
#
# Override the default path by setting ``USERSIM_SOV_AI_FACTS_BANK_<LOCALE>``,
# e.g. ``USERSIM_SOV_AI_FACTS_BANK_PT_BR=/abs/path/pt_BR_v1.yaml``. This is the
# hook SLURM / CI jobs use to pin a bank version.

_FACT_BANK_CACHE: dict[str, FactBank] = {}
_FACT_BANK_CACHE_LOCK = threading.Lock()


def default_fact_bank_path(locale: str) -> Path:
    """Default on-disk path for a locale's fact bank.

    Discovered via ``core._assets.default_assets_dir`` so the loader
    picks up ``assets/sov_ai_facts/<locale>/sample.yaml`` regardless
    of CWD. Asset directory matches the consuming probe name.
    """
    from usersim.engine.core._assets import probe_assets_dir

    return probe_assets_dir("sov_ai_facts") / locale / "sample.yaml"


def fact_bank_path_for(locale: str) -> Path:
    """Path the cached loader will try for ``locale``, honoring env overrides."""
    env_key = f"USERSIM_SOV_AI_FACTS_BANK_{locale.upper()}"
    override = os.environ.get(env_key)
    if override:
        return Path(override)
    return default_fact_bank_path(locale)


def load_fact_bank_for_locale(locale: str) -> FactBank:
    """Return the cached fact bank for ``locale``, loading on first use.

    Thread-safe. Shared across callers in this Python process so the
    simulator and the scorer see the same ``bank_version`` without
    re-parsing the YAML.
    """
    with _FACT_BANK_CACHE_LOCK:
        cached = _FACT_BANK_CACHE.get(locale)
        if cached is not None:
            return cached
        path = fact_bank_path_for(locale)
        bank = load_fact_bank(path)
        _FACT_BANK_CACHE[locale] = bank
        logger.info(
            "fact_bank: loaded %s v%s (%d facts) for locale=%s from %s",
            bank.bank_id,
            bank.bank_version,
            len(bank.facts),
            locale,
            path,
        )
        return bank


def reset_fact_bank_cache() -> None:
    """Drop all cached banks. For tests, CLI reloads, and bank-version bumps."""
    with _FACT_BANK_CACHE_LOCK:
        _FACT_BANK_CACHE.clear()


def _build_fact_bank(doc: dict[str, Any], *, src_path: str) -> FactBank:
    schema_version = _require_str(doc, "schema_version", src_path)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise FactBankError(
            f"{src_path}::schema_version: {schema_version!r} not supported; "
            f"loader understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    locale = _require_str(doc, "locale", src_path)
    bank_id = _require_str(doc, "bank_id", src_path)
    bank_version = _require_str(doc, "bank_version", src_path)

    raw_categories = doc.get("categories")
    if not isinstance(raw_categories, list) or not raw_categories:
        raise FactBankError(f"{src_path}::categories: must be a non-empty list")
    categories = tuple(str(c) for c in raw_categories)

    raw_tags = doc.get("tags", [])
    if not isinstance(raw_tags, list):
        raise FactBankError(f"{src_path}::tags: must be a list")
    tags = tuple(str(t) for t in raw_tags)

    raw_entries = doc.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise FactBankError(f"{src_path}::entries: must be a non-empty list")

    facts: list[Fact] = []
    seen_ids: set = set()
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise FactBankError(f"{src_path}::entries[{idx}]: must be a mapping")
        fact = _build_fact(
            entry,
            src_path=src_path,
            idx=idx,
            categories=categories,
            locale=locale,
            bank_id=bank_id,
            bank_version=bank_version,
        )
        if fact.id in seen_ids:
            raise FactBankError(f"{src_path}::{fact.id}: duplicate fact id")
        seen_ids.add(fact.id)
        facts.append(fact)

    bank = FactBank(
        schema_version=schema_version,
        locale=locale,
        bank_id=bank_id,
        bank_version=bank_version,
        categories=categories,
        tags=tags,
        facts=tuple(facts),
        source_path=src_path,
    )

    n_placeholder = sum(1 for f in facts if f.placeholder)
    if n_placeholder:
        logger.info(
            "fact_bank: loaded %s (v%s) with %d/%d placeholder entries — "
            "downstream scorecards will be marked preview-only",
            bank_id,
            bank_version,
            n_placeholder,
            len(facts),
        )
    return bank


def _build_fact(
    entry: dict[str, Any],
    *,
    src_path: str,
    idx: int,
    categories: tuple[str, ...],
    locale: str,
    bank_id: str,
    bank_version: str,
) -> Fact:
    fact_id = _require_str(entry, "id", src_path, ctx=f"entries[{idx}]")
    category = _require_str(entry, "category", src_path, ctx=fact_id)
    if category not in categories:
        raise FactBankError(f"{src_path}::{fact_id}: category {category!r} not in bank categories {list(categories)}")

    question_type = _require_str(entry, "question_type", src_path, ctx=fact_id)
    if question_type not in ALLOWED_QUESTION_TYPES:
        raise FactBankError(
            f"{src_path}::{fact_id}: question_type {question_type!r} must be one of {sorted(ALLOWED_QUESTION_TYPES)}"
        )

    placeholder = entry.get("placeholder", False)
    if not isinstance(placeholder, bool):
        raise FactBankError(f"{src_path}::{fact_id}: placeholder must be a bool, got {type(placeholder).__name__}")

    question = _require_str(entry, "question", src_path, ctx=fact_id)
    ground_truth = _require_str(entry, "ground_truth", src_path, ctx=fact_id)

    # false_premises — required (possibly empty) for factual_recall;
    # ignored for completion (stored empty to keep the dataclass uniform).
    raw_fp = entry.get("false_premises", [])
    if not isinstance(raw_fp, list):
        raise FactBankError(f"{src_path}::{fact_id}: false_premises must be a list")
    false_premises = tuple(str(p) for p in raw_fp)
    if question_type == "completion" and false_premises:
        raise FactBankError(
            f"{src_path}::{fact_id}: completion entries must not declare "
            "false_premises (that's a factual_recall concept); drop the key"
        )

    graceful_unknown = entry.get("graceful_unknown")
    if question_type == "completion":
        if not isinstance(graceful_unknown, str) or not graceful_unknown.strip():
            raise FactBankError(
                f"{src_path}::{fact_id}: completion entries require "
                "graceful_unknown (what the model should say when it "
                "doesn't know the canonical continuation)"
            )
    else:
        if graceful_unknown is not None and not isinstance(graceful_unknown, str):
            raise FactBankError(
                f"{src_path}::{fact_id}: graceful_unknown must be a string if "
                f"set, got {type(graceful_unknown).__name__}"
            )

    raw_persona_tags = entry.get("persona_tags", [])
    if not isinstance(raw_persona_tags, list) or not raw_persona_tags:
        raise FactBankError(f"{src_path}::{fact_id}: persona_tags must be a non-empty list")
    persona_tags: list[str] = []
    for t in raw_persona_tags:
        if not isinstance(t, str):
            raise FactBankError(f"{src_path}::{fact_id}: persona_tags entries must be strings")
        if not _PERSONA_TAG_RE.match(t):
            raise FactBankError(
                f"{src_path}::{fact_id}: persona_tag {t!r} does not match "
                "the `<prefix>:<value>` schema (see SCHEMA.md §Persona tags)"
            )
        persona_tags.append(t)

    difficulty = _require_str(entry, "difficulty", src_path, ctx=fact_id)
    if difficulty not in ALLOWED_DIFFICULTIES:
        raise FactBankError(
            f"{src_path}::{fact_id}: difficulty {difficulty!r} must be one of {sorted(ALLOWED_DIFFICULTIES)}"
        )

    provenance = _build_provenance(
        entry.get("provenance"),
        src_path=src_path,
        fact_id=fact_id,
        placeholder=placeholder,
    )

    return Fact(
        id=fact_id,
        category=category,
        question_type=question_type,
        placeholder=placeholder,
        question=question,
        false_premises=false_premises,
        ground_truth=ground_truth,
        graceful_unknown=graceful_unknown if isinstance(graceful_unknown, str) else None,
        persona_tags=tuple(persona_tags),
        difficulty=difficulty,
        provenance=provenance,
        locale=locale,
        bank_id=bank_id,
        bank_version=bank_version,
    )


def _build_provenance(
    raw: Any,
    *,
    src_path: str,
    fact_id: str,
    placeholder: bool,
) -> FactProvenance:
    if not isinstance(raw, dict):
        raise FactBankError(f"{src_path}::{fact_id}: provenance must be a mapping")
    source = _require_str(raw, "source", src_path, ctx=f"{fact_id}/provenance")
    last_reviewed = _require_str(raw, "last_reviewed", src_path, ctx=f"{fact_id}/provenance")
    if not _ISO_DATE_RE.match(last_reviewed):
        raise FactBankError(
            f"{src_path}::{fact_id}: provenance.last_reviewed must be YYYY-MM-DD, got {last_reviewed!r}"
        )
    try:
        date.fromisoformat(last_reviewed)
    except ValueError as e:
        raise FactBankError(f"{src_path}::{fact_id}: provenance.last_reviewed is not a real calendar date: {e}") from e

    raw_refs = raw.get("references", [])
    if not isinstance(raw_refs, list):
        raise FactBankError(f"{src_path}::{fact_id}: provenance.references must be a list")
    references = tuple(str(r) for r in raw_refs)

    if placeholder:
        # Placeholder entries may omit references and use "placeholder"
        # as source; the loader logs + downstream flags propagate.
        pass
    else:
        if source.strip().lower() == "placeholder":
            raise FactBankError(
                f"{src_path}::{fact_id}: non-placeholder entry must declare a "
                "real provenance.source (not 'placeholder')"
            )
        if not references:
            raise FactBankError(
                f"{src_path}::{fact_id}: non-placeholder entry must declare at least one provenance.references citation"
            )

    return FactProvenance(
        source=source,
        last_reviewed=last_reviewed,
        references=references,
    )


def _require_str(d: dict[str, Any], key: str, src_path: str, ctx: str | None = None) -> str:
    loc = f"{src_path}::{ctx}" if ctx else src_path
    if key not in d:
        raise FactBankError(f"{loc}: missing required field {key!r}")
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        raise FactBankError(f"{loc}: field {key!r} must be a non-empty string, got {type(v).__name__}")
    return v
