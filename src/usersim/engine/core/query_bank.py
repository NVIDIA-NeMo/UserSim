# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-locale query-bank loader, validator, and lookup helpers.

The query bank is the load-bearing primitive for the
``sov_ai_multilingual_parity`` probe. Schema is documented in
``assets/sov_ai_multilingual_parity/SCHEMA.md``.

The unit is the **conceptual question**, not the locale-specific
phrasing — each :class:`Query` carries N locale renderings of the
same underlying query, joined by a single ``query_id``. This is what
makes matched-pair cross-locale comparison possible: the same
``query_id`` appearing on trajectories in pt_BR and ja_JP means the
same conceptual question was asked, even though the actual injected
text differed across the two trajectories.

This module exposes:

- :class:`Query` — typed view of a single query template entry.
- :class:`QueryBank` — typed view of the whole bank + lookup helpers.
- :class:`QueryBankError` — raised for schema violations. The loader
  fails fast on every invariant violation so reviewers get an
  actionable error pointing at the entry id + field.
- :func:`load_query_bank` — parse + validate a single file.
- :func:`load_query_bank_default` / :func:`reset_query_bank_cache`
  — process-local cache for the (typically single) bank used by the
  probe and downstream scorers, mirroring the pattern in
  :mod:`core.fact_bank` and :mod:`core.probing_taxonomy`.

Architecturally a query bank is **shared across locales** (one file,
N renderings per entry) — this is the structural difference from the
fact bank (one file per locale) and probing taxonomy (one file per
locale). See ``assets/sov_ai_multilingual_parity/SCHEMA.md`` for the rationale.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

logger = logging.getLogger("usersim.engine")

SUPPORTED_SCHEMA_VERSIONS: FrozenSet[str] = frozenset({"v0.1"})
ALLOWED_DIFFICULTIES: FrozenSet[str] = frozenset({"easy", "medium", "hard"})
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class QueryBankError(ValueError):
    """User-facing schema or load problem.

    Message format: ``"<path>::<entry_id or field>: <what's wrong>"``
    so reviewers can grep the offending location.
    """


# ---------------------------------------------------------------------------
# Typed views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryProvenance:
    source: str
    last_reviewed: str
    references: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Query:
    id: str
    domain: str
    placeholder: bool
    difficulty: str
    persona_tags: Tuple[str, ...]
    concern: str
    # Locale → rendering text. Always a non-empty string for each locale
    # in the bank's `locales` list that is NOT in renderings_opted_out.
    renderings: Dict[str, str] = field(default_factory=dict)
    renderings_opted_out: Tuple[str, ...] = field(default_factory=tuple)
    provenance: Optional[QueryProvenance] = None
    # Set by the loader from the bank's top-level fields so consumers
    # can carry provenance per-query without re-looking-up the bank.
    bank_id: str = ""
    bank_version: str = ""

    def supports_locale(self, locale: str) -> bool:
        """True iff this query has a non-empty rendering for ``locale``."""
        return locale in self.renderings and bool(self.renderings[locale])

    def rendering_for(self, locale: str) -> Optional[str]:
        """Return the rendering text for ``locale``, or ``None`` if the
        query has no rendering for that locale (either because the
        locale isn't in the bank or because the entry opted out)."""
        return self.renderings.get(locale)


@dataclass(frozen=True)
class QueryBank:
    schema_version: str
    bank_id: str
    bank_version: str
    locales: Tuple[str, ...]
    domains: Tuple[str, ...]
    tags: Tuple[str, ...] = field(default_factory=tuple)
    queries: Tuple[Query, ...] = field(default_factory=tuple)
    source_path: Optional[str] = None

    # ── Lookups ─────────────────────────────────────────────────────

    def by_id(self, query_id: str) -> Optional[Query]:
        for q in self.queries:
            if q.id == query_id:
                return q
        return None

    def by_domain(self, domain: str) -> Tuple[Query, ...]:
        return tuple(q for q in self.queries if q.domain == domain)

    def supporting_locale(self, locale: str) -> Tuple[Query, ...]:
        """Queries that have a rendering for ``locale``. (Excludes
        opted-out entries.)"""
        return tuple(q for q in self.queries if q.supports_locale(locale))

    def matching_persona_tags(
        self,
        persona_tags: Iterable[str],
    ) -> Tuple[Query, ...]:
        """Queries whose persona_tags non-trivially intersect the
        supplied set, with ``*:any`` tags treated as wildcards over
        the prefix.

        Wildcard semantics: a query tagged ``age:any`` matches any
        persona regardless of age tags. A query tagged ``age:18-44``
        only matches personas with that exact tag (or with ``age:any``
        in their tag set, used by tests with mock personas).
        """
        persona_set = frozenset(persona_tags)
        out: List[Query] = []
        for q in self.queries:
            if _persona_tags_match(q.persona_tags, persona_set):
                out.append(q)
        return tuple(out)

    def query_ids(self) -> Tuple[str, ...]:
        return tuple(q.id for q in self.queries)

    def provenance_summary(self) -> Dict[str, Any]:
        """Surfaces that the downstream reporting layer reads."""
        return {
            "bank_id": self.bank_id,
            "bank_version": self.bank_version,
            "schema_version": self.schema_version,
            "n_queries": len(self.queries),
            "n_placeholders": sum(1 for q in self.queries if q.placeholder),
            "locales": list(self.locales),
            "domains": list(self.domains),
        }


# ---------------------------------------------------------------------------
# Persona-tag matcher
# ---------------------------------------------------------------------------


def _persona_tags_match(
    query_tags: Iterable[str],
    persona_tags: FrozenSet[str],
) -> bool:
    """Return True iff the query's required tags are satisfiable by
    the persona.

    Per-prefix semantics: for each prefix used in the query's tags
    (``age``, ``region``, ``occupation-family``, ``education``,
    ``interest``), at least one of the query's tags with that prefix
    must be present in the persona's tag set, OR be a ``*:any``
    wildcard that matches any persona along that axis.
    """
    by_prefix: Dict[str, List[str]] = {}
    for tag in query_tags:
        if ":" not in tag:
            # Tags without a prefix are treated as standalone — must
            # appear verbatim in the persona's set.
            by_prefix.setdefault("__bare__", []).append(tag)
            continue
        prefix = tag.split(":", 1)[0]
        by_prefix.setdefault(prefix, []).append(tag)

    for prefix, tags_for_prefix in by_prefix.items():
        if prefix == "__bare__":
            if not all(t in persona_tags for t in tags_for_prefix):
                return False
            continue
        # If the query carries a wildcard along this axis, the axis
        # is automatically satisfied.
        wildcard = f"{prefix}:any"
        if wildcard in tags_for_prefix:
            continue
        # Otherwise at least one of the query's tags for this prefix
        # must appear in the persona's tag set, OR the persona itself
        # carries the matching wildcard.
        persona_wildcard = wildcard in persona_tags
        if persona_wildcard:
            continue
        if not any(t in persona_tags for t in tags_for_prefix):
            return False
    return True


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_query_bank(path: str | Path) -> QueryBank:
    """Parse and validate a query bank from disk.

    Raises :class:`QueryBankError` with a message that names the
    offending file, entry id (or field), and the invariant violated.
    """
    import yaml  # lazy: keeps yaml off the plugin's top-level import graph

    src_path = Path(path)
    if not src_path.exists():
        raise QueryBankError(f"{path}: file not found")
    try:
        with src_path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise QueryBankError(f"{path}: YAML parse failure: {e}") from e
    if not isinstance(doc, dict):
        raise QueryBankError(f"{path}: top-level must be a mapping")

    return _build_query_bank(doc, src_path=str(src_path))


def _build_query_bank(doc: Dict[str, Any], *, src_path: str) -> QueryBank:
    schema_version = _require_str(doc, "schema_version", src_path)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise QueryBankError(
            f"{src_path}::schema_version: {schema_version!r} not supported; "
            f"loader understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    bank_id = _require_str(doc, "bank_id", src_path)
    bank_version = _require_str(doc, "bank_version", src_path)

    raw_locales = doc.get("locales")
    if not isinstance(raw_locales, list) or not raw_locales:
        raise QueryBankError(
            f"{src_path}::locales: must be a non-empty list of locale ids"
        )
    locales: List[str] = []
    for i, loc in enumerate(raw_locales):
        if not isinstance(loc, str) or not loc.strip():
            raise QueryBankError(
                f"{src_path}::locales[{i}]: must be a non-empty string"
            )
        locales.append(loc.strip())
    if len(set(locales)) != len(locales):
        raise QueryBankError(
            f"{src_path}::locales: contains duplicates: {locales}"
        )
    locales_tuple = tuple(locales)
    locales_set = frozenset(locales_tuple)

    raw_domains = doc.get("domains")
    if not isinstance(raw_domains, list) or not raw_domains:
        raise QueryBankError(
            f"{src_path}::domains: must be a non-empty list of domain names"
        )
    domains: List[str] = []
    for i, dom in enumerate(raw_domains):
        if not isinstance(dom, str) or not dom.strip():
            raise QueryBankError(
                f"{src_path}::domains[{i}]: must be a non-empty string"
            )
        domains.append(dom.strip())
    if len(set(domains)) != len(domains):
        raise QueryBankError(
            f"{src_path}::domains: contains duplicates: {domains}"
        )
    domains_tuple = tuple(domains)
    domains_set = frozenset(domains_tuple)

    raw_tags = doc.get("tags", [])
    if not isinstance(raw_tags, list):
        raise QueryBankError(f"{src_path}::tags: must be a list")
    tags_tuple = tuple(str(t) for t in raw_tags)

    raw_entries = doc.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise QueryBankError(
            f"{src_path}::entries: must be a non-empty list of query entries"
        )

    queries: List[Query] = []
    seen_ids: set[str] = set()
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise QueryBankError(
                f"{src_path}::entries[{idx}]: must be a mapping"
            )
        q = _build_query(
            entry,
            src_path=src_path,
            idx=idx,
            bank_id=bank_id,
            bank_version=bank_version,
            allowed_locales=locales_set,
            allowed_domains=domains_set,
        )
        if q.id in seen_ids:
            raise QueryBankError(
                f"{src_path}::{q.id}: duplicate query id"
            )
        seen_ids.add(q.id)
        queries.append(q)

    bank = QueryBank(
        schema_version=schema_version,
        bank_id=bank_id,
        bank_version=bank_version,
        locales=locales_tuple,
        domains=domains_tuple,
        tags=tags_tuple,
        queries=tuple(queries),
        source_path=src_path,
    )

    # Log placeholder banks at INFO so reviewers see them in CI / smoke
    # output without needing to grep the YAML.
    n_placeholder = sum(1 for q in bank.queries if q.placeholder)
    if n_placeholder:
        logger.info(
            "query_bank: loaded %s v%s with %d placeholder query(ies) of %d total "
            "— downstream sovereign-AI scorecards will refuse production-ready "
            "headlines for trajectories that probed against placeholder queries",
            bank_id,
            bank_version,
            n_placeholder,
            len(bank.queries),
        )
    return bank


def _build_query(
    entry: Dict[str, Any],
    *,
    src_path: str,
    idx: int,
    bank_id: str,
    bank_version: str,
    allowed_locales: FrozenSet[str],
    allowed_domains: FrozenSet[str],
) -> Query:
    qid = _require_str(entry, "id", src_path, ctx=f"entries[{idx}]")
    ctx = f"{qid}"

    domain = _require_str(entry, "domain", src_path, ctx=ctx)
    if domain not in allowed_domains:
        raise QueryBankError(
            f"{src_path}::{ctx}::domain: {domain!r} is not in the bank's "
            f"declared domains {sorted(allowed_domains)}"
        )

    placeholder = entry.get("placeholder")
    if not isinstance(placeholder, bool):
        raise QueryBankError(
            f"{src_path}::{ctx}::placeholder: must be a bool, got "
            f"{type(placeholder).__name__}"
        )

    difficulty = _require_str(entry, "difficulty", src_path, ctx=ctx)
    if difficulty not in ALLOWED_DIFFICULTIES:
        raise QueryBankError(
            f"{src_path}::{ctx}::difficulty: {difficulty!r} not in "
            f"{sorted(ALLOWED_DIFFICULTIES)}"
        )

    raw_tags = entry.get("persona_tags")
    if not isinstance(raw_tags, list) or not raw_tags:
        raise QueryBankError(
            f"{src_path}::{ctx}::persona_tags: must be a non-empty list"
        )
    persona_tags: List[str] = []
    for ti, tag in enumerate(raw_tags):
        if not isinstance(tag, str) or not tag.strip():
            raise QueryBankError(
                f"{src_path}::{ctx}::persona_tags[{ti}]: must be a non-empty "
                "string"
            )
        persona_tags.append(tag.strip())

    concern = _require_str(entry, "concern", src_path, ctx=ctx)
    if not concern.strip():
        raise QueryBankError(
            f"{src_path}::{ctx}::concern: must be a non-empty string "
            "describing what failure mode the query is designed to surface"
        )

    raw_renderings = entry.get("renderings")
    if not isinstance(raw_renderings, dict) or not raw_renderings:
        raise QueryBankError(
            f"{src_path}::{ctx}::renderings: must be a non-empty mapping of "
            "locale → rendering text"
        )
    renderings: Dict[str, str] = {}
    for loc, text in raw_renderings.items():
        if not isinstance(loc, str):
            raise QueryBankError(
                f"{src_path}::{ctx}::renderings: keys must be locale id "
                f"strings, got {type(loc).__name__}"
            )
        if loc not in allowed_locales:
            raise QueryBankError(
                f"{src_path}::{ctx}::renderings: locale {loc!r} is not in "
                f"the bank's declared locales {sorted(allowed_locales)}"
            )
        if not isinstance(text, str) or not text.strip():
            raise QueryBankError(
                f"{src_path}::{ctx}::renderings[{loc}]: must be a non-empty "
                "string"
            )
        renderings[loc] = text.strip()

    raw_opt_out = entry.get("renderings_opted_out", [])
    if not isinstance(raw_opt_out, list):
        raise QueryBankError(
            f"{src_path}::{ctx}::renderings_opted_out: must be a list (or omit)"
        )
    opted_out: List[str] = []
    for oi, loc in enumerate(raw_opt_out):
        if not isinstance(loc, str) or not loc.strip():
            raise QueryBankError(
                f"{src_path}::{ctx}::renderings_opted_out[{oi}]: must be a "
                "non-empty locale id string"
            )
        if loc not in allowed_locales:
            raise QueryBankError(
                f"{src_path}::{ctx}::renderings_opted_out[{oi}]: locale "
                f"{loc!r} is not in the bank's declared locales — "
                "opt-out keys must reference real bank locales (catches typos)"
            )
        if loc in renderings:
            raise QueryBankError(
                f"{src_path}::{ctx}: locale {loc!r} appears in BOTH "
                "renderings and renderings_opted_out; pick one"
            )
        opted_out.append(loc)

    # Cross-product check: every locale in the bank's `locales` must
    # either have a rendering OR be in renderings_opted_out.
    missing = [
        loc for loc in allowed_locales
        if loc not in renderings and loc not in opted_out
    ]
    if missing:
        raise QueryBankError(
            f"{src_path}::{ctx}::renderings: missing rendering for locale(s) "
            f"{sorted(missing)}; either add a rendering or list the locale in "
            "renderings_opted_out (the latter only when the query has no "
            "meaningful cross-locale equivalent)"
        )

    raw_provenance = entry.get("provenance")
    if not isinstance(raw_provenance, dict):
        raise QueryBankError(
            f"{src_path}::{ctx}::provenance: must be a mapping"
        )
    provenance = _build_provenance(
        raw_provenance,
        src_path=src_path,
        ctx=ctx,
        placeholder=placeholder,
    )

    return Query(
        id=qid,
        domain=domain,
        placeholder=placeholder,
        difficulty=difficulty,
        persona_tags=tuple(persona_tags),
        concern=concern.strip(),
        renderings=renderings,
        renderings_opted_out=tuple(opted_out),
        provenance=provenance,
        bank_id=bank_id,
        bank_version=bank_version,
    )


def _build_provenance(
    p: Dict[str, Any],
    *,
    src_path: str,
    ctx: str,
    placeholder: bool,
) -> QueryProvenance:
    source = _require_str(p, "source", src_path, ctx=f"{ctx}::provenance")
    if source.strip() == "placeholder" and not placeholder:
        raise QueryBankError(
            f"{src_path}::{ctx}::provenance::source: 'placeholder' is only "
            "allowed when the entry's placeholder field is true"
        )
    last_reviewed = _require_str(
        p, "last_reviewed", src_path, ctx=f"{ctx}::provenance"
    )
    if not _ISO_DATE.match(last_reviewed):
        raise QueryBankError(
            f"{src_path}::{ctx}::provenance::last_reviewed: must be an "
            f"ISO-8601 date YYYY-MM-DD; got {last_reviewed!r}"
        )
    raw_refs = p.get("references", [])
    if not isinstance(raw_refs, list):
        raise QueryBankError(
            f"{src_path}::{ctx}::provenance::references: must be a list of strings"
        )
    refs: List[str] = []
    for ri, ref in enumerate(raw_refs):
        if not isinstance(ref, str) or not ref.strip():
            raise QueryBankError(
                f"{src_path}::{ctx}::provenance::references[{ri}]: must be a "
                "non-empty string"
            )
        refs.append(ref.strip())
    if not placeholder and not refs:
        # Non-placeholder entries SHOULD have at least one reference but
        # we don't enforce — log at INFO so a CI run surfaces it as a
        # quality signal without failing the load.
        logger.info(
            "query_bank: %s::%s carries placeholder=False but has no "
            "references; consider adding at least one citation",
            src_path,
            ctx,
        )
    return QueryProvenance(
        source=source.strip(),
        last_reviewed=last_reviewed.strip(),
        references=tuple(refs),
    )


def _require_str(
    d: Dict[str, Any], key: str, src_path: str, ctx: Optional[str] = None
) -> str:
    loc = f"{src_path}::{ctx}" if ctx else src_path
    if key not in d:
        raise QueryBankError(f"{loc}: missing required field {key!r}")
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        raise QueryBankError(
            f"{loc}: field {key!r} must be a non-empty string, got "
            f"{type(v).__name__}"
        )
    return v


# ---------------------------------------------------------------------------
# Process-local cache (parallel to fact_bank / probing_taxonomy)
# ---------------------------------------------------------------------------
#
# Unlike the fact bank and probing taxonomy, the query bank is NOT
# locale-keyed (the unit IS cross-locale). The cache here is keyed by
# the resolved file path, so multiple banks (sample.yaml, v1.yaml, …)
# can coexist in one process without colliding.

_QUERY_BANK_CACHE: Dict[str, QueryBank] = {}
_QUERY_BANK_CACHE_LOCK = threading.Lock()


def default_query_bank_path() -> Path:
    """Default on-disk path for the query bank.

    Discovered via ``core._assets.default_assets_dir`` so the loader
    works regardless of CWD. Asset directory matches the consuming
    probe name (``sov_ai_multilingual_parity``).
    """
    from usersim.engine.core._assets import probe_assets_dir
    return probe_assets_dir("sov_ai_multilingual_parity") / "sample.yaml"


def query_bank_path() -> Path:
    """Path the cached loader will try, honoring the env override.

    Override via ``USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK`` (e.g.
    ``USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK=/abs/path/v1.yaml``).
    """
    override = os.environ.get("USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK")
    if override:
        return Path(override)
    return default_query_bank_path()


def load_query_bank_default() -> QueryBank:
    """Return the cached query bank from the default path, loading on first use.

    Thread-safe. Shared across callers in this Python process so the
    probe and downstream scorers see the same ``bank_version`` without
    re-parsing the YAML.
    """
    path = str(query_bank_path())
    with _QUERY_BANK_CACHE_LOCK:
        cached = _QUERY_BANK_CACHE.get(path)
        if cached is not None:
            return cached
        bank = load_query_bank(path)
        _QUERY_BANK_CACHE[path] = bank
        logger.info(
            "query_bank: loaded %s v%s (%d query/queries, %d locale(s)) from %s",
            bank.bank_id,
            bank.bank_version,
            len(bank.queries),
            len(bank.locales),
            path,
        )
        return bank


def reset_query_bank_cache() -> None:
    """Drop all cached banks. For tests, CLI reloads, version bumps."""
    with _QUERY_BANK_CACHE_LOCK:
        _QUERY_BANK_CACHE.clear()
