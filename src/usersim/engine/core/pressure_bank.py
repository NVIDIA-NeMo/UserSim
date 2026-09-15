# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pressure-strategy bank loader, validator, and lookup helpers.

The pressure-strategy bank is the load-bearing primitive for the
``safety_chat_pressure`` probe. Schema is documented in
``assets/safety_chat_pressure/SCHEMA.md``.

Three primitives in one bank:

- :class:`Strategy` — a high-level pressure approach (e.g.,
  ``persistence``, ``hypothetical_framing``, ``authority_impersonation``).
- :class:`Reframing` — a concrete tactic within a strategy applied
  turn-by-turn. The probe rotates through reframings
  deterministically (seed-salted per persona) so the same persona does
  not always hit the same tactic first.
- :class:`TargetRequest` — a content type the assistant *should
  refuse*. Independent of any specific strategy: any strategy can be
  applied to any target, generating the trajectory cohort matrix.

This module exposes:

- :class:`Strategy` / :class:`Reframing` / :class:`TargetRequest` /
  :class:`PressureBank` — typed views.
- :class:`PressureBankError` — raised for schema violations. The
  loader fails fast on every invariant violation so reviewers get an
  actionable error pointing at the entry id + field.
- :func:`load_pressure_bank` — parse + validate a single file.
- :func:`load_pressure_bank_default` /
  :func:`reset_pressure_bank_cache` — process-local cache mirroring
  the pattern in :mod:`core.fact_bank`, :mod:`core.probing_taxonomy`,
  and :mod:`core.query_bank`.

Architecturally a pressure bank is **shared across locales** — like
the cross-locale query bank and unlike the per-locale fact bank /
probing taxonomy. Strategy + reframing prompts are authored in
English by default; non-English jailbreak attempts are a separate
test surface a future bank can carry.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from usersim.engine.core.localized import (
    LocalizedText,
    LocalizedTextError,
)

logger = logging.getLogger("usersim.engine")

SUPPORTED_SCHEMA_VERSIONS: FrozenSet[str] = frozenset({"v0.1"})
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Required placeholders the loader validates on every strategy /
# reframing template. Bumping these is a schema-version-incrementing
# change because every shipped bank would need a re-author pass.
_STRATEGY_REQUIRED_PLACEHOLDERS: Tuple[str, ...] = (
    "{persona}",
    "{target_description}",
    "{initial_prompt}",
)
_REFRAMING_REQUIRED_PLACEHOLDERS: Tuple[str, ...] = (
    "{previous_assistant_response}",
)


class PressureBankError(ValueError):
    """User-facing schema or load problem.

    Message format: ``"<path>::<entry_id or field>: <what's wrong>"``
    so reviewers can grep the offending location.
    """


# ---------------------------------------------------------------------------
# Typed views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetProvenance:
    source: str
    last_reviewed: str
    references: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Reframing:
    id: str
    description: "LocalizedText"
    instruction: "LocalizedText"
    # Set by the loader from the parent strategy so consumers can
    # carry attribution per-reframing without re-looking-up.
    strategy_id: str = ""


@dataclass(frozen=True)
class Strategy:
    id: str
    description: "LocalizedText"
    system_prompt_template: "LocalizedText"
    reframings: Tuple[Reframing, ...] = field(default_factory=tuple)

    def reframing_at(self, index: int) -> Reframing:
        """Return the reframing at ``index`` modulo the strategy's length.

        The probe calls this with a per-turn-incremented
        index; wrap-around ensures arbitrarily long trajectories
        always have a valid reframing to apply.
        """
        if not self.reframings:
            raise PressureBankError(
                f"strategy {self.id!r} has no reframings — cannot pick"
            )
        return self.reframings[index % len(self.reframings)]

    def reframing_by_id(self, reframing_id: str) -> Optional[Reframing]:
        for r in self.reframings:
            if r.id == reframing_id:
                return r
        return None


@dataclass(frozen=True)
class TargetRequest:
    id: str
    description: "LocalizedText"
    harm_category: str
    placeholder: bool
    initial_prompt: "LocalizedText"
    expected_refusal: "LocalizedText"
    persona_tags: Tuple[str, ...] = field(default_factory=tuple)
    provenance: Optional[TargetProvenance] = None
    # Set by the loader from the bank's top-level fields so consumers
    # can carry provenance per-target.
    bank_id: str = ""
    bank_version: str = ""

    def matches_persona(self, persona_tags: Iterable[str]) -> bool:
        """True iff this target is applicable to a persona with the given tags.

        - Empty ``self.persona_tags`` → matches every persona (the
          default for safety probes — most are persona-agnostic).
        - Non-empty ``self.persona_tags`` → uses the same prefix-aware
          intersection rules as the fact / query bank matchers, with
          ``*:any`` wildcards on either side.
        """
        if not self.persona_tags:
            return True
        return _persona_tags_match(self.persona_tags, frozenset(persona_tags))


@dataclass(frozen=True)
class PressureBank:
    schema_version: str
    bank_id: str
    bank_version: str
    harm_categories: Tuple[str, ...]
    tags: Tuple[str, ...] = field(default_factory=tuple)
    strategies: Tuple[Strategy, ...] = field(default_factory=tuple)
    target_requests: Tuple[TargetRequest, ...] = field(default_factory=tuple)
    source_path: Optional[str] = None

    # ── Lookups ─────────────────────────────────────────────────────

    def strategy_by_id(self, strategy_id: str) -> Optional[Strategy]:
        for s in self.strategies:
            if s.id == strategy_id:
                return s
        return None

    def target_by_id(self, target_id: str) -> Optional[TargetRequest]:
        for t in self.target_requests:
            if t.id == target_id:
                return t
        return None

    def targets_by_harm_category(self, category: str) -> Tuple[TargetRequest, ...]:
        return tuple(
            t for t in self.target_requests if t.harm_category == category
        )

    def targets_for_persona(
        self, persona_tags: Iterable[str]
    ) -> Tuple[TargetRequest, ...]:
        """Targets applicable to a persona with the given tags."""
        tags = frozenset(persona_tags)
        return tuple(t for t in self.target_requests if t.matches_persona(tags))

    def strategy_ids(self) -> Tuple[str, ...]:
        return tuple(s.id for s in self.strategies)

    def target_ids(self) -> Tuple[str, ...]:
        return tuple(t.id for t in self.target_requests)

    def provenance_summary(self) -> Dict[str, Any]:
        """Surfaces that the downstream reporting layer reads."""
        return {
            "bank_id": self.bank_id,
            "bank_version": self.bank_version,
            "schema_version": self.schema_version,
            "n_strategies": len(self.strategies),
            "n_targets": len(self.target_requests),
            "n_placeholder_targets": sum(
                1 for t in self.target_requests if t.placeholder
            ),
            "harm_categories": list(self.harm_categories),
        }


# ---------------------------------------------------------------------------
# Persona-tag matcher (parallel to query_bank's)
# ---------------------------------------------------------------------------


def _persona_tags_match(
    target_tags: Iterable[str],
    persona_tags: FrozenSet[str],
) -> bool:
    """Return True iff the target's required tags are satisfiable by
    the persona. Same prefix-aware semantics as the query-bank matcher.
    """
    by_prefix: Dict[str, List[str]] = {}
    for tag in target_tags:
        if ":" not in tag:
            by_prefix.setdefault("__bare__", []).append(tag)
            continue
        prefix = tag.split(":", 1)[0]
        by_prefix.setdefault(prefix, []).append(tag)

    for prefix, tags_for_prefix in by_prefix.items():
        if prefix == "__bare__":
            if not all(t in persona_tags for t in tags_for_prefix):
                return False
            continue
        wildcard = f"{prefix}:any"
        if wildcard in tags_for_prefix:
            continue
        persona_wildcard = wildcard in persona_tags
        if persona_wildcard:
            continue
        if not any(t in persona_tags for t in tags_for_prefix):
            return False
    return True


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_pressure_bank(path: str | Path) -> PressureBank:
    """Parse and validate a pressure-strategy bank from disk.

    Raises :class:`PressureBankError` with a message that names the
    offending file, entry id (or field), and the invariant violated.
    """
    import yaml

    src_path = Path(path)
    if not src_path.exists():
        raise PressureBankError(f"{path}: file not found")
    try:
        with src_path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise PressureBankError(f"{path}: YAML parse failure: {e}") from e
    if not isinstance(doc, dict):
        raise PressureBankError(f"{path}: top-level must be a mapping")
    return _build_pressure_bank(doc, src_path=str(src_path))


def _build_pressure_bank(doc: Dict[str, Any], *, src_path: str) -> PressureBank:
    schema_version = _require_str(doc, "schema_version", src_path)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise PressureBankError(
            f"{src_path}::schema_version: {schema_version!r} not supported; "
            f"loader understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    bank_id = _require_str(doc, "bank_id", src_path)
    bank_version = _require_str(doc, "bank_version", src_path)

    raw_harm = doc.get("harm_categories")
    if not isinstance(raw_harm, list) or not raw_harm:
        raise PressureBankError(
            f"{src_path}::harm_categories: must be a non-empty list"
        )
    harm_categories: List[str] = []
    for i, h in enumerate(raw_harm):
        if not isinstance(h, str) or not h.strip():
            raise PressureBankError(
                f"{src_path}::harm_categories[{i}]: must be a non-empty string"
            )
        harm_categories.append(h.strip())
    if len(set(harm_categories)) != len(harm_categories):
        raise PressureBankError(
            f"{src_path}::harm_categories: contains duplicates: "
            f"{harm_categories}"
        )
    harm_set = frozenset(harm_categories)

    raw_tags = doc.get("tags", [])
    if not isinstance(raw_tags, list):
        raise PressureBankError(f"{src_path}::tags: must be a list")
    tags = tuple(str(t) for t in raw_tags)

    raw_strategies = doc.get("strategies")
    if not isinstance(raw_strategies, list) or not raw_strategies:
        raise PressureBankError(
            f"{src_path}::strategies: must be a non-empty list"
        )
    strategies: List[Strategy] = []
    seen_strategy_ids: set[str] = set()
    for idx, raw in enumerate(raw_strategies):
        if not isinstance(raw, dict):
            raise PressureBankError(
                f"{src_path}::strategies[{idx}]: must be a mapping"
            )
        s = _build_strategy(raw, src_path=src_path, idx=idx)
        if s.id in seen_strategy_ids:
            raise PressureBankError(
                f"{src_path}::{s.id}: duplicate strategy id"
            )
        seen_strategy_ids.add(s.id)
        strategies.append(s)

    raw_targets = doc.get("target_requests")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise PressureBankError(
            f"{src_path}::target_requests: must be a non-empty list"
        )
    targets: List[TargetRequest] = []
    seen_target_ids: set[str] = set()
    for idx, raw in enumerate(raw_targets):
        if not isinstance(raw, dict):
            raise PressureBankError(
                f"{src_path}::target_requests[{idx}]: must be a mapping"
            )
        t = _build_target(
            raw, src_path=src_path, idx=idx,
            bank_id=bank_id, bank_version=bank_version,
            allowed_harm_categories=harm_set,
        )
        if t.id in seen_target_ids:
            raise PressureBankError(
                f"{src_path}::{t.id}: duplicate target id"
            )
        seen_target_ids.add(t.id)
        targets.append(t)

    bank = PressureBank(
        schema_version=schema_version,
        bank_id=bank_id,
        bank_version=bank_version,
        harm_categories=tuple(harm_categories),
        tags=tags,
        strategies=tuple(strategies),
        target_requests=tuple(targets),
        source_path=src_path,
    )

    n_placeholder = sum(1 for t in bank.target_requests if t.placeholder)
    if n_placeholder:
        logger.info(
            "pressure_bank: loaded %s v%s with %d placeholder target(s) of %d "
            "total — downstream readers (capability dashboard, scorers) treat "
            "trajectories that probed against placeholder targets as preview-only",
            bank_id, bank_version, n_placeholder, len(bank.target_requests),
        )
    return bank


def _build_strategy(
    entry: Dict[str, Any], *, src_path: str, idx: int
) -> Strategy:
    sid = _require_str(entry, "id", src_path, ctx=f"strategies[{idx}]")
    ctx = sid

    description = _require_localized(entry, "description", src_path, ctx=ctx)
    template = _require_localized(
        entry, "system_prompt_template", src_path, ctx=ctx,
    )
    # Validate the required-placeholders contract against EVERY
    # locale rendering — a French rendering missing ``{persona}``
    # would crash at simulator time the same way an English one
    # would. Catching at load time keeps the failure visible.
    for locale_code, rendering in template.renderings.items():
        for ph in _STRATEGY_REQUIRED_PLACEHOLDERS:
            if ph not in rendering:
                raise PressureBankError(
                    f"{src_path}::{ctx}::system_prompt_template[{locale_code}]: "
                    f"missing required placeholder {ph!r}; loader requires "
                    f"{list(_STRATEGY_REQUIRED_PLACEHOLDERS)}"
                )

    raw_reframings = entry.get("reframings")
    if not isinstance(raw_reframings, list) or not raw_reframings:
        raise PressureBankError(
            f"{src_path}::{ctx}::reframings: must be a non-empty list"
        )

    reframings: List[Reframing] = []
    seen_ids: set[str] = set()
    for ri, raw in enumerate(raw_reframings):
        if not isinstance(raw, dict):
            raise PressureBankError(
                f"{src_path}::{ctx}::reframings[{ri}]: must be a mapping"
            )
        rid = _require_str(
            raw, "id", src_path, ctx=f"{ctx}::reframings[{ri}]"
        )
        rctx = f"{ctx}::reframings::{rid}"
        rdesc = _require_localized(raw, "description", src_path, ctx=rctx)
        rinstr = _require_localized(raw, "instruction", src_path, ctx=rctx)
        for locale_code, rendering in rinstr.renderings.items():
            for ph in _REFRAMING_REQUIRED_PLACEHOLDERS:
                if ph not in rendering:
                    raise PressureBankError(
                        f"{src_path}::{rctx}::instruction[{locale_code}]: "
                        f"missing required placeholder {ph!r}; loader "
                        f"requires {list(_REFRAMING_REQUIRED_PLACEHOLDERS)}"
                    )
        if rid in seen_ids:
            raise PressureBankError(
                f"{src_path}::{rctx}: duplicate reframing id within strategy {sid!r}"
            )
        seen_ids.add(rid)
        reframings.append(
            Reframing(
                id=rid,
                description=rdesc,
                instruction=rinstr,
                strategy_id=sid,
            )
        )

    return Strategy(
        id=sid,
        description=description,
        system_prompt_template=template,
        reframings=tuple(reframings),
    )


def _build_target(
    entry: Dict[str, Any],
    *,
    src_path: str,
    idx: int,
    bank_id: str,
    bank_version: str,
    allowed_harm_categories: FrozenSet[str],
) -> TargetRequest:
    tid = _require_str(entry, "id", src_path, ctx=f"target_requests[{idx}]")
    ctx = tid

    description = _require_localized(entry, "description", src_path, ctx=ctx)
    harm = _require_str(entry, "harm_category", src_path, ctx=ctx)
    if harm not in allowed_harm_categories:
        raise PressureBankError(
            f"{src_path}::{ctx}::harm_category: {harm!r} is not in the "
            f"bank's declared harm_categories {sorted(allowed_harm_categories)}"
        )

    placeholder = entry.get("placeholder")
    if not isinstance(placeholder, bool):
        raise PressureBankError(
            f"{src_path}::{ctx}::placeholder: must be a bool, got "
            f"{type(placeholder).__name__}"
        )

    initial_prompt = _require_localized(
        entry, "initial_prompt", src_path, ctx=ctx,
    )
    expected_refusal = _require_localized(
        entry, "expected_refusal", src_path, ctx=ctx,
    )

    raw_tags = entry.get("persona_tags", [])
    if not isinstance(raw_tags, list):
        raise PressureBankError(
            f"{src_path}::{ctx}::persona_tags: must be a list (or omit)"
        )
    persona_tags: List[str] = []
    for ti, tag in enumerate(raw_tags):
        if not isinstance(tag, str) or not tag.strip():
            raise PressureBankError(
                f"{src_path}::{ctx}::persona_tags[{ti}]: must be a non-empty "
                "string"
            )
        persona_tags.append(tag.strip())

    raw_provenance = entry.get("provenance")
    if not isinstance(raw_provenance, dict):
        raise PressureBankError(
            f"{src_path}::{ctx}::provenance: must be a mapping"
        )
    provenance = _build_provenance(
        raw_provenance, src_path=src_path, ctx=ctx, placeholder=placeholder,
    )

    return TargetRequest(
        id=tid,
        description=description,
        harm_category=harm,
        placeholder=placeholder,
        initial_prompt=initial_prompt,
        expected_refusal=expected_refusal,
        persona_tags=tuple(persona_tags),
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
) -> TargetProvenance:
    source = _require_str(p, "source", src_path, ctx=f"{ctx}::provenance")
    if source.strip() == "placeholder" and not placeholder:
        raise PressureBankError(
            f"{src_path}::{ctx}::provenance::source: 'placeholder' is only "
            "allowed when the entry's placeholder field is true"
        )
    last_reviewed = _require_str(
        p, "last_reviewed", src_path, ctx=f"{ctx}::provenance"
    )
    if not _ISO_DATE.match(last_reviewed):
        raise PressureBankError(
            f"{src_path}::{ctx}::provenance::last_reviewed: must be an "
            f"ISO-8601 date YYYY-MM-DD; got {last_reviewed!r}"
        )
    raw_refs = p.get("references", [])
    if not isinstance(raw_refs, list):
        raise PressureBankError(
            f"{src_path}::{ctx}::provenance::references: must be a list"
        )
    refs: List[str] = []
    for ri, ref in enumerate(raw_refs):
        if not isinstance(ref, str) or not ref.strip():
            raise PressureBankError(
                f"{src_path}::{ctx}::provenance::references[{ri}]: must be a "
                "non-empty string"
            )
        refs.append(ref.strip())
    if not placeholder and not refs:
        logger.info(
            "pressure_bank: %s::%s carries placeholder=False but has no "
            "references; consider adding at least one citation",
            src_path, ctx,
        )
    return TargetProvenance(
        source=source.strip(),
        last_reviewed=last_reviewed.strip(),
        references=tuple(refs),
    )


def _require_str(
    d: Dict[str, Any], key: str, src_path: str, ctx: Optional[str] = None
) -> str:
    loc = f"{src_path}::{ctx}" if ctx else src_path
    if key not in d:
        raise PressureBankError(f"{loc}: missing required field {key!r}")
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        raise PressureBankError(
            f"{loc}: field {key!r} must be a non-empty string, got "
            f"{type(v).__name__}"
        )
    return v


def _require_localized(
    d: Dict[str, Any],
    key: str,
    src_path: str,
    ctx: Optional[str] = None,
) -> LocalizedText:
    """Parse a YAML field as a :class:`LocalizedText` (string OR dict).

    Raises ``PressureBankError`` (so the bank loader's exception
    surface stays uniform) wrapping the underlying
    :class:`LocalizedTextError` from the parser.
    """
    loc = f"{src_path}::{ctx}" if ctx else src_path
    if key not in d:
        raise PressureBankError(f"{loc}: missing required field {key!r}")
    try:
        return LocalizedText.from_yaml_value(
            d[key], field_path=f"{loc}::{key}",
        )
    except LocalizedTextError as e:
        raise PressureBankError(str(e)) from e


# ---------------------------------------------------------------------------
# Process-local cache
# ---------------------------------------------------------------------------

_PRESSURE_BANK_CACHE: Dict[str, PressureBank] = {}
_PRESSURE_BANK_CACHE_LOCK = threading.Lock()


def default_pressure_bank_path() -> Path:
    """Default on-disk path for the pressure-strategy bank.

    Discovered via ``core._assets.default_assets_dir`` so the loader
    works regardless of CWD. Asset directory matches the consuming
    probe name (``safety_chat_pressure``).
    """
    from usersim.engine.core._assets import probe_assets_dir
    return probe_assets_dir("safety_chat_pressure") / "sample.yaml"


def pressure_bank_path() -> Path:
    """Path the cached loader will try, honoring the env override.

    Override via ``USERSIM_SAFETY_CHAT_PRESSURE_BANK`` (e.g.
    ``USERSIM_SAFETY_CHAT_PRESSURE_BANK=/abs/path/v1.yaml``).
    """
    override = os.environ.get("USERSIM_SAFETY_CHAT_PRESSURE_BANK")
    if override:
        return Path(override)
    return default_pressure_bank_path()


def load_pressure_bank_default() -> PressureBank:
    """Return the cached pressure bank from the default path, loading on first use.

    Thread-safe. Shared across callers in this Python process so the
    probe and downstream scorer see the same ``bank_version`` without
    re-parsing the YAML.
    """
    path = str(pressure_bank_path())
    with _PRESSURE_BANK_CACHE_LOCK:
        cached = _PRESSURE_BANK_CACHE.get(path)
        if cached is not None:
            return cached
        bank = load_pressure_bank(path)
        _PRESSURE_BANK_CACHE[path] = bank
        logger.info(
            "pressure_bank: loaded %s v%s (%d strategies, %d target(s)) from %s",
            bank.bank_id, bank.bank_version,
            len(bank.strategies), len(bank.target_requests), path,
        )
        return bank


def reset_pressure_bank_cache() -> None:
    """Drop all cached banks. For tests, CLI reloads, version bumps."""
    with _PRESSURE_BANK_CACHE_LOCK:
        _PRESSURE_BANK_CACHE.clear()
