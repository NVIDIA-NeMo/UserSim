# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic-action bank loader, validator, and lookup helpers.

The agentic-action bank is the load-bearing primitive for the
``safety_agentic`` probe. Schema is documented in
``assets/safety_agentic/SCHEMA.md``.

Three primitives:

- :class:`SubProtocol` — a high-level failure-mode identifier (the
  shipped default has ``unsanctioned_action`` / ``scope_creep`` /
  ``compromised_inputs`` / ``consequence_disclosure``).
- :class:`ToolSpec` — a simulated tool definition (name, description,
  JSON-Schema parameters block).
- :class:`ActionRequest` — a `(persona-prompt, expected-action,
  simulated-tools, mock-responses)` tuple that the probe consumes
  to drive one trajectory.

The loader **resolves** ``sanctioned_action_name`` and
``tempted_action_name`` against the shipped
``assets/safety_agentic/action_taxonomy.yaml`` at load time and rejects banks that
reference an unknown action — taxonomy drift is a thing the loader
should surface, not silently allow.

This module exposes:

- :class:`SubProtocol` / :class:`ToolSpec` / :class:`ActionRequest` /
  :class:`AgenticBank` — typed views.
- :class:`AgenticBankError` — raised for schema violations.
- :func:`load_agentic_bank` — parse + validate a single file.
- :func:`load_agentic_bank_default` /
  :func:`reset_agentic_bank_cache` — process-local cache mirroring
  the pattern in :mod:`core.fact_bank`, :mod:`core.probing_taxonomy`,
  :mod:`core.query_bank`, and :mod:`core.pressure_bank`.

Architecturally an agentic-action bank is **shared across locales** —
like the cross-locale query bank and the pressure-strategy bank. The
user-prompt and expected-behavior fields are authored in English by
default; non-English agentic-safety probes are a separate test surface
a future locale-aware bank can carry.
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


class AgenticBankError(ValueError):
    """User-facing schema or load problem.

    Message format: ``"<path>::<entry_id or field>: <what's wrong>"``
    so reviewers can grep the offending location.
    """


# ---------------------------------------------------------------------------
# Typed views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionProvenance:
    source: str
    last_reviewed: str
    references: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolSpec:
    """A simulated tool exposed to the assistant for one trajectory."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MockToolResponse:
    """A pre-canned tool response. ``content`` is the verbatim string
    appended to the conversation as the tool's reply."""

    role: str  # always "tool" — kept explicit for forward-compat
    name: str  # tool name; matches ToolSpec.name
    content: str


@dataclass(frozen=True)
class ActionRequest:
    id: str
    sub_protocol: str
    placeholder: bool
    description: "LocalizedText"
    sanctioned_action_name: Optional[str]  # None = "no action; refuse"
    tempted_action_name: Optional[str]
    initial_user_prompt: "LocalizedText"
    expected_behavior: "LocalizedText"
    simulated_tools: Tuple[ToolSpec, ...]
    mock_tool_responses: Dict[str, MockToolResponse] = field(default_factory=dict)
    persona_tags: Tuple[str, ...] = field(default_factory=tuple)
    provenance: Optional[ActionProvenance] = None
    # Set by the loader from the bank's top-level fields so consumers
    # can carry attribution per-entry without re-looking-up.
    bank_id: str = ""
    bank_version: str = ""

    def matches_persona(self, persona_tags: Iterable[str]) -> bool:
        """True iff this action request applies to a persona with the given tags.

        Empty ``persona_tags`` matches every persona (the common case
        for agentic-safety probes — refusal-to-overreach is persona-
        agnostic). Same prefix-aware matcher as the other curated
        asset banks.
        """
        if not self.persona_tags:
            return True
        return _persona_tags_match(self.persona_tags, frozenset(persona_tags))


@dataclass(frozen=True)
class AgenticBank:
    schema_version: str
    bank_id: str
    bank_version: str
    sub_protocols: Tuple[str, ...]
    tags: Tuple[str, ...] = field(default_factory=tuple)
    action_requests: Tuple[ActionRequest, ...] = field(default_factory=tuple)
    source_path: Optional[str] = None

    def by_id(self, request_id: str) -> Optional[ActionRequest]:
        for r in self.action_requests:
            if r.id == request_id:
                return r
        return None

    def by_sub_protocol(self, sub_protocol: str) -> Tuple[ActionRequest, ...]:
        return tuple(
            r for r in self.action_requests if r.sub_protocol == sub_protocol
        )

    def for_persona(self, persona_tags: Iterable[str]) -> Tuple[ActionRequest, ...]:
        tags = frozenset(persona_tags)
        return tuple(
            r for r in self.action_requests if r.matches_persona(tags)
        )

    def request_ids(self) -> Tuple[str, ...]:
        return tuple(r.id for r in self.action_requests)

    def provenance_summary(self) -> Dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "bank_version": self.bank_version,
            "schema_version": self.schema_version,
            "n_requests": len(self.action_requests),
            "n_placeholder_requests": sum(
                1 for r in self.action_requests if r.placeholder
            ),
            "sub_protocols": list(self.sub_protocols),
        }


# ---------------------------------------------------------------------------
# Persona-tag matcher (parallel to query / pressure / fact bank)
# ---------------------------------------------------------------------------


def _persona_tags_match(
    target_tags: Iterable[str],
    persona_tags: FrozenSet[str],
) -> bool:
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
# Action-taxonomy resolver — load + cache + name lookup
# ---------------------------------------------------------------------------


def _default_action_taxonomy_path() -> Path:
    """Discover the shipped ``assets/safety_agentic/action_taxonomy.yaml``
    from the packaged asset tree. Resolution is independent of CWD; set
    ``USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY`` to override."""
    from usersim.engine.core._assets import probe_assets_dir
    return probe_assets_dir("safety_agentic") / "action_taxonomy.yaml"


_ACTION_TAXONOMY_CACHE: Dict[str, FrozenSet[str]] = {}
_ACTION_TAXONOMY_CACHE_LOCK = threading.Lock()


def _load_action_taxonomy_names(path: Path) -> FrozenSet[str]:
    """Load the action taxonomy and return the set of valid ``name``s.

    The taxonomy file's full structure is rich (risk_tier,
    consequence_disclosure_required, p_user_detect, …); this loader
    only needs the set of valid action names so the agentic-bank
    loader can validate ``sanctioned_action_name`` and
    ``tempted_action_name`` references. The full taxonomy is read by
    the ``safety_agentic`` scorer when it joins on action names.
    """
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise AgenticBankError(
            f"action_taxonomy: {path}: top-level must be a mapping"
        )
    entries = doc.get("entries")
    if not isinstance(entries, list):
        raise AgenticBankError(
            f"action_taxonomy: {path}: missing top-level `entries` list"
        )
    names: List[str] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise AgenticBankError(
                f"action_taxonomy: {path}::entries[{i}]: must be a mapping"
            )
        name = e.get("name")
        if not isinstance(name, str) or not name:
            raise AgenticBankError(
                f"action_taxonomy: {path}::entries[{i}]: missing or "
                "non-string `name`"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise AgenticBankError(
            f"action_taxonomy: {path}: duplicate action names"
        )
    return frozenset(names)


def _action_taxonomy_names() -> FrozenSet[str]:
    """Return the cached set of valid action-taxonomy names.

    Honours ``USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY`` env override
    (parallel to the agentic-bank env override).
    """
    override = os.environ.get("USERSIM_SAFETY_AGENTIC_ACTION_TAXONOMY")
    path = Path(override) if override else _default_action_taxonomy_path()
    cache_key = str(path)
    with _ACTION_TAXONOMY_CACHE_LOCK:
        cached = _ACTION_TAXONOMY_CACHE.get(cache_key)
        if cached is not None:
            return cached
        names = _load_action_taxonomy_names(path)
        _ACTION_TAXONOMY_CACHE[cache_key] = names
        logger.info(
            "action_taxonomy: loaded %d valid action name(s) from %s",
            len(names), path,
        )
        return names


def reset_action_taxonomy_cache() -> None:
    """Drop the cached action-taxonomy name set. For tests / version bumps."""
    with _ACTION_TAXONOMY_CACHE_LOCK:
        _ACTION_TAXONOMY_CACHE.clear()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_agentic_bank(path: str | Path) -> AgenticBank:
    """Parse and validate an agentic-action bank from disk.

    Raises :class:`AgenticBankError` with a message that names the
    offending file, entry id (or field), and the invariant violated.
    """
    import yaml

    src_path = Path(path)
    if not src_path.exists():
        raise AgenticBankError(f"{path}: file not found")
    try:
        with src_path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise AgenticBankError(f"{path}: YAML parse failure: {e}") from e
    if not isinstance(doc, dict):
        raise AgenticBankError(f"{path}: top-level must be a mapping")
    return _build_bank(doc, src_path=str(src_path))


def _build_bank(doc: Dict[str, Any], *, src_path: str) -> AgenticBank:
    schema_version = _require_str(doc, "schema_version", src_path)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise AgenticBankError(
            f"{src_path}::schema_version: {schema_version!r} not supported; "
            f"loader understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    bank_id = _require_str(doc, "bank_id", src_path)
    bank_version = _require_str(doc, "bank_version", src_path)

    raw_sps = doc.get("sub_protocols")
    if not isinstance(raw_sps, list) or not raw_sps:
        raise AgenticBankError(
            f"{src_path}::sub_protocols: must be a non-empty list"
        )
    sub_protocols: List[str] = []
    for i, sp in enumerate(raw_sps):
        if not isinstance(sp, str) or not sp.strip():
            raise AgenticBankError(
                f"{src_path}::sub_protocols[{i}]: must be a non-empty string"
            )
        sub_protocols.append(sp.strip())
    if len(set(sub_protocols)) != len(sub_protocols):
        raise AgenticBankError(
            f"{src_path}::sub_protocols: contains duplicates: {sub_protocols}"
        )
    sp_set = frozenset(sub_protocols)

    raw_tags = doc.get("tags", [])
    if not isinstance(raw_tags, list):
        raise AgenticBankError(f"{src_path}::tags: must be a list")
    tags = tuple(str(t) for t in raw_tags)

    raw_entries = doc.get("action_requests")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise AgenticBankError(
            f"{src_path}::action_requests: must be a non-empty list"
        )

    valid_action_names = _action_taxonomy_names()

    requests: List[ActionRequest] = []
    seen_ids: set[str] = set()
    for idx, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise AgenticBankError(
                f"{src_path}::action_requests[{idx}]: must be a mapping"
            )
        ar = _build_action_request(
            raw,
            src_path=src_path,
            idx=idx,
            bank_id=bank_id,
            bank_version=bank_version,
            sub_protocol_set=sp_set,
            valid_action_names=valid_action_names,
        )
        if ar.id in seen_ids:
            raise AgenticBankError(
                f"{src_path}::{ar.id}: duplicate action_request id"
            )
        seen_ids.add(ar.id)
        requests.append(ar)

    bank = AgenticBank(
        schema_version=schema_version,
        bank_id=bank_id,
        bank_version=bank_version,
        sub_protocols=tuple(sub_protocols),
        tags=tags,
        action_requests=tuple(requests),
        source_path=src_path,
    )

    n_placeholder = sum(1 for r in bank.action_requests if r.placeholder)
    if n_placeholder:
        logger.info(
            "agentic_bank: loaded %s v%s with %d placeholder action request(s) "
            "of %d total — downstream readers (capability dashboard, scorers) "
            "treat placeholder-tainted trajectories as preview-only",
            bank_id, bank_version, n_placeholder, len(bank.action_requests),
        )
    return bank


def _build_action_request(
    entry: Dict[str, Any],
    *,
    src_path: str,
    idx: int,
    bank_id: str,
    bank_version: str,
    sub_protocol_set: FrozenSet[str],
    valid_action_names: FrozenSet[str],
) -> ActionRequest:
    rid = _require_str(entry, "id", src_path, ctx=f"action_requests[{idx}]")
    ctx = rid

    sub_protocol = _require_str(entry, "sub_protocol", src_path, ctx=ctx)
    if sub_protocol not in sub_protocol_set:
        raise AgenticBankError(
            f"{src_path}::{ctx}::sub_protocol: {sub_protocol!r} is not in "
            f"the bank's declared sub_protocols {sorted(sub_protocol_set)}"
        )

    placeholder = entry.get("placeholder")
    if not isinstance(placeholder, bool):
        raise AgenticBankError(
            f"{src_path}::{ctx}::placeholder: must be a bool, got "
            f"{type(placeholder).__name__}"
        )

    description = _require_localized_in_bank(
        entry, "description", src_path, ctx=ctx,
    )
    expected_behavior = _require_localized_in_bank(
        entry, "expected_behavior", src_path, ctx=ctx,
    )
    initial_user_prompt = _require_localized_in_bank(
        entry, "initial_user_prompt", src_path, ctx=ctx,
    )

    # sanctioned_action_name: either a valid taxonomy name OR null
    # (meaning "the assistant should refuse to call any tool"; e.g., a
    # compromised_inputs probe where the right behaviour is to flag).
    sanctioned_raw = entry.get("sanctioned_action_name", "<missing>")
    if sanctioned_raw == "<missing>":
        raise AgenticBankError(
            f"{src_path}::{ctx}::sanctioned_action_name: missing required "
            "field (set to null to indicate 'assistant should refuse')"
        )
    if sanctioned_raw is None:
        sanctioned_action_name = None
    elif isinstance(sanctioned_raw, str) and sanctioned_raw.strip():
        sanctioned_action_name = sanctioned_raw.strip()
        if sanctioned_action_name not in valid_action_names:
            raise AgenticBankError(
                f"{src_path}::{ctx}::sanctioned_action_name: "
                f"{sanctioned_action_name!r} not found in "
                "assets/safety_agentic/action_taxonomy.yaml — fix the bank entry or add "
                "the action to the taxonomy"
            )
    else:
        raise AgenticBankError(
            f"{src_path}::{ctx}::sanctioned_action_name: must be a "
            "non-empty string or null"
        )

    tempted_raw = entry.get("tempted_action_name")
    tempted_action_name: Optional[str]
    if tempted_raw is None:
        tempted_action_name = None
    elif isinstance(tempted_raw, str) and tempted_raw.strip():
        tempted_action_name = tempted_raw.strip()
        if tempted_action_name not in valid_action_names:
            raise AgenticBankError(
                f"{src_path}::{ctx}::tempted_action_name: "
                f"{tempted_action_name!r} not found in "
                "assets/safety_agentic/action_taxonomy.yaml"
            )
    else:
        raise AgenticBankError(
            f"{src_path}::{ctx}::tempted_action_name: must be a non-empty "
            "string or null/omitted"
        )

    raw_tools = entry.get("simulated_tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise AgenticBankError(
            f"{src_path}::{ctx}::simulated_tools: must be a non-empty list"
        )
    tools: List[ToolSpec] = []
    seen_tool_names: set[str] = set()
    for ti, raw in enumerate(raw_tools):
        if not isinstance(raw, dict):
            raise AgenticBankError(
                f"{src_path}::{ctx}::simulated_tools[{ti}]: must be a mapping"
            )
        tname = _require_str(
            raw, "name", src_path, ctx=f"{ctx}::simulated_tools[{ti}]"
        )
        tdesc = _require_str(
            raw, "description", src_path,
            ctx=f"{ctx}::simulated_tools[{ti}]::({tname})",
        )
        params = raw.get("parameters", {})
        if not isinstance(params, dict):
            raise AgenticBankError(
                f"{src_path}::{ctx}::simulated_tools::{tname}::parameters: "
                "must be a JSON-Schema-shaped mapping"
            )
        if tname in seen_tool_names:
            raise AgenticBankError(
                f"{src_path}::{ctx}::simulated_tools::{tname}: duplicate "
                "tool name"
            )
        seen_tool_names.add(tname)
        tools.append(ToolSpec(
            name=tname.strip(), description=tdesc.strip(), parameters=params,
        ))

    raw_responses = entry.get("mock_tool_responses", {}) or {}
    if not isinstance(raw_responses, dict):
        raise AgenticBankError(
            f"{src_path}::{ctx}::mock_tool_responses: must be a mapping "
            "(or omit)"
        )
    mock_responses: Dict[str, MockToolResponse] = {}
    for tname, raw in raw_responses.items():
        if not isinstance(tname, str) or not tname:
            raise AgenticBankError(
                f"{src_path}::{ctx}::mock_tool_responses: keys must be "
                "non-empty tool-name strings"
            )
        if tname not in seen_tool_names:
            raise AgenticBankError(
                f"{src_path}::{ctx}::mock_tool_responses[{tname}]: tool name "
                "is not declared in simulated_tools — typos here would mean "
                "the canned response never fires"
            )
        if not isinstance(raw, dict):
            raise AgenticBankError(
                f"{src_path}::{ctx}::mock_tool_responses[{tname}]: must be a "
                "mapping with role/name/content"
            )
        role = _require_str(
            raw, "role", src_path,
            ctx=f"{ctx}::mock_tool_responses::{tname}",
        )
        if role != "tool":
            raise AgenticBankError(
                f"{src_path}::{ctx}::mock_tool_responses[{tname}]::role: "
                f"must be 'tool', got {role!r}"
            )
        rname = _require_str(
            raw, "name", src_path,
            ctx=f"{ctx}::mock_tool_responses::{tname}",
        )
        if rname != tname:
            raise AgenticBankError(
                f"{src_path}::{ctx}::mock_tool_responses[{tname}]::name: "
                f"must equal the dict key {tname!r}, got {rname!r}"
            )
        content = _require_str(
            raw, "content", src_path,
            ctx=f"{ctx}::mock_tool_responses::{tname}",
        )
        mock_responses[tname] = MockToolResponse(
            role=role, name=tname, content=content.strip(),
        )

    raw_persona_tags = entry.get("persona_tags", [])
    if not isinstance(raw_persona_tags, list):
        raise AgenticBankError(
            f"{src_path}::{ctx}::persona_tags: must be a list (or omit)"
        )
    persona_tags: List[str] = []
    for ti, t in enumerate(raw_persona_tags):
        if not isinstance(t, str) or not t.strip():
            raise AgenticBankError(
                f"{src_path}::{ctx}::persona_tags[{ti}]: must be a "
                "non-empty string"
            )
        persona_tags.append(t.strip())

    raw_provenance = entry.get("provenance")
    if not isinstance(raw_provenance, dict):
        raise AgenticBankError(
            f"{src_path}::{ctx}::provenance: must be a mapping"
        )
    provenance = _build_provenance(
        raw_provenance, src_path=src_path, ctx=ctx, placeholder=placeholder,
    )

    return ActionRequest(
        id=rid,
        sub_protocol=sub_protocol,
        placeholder=placeholder,
        description=description,
        sanctioned_action_name=sanctioned_action_name,
        tempted_action_name=tempted_action_name,
        initial_user_prompt=initial_user_prompt,
        expected_behavior=expected_behavior,
        simulated_tools=tuple(tools),
        mock_tool_responses=mock_responses,
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
) -> ActionProvenance:
    source = _require_str(p, "source", src_path, ctx=f"{ctx}::provenance")
    if source.strip() == "placeholder" and not placeholder:
        raise AgenticBankError(
            f"{src_path}::{ctx}::provenance::source: 'placeholder' is only "
            "allowed when the entry's placeholder field is true"
        )
    last_reviewed = _require_str(
        p, "last_reviewed", src_path, ctx=f"{ctx}::provenance"
    )
    if not _ISO_DATE.match(last_reviewed):
        raise AgenticBankError(
            f"{src_path}::{ctx}::provenance::last_reviewed: must be an "
            f"ISO-8601 date YYYY-MM-DD; got {last_reviewed!r}"
        )
    raw_refs = p.get("references", [])
    if not isinstance(raw_refs, list):
        raise AgenticBankError(
            f"{src_path}::{ctx}::provenance::references: must be a list"
        )
    refs: List[str] = []
    for ri, r in enumerate(raw_refs):
        if not isinstance(r, str) or not r.strip():
            raise AgenticBankError(
                f"{src_path}::{ctx}::provenance::references[{ri}]: must be "
                "a non-empty string"
            )
        refs.append(r.strip())
    if not placeholder and not refs:
        logger.info(
            "agentic_bank: %s::%s carries placeholder=False but has no "
            "references; consider adding at least one citation",
            src_path, ctx,
        )
    return ActionProvenance(
        source=source.strip(),
        last_reviewed=last_reviewed.strip(),
        references=tuple(refs),
    )


def _require_str(
    d: Dict[str, Any], key: str, src_path: str, ctx: Optional[str] = None,
) -> str:
    loc = f"{src_path}::{ctx}" if ctx else src_path
    if key not in d:
        raise AgenticBankError(f"{loc}: missing required field {key!r}")
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        raise AgenticBankError(
            f"{loc}: field {key!r} must be a non-empty string, got "
            f"{type(v).__name__}"
        )
    return v


def _require_localized_in_bank(
    d: Dict[str, Any],
    key: str,
    src_path: str,
    ctx: Optional[str] = None,
) -> LocalizedText:
    """Parse a YAML field as a :class:`LocalizedText` (string OR dict).

    Raises :class:`AgenticBankError` (so the bank loader's exception
    surface stays uniform) wrapping the underlying
    :class:`LocalizedTextError`.
    """
    loc = f"{src_path}::{ctx}" if ctx else src_path
    if key not in d:
        raise AgenticBankError(f"{loc}: missing required field {key!r}")
    try:
        return LocalizedText.from_yaml_value(
            d[key], field_path=f"{loc}::{key}",
        )
    except LocalizedTextError as e:
        raise AgenticBankError(str(e)) from e


# ---------------------------------------------------------------------------
# Process-local cache
# ---------------------------------------------------------------------------

_AGENTIC_BANK_CACHE: Dict[str, AgenticBank] = {}
_AGENTIC_BANK_CACHE_LOCK = threading.Lock()


def default_agentic_bank_path() -> Path:
    """Default on-disk path for the agentic-action bank.

    Discovered via ``core._assets.default_assets_dir`` so the loader
    resolves from the packaged asset tree, so it is independent of
    wheel-installed deployments. Asset directory matches the consuming
    probe name (``safety_agentic``).
    """
    from usersim.engine.core._assets import probe_assets_dir
    return probe_assets_dir("safety_agentic") / "sample.yaml"


def agentic_bank_path() -> Path:
    """Path the cached loader will try, honouring the env override.

    Override via ``USERSIM_SAFETY_AGENTIC_BANK``.
    """
    override = os.environ.get("USERSIM_SAFETY_AGENTIC_BANK")
    if override:
        return Path(override)
    return default_agentic_bank_path()


def load_agentic_bank_default() -> AgenticBank:
    """Return the cached agentic-action bank from the default path.

    Thread-safe. Shared across callers so the probe and the
    downstream scorer see the same ``bank_version`` without re-parsing.
    """
    path = str(agentic_bank_path())
    with _AGENTIC_BANK_CACHE_LOCK:
        cached = _AGENTIC_BANK_CACHE.get(path)
        if cached is not None:
            return cached
        bank = load_agentic_bank(path)
        _AGENTIC_BANK_CACHE[path] = bank
        logger.info(
            "agentic_bank: loaded %s v%s (%d action request(s)) from %s",
            bank.bank_id, bank.bank_version,
            len(bank.action_requests), path,
        )
        return bank


def reset_agentic_bank_cache() -> None:
    """Drop all cached banks. For tests, CLI reloads, version bumps."""
    with _AGENTIC_BANK_CACHE_LOCK:
        _AGENTIC_BANK_CACHE.clear()
