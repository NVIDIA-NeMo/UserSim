# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clinical-profile bank — versioned ground truth for the health_disclosure family.

Models the hidden clinical profile (gated topics + concealment / risk ground
truth, keyed by persona uuid) as a first-class, versioned asset bank, exactly
like ``fact_bank`` / ``pressure_bank`` / ``agentic_bank``. This buys the family
everything the ad-hoc JSON loader it replaces did not have:

- ``provenance.bank_version`` pinning (so a trajectory records which profile set
  it ran against — no replay/drift hole),
- ``placeholder``-entry discipline (the shipped bundled banks are SYNTHETIC
  placeholders; using one raises ``WarningKind.USED_PLACEHOLDER_CLINICAL_PROFILE``
  so a scorecard never claims production readiness on placeholder ground truth),
- asset-path discovery + audit-test coverage.

Unlike the locale-keyed banks (fact_bank), clinical profiles are **per-client**
(one bank per registered probe: ``health_therapy_disclosure`` / ``health_triage_disclosure`` /
``health_decision_support_disclosure`` / ``health_general_disclosure``) and
**locale-independent**. A customer ships their own real
profiles by pointing the client's
``profiles_env`` env var at a bank file of this same shape; absent that, the
bundled synthetic placeholder bank is used (and flagged).
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger("usersim.engine")

SUPPORTED_SCHEMA_VERSIONS: FrozenSet[str] = frozenset({"v0.1"})

#: Profile id the derived task falls back to when a persona uuid isn't in the
#: bank (the usual case for a bundled synthetic bank). Ships in every bundled
#: bank as a single placeholder entry.
DEFAULT_PROFILE_ID = "__default__"

#: The registered client probe labels that own a clinical profile bank.
CLIENT_PROBE_LABELS: Tuple[str, ...] = (
    "health_therapy_disclosure",
    "health_triage_disclosure",
    "health_decision_support_disclosure",
    "health_general_disclosure",
)


class ClinicalProfileBankError(ValueError):
    """Raised on a malformed clinical profile bank ("<path>::<ctx>: <what>")."""


@dataclass(frozen=True)
class ClinicalProfile:
    """One persona's hidden clinical profile (bank fields stamped by the loader)."""

    id: str  # persona uuid (join key), or DEFAULT_PROFILE_ID for the fallback
    placeholder: bool
    profile: Dict[str, Any]  # concealment / gated-topic / risk ground truth
    client: str = ""
    bank_id: str = ""
    bank_version: str = ""


@dataclass(frozen=True)
class ClinicalProfileBank:
    schema_version: str
    client: str
    bank_id: str
    bank_version: str
    profiles: Tuple[ClinicalProfile, ...] = field(default_factory=tuple)
    source_path: Optional[str] = None

    def by_id(self, profile_id: str) -> Optional[ClinicalProfile]:
        for p in self.profiles:
            if p.id == profile_id:
                return p
        return None


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------
def _require_str(
    d: Dict[str, Any], key: str, src_path: str, ctx: Optional[str] = None
) -> str:
    loc = f"{src_path}::{ctx}" if ctx else src_path
    if key not in d:
        raise ClinicalProfileBankError(f"{loc}: missing required field {key!r}")
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        raise ClinicalProfileBankError(
            f"{loc}: field {key!r} must be a non-empty string, got "
            f"{type(v).__name__}"
        )
    return v


def _build_clinical_profile_bank(doc: Dict[str, Any], *, src_path: str) -> ClinicalProfileBank:
    schema_version = _require_str(doc, "schema_version", src_path)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ClinicalProfileBankError(
            f"{src_path}: unsupported schema_version {schema_version!r} "
            f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )
    client = _require_str(doc, "client", src_path)
    bank_id = _require_str(doc, "bank_id", src_path)
    bank_version = _require_str(doc, "bank_version", src_path)

    raw_entries = doc.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ClinicalProfileBankError(
            f"{src_path}::entries: must be a non-empty list"
        )

    profiles: list[ClinicalProfile] = []
    seen: set[str] = set()
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ClinicalProfileBankError(
                f"{src_path}::entries[{idx}]: each entry must be a mapping"
            )
        profile_id = _require_str(entry, "id", src_path, ctx=f"entries[{idx}]")
        placeholder = entry.get("placeholder", False)
        if not isinstance(placeholder, bool):
            raise ClinicalProfileBankError(
                f"{src_path}::{profile_id}: placeholder must be a bool, got "
                f"{type(placeholder).__name__}"
            )
        payload = entry.get("profile")
        if not isinstance(payload, dict):
            raise ClinicalProfileBankError(
                f"{src_path}::{profile_id}: profile must be a mapping, got "
                f"{type(payload).__name__}"
            )
        if profile_id in seen:
            raise ClinicalProfileBankError(
                f"{src_path}::{profile_id}: duplicate profile id"
            )
        seen.add(profile_id)
        profiles.append(
            ClinicalProfile(
                id=profile_id, placeholder=placeholder, profile=payload,
                client=client, bank_id=bank_id, bank_version=bank_version,
            )
        )

    n_placeholder = sum(1 for p in profiles if p.placeholder)
    if n_placeholder:
        logger.info(
            "clinical_profile_bank: loaded %s (v%s) with %d/%d placeholder "
            "entries — downstream scorecards will be marked preview-only",
            bank_id, bank_version, n_placeholder, len(profiles),
        )
    return ClinicalProfileBank(
        schema_version=schema_version, client=client, bank_id=bank_id,
        bank_version=bank_version, profiles=tuple(profiles), source_path=src_path,
    )


def load_clinical_profile_bank(path: str | Path) -> ClinicalProfileBank:
    """Parse a clinical profile bank from a YAML/JSON file (JSON is valid YAML)."""
    src_path = Path(path)
    if not src_path.exists():
        raise ClinicalProfileBankError(f"{src_path}: file not found")
    import yaml  # lazy (matches fact_bank)

    try:
        doc = yaml.safe_load(src_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:  # pragma: no cover - defensive
        raise ClinicalProfileBankError(f"{src_path}: parse failure: {e}") from e
    if not isinstance(doc, dict):
        raise ClinicalProfileBankError(f"{src_path}: top-level must be a mapping")
    return _build_clinical_profile_bank(doc, src_path=str(src_path))


# ---------------------------------------------------------------------------
# Default path resolution + cache
# ---------------------------------------------------------------------------
def default_clinical_profile_bank_path(client: str) -> Path:
    """Bundled bank path: ``assets/<client>/sample.yaml`` (locale-independent)."""
    from usersim.engine.core._assets import probe_assets_dir

    if client not in CLIENT_PROBE_LABELS:
        raise ClinicalProfileBankError(f"unknown client probe label: {client!r}")
    return probe_assets_dir(client) / "sample.yaml"


def clinical_profile_bank_path_for(
    client: str, env_override: Optional[str] = None,
) -> Path:
    """Resolve the bank path for a client: customer ``env_override`` var wins,
    else the bundled synthetic default."""
    if env_override:
        override_path = os.environ.get(env_override, "").strip()
        if override_path:
            return Path(override_path)
    return default_clinical_profile_bank_path(client)


# Keyed by the RESOLVED bank path, not the client label: the customer
# ``env_override`` can repoint a client at a different file within one process
# (tests, config reloads), and a label-keyed cache would then serve the stale
# bank. The path is what actually determines the bank contents.
_BANK_CACHE: Dict[str, ClinicalProfileBank] = {}
_BANK_CACHE_LOCK = threading.Lock()


def load_clinical_profile_bank_for_client(
    client: str, env_override: Optional[str] = None,
) -> ClinicalProfileBank:
    """Load (and cache) the clinical profile bank for a client probe label.

    Cached by resolved path so that repointing ``env_override`` at a new file
    loads the new bank rather than returning the previously cached one.
    """
    path = clinical_profile_bank_path_for(client, env_override)
    cache_key = str(Path(path).resolve()) if Path(path).exists() else str(path)
    with _BANK_CACHE_LOCK:
        cached = _BANK_CACHE.get(cache_key)
        if cached is not None:
            return cached
        bank = load_clinical_profile_bank(path)
        _BANK_CACHE[cache_key] = bank
        logger.info(
            "clinical_profile_bank: loaded %s v%s (%d profiles) for client=%s "
            "from %s", bank.bank_id, bank.bank_version, len(bank.profiles),
            client, path,
        )
        return bank


def reset_clinical_profile_bank_cache() -> None:
    """Clear the process cache (tests + config reloads)."""
    with _BANK_CACHE_LOCK:
        _BANK_CACHE.clear()
