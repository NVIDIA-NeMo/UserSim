# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset-bank audit primitives for ``test_no_template_text.py``.

The audit scans every bank ``shipped_banks()`` lists for template-fill
placeholder strings and for ``LocalizedText`` fields that carry one identical
string across three or more locales. ``test_no_template_text.py`` runs it on
every push (strict by default) and fails CI if any of the patterns below
appear.

Public surface:

- ``shipped_banks()``: every YAML file in the asset folder of every
  registered probe, by name.
- ``audit_one_bank(path)``: return a per-bank ``{template_hits,
  duplicate_locale_dicts}`` finding dict.
- ``YAML_LOADER``: the YAML loader the audit uses, libyaml's when it is
  available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from usersim.engine.core._assets import probe_assets_dir
from usersim.engine.core.probes import known_probes

YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# All Nemotron-Personas locale ids the simulator currently supports.
SUPPORTED_LOCALES: tuple[str, ...] = (
    "en_US",
    "en_IN",
    "en_SG",
    "en_GB",
    "pt_BR",
    "fr_FR",
    "de_DE",
    "ja_JP",
    "hi_Deva_IN",
    "hi_Latn_IN",
    "ko_KR",
)

# Regex patterns that indicate template-fill / placeholder text masquerading
# as bank content. Each entry is ``(name, pattern)``; the pattern is matched
# case-insensitively against every prose string in every shipped bank.
TEMPLATE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("variant_with_refusal_should_hold", r"variant \d+;? .*refusal should hold"),
    ("would_cross_into", r"would cross into"),
    ("usable_not_just_theory", r"usable,?\s+not just theory"),
    ("low_real_world_harm", r"low-real-world-harm"),
    (
        "fictional_example_safety",
        r"fictional example .* (safety bypass|manipulation|deception|fraud|circumvention|privacy|impersonation|harassment)",
    ),
    ("wrong_national_agency", r"wrong national agency"),
    ("half_remember_proverb", r"proverb.*half[- ]remember|half[- ]remember.*proverb"),
    ("main_X_rule_handled", r"main \w+ rule (is )?handled"),
    ("beta_scenario", r"\bbeta\s+\w+(?:_\w+)*\s+scenario\b"),
    ("decline_to_provide_content_that_enables", r"decline to provide content that enables"),
    ("placeholder_request_filler", r"placeholder request"),
    ("sample_target_filler", r"sample target"),
    ("placeholder_action_filler", r"placeholder action"),
    (
        "english_noun_in_native_scaffold",
        r"poser une question sur (geography|history|culture|public services|health systems|education|labor|finance|digital)",
    ),
)

_TEMPLATE_RES = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in TEMPLATE_PATTERNS]


def shipped_banks() -> dict[str, Path]:
    """Every YAML file in the asset folder of every registered probe.

    Keyed by ``<label>/<path inside the probe's folder>``, so a bank is named
    the same wherever its probe keeps it. Raises ``FileNotFoundError`` when a
    probe's folder does not resolve, so banks that moved out of reach fail
    the audit instead of dropping out of it.
    """
    banks: dict[str, Path] = {}
    for label in known_probes():
        folder = probe_assets_dir(label)
        if not folder.is_dir():
            raise FileNotFoundError(f"{label}: asset folder {folder} does not exist")
        for path in sorted(folder.rglob("*.y*ml")):
            banks[f"{label}/{path.relative_to(folder).as_posix()}"] = path
    return banks


def _is_locale_dict(node: Any) -> bool:
    """``True`` when ``node`` is a LocalizedText-shaped dict (locale-keyed)."""
    return isinstance(node, dict) and len(node) >= 2 and any(loc in node for loc in SUPPORTED_LOCALES)


def _walk_strings(node: Any, parent_id: Any = None, parent_field: str | None = None, locale: str | None = None):
    """Yield ``(entry_id, field_name, string_value, locale)`` for every string in a YAML doc.

    Strings inside lists count. A string under a LocalizedText dict keeps the
    field that holds the dict and carries its locale.
    """
    if isinstance(node, str):
        yield (parent_id, parent_field, node, locale)
    elif isinstance(node, dict):
        entry_id = node.get("id", parent_id)
        for k, v in node.items():
            if _is_locale_dict(v):
                for loc, s in v.items():
                    yield from _walk_strings(s, entry_id, k, loc)
            else:
                yield from _walk_strings(v, entry_id, k, locale)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_strings(item, parent_id, parent_field, locale)


def _detect_template_in_string(s: str) -> list[str]:
    """Return list of pattern names that match ``s``."""
    return [name for name, rx in _TEMPLATE_RES if rx.search(s)]


def audit_one_bank(path: Path) -> dict[str, Any]:
    """Audit a single bank YAML file for template-text and cross-locale-duplicate
    LocalizedText fields. Returns a finding dict with keys ``template_hits``
    and ``duplicate_locale_dicts``.
    """
    with path.open(encoding="utf-8") as fh:
        data = yaml.load(fh, Loader=YAML_LOADER)

    findings: dict[str, Any] = {
        "template_hits": [],  # list of (entry_id, field, locale, pattern_name)
        "duplicate_locale_dicts": [],  # list of (entry_id, field, en_value_truncated)
    }

    seen = set()
    for entry_id, field, s, loc in _walk_strings(data):
        if field in {
            "id",
            "schema_version",
            "bank_id",
            "bank_version",
            "taxonomy_id",
            "taxonomy_version",
            "category",
            "harm_category",
            "sub_protocol",
            "question_type",
            "difficulty",
            "last_reviewed",
            "source",
            "name",
            "type",
        }:
            continue
        for pat_name in _detect_template_in_string(s):
            key = (entry_id, field, loc, pat_name)
            if key in seen:
                continue
            seen.add(key)
            findings["template_hits"].append(
                {"entry_id": entry_id, "field": field, "locale": loc, "pattern": pat_name, "snippet": s[:120]}
            )

    def _walk_for_locale_dicts(node, parent_id=None):
        if isinstance(node, dict):
            entry_id = node.get("id", parent_id) if "id" in node else parent_id
            for k, v in node.items():
                if _is_locale_dict(v):
                    string_vals = {loc: s for loc, s in v.items() if isinstance(s, str)}
                    if len(string_vals) >= 2:
                        unique = set(string_vals.values())
                        if len(unique) == 1 and len(string_vals) > 2:
                            findings["duplicate_locale_dicts"].append(
                                {
                                    "entry_id": entry_id,
                                    "field": k,
                                    "n_locales": len(string_vals),
                                    "snippet": next(iter(string_vals.values()))[:120],
                                }
                            )
                else:
                    _walk_for_locale_dicts(v, entry_id)
        elif isinstance(node, list):
            for item in node:
                _walk_for_locale_dicts(item, parent_id)

    _walk_for_locale_dicts(data)
    return findings
