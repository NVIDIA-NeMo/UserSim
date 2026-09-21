# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Asset-bank audit primitives for the regression guard in
``test_no_template_text.py``.

Built during the asset-rewrite project that replaced ~3,500 template-fill
placeholder strings + ~300 cross-locale-duplicate ``LocalizedText`` fields
with AI-authored beta content across all 19 shipped banks (8 locales of
``sov_ai_facts``, 8 locales of ``sov_ai_dynamic``, the safety /
multilingual banks).

The rewrite itself is complete; this module is the **defence against
regression**: ``test_no_template_text.py`` runs the audit on every push
(strict by default) and fails CI if any of the patterns below reappear.

Public surface:

- ``BANK_PATHS`` — tuple of repo-relative YAML paths to every shipped bank.
- ``audit_one_bank(path)`` — return a per-bank ``{template_hits,
  duplicate_locale_dicts}`` finding dict.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# All Nemotron-Personas locale ids the simulator currently supports.
SUPPORTED_LOCALES: tuple[str, ...] = (
    "en_US",
    "en_IN",
    "en_SG",
    "pt_BR",
    "fr_FR",
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
    ("half_remember_proverb", r"half remember"),
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


BANK_PATHS: tuple[str, ...] = (
    "assets/sov_ai_facts/en_IN/sample.yaml",
    "assets/sov_ai_facts/en_SG/sample.yaml",
    "assets/sov_ai_facts/en_US/sample.yaml",
    "assets/sov_ai_facts/fr_FR/sample.yaml",
    "assets/sov_ai_facts/hi_Deva_IN/sample.yaml",
    "assets/sov_ai_facts/hi_Latn_IN/sample.yaml",
    "assets/sov_ai_facts/ja_JP/sample.yaml",
    "assets/sov_ai_facts/ko_KR/sample.yaml",
    "assets/sov_ai_facts/pt_BR/sample.yaml",
    "assets/safety_chat_pressure/sample.yaml",
    "assets/safety_agentic/sample.yaml",
    "assets/safety_agentic/action_taxonomy.yaml",
    "assets/sov_ai_multilingual_parity/sample.yaml",
    "assets/sov_ai_dynamic/en_IN/categories.yaml",
    "assets/sov_ai_dynamic/en_SG/categories.yaml",
    "assets/sov_ai_dynamic/en_US/categories.yaml",
    "assets/sov_ai_dynamic/fr_FR/categories.yaml",
    "assets/sov_ai_dynamic/hi_Deva_IN/categories.yaml",
    "assets/sov_ai_dynamic/hi_Latn_IN/categories.yaml",
    "assets/sov_ai_dynamic/ja_JP/categories.yaml",
    "assets/sov_ai_dynamic/ko_KR/categories.yaml",
    "assets/sov_ai_dynamic/pt_BR/categories.yaml",
)


def _is_locale_dict(node: Any) -> bool:
    """``True`` when ``node`` is a LocalizedText-shaped dict (locale-keyed)."""
    return isinstance(node, dict) and len(node) >= 2 and any(loc in node for loc in SUPPORTED_LOCALES)


def _walk_strings(node: Any, parent_id: str | None = None, parent_field: str | None = None):
    """Yield ``(entry_id, field_name, string_value, is_locale_dict_member, locale)``
    for every string in a YAML doc."""
    if isinstance(node, dict):
        entry_id = node.get("id", parent_id) if "id" in node else parent_id
        for k, v in node.items():
            if isinstance(v, str):
                yield (entry_id, k, v, False, None)
            elif _is_locale_dict(v):
                for loc, s in v.items():
                    if isinstance(s, str):
                        yield (entry_id, k, s, True, loc)
            else:
                yield from _walk_strings(v, entry_id, k)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_strings(item, parent_id, parent_field)


def _detect_template_in_string(s: str) -> list[str]:
    """Return list of pattern names that match ``s``."""
    return [name for name, rx in _TEMPLATE_RES if rx.search(s)]


def audit_one_bank(path: Path) -> dict[str, Any]:
    """Audit a single bank YAML file for template-text and cross-locale-duplicate
    LocalizedText fields. Returns a finding dict with keys ``path``,
    ``template_hits``, and ``duplicate_locale_dicts``.
    """
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    findings: dict[str, Any] = {
        "path": str(path.relative_to(REPO_ROOT)),
        "template_hits": [],  # list of (entry_id, field, locale, pattern_name)
        "duplicate_locale_dicts": [],  # list of (entry_id, field, en_value_truncated)
    }

    seen = set()
    for entry_id, field, s, _is_loc_member, loc in _walk_strings(data):
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
