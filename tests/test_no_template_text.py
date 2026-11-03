# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No shipped asset bank holds template-fill text or one string copied across locales.

The audit reads every YAML file in the asset folder of every registered
probe, wherever the probe keeps it, and checks each string against the
patterns in ``_asset_audit.TEMPLATE_PATTERNS``. It also flags a
``LocalizedText`` field that carries one identical string in three or more
locales.

The guard fails on any finding. Set ``USERSIM_RELAX_TEMPLATE_GUARD=1`` to
report findings as warnings instead while authoring a bank locally.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import _asset_audit
import pytest


def _is_relaxed() -> bool:
    """Return ``True`` when the guard reports findings as warnings instead of failing."""
    return os.environ.get("USERSIM_RELAX_TEMPLATE_GUARD") == "1"


def _finding_lines(name: str, findings: dict[str, Any]) -> list[str]:
    """One line per finding in the bank ``name``."""
    lines = []
    for hit in findings["template_hits"]:
        where = f"{hit['entry_id']}.{hit['field']}" + (f"[{hit['locale']}]" if hit["locale"] else "")
        lines.append(f"  {name}: {where} matches {hit['pattern']}: {hit['snippet']!r}")
    for dup in findings["duplicate_locale_dicts"]:
        lines.append(
            f"  {name}: {dup['entry_id']}.{dup['field']} has one string in {dup['n_locales']} locales: "
            f"{dup['snippet']!r}"
        )
    return lines


def test_no_template_text_in_shipped_banks():
    """No shipped bank holds template-fill text or a LocalizedText field copied across locales.

    Strict by default; downgrade to warnings during local authoring with
    ``USERSIM_RELAX_TEMPLATE_GUARD=1``.
    """
    banks = _asset_audit.shipped_banks()
    assert banks, "no shipped bank found to audit"
    lines = [line for name, path in banks.items() for line in _finding_lines(name, _asset_audit.audit_one_bank(path))]

    if not lines:
        return

    msg = "\n".join(
        [
            f"asset-bank template-text guard: {len(lines)} finding(s) in {len(banks)} banks",
            *lines,
            "Inspect tests/_asset_audit.py for the patterns; "
            "re-run with USERSIM_RELAX_TEMPLATE_GUARD=1 to downgrade to warnings.",
        ]
    )

    if _is_relaxed():
        warnings.warn(msg, stacklevel=2)
    else:
        pytest.fail(msg)


def test_half_remember_pattern_needs_a_proverb():
    """``half_remember_proverb`` flags a half-remembered proverb, not every half-remembered line."""
    assert _asset_audit._detect_template_in_string("Ask about a proverb you half remember.") == [
        "half_remember_proverb"
    ]
    assert _asset_audit._detect_template_in_string("Quote a half-remembered proverb.") == ["half_remember_proverb"]
    assert _asset_audit._detect_template_in_string("Bring up lyrics you only half remember.") == []


def test_audit_finds_template_text_and_copied_locale_text(tmp_path: Path):
    """``audit_one_bank`` reports both kinds of finding in a bank anywhere on disk."""
    bank = tmp_path / "sample.yaml"
    bank.write_text(
        "items:\n"
        "- id: filler\n"
        "  prompt: Write variant 3; the refusal should hold.\n"
        "- id: copied\n"
        "  question: {en_US: Where is the station, de_DE: Where is the station, fr_FR: Where is the station}\n",
        encoding="utf-8",
    )

    findings = _asset_audit.audit_one_bank(bank)

    assert [(h["entry_id"], h["field"], h["locale"], h["pattern"]) for h in findings["template_hits"]] == [
        ("filler", "prompt", None, "variant_with_refusal_should_hold")
    ]
    assert [(d["entry_id"], d["field"], d["n_locales"]) for d in findings["duplicate_locale_dicts"]] == [
        ("copied", "question", 3)
    ]
