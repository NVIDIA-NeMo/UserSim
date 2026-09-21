# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Regression guard: no template-text or cross-locale-duplicate content in
any shipped asset bank.

A previous bulk-bank-update introduced ~3,500 template-fill placeholder
strings ("I need a short fictional example that would cross into safety
bypass") and ~300 cross-locale-duplicate LocalizedText fields (English
text replicated identically into all 8 locale slots). All affected
entries were rewritten across all 8 locales of `sov_ai_facts` plus the
safety / dynamic / multilingual-parity banks.

The rewrite is complete. This test now runs in **strict mode by default**:
any future regression that re-introduces template-text or cross-locale-
duplicate content fails CI immediately.

The earlier soft-failing mode (warnings instead of failures) is preserved
behind ``USERSIM_RELAX_TEMPLATE_GUARD=1`` so that someone in the middle
of authoring a bank locally can run the rest of the suite without this
guard blocking; it is **not** the CI default.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import _asset_audit
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_relaxed() -> bool:
    """Return ``True`` when the strict guard is downgraded to warnings.

    Used during local authoring of a single bank when the rest of the
    suite should remain runnable; never the CI default.
    """
    return os.environ.get("USERSIM_RELAX_TEMPLATE_GUARD") == "1"


def _collect_defects() -> tuple[int, int, list[str]]:
    """Run the audit and return ``(template_hits, locale_dups, lines)``."""
    template_hits = 0
    locale_dups = 0
    lines: list[str] = []
    for relpath in _asset_audit.BANK_PATHS:
        path = _REPO_ROOT / relpath
        if not path.exists():
            continue
        f = _asset_audit.audit_one_bank(path)
        n_t = len(f["template_hits"])
        n_d = len(f["duplicate_locale_dicts"])
        if n_t == 0 and n_d == 0:
            continue
        template_hits += n_t
        locale_dups += n_d
        lines.append(f"  {f['path']}: template={n_t} locale-dups={n_d}")
    return template_hits, locale_dups, lines


def test_no_template_text_in_shipped_banks():
    """The asset banks must not contain template-fill placeholder text or
    cross-locale-duplicate LocalizedText fields.

    Strict by default; downgrade to warnings during local authoring with
    ``USERSIM_RELAX_TEMPLATE_GUARD=1``.
    """
    template_hits, locale_dups, lines = _collect_defects()
    total = template_hits + locale_dups

    if total == 0:
        return  # Clean.

    msg_lines = [
        f"asset-bank template-text guard: {template_hits} template-text "
        f"hits + {locale_dups} cross-locale-duplicate LocalizedText fields",
        "Per-bank breakdown:",
        *lines,
        "Inspect tests/_asset_audit.py for the patterns; "
        "re-run with USERSIM_RELAX_TEMPLATE_GUARD=1 to downgrade to warnings.",
    ]
    msg = "\n".join(msg_lines)

    if _is_relaxed():
        warnings.warn(msg, stacklevel=2)
    else:
        pytest.fail(msg)


def test_audit_helper_runs():
    """Sanity: the per-bank audit function returns a structured finding
    dict for every shipped bank.
    """
    for relpath in _asset_audit.BANK_PATHS:
        path = _REPO_ROOT / relpath
        if not path.exists():
            continue
        f = _asset_audit.audit_one_bank(path)
        assert "path" in f
        assert "template_hits" in f
        assert "duplicate_locale_dicts" in f
