# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for each probe's India-variant wiring tests; their fixtures are in ``tests/probes/conftest.py``.

Each verbatim probe's ``tests/probes/<package>/test_india_variant.py`` exercises
its real first-turn call site for a variant locale (``ta_Taml_IN``) against the
shipped en_IN or shared banks:

- asset resolution falls back to en_IN (``self._asset_locale``),
- the verbatim first turn is machine-translated into the conversation language,
- side-channel metadata is still seeded, so ``should_succeed`` and scorers work,
- a ``USED_MACHINE_TRANSLATION`` warning is recorded.

``tests/engine/core/test_india_variants.py`` covers the substrate on its own.
Translation is mocked at ``core.translation.acall_llm``, its own binding, so no
network is reached.
"""

from __future__ import annotations

from unittest.mock import patch

from usersim.engine.core import translation as T
from usersim.engine.core.outcomes import OutcomeBuilder, Provenance, WarningKind


def _as_async(fn):
    """Wrap a canned reply so the awaiting call site can await it."""

    async def _call(*args, **kwargs):
        return fn(*args, **kwargs)

    return _call


_MARKER = "‹TRANSLATED-turn1›"


def _mock_translation():
    """Patch the translation module's own awaited model call."""
    return patch.object(T, "acall_llm", _as_async(lambda *a, **k: {"content": _MARKER}))


def _builder() -> OutcomeBuilder:
    return OutcomeBuilder(provenance=Provenance())


def _mt_warned(ob: OutcomeBuilder) -> bool:
    return WarningKind.USED_MACHINE_TRANSLATION in [w.kind for w in ob._outcome.warnings]
