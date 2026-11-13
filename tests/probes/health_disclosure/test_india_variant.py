# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The health probes in India variant locales: the en_IN scaffold, the romanized directive, and unchanged shipped locales."""

from __future__ import annotations

import pytest
from _india_variant import _builder

import usersim.engine.generator  # noqa: F401  (registers the built-in probes)
from usersim.engine.core.outcomes import Provenance

_HEALTH_PROBE_LABELS = (
    "health_therapy_disclosure",
    "health_triage_disclosure",
    "health_decision_support_disclosure",
)


def _health_probe(label: str, persona, cfg, locale: str, language: str):
    import usersim.engine.probes.health_disclosure  # noqa: F401 — registers
    from usersim.engine.core.behavioral import compute_behavioral_profile
    from usersim.engine.core.probes import resolve_probe

    return resolve_probe(label)(
        persona=persona,
        locale=locale,
        language=language,
        models={"summary_model": object()},
        cfg=cfg,
        provenance=Provenance(),
        profile=compute_behavioral_profile(persona, locale),
        data={},
        outcome_builder=_builder(),
    )


@pytest.mark.parametrize("label", _HEALTH_PROBE_LABELS)
def test_health_disclosure_variant_resolves_en_in_scaffold(
    label,
    indian_persona,
    cfg,
) -> None:
    probe = _health_probe(label, indian_persona, cfg, "ta_Taml_IN", "Tamil")

    assert probe._asset_locale == "en_IN"  # scaffold pack fell back to the base
    assert probe._locale == "ta_Taml_IN"  # conversation locale preserved
    prompt = probe.get_user_system_prompt()
    assert prompt  # en_IN pack resolved (no abort)
    # The scaffold is English, so the target-language directive must carry the
    # conversation language — otherwise the user agent writes English and the
    # script gate exhausts every row.
    assert "You MUST write your messages in Tamil" in prompt
    assert "native Tamil script" in prompt
    assert probe.format_gate_prompt("hi", "history")  # gate pack resolved too


@pytest.mark.parametrize("label", _HEALTH_PROBE_LABELS)
def test_health_disclosure_romanized_variant_gets_romanized_directive(
    label,
    indian_persona,
    cfg,
) -> None:
    probe = _health_probe(label, indian_persona, cfg, "ta_Latn_IN", "Tamil")
    assert "romanized Tamil" in probe.get_user_system_prompt()


@pytest.mark.parametrize("label", _HEALTH_PROBE_LABELS)
def test_health_disclosure_shipped_locale_unchanged(
    label,
    indian_persona,
    cfg,
) -> None:
    # A shipped locale resolves its own pack, with no fallback, and its directive names its own language.
    probe = _health_probe(label, indian_persona, cfg, "en_US", "English")
    assert probe._asset_locale == "en_US"
    prompt = probe.get_user_system_prompt()
    assert "You MUST write your messages in English." in prompt
    assert "romanized" not in prompt
