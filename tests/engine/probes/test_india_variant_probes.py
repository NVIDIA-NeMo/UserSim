# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-probe wiring tests for the India language-variant stop-gap.

Complements ``tests/core/test_india_variants.py`` (which covers the substrate
``_load_bank_for_locale`` fallback + ``_localize_verbatim`` helper in
isolation) by exercising each verbatim probe's ACTUAL ``get_verbatim_first_
user_turn`` call site for a variant locale (``ta_Taml_IN``) against the real
shipped en_IN / shared banks:

- asset resolution falls back to en_IN (``self._asset_locale``),
- the verbatim turn-1 is machine-translated into the conversation language,
- side-channel metadata is still seeded (so ``should_succeed`` + scorers work),
- a ``USED_MACHINE_TRANSLATION`` warning is recorded (preview-only discipline).

Translation is mocked at ``core.translation.call_llm`` (its own binding — the
loop's ``_patched_call_llm`` helper does not cover it) so no network is hit.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

import usersim.engine.generator  # noqa: F401 — triggers probe registration
from usersim.engine.core import translation as T
from usersim.engine.core.outcomes import (
    OutcomeBuilder,
    Provenance,
    WarningKind,
)
from usersim.engine.core.simulation import ConversationState


_MARKER = "‹TRANSLATED-turn1›"


@pytest.fixture
def indian_persona() -> Dict[str, Any]:
    return {
        "first_name": "Arjun",
        "last_name": "Kumar",
        "age": 34,
        "sex": "male",
        "education_level": "Graduate & above",
        "occupation": "Schoolteacher",
        "country": "India",
        "region": "Tamil Nadu",
        "district": "Chennai",
        "openness": 0.6,
        "conscientiousness": 0.6,
        "extraversion": 0.5,
        "agreeableness": 0.7,
        "neuroticism": 0.4,
        "hobbies_and_interests": "cricket, cooking, local history",
    }


@pytest.fixture
def cfg() -> Any:
    return type("Cfg", (), {"random_seed": 7, "max_turns": 5})()


def _mock_translation():
    """Patch the translation module's own ``call_llm`` binding."""
    return patch.object(T, "call_llm", lambda *a, **k: {"content": _MARKER})


def _builder() -> OutcomeBuilder:
    return OutcomeBuilder(provenance=Provenance())


def _mt_warned(ob: OutcomeBuilder) -> bool:
    return WarningKind.USED_MACHINE_TRANSLATION in [w.kind for w in ob._outcome.warnings]


# ---------------------------------------------------------------------------
# sov_ai_facts — verbatim fact question translated
# ---------------------------------------------------------------------------


def test_sov_ai_facts_variant_translates_turn1(indian_persona, cfg) -> None:
    from usersim.engine.probes.sov_ai_facts.generator import SovAiFactsProbe

    T.reset_translation_cache()
    ob = _builder()
    with _mock_translation():
        probe = SovAiFactsProbe(
            persona=indian_persona, locale="ta_Taml_IN", language="Tamil",
            models={"summary_model": object()}, cfg=cfg,
            provenance=Provenance(), profile={}, data={}, outcome_builder=ob,
        )
        assert probe._asset_locale == "en_IN"     # fell back to the en_IN bank
        assert probe._locale == "ta_Taml_IN"       # conversation locale preserved
        state = ConversationState(outcome=ob)
        turn1 = probe.get_verbatim_first_user_turn(state)

    assert turn1 == _MARKER                          # translated
    assert state.metadata.get("facts_probed")        # metadata still seeded
    assert _mt_warned(ob)                            # preview-only flag


# ---------------------------------------------------------------------------
# sov_ai_multilingual_parity — verbatim query rendering translated
# ---------------------------------------------------------------------------


def test_parity_variant_translates_turn1(indian_persona, cfg) -> None:
    from usersim.engine.probes.sov_ai_multilingual_parity.generator import (
        SovAiMultilingualParityProbe,
    )

    T.reset_translation_cache()
    ob = _builder()
    with _mock_translation():
        probe = SovAiMultilingualParityProbe(
            persona=indian_persona, locale="ta_Taml_IN", language="Tamil",
            models={"summary_model": object()}, cfg=cfg,
            provenance=Provenance(), profile={}, data={}, outcome_builder=ob,
        )
        # Shared bank -> asset_locale resolves to the en_IN base rendering.
        assert probe._asset_locale == "en_IN"
        state = ConversationState(outcome=ob)
        turn1 = probe.get_verbatim_first_user_turn(state)

    assert turn1 == _MARKER
    assert state.metadata.get("query_id")            # matched-pair join key seeded
    assert _mt_warned(ob)


def test_parity_variants_share_query_id_matched_pair(indian_persona, cfg) -> None:
    """Two India-language variants drawing the same en_IN persona pick the
    SAME query_id — so cross-language parity joins on query_id work."""
    from usersim.engine.probes.sov_ai_multilingual_parity.generator import (
        SovAiMultilingualParityProbe,
    )

    def _query_id_for(locale: str) -> str:
        probe = SovAiMultilingualParityProbe(
            persona=indian_persona, locale=locale, language="X",
            models={"summary_model": object()}, cfg=cfg,
            provenance=Provenance(), profile={}, data={},
            outcome_builder=_builder(),
        )
        assert probe._asset_locale == "en_IN"
        return probe._task.id

    assert _query_id_for("ta_Taml_IN") == _query_id_for("te_Telu_IN")


# ---------------------------------------------------------------------------
# safety_chat_pressure — verbatim target prompt translated
# ---------------------------------------------------------------------------


def test_safety_chat_pressure_variant_translates_turn1(indian_persona, cfg) -> None:
    from usersim.engine.probes.safety_chat_pressure.generator import (
        SafetyChatPressureProbe,
    )

    T.reset_translation_cache()
    ob = _builder()
    with _mock_translation():
        probe = SafetyChatPressureProbe(
            persona=indian_persona, locale="ta_Taml_IN", language="Tamil",
            models={"summary_model": object()}, cfg=cfg,
            provenance=Provenance(), profile={}, data={}, outcome_builder=ob,
        )
        assert probe._asset_locale == "en_IN"
        state = ConversationState(outcome=ob)
        turn1 = probe.get_verbatim_first_user_turn(state)

    assert turn1 == _MARKER
    assert state.metadata.get("target_request_id")
    assert _mt_warned(ob)


# ---------------------------------------------------------------------------
# safety_agentic — construction + asset routing (turn-1 translation lives in
# the custom run_dispatch, covered by the substrate _localize_verbatim test)
# ---------------------------------------------------------------------------


def test_safety_agentic_variant_constructs_and_routes(indian_persona, cfg) -> None:
    from usersim.engine.probes.safety_agentic.generator import SafetyAgenticProbe

    ob = _builder()
    probe = SafetyAgenticProbe(
        persona=indian_persona, locale="ta_Taml_IN", language="Tamil",
        models={"summary_model": object()}, cfg=cfg,
        provenance=Provenance(), profile={}, data={}, outcome_builder=ob,
    )
    assert probe._asset_locale == "en_IN"
    assert probe._locale == "ta_Taml_IN"
    # The translation gate fires because asset_locale != locale.
    with _mock_translation():
        assert probe._localize_verbatim("english action prompt") == _MARKER
    assert _mt_warned(ob)


# ---------------------------------------------------------------------------
# sov_ai_dynamic — generate-and-gate (no verbatim translation), en_IN taxonomy
# ---------------------------------------------------------------------------


def test_sov_ai_dynamic_variant_uses_en_in_taxonomy(indian_persona, cfg) -> None:
    from usersim.engine.probes.sov_ai_dynamic.generator import SovAiDynamicProbe

    ob = _builder()
    probe = SovAiDynamicProbe(
        persona=indian_persona, locale="ta_Taml_IN", language="Tamil",
        models={"summary_model": object()}, cfg=cfg,
        provenance=Provenance(), profile={}, data={}, outcome_builder=ob,
    )
    assert probe._asset_locale == "en_IN"
    assert probe._bank.taxonomy_id.startswith("en_IN")
    # No verbatim turn-1 -> generate-and-gate; no MT warning on construction.
    assert not _mt_warned(ob)
    prompt = probe.get_user_system_prompt()
    assert prompt  # en_IN prompt pack resolved (no abort)
    # CRITICAL: turn-1 is generated from this prompt, and the en_IN pack is
    # English — so the target-language directive MUST be appended, else the
    # agent writes English and the script gate exhausts (the bug that made
    # every sov_ai_dynamic variant row fail with USER_LANGUAGE_GATE_EXHAUSTED).
    assert "You MUST write your messages in Tamil" in prompt
    assert "native Tamil script" in prompt


def test_sov_ai_dynamic_romanized_variant_gets_romanized_directive(indian_persona, cfg) -> None:
    from usersim.engine.probes.sov_ai_dynamic.generator import SovAiDynamicProbe

    probe = SovAiDynamicProbe(
        persona=indian_persona, locale="ta_Latn_IN", language="Tamil",
        models={"summary_model": object()}, cfg=cfg,
        provenance=Provenance(), profile={}, data={}, outcome_builder=_builder(),
    )
    assert "romanized Tamil" in probe.get_user_system_prompt()


def test_sov_ai_dynamic_shipped_locale_prompt_unchanged(indian_persona, cfg) -> None:
    # Shipped locales must not get an appended directive (asset_locale == locale).
    from usersim.engine.probes.sov_ai_dynamic.generator import SovAiDynamicProbe

    probe = SovAiDynamicProbe(
        persona=indian_persona, locale="en_US", language="English",
        models={"summary_model": object()}, cfg=cfg,
        provenance=Provenance(), profile={}, data={}, outcome_builder=_builder(),
    )
    assert "You MUST write your messages in" not in probe.get_user_system_prompt()


# ---------------------------------------------------------------------------
# health_disclosure — English scaffold packs resolved against the en_IN base
# ---------------------------------------------------------------------------
#
# The family's prompt packs are English scaffolds registered under
# ``SHIPPED_LOCALES`` only, so resolving them against the CONVERSATION locale
# raises ``LocalePromptPackError`` for every variant. They must key off
# ``self._asset_locale`` (en_IN) instead, with the target language carried by
# the interpolated ``language_instruction``.

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
        persona=persona, locale=locale, language=language,
        models={"summary_model": object()}, cfg=cfg, provenance=Provenance(),
        profile=compute_behavioral_profile(persona, locale), data={},
        outcome_builder=_builder(),
    )


@pytest.mark.parametrize("label", _HEALTH_PROBE_LABELS)
def test_health_disclosure_variant_resolves_en_in_scaffold(
    label, indian_persona, cfg,
) -> None:
    probe = _health_probe(label, indian_persona, cfg, "ta_Taml_IN", "Tamil")

    assert probe._asset_locale == "en_IN"   # scaffold pack fell back to the base
    assert probe._locale == "ta_Taml_IN"     # conversation locale preserved
    prompt = probe.get_user_system_prompt()
    assert prompt                            # en_IN pack resolved (no abort)
    # The scaffold is English, so the target-language directive must carry the
    # conversation language — otherwise the user agent writes English and the
    # script gate exhausts every row.
    assert "You MUST write your messages in Tamil" in prompt
    assert "native Tamil script" in prompt
    assert probe.format_gate_prompt("hi", "history")  # gate pack resolved too


@pytest.mark.parametrize("label", _HEALTH_PROBE_LABELS)
def test_health_disclosure_romanized_variant_gets_romanized_directive(
    label, indian_persona, cfg,
) -> None:
    probe = _health_probe(label, indian_persona, cfg, "ta_Latn_IN", "Tamil")
    assert "romanized Tamil" in probe.get_user_system_prompt()


@pytest.mark.parametrize("label", _HEALTH_PROBE_LABELS)
def test_health_disclosure_shipped_locale_unchanged(
    label, indian_persona, cfg,
) -> None:
    # Shipped locales resolve their own pack — no fallback, no directive.
    probe = _health_probe(label, indian_persona, cfg, "en_US", "English")
    assert probe._asset_locale == "en_US"
    assert "You MUST write your messages in" not in probe.get_user_system_prompt()
