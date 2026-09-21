# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for the India multi-language stop-gap layer.

Covers three buckets:

- **New behavior** — variant locale resolution (presence-aware), simulation
  directives, trajectory_id locale discrimination, behavioral fall-through,
  the translation helper, and the substrate asset_locale routing + verbatim
  translation.
- **Graduation** — a native asset present -> no fallback, no translation,
  no ``USED_MACHINE_TRANSLATION``; absent -> fallback + translate + flag;
  an unknown/typo locale still fails fast (is NOT silently redirected).
- **Don't-break regressions** — ``available_locales()`` for the asset-driven
  probes stays ``== SHIPPED_LOCALES`` and the exact-parity locale maps are
  unchanged.
"""

from __future__ import annotations

from typing import Any

from usersim.engine.core import behavioral as B
from usersim.engine.core import locale as L
from usersim.engine.core import simulation as S
from usersim.engine.core import translation as T
from usersim.engine.core.identity import trajectory_id
from usersim.engine.core.outcomes import (
    OutcomeBuilder,
    Provenance,
    WarningKind,
)
from usersim.engine.core.probes import BankBackedProbe, BankVerbatimMixin

# ---------------------------------------------------------------------------
# core/locale.py — registry + presence-aware resolvers
# ---------------------------------------------------------------------------


class TestVariantRegistry:
    def test_expected_count(self) -> None:
        # 11 languages x (native + romanized) = 22.
        assert len(L.INDIA_VARIANT_LOCALES) == 22
        assert all(v.base_locale == "en_IN" for v in L.INDIA_VARIANT_LOCALES.values())

    def test_variants_excluded_from_shipped_locales(self) -> None:
        # The stop-gap must NOT touch the asset-complete set.
        assert set(L.INDIA_VARIANT_LOCALES).isdisjoint(set(L.SHIPPED_LOCALES))

    def test_native_and_romanized_twins(self) -> None:
        native = L.INDIA_VARIANT_LOCALES["ta_Taml_IN"]
        latin = L.INDIA_VARIANT_LOCALES["ta_Latn_IN"]
        assert native.language_display == latin.language_display == "Tamil"
        assert native.lingua_name == "TAMIL" and native.romanized is False
        assert latin.lingua_name is None and latin.romanized is True
        assert native.script_ranges == L.TAMIL_RANGES
        assert latin.script_ranges == L.LATIN_RANGES

    def test_lingua_less_native_variants_have_none(self) -> None:
        # Kannada/Malayalam/Odia/Nepali are absent from the lingua build;
        # Marathi is deliberately None too (Devanagari-shared with the shipped
        # Hindi locale -> excluded from the detector to protect Hindi accuracy).
        for loc in ("kn_Knda_IN", "ml_Mlym_IN", "or_Orya_IN", "ne_Deva_IN", "mr_Deva_IN"):
            assert L.INDIA_VARIANT_LOCALES[loc].lingua_name is None
            # ...but they still carry native script ranges for in-sim checks.
            assert L.INDIA_VARIANT_LOCALES[loc].script_ranges

    def test_marathi_is_script_only_devanagari(self) -> None:
        mr = L.INDIA_VARIANT_LOCALES["mr_Deva_IN"]
        assert mr.lingua_name is None  # avoid Hindi/Marathi lingua confusion
        assert mr.script_ranges == L.DEVANAGARI_RANGES

    def test_detector_includes_distinct_script_variant_languages(self) -> None:
        # The language_compliance detector must include the distinct-script
        # variant languages, else their text is misclassified as the nearest
        # shipped language -> requested_language_match_rate silently 0.
        from usersim.engine.core.language_detection import (
            detect_language_name,
            reset_detector,
        )

        reset_detector()
        try:
            assert detect_language_name("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?") == "TAMIL"
            assert detect_language_name("আপনি আজকে কেমন আছেন বলুন তো?") == "BENGALI"
            assert detect_language_name("మీరు ఈరోజు ఎలా ఉన్నారు?") == "TELUGU"
            # Shipped Hindi still detects as Hindi (Marathi excluded to protect it).
            assert detect_language_name("नमस्ते") == "HINDI"
        finally:
            reset_detector()

    def test_lingua_names_resolve(self) -> None:
        from lingua import Language

        for v in L.INDIA_VARIANT_LOCALES.values():
            if v.lingua_name is not None:
                assert hasattr(Language, v.lingua_name)

    def test_accessors_fall_through_for_variants(self) -> None:
        assert L.expected_language_name("ta_Taml_IN") == "TAMIL"
        assert L.expected_language_name("ta_Latn_IN") is None
        assert L.expected_language_display("bn_Beng_IN") == "Bengali"
        assert L.expected_script_ranges("bn_Beng_IN") == L.BENGALI_RANGES
        assert L.expected_script_ranges("bn_Latn_IN") == L.LATIN_RANGES

    def test_unknown_locale_still_unregistered(self) -> None:
        assert L.expected_language_name("xx_YY") is None
        assert L.expected_script_ranges("xx_YY") == ()

    def test_new_script_ranges_in_buckets(self) -> None:
        bucket_ranges = {r for ranges in L.SCRIPT_BUCKETS.values() for r in ranges}
        for v in L.INDIA_VARIANT_LOCALES.values():
            for r in v.script_ranges:
                assert r in bucket_ranges


class TestPresenceAwareResolvers:
    def test_persona_dataset_locale_fallback(self) -> None:
        assert L.persona_dataset_locale("ta_Taml_IN") == "en_IN"
        assert L.persona_dataset_locale("en_US") == "en_US"  # non-variant unchanged
        assert L.persona_dataset_locale("xx_YY") == "xx_YY"  # unknown unchanged

    def test_persona_dataset_locale_prefers_native_when_present(self, tmp_path) -> None:
        # Graduation: a native parquet present -> use it.
        (tmp_path / "ta_Taml_IN.parquet").write_bytes(b"x")
        assert L.persona_dataset_locale("ta_Taml_IN", datasets_dir=tmp_path) == "ta_Taml_IN"
        # Absent -> base.
        assert L.persona_dataset_locale("te_Telu_IN", datasets_dir=tmp_path) == "en_IN"

    def test_asset_locale_presence_aware(self) -> None:
        assert L.asset_locale("ta_Taml_IN") == "en_IN"
        assert L.asset_locale("ta_Taml_IN", exists=lambda loc: True) == "ta_Taml_IN"
        assert L.asset_locale("ta_Taml_IN", exists=lambda loc: False) == "en_IN"
        assert L.asset_locale("pt_BR", exists=lambda loc: False) == "pt_BR"

    def test_prompt_pack_locale_presence_aware(self) -> None:
        assert L.prompt_pack_locale("bn_Beng_IN") == "en_IN"
        assert L.prompt_pack_locale("bn_Beng_IN", has=lambda loc: True) == "bn_Beng_IN"
        assert L.prompt_pack_locale("ja_JP") == "ja_JP"


# ---------------------------------------------------------------------------
# core/simulation.py — directives derived from the registry
# ---------------------------------------------------------------------------


class TestSimulationDirectives:
    def test_native_scripts_are_token_dense(self) -> None:
        assert S._is_non_ascii_locale("ta_Taml_IN")
        assert not S._is_non_ascii_locale("ta_Latn_IN")  # Latin twin is ASCII-ish

    def test_native_script_directive(self) -> None:
        d = S._script_directive("ta_Taml_IN")
        assert "native Tamil script" in d and "romanize" in d

    def test_romanized_directive_and_judge_clause(self) -> None:
        assert "ta_Latn_IN" in S.ROMANIZED_LOCALES
        directive = S._script_directive("ta_Latn_IN")
        assert "romanized Tamil" in directive and "Latin" in directive
        clause = S._language_judge_clause("ta_Latn_IN")
        assert "[LANGUAGE]" in clause and "Tamil" in clause

    def test_shipped_locale_directives_unchanged(self) -> None:
        assert "hi_Latn_IN" in S.ROMANIZED_LOCALES
        assert S._is_non_ascii_locale("hi_Deva_IN")


# ---------------------------------------------------------------------------
# trajectory_id — locale discrimination (the collision fix)
# ---------------------------------------------------------------------------


class TestTrajectoryIdLocale:
    BASE = dict(
        persona_uuid="abcd",
        probe_family="sov_ai_facts",
        probe_variant="v",
        scenario_seed=None,
        user_model="u",
        assistant_model="a",
        prompt_version="v1.0",
    )

    def test_locale_salt_discriminates(self) -> None:
        # Same persona + probe + models, two India variants -> different ids.
        ta = trajectory_id(**self.BASE, extra_keys=[("locale", "ta_Taml_IN")])
        te = trajectory_id(**self.BASE, extra_keys=[("locale", "te_Telu_IN")])
        assert ta != te

    def test_same_locale_stable(self) -> None:
        a = trajectory_id(**self.BASE, extra_keys=[("locale", "ta_Taml_IN")])
        b = trajectory_id(**self.BASE, extra_keys=[("locale", "ta_Taml_IN")])
        assert a == b


# ---------------------------------------------------------------------------
# core/behavioral.py
# ---------------------------------------------------------------------------


class TestBehavioral:
    def test_conversation_language_falls_through(self) -> None:
        assert B.get_conversation_language("ta_Taml_IN") == "Tamil"
        assert B.get_conversation_language("ta_Latn_IN") == "Tamil"
        assert B.get_conversation_language("en_US") == "English"
        assert B.get_conversation_language("zz_ZZ") == "English"

    def test_education_uses_en_in_via_persona_dataset_locale(self) -> None:
        # The generator computes the profile with persona_dataset_locale(locale);
        # for a variant that resolves to en_IN's exact vocabulary.
        loc = L.persona_dataset_locale("ta_Taml_IN")
        assert loc == "en_IN"
        assert B._education_ordinal("Graduate & above", loc) == 1.0
        # Without the routing, the variant locale would miss the exact map.
        assert B._education_ordinal("Graduate & above", "ta_Taml_IN") != 1.0


# ---------------------------------------------------------------------------
# core/translation.py
# ---------------------------------------------------------------------------


class TestTranslation:
    def setup_method(self) -> None:
        T.reset_translation_cache()

    def teardown_method(self) -> None:
        T.reset_translation_cache()

    def test_translate_and_cache(self, monkeypatch) -> None:
        calls = []

        def fake(models, alias, msgs, **kw):
            calls.append(msgs[0]["content"])
            return {"content": "अनुवादित"}

        monkeypatch.setattr(T, "call_llm", fake)
        out = T.translate_user_turn({}, "How often can I take this?", target_language="Tamil")
        assert out == "अनुवादित"
        # Cached: a second identical call does not re-invoke the model.
        T.translate_user_turn({}, "How often can I take this?", target_language="Tamil")
        assert len(calls) == 1
        # The prompt preserves false premises + names the language.
        assert "Tamil" in calls[0] and "premise" in calls[0].lower()

    def test_romanize_flag_changes_prompt(self, monkeypatch) -> None:
        seen = {}

        def fake(models, alias, msgs, **kw):
            seen["prompt"] = msgs[0]["content"]
            return {"content": "x"}

        monkeypatch.setattr(T, "call_llm", fake)
        T.translate_user_turn({}, "hi", target_language="Tamil", romanize=True)
        assert "Latin/Roman alphabet" in seen["prompt"]

    def test_error_passthrough(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise RuntimeError("provider down")

        monkeypatch.setattr(T, "call_llm", boom)
        assert T.translate_user_turn({}, "unchanged", target_language="Odia") == "unchanged"

    def test_empty_passthrough(self) -> None:
        assert T.translate_user_turn({}, "", target_language="Tamil") == ""


# ---------------------------------------------------------------------------
# BankBackedProbe substrate — asset_locale routing + verbatim translation
# ---------------------------------------------------------------------------


class _FakeBank:
    bank_version = "vfake"
    bank_id = "fake_bank"


class _FakeTask:
    id = "t1"
    placeholder = False
    text = "How often can I take this medication? It cures diabetes, right?"


class _FakeProbe(BankVerbatimMixin, BankBackedProbe):
    """Minimal bank-backed probe: a per-locale-file loader that only 'has'
    an en_IN bank (so a variant falls back to en_IN)."""

    label = "fake_probe"
    verbatim_field = "text"
    bank_version_key = None
    _native_locales: set[str] = {"en_IN", "en_US", "pt_BR"}

    @staticmethod
    def bank_loader(locale: str) -> Any:
        if locale in _FakeProbe._native_locales:
            return _FakeBank()
        raise FileNotFoundError(f"no bank for {locale}")

    def derive_task(self, persona: dict[str, Any], bank: Any, *, cfg: Any) -> Any:
        return _FakeTask()

    def get_user_system_prompt(self) -> str:
        return "system"

    def get_assistant_system_prompt(self) -> str:
        return ""


def _make_probe(locale: str) -> _FakeProbe:
    return _FakeProbe(
        persona={"first_name": "A", "age": 30},
        locale=locale,
        language="Tamil",
        models={"summary_model": object()},
        cfg=type("Cfg", (), {"random_seed": None})(),
        provenance=Provenance(),
        outcome_builder=OutcomeBuilder(provenance=Provenance()),
    )


class TestAssetLocaleRouting:
    def test_variant_falls_back_to_base(self) -> None:
        probe = _make_probe("ta_Taml_IN")
        assert probe._asset_locale == "en_IN"  # native ta bank absent -> en_IN
        assert probe._locale == "ta_Taml_IN"  # conversation locale unchanged

    def test_non_variant_unchanged(self) -> None:
        probe = _make_probe("pt_BR")
        assert probe._asset_locale == "pt_BR"

    def test_native_asset_present_no_fallback(self) -> None:
        # Graduation: a native ta bank exists -> asset_locale stays the variant.
        _FakeProbe._native_locales.add("ta_Taml_IN")
        try:
            probe = _make_probe("ta_Taml_IN")
            assert probe._asset_locale == "ta_Taml_IN"
        finally:
            _FakeProbe._native_locales.discard("ta_Taml_IN")


class TestVerbatimTranslation:
    def setup_method(self) -> None:
        T.reset_translation_cache()

    def teardown_method(self) -> None:
        T.reset_translation_cache()

    def test_variant_translates_and_flags(self, monkeypatch) -> None:
        monkeypatch.setattr(T, "call_llm", lambda *a, **k: {"content": "TAMIL_TEXT"})
        probe = _make_probe("ta_Taml_IN")
        out = probe._localize_verbatim(_FakeTask.text)
        assert out == "TAMIL_TEXT"
        # A USED_MACHINE_TRANSLATION warning is recorded on the outcome builder.
        kinds = [w.kind for w in probe._outcome_builder._outcome.warnings]
        assert WarningKind.USED_MACHINE_TRANSLATION in kinds

    def test_shipped_locale_no_translation(self, monkeypatch) -> None:
        called = []
        monkeypatch.setattr(T, "call_llm", lambda *a, **k: called.append(1) or {"content": "x"})
        probe = _make_probe("pt_BR")
        out = probe._localize_verbatim(_FakeTask.text)
        assert out == _FakeTask.text  # unchanged
        assert not called  # no translation call

    def test_graduated_native_variant_no_translation(self, monkeypatch) -> None:
        called = []
        monkeypatch.setattr(T, "call_llm", lambda *a, **k: called.append(1) or {"content": "x"})
        _FakeProbe._native_locales.add("ta_Taml_IN")
        try:
            probe = _make_probe("ta_Taml_IN")  # asset_locale == locale
            out = probe._localize_verbatim(_FakeTask.text)
        finally:
            _FakeProbe._native_locales.discard("ta_Taml_IN")
        assert out == _FakeTask.text
        assert not called


# ---------------------------------------------------------------------------
# Don't-break regressions
# ---------------------------------------------------------------------------


class TestDontBreak:
    def test_prompt_pack_available_locales_unchanged(self) -> None:
        # The variant fallback is a call-time redirect (asset_locale=en_IN),
        # NOT a dict mutation — so available_locales() must still equal the
        # shipped set (guards the existing all-shipped-locale suites).
        from usersim.engine.probes.sov_ai_dynamic import prompts as dyn_p
        from usersim.engine.probes.sov_ai_facts import prompts as facts_p
        from usersim.engine.probes.sov_ai_multilingual_parity import (
            prompts as par_p,
        )

        shipped = set(L.SHIPPED_LOCALES)
        assert set(facts_p.available_locales()) == shipped
        assert set(dyn_p.available_locales()) == shipped
        assert set(par_p.available_locales()) == shipped

    def test_locale_parity_maps_unchanged(self) -> None:
        # Variants live only in INDIA_VARIANT_LOCALES, never in the three
        # exact-parity maps keyed on SHIPPED_LOCALES.
        assert set(L.LOCALE_TO_LANGUAGE_NAME) == set(L.SHIPPED_LOCALES)
        assert set(L.LOCALE_TO_LANGUAGE_DISPLAY) == set(L.SHIPPED_LOCALES)
        assert set(L.LOCALE_TO_EXPECTED_SCRIPT_RANGES) == set(L.SHIPPED_LOCALES)
        assert set(L.supported_locales()) == set(L.SHIPPED_LOCALES)
