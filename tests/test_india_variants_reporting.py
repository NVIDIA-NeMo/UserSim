# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""India language-variant coverage for the reporting / preview layer.

Verifies the locale->country/flag maps handle the India variants, and that
persona-facet lookups resolve the source parquet via the presence-aware
``persona_dataset_locale`` rather than a non-existent ``ta_Taml_IN.parquet``.
"""

from __future__ import annotations

from usersim.engine.core.locale import (
    INDIA_VARIANT_LOCALES,
    persona_dataset_locale,
)


def test_preview_country_maps_cover_variants() -> None:
    from usersim.engine.core.preview import (
        _LOCALE_COUNTRIES,
        _LOCALE_FLAGS,
    )

    for loc in INDIA_VARIANT_LOCALES:
        assert _LOCALE_COUNTRIES.get(loc) == "India"
        assert _LOCALE_FLAGS.get(loc) == "🇮🇳"


def test_dashboard_locale_label_variant_flag() -> None:
    from usersim.reporting.dashboard import _locale_label

    # Any India locale (variant included) flies the India flag; the code is
    # still shown so ta_Taml_IN vs ta_Latn_IN remain distinguishable.
    label = _locale_label("ta_Taml_IN")
    assert "🇮🇳" in label and "ta_Taml_IN" in label
    assert "🌐" in _locale_label("zz_ZZ")


def test_persona_source_resolves_via_persona_dataset_locale() -> None:
    # A variant has no parquet of its own; persona_dataset_locale routes it to
    # en_IN so persona-facet lookups scan the right source.
    assert persona_dataset_locale("ta_Taml_IN") == "en_IN"
    assert persona_dataset_locale("en_US") == "en_US"
