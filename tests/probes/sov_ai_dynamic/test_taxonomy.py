# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``usersim.engine.probes.sov_ai_dynamic.taxonomy`` and the taxonomies it ships.

The format itself is tested in ``tests/engine/core/test_probing_taxonomy.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from usersim.engine.core._assets import probe_assets_dir
from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.core.probing_taxonomy import (
    MIN_SUBTOPIC_HINTS,
    ProbingTaxonomy,
    load_probing_taxonomy,
    reset_probing_taxonomy_cache,
)
from usersim.engine.probes.sov_ai_dynamic.taxonomy import (
    default_probing_taxonomy_path,
    load_probing_taxonomy_for_locale,
    probing_taxonomy_path_for,
)


def _write(tmp_path: Path, content: str, name: str = "tax.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _minimal_category(cid: str = "cat-a", hints: int = 2) -> str:
    """Render one category block (already indented for inclusion under `categories:`)."""
    hint_lines = "\n".join(f"      - 'hint {cid} #{i}'" for i in range(hints))
    return f"  - id: {cid}\n    invitation: 'invitation text for {cid}'\n    subtopic_hints:\n{hint_lines}\n"


def _minimal_taxonomy(extra_categories: str = "") -> str:
    return (
        'schema_version: "v0.1"\n'
        "locale: en_US\n"
        "taxonomy_id: test_taxonomy\n"
        "taxonomy_version: v0.1.0\n"
        "categories:\n" + _minimal_category() + extra_categories
    )


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    """Keep the module-level cache from leaking across tests."""
    reset_probing_taxonomy_cache()
    yield
    reset_probing_taxonomy_cache()


class TestCache:
    def test_load_once_cached(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _minimal_taxonomy().replace("locale: en_US", "locale: pt_BR"))
        with patch.dict(os.environ, {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(path)}):
            t1 = load_probing_taxonomy_for_locale("pt_BR")
            t2 = load_probing_taxonomy_for_locale("pt_BR")
        assert t1 is t2, "cache returned a different instance on second call"

    def test_env_override_picks_up_different_file(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            _minimal_taxonomy()
            .replace("locale: en_US", "locale: pt_BR")
            .replace("taxonomy_id: test_taxonomy", "taxonomy_id: override_id"),
        )
        with patch.dict(os.environ, {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(path)}):
            tax = load_probing_taxonomy_for_locale("pt_BR")
        assert tax.taxonomy_id == "override_id"

    def test_reset_drops_cache(self, tmp_path: Path) -> None:
        path_a = _write(
            tmp_path,
            _minimal_taxonomy()
            .replace("locale: en_US", "locale: pt_BR")
            .replace("taxonomy_id: test_taxonomy", "taxonomy_id: id_a"),
            name="a.yaml",
        )
        path_b = _write(
            tmp_path,
            _minimal_taxonomy()
            .replace("locale: en_US", "locale: pt_BR")
            .replace("taxonomy_id: test_taxonomy", "taxonomy_id: id_b"),
            name="b.yaml",
        )

        with patch.dict(os.environ, {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(path_a)}):
            assert load_probing_taxonomy_for_locale("pt_BR").taxonomy_id == "id_a"
        reset_probing_taxonomy_cache()
        with patch.dict(os.environ, {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(path_b)}):
            assert load_probing_taxonomy_for_locale("pt_BR").taxonomy_id == "id_b"

    def test_default_path_resolution(self) -> None:
        # No env override → default relative path under assets/.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR", None)
            assert probing_taxonomy_path_for("pt_BR") == default_probing_taxonomy_path("pt_BR")
            assert default_probing_taxonomy_path("pt_BR").parts[-3:] == (
                "sov_ai_dynamic",
                "pt_BR",
                "categories.yaml",
            )


# ---------------------------------------------------------------------------
# Integration: every shipped per-locale taxonomy
# ---------------------------------------------------------------------------


# Taxonomies not yet audited by a native-speaker reviewer, which therefore
# carry ``placeholder: true``.
_AWAITING_NATIVE_REVIEW = frozenset({"en_GB", "de_DE"})

# Locales that share a country topic universe (en_IN + hi_Deva_IN both
# cover India) get the same category prefix.
_CATEGORY_PREFIX_BY_LOCALE = {
    "en_US": "usa",
    "pt_BR": "brazil",
    "ja_JP": "japan",
    "fr_FR": "france",
    "de_DE": "germany",
    "hi_Deva_IN": "india",
    "hi_Latn_IN": "india",
    "en_IN": "india",
    "en_SG": "singapore",
    "en_GB": "uk",
    "ko_KR": "korea",
}


# Per-locale heuristic markers — every shipped invitation should
# contain at least one of these to catch accidental English drift in
# non-English locales (and accidental non-English drift in en_*).
# Keep these conservative; the goal is to fire on obvious mistakes,
# not to police vocabulary.
_LANGUAGE_MARKERS = {
    "en_US": ("the", "you", "your"),
    "en_IN": ("the", "you", "your"),
    "pt_BR": ("você", "sua", "seu", "para"),
    "ja_JP": ("について", "です", "ます", "あなた"),
    "fr_FR": ("vous", "votre", "votre"),
    "de_DE": ("sie", "ihre", "ihnen"),
    "hi_Deva_IN": ("आप", "के", "में", "है"),
    "hi_Latn_IN": ("aap", "mein", "ke", "hai"),
    "en_SG": ("you", "your", "singapore"),
    "en_GB": ("the", "you", "your"),
    "ko_KR": ("당신", "한국", "대해", "하세요"),
}


def test_taxonomy_prefix_metadata_covers_shipped_locales() -> None:
    assert set(_CATEGORY_PREFIX_BY_LOCALE) == set(SHIPPED_LOCALES)
    assert set(_LANGUAGE_MARKERS) == set(SHIPPED_LOCALES)


@pytest.mark.parametrize(
    "locale,prefix",
    [(loc, _CATEGORY_PREFIX_BY_LOCALE[loc]) for loc in SHIPPED_LOCALES],
)
class TestEveryShippedLocaleTaxonomy:
    """Sanity-check every per-locale taxonomy ships in the right shape."""

    def _load(self, locale: str) -> ProbingTaxonomy:
        return load_probing_taxonomy(probe_assets_dir("sov_ai_dynamic") / locale / "categories.yaml")

    def test_loads(self, locale: str, prefix: str) -> None:
        tax = self._load(locale)
        assert tax.locale == locale
        assert tax.schema_version == "v0.1"
        assert tax.taxonomy_id.startswith(locale)

    def test_placeholder_flag_matches_review_status(self, locale: str, prefix: str) -> None:
        tax = self._load(locale)
        assert tax.placeholder is (locale in _AWAITING_NATIVE_REVIEW)

    def test_10_categories_with_country_prefix(self, locale: str, prefix: str) -> None:
        tax = self._load(locale)
        assert len(tax.categories) == 10
        for cat in tax.categories:
            assert cat.id.startswith(f"{prefix}-"), (
                f"{locale}: category {cat.id!r} does not start with {prefix!r}- prefix"
            )

    def test_canonical_10_topic_set(self, locale: str, prefix: str) -> None:
        # Every shipped locale uses the canonical topic set so per-
        # category comparison across locales is direct.
        tax = self._load(locale)
        topics = {cat.id.removeprefix(f"{prefix}-") for cat in tax.categories}
        assert topics == {
            "geography",
            "history",
            "culture",
            "poetry-and-music",
            "public-services",
            "health-systems",
            "education",
            "labor-and-benefits",
            "finance-and-consumer",
            "digital-services",
        }, f"{locale}: unexpected topic set {topics}"

    def test_every_category_has_invitation_and_hints(self, locale: str, prefix: str) -> None:
        tax = self._load(locale)
        for cat in tax.categories:
            assert cat.invitation, f"{locale}::{cat.id}: empty invitation"
            assert len(cat.subtopic_hints) >= MIN_SUBTOPIC_HINTS, (
                f"{locale}::{cat.id}: only {len(cat.subtopic_hints)} hints; loader requires {MIN_SUBTOPIC_HINTS}"
            )

    def test_invitations_use_locale_language(self, locale: str, prefix: str) -> None:
        # Light heuristic that catches accidental English drift in
        # non-English locales (and the reverse for English locales).
        markers = _LANGUAGE_MARKERS[locale]
        tax = self._load(locale)
        for cat in tax.categories:
            inv = cat.invitation.lower()
            assert any(m.lower() in inv for m in markers), (
                f"{locale}::{cat.id}: invitation has no expected "
                f"language markers ({markers}); first 100 chars: "
                f"{cat.invitation[:100]!r}"
            )


class TestIndiaSharesTopicUniverse:
    """en_IN and hi_Deva_IN cover the same country with two language surfaces."""

    def test_same_category_ids(self) -> None:
        en = load_probing_taxonomy((probe_assets_dir("sov_ai_dynamic") / "en_IN/categories.yaml"))
        hi = load_probing_taxonomy((probe_assets_dir("sov_ai_dynamic") / "hi_Deva_IN/categories.yaml"))
        assert set(en.category_ids()) == set(hi.category_ids())

    def test_taxonomy_ids_remain_locale_distinct(self) -> None:
        en = load_probing_taxonomy((probe_assets_dir("sov_ai_dynamic") / "en_IN/categories.yaml"))
        hi = load_probing_taxonomy((probe_assets_dir("sov_ai_dynamic") / "hi_Deva_IN/categories.yaml"))
        # Same topics, different language surfaces → different ids.
        assert en.taxonomy_id != hi.taxonomy_id
