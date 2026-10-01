# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for language-matched persona sampling (``cli/_pipeline.py``).

Scope: the three helpers that narrow the persona population to speakers of
the conversation language — ``persona_language_for_locale``,
``persona_language_filter``, and ``persona_sampler_params`` — plus the
cached population diagnostic. No LLM calls or managed persona assets:
the expected language strings were read off the shipped NGC corpora once
and are pinned here; count tests use tiny temporary parquet fixtures.

The India locales are the interesting case. All three India corpora
(``en_IN`` / ``hi_Deva_IN`` / ``hi_Latn_IN``) are the SAME pan-India
population in three renderings, so:

- a locale having its own parquet does not make it language-appropriate
  (Hindi is the first language of 42% of ``hi_Deva_IN``), and
- the spelling of ``first_language`` follows the CORPUS, not the
  conversation locale (``hi_Deva_IN`` stores "हिंदी", ``hi_Latn_IN``
  stores "Hindi").
"""

from __future__ import annotations

import logging

import pytest

from usersim import cli
from usersim.cli._persona_language import (
    _column_value_counts,
    log_persona_language_population,
    persona_language_filter,
    persona_language_for_locale,
    persona_sampler_params,
)
from usersim.cli._pipeline import build_simulator_config_builder


class _FakeDD:
    """Stand-in for the ``data_designer`` module's sampler params.

    ``persona_sampler_params`` only ever constructs ``PersonSamplerParams``,
    so a dict subclass is enough to assert on the kwargs it assembles
    without importing Data Designer's real (validating) model.
    """

    class PersonSamplerParams(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)


# ---------------------------------------------------------------------------
# Which language, and how the sampled corpus spells it
# ---------------------------------------------------------------------------


class TestPersonaLanguageForLocale:
    @pytest.mark.parametrize(
        "locale,expected",
        [
            ("ta_Taml_IN", "Tamil"),
            ("ml_Mlym_IN", "Malayalam"),
            ("bn_Beng_IN", "Bengali"),
            ("te_Telu_IN", "Telugu"),
            ("pa_Guru_IN", "Punjabi"),
            ("ur_Arab_IN", "Urdu"),
        ],
    )
    def test_variants_use_en_in_latin_names(self, locale, expected):
        """Preview variants sample ``en_IN``, which spells languages in Latin."""
        assert persona_language_for_locale(locale, "en_IN") == expected

    def test_romanized_twin_matches_its_native_twin(self):
        """A language's romanized locale resolves to the same language."""
        assert persona_language_for_locale("ta_Latn_IN", "en_IN") == persona_language_for_locale("ta_Taml_IN", "en_IN")

    def test_devanagari_corpus_uses_devanagari_name(self):
        """``hi_Deva_IN`` stores "हिंदी" — the Latin "Hindi" matches 0 rows."""
        assert persona_language_for_locale("hi_Deva_IN", "hi_Deva_IN") == "हिंदी"

    def test_romanized_hindi_corpus_uses_latin_name(self):
        assert persona_language_for_locale("hi_Latn_IN", "hi_Latn_IN") == "Hindi"

    def test_spelling_follows_the_corpus_not_the_locale(self):
        """Same conversation locale, different corpus -> different spelling.

        Pins the keying decision: were this keyed on the conversation
        locale, a Hindi run drawing from ``en_IN`` would filter on the
        Devanagari form and match nothing.
        """
        assert persona_language_for_locale("hi_Deva_IN", "en_IN") == "Hindi"
        assert persona_language_for_locale("hi_Deva_IN", "hi_Deva_IN") == "हिंदी"

    def test_display_qualifiers_are_stripped(self):
        """Corpus names languages plainly; display names carry qualifiers."""
        assert persona_language_for_locale("en_IN", "en_IN") == "English"
        assert persona_language_for_locale("hi_Latn_IN", "en_IN") == "Hindi"

    def test_persona_locale_defaults_to_conversation_locale(self):
        assert persona_language_for_locale("hi_Deva_IN") == persona_language_for_locale("hi_Deva_IN", "hi_Deva_IN")

    def test_unknown_locale_returns_none(self):
        """None so callers no-op rather than filter on a bogus value."""
        assert persona_language_for_locale("zz_ZZ") is None


# ---------------------------------------------------------------------------
# Default policy: which locales get matched at all
# ---------------------------------------------------------------------------


class TestBothEntryPointsDefaultOff:
    """The CLI and the notebook must agree for a given locale.

    A per-locale default policy consulted only when the caller leaves
    ``match_persona_language`` unset is unreachable from the CLI --
    ``--match-persona-language`` is ``store_true``, so argparse always
    supplies a real bool -- while the notebook, passing nothing, always
    reaches it. The same ``--locale ta_Taml_IN`` would then draw
    a pan-India population from the CLI and Tamil speakers from a
    notebook cell, in the module whose stated job is keeping the two
    surfaces in lockstep.

    Now there is one behaviour: off unless asked for, from either path.
    """

    def test_builder_default_is_off(self):
        import inspect

        sig = inspect.signature(build_simulator_config_builder)
        assert sig.parameters["match_persona_language"].default is False

    def test_cli_default_is_off(self):
        args = cli._build_parser().parse_args(
            ["simulate", "--locale", "ta_Taml_IN", "--num-rows", "1", "--out", "/tmp/x"]
        )
        assert args.match_persona_language is False

    def test_cli_flag_turns_it_on(self):
        args = cli._build_parser().parse_args(
            ["simulate", "--locale", "ta_Taml_IN", "--num-rows", "1", "--out", "/tmp/x", "--match-persona-language"]
        )
        assert args.match_persona_language is True

    def test_panel_defaults_off_like_simulate(self):
        """`usersim panel` must agree with `usersim simulate`.

        A panel exists to freeze the population a run draws from, so a
        panel built from a different population than the run it seeds
        defeats the point. panel.py used to build PersonSamplerParams
        inline instead of calling persona_sampler_params, so it could
        never language-match at all.
        """
        args = cli._build_parser().parse_args(
            ["panel", "--locale", "ta_Taml_IN", "--num-personas", "5", "--out", "/tmp/p.parquet"]
        )
        assert args.match_persona_language is False

    def test_panel_flag_turns_it_on(self):
        args = cli._build_parser().parse_args(
            [
                "panel",
                "--locale",
                "ta_Taml_IN",
                "--num-personas",
                "5",
                "--out",
                "/tmp/p.parquet",
                "--match-persona-language",
            ]
        )
        assert args.match_persona_language is True

    def test_panel_help_explains_en_in_full_population_mode(self, capsys):
        with pytest.raises(SystemExit):
            cli._build_parser().parse_args(["panel", "--help"])
        out = " ".join(capsys.readouterr().out.split())
        assert "en_IN" in out
        assert "full India persona corpus" in out

    def test_panel_uses_the_shared_sampler_helper(self):
        """Pinning the call, not just the flag: an inline
        PersonSamplerParams would satisfy the two tests above while still
        ignoring the language."""
        import inspect

        from usersim.cli import panel

        src = inspect.getsource(panel)
        assert "persona_sampler_params(" in src
        assert "dd.PersonSamplerParams(" not in src

    def test_no_locale_default_policy_survives(self):
        """The policy is gone, not merely bypassed -- a reintroduced one
        would recreate the divergence."""
        import usersim.cli._pipeline as pipeline

        assert not hasattr(pipeline, "matches_persona_language_by_default")


# ---------------------------------------------------------------------------
# Assembling the sampler params
# ---------------------------------------------------------------------------


class TestPersonaSamplerParams:
    def test_no_match_leaves_population_unfiltered(self):
        p = persona_sampler_params(_FakeDD(), "en_US", "en_US", match_language=False)
        assert p == {"locale": "en_US", "with_synthetic_personas": True}

    def test_match_adds_first_language_filter(self):
        p = persona_sampler_params(_FakeDD(), "en_IN", "ml_Mlym_IN", match_language=True)
        assert p["select_field_values"] == {"first_language": ["Malayalam"]}
        assert p["locale"] == "en_IN"

    def test_match_uses_corpus_spelling_for_hindi(self):
        p = persona_sampler_params(
            _FakeDD(),
            "hi_Deva_IN",
            "hi_Deva_IN",
            match_language=True,
        )
        assert p["select_field_values"] == {"first_language": ["हिंदी"]}

    def test_match_is_noop_for_unknown_language(self):
        p = persona_sampler_params(_FakeDD(), "en_US", "zz_ZZ", match_language=True)
        assert "select_field_values" not in p

    @pytest.mark.parametrize(
        "corpus,locale",
        [
            ("ja_JP", "ja_JP"),
            ("en_US", "en_US"),
            ("pt_BR", "pt_BR"),
            ("ko_KR", "ko_KR"),
            ("fr_FR", "fr_FR"),
            ("en_SG", "en_SG"),
        ],
    )
    def test_match_is_noop_where_corpus_lacks_the_column(self, corpus, locale):
        """Only the India datasets carry ``first_language``.

        Constraining elsewhere would filter on a field the corpus does not
        have, so the flag must be safe to leave on across a mixed panel
        rather than erroring on the first non-India row.
        """
        p = persona_sampler_params(_FakeDD(), corpus, locale, match_language=True)
        assert "select_field_values" not in p


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestSimulateFlag:
    def test_defaults_to_off(self):
        """Opt-in from the CLI: narrowing the population is never implicit."""
        args = cli._build_parser().parse_args(
            ["simulate", "--locale", "ta_Taml_IN", "--num-rows", "1", "--out", "/tmp/x"]
        )
        assert args.match_persona_language is False

    def test_flag_turns_it_on(self):
        args = cli._build_parser().parse_args(
            ["simulate", "--locale", "ta_Taml_IN", "--num-rows", "1", "--out", "/tmp/x", "--match-persona-language"]
        )
        assert args.match_persona_language is True

    def test_help_renders(self, capsys):
        """argparse %-expands help strings — a bare '%' raises at --help time."""
        with pytest.raises(SystemExit):
            cli._build_parser().parse_args(["simulate", "--help"])
        out = " ".join(capsys.readouterr().out.split())
        assert "--match-persona-language" in out
        assert "en_IN" in out
        assert "full India persona corpus" in out


class TestManifestRecordsWhatWasApplied:
    """A run must say which people it was allowed to draw.

    ``ScopeInfo`` already records locales, probe mix and panel path.
    Language matching changes *which people were measured*, so two runs
    of the same locale that differ on it are not comparable -- and
    nothing in the output said so.

    Recorded as the filter APPLIED, not the flag REQUESTED, because the
    two disagree: a corpus with no language column silently no-ops, and
    the two Hindi corpora apply different values for the same language.
    """

    def test_absent_when_not_requested(self):
        assert persona_language_filter("en_IN", "ta_Taml_IN", match_language=False) is None

    def test_records_the_value_actually_used(self):
        assert persona_language_filter("en_IN", "ta_Taml_IN", match_language=True) == {"first_language": ["Tamil"]}

    def test_hindi_corpora_apply_different_values(self):
        """Same language, two spellings -- the reason the flag alone is
        not enough to tell whether two runs match."""
        deva = persona_language_filter("hi_Deva_IN", "hi_Deva_IN", match_language=True)
        latn = persona_language_filter("hi_Latn_IN", "hi_Latn_IN", match_language=True)
        assert deva == {"first_language": ["हिंदी"]}
        assert latn == {"first_language": ["Hindi"]}
        assert deva != latn

    def test_requested_but_not_applicable_records_nothing(self):
        """The case the flag alone gets wrong: matching asked for, corpus
        has no language column, nothing narrowed."""
        assert persona_language_filter("en_US", "en_US", match_language=True) is None
        assert persona_language_filter("ja_JP", "ja_JP", match_language=True) is None

    def test_scope_field_round_trips(self):
        import dataclasses

        from usersim.engine.core.manifest import ScopeInfo, _scope_from_dict

        scope = ScopeInfo(
            locales=["ta_Taml_IN"],
            probes=[],
            probe_mix={},
            persona_language_filters={"ta_Taml_IN": {"first_language": ["Tamil"]}},
        )
        back = _scope_from_dict(dataclasses.asdict(scope))
        assert back.persona_language_filters == scope.persona_language_filters


class TestPopulationCountIsLogged:
    """The optional diagnostic scans each corpus snapshot only once."""

    @pytest.fixture(autouse=True)
    def _clear_population_cache(self):
        _column_value_counts.cache_clear()
        yield
        _column_value_counts.cache_clear()

    @staticmethod
    def _corpus(tmp_path, values, *, locale="xx_XX", column="first_language"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(
            pa.table({column: values}),
            tmp_path / f"{locale}.parquet",
        )
        return tmp_path

    def test_counts_matching_rows(self, tmp_path, caplog):
        root = self._corpus(tmp_path, ["A"] * 3 + ["B"] * 97)
        with caplog.at_level(logging.INFO):
            stats = log_persona_language_population(
                persona_locale="xx_XX",
                conversation_locale="xx_XX",
                constraint={"first_language": ["A"]},
                datasets_dir=root,
            )
        assert stats is not None
        assert stats.eligible_rows == 3
        assert stats.total_rows == 100
        assert "3 of 100 eligible synthetic persona records" in caplog.text
        assert "3.00%" in caplog.text

    def test_cache_reuses_one_scan_for_different_languages(self, tmp_path):
        root = self._corpus(tmp_path, ["A", "B", "B"])
        for language in ("A", "B"):
            log_persona_language_population(
                persona_locale="xx_XX",
                conversation_locale="xx_XX",
                constraint={"first_language": [language]},
                datasets_dir=root,
            )
        cache = _column_value_counts.cache_info()
        assert cache.misses == 1
        assert cache.hits == 1

    def test_cache_misses_after_the_corpus_changes(self, tmp_path):
        root = self._corpus(tmp_path, ["A"])
        kwargs = {
            "persona_locale": "xx_XX",
            "conversation_locale": "xx_XX",
            "constraint": {"first_language": ["A"]},
            "datasets_dir": root,
        }
        first = log_persona_language_population(**kwargs)
        self._corpus(tmp_path, ["A", "A"])
        second = log_persona_language_population(**kwargs)
        assert first is not None and first.total_rows == 1
        assert second is not None and second.total_rows == 2
        assert _column_value_counts.cache_info().misses == 2

    def test_supports_a_corpus_defined_non_india_column(self, tmp_path):
        root = self._corpus(
            tmp_path,
            ["English", "French"],
            column="primary_language",
        )
        stats = log_persona_language_population(
            persona_locale="xx_XX",
            conversation_locale="xx_XX",
            constraint={"primary_language": ["English"]},
            datasets_dir=root,
        )
        assert stats is not None
        assert stats.eligible_rows == 1

    def test_en_in_warning_explains_the_full_population_mode(
        self,
        tmp_path,
        caplog,
    ):
        root = self._corpus(
            tmp_path,
            ["English", "Hindi", "Tamil"],
            locale="en_IN",
        )
        with caplog.at_level(logging.WARNING):
            stats = log_persona_language_population(
                persona_locale="en_IN",
                conversation_locale="en_IN",
                constraint={"first_language": ["English"]},
                datasets_dir=root,
            )
        assert stats is not None and stats.eligible_rows == 1
        assert "first_language=English" in caplog.text
        assert "corpus-defined matched cohort" in caplog.text
        assert "not everyone who speaks English" in caplog.text
        assert "Omit the flag" in caplog.text
        assert "1 of 3 eligible synthetic persona records" in caplog.text

    def test_missing_corpus_is_nonfatal_and_visible(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO):
            stats = log_persona_language_population(
                persona_locale="absent_XX",
                conversation_locale="absent_XX",
                constraint={"language": ["A"]},
                datasets_dir=tmp_path,
            )
        assert stats is None
        assert "count unavailable" in caplog.text

    def test_no_constraint_does_not_read_or_log(self, tmp_path, caplog):
        with caplog.at_level(logging.INFO):
            stats = log_persona_language_population(
                persona_locale="xx_XX",
                conversation_locale="xx_XX",
                constraint=None,
                datasets_dir=tmp_path,
            )
        assert stats is None
        assert caplog.text == ""
