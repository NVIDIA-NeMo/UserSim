# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona-language cohort resolution and user-facing diagnostics.

The sampler configuration is corpus-driven: each persona dataset declares
which field represents its matchable language and how language names are
spelled. Population counting is diagnostic only. It reads that one declared
column with PyArrow, caches its frequency table for the life of the process,
and never changes whether a run proceeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger("usersim.cli.persona_language")

DEFAULT_PERSONA_DATASETS_DIR = (
    Path.home() / ".data-designer" / "managed-assets" / "datasets"
)


def persona_language_for_locale(
    locale: str, persona_locale: Optional[str] = None,
) -> Optional[str]:
    """Return the sampled corpus's language value for a conversation locale."""
    from usersim.engine.core.locale import (
        INDIA_VARIANT_LOCALES,
        PERSONA_CORPUS_LANGUAGE,
        expected_language_display,
    )

    variant = INDIA_VARIANT_LOCALES.get(locale)
    if variant is not None:
        name = variant.language_display
    else:
        display = expected_language_display(locale)
        if not display:
            return None
        # "Hindi Devanagari"/"Hindi Latin" -> "Hindi";
        # "Indian English" -> "English".
        words = display.split()
        if len(words) == 1:
            name = words[0]
        elif words[-1] == "English":
            name = "English"
        else:
            name = words[0]

    corpus = PERSONA_CORPUS_LANGUAGE.get(persona_locale or locale, {})
    return corpus.get("language_name_remap", {}).get(name, name)


def persona_language_filter(
    persona_locale: str,
    conversation_locale: str,
    *,
    match_language: bool = False,
) -> Optional[dict[str, list[str]]]:
    """Return the corpus-defined language constraint actually applied.

    ``None`` means the population is not narrowed: matching was not requested,
    the corpus has no language field, or the conversation language could not
    be resolved. The current India assets match on ``first_language``; another
    corpus may declare a different field.

    Matching changes the demographic cohort as well as language. Cross-locale
    comparisons of matched runs therefore do not isolate language alone.
    """
    from usersim.engine.core.locale import PERSONA_CORPUS_LANGUAGE

    if not match_language:
        return None
    corpus = PERSONA_CORPUS_LANGUAGE.get(persona_locale)
    if not corpus:
        return None
    language = persona_language_for_locale(conversation_locale, persona_locale)
    if not language:
        return None
    return {corpus["language_column"]: [language]}


def persona_sampler_params(
    dd: Any,
    persona_locale: str,
    conversation_locale: str,
    *,
    match_language: bool = False,
) -> Any:
    """Build ``PersonSamplerParams`` with the corpus-defined language filter.

    Data Designer combines different filter fields with AND, so the current
    India configuration selects the declared primary-language cohort rather
    than everyone listing that language in any language field.
    """
    kwargs: dict[str, Any] = {
        "locale": persona_locale,
        "with_synthetic_personas": True,
    }
    constraint = persona_language_filter(
        persona_locale,
        conversation_locale,
        match_language=match_language,
    )
    if constraint:
        kwargs["select_field_values"] = constraint
    return dd.PersonSamplerParams(**kwargs)


@dataclass(frozen=True)
class PersonaLanguagePopulation:
    """Eligibility counts for one corpus-defined language constraint."""

    persona_locale: str
    column: str
    values: tuple[str, ...]
    eligible_rows: int
    total_rows: int

    @property
    def eligible_percent(self) -> float:
        return (
            100.0 * self.eligible_rows / self.total_rows
            if self.total_rows
            else 0.0
        )


@lru_cache(maxsize=8)
def _column_value_counts(
    parquet_path: str,
    _mtime_ns: int,
    _size_bytes: int,
    column: str,
) -> tuple[int, tuple[tuple[str, int], ...]]:
    """Read one column once per corpus snapshot and cache its frequencies."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(
        parquet_path,
        columns=[column],
        memory_map=True,
    )
    frequencies = tuple(
        sorted(
            (str(entry["values"]), int(entry["counts"]))
            for entry in pc.value_counts(table[column]).to_pylist()
            if entry["values"] is not None
        )
    )
    return table.num_rows, frequencies


def log_persona_language_population(
    *,
    persona_locale: str,
    conversation_locale: str,
    constraint: Optional[Mapping[str, Sequence[str]]],
    datasets_dir: Path = DEFAULT_PERSONA_DATASETS_DIR,
) -> Optional[PersonaLanguagePopulation]:
    """Log the eligible synthetic-persona pool for an applied constraint.

    Best-effort only: a missing or unreadable managed asset never blocks
    sampling. The cache is process-local and keyed by resolved path, mtime,
    size, and language column, so several stop-gap locales sharing ``en_IN``
    incur one PyArrow scan while a changed parquet automatically misses.
    """
    if not constraint:
        return None
    if len(constraint) != 1:
        logger.warning(
            "persona language population unavailable for locale=%s: "
            "expected one corpus language constraint, got %s",
            conversation_locale,
            dict(constraint),
        )
        return None

    column, raw_values = next(iter(constraint.items()))
    values = tuple(str(value) for value in raw_values)
    path = (Path(datasets_dir) / f"{persona_locale}.parquet").resolve()
    english_india = (
        conversation_locale == "en_IN"
        and values == ("English",)
    )

    if not path.is_file():
        logger.warning(
            "persona language cohort count unavailable for locale=%s "
            "(corpus=%s, %s=%s): %s is not present yet",
            conversation_locale,
            persona_locale,
            column,
            ", ".join(values),
            path,
        )
        if english_india:
            logger.warning(
                "en_IN with --match-persona-language selects only records "
                "whose %s is English, not the full India corpus. Omit the "
                "flag to run in English across all en_IN personas.",
                column,
            )
        return None

    try:
        stat = path.stat()
        total, frequency_items = _column_value_counts(
            str(path),
            stat.st_mtime_ns,
            stat.st_size,
            column,
        )
    except Exception as error:  # diagnostics must never fail a run
        logger.warning(
            "persona language cohort count unavailable for locale=%s "
            "(corpus=%s, %s=%s): %s; continuing without the diagnostic",
            conversation_locale,
            persona_locale,
            column,
            ", ".join(values),
            error,
        )
        return None

    frequencies = dict(frequency_items)
    eligible = sum(frequencies.get(value, 0) for value in values)
    result = PersonaLanguagePopulation(
        persona_locale=persona_locale,
        column=column,
        values=values,
        eligible_rows=eligible,
        total_rows=total,
    )

    if english_india:
        logger.warning(
            "en_IN with --match-persona-language applies %s=%s: "
            "%s of %s eligible synthetic persona records (%.2f%%). "
            "This is the corpus-defined matched cohort, not everyone who speaks "
            "English. Omit the flag to run in English across all en_IN personas.",
            column,
            ", ".join(values),
            f"{eligible:,}",
            f"{total:,}",
            result.eligible_percent,
        )
    else:
        logger.warning(
            "persona language cohort for locale=%s uses corpus=%s, %s=%s: "
            "%s of %s eligible synthetic persona records (%.2f%%). "
            "Language matching narrows correlated demographics as well as language.",
            conversation_locale,
            persona_locale,
            column,
            ", ".join(values),
            f"{eligible:,}",
            f"{total:,}",
            result.eligible_percent,
        )
    return result
