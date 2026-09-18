# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin wrappers around lingua + Unicode-block helpers for language and
script detection.

Used by the deterministic out-of-sim scorers
(``evaluator/scorers/language_compliance.py`` and friends). Two design
constraints worth flagging:

- **Lazy detector construction.** The lingua detector is ~170 MB of
  compiled language models. Building it costs about a second on import
  and the memory stays resident. We defer construction until the first
  call to :func:`detect_language`, behind a lock so multi-row evaluators
  don't race. If a scorer that doesn't need it (e.g., ``response_shape``)
  runs alone, lingua never loads.
- **Graceful fallback on empty / very short / whitespace-only input.**
  Assistant turns can legitimately be empty (truncation, error path) or
  near-empty ("ok", "thanks"). All public functions return ``None`` /
  default values rather than raising. The caller decides whether an
  empty turn should count toward an aggregate or be skipped.

Scoped to the languages our supported locales need — the shipped-locale set
(English, Portuguese, French, Japanese, Hindi, Korean) plus every India
language-variant language lingua can detect (Bengali, Gujarati, Marathi,
Punjabi, Tamil, Telugu, Urdu). Restricting the detector's candidate set
materially improves accuracy on short text — lingua's docs explicitly
recommend it — but the candidate set MUST include a language for it to be
detectable at all: a language absent from the set is misclassified as the
nearest included one (this silently zeroed `language.requested_language_match_rate`
for the India variants before they were added here). Adding a new locale's
language therefore means registering it in
:mod:`usersim.engine.core.locale` (``LOCALE_TO_LANGUAGE_NAME`` or
``INDIA_VARIANT_LOCALES``), which this module unions automatically.
"""

from __future__ import annotations

import threading

from usersim.engine.core.locale import (
    INDIA_VARIANT_LOCALES,
    LATIN_RANGES,
    LOCALE_TO_LANGUAGE_NAME,
    SCRIPT_BUCKETS,
)

# ---------------------------------------------------------------------------
# Lazy detector
# ---------------------------------------------------------------------------


_DETECTOR = None
_DETECTOR_LOCK = threading.Lock()


def _get_detector():
    """Return the shared ``lingua.LanguageDetector``, building on first call.

    The detector is restricted to the union of expected languages across
    every supported locale. Restricting the candidate set materially improves accuracy
    on short conversational text.
    """
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            from lingua import Language, LanguageDetectorBuilder

            # Union the shipped-locale languages with every India
            # language-variant language lingua can detect. Some locales map to
            # ``None`` (romanized twins; native scripts lingua lacks like
            # Odia/Kannada/Malayalam/Nepali) — deliberately undetectable and
            # handled by the script check elsewhere; filter them out. A language
            # MUST be in this set to be detectable: omitting the variant
            # languages silently zeroed their requested_language_match_rate.
            language_names = sorted(
                {n for n in LOCALE_TO_LANGUAGE_NAME.values() if n}
                | {v.lingua_name for v in INDIA_VARIANT_LOCALES.values() if v.lingua_name}
            )
            languages = [getattr(Language, n) for n in language_names]
            _DETECTOR = LanguageDetectorBuilder.from_languages(*languages).build()
    return _DETECTOR


def reset_detector() -> None:
    """Drop the cached detector. For tests that need a clean state."""
    global _DETECTOR
    with _DETECTOR_LOCK:
        _DETECTOR = None


# ---------------------------------------------------------------------------
# Public detection helpers
# ---------------------------------------------------------------------------


def detect_language_name(text: str) -> str | None:
    """Return the lingua language enum NAME for the dominant language in
    ``text``, or ``None`` if the text is empty / whitespace / too
    ambiguous for the detector to decide.

    Examples (with the default locale-restricted detector):
      - ``"Hello, how are you?"``  → ``"ENGLISH"``
      - ``"Bonjour, ça va?"``      → ``"FRENCH"``
      - ``"नमस्ते"``                → ``"HINDI"``
      - ``""``                     → ``None``
      - ``"   \\n  "``               → ``None``
    """
    if not text or not text.strip():
        return None
    detector = _get_detector()
    lang = detector.detect_language_of(text)
    return lang.name if lang is not None else None


def script_compliance_fraction(
    text: str,
    expected_ranges: tuple[tuple[int, int], ...],
) -> float:
    """Return the fraction of letter codepoints in ``text`` that fall
    within any of ``expected_ranges`` (each range is a
    ``(lo_codepoint, hi_codepoint)`` inclusive pair).

    Non-letter codepoints (digits, punctuation, whitespace, symbols,
    emoji) are excluded from both numerator and denominator — they're
    script-neutral. So a Hindi response containing the digit ``5`` and
    the word ``"computer"`` written in Latin would have its compliance
    fraction reflect only the letters that picked a script.

    Returns ``1.0`` if the text contains no letters (vacuously
    compliant — the alternative would be 0.0, which would penalize
    legitimate empty / numeric / emoji responses).

    Returns ``1.0`` if ``expected_ranges`` is empty (no expected script
    declared for this locale).
    """
    if not text:
        return 1.0
    if not expected_ranges:
        return 1.0
    n_letter = 0
    n_compliant = 0
    for ch in text:
        if not ch.isalpha():
            continue
        n_letter += 1
        cp = ord(ch)
        for lo, hi in expected_ranges:
            if lo <= cp <= hi:
                n_compliant += 1
                break
    if n_letter == 0:
        return 1.0
    return n_compliant / n_letter


def foreign_scripts_used(
    text: str,
    expected_ranges: tuple[tuple[int, int], ...],
) -> tuple[str, ...]:
    """Return the names of scripts in ``text`` that the locale never expects.

    A script is foreign when it is neither the locale's own script nor Latin.
    Latin is always allowed: an in-language answer legitimately carries Latin
    product names, tool identifiers and initialisms (``UPI``, ``PAN``,
    ``freeze_card``), and the finance corpora deliberately keep those stable
    across every language.

    This is a different question from
    :func:`script_compliance_fraction`, which asks "how much of this is in
    the right script" and therefore counts those legitimate Latin
    identifiers as failures. This asks "is there anything in here that has
    no business being here at all" — Hangul, Cyrillic or Hebrew glyphs
    inside Kannada prose, which is not a language choice but a broken
    generation (``"MAB ನListener"``, ``"формаಟ್"``).

    Returns an empty tuple when the text is clean, has no letters, or the
    locale declares no expected script (nothing to be foreign to). Names come
    from :data:`~usersim.engine.core.locale.SCRIPT_BUCKETS`; anything
    outside every named bucket reports as ``"other"``.
    """
    if not text or not expected_ranges:
        return ()
    allowed = tuple(expected_ranges) + LATIN_RANGES
    found: set[str] = set()
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in allowed):
            continue
        found.add(_script_name(cp))
    return tuple(sorted(found))


def _script_name(cp: int) -> str:
    """Name the script bucket ``cp`` belongs to, or ``"other"``."""
    for name, ranges in SCRIPT_BUCKETS.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return "other"


def script_dominance(text: str) -> dict[str, float]:
    """Return per-script-bucket fractions of the letter codepoints in
    ``text``. Buckets: ``latin``, ``devanagari``, ``hiragana``,
    ``katakana``, ``han``, ``other``.

    Useful for surfacing *what the model actually used* when the
    expected-script check fails — "expected Devanagari but the response
    was 92% Latin" gives the reviewer the actionable signal that the
    aggregate compliance rate alone would not.

    Returns all-zero fractions if the text contains no letters.
    """
    counts: dict[str, int] = {k: 0 for k in list(SCRIPT_BUCKETS) + ["other"]}
    n_letter = 0
    for ch in text:
        if not ch.isalpha():
            continue
        n_letter += 1
        cp = ord(ch)
        matched = False
        for name, ranges in SCRIPT_BUCKETS.items():
            for lo, hi in ranges:
                if lo <= cp <= hi:
                    counts[name] += 1
                    matched = True
                    break
            if matched:
                break
        if not matched:
            counts["other"] += 1
    if n_letter == 0:
        return {k: 0.0 for k in counts}
    return {k: round(v / n_letter, 4) for k, v in counts.items()}
