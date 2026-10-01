# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Language-compliance scorer — deterministic, no LLM call.

Operates on a trajectory's ``conversation_messages`` + ``locale``
columns. Computes per-trajectory rates over **assistant turns only**:

================================================ ==========================
axis                                             interpretation (1.0 best)
================================================ ==========================
``language.requested_language_match_rate``       fraction of assistant turns whose detected language matches the locale
``language.script_compliance_rate``              mean fraction of letter codepoints in expected Unicode script(s)
``language.first_turn_match``                    1.0 if the very first assistant turn matched the expected language, else 0.0
``language.script_integrity_rate``               fraction of substantial assistant turns free of glyphs from a script the locale never expects
================================================ ==========================

Not every locale gets every axis. The first three need a lingua model for the
locale's language, and several supported locales have none (Kannada, Malayalam,
Odia and Nepali are absent from lingua; Marathi is excluded deliberately because
it collides with Hindi). Those locales still get the two script axes, which need
only Unicode ranges — before that, they were skipped wholesale and reported as
"Untested" across 500 scored rows. ``detector_available`` on the envelope says
which path ran.

Why these four:

- ``requested_language_match_rate`` is the load-bearing gate — *did the
  model actually respond in the language the user wrote in?*. This is
  the metric the LLM-judge ``language_appropriateness`` axis is bad at
  (the judge is happy to score "appropriate English" on a Hindi
  trajectory because the English IS appropriate-level English; it just
  isn't Hindi). A deterministic check makes this inescapable.
- ``script_compliance_rate`` catches the specific "model speaks Hindi
  but in Latin transliteration" failure mode — `langdetect`-style
  language detection alone returns ``"HINDI"`` for both
  ``"नमस्ते"`` and ``"namaste"``, so a script-level check is the only
  way to flag the latter.
- ``first_turn_match`` is broken out separately because a wrong-language
  turn 1 often corrupts the rest of the conversation downstream (the
  user-agent loops in the wrong language too) and the aggregate metric
  hides that. CI can gate on this single value cheaply.
- ``script_integrity_rate`` asks a different question from
  ``script_compliance_rate``: not "how much of this is in the right script"
  but "is there anything in here that has no business being here at all".
  Compliance counts a legitimate Latin identifier (``UPI``, ``freeze_card``)
  as a failure, which gives it a systematic floor on locales whose corpora
  keep identifiers stable across languages. Integrity allows Latin and flags
  only glyphs from a third script — Hangul, Cyrillic or Hebrew appearing
  mid-word in Kannada prose. That is not a language choice; it is a broken
  generation, and it needs a name of its own so a reader is not told
  "script compliance 45%" and left to assume the model answered in English.

``status_proposal``: ``False`` whenever any rate that was actually computed is
below ``1.0`` (strict default). Axes the locale cannot support are absent
rather than zero, so they neither pass nor fail. The capability dashboard lays
configurable thresholds on top — strict-by-default keeps individual trajectory
inspection sharp.

Trajectories without assistant turns, without a recognizable locale, or in a
romanized locale (where both script axes are vacuous because the expected
script IS Latin and there is no detector to tell the target language from
English) short-circuit to a no-op envelope (``error`` populated) so the scorer
degrades gracefully when run against the wrong probe family.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from usersim.engine.core.language_detection import (
    detect_language_name,
    foreign_scripts_used,
    script_compliance_fraction,
    script_dominance,
)
from usersim.engine.core.locale import (
    LATIN_RANGES,
    expected_language_name,
    expected_script_ranges,
)
from usersim.engine.evaluator.scorers import register_scorer

logger = logging.getLogger("usersim.engine")


# Public so reporting and tests can enumerate the axes without re-deriving them.
LANGUAGE_AXES: tuple[str, ...] = (
    "language.requested_language_match_rate",
    "language.script_compliance_rate",
    "language.first_turn_match",
    "language.script_integrity_rate",
)

# Turns shorter than this are excluded from the script-integrity rate. A single
# stray glyph is the whole signal, so on a 3-character turn ("ok") one bad
# codepoint would sink the turn on noise. The corpus-wide intrusion figures this
# axis was calibrated against were measured with the same floor.
INTEGRITY_MIN_CHARS: int = 80


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def score_language_compliance_trajectory(
    trajectory: dict[str, Any],
    models: dict[str, Any],  # unused — deterministic scorer, no LLM call
) -> dict[str, Any]:
    """Score a trajectory's assistant turns for language + script compliance.

    Contract:

    - Reads ``locale`` and ``conversation_messages`` (JSON or list)
      off the trajectory dict.
    - Skips with a structured no-op envelope when the locale is unregistered,
      when it is romanized (both script axes would be vacuous), or when the
      trajectory carries no assistant turns.
    - Iterates over assistant turns computing the per-turn script compliance
      fraction and foreign-script intrusions, plus — only when lingua has a
      model for the locale's language — the detected language.
    - Emits the two detector-dependent axes only when a detector exists. A
      locale lingua cannot handle still gets both script axes rather than
      nothing at all.
    - Returns ``status_proposal=False`` whenever any axis it computed < 1.0.
    """
    locale = trajectory.get("locale")
    expected_lang = expected_language_name(locale) if isinstance(locale, str) else None
    expected_ranges = expected_script_ranges(locale) if isinstance(locale, str) else ()
    detector_available = expected_lang is not None
    # Latin-only expected script + no detector = nothing measurable. This is the
    # romanized case (Hindi or Telugu written in Latin), where both script axes
    # come back a meaningless 1.0 for an answer that is plain English, so
    # scoring them would manufacture a pass. Tested on the ranges rather than a
    # romanized flag on purpose: ``hi_Latn_IN`` ships as a first-class locale
    # and is absent from ``INDIA_VARIANT_LOCALES``, so a flag lookup misses it
    # and it silently scores 1.0 on both axes.
    latin_only = bool(expected_ranges) and set(expected_ranges) <= set(LATIN_RANGES)

    if not detector_available and (latin_only or not expected_ranges):
        why = (
            "expected script is Latin and no language detector exists, so the "
            "script axes cannot tell the target language from English"
            if latin_only
            else "locale not registered in core/locale.py"
        )
        return _noop(
            error=f"locale={locale!r}: {why} — language_compliance scorer skipped",
        )

    assistant_messages = _extract_assistant_messages(
        trajectory.get("conversation_messages"),
    )
    if not assistant_messages:
        return _noop(
            error=("no assistant turns in conversation_messages — language_compliance scorer skipped"),
        )

    per_turn: list[dict[str, Any]] = []
    matches = 0
    script_fractions: list[float] = []

    for idx, content in enumerate(assistant_messages):
        text = content or ""
        detected = detect_language_name(text) if detector_available else None
        match = detector_available and detected == expected_lang
        script_frac = script_compliance_fraction(text, expected_ranges)
        foreign = foreign_scripts_used(text, expected_ranges)
        if match:
            matches += 1
        script_fractions.append(script_frac)
        per_turn.append(
            {
                "turn_idx": idx,
                "n_chars": len(text),
                "detected_language": detected,
                "expected_language": expected_lang,
                "language_match": match,
                "script_compliance": round(script_frac, 4),
                # Surface dominant-script breakdown when the per-turn
                # compliance failed — gives the reviewer the actionable
                # signal of what the model used instead.
                "script_dominance": (script_dominance(text) if script_frac < 1.0 else None),
                # Names the scripts that had no business being here, so the
                # reviewer sees "hangul, cyrillic" rather than a bare rate.
                "foreign_scripts": list(foreign),
                "counts_toward_integrity": len(text.strip()) >= INTEGRITY_MIN_CHARS,
            }
        )

    n_turns = len(assistant_messages)
    script_compliance_rate = sum(script_fractions) / n_turns if script_fractions else 1.0

    scores: dict[str, Any] = {
        "language.script_compliance_rate": _score_cell(
            round(script_compliance_rate, 4),
            n_turns,
            _summarize_script_compliance(per_turn, script_compliance_rate, n_turns),
        ),
        "language.script_integrity_rate": _script_integrity_cell(per_turn),
    }
    if detector_available:
        # Both of these read a detected language, so they exist only where
        # lingua has a model. Absent beats zero: a missing axis is skipped by
        # the capability aggregation, whereas a 0.0 would read as a real failure.
        scores["language.requested_language_match_rate"] = _score_cell(
            matches / n_turns,
            n_turns,
            _summarize_match_rate(per_turn, expected_lang, matches, n_turns),
        )
        scores["language.first_turn_match"] = _score_cell(
            1.0 if per_turn[0]["language_match"] else 0.0,
            1,
            _summarize_first_turn(per_turn[0], expected_lang),
        )

    # Only judge what was measured. A ``None`` score means the locale could not
    # support that axis (or no turn was long enough), which is absence of
    # evidence rather than a failure.
    computed = [c["score"] for c in scores.values() if c["score"] is not None]
    status_proposal = all(s >= 1.0 for s in computed)

    return {
        "scorer_kind": "deterministic",
        "locale": locale,
        "expected_language": expected_lang,
        # False means lingua has no model for this locale's language, so the
        # two language-identity axes are absent from ``scores``.
        "detector_available": detector_available,
        "n_assistant_turns": n_turns,
        "scores": scores,
        "per_turn_details": per_turn,
        "status_proposal": status_proposal,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_assistant_messages(raw: Any) -> list[str]:
    """Pull the assistant turns out of ``conversation_messages`` as a
    list of content strings, in turn order. Tolerates both list and
    JSON-string forms.

    Skips two kinds of "assistant" messages that aren't natural-language
    output:

    - **Tool-call envelopes** (``tool_calls`` is set) — infrastructure
      messages where the assistant is invoking a function. Some shims
      store these with ``content=""`` rather than ``content=None``, so a
      ``content is None`` filter alone lets them through and drags the
      language-match rate down.
    - **Empty / whitespace-only content** — provides zero language
      signal and would otherwise be detected as ``None``, falsely
      flagging the trajectory as a language-mismatch failure.
    """
    messages = _normalize_conversation(raw)
    out: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        if m.get("tool_calls"):
            continue
        content = m.get("content")
        if content is None:
            continue
        text = str(content)
        if not text.strip():
            continue
        out.append(text)
    return out


def _normalize_conversation(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _summarize_match_rate(
    per_turn: list[dict[str, Any]],
    expected_lang: str,
    matches: int,
    n_turns: int,
) -> str:
    """Narrate which assistant turns failed the language check, if any."""
    if matches == n_turns:
        return f"All {n_turns} assistant turn(s) detected as {expected_lang}."
    failing = [d for d in per_turn if not d["language_match"]]
    failing_chunks = ", ".join(
        f"#{d['turn_idx'] + 1} ({d['detected_language'] or 'undetected'}, {d['n_chars']} chars)" for d in failing[:3]
    )
    suffix = "" if len(failing) <= 3 else f" + {len(failing) - 3} more"
    return (
        f"Matched on {matches}/{n_turns} assistant turn(s); expected "
        f"{expected_lang}. Failing turns: {failing_chunks}{suffix}."
    )


def _summarize_script_compliance(
    per_turn: list[dict[str, Any]],
    rate: float,
    n_turns: int,
) -> str:
    """Mean script compliance + the worst-offending turns when below 100 %.

    The per-turn ``script_dominance`` payload always carries every Unicode
    bucket (latin / devanagari / hangul / hiragana / katakana / han / other),
    most of them with value 0.0. We sort by descending share and drop zeros
    before naming the dominant scripts so the reviewer sees what the model
    *actually* used (e.g. ``latin 96%, han 4%`` for a Japanese trajectory
    that came back in English) instead of the misleading dict-insertion
    order (``latin, devanagari``).
    """
    if all(d["script_compliance"] >= 1.0 for d in per_turn):
        return f"All {n_turns} assistant turn(s) used the expected script(s)."
    below = sorted(
        (d for d in per_turn if d["script_compliance"] < 1.0),
        key=lambda d: d["script_compliance"],
    )
    chunks = []
    for d in below[:3]:
        dom = d.get("script_dominance") or {}
        dominant_items = sorted(
            ((k, v) for k, v in dom.items() if v > 0),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if dominant_items:
            dom_text = ", ".join(f"{k} {round(v * 100)}%" for k, v in dominant_items[:2])
        else:
            dom_text = "n/a"
        chunks.append(f"#{d['turn_idx'] + 1} ({d['script_compliance']:.0%}, dominant: {dom_text})")
    suffix = "" if len(below) <= 3 else f" + {len(below) - 3} more"
    return f"Mean script compliance {rate:.0%} across {n_turns} turn(s). Below threshold: {', '.join(chunks)}{suffix}."


def _script_integrity_cell(per_turn: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of substantial assistant turns carrying no foreign-script glyph.

    A turn fails on *any* foreign glyph rather than on a fraction of them. One
    stray Hangul character mid-word in Kannada prose is already the defect;
    waiting for it to reach some share of the text would only hide the milder
    cases. On the run this was calibrated against, the all-or-nothing rule
    separates cleanly — Kannada 95% of turns affected against Marathi 0.9%.

    Scores ``None`` when no turn cleared :data:`INTEGRITY_MIN_CHARS`, because a
    vacuous 1.0 would read as a clean bill of health the evidence cannot
    support.
    """
    counted = [d for d in per_turn if d["counts_toward_integrity"]]
    if not counted:
        return _score_cell(
            None,
            0,
            f"No assistant turn reached {INTEGRITY_MIN_CHARS} characters, the "
            "floor below which a single stray glyph would dominate the rate.",
        )
    dirty = [d for d in counted if d["foreign_scripts"]]
    rate = (len(counted) - len(dirty)) / len(counted)
    if not dirty:
        return _score_cell(
            1.0,
            len(counted),
            f"All {len(counted)} substantial assistant turn(s) used only the expected script plus Latin.",
        )
    chunks = ", ".join(f"#{d['turn_idx'] + 1} ({', '.join(d['foreign_scripts'])})" for d in dirty[:3])
    suffix = "" if len(dirty) <= 3 else f" + {len(dirty) - 3} more"
    intruders = sorted({s for d in dirty for s in d["foreign_scripts"]})
    return _score_cell(
        round(rate, 4),
        len(counted),
        f"{len(dirty)}/{len(counted)} substantial assistant turn(s) contain "
        f"glyphs from a script this locale never expects ({', '.join(intruders)}) "
        f"— a broken generation rather than a language choice. Turns: "
        f"{chunks}{suffix}.",
    )


def _summarize_first_turn(first: dict[str, Any], expected_lang: str) -> str:
    if first["language_match"]:
        return f"Turn 1 detected as {expected_lang} ({first['n_chars']} chars)."
    detected = first["detected_language"] or "None (empty / too-short content)"
    return f"Turn 1 detected as {detected}, expected {expected_lang} ({first['n_chars']} chars)."


def _score_cell(score: float | None, n: int, reasoning: str) -> dict[str, Any]:
    return {
        "score": score,
        "reasoning": reasoning,
        "n": n,
    }


def _noop(*, error: str) -> dict[str, Any]:
    """No-op envelope shape returned when the trajectory can't be scored."""
    return {
        "scorer_kind": "deterministic",
        "locale": None,
        "expected_language": None,
        "n_assistant_turns": 0,
        "scores": {},
        "per_turn_details": [],
        "status_proposal": True,  # vacuously — no evidence to fail on
        "error": error,
    }


# Auto-register on import so the evaluator picks this up once the
# module is listed in DEFAULT_SCORER_MODULES.
register_scorer("language_compliance", score_language_compliance_trajectory)
