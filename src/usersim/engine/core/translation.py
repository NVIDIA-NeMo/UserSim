# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Machine translation across the asset/conversation language boundary
(India language-variant stop-gap).

The India language-variant locales reuse the existing India (en_IN) asset
banks. For the *verbatim* turn-1 that some probes inject (a fact question, a
parity query rendering, a safety target/action prompt), the bank text is in
English, but the conversation must be conducted in the target language. This
module machine-translates that single verbatim turn into the target language
via the cheap ``summary_model`` alias, with a strict prompt that preserves any
false/incorrect premises exactly (so a ``sov_ai_facts`` trap is not softened).

Traffic crosses that boundary in BOTH directions, so there are two helpers:

- :func:`translate_user_turn` — bank text OUT to the conversation language.
- :func:`translate_search_query` — a retrieval query IN to the corpus's
  language. Only ``financial_services`` needs this, because it is the one probe
  whose borrowed asset is a corpus the ASSISTANT reads rather than a question
  handed to the user; translating turn-1 localizes the whole asset for every
  other probe.

This is a stop-gap. It runs ONLY when a probe fell back to a base-locale asset
(i.e. no native per-language bank exists yet); once native assets land, the
probe resolves the variant's own asset and no translation happens. Translated
rows are flagged ``WarningKind.USED_MACHINE_TRANSLATION`` so downstream
scorecards treat them as preview-only.

The helper is deliberately defensive: it never raises. On any failure (LLM
error, empty output) it returns the input text unchanged, so the caller's
verbatim-injection path continues with the untranslated English turn rather
than silently switching to a different code path.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Dict

from usersim.engine.core.llm import call_llm

logger = logging.getLogger("usersim.engine")

# The cheapest configured alias; also used for early-stop / capitulation checks
# in ``core/simulation.py``, so it is always present in the models dict.
MODEL_SUMMARY = "summary_model"

# Process-local translation cache. Bank entries repeat across rows, so caching
# by (text, target_language, romanize) collapses the cost to one call per
# unique verbatim turn per language. Guarded by a lock because Data Designer
# fans rows out across worker threads.
_cache: Dict[str, str] = {}
_cache_lock = threading.Lock()


def _cache_key(text: str, target_language: str, romanize: bool) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{digest}|{target_language}|{int(romanize)}"


def _build_prompt(text: str, target_language: str, romanize: bool) -> str:
    script_line = (
        f"Write your answer in {target_language} using the Latin/Roman "
        f"alphabet (romanized {target_language}) — NOT the native script, "
        "and NOT English.\n"
        if romanize
        else f"Write your answer in {target_language}'s native script.\n"
    )
    return (
        f"You are a professional translator. Translate the USER MESSAGE below "
        f"into {target_language}.\n"
        "Rules:\n"
        "- Preserve the meaning, tone, and register exactly.\n"
        "- CRITICAL: if the message contains an incorrect, false, or mistaken "
        "assumption or premise, preserve it EXACTLY — do NOT correct, fix, "
        "soften, question, or answer it. You are translating, not responding.\n"
        "- Do not add, omit, or explain anything.\n"
        "- Output ONLY the translation: no preamble, quotes, or notes.\n"
        f"{script_line}"
        "\nUSER MESSAGE:\n"
        f"{text}"
    )


def translate_user_turn(
    models: Dict[str, Any],
    text: str,
    *,
    target_language: str,
    romanize: bool = False,
) -> str:
    """Translate a verbatim user turn into ``target_language`` (cached).

    Returns the input ``text`` unchanged on empty input or any failure —
    this function never raises, so callers can wrap a verbatim-injection
    return value without risking a control-flow change.
    """
    if not text or not text.strip():
        return text
    key = _cache_key(text, target_language, romanize)
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached

    out = text
    try:
        resp = call_llm(
            models,
            MODEL_SUMMARY,
            [{"role": "user", "content": _build_prompt(text, target_language, romanize)}],
        )
        content = resp.get("content", "") if isinstance(resp, dict) else ""
        content = (content or "").strip()
        if content:
            out = content
        else:
            logger.warning(
                "translate_user_turn: empty translation into %s; using untranslated text",
                target_language,
            )
    except Exception as e:  # noqa: BLE001 — never propagate from a translation
        logger.warning(
            "translate_user_turn: translation into %s failed (%s: %s); using untranslated text",
            target_language,
            type(e).__name__,
            e,
        )
        out = text

    with _cache_lock:
        _cache[key] = out
    return out


def _build_query_prompt(query: str, target_language: str) -> str:
    return (
        f"Translate the following knowledge-base SEARCH QUERY into "
        f"{target_language}.\n"
        "Rules:\n"
        "- This is a search query, not a message: translate the search TERMS. "
        "Do not answer it, expand it, or add words it does not contain.\n"
        "- Keep numbers, amounts, percentages, dates, product names, brand "
        "names and identifiers (snake_case tool names, ids) EXACTLY as written.\n"
        "- Use the domain's standard terminology in the target language, so the "
        "translated terms match how a document in that language would word them.\n"
        f"- Output ONLY the translated query in {target_language}: no preamble, "
        "quotes, explanation, or alternatives.\n"
        "\nSEARCH QUERY:\n"
        f"{query}"
    )


def translate_search_query(
    models: Dict[str, Any],
    query: str,
    *,
    target_language: str,
) -> str:
    """Translate a retrieval query into the CORPUS's language (cached).

    The mirror image of :func:`translate_user_turn`. That one carries a bank's
    text *out* to the conversation language; this one carries the assistant's
    ``kb_search`` query *in* to the language the borrowed corpus is written in.

    Needed because the stop-gap lends a locale someone else's corpus: the user
    speaks Tamil, so the assistant tends to search in Tamil, but the documents
    are English. Lexical ranking tokenizes on ``[a-z0-9]+``, so a native-script
    query yields ZERO tokens — and an empty token set makes the lexical fallback
    return the corpus's first k documents regardless of the query, while in
    ``hybrid`` it zeroes the lexical term and leaves pure dense, the mode
    measured at 4% precision@8 on this corpus against hybrid's 88%. A romanized
    query tokenizes but its tokens do not overlap English prose, so it ranks on
    accidental collisions instead.

    Distinct prompt from ``translate_user_turn`` on purpose: a query wants its
    terms rendered in the corpus's domain vocabulary, not its tone and register
    preserved, and it must not be answered or expanded.

    Never raises: returns ``query`` unchanged on empty input or any failure, so
    retrieval proceeds exactly as it does today rather than changing code path.
    """
    if not query or not query.strip():
        return query
    key = f"q|{_cache_key(query, target_language, False)}"
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached

    out = query
    try:
        resp = call_llm(
            models,
            MODEL_SUMMARY,
            [{"role": "user", "content": _build_query_prompt(query, target_language)}],
        )
        content = resp.get("content", "") if isinstance(resp, dict) else ""
        content = (content or "").strip()
        if content:
            out = content
        else:
            logger.warning(
                "translate_search_query: empty translation into %s; searching with the untranslated query",
                target_language,
            )
    except Exception as e:  # noqa: BLE001 — never propagate from a translation
        logger.warning(
            "translate_search_query: translation into %s failed (%s: %s); searching with the untranslated query",
            target_language,
            type(e).__name__,
            e,
        )
        out = query

    with _cache_lock:
        _cache[key] = out
    return out


def reset_translation_cache() -> None:
    """Clear the process-local translation cache (test hook)."""
    with _cache_lock:
        _cache.clear()
