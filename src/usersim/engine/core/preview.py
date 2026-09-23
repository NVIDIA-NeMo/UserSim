# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Rich-based preview utilities for inspecting generated conversations.

The preview surface answers three questions, in order, for each
trajectory:

1. **Did this run cleanly?** ``status × warnings`` collapse into a
   tri-state badge (``SIM OK`` / ``SIM WARN`` / ``SIM FAIL``) plus a colour-coded
   header rule. ``completed_with_warnings`` rows render yellow with
   warning chips listing each :class:`WarningKind` value, distinct
   from clean ``ok`` rows.
2. **What was the model probed against?** Side-channel asset metadata
   (``action_request_id`` / ``sub_protocol`` / ``query_id`` / etc.)
   is resolved against ``conversation_metadata`` and rendered in a
   dedicated *Probe Inputs* panel — no grep-the-bank dance to figure
   out what ``probe_aadhaar_001`` means.
3. **What did the model do?** The conversation panel renders turns
   with an inline judge chip (``✓`` / ``✗``) lifted from the
   in-sim ``assistant_judge_ratings`` so you can see where things
   degraded without scrolling to a separate panel.

A small dim trajectory metadata footer at the bottom carries
``trajectory_id`` / ``probe_variant`` / model call counts /
tokens / wall-clock / bank versions for correlation back to the
parquet and the asset banks.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pandas as pd
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Flags + country names cover only the LOCALES we actually ship
# today (the keys of
# ``usersim.engine.core.behavioral.LOCALE_LANGUAGE_MAP``).
# Adding a new locale upstream means adding its entry here too —
# kept in lock-step deliberately rather than auto-derived because
# the canonical lookup is the language map and we don't want a
# circular import at module load.
_LOCALE_FLAGS = {
    "en_US": "🇺🇸",
    "pt_BR": "🇧🇷",
    "ja_JP": "🇯🇵",
    "fr_FR": "🇫🇷",
    "en_IN": "🇮🇳",
    "hi_Deva_IN": "🇮🇳",
    "hi_Latn_IN": "🇮🇳",
    "en_SG": "🇸🇬",
    "ko_KR": "🇰🇷",
}

# Render-time rewrites for older-format ``conversation_language`` strings.
# The single source of truth is
# ``usersim.engine.core.behavioral.LOCALE_LANGUAGE_MAP`` and
# is written to the ``conversation_language`` parquet column at
# *simulation time*. Whenever that map's labels are shortened (we
# went ``Hindi (Devanagari script)`` → ``Hindi (Devanagari)`` →
# ``Hindi Devanagari`` over the course of polishing this surface),
# pre-existing parquets still carry whichever form was current
# when they were generated. This map normalises every historical
# form to the latest at *render time* so the preview stays
# consistent without forcing a re-simulation after each label
# shortening. Strictly cosmetic — downstream filters / scorers /
# evaluators see the original parquet value untouched.
#
# Add an entry here whenever an existing label is shortened.
# Keep the inline lookup at the two call sites (rather than a
# helper function) so Jupyter's %autoreload partial-reload mode
# can't fail mid-render with a NameError on a missing helper.
_LANGUAGE_REWRITES = {
    "Hindi (Devanagari script)": "Hindi Devanagari",
    "Hindi (Latin script)": "Hindi Latin",
    "English (Indian English)": "Indian English",
    "Hindi (Devanagari)": "Hindi Devanagari",
    "Hindi (Latin)": "Hindi Latin",
    "Singapore English": "English",
}


# Country names used in the per-row preview header — replaces the bare
# locale code (``en_US`` / ``hi_Deva_IN``) with a human-readable
# country name. The ``conversation_language`` column on the parquet
# already carries a human-readable language string (``"English"``,
# ``"Hindi Devanagari"`` etc.) so we don't need a parallel
# language map. The summary table still uses the compact ``locale``
# code because table cells trade verbosity for column width.
_LOCALE_COUNTRIES = {
    "en_US": "USA",
    "pt_BR": "Brazil",
    "ja_JP": "Japan",
    "fr_FR": "France",
    "en_IN": "India",
    "hi_Deva_IN": "India",
    "hi_Latn_IN": "India",
    "en_SG": "Singapore",
    "ko_KR": "South Korea",
}

# India language-variants (ta_Taml_IN / ta_Latn_IN / ...) all resolve to
# India; derived from the registry rather than hand-listed. The maps
# ``.get(locale, ...)`` gracefully, so this only improves the label
# (India flag / "India") rather than fixing a crash.
from usersim.engine.core.locale import INDIA_VARIANT_LOCALES as _INDIA_VARIANTS

for _india_loc in _INDIA_VARIANTS:
    _LOCALE_FLAGS.setdefault(_india_loc, "🇮🇳")
    _LOCALE_COUNTRIES.setdefault(_india_loc, "India")

# Width budget for the summary table's ``Country`` column. Computed
# from the longest shipped country name + 2 cells for the flag
# emoji + 1 separating space — auto-widens when a new locale with a
# longer country name is added to ``_LOCALE_COUNTRIES``. Rich's
# auto-sizer underestimates flag-emoji cell width (each is 2 cells
# but Rich's measurer can report 1), so an explicit ``min_width``
# is what keeps short names like ``Brazil`` from getting squeezed
# mid-table.
_COUNTRY_COLUMN_MIN_WIDTH = max(len(c) for c in _LOCALE_COUNTRIES.values()) + 3


# ---------------------------------------------------------------------------
# Probe terminology
# ---------------------------------------------------------------------------
#
# Codebase columns are ``probe_type`` / ``probe_family`` /
# ``probe_variant``. The UI surfaces this as **"Probe"** to keep the
# vocabulary consistent across persona / probe / trajectory:
# persona is *who*, probe is *what we're testing for*, trajectory is
# *what came out*. Mild semantic overlap with the
# ``sov_ai_dynamic`` probe is acceptable — that one is
# specifically about probing the population's experience under
# persona-driven exploration; "Probe" as the meta-label means
# "the test protocol for this row".
#
# Short descriptions for each shipped probe. Surface in the
# "Probe" panel right above Persona.
_PROBE_DESCRIPTIONS = {
    "tool_calling": (
        "Tests the assistant's ability to invoke tools correctly "
        "in response to user requests, including argument shaping, "
        "schema validation, and recovery from tool-call failures."
    ),
    "general_open_ended": (
        "Tests open-ended conversational quality across diverse "
        "personas — coherence, follow-through, calibration to the "
        "user's apparent expertise."
    ),
    "general_educational": (
        "Tests general_educational / explanatory response quality — "
        "structure, accuracy, calibration to the user's stated "
        "background."
    ),
    "sov_ai_facts": (
        "Tests recall and reasoning over locale-specific (sovereign) "
        "facts drawn from a curated bank — ground-truth answers + "
        "false-premise probes."
    ),
    "sov_ai_dynamic": (
        "Tests for biased framing, condescension, or stereotyped "
        "assumptions when a persona-driven user explores topics "
        "tied to their demographic profile."
    ),
    "sov_ai_multilingual_parity": (
        "Tests response consistency across languages for the same "
        "matched-pair query — does the assistant treat speakers of "
        "low-resource languages with the same respect / depth as "
        "English speakers?"
    ),
    "safety_chat_pressure": (
        "Tests safety under multi-turn conversational pressure — "
        "does the assistant hold the line when the user persists, "
        "reframes, or escalates a target request?"
    ),
    "safety_agentic": (
        "Tests safety under agentic tool-call probes across four "
        "sub-protocols (unsanctioned-action / scope-creep / "
        "consequence-disclosure / compromised-inputs) in a "
        "mock-environment loop. Single-user-turn by design: after "
        "the initial action request, the probe observes the assistant's "
        "tool use and whether it stops to ask for clarification / "
        "confirmation before risky actions. A conversation that ends "
        "after the assistant asks what to do next is expected, not a "
        "truncated transcript."
    ),
    "financial_services": (
        "Tests financial customer-support / advisory grounded in a "
        "per-institution knowledge base, across two tiers: verifiable "
        "(param-locked tasks scored by a deterministic tool/retrieval "
        "verifier) and dynamic (open-ended questions from an authored "
        "taxonomy, scored by a reference-grounded judge on grounding, "
        "numeric faithfulness, boundary adherence, and no-fabrication). "
        "Retrieval is scoped to a single institution."
    ),
}


# Warning kinds that indicate the trajectory probed against an asset
# the bank itself flagged as ``placeholder: true`` (i.e. the asset is
# not yet reviewer-blessed for production claims). These are an
# asset-bank concern, not a simulation-quality concern — the
# *simulation* still ran cleanly, the *content* is what's tainted.
# Filtered out of:
#   - warning chips in the per-row preview header (kept noise-free
#     so simulation-health warnings stand out),
#   - status_kind determination (a row whose only warnings are
#     placeholder-kind reads as SIM OK in the preview, not WARN —
#     the upstream ``simulation_outcome.status`` is unchanged on
#     the parquet, this is a UI-layer override only).
# Downstream reporting still treats placeholder-tainted trajectories
# as preview-only via the ``placeholder_*`` counters surfaced on the
# capability dashboard, which is the correct surface for that signal.
# Panel box style. Side borders (vertical pipes) jitter for CJK /
# Devanagari content because Rich measures glyphs in fixed
# character cells but the glyphs render at variable pixel widths
# in browser monospace fonts — the right pipe ends up ragged
# row-by-row. ``box.HORIZONTALS`` keeps the top + bottom rules
# (which still cleanly delimit panels and carry the title) and
# drops the side pipes entirely, eliminating the alignment
# problem at the cost of slightly less visual containment.
_PANEL_BOX = box.HORIZONTALS


_PLACEHOLDER_WARNING_KINDS = frozenset(
    {
        "used_placeholder_fact",
        "used_placeholder_taxonomy",
        "used_placeholder_query",
        "used_placeholder_target",
        "used_placeholder_agentic_action",
        "used_placeholder_identity",
    }
)


# Tri-state status. A ``completed_with_warnings`` row is functionally
# successful but carries one or more :class:`WarningKind` flags
# (placeholder asset, id fabrication, poor assistant heuristic, etc.)
# — the badge / rule colour distinguish it from a clean ``ok`` so you
# don't conflate the two when scanning a parquet.
_STATUS_OK = "ok"
_STATUS_WARN = "warn"
_STATUS_FAIL = "fail"

_STATUS_COLOURS = {
    _STATUS_OK: "green",
    _STATUS_WARN: "yellow",
    _STATUS_FAIL: "red",
}

# OCEAN trait pill styling. Five-tier red → orange → yellow → light
# green → green gradient so any tier is instantly readable as "how
# extreme on the magnitude scale". The gradient is purely
# perceptual (low → high), NOT semantic — "low neuroticism" maps
# to red here even though low neuroticism is desirable. Reading
# the trait label (Openness vs Neuroticism) is on the user.
_OCEAN_PILL_STYLES = {
    "very_low": "bold white on red",
    "low": "bold white on dark_orange3",
    "moderate": "bold black on yellow",
    "average": "bold black on yellow",
    "high": "bold black on green_yellow",
    "very_high": "bold white on green3",
}


def _locale_label(locale: str) -> str:
    """Compact ``flag locale_code`` label — kept for backward compat
    with any external caller; no longer used by the preview itself."""
    flag = _LOCALE_FLAGS.get(locale, "🌐")
    return f"{flag} {locale}"


def _country_label(locale: str) -> str:
    """``flag Country`` label used in the summary table's Country column.

    Falls back to the bare locale code when the country lookup
    misses, so a new locale shows up as something readable rather
    than a default-globe-with-no-name.
    """
    flag = _LOCALE_FLAGS.get(locale, "🌐")
    country = _LOCALE_COUNTRIES.get(locale, locale)
    return f"{flag} {country}"


def _format_summary_age(value: Any) -> str:
    """Render a persona age cell. Returns ``"—"`` for missing values."""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or "—"
    return str(value)


def _format_summary_sex(value: Any) -> str:
    """Render a persona sex cell. Returns ``"—"`` for missing values.

    Existing parquets predating the ``persona_sex`` projection won't
    carry the column at all; pandas surfaces the missing column as
    None / NaN through ``row.get()`` and we fall back to ``"—"``.
    """
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or "—"
    return str(value)


def _country_language_label(locale: str, language: Any) -> str:
    """Human-readable ``flag Country / Language`` label for the per-row header.

    Replaces the compact ``flag locale_code`` form because in a
    full-row header there's room to be readable and the locale code
    (``hi_Deva_IN``) is harder to parse at a glance than the country
    name + the already-human-readable conversation_language string
    (``"Hindi Devanagari"``). Falls back to the locale code
    when the country lookup misses, and to ``"?"`` when language is
    missing / NaN.
    """
    flag = _LOCALE_FLAGS.get(locale, "🌐")
    country = _LOCALE_COUNTRIES.get(locale, locale)
    if isinstance(language, str) and language.strip():
        # Inline the older-format-string rewrite so this function stays
        # self-contained against Jupyter's %autoreload partial-
        # reload mode (which sometimes picks up a function body
        # change but misses newly-added module-level helpers).
        lang_str = _LANGUAGE_REWRITES.get(language, language)
    else:
        lang_str = "?"
    return f"{flag} {country} / {lang_str}"


def filter_preview_locale(
    df: pd.DataFrame,
    locale: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return preview rows matching one or more locale/language selectors.

    ``locale`` accepts either concrete locale codes (``"en_US"``,
    ``"ja_JP"``) or human-readable conversation-language labels
    (``"English"``, ``"Japanese"``, ``"Hindi Devanagari"``). The
    special selector ``"English"`` also matches English locale codes
    such as ``en_IN`` whose rendered language label is ``"Indian
    English"``.
    """
    if locale is None:
        return df
    if isinstance(locale, str):
        selectors = {locale.strip().lower()}
    else:
        selectors = {str(item).strip().lower() for item in locale if str(item).strip()}
    if not selectors:
        return df

    mask = pd.Series(False, index=df.index)
    if "locale" in df.columns:
        locale_values = df["locale"].map(
            lambda v: str(v).strip().lower() if pd.notna(v) else "",
        )
        mask = mask | locale_values.isin(selectors)
        if "english" in selectors:
            mask = mask | locale_values.str.startswith("en_")

    if "conversation_language" in df.columns:
        language_values = df["conversation_language"].map(
            lambda v: (_LANGUAGE_REWRITES.get(v, v).strip().lower() if isinstance(v, str) and v.strip() else ""),
        )
        mask = mask | language_values.isin(selectors)
        if "english" in selectors:
            mask = mask | language_values.str.contains("english", na=False)

    return df[mask]


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _format_style_pair(interaction: Any, disclosure: Any) -> str:
    """Render the ``Style`` summary cell as ``<interaction>/<disclosure>``.

    Returns ``"—"`` when both halves are missing (pandas-NaN, None, or
    empty). Failed-early trajectories that bail before behavioural
    profiling write neither field — showing ``"None/None"`` or
    ``"<NA>/<NA>"`` for them is noise that crowds out the signal in
    the rest of the column.
    """
    parts: list[str] = []
    for half in (interaction, disclosure):
        if isinstance(half, str) and half.strip():
            parts.append(half)
        else:
            parts.append("")
    if not any(parts):
        return "—"
    return "/".join(p or "?" for p in parts)


def _str_or_none(value: Any) -> str | None:
    """Return ``value`` as a non-empty string or None.

    Coerces pandas-NaN (float) and missing values to None so the
    preview's side-channel detail line doesn't show ``id=nan`` for
    ``general`` family rows in a mixed-probe DataFrame (the
    general-purpose conversational probes — tool_calling /
    general_open_ended / general_educational — never populate the
    sovereign-AI / safety side-channel columns).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    if not value.strip():
        return None
    return value


def _parse_simulation_outcome(row: pd.Series) -> dict | None:
    """Parse the ``simulation_outcome`` JSON column, defensively.

    Returns ``None`` when absent / unparseable so callers can fall
    back to the plain-bool ``conversation_status`` field without
    branching everywhere.
    """
    raw = row.get("simulation_outcome")
    if raw is None or not isinstance(raw, str):
        return None
    try:
        outcome = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return outcome if isinstance(outcome, dict) else None


def _extract_warnings(outcome: dict | None) -> list[str]:
    """Return de-duplicated, count-suffixed warning kinds.

    ``WarningKind`` values like ``poor_assistant_response`` can fire
    multiple times in one trajectory (once per offending turn); we
    collapse them to ``"poor_assistant_response×N"`` so the header
    chip stays compact.

    Placeholder-asset warnings (``used_placeholder_query`` / etc.)
    are filtered out: they're an asset-bank concern, not a
    simulation-quality concern. The simulation ran cleanly; the
    *content* it ran against is what's not yet reviewer-blessed for
    production claims. Downstream reporting bundles still gate
    production-readiness on these via dedicated counters
    (``placeholder_*_trajectories`` etc.) — that's the correct
    surface, not the per-row preview header.
    """
    if not outcome:
        return []
    raw = outcome.get("warnings", [])
    if not isinstance(raw, list):
        return []
    counts: dict[str, int] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("kind")
        if not isinstance(kind, str) or not kind:
            continue
        if kind in _PLACEHOLDER_WARNING_KINDS:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    chips: list[str] = []
    for kind, n in counts.items():
        chips.append(f"{kind}×{n}" if n > 1 else kind)
    return chips


def _has_non_placeholder_warning(outcome: dict | None) -> bool:
    """Return True when outcome carries any warning that is *not*
    asset-placeholder-related.

    Used by :func:`_status_kind` to decide whether to honour an
    upstream ``completed_with_warnings`` status. A row whose only
    warnings are placeholder kinds reads as SIM OK in the preview —
    the simulation itself was clean, the asset bank just isn't
    reviewer-blessed yet (an orthogonal track tracked by the
    reporting bundles' placeholder counters).
    """
    if not outcome:
        return False
    raw = outcome.get("warnings", [])
    if not isinstance(raw, list):
        return False
    for entry in raw:
        if isinstance(entry, dict):
            kind = entry.get("kind")
            if isinstance(kind, str) and kind not in _PLACEHOLDER_WARNING_KINDS:
                return True
    return False


def _status_kind(row: pd.Series, outcome: dict | None) -> str:
    """Resolve the tri-state status for a row.

    Prefers the explicit ``simulation_outcome.status`` field
    (``"ok"`` / ``"completed_with_warnings"`` / ``"failed"``) when
    available, else falls back to the plain ``conversation_status``
    bool plus warning detection on the outcome envelope.

    A ``completed_with_warnings`` outcome demotes to ``ok`` when its
    *only* warnings are placeholder-asset kinds (see
    ``_PLACEHOLDER_WARNING_KINDS``) — those are content-track issues,
    not simulation-track issues, and showing them as WARN in the
    preview drowns out actually-concerning simulation-quality
    warnings (``poor_assistant_response``, ``id_fabrication``,
    ``compression_fallback``).
    """
    if outcome:
        status_str = outcome.get("status")
        if status_str == "failed":
            return _STATUS_FAIL
        if status_str == "completed_with_warnings":
            return _STATUS_WARN if _has_non_placeholder_warning(outcome) else _STATUS_OK
        if status_str == "ok":
            return _STATUS_OK
    if not bool(row.get("conversation_status", False)):
        return _STATUS_FAIL
    return _STATUS_WARN if _extract_warnings(outcome) else _STATUS_OK


def _status_badge(kind: str) -> Text:
    """Render the tri-state status as a coloured pill."""
    if kind == _STATUS_FAIL:
        return Text(" SIM FAIL ", style="bold white on red")
    if kind == _STATUS_WARN:
        return Text(" SIM WARN ", style="bold black on yellow")
    return Text(" SIM OK ", style="bold white on green")


def _ocean_pill_style(label: Any) -> str:
    """Resolve the Rich style for an OCEAN trait-tier pill.

    Unknown / missing labels get plain text — we don't want to over-
    fit on a vocabulary that the persona generator might tweak.
    """
    if not isinstance(label, str):
        return ""
    return _OCEAN_PILL_STYLES.get(label.strip().lower().replace(" ", "_"), "")


def _score_color(score: int | float | None) -> str:
    if score is None:
        return "dim"
    if score >= 4:
        return "green"
    if score >= 3:
        return "yellow"
    return "red"


def _judge_chip_for_turn(turn_idx: int, ratings: list[dict] | None) -> Text | None:
    """Return the ``✓`` / ``✗`` chip for an assistant turn.

    The in-sim ``assistant_judge_ratings`` are 0-indexed by turn so
    we look up by ``turn_idx`` rather than positional order in the
    ratings list (some probes skip turns, others retry). With
    assistant resampling there can be several entries per turn (one
    per attempt); the chip reflects the FINAL attempt — the candidate
    that is actually in the transcript.
    """
    if not ratings:
        return None
    final = _final_attempt_per_turn(ratings).get(turn_idx)
    if final is None:
        return None
    chip = Text()
    if bool(final.get("success", True)):
        chip.append(" ✓", style="green")
    else:
        chip.append(" ✗", style="bold red")
    return chip


def _final_attempt_per_turn(ratings: list) -> dict:
    """Collapse attempt-indexed judge entries to the last attempt per turn.

    Entries without an ``attempt`` key (pre-resampling data) keep list
    order: the later entry for a turn wins, which for single-attempt
    data is the only entry.
    """
    final: dict = {}
    for entry in ratings:
        if not isinstance(entry, dict):
            continue
        turn = entry.get("turn_idx")
        prev = final.get(turn)
        if prev is None or entry.get("attempt", 0) >= prev.get("attempt", 0):
            final[turn] = entry
    return final


_ROLE_STYLES = {
    "system": "bold magenta",
    "user": "bold cyan",
    "assistant": "bold yellow",
    "tool": "dark_orange",
}
_ROLE_EMOJI = {"user": "👤", "assistant": "🤖", "tool": "📡"}


def _is_tool_call_envelope(msg: dict[str, Any]) -> bool:
    """True for assistant messages that only carry tool calls.

    In OpenAI-style tool use, the assistant first emits a protocol
    envelope with ``tool_calls`` and no natural-language content; the
    tool response follows; then the assistant emits the user-visible
    answer. The protocol envelope should be rendered as a tool-call
    step, not counted as a visible assistant turn.
    """
    if msg.get("role") != "assistant":
        return False
    if not msg.get("tool_calls"):
        return False
    return not (msg.get("content") or "").strip()


def _format_message_label(
    msg: dict,
    turn_label: str,
    judge_chip: Text | None = None,
) -> Text:
    """Build the left-column label cell for a conversation row.

    Renders ``[👤 user, turn #1]`` (or equivalent) in the role's
    colour, with the optional ``✓`` / ``✗`` judge chip
    appended for assistant turns.
    """
    role = msg.get("role", "?")
    style = _ROLE_STYLES.get(role, "bold white")
    emoji = _ROLE_EMOJI.get(role, "")
    label = turn_label or role
    text = Text()
    text.append(f"[{emoji} {label}]", style=style)
    if judge_chip is not None:
        text.append_text(judge_chip)
    return text


# When an assistant message exceeds ``max_content`` chars, render
# as ``<head> … <tail>`` rather than ``<head>…``. Rationale: the
# user's next turn often hinges on the assistant's *closing*
# question or invitation ("...does that help?", "...want to dig
# into option B?"), which a head-only truncation drops on the
# floor. The smart truncation budget splits roughly equally
# between head and tail (the last sentence). Applied to every
# role uniformly via ``_smart_truncate``.
_TAIL_FRACTION = 0.5
_TAIL_MIN_CHARS = 80


def _last_sentence(text: str) -> str:
    """Return the last sentence (or sentence-ish chunk) of ``text``.

    Walks backward from the end looking for a sentence terminator
    (``. ? !`` plus the Devanagari danda ``।`` and CJK full-stops
    ``。``). Returns everything *after* the terminator, stripped.
    Falls back to the whole string when no terminator is found.
    """
    s = text.rstrip()
    terminators = ".?!।。"
    # Skip any trailing terminator chars (so "Foo bar." doesn't
    # return an empty string).
    end = len(s)
    while end > 0 and s[end - 1] in terminators:
        end -= 1
    # Walk backward from there for the previous terminator.
    for i in range(end - 1, -1, -1):
        if s[i] in terminators:
            return s[i + 1 : end].strip()
    return s[:end].strip()


def _smart_truncate(content: str, max_chars: int) -> str:
    """Truncate ``content`` to ``max_chars`` preserving the last sentence.

    For short content (fits in budget): return as-is.
    For long content: return ``<head>\\n\\n…\\n\\n<tail>`` where
    ``<tail>`` is the last sentence (or the last
    ``_TAIL_MIN_CHARS`` chars if no sentence boundary is found
    late enough). The ``…`` lands on its own line surrounded by
    blank lines so the truncation is visually unmissable —
    inline ellipses get lost in long markdown / table content.
    """
    if len(content) <= max_chars:
        return content
    sep = "\n\n…\n\n"
    tail_budget = max(int(max_chars * _TAIL_FRACTION), _TAIL_MIN_CHARS)
    tail = _last_sentence(content)
    if len(tail) > tail_budget:
        # Last sentence is itself longer than the tail budget —
        # keep just the last `tail_budget` chars from the end.
        tail = "…" + content[-tail_budget:].lstrip()
    head_budget = max_chars - len(tail) - len(sep)
    if head_budget < 40:
        # Not enough room for a meaningful head with the chosen
        # tail; fall back to plain head-only truncation.
        return content[: max_chars - 1].rstrip() + "…"
    head = content[:head_budget].rstrip()
    return f"{head}{sep}{tail}"


def _format_message_content(msg: dict, max_content: int = 300) -> Text:
    """Build the right-column content cell for a conversation row.

    Renders the assistant's tool calls (if any) followed by the
    message ``content``, smart-truncated at ``max_content`` (head +
    last sentence; see ``_smart_truncate``). Tool responses also
    surface their ``tool_call_id`` so a reader can correlate to
    the matching tool_call upstream.
    """
    role = msg.get("role", "?")
    content = msg.get("content", "") or ""
    text = Text()

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            args = fn.get("arguments", "{}")
            if isinstance(args, str) and len(args) > 120:
                args = args[:120] + "..."
            if text.plain:
                text.append("\n")
            text.append(f"🔧 tool_call: {name}({args})", style="bold dark_orange")

    if content:
        display = _smart_truncate(content, max_content)
        if text.plain:
            text.append("\n")
        text.append(display)

    tool_call_id = msg.get("tool_call_id")
    if tool_call_id and role == "tool":
        text.append(f" [tool_call_id={tool_call_id}]", style="dim")

    return text


# ---------------------------------------------------------------------------
# Probe panel — what we're measuring
# ---------------------------------------------------------------------------


def _render_probe_panel(row: pd.Series) -> Panel:
    """Render the Probe description panel.

    Always renders (every row has a probe). Lives above the Persona
    panel in the per-row preview so the reader's mental flow is
    *what is being tested* → *who is being tested* → *which curated
    asset* → *what happened*. The header rule carries only the probe
    *name* for at-a-glance scanning; the variant + family +
    description live here.
    """
    probe = row.get("probe_type", "?")
    family = _str_or_none(row.get("probe_family")) or "—"
    variant = _str_or_none(row.get("probe_variant"))
    description = _PROBE_DESCRIPTIONS.get(probe, "(no description registered)")

    body = Text()
    body.append("Probe: ", style="bold")
    body.append(f"{probe}\n")
    body.append("Family: ", style="bold")
    body.append(f"{family}\n")
    if variant and variant != probe:
        body.append("Variant: ", style="bold")
        body.append(f"{variant}\n")
    body.append("\n")
    body.append(description, style="dim")

    return Panel(
        body,
        title="[bold]Probe[/bold]",
        border_style="magenta",
        box=_PANEL_BOX,
        expand=True,
    )


# ---------------------------------------------------------------------------
# Scenario inputs panel — resolves side-channel asset metadata
# ---------------------------------------------------------------------------

# Probe-specific metadata keys to surface in the Probe Inputs
# panel. Order matters — items are rendered top-to-bottom in the
# panel. Values come from ``conversation_metadata`` (set by each
# probe via ``state.metadata[...]``) and are joined read-only into
# the panel; we never re-load the asset bank from inside the preview
# (cross-module coupling we don't want).
_PROBE_INPUT_FIELDS: dict[str, list[tuple[str, str, str]]] = {
    # (column-or-meta-key, display-label, source: "row" | "meta")
    # General family: theme-driven probes (no curated asset bank).
    # The ``theme`` column carries whatever was sampled from the
    # locale's seed YAML (open_ended_topics / educational_subjects)
    # or the toolset bundle (for tool_calling). Showing it here +
    # the toolset metadata makes "what was the probe seeded with"
    # answerable at a glance for the general family too.
    "tool_calling": [
        ("theme", "Theme", "row"),
        ("toolset_name", "Toolset", "row"),
        ("toolset_category", "Toolset category", "row"),
        ("num_tools", "Tools available", "row"),
    ],
    "general_open_ended": [
        ("theme", "Topic", "row"),
    ],
    "general_educational": [
        ("theme", "Subject", "row"),
    ],
    # Inclusion / safety family: curated-asset-driven.
    "safety_agentic": [
        ("action_request_id", "Action request id", "row"),
        ("sub_protocol", "Sub-protocol", "row"),
        ("sanctioned_action_name", "Sanctioned action", "meta"),
        ("tempted_action_name", "Tempted action", "meta"),
    ],
    "safety_chat_pressure": [
        ("target_request_id", "Target request id", "row"),
        ("strategy_id", "Pressure strategy", "row"),
        ("harm_category", "Harm category", "meta"),
    ],
    "sov_ai_multilingual_parity": [
        ("query_id", "Query id", "row"),
        ("query_concern", "Concern axis", "meta"),
        ("query_domain", "Domain", "meta"),
        ("query_difficulty", "Difficulty", "meta"),
    ],
    "sov_ai_facts": [
        ("facts_probed", "Facts probed", "meta"),
        ("fact_category", "Fact category", "meta"),
        ("fact_question_type", "Question type", "meta"),
        ("fact_difficulty", "Difficulty", "meta"),
    ],
    "sov_ai_dynamic": [
        ("taxonomy_id", "Taxonomy id", "meta"),
        ("probing_categories_explored", "Categories explored", "row"),
        ("probing_subtopic_hints_used", "Sub-topic hints", "row"),
    ],
}


def _render_probe_inputs(row: pd.Series, metadata: dict | None) -> Panel | None:
    """Build the Probe Inputs panel for the row, if applicable.

    Returns ``None`` for the ``general`` family (no curated-asset
    side channel) or when none of the probe-specific fields are
    populated. The panel deliberately surfaces *what we have on the
    parquet* — full asset text (the actual concern/query/action body)
    lives in the asset YAML; we surface enough metadata
    (id + category + difficulty / sub_protocol + sanctioned action /
    etc.) to disambiguate without making the preview reach into the
    bank loaders.
    """
    probe = row.get("probe_type", "")
    fields = _PROBE_INPUT_FIELDS.get(probe)
    if not fields:
        return None

    rows: list[tuple[str, str]] = []
    for key, label, source in fields:
        if source == "row":
            value = row.get(key)
        else:
            value = metadata.get(key) if isinstance(metadata, dict) else None
        rendered = _render_field_value(value)
        if rendered is None:
            continue
        rows.append((label, rendered))

    if not rows:
        return None

    body = Text()
    for i, (label, value) in enumerate(rows):
        if i > 0:
            body.append("\n")
        body.append(f"{label}: ", style="bold")
        body.append(value)

    return Panel(
        body,
        title="[bold]Probe Inputs[/bold]",
        border_style="cyan",
        box=_PANEL_BOX,
        expand=True,
    )


def _format_dict_for_display(d: dict) -> str | None:
    """Render a dict as ``key: value · key: value`` for human reading.

    Skips empty / None / False values. Strings get whitespace-
    normalised. Nested dicts / lists get their JSON repr (rare in
    practice; the dicts we render here are flat ``{type, description,
    ...}`` shapes from the theme seed YAMLs).
    """
    parts: list[str] = []
    for k, v in d.items():
        if v in (None, "", False):
            continue
        if isinstance(v, str):
            v_str = " ".join(v.split())
        elif isinstance(v, (dict, list, tuple)):
            v_str = json.dumps(v, ensure_ascii=False, default=str)
        else:
            v_str = str(v)
        if not v_str.strip():
            continue
        parts.append(f"{k}: {v_str}")
    return " · ".join(parts) if parts else None


def _render_field_value(value: Any) -> str | None:
    """Coerce a metadata value to a renderable string.

    Skips empties / None / NaN so the panel doesn't show
    ``Sanctioned action: None`` for trajectories where the action
    request lacked a sanctioned variant.

    For string values we collapse internal whitespace runs (including
    embedded ``\\n`` linebreaks from the YAML asset source) into
    single spaces. Rationale: YAML multi-line strings carry hard
    linebreaks at the source's wrap column (typically ~80 chars),
    which Rich's panel renderer treats as fixed line breaks rather
    than soft wrap points — the result is the panel rendering text
    at ~80-char width inside a much wider panel border. Normalising
    the whitespace lets Rich reflow to the panel's actual width.

    Strings that parse as JSON dicts / lists are re-rendered as
    human-friendly ``key: value · key: value`` rather than dumped
    raw — themes for the general family (``tool_calling`` /
    ``general_open_ended`` / ``general_educational``) are stored in the parquet
    as JSON-encoded ``{type, description, ...}`` blobs and
    nobody wants to read raw JSON in a human preview.
    """
    if value is None:
        return None
    if isinstance(value, float):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return str(value)
    if isinstance(value, str):
        normalised = " ".join(value.split())
        if not normalised:
            return None
        # Try JSON: if it parses as a dict/list, render as
        # readable key:value pairs; otherwise return as-is.
        if normalised[0] in "{[":
            try:
                parsed = json.loads(normalised)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                rendered = _format_dict_for_display(parsed)
                return rendered if rendered is not None else normalised
            if isinstance(parsed, list):
                items = [_render_field_value(item) for item in parsed]
                items = [s for s in items if s]
                if items:
                    return ", ".join(items)
        return normalised
    if isinstance(value, (list, tuple)):
        items = [" ".join(str(v).split()) for v in value if v not in (None, "")]
        items = [s for s in items if s]
        return ", ".join(items) if items else None
    if isinstance(value, dict):
        return _format_dict_for_display(value)
    return str(value)


# ---------------------------------------------------------------------------
# Attempted actions table (safety_agentic)
# ---------------------------------------------------------------------------


def _render_attempted_actions_table(
    metadata: dict | None,
) -> Table | None:
    """Render the per-call attempted-actions audit for safety_agentic.

    The bare count (``attempted_actions=3`` in the header) is enough
    for triage; this table is for the next step — *which* tools, with
    *what* arguments, and were they sanctioned / tempted / fabricated
    against the bank's allow-list. Categories come from the in-sim
    classifier (``state.metadata["attempted_actions"]``).
    """
    if not isinstance(metadata, dict):
        return None
    actions = metadata.get("attempted_actions")
    if not isinstance(actions, list) or not actions:
        return None

    table = Table(
        show_header=True,
        header_style="bold",
        expand=True,
        box=box.SIMPLE,
    )
    table.add_column("#", justify="right", width=3, style="dim")
    table.add_column("Turn", justify="center", width=5)
    table.add_column("Tool", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column("In simulated tools", justify="center", width=18)
    table.add_column("Args", overflow="fold")

    for i, entry in enumerate(actions):
        if not isinstance(entry, dict):
            continue
        tool = str(entry.get("tool_name", "?"))
        turn = str(entry.get("turn_idx", "?"))
        category = str(entry.get("category") or "—")
        in_sim = entry.get("was_in_simulated_tools")
        in_sim_str = "✓" if in_sim else "✗ fabricated" if in_sim is False else "—"
        args = entry.get("tool_args", {})
        args_str = json.dumps(args, ensure_ascii=False, default=str)
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."

        # Colour the category cell by classification: sanctioned →
        # green (allowed), tempted → yellow (testable boundary),
        # fabricated → red (out-of-allowlist hallucination).
        cat_styles = {
            "sanctioned": "green",
            "tempted": "yellow",
            "fabricated": "red",
        }
        cat_style = cat_styles.get(category.lower(), "")
        table.add_row(
            str(i),
            turn,
            tool,
            Text(category, style=cat_style),
            in_sim_str,
            args_str,
        )

    return table


# ---------------------------------------------------------------------------
# Trajectory metadata footer
# ---------------------------------------------------------------------------


def _render_trajectory_footer(row: pd.Series, outcome: dict | None) -> Text | None:
    """Render the compact provenance footer for a trajectory.

    Three load-bearing correlation handles only — kept small so
    the trajectory header box stays scannable:

    - ``trajectory_id`` — full hash, no truncation, for parquet
      grep / ``df.query("trajectory_id == ...")`` round-trip.
    - ``wall_clock_s`` — at-a-glance perf signal.
    - ``code_sha`` (short) — which build produced this row.

    Other provenance fields (variant, per-model token counts,
    bank versions) used to render here but are now reserved for
    the panels below or downstream analysis on the parquet —
    they crowded the header out of legibility.

    Returns ``None`` when none of the fields are populated (an
    older-format parquet without ``simulation_outcome`` and without a
    ``trajectory_id`` column).
    """
    parts: list[str] = []

    traj_id = _str_or_none(row.get("trajectory_id"))
    if traj_id:
        parts.append(f"trajectory_id={traj_id}")

    if outcome:
        wall = outcome.get("wall_clock_s")
        if isinstance(wall, (int, float)) and wall > 0:
            parts.append(f"wall={wall:.1f}s")

        provenance = outcome.get("provenance") or {}
        if isinstance(provenance, dict):
            sha = provenance.get("code_sha")
            if isinstance(sha, str) and sha:
                parts.append(f"sha={sha[:7]}")

    if not parts:
        return None
    return Text(" · ".join(parts), style="dim")


def preview_row(
    row: pd.Series,
    console: Console | None = None,
    show_conversation: bool = True,
    show_eval: bool = True,
    show_metadata: bool = True,
    show_reasoning: bool = False,
    max_message_length: int = 300,
    max_messages: int = 20,
) -> None:
    """Render a rich preview of a single generated row.

    Output is written to the caller-supplied ``console`` so previews
    can be captured (e.g. in tests via ``Console(record=True)``) or
    redirected. When ``console`` is None, a default 150-wide console
    is created. Both ``width`` and ``height`` are passed because
    Rich 14.x silently ignores ``width`` alone (it falls back to the
    terminal size unless *both* are set).

    The row's pandas-Index value (``row.name``) is rendered as
    ``#N`` at the head of the rule — reuses the same ``#`` column
    the summary table shows so the reader can map a preview back to
    the summary row without an explicit ``position`` argument. If
    the caller wants distinct numbering they can override with
    ``df.iloc[i].rename(my_label)`` before passing.

    ``show_reasoning`` adds each assistant message's captured
    thinking trace (``reasoning_content``, present only when the
    model emits one) as a dim continuation row under the message it
    belongs to. Unlike the other ``show_*`` toggles it defaults to
    **off**: rendering them by default would bury the conversation
    the preview exists to show. Trace rows are
    annotations rather than messages — they do not count against
    ``max_messages``, so toggling this never changes how many
    messages are displayed.

    Layout (top → bottom):

    1. **Trajectory delimiter box** — top + bottom heavy ``═`` rules
       in the status colour bracket two lines of metadata:
       (a) the header (``# / STATUS / warning chips / probe /
       interaction-disclosure / turns / tools / Country/Language``)
       and (b) the dim trajectory-metadata footer (``trajectory_id``
       / ``variant`` / per-model call+token counts / wall-clock /
       bank versions / code SHA). Persona name + variant +
       family + description all live in the panels below to keep
       the box compact and CJK alignment safe.
    2. **Side-channel detail line** — IDs + sub-protocol +
       attempted-actions count for inclusion / safety probes
       (``general`` family rows omit this entirely).
    3. **Probe panel** (always renders) — what is being tested.
       Probe name, family (general / inclusion / safety), variant
       (when distinct from the probe name), and a short hardcoded
       description. The reader's mental flow is *what is being
       tested* (this panel) → *which curated asset* (Probe Inputs
       panel) → *who is being tested* (Persona panel) → *what
       happened* (Full Conversation panel). Magenta border.
    4. **Probe Inputs panel** (inclusion / safety only) — resolved
       metadata for the curated asset that drove the run. Cyan
       border, sits adjacent to the Probe panel so the two
       "what we're measuring" surfaces stay visually grouped.
    5. **Persona panel** — location / age / sex / education /
       occupation / interaction-disclosure styles, plus OCEAN
       traits with magnitude pills (low/very_low/high/very_high
       get coloured backgrounds; average stays dim). Green border —
       distinguishes "who is being tested" from the magenta /
       cyan probe surfaces above.
    6. **Full Conversation panel** — turn-by-turn render with role
       emojis and inline assistant judge chips lifted from
       ``assistant_judge_ratings``.
    7. **Attempted actions table** (``safety_agentic`` only) —
       per-call audit with category × in-allowlist colour-coding.
    8. **Trajectory judgment / Eval scores panels** (when present in
       the row).
    """
    if console is None:
        console = Console(width=150, height=50)

    outcome = _parse_simulation_outcome(row)
    metadata = _parse_json_field(row.get("conversation_metadata"))
    if not isinstance(metadata, dict):
        metadata = None

    status_kind = _status_kind(row, outcome)
    rule_colour = _STATUS_COLOURS[status_kind]

    probe = row.get("probe_type", "?")
    persona_name = row.get("persona_name", "?")
    locale = row.get("locale", "?")
    language = row.get("conversation_language", "?")

    n_turns = row.get("num_turns", 0)
    n_tools = row.get("num_tool_calls", 0)
    style_pair = _format_style_pair(
        row.get("user_interaction_style"),
        row.get("disclosure_style"),
    )

    header = Text()

    # Field order mirrors preview_summary so a reader can glance
    # between the summary row and the per-row preview without
    # re-parsing. Persona ALWAYS lands last (variable-width CJK /
    # Devanagari glyphs would push every following field out of
    # alignment line-by-line in monospace fonts).
    # Header fields joined by ``·`` separators for visual
    # consistency with the footer line below (``trajectory_id=… ·
    # wall=… · sha=…``). Each separator is a dim-grey middle dot
    # so it doesn't compete with the field content for attention.
    sep_text = Text(" · ", style="dim")

    def _sep() -> None:
        if header.plain:
            header.append_text(sep_text)

    row_idx = row.name
    # Accept Python ints, numpy ints (the default after iterrows()
    # on a DataFrame with the pyarrow backend), and meaningful
    # strings. Filter NaN / None / bools defensively — pandas
    # surfaces missing values as NaN floats and we don't want
    # ``#nan`` to leak through.
    if row_idx is not None and not isinstance(row_idx, bool):
        try:
            is_missing = bool(pd.isna(row_idx))
        except (TypeError, ValueError):
            is_missing = False
        if not is_missing:
            header.append(f"#{row_idx}", style="dim")

    _sep()
    header.append_text(_status_badge(status_kind))

    # Warning chips after the status badge so the status pill +
    # its justification read as one logical group.
    for chip in _extract_warnings(outcome):
        _sep()
        header.append(f"⚠ {chip}", style="bold yellow")

    # Probe name only — the variant + family + description live
    # in the Probe panel below, keeping the rule readable.
    _sep()
    header.append(f"{probe}", style="bold magenta")

    if style_pair != "—":
        _sep()
        header.append(f"{style_pair}", style="dim")

    _sep()
    header.append(f"turns={n_turns}", style="dim")

    if probe == "tool_calling":
        _sep()
        header.append(f"tool_calls={n_tools}", style="dim")

    _sep()
    header.append(_country_language_label(locale, language), style="dim")
    # Persona name deliberately NOT in the header — variable-width
    # CJK / Devanagari glyphs in persona names would cause the top
    # / bottom ``═`` rules to render at different lengths. The name
    # surfaces as the first field in the Persona panel below
    # instead.

    # Trajectory delimiter: a four-sided ``╔══╗`` box in the status
    # colour bracketing two centered lines — the header (status /
    # probe / style / turns / locale / language) and the dim
    # provenance footer (trajectory_id / wall-clock / code SHA).
    # The header content is pure ASCII (the persona name, with its
    # variable-width CJK glyphs, lives in the Persona panel
    # below) so side pipes don't cause alignment jitter here.
    # Centring inside the box reads as a deliberate "trajectory
    # banner" rather than a casual separator.
    console.print()
    footer = _render_trajectory_footer(row, outcome)
    if footer is not None:
        box_content = Group(Align.center(header), Align.center(footer))
    else:
        box_content = Align.center(header)
    console.print(
        Panel(
            box_content,
            box=box.DOUBLE,
            border_style=rule_colour,
            expand=True,
            padding=(0, 2),
        )
    )

    # Probe panel — what we're testing. Lives above the Probe Inputs
    # + Persona panels so the reader's mental flow is *what is being
    # tested* → *which curated asset* → *who is being tested* →
    # *what happened*. Probe Inputs (the asset metadata that
    # parameterises the probe) sits adjacent to the Probe panel so
    # the two "what we're measuring" surfaces stay visually grouped;
    # Persona (the variable side: who we're measuring against) lives
    # right above the conversation panel where the persona's
    # behavioural choices became messages.
    console.print(_render_probe_panel(row))

    # Probe Inputs panel — only for inclusion / safety probes that
    # wrote curated-asset metadata. ``general`` family (tool_calling
    # / general_open_ended / general_educational) skips this entirely (returns None).
    inputs_panel = _render_probe_inputs(row, metadata)
    if inputs_panel is not None:
        console.print(inputs_panel)

    # Persona panel
    profile = _parse_json_field(row.get("behavioral_profile"))

    # Location string: prefer the ``persona_location`` column
    # (deduped concatenation of every location-relevant persona
    # field, projected by ``cli/_pipeline.py``), falling back to
    # ``persona_city`` + ``persona_region`` for parquets that
    # don't carry the new column. Both paths filter out the
    # ``"-"`` placeholder the upstream Jinja template emits when
    # no location field is populated for the locale (a Data
    # Designer "no-empty-render" workaround — see the projection
    # block in ``cli/_pipeline.py`` for the rationale).
    raw_location = row.get("persona_location")
    if isinstance(raw_location, str) and raw_location.strip() and raw_location.strip() != "-":
        location = raw_location
    else:
        fallback_parts = [row.get("persona_city"), row.get("persona_region")]
        location = (
            ", ".join(str(p) for p in fallback_parts if isinstance(p, str) and p.strip() and p.strip() != "-") or "?"
        )

    persona_text = Text()
    # Name is the first persona field (lifted out of the trajectory
    # header rule because variable-width CJK / Devanagari name
    # glyphs would cause the heavy ``═`` rules around the header
    # to render at different widths between adjacent trajectories).
    persona_text.append("Name: ", style="bold")
    persona_text.append(f"{persona_name}\n")
    persona_text.append("Location: ", style="bold")
    persona_text.append(f"{location}\n")
    persona_text.append("Age: ", style="bold")
    persona_text.append(f"{row.get('persona_age', '?')}\n")
    # Sex line is conditional: the column is new (added after the
    # original 7-field persona projection) so existing parquets
    # don't carry it; ``_format_summary_sex`` returns ``"—"`` for
    # missing values, which would crowd the panel — skip the line
    # entirely instead.
    sex_value = _format_summary_sex(row.get("persona_sex"))
    if sex_value != "—":
        persona_text.append("Sex: ", style="bold")
        persona_text.append(f"{sex_value}\n")
    persona_text.append("Education: ", style="bold")
    persona_text.append(f"{row.get('persona_education_level', '?')}\n")
    persona_text.append("Occupation: ", style="bold")
    persona_text.append(f"{row.get('persona_occupation', '?')}\n")

    disclosure = row.get("disclosure_style", "?")
    interaction = row.get("user_interaction_style", "?")
    persona_text.append("Interaction style: ", style="bold")
    persona_text.append(f"{interaction}\n")
    persona_text.append("Disclosure: ", style="bold")
    persona_text.append(f"{disclosure}\n")

    if isinstance(profile, dict):
        ocean_labels = profile.get("ocean_labels", {})
        ocean_descs = profile.get("ocean_descriptions", {})
        ocean_scores = profile.get("ocean", {})
        if ocean_labels or ocean_scores:
            persona_text.append("\n")
            trait_names = {
                "openness": "Openness",
                "conscientiousness": "Conscientiousness",
                "extraversion": "Extraversion",
                "agreeableness": "Agreeableness",
                "neuroticism": "Neuroticism",
            }
            for key, display_name in trait_names.items():
                label = ocean_labels.get(key, "?")
                score = ocean_scores.get(key)
                desc = ocean_descs.get(key, "")
                score_str = f"{score:.2f}" if score is not None else "?"
                persona_text.append(f"{display_name}: ", style="bold")
                pill_style = _ocean_pill_style(label)
                if pill_style and pill_style != "dim":
                    # Wrap label in a pill-style span. Pad with
                    # single spaces so the background colour reads as
                    # a chip rather than a tightly-clipped word.
                    persona_text.append(f" {label} ", style=pill_style)
                else:
                    persona_text.append(label, style=pill_style or "")
                persona_text.append(f" ({score_str})")
                if desc:
                    persona_text.append(f" — {desc}", style="dim")
                persona_text.append("\n")

    console.print(
        Panel(
            persona_text,
            title="[bold]Persona[/bold]",
            # Green border distinguishes the persona panel from the
            # cyan Probe Inputs and magenta Probe panels above —
            # readers parse "who is being tested" at a glance from
            # the colour rather than scanning the title.
            border_style="green",
            box=_PANEL_BOX,
            expand=True,
        )
    )

    # Full Conversation
    if show_conversation:
        # Lift assistant_judge_ratings up here so the message
        # formatter can attach an inline ✓ / ✗ chip to each
        # assistant turn label. The standalone "Assistant Quality"
        # panel below still fires for failure summaries — the chip
        # just adds at-a-glance visibility for clean / warn turns
        # too.
        assistant_ratings = metadata.get("assistant_judge_ratings", []) if metadata else []
        if not isinstance(assistant_ratings, list):
            assistant_ratings = []

        messages = _parse_json_field(row.get("conversation_messages"))
        if isinstance(messages, list) and messages:
            visible = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")]

            # Two-column conversation table: left column is the
            # turn label (fixed width, no-wrap), right column is
            # the message content (wraps to fit). Makes turn
            # boundaries immediately scannable: every new row =
            # new message; the label column stays vertically
            # aligned even when content spans many lines.
            conv_table = Table(
                show_header=False,
                box=None,
                expand=True,
                padding=(0, 1, 0, 1),
                show_lines=True,
                pad_edge=False,
            )
            # Label column width tuned to fit the longest label
            # (e.g. ``[🤖 assistant, turn #99]`` ≈ 24 cells with
            # the emoji as 2 cells) plus 2 cells of breathing
            # room. ``no_wrap=True`` clamps to width even for
            # double-digit turn counts.
            conv_table.add_column(
                "label",
                width=27,
                no_wrap=True,
                vertical="top",
                style="bold",
            )
            conv_table.add_column("content", overflow="fold")

            # Leading blank row gives the conversation breathing
            # room from the panel's top horizontal border — easier
            # to scan than messages flush against the rule.
            conv_table.add_row("", "")

            displayed = 0
            turn_num = 0
            assistant_in_turn = 0
            for msg in visible:
                role = msg.get("role", "?")
                is_tool_call_envelope = _is_tool_call_envelope(msg)

                if role == "user":
                    turn_num += 1
                    assistant_in_turn = 0

                if is_tool_call_envelope:
                    turn_label = f"tool call, turn #{turn_num}"
                elif role in ("user", "assistant"):
                    turn_label = f"{role}, turn #{turn_num}"
                else:
                    turn_label = "api response"

                judge_chip = None
                if role == "assistant" and not is_tool_call_envelope and assistant_in_turn == 0:
                    # Only chip the FIRST assistant message in a
                    # user-visible turn. Tool-call envelopes are
                    # implementation details; the final natural-
                    # language assistant message carries the chip.
                    judge_chip = _judge_chip_for_turn(turn_num - 1, assistant_ratings)
                    assistant_in_turn += 1

                # Blank row before every message except the first
                # — gives clear visual separation between turns
                # without per-row borders. The label column is
                # empty so the spacing reads as gutter rather
                # than as a row of its own.
                if displayed > 0:
                    conv_table.add_row("", "")

                label_cell = _format_message_label(msg, turn_label, judge_chip)
                # Assistants get 1.5× the truncation budget — their
                # responses are typically the longest in the
                # conversation AND the most informative for
                # follow-up triage (the user often reacts to the
                # assistant's closing question or recommendation).
                msg_max = int(max_message_length * 1.5) if role == "assistant" else max_message_length
                content_cell = _format_message_content(msg, max_content=msg_max)
                conv_table.add_row(label_cell, content_cell)

                # Thinking trace as a continuation row under its own
                # message, so a multi-call assistant turn keeps each
                # trace next to the call that produced it. Deliberately
                # outside the ``displayed`` accounting below: this is an
                # annotation on a message already counted, not a message,
                # and letting it consume the budget would make
                # ``max_messages`` mean different things per toggle.
                if show_reasoning:
                    trace = msg.get("reasoning_content") or ""
                    if isinstance(trace, str) and trace.strip():
                        conv_table.add_row(
                            Text("↳ reasoning", style="dim italic"),
                            Text(
                                _smart_truncate(trace.strip(), msg_max),
                                style="dim italic",
                            ),
                        )

                displayed += 1
                if displayed >= max_messages:
                    conv_table.add_row(
                        "",
                        Text(
                            f"... ({len(visible) - displayed} more)",
                            style="dim",
                        ),
                    )
                    break

            # Trailing blank row mirrors the leading one — gives
            # the last message breathing room from the panel's
            # bottom border.
            conv_table.add_row("", "")

            n_user = sum(1 for m in visible if m.get("role") == "user")
            n_asst = sum(1 for m in visible if m.get("role") == "assistant" and not _is_tool_call_envelope(m))
            n_tool_calls = sum(1 for m in visible if _is_tool_call_envelope(m))
            n_tool = sum(1 for m in visible if m.get("role") == "tool")
            counts = f"{n_user} user, {n_asst} assistant"
            if n_tool_calls:
                counts += f", {n_tool_calls} tool call"
            if n_tool:
                counts += f", {n_tool} tool"
            console.print(
                Panel(
                    conv_table,
                    title=f"[bold]Full Conversation ({counts})[/bold]",
                    border_style="white",
                    box=_PANEL_BOX,
                    expand=True,
                )
            )
        elif status_kind == _STATUS_FAIL:
            reason = "unknown"
            if isinstance(metadata, dict):
                reason = metadata.get("reason", reason)
            console.print(
                Panel(
                    f"[dim italic]No conversation generated. Reason: {reason}[/dim italic]",
                    border_style="red",
                    box=_PANEL_BOX,
                )
            )

    # Per-call attempted-actions audit (safety_agentic). Lives below
    # the conversation panel so the reader has the conversational
    # context first; the table is the next "what did the model
    # actually try to invoke" answer.
    if show_metadata and probe == "safety_agentic":
        actions_table = _render_attempted_actions_table(metadata)
        if actions_table is not None:
            console.print(
                Panel(
                    actions_table,
                    title="[bold]Attempted Actions[/bold]",
                    border_style="red",
                    box=_PANEL_BOX,
                )
            )

    # Per-turn assistant quality (only shown when at least one turn's
    # FINAL attempt failed — a turn resampled and then accepted is clean
    # in the delivered transcript, so its rejected attempts don't fire
    # the panel; the full attempt-indexed audit trail stays available in
    # ``assistant_judge_ratings``).
    if show_metadata and isinstance(metadata, dict):
        asst_ratings = metadata.get("assistant_judge_ratings", [])
        final_ratings = sorted(
            _final_attempt_per_turn(asst_ratings).values(),
            key=lambda r: (r.get("turn_idx") is None, r.get("turn_idx")),
        )
        failures = [r for r in final_ratings if not r.get("success", True)]
        if failures:
            quality_text = Text()
            for r in final_ratings:
                turn = r.get("turn_idx", "?")
                rating = r.get("rating", "?")
                expl = r.get("explanation", "")
                if r.get("success", True):
                    quality_text.append(f"Turn {int(turn) + 1}: ", style="bold")
                    quality_text.append(f"{rating}\n", style="green")
                else:
                    quality_text.append(f"Turn {int(turn) + 1}: ", style="bold")
                    quality_text.append(f"{rating}", style="red")
                    if expl:
                        quality_text.append(f" — {expl}", style="dim")
                    quality_text.append("\n")
            console.print(
                Panel(
                    quality_text,
                    title="[bold]Assistant Quality (per-turn)[/bold]",
                    border_style="red",
                    box=_PANEL_BOX,
                )
            )

    # Trajectory judgment
    if show_metadata:
        traj = _parse_json_field(row.get("trajectory_judgment"))
        if isinstance(traj, dict) and traj:
            has_axis_scores = any(isinstance(v, dict) and v.get("score") is not None for v in traj.values())
            if has_axis_scores:
                traj_table = Table(show_header=True, header_style="bold", expand=True, show_lines=False)
                traj_table.add_column("Axis", style="bold")
                traj_table.add_column("Score", justify="center", width=6)
                traj_table.add_column("Reasoning")
                for axis, val in traj.items():
                    if isinstance(val, dict):
                        score = val.get("score")
                        reasoning = val.get("reasoning", "")
                        score_str = str(score) if score is not None else "—"
                        traj_table.add_row(axis, Text(score_str, style=_score_color(score)), reasoning)
                console.print(
                    Panel(traj_table, title="[bold]Trajectory Judgment[/bold]", border_style="magenta", box=_PANEL_BOX)
                )
            elif "rating" in traj:
                rating = traj.get("rating", "?")
                explanation = traj.get("explanation", "")
                rating_style = "green" if rating == "success" else "red"
                traj_text = Text()
                traj_text.append("Rating: ", style="bold")
                traj_text.append(f"{rating}\n", style=rating_style)
                if explanation:
                    traj_text.append("Explanation: ", style="bold")
                    traj_text.append(explanation)
                console.print(
                    Panel(traj_text, title="[bold]Trajectory Judgment[/bold]", border_style="magenta", box=_PANEL_BOX)
                )

    # Eval scores
    if show_eval:
        scores = _parse_json_field(row.get("eval_scores"))
        if isinstance(scores, dict) and any(v is not None for v in scores.values()):
            score_table = Table(show_header=True, header_style="bold", expand=True, show_lines=False)
            score_table.add_column("Axis", style="bold")
            score_table.add_column("Score", justify="center", width=6)
            score_table.add_column("Reasoning")
            for axis, val in scores.items():
                if val is None:
                    continue
                if isinstance(val, dict):
                    score = val.get("score")
                    reasoning = val.get("reasoning", "")
                    if isinstance(reasoning, str) and len(reasoning) > 120:
                        reasoning = reasoning[:120] + "..."
                    score_str = f"{score}/5" if score is not None else "—"
                    score_table.add_row(axis, Text(score_str, style=_score_color(score)), reasoning)
            console.print(
                Panel(score_table, title="[bold]Evaluation Scores[/bold]", border_style="green", box=_PANEL_BOX)
            )

    # Tool-calling metadata
    if show_metadata and probe == "tool_calling" and isinstance(metadata, dict):
        tools_called = metadata.get("tools_called", [])
        verifier_results = metadata.get("verifier_results", [])
        if tools_called or verifier_results:
            meta_parts = []
            if tools_called:
                meta_parts.append(f"[bold]Tools called:[/bold] {', '.join(tools_called)}")
            for vr in verifier_results:
                valid = "✓" if vr.get("valid") else "✗"
                err = f" ({vr.get('error')})" if vr.get("error") else ""
                meta_parts.append(f"  {valid} {vr.get('tool_name', '?')}{err}")
            console.print(
                Panel(
                    "\n".join(meta_parts), title="[bold]Tool Call Details[/bold]", border_style="blue", box=_PANEL_BOX
                )
            )


def preview_summary(df: pd.DataFrame, console: Console | None = None) -> None:
    """Print a summary table of all rows in the preview DataFrame.

    Same console-honouring contract as :func:`preview_row`: writes to
    the caller-supplied console when present, else falls back to a
    200-wide console (wider than 150 to fit long probe names like
    ``sov_ai_multilingual_parity`` without truncation). Both
    ``width`` and ``height`` are passed because Rich 14.x silently
    ignores ``width`` alone (it falls back to the terminal size unless
    *both* are set).

    Columns: ``# / Status / Probe / Style / Turns / Tools / Country
    / Language / Age / Sex / Persona``. The ``Probe`` column carries
    what the rest of the codebase calls ``probe_type`` — see the
    ``_PROBE_DESCRIPTIONS`` block for the rationale. ``Country`` /
    ``Language`` replace the older combined ``Locale`` column —
    breaking them out makes filtering / scanning easier and the
    flag-prefixed country name carries more signal than the bare
    locale code (``hi_Deva_IN``). ``Age`` / ``Sex`` are the
    persona-demographic distinguishers most reviewers want at a
    glance; they sit just before ``Persona``. The ``Persona``
    column lands LAST on purpose: persona
    names span Latin / Cyrillic / Devanagari / Hiragana+Kanji /
    Katakana scripts, all rendered in a browser monospace font where
    CJK / Devanagari conjuncts have non-uniform pixel widths. With
    ``Persona`` first (or middle), the row-by-row width drift
    cascades leftward / rightward through every following column,
    making boundaries jitter. With it last, the drift only affects
    what's *after* it (nothing), so all the structured columns stay
    aligned.

    The status badge is tri-state (``SIM OK`` / ``SIM WARN`` / ``SIM FAIL``) so
    a ``completed_with_warnings`` row reads visually distinct from a
    clean ``ok`` — placeholder-tainted rows surface rather than
    hiding under a green pill.

    Use :func:`filter_preview_locale` before sampling rows when you
    want to restrict only the detailed record previews by locale while
    leaving this summary table global.

    The table sizes to its content (no ``expand=True``) so it never
    over-truncates a single column to fit a fixed width — long
    probe names just make the row a bit wider.
    """
    wide = console if console is not None else Console(width=200, height=50)

    table = Table(
        title="Preview Summary",
        show_header=True,
        header_style="bold",
        # No per-row separators (``show_lines=False``): in a Jupyter
        # cell, Rich emits the table inside a ``<pre>`` element
        # rendered in a monospace font, but CJK glyphs (Japanese /
        # Hindi / Korean / Chinese persona rows) render at
        # non-uniform pixel widths that don't match Rich's
        # character-cell calculations. Row separators amplify the
        # visible misalignment between ASCII and CJK rows; omitting
        # them keeps the table readable across all locales.
        # ``box.SIMPLE_HEAVY`` keeps a clear header rule but no
        # internal vertical lines that would also shift.
        box=box.SIMPLE_HEAVY,
        # No expand=True: table sizes to content, not console width.
        # Long probe names make the row wider rather than truncated.
    )
    # Width 5 fits a 4-digit row index plus a cell of breathing
    # room — a single simulation can easily produce 3-digit row
    # counts (8 locales × 20 rows × 8 probes = 1280) and the
    # ``#`` index is the load-bearing handle for mapping a
    # per-row preview back to its summary entry.
    table.add_column("#", justify="right", width=5, style="dim")
    table.add_column("Status", justify="center", width=11, no_wrap=True)
    # "Probe" names the measurement protocol the row was simulated
    # under (tool_calling, sov_ai_facts, etc.) — see
    # ``_PROBE_DESCRIPTIONS`` for the per-probe descriptions surfaced
    # in the per-row Probe panel.
    table.add_column("Probe", no_wrap=True)
    table.add_column("Style", no_wrap=True)
    # Headers ``Turns`` / ``Tools`` are 5 chars but Rich needs an
    # extra cell of breathing room for the bold-header padding plus
    # the value's float-formatted width (``5.0`` is 3 chars). Width
    # 7 fits both header and value without the header showing as
    # ``Turn…``.
    table.add_column("Turns", justify="center", width=7)
    table.add_column("Tools", justify="center", width=7)
    # Country + Language replaced the older combined ``Locale``
    # column. Splitting them lets reviewers filter on either axis
    # and reads cleaner than ``🇮🇳 hi_Deva_IN`` for the
    # multi-script locales. ``Country`` width is explicit and
    # derived from the longest shipped country name (see
    # ``_COUNTRY_COLUMN_MIN_WIDTH``); Rich's auto-sizer
    # underestimates flag-emoji cells without it.
    table.add_column(
        "Country",
        no_wrap=True,
        min_width=_COUNTRY_COLUMN_MIN_WIDTH,
    )
    table.add_column("Language", no_wrap=True)
    # Age + Sex sit just before Persona — the demographic
    # distinguishers reviewers want at a glance for inclusion /
    # population probes.
    table.add_column("Age", justify="center", width=4)
    # Left-aligned and wide enough for the longest localized value
    # in the shipped locales — Portuguese ``masculino`` is 9 chars,
    # so 11 leaves a couple chars of breathing room. Matches the
    # default left-alignment of the surrounding string columns.
    table.add_column("Sex", width=11, no_wrap=True)
    # Persona last: variable-width Devanagari / Hiragana / Kanji /
    # Cyrillic glyphs render at non-uniform widths in browser
    # monospace fonts and would push every following column out of
    # alignment row-by-row. Trailing position contains the drift.
    table.add_column("Persona", no_wrap=True)

    pass_count = 0
    warn_count = 0
    for idx, row in df.iterrows():
        outcome = _parse_simulation_outcome(row)
        kind = _status_kind(row, outcome)
        status_text = _status_badge(kind)
        if kind == _STATUS_OK:
            pass_count += 1
        elif kind == _STATUS_WARN:
            warn_count += 1

        locale = str(row.get("locale", "?"))
        n_turns = str(row.get("num_turns", 0))
        n_tools = str(row.get("num_tool_calls", 0))

        # Style: only render when both halves are present and string.
        # Failed-early trajectories (e.g. sov_ai_facts
        # bailing on a missing prompt pack) leave these unset; show `—`
        # rather than literal "None/None" or "<NA>/<NA>".
        style = _format_style_pair(
            row.get("user_interaction_style"),
            row.get("disclosure_style"),
        )

        # Language column reuses the human-readable
        # ``conversation_language`` string written by every adapter
        # (``"English"`` / ``"Hindi Devanagari"`` etc.). Routed
        # through ``_LANGUAGE_REWRITES`` so parquets carrying an
        # older form of the label render with the current short
        # form. Inlined rather than calling a helper for the same
        # partial-reload-safety reason as ``_country_language_label``.
        raw_language = row.get("conversation_language")
        if isinstance(raw_language, str) and raw_language.strip():
            language_str = _LANGUAGE_REWRITES.get(raw_language, raw_language)
        else:
            language_str = "?"

        table.add_row(
            str(idx),
            status_text,
            str(row.get("probe_type", "?")),
            style,
            n_turns,
            n_tools,
            _country_label(locale),
            language_str,
            _format_summary_age(row.get("persona_age")),
            _format_summary_sex(row.get("persona_sex")),
            str(row.get("persona_name", "?")),
        )

    wide.print()
    wide.print(table)

    total = len(df)
    fail_count = total - pass_count - warn_count
    summary_parts = [f"[bold]{pass_count}/{total}[/bold] clean"]
    if warn_count:
        summary_parts.append(f"[yellow]{warn_count} with warnings[/yellow]")
    if fail_count:
        summary_parts.append(f"[red]{fail_count} failed[/red]")
    wide.print("\n" + " · ".join(summary_parts), style="dim")
