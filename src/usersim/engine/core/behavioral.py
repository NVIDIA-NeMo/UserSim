# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral augmentation: OCEAN traits -> behavioral parameters for user agent prompts.

Includes Sim2Real gap mitigations inspired by Zhou et al. (2026),
"Mind the Sim2Real Gap in User Simulation for Agentic Tasks":
- Disclosure style: incremental vs upfront information sharing
- User interaction style: OCEAN-grounded frustration/cooperation spectrum
- Reactive frustration prompts for mid-conversation injection
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

_OCEAN_TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

LOCALE_LANGUAGE_MAP = {
    "en_US": "English",
    "pt_BR": "Portuguese",
    "ja_JP": "Japanese",
    "fr_FR": "French",
    "en_IN": "Indian English",
    "hi_Deva_IN": "Hindi Devanagari",
    "hi_Latn_IN": "Hindi Latin",
    "en_SG": "English",
    "ko_KR": "Korean",
}


# ---------------------------------------------------------------------------
# Education ordinal mapping per locale
# Maps exact education_level values to [0, 1] (0 = lowest, 1 = highest).
# Values sourced from actual Nemotron-Personas datasets.
# ---------------------------------------------------------------------------

_EDUCATION_ORDINAL: dict[str, dict[str, float]] = {
    "en_US": {
        "less_than_9th": 0.0,
        "9th_12th_no_diploma": 0.15,
        "high_school": 0.3,
        "some_college": 0.45,
        "associates": 0.5,
        "bachelors": 0.7,
        "graduate": 1.0,
    },
    "pt_BR": {
        "Sem instrução e fundamental incompleto": 0.0,
        "Fundamental completo e médio incompleto": 0.3,
        "Médio completo e superior incompleto": 0.5,
        "Superior completo": 1.0,
    },
    "ja_JP": {
        "中学卒": 0.15,
        "高校卒": 0.3,
        "短大卒": 0.5,
        "高専卒": 0.55,
        "大学卒 文系": 0.7,
        "大学卒 理系": 0.75,
        "大学院卒 文系": 0.9,
        "大学院卒 理系": 1.0,
    },
    "en_IN": {
        "Illiterate": 0.0,
        "Literate without education level": 0.05,
        "Below Primary": 0.1,
        "Primary": 0.2,
        "Middle": 0.3,
        "Matric/Secondary": 0.4,
        "Higher Secondary/Intermediate Pre-University/Senior Secondary": 0.5,
        "Technical diploma or certificate not equal to degree": 0.6,
        "Graduate & above": 1.0,
    },
    "fr_FR": {
        "Sans diplôme": 0.0,
        "CEP": 0.1,
        "BEPC, brevet des collèges, DNB": 0.2,
        "CAP, BEP": 0.35,
        "Baccalauréat": 0.5,
        "BTS, DUT, DEUG": 0.6,
        "Licence": 0.7,
        "Master": 0.85,
        "Doctorat": 1.0,
    },
}
_EDUCATION_ORDINAL["hi_Deva_IN"] = _EDUCATION_ORDINAL["en_IN"]
# ``hi_Latn_IN`` cannot alias ``en_IN``: the India persona dataset, when
# rendered in Latin/romanized form, romanizes the education_level strings
# too ("Ashikshit" rather than "Illiterate"), so the en_IN English keys
# never match. Map the verified romanized values onto the same India scale.
_EDUCATION_ORDINAL["hi_Latn_IN"] = {
    "Ashikshit": 0.0,
    "Shiksha Star Ke Bina Sakshar": 0.05,
    "Prathamik Se Neeche": 0.1,
    "Prathamik": 0.2,
    "Middle School": 0.3,
    "Matric/Secondary": 0.4,
    "Uchchtar Madhyamik/Intermediate Pre-University/Senior Madhyamik": 0.5,
    "Takneeki Diploma Ya Pramanpatra Jo Degree Ke Barabar Nahi Hai": 0.6,
    "Gair-Takniki Diploma Ya Pramanpatra Jo Degree Ke Barabar Nahin Hai": 0.6,
    "Snatak Aur Usse Upar": 1.0,
}
# Note: ``en_SG`` (Singapore) deliberately not aliased here.
# Singapore uses its own education vocabulary (PSLE / O-Level /
# A-Level / Polytechnic diploma / etc.) which doesn't map cleanly
# to the US scheme — letting it fall through to the keyword
# fallback (which catches Bachelor / Master / PhD reliably) is
# safer than baking in a wrong cross-system mapping. Add an
# explicit ``_EDUCATION_ORDINAL["en_SG"] = {...}`` block when the
# Singapore-specific vocabulary lands.

# Keyword fallback for locales not yet mapped or test fixtures.
_ED_HIGH_KEYWORDS = ["master", "doctor", "phd", "bachelor", "graduate", "university"]
_ED_LOW_KEYWORDS = ["illiterate", "no_formal", "less_than", "below_primary"]


def _education_ordinal(education_level: str, locale: str = "en_US") -> float:
    """Map an education_level string to a [0, 1] ordinal score.

    Uses exact per-locale lookup first, then keyword fallback, then 0.5.
    """
    locale_map = _EDUCATION_ORDINAL.get(locale, {})
    if education_level in locale_map:
        return locale_map[education_level]

    ed_lower = education_level.lower()
    if any(kw in ed_lower for kw in _ED_HIGH_KEYWORDS):
        return 0.8
    if any(kw in ed_lower for kw in _ED_LOW_KEYWORDS):
        return 0.1
    return 0.5


# ---------------------------------------------------------------------------
# Tech-savvy occupation matching (en_US exact set + keyword fallback)
# ---------------------------------------------------------------------------

_TECH_OCCUPATIONS_EXACT = {
    "software_developer",
    "software_quality_assurance_analyst_or_tester",
    "computer_programmer",
    "computer_or_information_research_scientist",
    "computer_hardware_engineer",
    "computer_or_information_systems_manager",
    "computer_systems_analyst",
    "computer_support_specialist",
    "computer_network_architect",
    "computer_occupation",
    "web_developer",
    "web_or_digital_interface_designer",
    "database_administrator_or_architect",
    "network_or_computer_systems_administrator",
    "information_security_analyst",
}

_TECH_KEYWORDS = {"software", "computer", "programmer", "developer", "database"}


def _is_tech_occupation(occupation: str) -> bool:
    """Check whether an occupation indicates digital tech literacy.

    Uses exact match against curated set, then word-boundary keyword
    fallback for test fixtures and non-standard formats.
    """
    occ_lower = occupation.lower().strip()
    if occ_lower in _TECH_OCCUPATIONS_EXACT:
        return True
    words = set(re.split(r"[_\s,/]+", occ_lower))
    return bool(words & _TECH_KEYWORDS)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _parse_ocean_dict(d: dict) -> float:
    """Convert an OCEAN trait dict with t_score to [0, 1]."""
    t_score = d.get("t_score", 50)
    return _clamp((float(t_score) - 20) / 60)


def _parse_ocean_value(raw: Any) -> float:
    """Parse a single OCEAN trait value to [0, 1].

    DD's persona sampler returns OCEAN traits as dicts:
        {"description": "...", "label": "high", "t_score": 57}
    where t_score is a T-score (mean=50, sd=10, range ~20-80).
    We normalize to [0, 1] using: (t_score - 20) / 60.

    For non-en_US locales the same dicts arrive as JSON strings
    due to Arrow type differences in the managed datasets.

    Also handles plain floats/ints for testing convenience.
    """
    if isinstance(raw, dict):
        return _parse_ocean_dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return _parse_ocean_dict(parsed)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if isinstance(raw, (int, float)):
        return _clamp(float(raw))
    try:
        return _clamp(float(raw))
    except (ValueError, TypeError):
        return 0.5


def _extract_ocean(persona: dict[str, Any]) -> dict[str, float]:
    """Extract OCEAN trait scores from a persona dict.

    DD's persona sampler provides OCEAN traits as dicts with t_scores.
    Returns normalized scores in [0, 1] range.
    """
    return {t: _parse_ocean_value(persona.get(t, 0.5)) for t in _OCEAN_TRAITS}


def _extract_ocean_metadata(persona: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Extract OCEAN labels and descriptions from persona dict.

    Returns:
        (labels, descriptions) — labels like "high"/"average", descriptions
        like "Sociable, outgoing, and energetic...".
    Handles both dict values (en_US) and JSON-string values (other locales).
    """
    labels: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    for trait in _OCEAN_TRAITS:
        raw = persona.get(trait)
        d: dict | None = None
        if isinstance(raw, dict):
            d = raw
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    d = parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if d is not None:
            labels[trait] = d.get("label", "?")
            descriptions[trait] = d.get("description", "")
        else:
            labels[trait] = "?"
            descriptions[trait] = ""
    return labels, descriptions


def _infer_tech_literacy(persona: dict[str, Any], locale: str = "en_US") -> float:
    """Infer tech literacy from education, occupation, and age.

    Returns a score in [0, 1] where 1 = highly tech literate.
    Uses per-locale education ordinal mapping and curated occupation sets
    to avoid false positives from naive substring matching.
    """
    score = 0.5

    education = str(persona.get("education_level", ""))
    ed_ord = _education_ordinal(education, locale)
    if ed_ord >= 0.7:
        score += 0.15
    elif ed_ord <= 0.15:
        score -= 0.2

    occupation = str(persona.get("occupation", ""))
    if _is_tech_occupation(occupation):
        score += 0.25

    age = persona.get("age")
    if age is not None:
        age = int(age)
        if age > 75:
            score -= 0.25
        elif age > 60:
            score -= 0.15
        elif age < 30:
            score += 0.1

    return _clamp(score)


def compute_behavioral_profile(
    persona: dict[str, Any],
    locale: str = "en_US",
) -> dict[str, Any]:
    """Derive behavioral parameters from persona attributes.

    Returns a BehavioralProfile dict with parameters that control
    the user agent's simulated behavior.
    """
    ocean = _extract_ocean(persona)

    patience = _clamp(0.5 + 0.3 * ocean["agreeableness"] - 0.3 * ocean["neuroticism"])
    verbosity = _clamp(0.3 + 0.35 * ocean["extraversion"] + 0.2 * ocean["openness"])
    cooperativeness = _clamp(0.3 + 0.5 * ocean["agreeableness"] + 0.1 * ocean["conscientiousness"])
    tech_literacy = _infer_tech_literacy(persona, locale)

    education = str(persona.get("education_level", ""))
    ed_ord = _education_ordinal(education, locale)
    error_proneness = 0.3
    if ed_ord <= 0.15:
        error_proneness += 0.3
    if tech_literacy < 0.3:
        error_proneness += 0.2
    error_proneness = _clamp(error_proneness)

    ocean_labels, ocean_descriptions = _extract_ocean_metadata(persona)

    return {
        "patience": round(patience, 2),
        "verbosity": round(verbosity, 2),
        "cooperativeness": round(cooperativeness, 2),
        "tech_literacy": round(tech_literacy, 2),
        "error_proneness": round(error_proneness, 2),
        "ocean": {k: round(v, 2) for k, v in ocean.items()},
        "ocean_labels": ocean_labels,
        "ocean_descriptions": ocean_descriptions,
    }


_CHAT_REALISM_STYLE_EN = (
    "- Write like you talk: use contractions (don't, I'm, can't), fragments "
    "('Makes sense', 'Got it'), and hedging ('I think', 'maybe', 'not sure').\n"
    "- React first, then ask: 'Oh interesting — but what about X?' or "
    "'Hmm ok, so does that mean Y?'\n"
)

_CHAT_REALISM_STYLE_NON_EN = (
    "- Write naturally in your language. Use casual, everyday phrasing "
    "rather than formal or literary style.\n"
    "- Do NOT use any formatting: no bullet points, no numbered lists, "
    "no bold, no headers. Plain sentences only.\n"
    "- When ending the conversation, close briefly as a person who "
    "received help — do not offer ongoing assistance.\n"
)


def _chat_realism_style(language: str) -> str:
    if language == "English":
        return _CHAT_REALISM_STYLE_EN
    return _CHAT_REALISM_STYLE_NON_EN


def _build_chat_realism(language: str, probe_type: str, *, pressing: bool = False) -> str:
    style = _chat_realism_style(language)
    if pressing:
        return (
            "<CHAT_REALISM>\n"
            "CRITICAL — You are typing in a chat window, not writing a document:\n"
            "- Use plain text ONLY. No markdown: no headers (#), no bold (**), "
            "no bullet lists, no numbered lists, no tables.\n"
            "- Keep each reply SHORT — 1 to 3 sentences is typical for a chat message.\n"
            "- Focus on ONE point per message.\n"
            + style
            + "- You believe something about the assistant and you are pressing the point: "
            "say what you believe, back it up the way this person would, and push back "
            "when it disagrees.\n"
            "- Never echo or restate the assistant's reply at length.\n"
            "- Never offer to help the assistant.\n"
            "- Never mention that you are simulated, or how these messages are produced.\n"
            "</CHAT_REALISM>"
        )
    if probe_type == "tool_calling":
        return (
            "<CHAT_REALISM>\n"
            "CRITICAL — You are typing in a chat window, not writing a document:\n"
            "- Use plain text ONLY. No markdown: no headers (#), no bold (**), "
            "no bullet lists, no numbered lists, no tables.\n"
            "- Keep each reply SHORT — 1 to 4 sentences is typical for a chat message.\n"
            "- Focus on ONE thing per message: one question, one reaction, one concern.\n"
            + style
            + "- You are the person ASKING for help. You do NOT have the answers.\n"
            "- When the assistant asks for details, provide them naturally. "
            "It is OK to share IDs, reference numbers, short lists of names, "
            "dates, or other structured data when the workflow requires it.\n"
            "- Never echo, restate, or restructure what the assistant said. "
            "If you understood, say so briefly and move on.\n"
            "- Never offer to help the assistant — you are the one receiving help, "
            "not giving it.\n"
            "- Never refer to the conversation itself, your role in it, or how "
            "messages are being written.\n"
            "</CHAT_REALISM>"
        )
    return (
        "<CHAT_REALISM>\n"
        "CRITICAL — You are typing in a chat window, not writing a document:\n"
        "- Use plain text ONLY. Absolutely NO markdown: no headers (#), no bold (**), "
        "no tables, no horizontal rules (---), no bullet lists with dashes.\n"
        "- Keep each reply SHORT — 1 to 4 sentences is typical for a chat message.\n"
        "- Focus on ONE thing per message: one question, one reaction, one concern.\n"
        + style
        + "- You are the person ASKING for help. You do NOT have the answers. "
        "Never provide information, data, analysis, or explanations — that is "
        "the assistant's job.\n"
        "- When the assistant asks for details (names, dates, numbers, preferences), "
        "provide ONLY what was asked and then stop. Do NOT continue to do the task "
        "yourself — let the assistant use your input to produce the result.\n"
        "- Never echo, restate, or restructure what the assistant said. "
        "If you understood, say so briefly and move on.\n"
        "- Do NOT number your points or create organized lists. Real people "
        "in chat do not structure replies that way.\n"
        "- Never offer to help the assistant — you are the one receiving help, "
        "not giving it.\n"
        "- Never refer to the conversation itself, your role in it, or how "
        "messages are being written.\n"
        "</CHAT_REALISM>"
    )


USER_QUERY_INSTRUCTION = (
    "You are chatting with an AI assistant right now. "
    "Ask your question or make your request — a short, natural message "
    "in plain text, no formatting. Stay in character as the user."
)

USER_QUERY_INSTRUCTION_GROUNDED = (
    "You are chatting with an AI assistant right now. "
    "Ask your question or make your request — a short, natural message "
    "in plain text, no formatting. "
    "This is your first conversation with this assistant; it knows "
    "nothing about you yet. Your question should reflect who you are — "
    "make it a question only YOU would ask, not something anyone could "
    "ask. Do not vaguely reference personal attributes without "
    "specifying them. "
    "Stay in character as the user."
)

ROLE_ANCHOR_PROMPT = (
    "[Stay in character as the USER. Read the assistant's last response "
    "before replying. Does it actually answer your question? "
    "If the response does not make sense or does not address what you "
    "asked, react to that naturally — do not ignore it and move on to "
    "something else. "
    "Otherwise, write a short, natural chat reply: a follow-up question, "
    "a reaction, a clarification, or — if you feel your question has been "
    "answered — a brief closing remark. "
    "Do NOT provide information, analysis, tables, or formatted content. "
    "You are asking for help, not giving it.]"
)

ROLE_ANCHOR_PROMPT_WRAPUP = (
    "[Stay in character as the USER. Read the assistant's last response "
    "before replying. Does it actually answer your question? "
    "If the response does not make sense or does not address what you "
    "asked, react to that naturally — do not ignore it and move on to "
    "something else. "
    "The conversation has been going on for a while. If your original "
    "question has been answered, it is perfectly natural to thank the "
    "assistant and wrap up. You do NOT need to ask another question. "
    "Close briefly as a person who just received help. Do NOT sound "
    "like a service agent offering ongoing assistance — YOU are the "
    "one who needed help, not the other way around. "
    "Do NOT provide information, analysis, tables, or formatted content.]"
)


def format_behavioral_profile_for_prompt(
    profile: dict[str, Any],
    probe_type: str = "",
    language: str = "English",
    *,
    pressing: bool = False,
) -> str:
    """Format a BehavioralProfile dict into a prompt injection block.

    Combines OCEAN personality descriptions (rich, grounded in population
    data) with derived tech-literacy and error-proneness traits that are
    NOT directly captured by OCEAN.

    Patience, verbosity, and cooperativeness are intentionally omitted
    from the prompt -- they are already richly expressed by the OCEAN
    descriptions (extraversion -> verbosity, agreeableness -> cooperativeness,
    agreeableness + neuroticism -> patience).

    For tool_calling, uses relaxed content rules that allow the
    user to share structured data (IDs, lists, reference numbers) when the
    workflow requires it. ``pressing`` swaps in rules for a user who presses
    a point rather than asks for help, since the default rules say the user
    has no answers and never provides information.
    """
    lines: list[str] = []

    ocean_descs = profile.get("ocean_descriptions", {})
    if any(ocean_descs.values()):
        lines.append("Personality profile:")
        for trait in _OCEAN_TRAITS:
            desc = ocean_descs.get(trait, "")
            if desc:
                lines.append(f"- {desc}")

    tech_desc = (
        "tech-savvy"
        if profile["tech_literacy"] > 0.7
        else "not very tech literate"
        if profile["tech_literacy"] < 0.4
        else "moderately tech literate"
    )
    error_desc = (
        "prone to typos and ambiguous phrasing"
        if profile["error_proneness"] > 0.6
        else "generally clear in communication"
    )

    lines.append("")
    lines.append("Additional traits:")
    lines.append(f"- {tech_desc}")
    lines.append(f"- {error_desc}")

    lines.append("")
    lines.append(_build_chat_realism(language, probe_type, pressing=pressing))

    return "\n".join(lines)


def get_conversation_language(locale: str) -> str:
    """Get the conversation language for a locale.

    Falls through to the India language-variant table (``core/locale.py``)
    for variant locales (e.g. ``ta_Taml_IN`` -> ``"Tamil"``) before
    defaulting to English, so a registered variant never silently
    generates English.
    """
    if locale in LOCALE_LANGUAGE_MAP:
        return LOCALE_LANGUAGE_MAP[locale]
    from usersim.engine.core.locale import india_variant

    variant = india_variant(locale)
    if variant is not None:
        return variant.language_display
    return "English"


# ---------------------------------------------------------------------------
# Sim2Real: Disclosure style (Option B)
# ---------------------------------------------------------------------------

#: This style is about PACING -- don't dump everything into the opening message. It is
#: not about refusing to answer. The last bullet used to read "It is OK to say 'I'm not
#: sure' or 'let me check' when appropriate", which contradicted the bullet above it
#: ("reveal ... only when ASKED", i.e. you do answer when asked) and handed the
#: user-simulator a blanket excuse to stall on any direct question. Someone who
#: contacted support is there for help and knows their own details.
#:
#: It cost real signal: on the financial_services probe, `incremental` trajectories
#: scored 0.424 on tool selection against 0.667 for `upfront` (n=24/22), because the
#: user withheld the inputs a task needed and the assistant could never act. Narrowed
#: to genuine ignorance -- things a customer plausibly would not know offhand -- so the
#: pacing intent survives without the escape hatch.
DISCLOSURE_INSTRUCTIONS = {
    "incremental": (
        "<INFORMATION_DISCLOSURE>\n"
        "IMPORTANT — Realistic information sharing:\n"
        "- Do NOT provide all relevant details in your first message.\n"
        "- State your general need first, without a comprehensive brief.\n"
        "- Reveal specific details (names, IDs, numbers, dates) only when "
        "asked or when it becomes naturally relevant.\n"
        "- If you have identifying information, wait for the assistant to ask.\n"
        "- Keep your initial message short and conversational.\n"
        "- When the assistant DOES ask for something you would know — your own "
        "details, or those of someone you brought up yourself — answer it. Only "
        "say you are unsure about things you plausibly would not know offhand "
        "(an exact transaction date, a reference number, a precise balance).\n"
        "</INFORMATION_DISCLOSURE>"
    ),
    "upfront": "",
}


def compute_disclosure_style(
    persona_seed: int | None = None,
    incremental_ratio: float = 0.6,
) -> str:
    """Determine disclosure style for this conversation.

    Args:
        persona_seed: Optional seed for reproducibility (derived from persona hash).
        incremental_ratio: Fraction of conversations using incremental disclosure.
    """
    rng = random.Random(persona_seed)
    return "incremental" if rng.random() < incremental_ratio else "upfront"


def format_disclosure_instructions(style: str) -> str:
    """Return the prompt block for the given disclosure style."""
    return DISCLOSURE_INSTRUCTIONS.get(style, "")


# ---------------------------------------------------------------------------
# Sim2Real: User interaction style (Option C + G)
# ---------------------------------------------------------------------------

INTERACTION_STYLES = {
    "confrontational": {
        "base_frustration": 2,
        "escalation_rate": 2.0,
        "description": (
            "You have a confrontational personality. You are direct, blunt, "
            "and do not tolerate inefficiency. You speak your mind plainly "
            "and do not sugarcoat your reactions."
        ),
    },
    "frustrated": {
        "base_frustration": 1,
        "escalation_rate": 1.5,
        "description": (
            "You are easily frustrated and have little patience. "
            "When things feel slow or unclear, you become terse and "
            "let your annoyance show through your tone."
        ),
    },
    "impatient": {
        "base_frustration": 0,
        "escalation_rate": 1.5,
        "description": (
            "You are impatient and value your time. You give short, "
            "direct replies — a sentence or two at most. You prefer "
            "concise answers and do not want lengthy explanations."
        ),
    },
    "neutral": {
        "base_frustration": 0,
        "escalation_rate": 1.0,
        "description": "",
    },
    "cooperative": {
        "base_frustration": 0,
        "escalation_rate": 0.3,
        "description": (
            "You are patient and cooperative. You give the assistant the "
            "benefit of the doubt and respond politely even when things "
            "take longer than expected."
        ),
    },
}

FRUSTRATION_PROMPTS = {
    1: (
        "[Your tone is becoming mildly impatient. Keep your reply short "
        "and to the point. You prefer brevity — do not elaborate more "
        "than necessary.]"
    ),
    2: (
        "[You are noticeably frustrated. Show it through tone, not claims:\n"
        "- Very short sentences, no pleasantries or padding\n"
        "- You want a direct answer, not a long explanation\n"
        "- Your patience is wearing thin — let that come through naturally\n"
        "- Do NOT fabricate claims about what you previously said]"
    ),
    3: (
        "[You are strongly displeased and want this wrapped up:\n"
        "- Be curt and blunt — one or two sentences at most\n"
        "- Make it clear you are done with lengthy back-and-forth\n"
        "- Your tone is cold and clipped\n"
        "- Do NOT fabricate claims about prior exchanges or make threats]"
    ),
}


def compute_user_interaction_style(profile: dict[str, Any]) -> str:
    """Derive user interaction style from OCEAN-grounded behavioral profile.

    Maps personality traits to one of five interaction styles that control
    how the simulated user reacts to assistant behavior during conversation.
    """
    ocean = profile.get("ocean", {})
    neuroticism = ocean.get("neuroticism", 0.5)
    agreeableness = ocean.get("agreeableness", 0.5)
    patience = profile.get("patience", 0.5)

    if neuroticism > 0.6 and agreeableness < 0.45:
        return "confrontational"
    if neuroticism > 0.5 and patience < 0.45:
        return "frustrated"
    if agreeableness < 0.45 or patience < 0.4:
        return "impatient"
    if agreeableness > 0.6 and neuroticism < 0.4:
        return "cooperative"
    return "neutral"


def format_interaction_style_instructions(style: str) -> str:
    """Return prompt block describing the user's interaction temperament."""
    info = INTERACTION_STYLES.get(style, INTERACTION_STYLES["neutral"])
    desc = info["description"]
    if not desc:
        return ""
    return f"<INTERACTION_STYLE>\n{desc}\n</INTERACTION_STYLE>"


def compute_frustration_level(
    style: str,
    turn_idx: int,
    assistant_failures: int,
    patience: float,
    max_turns: int = 3,
) -> int:
    """Compute current frustration level (0-3) based on conversation state.

    Frustration escalates based on the user's interaction style, how many
    turns have elapsed, and how many times the assistant has been judged
    inadequate.  Thresholds scale proportionally with max_turns so that
    frustration ramps up regardless of conversation length.

    Returns:
        0 = calm, 1 = mildly annoyed, 2 = frustrated, 3 = very frustrated
    """
    info = INTERACTION_STYLES.get(style, INTERACTION_STYLES["neutral"])
    base = info["base_frustration"]
    rate = info["escalation_rate"]

    frustration = base
    if assistant_failures >= 2:
        frustration += assistant_failures * rate

    mid_threshold = max(1, int(max_turns * 0.4))
    late_threshold = max(2, int(max_turns * 0.7))
    if turn_idx >= mid_threshold and patience < 0.5:
        frustration += 0.5
    if turn_idx >= late_threshold:
        frustration += 0.5

    return min(3, max(0, int(frustration)))


def get_frustration_prompt(level: int) -> str:
    """Get the frustration injection prompt for a given level.

    Returns empty string for level 0 (calm).
    """
    if level <= 0:
        return ""
    return FRUSTRATION_PROMPTS.get(min(level, 3), "")
