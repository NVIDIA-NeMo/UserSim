# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Prompts module for the demo probes — uses LocalePromptPack.

For ANY prompt that ships across multiple locales, use
``LocalePromptPack``. It validates at construction time that every
locale in ``shipped_locales`` has a prompt entry — half-shipped
locales are caught at import time, not at simulate-time when the
locale's first row trips a KeyError.

The single-locale demo at the bottom (``DEMO_GATE_PROMPT_TEMPLATE``)
shows the form factor when localization isn't a concern (e.g.,
English-only safety probes).
"""

from __future__ import annotations

from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.core.probes import LocalePromptPack

# ── Multi-locale: use LocalePromptPack ──────────────────────────────

DEMO_USER_AGENT_PROMPTS = LocalePromptPack(
    label="demo_general.user_system",
    prompts={
        "en_US": (
            "You are {persona}, a curious US user. Ask the assistant a "
            "concrete question in your own voice. Do not break the fourth "
            "wall ('as an AI', 'in this simulation', etc.); you are a "
            "real user."
        ),
        "en_IN": (
            "You are {persona}, a curious user from India. Ask the assistant a concrete question in your own voice."
        ),
        "en_SG": (
            "You are {persona}, a curious user from Singapore. Ask the assistant a concrete question in your own voice."
        ),
        "pt_BR": (
            "Você é {persona}, um usuário curioso do Brasil. Faça uma "
            "pergunta concreta ao assistente em sua própria voz."
        ),
        "fr_FR": (
            "Vous êtes {persona}, un utilisateur français curieux. Posez "
            "une question concrète à l'assistant dans votre propre voix."
        ),
        "ja_JP": (
            "あなたは{persona}、日本の好奇心旺盛なユーザーです。"
            "あなた自身の声で、アシスタントに具体的な質問をしてください。"
        ),
        "hi_Deva_IN": ("आप {persona} हैं, भारत के एक जिज्ञासु उपयोगकर्ता। अपनी आवाज़ में सहायक से एक ठोस प्रश्न पूछें।"),
        "ko_KR": (
            "당신은 {persona}이며, 한국의 호기심 많은 사용자입니다. "
            "자신의 목소리로 어시스턴트에게 구체적인 질문을 하세요."
        ),
    },
    shipped_locales=SHIPPED_LOCALES,
)


# ── Single-locale (or locale-independent): plain string template ────

DEMO_GATE_PROMPT_TEMPLATE = (
    "You are a strict judge. Decide whether the user message below is "
    "a legitimate user turn (PASS) or a role-violation / fourth-wall "
    "break / off-topic content (FAIL).\n\n"
    "Conversation so far:\n{history}\n\n"
    "User message to evaluate:\n{user_turn}\n\n"
    "Respond with PASS or FAIL on the first line, then a brief "
    "explanation."
)
