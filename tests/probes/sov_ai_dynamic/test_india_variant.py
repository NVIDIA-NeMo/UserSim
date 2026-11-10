# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sov_ai_dynamic in India variant locales: it reads the en_IN taxonomy and gets the script directive."""

from __future__ import annotations

from _india_variant import _builder, _mt_warned

import usersim.engine.generator  # noqa: F401  (registers the built-in probes)
from usersim.engine.core.outcomes import Provenance


def test_sov_ai_dynamic_variant_uses_en_in_taxonomy(indian_persona, cfg) -> None:
    from usersim.engine.probes.sov_ai_dynamic.probe import SovAiDynamicProbe

    ob = _builder()
    probe = SovAiDynamicProbe(
        persona=indian_persona,
        locale="ta_Taml_IN",
        language="Tamil",
        models={"summary_model": object()},
        cfg=cfg,
        provenance=Provenance(),
        profile={},
        data={},
        outcome_builder=ob,
    )
    assert probe._asset_locale == "en_IN"
    assert probe._bank.taxonomy_id.startswith("en_IN")
    # No verbatim turn-1 -> generate-and-gate; no MT warning on construction.
    assert not _mt_warned(ob)
    prompt = probe.get_user_system_prompt()
    assert prompt  # en_IN prompt pack resolved (no abort)
    # CRITICAL: turn-1 is generated from this prompt, and the en_IN pack is
    # English — so the target-language directive MUST be appended, else the
    # agent writes English and the script gate exhausts (the bug that made
    # every sov_ai_dynamic variant row fail with USER_LANGUAGE_GATE_EXHAUSTED).
    assert "You MUST write your messages in Tamil" in prompt
    assert "native Tamil script" in prompt


def test_sov_ai_dynamic_romanized_variant_gets_romanized_directive(indian_persona, cfg) -> None:
    from usersim.engine.probes.sov_ai_dynamic.probe import SovAiDynamicProbe

    probe = SovAiDynamicProbe(
        persona=indian_persona,
        locale="ta_Latn_IN",
        language="Tamil",
        models={"summary_model": object()},
        cfg=cfg,
        provenance=Provenance(),
        profile={},
        data={},
        outcome_builder=_builder(),
    )
    assert "romanized Tamil" in probe.get_user_system_prompt()


def test_sov_ai_dynamic_shipped_locale_prompt_unchanged(indian_persona, cfg) -> None:
    # Shipped locales must not get an appended directive (asset_locale == locale).
    from usersim.engine.probes.sov_ai_dynamic.probe import SovAiDynamicProbe

    probe = SovAiDynamicProbe(
        persona=indian_persona,
        locale="en_US",
        language="English",
        models={"summary_model": object()},
        cfg=cfg,
        provenance=Provenance(),
        profile={},
        data={},
        outcome_builder=_builder(),
    )
    assert "You MUST write your messages in" not in probe.get_user_system_prompt()


def test_the_prompt_pack_lists_only_shipped_locales() -> None:
    """The variant fallback redirects at call time (asset_locale=en_IN) rather
    than adding the variants to the pack, so its locales stay the shipped set."""
    from usersim.engine.core import locale as L
    from usersim.engine.probes.sov_ai_dynamic import prompts

    assert set(prompts.available_locales()) == set(L.SHIPPED_LOCALES)
