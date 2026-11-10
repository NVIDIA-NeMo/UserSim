# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sov_ai_facts in an India variant locale: its verbatim fact question is machine-translated."""

from __future__ import annotations

from _india_variant import _MARKER, _builder, _mock_translation, _mt_warned

import usersim.engine.generator  # noqa: F401  (registers the built-in probes)
from usersim.engine.core import translation as T
from usersim.engine.core.outcomes import Provenance
from usersim.engine.core.simulation import ConversationState


async def test_sov_ai_facts_variant_translates_turn1(indian_persona, cfg) -> None:
    from usersim.engine.probes.sov_ai_facts.probe import SovAiFactsProbe

    T.reset_translation_cache()
    ob = _builder()
    with _mock_translation():
        probe = SovAiFactsProbe(
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
        assert probe._asset_locale == "en_IN"  # fell back to the en_IN bank
        assert probe._locale == "ta_Taml_IN"  # conversation locale preserved
        state = ConversationState(outcome=ob)
        turn1 = await probe.get_verbatim_first_user_turn(state)

    assert turn1 == _MARKER  # translated
    assert state.metadata.get("facts_probed")  # metadata still seeded
    assert _mt_warned(ob)  # machine-translation warning


def test_the_prompt_pack_lists_only_shipped_locales() -> None:
    """The variant fallback redirects at call time (asset_locale=en_IN) rather
    than adding the variants to the pack, so its locales stay the shipped set."""
    from usersim.engine.core import locale as L
    from usersim.engine.probes.sov_ai_facts import prompts

    assert set(prompts.available_locales()) == set(L.SHIPPED_LOCALES)
