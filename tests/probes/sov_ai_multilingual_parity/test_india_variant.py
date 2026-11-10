# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sov_ai_multilingual_parity in India variant locales: the first turn is translated, and variants share query ids."""

from __future__ import annotations

from _india_variant import _MARKER, _builder, _mock_translation, _mt_warned

import usersim.engine.generator  # noqa: F401  (registers the built-in probes)
from usersim.engine.core import translation as T
from usersim.engine.core.outcomes import Provenance
from usersim.engine.core.simulation import ConversationState


async def test_parity_variant_translates_turn1(indian_persona, cfg) -> None:
    from usersim.engine.probes.sov_ai_multilingual_parity.probe import (
        SovAiMultilingualParityProbe,
    )

    T.reset_translation_cache()
    ob = _builder()
    with _mock_translation():
        probe = SovAiMultilingualParityProbe(
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
        # Shared bank -> asset_locale resolves to the en_IN base rendering.
        assert probe._asset_locale == "en_IN"
        state = ConversationState(outcome=ob)
        turn1 = await probe.get_verbatim_first_user_turn(state)

    assert turn1 == _MARKER
    assert state.metadata.get("query_id")  # matched-pair join key seeded
    assert _mt_warned(ob)


def test_parity_variants_share_query_id_matched_pair(indian_persona, cfg) -> None:
    """Two India-language variants drawing the same en_IN persona pick the
    SAME query_id — so cross-language parity joins on query_id work."""
    from usersim.engine.probes.sov_ai_multilingual_parity.probe import (
        SovAiMultilingualParityProbe,
    )

    def _query_id_for(locale: str) -> str:
        probe = SovAiMultilingualParityProbe(
            persona=indian_persona,
            locale=locale,
            language="X",
            models={"summary_model": object()},
            cfg=cfg,
            provenance=Provenance(),
            profile={},
            data={},
            outcome_builder=_builder(),
        )
        assert probe._asset_locale == "en_IN"
        return probe._task.id

    assert _query_id_for("ta_Taml_IN") == _query_id_for("te_Telu_IN")


def test_the_prompt_pack_lists_only_shipped_locales() -> None:
    """The variant fallback redirects at call time (asset_locale=en_IN) rather
    than adding the variants to the pack, so its locales stay the shipped set."""
    from usersim.engine.core import locale as L
    from usersim.engine.probes.sov_ai_multilingual_parity import prompts

    assert set(prompts.available_locales()) == set(L.SHIPPED_LOCALES)
