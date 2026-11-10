# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_chat_pressure in an India variant locale: its verbatim first turn is machine-translated."""

from __future__ import annotations

from _india_variant import _MARKER, _builder, _mock_translation, _mt_warned

import usersim.engine.generator  # noqa: F401  (registers the built-in probes)
from usersim.engine.core import translation as T
from usersim.engine.core.outcomes import Provenance
from usersim.engine.core.simulation import ConversationState


async def test_safety_chat_pressure_variant_translates_turn1(indian_persona, cfg) -> None:
    from usersim.engine.probes.safety_chat_pressure.probe import (
        SafetyChatPressureProbe,
    )

    T.reset_translation_cache()
    ob = _builder()
    with _mock_translation():
        probe = SafetyChatPressureProbe(
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
        state = ConversationState(outcome=ob)
        turn1 = await probe.get_verbatim_first_user_turn(state)

    assert turn1 == _MARKER
    assert state.metadata.get("target_request_id")
    assert _mt_warned(ob)
