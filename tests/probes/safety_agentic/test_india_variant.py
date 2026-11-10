# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_agentic in an India variant locale: the probe constructs and routes."""

from __future__ import annotations

from _india_variant import _MARKER, _builder, _mock_translation, _mt_warned

import usersim.engine.generator  # noqa: F401  (registers the built-in probes)
from usersim.engine.core.outcomes import Provenance


async def test_safety_agentic_variant_constructs_and_routes(indian_persona, cfg) -> None:
    from usersim.engine.probes.safety_agentic.probe import SafetyAgenticProbe

    ob = _builder()
    probe = SafetyAgenticProbe(
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
    assert probe._locale == "ta_Taml_IN"
    # The translation gate fires because asset_locale != locale.
    with _mock_translation():
        assert await probe._localize_verbatim("english action prompt") == _MARKER
    assert _mt_warned(ob)
