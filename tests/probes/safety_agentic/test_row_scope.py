# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_agentic installs its own outcome builder for a step, and puts the row's back afterwards."""

from __future__ import annotations

import pytest


async def test_a_nested_scope_puts_the_row_builder_back() -> None:
    """A probe that installs its own builder must not detach the row's.

    Every model call feeds the installed builder, so a scope that clears
    instead of restoring leaves the rest of the row unattributed and its
    token totals short.
    """
    import types

    from usersim.engine.core import llm
    from usersim.engine.core.outcomes import OutcomeBuilder, Provenance
    from usersim.engine.probes.safety_agentic import probe as safety_agentic

    row_builder = OutcomeBuilder(provenance=Provenance())
    probe = safety_agentic.SafetyAgenticProbe.__new__(safety_agentic.SafetyAgenticProbe)
    probe._provenance = Provenance()
    probe._outcome_builder = OutcomeBuilder(provenance=Provenance())
    probe._task = types.SimpleNamespace(
        action_request=types.SimpleNamespace(
            id="ar-1",
            sub_protocol="sp",
            sanctioned_action_name="a",
            tempted_action_name="b",
            bank_id="bank",
            bank_version="v1",
        )
    )

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("stop once the nested builder is installed")

    with llm.outcome_builder_scope(row_builder):
        with pytest.raises(RuntimeError):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(safety_agentic, "ConversationState", _explode)
                await probe.run_dispatch(models={}, data={}, cfg=types.SimpleNamespace())
        assert llm._OUTCOME_BUILDER.get() is row_builder, (
            "the nested scope cleared the row's outcome builder instead of "
            "restoring it, so later calls in this row record nowhere"
        )
