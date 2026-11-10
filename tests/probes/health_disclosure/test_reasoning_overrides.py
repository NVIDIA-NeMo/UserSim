# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The health probes' reasoning-effort overrides use only settings ``acall_llm`` passes on to a model."""

from __future__ import annotations

import pytest

from usersim.engine.core import llm
from usersim.engine.probes.health_disclosure import move_runtime, realized_audit


@pytest.mark.parametrize("value", ["off", "low", "medium", "high"])
def test_the_reasoning_overrides_use_only_settings_acall_llm_routes(monkeypatch, value) -> None:
    monkeypatch.setenv("USERSIM_AUDIT_REASONING_EFFORT", value)
    monkeypatch.setattr(move_runtime, "_MOVE_REASONING_EFFORT", value)
    for produced in (realized_audit._audit_reasoning_kwargs(), move_runtime._move_reasoning_kwargs()):
        assert produced and set(produced) <= set(llm._EXTRA_BODY_SETTINGS)
