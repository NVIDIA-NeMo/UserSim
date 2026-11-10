# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_chat_pressure's preview shows each row's target request id."""

from __future__ import annotations

from engine.core.test_preview import _agentic_row, _record


def test_target_request_id_in_probe_inputs() -> None:
    # safety_chat_pressure carries target_request_id
    # — surfaces as a Probe Inputs field, not as a side-channel
    # ``id=...`` prefix.
    row = _agentic_row()
    del row["action_request_id"]
    del row["sub_protocol"]
    del row["attempted_actions"]
    row["probe_type"] = "safety_chat_pressure"
    row["probe_family"] = "safety_chat_pressure"
    row["target_request_id"] = "TR-SB-B001"
    out = _record([row])
    assert "TR-SB-B001" in out
    assert "id=TR-SB-B001" not in out
