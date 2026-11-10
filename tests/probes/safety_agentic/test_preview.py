# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_agentic's preview: its scenario inputs and the single-user-turn description."""

from __future__ import annotations

import json

from engine.core.test_preview import _agentic_row, _record


def test_action_request_id_in_probe_inputs() -> None:
    # Curated-asset id surfaces in the Probe Inputs panel as a
    # labeled field rather than the older ``id=...`` side-
    # channel detail line. Pin the asset id is still present
    # in the rendered output (just relocated).
    out = _record([_agentic_row()])
    assert "AR-CONSEQ-001" in out
    # The ``id=`` side-channel prefix was dropped — Probe
    # Inputs uses ``Action request id: ...`` instead.
    assert "id=AR-CONSEQ-001" not in out


def test_sub_protocol_in_probe_inputs() -> None:
    out = _record([_agentic_row()])
    assert "consequence_disclosure" in out
    assert "sub_protocol=consequence_disclosure" not in out


def test_agentic_probe_description_explains_single_user_turn_design() -> None:
    out = _record([_agentic_row()])
    assert "Single-user-turn by design" in out
    assert "expected, not a truncated transcript" in out


def test_safety_agentic_inputs_panel() -> None:
    row = _agentic_row()
    row["conversation_metadata"] = json.dumps(
        {
            "sanctioned_action_name": "send_email_internal",
            "tempted_action_name": "transfer_funds_external",
            "attempted_actions": [],
        }
    )
    out = _record([row])
    assert "Probe Inputs" in out
    # Both row-source and meta-source fields surface in the panel.
    assert "Action request id" in out
    assert "Sanctioned action" in out
    assert "send_email_internal" in out
    assert "transfer_funds_external" in out
