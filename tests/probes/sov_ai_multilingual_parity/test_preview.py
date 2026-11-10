# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sov_ai_multilingual_parity's preview shows each row's query id and concern."""

from __future__ import annotations

import json

from engine.core.test_preview import _agentic_row, _general_row, _record


def test_query_id_in_probe_inputs() -> None:
    # sov_ai_multilingual_parity carries query_id —
    # surfaces in Probe Inputs.
    row = _agentic_row()
    del row["action_request_id"]
    del row["sub_protocol"]
    del row["attempted_actions"]
    row["probe_type"] = "sov_ai_multilingual_parity"
    row["probe_family"] = "sov_ai_multilingual_parity"
    row["query_id"] = "Q-HEALTH-001"
    out = _record([row])
    assert "Q-HEALTH-001" in out
    assert "id=Q-HEALTH-001" not in out


def test_multilingual_consistency_inputs_panel() -> None:
    row = _general_row()
    row["probe_type"] = "sov_ai_multilingual_parity"
    row["probe_family"] = "sov_ai_multilingual_parity"
    row["query_id"] = "Q-HEALTH-001"
    row["conversation_metadata"] = json.dumps(
        {
            "query_concern": "respect.condescension",
            "query_domain": "health",
            "query_difficulty": "medium",
        }
    )
    out = _record([row])
    assert "Probe Inputs" in out
    assert "Query id" in out
    assert "Q-HEALTH-001" in out
    assert "respect.condescension" in out
