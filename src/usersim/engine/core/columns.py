# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The result columns the engine writes for every probe.

A Data Designer run keeps a row's input columns, the simulator's configured
column and its declared side effects, and drops everything else.
``ConversationSimulatorConfig.side_effect_columns`` lists these core columns
first, then the columns individual probes write.
"""

from __future__ import annotations

CORE_RESULT_COLUMNS: tuple[str, ...] = (
    "conversation_messages",
    "user_query",
    "conversation_metadata",
    "conversation_status",
    "behavioral_profile",
    "locale",
    "conversation_language",
    "num_turns",
    "num_tool_calls",
    "tool_subset",
    "disclosure_style",
    "user_interaction_style",
    "persona_grounding",
    # Normalized protected fields rendered into the user-agent prompt.
    # The raw persona input is dropped, so this compact JSON audit view
    # preserves what influenced the simulated user.
    "persona_religion_language_context",
    # Structured per-trajectory outcome + per-turn telemetry sidecar.
    # See core/outcomes.py for the schema.
    "simulation_outcome",
    "simulation_traces",
    "probe_family",
    "probe_variant",
    # Stable identity for idempotent re-runs. See core/identity.py.
    "persona_uuid",
    "trajectory_id",
    "usersim_provenance",
)
