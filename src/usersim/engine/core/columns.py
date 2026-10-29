# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The result columns the engine writes for every probe.

A Data Designer run keeps a row's input columns, the simulator's configured
column and its declared side effects, and drops everything else. The side
effects are these core columns plus the ``output_columns`` every registered
probe class declares (``ConversationSimulatorConfig.side_effect_columns``).
A probe may not declare one of these names: ``@register_probe`` refuses it.
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


def output_columns_problem(columns: object) -> str | None:
    """Why ``columns`` cannot be a probe's ``output_columns``, or ``None`` when it can.

    A bare string is refused, since it would read as one column per
    character, and so is a core column name, which the engine already writes.
    """
    if isinstance(columns, str) or not isinstance(columns, (tuple, list)):
        return f"must be a tuple of column names, got {columns!r}"
    if not all(isinstance(column, str) and column for column in columns):
        return f"must be a tuple of non-empty column names, got {columns!r}"
    core = sorted(set(columns) & set(CORE_RESULT_COLUMNS))
    if core:
        return (
            f"names {core}, which the engine writes for every probe. "
            "Give the probe's own columns names no core column uses."
        )
    return None
