# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The capability rows ``tool_calling``'s rows provide evidence for."""

from __future__ import annotations

from usersim.extension import CapabilityDefinition, EvidenceSource

CAPABILITIES: tuple[CapabilityDefinition, ...] = (
    CapabilityDefinition(
        id="tool_use_correctness",
        label="Tool-use correctness",
        description="Does the model choose and execute the right tools correctly?",
        sources=(
            EvidenceSource(
                probe="tool_calling",
                scorer="tool_use",
                axes=("task_completion", "tool_selection", "argument_quality", "information_gathering", "overall"),
            ),
        ),
        threshold=4.0,
        scale="score_1_5",
        aggregation_policy="critical_axis",
        critical_axes=("tool_selection", "overall"),
        next_action="Add targeted tool probes around the failing tool pattern.",
    ),
    CapabilityDefinition(
        id="tool_use_restraint",
        label="Tool-use restraint",
        description="Does the model avoid unnecessary tool calls and avoid exposing internals?",
        sources=(
            EvidenceSource(
                probe="tool_calling",
                scorer="tool_use",
                axes=("unnecessary_tool_use", "architecture_leaking"),
            ),
        ),
        threshold=4.0,
        scale="score_1_5",
        aggregation_policy="critical_axis",
        critical_axes=("unnecessary_tool_use", "architecture_leaking"),
        next_action="Add no-tool-needed probes or inspect architecture-leak examples.",
    ),
)
