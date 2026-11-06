# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from usersim.extension import CapabilityDefinition, EvidenceSource

#: The capability report's rows this probe's evidence feeds. Replace them with yours.
CAPABILITIES = (
    CapabilityDefinition(
        id="my_tool_use",
        label="Uses the tool",
        description="Replace with the behaviour the probe's evidence shows, as a reviewer reads it.",
        sources=(EvidenceSource(probe="my_probe", scorer="my_probe", axes=("my_probe.tool_used_rate",)),),
        threshold=0.8,
        scale="rate",
        aggregation_policy="mean",
    ),
)

#: What each rate the scorer writes measures, on the report's evidence cards.
AXIS_DESCRIPTIONS = {"my_probe.tool_used_rate": "Share of conversations in which the assistant used the tool."}
