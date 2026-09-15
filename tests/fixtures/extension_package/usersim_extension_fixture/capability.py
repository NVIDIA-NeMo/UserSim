# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from usersim.taxonomy.capabilities import CapabilityDefinition, EvidenceSource

CAPABILITY = CapabilityDefinition(
    id="fixture_capability",
    label="Fixture Capability",
    description="Contributed by an out-of-tree package.",
    sources=(
        EvidenceSource(
            probe="fixture_probe",
            scorer="fixture_scorer",
            axes=("fixture_axis",),
        ),
    ),
    threshold=0.8,
    scale="rate",
    aggregation_policy="mean",
)
