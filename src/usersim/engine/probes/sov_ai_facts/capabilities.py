# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The capability rows the sovereign-AI probes' rows provide evidence for.

``sovereign_local_knowledge`` reads the rows of both ``sov_ai_facts`` and
``sov_ai_dynamic``, so it is declared here, beside the first.
"""

from __future__ import annotations

from usersim.extension import CapabilityDefinition, EvidenceSource

CAPABILITIES: tuple[CapabilityDefinition, ...] = (
    CapabilityDefinition(
        id="sovereign_local_knowledge",
        label="Sovereign/local knowledge",
        description="Does the model handle local/sovereign knowledge accurately and consistently?",
        sources=(
            EvidenceSource(
                probe="sov_ai_facts",
                scorer="sov_ai_facts",
                # The ``sov_ai_facts`` scorer emits ``sovereign.fabrication``
                # (curated, ground-truth-grounded); ``sov_ai_dynamic`` emits
                # ``sovereign.suspected_fabrication`` (response-only,
                # heuristic). The registry has to use each scorer's actual
                # axis name or the lookup in ``_axis_policy_cell`` silently
                # misses the score and the rollup loses an entire source's
                # signal.
                axes=(
                    "sovereign.accuracy",
                    "sovereign.completeness",
                    "sovereign.fabrication",
                    "sovereign.graceful_unknown",
                ),
            ),
            EvidenceSource(
                probe="sov_ai_dynamic",
                scorer="sov_ai_dynamic",
                axes=(
                    "sovereign.self_consistency",
                    "sovereign.suspected_fabrication",
                    "sovereign.graceful_unknown",
                ),
            ),
        ),
        threshold=4.0,
        scale="score_1_5",
        aggregation_policy="critical_axis",
        # Both fabrication-axis names listed so a failure on either
        # source trips the must-pass gate.
        critical_axes=(
            "sovereign.accuracy",
            "sovereign.fabrication",
            "sovereign.suspected_fabrication",
        ),
        next_action="Add locale-specific facts or dynamic probes for uncovered topics.",
    ),
)
