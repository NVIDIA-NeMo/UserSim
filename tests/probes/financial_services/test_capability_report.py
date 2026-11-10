# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services' deterministic axes describe themselves in the capability report."""

from __future__ import annotations


def test_deterministic_axes_carry_a_description_instead_of_empty_reasoning():
    """A rate computed by a deterministic scorer has no per-conversation reasoning.

    The evidence card used to say "No reasoning provided by this scorer", which
    tells a reviewer neither what was counted nor why 0% is bad. Every axis a
    deterministic scorer emits should describe itself instead.
    """
    from usersim.taxonomy.capabilities import axis_description, capability_definitions

    finance_axes = {
        axis
        for d in capability_definitions()
        for s in d.sources
        if s.scorer == "financial_services"
        for axis in s.axes
        if axis.startswith("finance.")  # dynamic.* are judged and carry reasoning
    }
    assert finance_axes, "no deterministic finance axes found"
    for axis in finance_axes:
        assert axis_description(axis), axis
