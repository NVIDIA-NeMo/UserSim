# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Taxonomy — the vocabulary of capabilities and evaluation cells.

This package sits below both ``reporting`` and ``selection`` and depends on
neither. It holds the shared definitions those layers agree on: what a
capability is, which evidence supports it, and how a single evaluation cell
is addressed.

The pipeline runs ``config -> generation -> storage -> reporting ->
selection``. Capability definitions are needed at the *config* end -- to
validate a probe mix and to warn about capabilities with no scorer -- while
also being needed at the reporting end to build report cells. Keeping them
in ``reporting`` made config-time construction import the reporting layer,
which inverted that chain. A neutral package below both removes the cycle
without either side importing the other.

Cell-decoding helpers live in :mod:`taxonomy.eval_cell` and are imported
from there directly rather than re-exported here, so a reader can tell which
of the two concerns a call site depends on.
"""

from usersim.taxonomy.capabilities import (
    ALL_PROBES,
    CapabilityDefinition,
    EvidenceSource,
    capabilities_missing_scorer,
    capability_definitions,
)

__all__ = [
    "ALL_PROBES",
    "CapabilityDefinition",
    "EvidenceSource",
    "capabilities_missing_scorer",
    "capability_definitions",
]
