# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services probe.

Population-grounded, hybrid-agentic financial/banking customer-support and
advisory probe. The assistant retrieves institution-scoped documents from a
knowledge base, discovers the tools those documents expose, and calls them to
resolve the user's request. Two tiers: ``verifiable`` tasks carry a
deterministically-derived gold spec (gold docs + gold tool sequence + expected
state deltas) checked by a deterministic verifier; ``dynamic`` tasks are
generated open-ended questions scored by a reference-grounded judge.

See ``README.md`` (co-located) for the full design and package map, and
``assets/financial_services/SCHEMA.md`` for the asset contract.
"""
