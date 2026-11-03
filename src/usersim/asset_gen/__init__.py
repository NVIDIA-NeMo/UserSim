# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""asset_gen: the generic dispatch for building asset banks offline.

``registry.py`` holds :class:`AssetDomain` and the built-in domains, and
``usersim gen-assets``, ``validate-assets`` and ``evaluate-assets`` dispatch
through it. A domain's generator lives beside the probe whose banks it builds:
``usersim.engine.probes.financial_services.asset_gen`` authors the committed
``assets/financial_services/<locale>/`` banks the probe and its scorer read.
"""
