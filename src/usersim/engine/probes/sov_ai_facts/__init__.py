# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sovereign-knowledge probe.

Probes the assistant for locale-specific factual knowledge grounded in
a curated fact bank. Tasks are derived from persona attributes: a
Northeastern Brazilian teacher gets different questions than a São
Paulo business owner. Ground truth comes from the bank, never from
LLM fabrication.

The probe is ``SovAiFactsProbe`` in :mod:`.probe`, registered as ``sov_ai_facts``.
"""
