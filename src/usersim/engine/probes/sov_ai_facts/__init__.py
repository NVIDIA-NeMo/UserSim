# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Sovereign-knowledge probe.

Probes the assistant for locale-specific factual knowledge grounded in
a curated fact bank. Tasks are derived from persona attributes: a
Northeastern Brazilian teacher gets different questions than a São
Paulo business owner. Ground truth comes from the bank, never from
LLM fabrication.

Entry point: :func:`simulate_sov_ai_facts` in
:mod:`.generator` — registered as ``sov_ai_facts``
on ``_PROBE_REGISTRY`` in :mod:`usersim.engine.generator`.
"""
