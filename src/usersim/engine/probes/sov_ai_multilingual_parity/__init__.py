# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Multilingual-consistency probe.

Probes the assistant for **respect parity** within a locale and **quality
consistency** across locales by injecting the same conceptual question
verbatim — drawn from a curated cross-locale query bank where each entry
holds N locale renderings joined by a single ``query_id``.

The matched-pair join key is ``query_id``: trajectories that probed the
same conceptual question (in different locales by demographically
matched personas) are joinable downstream by joining on
``query_id`` and stratifying by persona attributes.

Entry point: :func:`simulate_sov_ai_multilingual_parity` in
:mod:`.generator` — registered as ``sov_ai_multilingual_parity``
on ``_PROBE_REGISTRY`` in :mod:`usersim.engine.generator`.
"""
