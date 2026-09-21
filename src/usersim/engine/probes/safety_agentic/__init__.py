# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Agentic-safety probe.

Probes the assistant for safe tool-use behaviour in a mocked
environment: each trajectory runs the assistant against a curated
``ActionRequest`` (from the agentic-action bank) where the
*sanctioned* action and the *tempted* action are pinned in advance.
Every tool call the assistant attempts is **intercepted** (no
external execution) and answered with a pre-canned mock response from
the action request's ``mock_tool_responses`` map.

The simulator records every attempted action — name, arguments,
turn index — as a side channel for the ``safety_agentic`` scorer to
read. Whether the assistant called the right tool, whether it disclosed
consequences before high-blast-radius calls, whether it asked for
confirmation when the action taxonomy says it should — those are
the scorer's questions, not the simulator's.

Entry point: :func:`simulate_safety_agentic` in :mod:`.generator` —
registered as ``safety_agentic`` on ``_PROBE_REGISTRY`` in
:mod:`usersim.engine.generator`.
"""
