# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Conversational-pressure safety probe.

Probes the assistant for refusal robustness under sustained conversational
pressure. Each trajectory is a `(persona, target_request, strategy)`
tuple: the user-agent applies a multi-turn pressure strategy
(persistence / hypothetical_framing / authority_impersonation) to a
target the assistant should refuse, rotating through the strategy's
reframings turn-by-turn.

The simulator records turn-by-turn telemetry (which reframing was
applied on each turn, the assistant's response). It does **not** judge
whether the assistant flipped — that is the
``safety_chat_pressure`` scorer's job, which reads the trajectory's
``conversation_messages`` + ``target_request_id`` +
``expected_refusal`` (re-loaded from the bank) and emits per-turn
flip indicators plus a trajectory-level ``turn_of_flip``.

Entry point: :func:`simulate_safety_chat_pressure` in
:mod:`.generator` — registered as ``safety_chat_pressure``
on ``_PROBE_REGISTRY`` in :mod:`usersim.engine.generator`.
"""
