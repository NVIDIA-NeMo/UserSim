# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""health_disclosure — a config-driven family of health disclosure probes.

One mechanism (a move-space + a deterministic Guard on the *user* side), three
client-shaped probes that differ only by configuration:

  * ``health_therapy_disclosure``       — a therapy patient discloses to a therapist (mental health).
  * ``health_triage_disclosure``        — a patient discloses to a care-navigation / triage assistant.
  * ``health_decision_support_disclosure``— a clinician presents a case to a decision-support assistant.

The *user* side is the product in every case: a persona-grounded simulated
user who reveals sensitive information **gradually and strategically**, only
surfacing red-flag / risk content once the assistant earns it. The *assistant*
is the system under test (the client's real model at serve time; a flagged
stand-in for local runs).

What is shared vs what varies:
  * shared, verbatim: the move-space + Guard (``moves.py``) and the propose →
    guard → instruct runtime (``move_runtime.py``).
  * per client, in ``clients.py``: topic vocabulary + which topics are gated,
    what "risk" means, who the user plays (patient vs clinician), the stand-in
    assistant prompt, move-proposal framing, and the (PROVISIONAL) judge axes.

See README.md in this directory for the mechanism, the env flags, and how to
change the judge axes.
"""
