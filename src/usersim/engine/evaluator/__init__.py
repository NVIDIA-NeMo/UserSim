# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-sim assistant evaluation: the second DD column generator.

The simulator (``usersim.engine.generator``) writes trajectories
to the trajectory store. The evaluator reads those trajectories and
applies one or more LLM judges plus probe-specific scorers
to score the *assistant* under test. The two are physically separate
DD column generators registered as separate plugin entry points so
re-evaluating with a new judge or new axes does not require re-running
the simulator.

Design:
- judging happens outside the simulation, so judges and axes change
  without re-simulating;
- the evaluator is a Data Designer plugin, like the simulator;
- each row stores one wide cell, which the reporting layer reads directly.
"""
