# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Out-of-sim assistant evaluation — the second DD column generator.

The simulator (``usersim.engine.generator``) writes trajectories
to the trajectory store. The evaluator reads those trajectories and
applies a judge ensemble + probe-specific deterministic scorers
to score the *assistant* under test. The two are physically separate
DD column generators registered as separate plugin entry points so
re-evaluating with a new judge or new axes does not require re-running
the simulator.

The load-bearing decisions:
- in-sim vs out-of-sim judge boundary,
- staying with the plugin entry-point pattern (SLURM compatibility),
- wide-per-row storage with reporting-layer wide -> long via pd.melt.
"""
