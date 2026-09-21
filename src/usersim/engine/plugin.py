# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Data Designer plugin entry points.

Two plugins ship in this package:

- ``conversation_simulator`` (column type ``"conversation-simulator"``) —
  the trajectory simulator. Runs probe-dispatched user/assistant
  conversations with in-sim judges, emits trajectory parquet rows.
- ``trajectory_evaluator`` (column type ``"trajectory-evaluator"``) —
  the out-of-sim assistant evaluator. Reads stored trajectories,
  applies a judge ensemble + probe-specific scorers, writes a
  wide eval cell. Re-runnable against an unchanged trajectory store
  without re-invoking the simulator.

The two are physically separate plugins so iteration on the evaluator
(swap a judge, change an axis rubric, add a scorer) does not require
re-running the (expensive) simulation.
"""

from data_designer.plugins.plugin import Plugin, PluginType

conversation_simulator = Plugin(
    impl_qualified_name="usersim.engine.generator.ConversationSimulatorGenerator",
    config_qualified_name="usersim.engine.config.ConversationSimulatorConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

trajectory_evaluator = Plugin(
    impl_qualified_name="usersim.engine.evaluator.generator.TrajectoryEvaluatorGenerator",
    config_qualified_name="usersim.engine.evaluator.config.TrajectoryEvaluatorConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
