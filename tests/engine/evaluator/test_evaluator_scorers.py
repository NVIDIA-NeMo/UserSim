# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator/scorers/__init__.py — scorer registry."""

from __future__ import annotations

import pytest

from usersim.engine.evaluator import scorers as scorers_module


class TestScorerRegistry:
    # The registry is process-global, so teardown restores what was there
    # instead of leaving it empty for whatever runs next in this worker.
    def setup_method(self) -> None:
        self._snapshot = dict(scorers_module._REGISTRY)
        scorers_module.clear_registry()

    def teardown_method(self) -> None:
        scorers_module.clear_registry()
        scorers_module._REGISTRY.update(self._snapshot)

    def test_register_and_get(self) -> None:
        def fake_scorer(traj: dict, models: dict) -> dict:
            return {"score": 0.5}

        scorers_module.register_scorer("fake", fake_scorer)
        assert scorers_module.get_scorer("fake") is fake_scorer
        assert scorers_module.list_scorers() == ["fake"]

    def test_unknown_raises_with_helpful_message(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            scorers_module.get_scorer("nope")

    def test_a_second_scorer_under_a_taken_name_is_refused(self) -> None:
        def a(traj, models):
            return {"v": 1}

        def b(traj, models):
            return {"v": 2}

        scorers_module.register_scorer("x", a)
        with pytest.raises(
            scorers_module.ScorerRegistrationError,
            match=r"'x' is already registered by .*\.a, so .*\.b cannot take it",
        ):
            scorers_module.register_scorer("x", b)
        assert scorers_module.get_scorer("x") is a

    def test_registering_the_same_scorer_again_keeps_it(self) -> None:
        def a(traj, models):
            return {"v": 1}

        scorers_module.register_scorer("x", a)
        scorers_module.register_scorer("x", a)
        assert scorers_module.get_scorer("x") is a

    def test_reloading_a_scorer_module_updates_its_scorer(self) -> None:
        scorers_module.reload_scorers(["usersim.engine.evaluator.scorers.tool_use"])
        first = scorers_module.get_scorer("tool_use")
        scorers_module.reload_scorers(["usersim.engine.evaluator.scorers.tool_use"])
        assert scorers_module.get_scorer("tool_use") is not first

    def test_clear_registry(self) -> None:
        scorers_module.register_scorer("a", lambda t, m: {})
        scorers_module.register_scorer("b", lambda t, m: {})
        assert len(scorers_module.list_scorers()) == 2
        scorers_module.clear_registry()
        assert scorers_module.list_scorers() == []

    def test_reload_scorers_imports(self) -> None:
        # reload_scorers should not raise on a valid module path even
        # if it doesn't define any scorers (e.g., the scorers package
        # __init__ itself).
        scorers_module.reload_scorers(["usersim.engine.evaluator.scorers"])
