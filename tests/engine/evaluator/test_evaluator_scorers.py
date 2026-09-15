# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator/scorers/__init__.py — scorer registry."""

from __future__ import annotations

import pytest

from usersim.engine.evaluator import scorers as scorers_module


class TestScorerRegistry:
    def setup_method(self) -> None:
        scorers_module.clear_registry()

    def teardown_method(self) -> None:
        scorers_module.clear_registry()

    def test_register_and_get(self) -> None:
        def fake_scorer(traj: dict, models: dict) -> dict:
            return {"score": 0.5}

        scorers_module.register_scorer("fake", fake_scorer)
        assert scorers_module.get_scorer("fake") is fake_scorer
        assert scorers_module.list_scorers() == ["fake"]

    def test_unknown_raises_with_helpful_message(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            scorers_module.get_scorer("nope")

    def test_idempotent_re_registration(self) -> None:
        def a(traj, models): return {"v": 1}
        def b(traj, models): return {"v": 2}

        scorers_module.register_scorer("x", a)
        scorers_module.register_scorer("x", b)
        # Last write wins; no error.
        assert scorers_module.get_scorer("x") is b

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
