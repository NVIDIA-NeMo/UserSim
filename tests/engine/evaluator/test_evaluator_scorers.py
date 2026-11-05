# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator/scorers/__init__.py — scorer registry."""

from __future__ import annotations

import pytest

from usersim.engine.evaluator import scorers as scorers_module


@pytest.mark.usefixtures("empty_scorer_registry")
class TestScorerRegistry:
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

    def test_the_auto_entry_cannot_be_a_scorer_name(self) -> None:
        """``--scorers auto`` stands for a set of scorers, so no one scorer may answer to it."""
        with pytest.raises(scorers_module.ScorerRegistrationError, match=r"'auto' is reserved"):
            scorers_module.register_scorer("auto", lambda t, m: {})
        assert "auto" not in scorers_module.list_scorers()

    def test_registering_the_same_scorer_again_keeps_it(self) -> None:
        def a(traj, models):
            return {"v": 1}

        scorers_module.register_scorer("x", a)
        scorers_module.register_scorer("x", a)
        assert scorers_module.get_scorer("x") is a

    def test_a_scorer_defined_again_in_the_same_place_takes_its_name(self) -> None:
        """A module reloaded in a notebook defines its scorer again, as a new function of the same name."""

        def define():
            def scorer(traj, models):
                return {"v": 1}

            return scorer

        first, again = define(), define()
        scorers_module.register_scorer("x", first)
        scorers_module.register_scorer("x", again)
        assert scorers_module.get_scorer("x") is again

    def test_clear_registry(self) -> None:
        scorers_module.register_scorer("a", lambda t, m: {})
        scorers_module.register_scorer("b", lambda t, m: {})
        assert len(scorers_module.list_scorers()) == 2
        scorers_module.clear_registry()
        assert scorers_module.list_scorers() == []


def test_loading_the_bundled_scorers_again_keeps_the_registered_functions() -> None:
    """Each scorer module is imported once, so the registry holds the function the module defines."""
    from usersim.engine.evaluator.scorers import tool_use

    registered = dict(scorers_module._REGISTRY)
    scorers_module.load_default_scorers()

    assert scorers_module._REGISTRY == registered
    assert scorers_module.get_scorer("tool_use") is tool_use.score_tool_use_trajectory
