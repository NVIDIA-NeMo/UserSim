# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""The published conformance suite, run against this project's own code.

If ``usersim.testing`` describes a contract the built-ins do not keep, it is
wrong — either the check or the contract. Running it here is what stops it
from becoming aspirational documentation that fails for every extension
author who trusts it.
"""

from __future__ import annotations

import pytest

import usersim.engine.generator  # noqa: F401  (populates the registry)
from usersim.engine.core.probes import known_probes
from usersim.engine.evaluator.scorers import list_scorers
from usersim.testing import (
    assert_probe_conforms,
    assert_scorer_conforms,
    probe_conformance_problems,
    scorer_conformance_problems,
)


@pytest.mark.parametrize("probe", known_probes())
def test_shipped_probe_conforms(probe: str) -> None:
    assert_probe_conforms(probe)


@pytest.mark.parametrize("scorer", sorted(list_scorers()))
async def test_shipped_scorer_conforms(scorer: str) -> None:
    await assert_scorer_conforms(scorer)


class TestTheChecksCanFail:
    """A check that cannot fail proves nothing about the ones that pass."""

    async def test_unregistered_probe_is_reported(self) -> None:
        problems = probe_conformance_problems("not_a_probe")
        assert problems and "not registered" in problems[0]
        assert "usersim.probes" in problems[0]

    async def test_unset_label_is_reported(self) -> None:
        from usersim.engine.core.probes import BaseProbe

        class Unlabelled(BaseProbe):
            pass

        problems = probe_conformance_problems(Unlabelled)
        assert any("label" in p for p in problems)

    async def test_missing_overrides_are_reported(self) -> None:
        """A probe inheriting the abstract prompt methods registers fine and
        then raises once a run reaches it."""
        from usersim.engine.core.probes import BaseProbe

        class Hollow(BaseProbe):
            label = "hollow"

        problems = probe_conformance_problems(Hollow)
        assert any("get_user_system_prompt" in p for p in problems)

    async def test_unregistered_scorer_is_reported(self) -> None:
        problems = await scorer_conformance_problems("not_a_scorer")
        assert problems and "usersim.scorers" in problems[0]

    async def test_raising_scorer_is_reported(self) -> None:
        """The evaluator calls scorers on rows missing any given field."""
        from usersim.engine.evaluator import scorers

        async def brittle(row, models):
            return {"score": row["definitely_absent"]}

        snapshot = dict(scorers._REGISTRY)
        try:
            scorers.register_scorer("brittle", brittle)
            problems = await scorer_conformance_problems("brittle")
        finally:
            scorers._REGISTRY.clear()
            scorers._REGISTRY.update(snapshot)
        assert any("KeyError" in p for p in problems)

    async def test_non_dict_scorer_is_reported(self) -> None:
        from usersim.engine.evaluator import scorers

        snapshot = dict(scorers._REGISTRY)
        try:

            async def _stringly(row, models):
                return "nope"

            scorers.register_scorer("stringly", _stringly)
            problems = await scorer_conformance_problems("stringly")
        finally:
            scorers._REGISTRY.clear()
            scorers._REGISTRY.update(snapshot)
        assert any("expected a dict" in p for p in problems)

    def test_assert_form_lists_every_problem(self) -> None:
        with pytest.raises(AssertionError) as exc:
            assert_probe_conforms("not_a_probe")
        assert "does not conform" in str(exc.value)
