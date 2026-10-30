# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What ``usersim eval --scorers auto`` stands for.

``auto`` is the rule-based scorers that apply to every probe plus the
``default_scorers`` of each probe whose rows the pass evaluates, found by the
``probe_family`` the rows are stored under.
"""

from __future__ import annotations

import logging

import pytest

from usersim.cli.evaluate import _resolve_scorers
from usersim.engine.core.probes import known_probes, resolve_probe
from usersim.engine.evaluator.scorers import GENERIC_SCORERS
from usersim.taxonomy.capabilities import ALL_PROBES, capabilities_missing_scorer, capability_definitions


def test_auto_is_the_generic_scorers_plus_the_run_probes_own() -> None:
    assert _resolve_scorers(["auto"], ["tool_calling", "general_open_ended"]) == [*GENERIC_SCORERS, "tool_use"]


def test_names_beside_auto_are_kept_in_place_each_once() -> None:
    resolved = _resolve_scorers(["tool_use", "auto", "language_compliance", "my_scorer"], ["tool_calling"])

    assert resolved == ["tool_use", *GENERIC_SCORERS, "my_scorer"]


def test_a_list_without_auto_is_passed_on_as_given() -> None:
    assert _resolve_scorers(["tool_use"], ["financial_services"]) == ["tool_use"]


def test_a_family_no_installed_probe_stores_is_named(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="usersim.cli.eval"):
        resolved = _resolve_scorers(["auto"], ["retired_probe", "tool_calling"])

    assert resolved == [*GENERIC_SCORERS, "tool_use"]
    (warning,) = [r.getMessage() for r in caplog.records]
    assert "probe_family retired_probe" in warning
    assert "tool_calling" not in warning


def test_a_blank_family_adds_no_scorer_and_no_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="usersim.cli.eval"):
        resolved = _resolve_scorers(["auto"], ["", "tool_calling"])

    assert resolved == [*GENERIC_SCORERS, "tool_use"]
    assert not caplog.records


def test_a_probe_is_found_by_the_family_its_rows_are_stored_under(register_test_probe) -> None:
    register_test_probe("ext_recipes", family="ext_cooking", default_scorers=("tool_use",))

    assert _resolve_scorers(["auto"], ["ext_cooking"]) == [*GENERIC_SCORERS, "tool_use"]
    assert _resolve_scorers(["auto"], ["ext_recipes"]) == list(GENERIC_SCORERS)


def test_auto_over_every_built_in_probe_runs_every_scorer_the_capability_report_reads() -> None:
    families = [resolve_probe(label).family or label for label in known_probes()]

    assert capabilities_missing_scorer(_resolve_scorers(["auto"], families)) == {}


@pytest.mark.parametrize(
    ("probe", "scorer"),
    sorted(
        {
            (source.probe, source.scorer)
            for definition in capability_definitions()
            for source in definition.sources
            if source.scorer and source.probe and source.probe != ALL_PROBES
        }
    ),
)
def test_auto_over_one_probe_runs_the_scorers_its_capabilities_read(probe: str, scorer: str) -> None:
    assert scorer in _resolve_scorers(["auto"], [resolve_probe(probe).family or probe])
