# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the capability registry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from usersim.engine.core.builtin_packages import builtin_modules
from usersim.taxonomy.capabilities import (
    ALL_PROBES,
    _probe_ref,
    _scorer_ref,
    capabilities_missing_scorer,
    capability_by_id,
    capability_definitions,
    implementation_refs_for_capability,
    probe_label_for_capability,
    scorers_required_by_capabilities,
)


def test_every_capability_has_sources_and_policy() -> None:
    for definition in capability_definitions():
        assert definition.sources, definition.id
        assert definition.aggregation_policy, definition.id
        if definition.aggregation_policy == "critical_axis":
            assert definition.critical_axes, definition.id


def test_probe_labels_are_explicit() -> None:
    """A capability is labelled with the probes it reads, in order, or "All" when it reads every probe."""
    assert probe_label_for_capability(capability_by_id("assistant_quality")) == "All"
    for definition in capability_definitions():
        probes = [source.probe for source in definition.sources]
        expected = "All" if ALL_PROBES in probes else ", ".join(dict.fromkeys(probes))
        assert probe_label_for_capability(definition) == expected, definition.id


def test_expected_capabilities_present() -> None:
    ids = {definition.id for definition in capability_definitions()}
    assert "cultural_sensitivity" in ids
    assert "conversation_conciseness" in ids
    assert "conversation_formatting" in ids
    assert "general_safety" in ids
    assert "tool_use_restraint" in ids


def test_language_appropriateness_is_consumed_by_a_capability() -> None:
    """Guards the "judged but never read" class of bug.

    Every trajectory pays an LLM judge for ``language_appropriateness``, and for
    a long time no capability consumed it -- so the only signal that covered the
    five locales lingua cannot detect sat unused while those cells rendered as
    untested. If a judge axis is worth paying for, something has to read it.
    """
    consumed = {a for d in capability_definitions() for s in d.sources for a in s.axes}
    assert "language_appropriateness" in consumed


def test_language_capabilities_answer_different_questions() -> None:
    """Compliance and quality must stay separate.

    ``language_appropriateness`` is COMPOUND -- its rubric blends register fit
    with fluency and language choice, and only its score of 1 mentions language
    at all. Feeding it into the "did you answer in the right language" capability
    would report language failures that do not exist: on a real run the judge
    said "the conversation is in Marathi as requested" and still scored 2 for
    unnatural Marathi. Conversely the deterministic axes cannot see fluency.
    """
    compliance = capability_by_id("multilingual_reliability")
    quality = capability_by_id("language_quality_register")

    compliance_axes = {a for s in compliance.sources for a in s.axes}
    quality_axes = {a for s in quality.sources for a in s.axes}
    assert not compliance_axes & quality_axes
    assert "language_appropriateness" not in compliance_axes
    assert quality_axes == {"language_appropriateness"}

    # Script integrity is the gate: it is the only axis every locale can be
    # held to, because it needs no detector model.
    assert compliance.critical_axes == ("language.script_integrity_rate",)

    # Under `minimum` the score IS the lowest axis, so anything listed gates
    # regardless of critical_axes. script_compliance_rate is therefore kept out:
    # it counts legitimate Latin (UPI, KYC, PAN, tool names) against the locale
    # and floors near 0.83 on clean Devanagari, which failed the two locales
    # with the cleanest script of the whole Indic set.
    assert compliance.aggregation_policy == "minimum"
    assert "language.script_compliance_rate" not in compliance_axes
    assert "language.first_turn_match" not in compliance_axes


def test_language_quality_threshold_can_actually_discriminate() -> None:
    """At 4.0 this axis passed ZERO of 13 locales on the calibration run,
    English included (en_US 3.94). A capability that fails everything lowers
    every model's comparison composite by the same amount and ranks nothing.
    """
    quality = capability_by_id("language_quality_register")
    assert quality.scale == "score_1_5"
    assert quality.threshold == 3.5


def test_all_probe_capabilities_are_declared_with_sentinel() -> None:
    all_probe_ids = {
        "assistant_quality",
        "cultural_sensitivity",
        "conversation_conciseness",
        "conversation_formatting",
        "general_safety",
        "multilingual_reliability",
        "simulation_reliability",
    }
    for capability_id in all_probe_ids:
        assert any(source.probe == ALL_PROBES for source in capability_by_id(capability_id).sources)


def test_capabilities_expose_implementation_refs() -> None:
    """A capability's refs include the module of each probe it names."""
    for definition in capability_definitions():
        refs = implementation_refs_for_capability(definition)
        for source in definition.sources:
            if source.probe and source.probe != ALL_PROBES:
                assert _probe_ref(source.probe) in refs, (definition.id, source.probe)


def test_each_probe_and_scorer_a_capability_names_has_a_ref() -> None:
    """A probe's or scorer's ref is the file its registered class or function is defined in."""
    sources = [source for definition in capability_definitions() for source in definition.sources]
    probes = {source.probe for source in sources if source.probe and source.probe != ALL_PROBES}
    scorers = {source.scorer for source in sources if source.scorer}
    assert probes and scorers
    assert sorted(probe for probe in probes if _probe_ref(probe) is None) == []
    assert sorted(scorer for scorer in scorers if _scorer_ref(scorer) is None) == []


def test_every_implementation_ref_names_a_file_in_the_repo() -> None:
    """A ref is a path from the repo root; a probe's module, scorer or bank that moves takes its refs along."""
    root = Path(__file__).resolve().parents[1]
    refs = {ref for definition in capability_definitions() for ref in implementation_refs_for_capability(definition)}
    assert refs
    assert sorted(ref for ref in refs if not (root / ref).is_file()) == []


def test_no_capability_uses_status_rate() -> None:
    """No capability should use ``status_rate`` aggregation.

    A binary ``status_proposal`` gate over multi-turn trajectories
    treats a single hedged or capitulated turn as catastrophic, which
    drives the locale rate to 0% even when the model held the line on
    most turns. New capabilities should use ``critical_axis`` or
    ``mean`` over per-axis 1-5 scores instead.
    """
    for definition in capability_definitions():
        assert definition.aggregation_policy != "status_rate", definition.id


# ── a capability is only as good as the scorer the run actually dispatches ────


def test_required_scorers_covers_every_declared_scorer() -> None:
    required = set(scorers_required_by_capabilities())
    for definition in capability_definitions():
        for source in definition.sources:
            if source.scorer:
                assert source.scorer in required, f"{definition.id} -> {source.scorer}"


def test_every_declared_scorer_actually_exists() -> None:
    """A capability naming a scorer that is not registered can never be scored.

    The symptom is indistinguishable from an untested model -- the cell reads
    Untested with n=0 -- so a typo here is invisible without this check.
    """
    from usersim.engine.evaluator.scorers import get_scorer

    for scorer in scorers_required_by_capabilities():
        get_scorer(scorer)  # raises KeyError if unregistered


def test_missing_scorer_map_is_empty_when_everything_is_dispatched() -> None:
    assert capabilities_missing_scorer(scorers_required_by_capabilities()) == {}


def test_eval_notebook_derives_its_scorer_list() -> None:
    """The notebook must not hand-maintain a copy of the scorer list.

    It did, and the copy drifted: nine of the ten scorers were listed, and the
    missing one silently zeroed 11 capabilities. Asserting on the raw text rather
    than parsing keeps this cheap -- the notebook is ~32MB of embedded output.
    """
    import re
    from pathlib import Path

    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "02_evaluate_simulation.ipynb"
    assert nb_path.is_file(), f"evaluation notebook is missing: {nb_path}"
    raw = nb_path.read_text()
    assert "scorers_required_by_capabilities()" in raw
    # No literal list of scorer names assigned to SCORERS.
    assert not re.search(r"SCORERS = \[\\n", raw), "SCORERS is hand-listed again"


def test_missing_scorer_map_names_the_capabilities_that_go_untested() -> None:
    """A scorer left out of an evaluation leaves every capability it feeds untested,
    and the missing-scorer map names each of them, so the gap is visible."""
    for left_out in scorers_required_by_capabilities():
        others = [s for s in scorers_required_by_capabilities() if s != left_out]
        missing = capabilities_missing_scorer(others)
        assert set(missing) == {left_out}
        assert set(missing[left_out]) == {
            d.id for d in capability_definitions() if any(s.scorer == left_out for s in d.sources)
        }


def test_reading_the_capabilities_loads_no_probe() -> None:
    """The report reads every capability without importing a probe, so a probe that fails
    to import still leaves its capability rows, and every other, in the report."""
    code = (
        "import sys\n"
        "from usersim.taxonomy.capabilities import axis_description, capability_definitions\n"
        "capability_definitions(); axis_description('finance.tool_selection_rate')\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('usersim.engine.probes.') and m.count('.') > 3)\n"
        "print([m for m in loaded if not m.endswith('.capabilities')], len(loaded))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"[] {len(builtin_modules('capabilities'))}", result.stdout
