# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Cross-probe invariant tests.

Every probe in the framework must honour a small contract that the
unified ``ConversationLoop`` + ``TrajectoryEvaluatorGenerator``
machinery rely on. This file consolidates the cross-cutting checks
into one parametrized test surface so per-probe files can focus on
probe-specific behavior.

What is asserted here (vs in per-probe test files):

- **Registry completeness** — the canonical 12 probes are registered
  with the canonical labels. Catches the regression where a rename
  drops one or where the bootstrap silently fails to import a
  module.
- **Module constants** — every registered probe's metadata module
  exposes ``PROBE_FAMILY`` / ``PROMPT_VERSION`` / ``PROBE_VARIANTS``
  with the right shapes. Mirrors the smoke-check #8 surface as a
  fast unit test, so a CI-time regression doesn't require a full
  smoke run to surface.
- **ProbeAdapter conformance** — every registered probe class
  satisfies the ``ProbeAdapter`` Protocol's required surface.
  Catches the regression where a method is renamed on the Protocol
  but a probe forgot to update.
- **should_succeed invariant** — every probe's sim-side guardrail
  returns False on a minimally-empty state and True when the probe-
  specific side-channel key is present. Catches the regression
  where a probe stops emitting its primary join key but still
  produces "successful" trajectories.
- **BaseProbe.after_assistant_turn default contract** — the default
  (un-overridden) ``after_assistant_turn`` appends to
  ``state.messages`` and returns the appended content. The
  ``ConversationLoop`` relies on this equality (it pushes the
  return value into ``state.user_history`` while assuming the same
  text already lives at ``state.messages[-1]["content"]``).

What is NOT asserted here (and where it lives instead):

- Per-probe end-to-end behavior (verbatim turn-1 shape, two-turn
  follow-up, placeholder warnings, bank-load failures): the
  per-probe test files (``test_sov_ai_facts.py``,
  ``test_safety_agentic.py``, etc.) own these.
- Substrate behavior (mixins, ``BankBackedProbe``,
  ``LocalePromptPack``, ``@register_probe`` decorator semantics):
  ``test_probes_substrate.py`` owns these.
- ``ConversationLoop`` mechanics (fourth-wall prefilter, gate
  retries, telemetry, on_followup_failure consultation):
  ``test_simulation.py`` owns these.
- Smoke-test #8 (full integration check, slower):
  ``cli/smoke.py::_check_probe_registry`` and ``tests/test_smoke.py``.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

import pytest

import usersim.engine.generator  # noqa: F401 — bootstraps the registry
from usersim.engine.core.outcomes import OutcomeBuilder
from usersim.engine.core.probes import (
    _PROBE_REGISTRY,
    BaseProbe,
    known_probes,
    resolve_probe,
)
from usersim.engine.core.simulation import ConversationState

# The canonical 12 probe labels the framework ships. A regression
# test: if any rename drops a label or a new probe is added without
# updating the canonical roster, this test fails first.
EXPECTED_PROBES: tuple[str, ...] = (
    "general_open_ended",
    "general_educational",
    "tool_calling",
    "sov_ai_facts",
    "sov_ai_dynamic",
    "sov_ai_multilingual_parity",
    "safety_chat_pressure",
    "safety_agentic",
    "financial_services",
    # health_disclosure family (shared base, per-client labels)
    "health_therapy_disclosure",
    "health_triage_disclosure",
    "health_decision_support_disclosure",
    "health_general_disclosure",
)


# Per-probe should_succeed contract. Each entry maps the probe label
# to the (metadata_key, value) pair that makes the sim-side guardrail
# pass. Probes whose ``should_succeed`` defers to ``BaseProbe`` (the
# always-True default — currently the two general probes) carry None
# for both, and the test asserts True regardless of metadata.
SHOULD_SUCCEED_CONTRACT: dict[str, tuple[str | None, Any]] = {
    "general_open_ended": (None, None),
    "general_educational": (None, None),
    "tool_calling": ("tools_called", [{"name": "x"}]),
    "sov_ai_facts": ("facts_probed", ["F-001"]),
    "sov_ai_dynamic": ("probing_categories_explored", ["category-1"]),
    "sov_ai_multilingual_parity": ("query_id", "Q-001"),
    "safety_chat_pressure": ("target_request_id", "T-001"),
    "safety_agentic": ("action_request_id", "AR-001"),
    "financial_services": ("finance_task_id", "TPL-DISPUTE-001"),
    # health_disclosure family — BaseProbe-only probes that inherit the
    # always-True should_succeed (like the general probes).
    "health_therapy_disclosure": (None, None),
    "health_triage_disclosure": (None, None),
    "health_decision_support_disclosure": (None, None),
    "health_general_disclosure": (None, None),
}


def _resolve_target_class(probe_label: str) -> type:
    """Return the registered ``BaseProbe`` subclass for a probe.

    The registry maps every label directly to its ``BaseProbe``
    subclass. Kept as a function (rather than inlining
    ``resolve_probe``) so future invariant tests have one place to
    add probe-class normalization if it's ever needed.
    """
    return resolve_probe(probe_label)


def _make_state(metadata: dict[str, Any] | None = None) -> ConversationState:
    """Build a minimally-empty ConversationState for invariant checks."""
    state = ConversationState(outcome=OutcomeBuilder())
    if metadata:
        state.metadata.update(metadata)
    return state


def _resolve_should_succeed(probe_label: str) -> Callable[[ConversationState], bool]:
    """Return the canonical ``should_succeed`` callable for a probe.

    For the 5 BankBackedProbe-derived probes, the probe module
    exposes a top-level ``should_succeed(state) -> bool`` function
    that the class method delegates to (preserved so existing test
    surfaces can call this directly without an instance).

    For the other probes (general probes inheriting BaseProbe's
    default, plus tool_calling), we cannot cheaply construct an
    instance — the constructors validate ``cfg.tools_column`` /
    ``profile["tech_literacy"]`` etc. and would require a richer
    fixture than this invariant test should carry. Instead we
    invoke the unbound class method with ``self=None``: every
    probe's ``should_succeed`` only reads from ``state`` (the
    BaseProbe default returns True unconditionally; tool_calling
    reads ``state.metadata.get("tools_called", [])``), so the
    `self` argument is effectively unused. This keeps the
    invariant test free of probe-specific construction-fixture
    leakage.
    """
    target_cls = _resolve_target_class(probe_label)
    module = sys.modules[target_cls.__module__]
    if hasattr(module, "should_succeed"):
        return module.should_succeed  # type: ignore[no-any-return]
    method = target_cls.should_succeed
    return lambda state: method(None, state)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Registry completeness
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    """Every canonical probe label is registered after the bootstrap."""

    def test_known_probes_matches_expected_roster(self) -> None:
        registered = set(known_probes())
        expected = set(EXPECTED_PROBES)
        missing = expected - registered
        extra = registered - expected
        assert not missing, (
            f"expected probes missing from registry: {sorted(missing)}; a rename or import failure dropped them"
        )
        assert not extra, f"unexpected probes in registry: {sorted(extra)}; update EXPECTED_PROBES if intentional"

    def test_registry_is_non_empty(self) -> None:
        assert len(_PROBE_REGISTRY) >= len(EXPECTED_PROBES)

    @pytest.mark.parametrize("label", EXPECTED_PROBES)
    def test_each_probe_resolves_to_a_class(self, label: str) -> None:
        cls = resolve_probe(label)
        assert isinstance(cls, type)
        assert issubclass(cls, BaseProbe)


# ---------------------------------------------------------------------------
# 2. Module constants — fast unit-test mirror of smoke check #8
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Each registered probe's module exposes well-formed metadata constants."""

    @pytest.mark.parametrize("label", EXPECTED_PROBES)
    def test_probe_family_is_non_empty_string(self, label: str) -> None:
        cls = resolve_probe(label)
        module = sys.modules[cls.__module__]
        family = getattr(module, "PROBE_FAMILY", None)
        assert isinstance(family, str) and family, f"{label}: PROBE_FAMILY missing or non-string"

    @pytest.mark.parametrize("label", EXPECTED_PROBES)
    def test_prompt_version_is_non_empty_string(self, label: str) -> None:
        cls = resolve_probe(label)
        module = sys.modules[cls.__module__]
        prompt_version = getattr(module, "PROMPT_VERSION", None)
        assert isinstance(prompt_version, str) and prompt_version, f"{label}: PROMPT_VERSION missing or non-string"

    @pytest.mark.parametrize("label", EXPECTED_PROBES)
    def test_probe_variants_is_non_empty_sequence(self, label: str) -> None:
        cls = resolve_probe(label)
        module = sys.modules[cls.__module__]
        variants = getattr(module, "PROBE_VARIANTS", None)
        assert isinstance(variants, (list, tuple)) and variants, f"{label}: PROBE_VARIANTS missing or empty"
        assert all(isinstance(v, str) and v for v in variants), (
            f"{label}: PROBE_VARIANTS contains non-string or empty entries: {variants!r}"
        )


# ---------------------------------------------------------------------------
# 3. ProbeAdapter Protocol conformance
# ---------------------------------------------------------------------------


# The 8 required ProbeAdapter hooks every probe class must define
# (per the Protocol declared in core/probes.py). We assert presence
# at the class level — a runtime ``isinstance(instance, ProbeAdapter)``
# check would pass via duck-typing on BaseProbe defaults; we want to
# catch the regression where a method is renamed on the Protocol and
# a probe class silently keeps the old name.
PROBE_ADAPTER_REQUIRED_METHODS: tuple[str, ...] = (
    "get_user_system_prompt",
    "get_assistant_system_prompt",
    "get_tools_for_assistant",
    "format_gate_prompt",
    "format_assistant_judge_prompt",
    "after_assistant_turn",
    "build_result_extras",
)


class TestProbeAdapterConformance:
    """Every registered probe class satisfies the Protocol surface."""

    @pytest.mark.parametrize("label", EXPECTED_PROBES)
    @pytest.mark.parametrize("method", PROBE_ADAPTER_REQUIRED_METHODS)
    def test_method_present_on_class(self, label: str, method: str) -> None:
        target_cls = _resolve_target_class(label)
        attr = getattr(target_cls, method, None)
        assert attr is not None, f"{label}: required ProbeAdapter method {method!r} missing on {target_cls.__name__}"
        assert callable(attr), f"{label}: ProbeAdapter method {method!r} is not callable on {target_cls.__name__}"

    @pytest.mark.parametrize("label", EXPECTED_PROBES)
    def test_label_attribute_matches_registry(self, label: str) -> None:
        target_cls = _resolve_target_class(label)
        assert getattr(target_cls, "label", None) == label, (
            f"{label}: target class {target_cls.__name__} has "
            f"label={getattr(target_cls, 'label', None)!r}; expected {label!r}"
        )

    def test_protocol_runtime_check_on_baseprobe(self) -> None:
        # BaseProbe alone (without subclass overrides for the abstract
        # methods) does not satisfy the Protocol because the abstract
        # methods raise NotImplementedError; but the Protocol only
        # checks attribute presence (since runtime_checkable Protocols
        # check structural shape, not implementation). This sanity-
        # check ensures the Protocol decorator itself is wired.
        from usersim.engine.core.probes import ProbeAdapter as _PA

        # Protocol is runtime-checkable; isinstance dispatches on the
        # presence of the required methods.
        assert hasattr(_PA, "_is_runtime_protocol") or hasattr(_PA, "_is_protocol")


# ---------------------------------------------------------------------------
# 4. should_succeed invariant — five overridden probes + three defaults
# ---------------------------------------------------------------------------


class TestShouldSucceedInvariant:
    """Every probe's sim-side guardrail honours a minimal contract.

    For probes whose ``should_succeed`` is overridden (5 of 8), the
    probe-specific side-channel key is required for True. For the
    three probes that inherit ``BaseProbe.should_succeed`` (open_ended,
    educational — and historically tool_calling, but tool_calling
    overrides), the default returns True regardless of metadata.
    """

    @pytest.mark.parametrize("label", EXPECTED_PROBES)
    def test_returns_bool(self, label: str) -> None:
        fn = _resolve_should_succeed(label)
        result = fn(_make_state())
        assert isinstance(result, bool), f"{label}: should_succeed returned {type(result).__name__}; expected bool"

    @pytest.mark.parametrize("label", EXPECTED_PROBES)
    def test_minimal_empty_state_returns_expected(self, label: str) -> None:
        key, value = SHOULD_SUCCEED_CONTRACT[label]
        fn = _resolve_should_succeed(label)
        result = fn(_make_state())
        if key is None:
            # Default-True probes (general probes inheriting BaseProbe).
            assert result is True, (
                f"{label}: should_succeed on empty state returned False; BaseProbe default should return True"
            )
        else:
            assert result is False, (
                f"{label}: should_succeed on empty state returned True; "
                f"expected False until metadata[{key!r}] is populated"
            )

    @pytest.mark.parametrize("label", EXPECTED_PROBES)
    def test_populated_side_channel_returns_true(self, label: str) -> None:
        key, value = SHOULD_SUCCEED_CONTRACT[label]
        if key is None:
            # Default-True probes — already covered above.
            return
        fn = _resolve_should_succeed(label)
        result = fn(_make_state(metadata={key: value}))
        assert result is True, f"{label}: should_succeed with metadata[{key!r}]={value!r} returned False; expected True"


# ---------------------------------------------------------------------------
# 5. BaseProbe.after_assistant_turn default contract
# ---------------------------------------------------------------------------


class _MinimalBaseProbe(BaseProbe):
    """Smallest valid BaseProbe subclass for unit-testing defaults."""

    label = "_test_minimal_baseprobe"

    def get_user_system_prompt(self) -> str:
        return ""

    def get_assistant_system_prompt(self) -> str:
        return ""


class TestBaseProbeAfterAssistantTurnContract:
    """The default ``after_assistant_turn`` honours the loop's invariant.

    ``ConversationLoop.run`` pushes the ``after_assistant_turn``
    return value into ``state.user_history`` while assuming the same
    text already lives at ``state.messages[-1]["content"]``. This
    invariant is what lets the loop drive both surfaces in lock-step
    without a separate "synthesis" channel. The default appends an
    assistant message and returns its content; subclasses that
    override (tool_calling, safety_agentic) own their own append.
    """

    def _probe(self) -> _MinimalBaseProbe:
        return _MinimalBaseProbe(
            persona={"first_name": "T", "last_name": "U"},
            locale="en_US",
            language="English",
            models={},
        )

    def test_appends_assistant_message_and_returns_content(self) -> None:
        probe = self._probe()
        state = _make_state()
        before = list(state.messages)
        returned = probe.after_assistant_turn(
            models={},
            state=state,
            assistant_response={"content": "Hello, world."},
            cfg=None,
        )
        assert returned == "Hello, world."
        assert state.messages == before + [
            {"role": "assistant", "content": "Hello, world."},
        ]

    def test_returned_text_equals_last_message_content(self) -> None:
        probe = self._probe()
        state = _make_state()
        returned = probe.after_assistant_turn(
            models={},
            state=state,
            assistant_response={"content": "Some reply."},
            cfg=None,
        )
        assert returned == state.messages[-1]["content"]

    def test_handles_empty_content(self) -> None:
        probe = self._probe()
        state = _make_state()
        returned = probe.after_assistant_turn(
            models={},
            state=state,
            assistant_response={"content": ""},
            cfg=None,
        )
        assert returned == ""
        assert state.messages[-1] == {"role": "assistant", "content": ""}

    def test_handles_non_dict_response_gracefully(self) -> None:
        probe = self._probe()
        state = _make_state()
        returned = probe.after_assistant_turn(
            models={},
            state=state,
            assistant_response="not a dict",  # type: ignore[arg-type]
            cfg=None,
        )
        assert returned == ""
        assert state.messages[-1] == {"role": "assistant", "content": ""}
