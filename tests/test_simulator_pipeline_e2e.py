# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""End-to-end wiring test: every shipped probe survives construction +
single-turn dispatch when driven through ``cli/_pipeline.build_simulator_config_builder``.

This is the test that would have caught the
``cfg.tools_column not configured`` regression where every
``tool_calling`` trajectory failed at probe construction. The
existing per-probe tests (``tests/probes/test_*.py``) all use
synthetic ``ConversationSimulatorConfig`` fixtures that hard-code the
correct ``tools_column`` / ``persona_column`` / etc. values; they
therefore never exercise the wiring chain
``cli/simulate.py`` → ``cli/_pipeline.build_simulator_config_builder``
→ ``ConversationSimulatorConfig`` → probe construction.

This test fills that gap. For every probe in the registry it:

1. Computes the toolset-seed path the same way the CLI does
   (``cli._pipeline.resolve_toolset_seed_for_mix``) — so a stale path
   literal there fails this test.
2. Builds the simulator ``DataDesignerConfigBuilder`` via the same
   helper the CLI uses (``cli._pipeline.build_simulator_config_builder``).
3. Extracts the ``ConversationSimulatorConfig`` the helper produced.
4. Constructs the probe class (``resolve_probe(probe_name)(...)``)
   with that cfg and a synthetic persona — the same calling pattern
   ``ConversationSimulatorGenerator.generate`` uses.
5. Runs ``await probe.run_dispatch(...)`` against a mocked ``acall_llm`` so
   no real network calls happen.
6. Asserts the result is **not** a scenario-aborted outcome and
   that at least one assistant turn was produced.

If any of these steps fails, the test surfaces it loud and red at
PR time — before any expensive real-LLM run discovers it. Catches:

- Path-composition regressions in ``cli/_pipeline.py`` (the original
  ``"toolsets"`` literal bug).
- Probe ``__init__`` validators getting stricter without the CLI's
  cfg builder being updated to satisfy them.
- A new probe added to the registry that fails to construct against
  the wiring the CLI produces.
- Renames of ``ConversationSimulatorConfig`` fields where the CLI's
  builder forgets to populate the new field name.

Cost: < 5 seconds for the full parametrized sweep across 8 probes,
all locally with mocked LLMs and the shipped fixture personas.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Importing the plugin generator triggers the probe bootstrap, which
# self-registers every probe class.
import usersim.engine.generator  # noqa: F401
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.outcomes import OutcomeBuilder, Provenance
from usersim.engine.core.probes import known_probes, resolve_probe

# ---------------------------------------------------------------------------
# Synthetic personas keyed by locale
# ---------------------------------------------------------------------------
#
# Per-locale because some probes' construction (notably ``tool_calling``)
# reads behavioral fields that are locale-aware. We pick one persona
# per locale that satisfies the persona-tag matching for every shipped
# probe (i.e. lives in the locale, has fields the per-locale fact /
# query / pressure / agentic banks expect).

_PERSONAS: dict[str, dict[str, Any]] = {
    "en_US": {
        "first_name": "Sarah",
        "last_name": "Johnson",
        "age": 42,
        "city": "Austin",
        "state": "TX",
        "state_abbrev": "TX",
        "education_level": "Bachelor's degree",
        "occupation": "Public school teacher",
    },
    "pt_BR": {
        "first_name": "Mariana",
        "last_name": "Silva",
        "age": 37,
        "city": "São Paulo",
        "state": "SP",
        "state_abbrev": "SP",
        "education_level": "Ensino superior (graduação)",
        "occupation": "Professora de história em escola pública",
    },
}

# Default to en_US for probes whose tests don't need locale-specific
# matching. Using en_US keeps the fact-bank / probing-taxonomy
# dependencies on the most-shipped fixtures and avoids exercising
# locale-specific persona-tag matching at the same time as the wiring
# contract.
_DEFAULT_LOCALE = "en_US"


# ---------------------------------------------------------------------------
# Mocked acall_llm side-effect — alias-dispatching like the per-probe tests
# ---------------------------------------------------------------------------


def _mock_call_llm_side_effect(models, alias, msgs, **kwargs):
    """Return a plausible payload per model alias.

    The shape matches what each call site expects:

    - ``assistant_model``: ``{"role": "assistant", "content": "..."}``
      with non-empty content so the ``ConversationLoop`` doesn't trip
      its empty-response retry path.
    - ``user_model``: ``{"role": "assistant", "content": "..."}``
      (the user-LLM also returns assistant-role payloads — it's
      role-playing, not the user role itself).
    - ``judge_model``: well-formed XML so the judge parser doesn't
      retry (success rating to keep the loop happy).
    - ``summary_model``: ``"no"`` so the early-stop check returns
      "do not stop".
    - ``api_response_model``: a simple JSON-shaped tool response for
      ``tool_calling`` and ``safety_agentic`` mock-environment paths.
    """
    if alias == "assistant_model":
        return {
            "role": "assistant",
            "content": "Sure — happy to help with that.",
        }
    if alias == "user_model":
        return {"role": "assistant", "content": "Thanks, that's helpful."}
    if alias == "judge_model":
        return {
            "role": "assistant",
            "content": ("<explanation>looks fine</explanation>\n<rating>success</rating>"),
        }
    if alias == "summary_model":
        return {"role": "assistant", "content": "no"}
    if alias == "api_response_model":
        return {"role": "tool", "content": '{"status": "ok"}'}
    raise AssertionError(f"unexpected model alias in mock: {alias!r}")


@contextmanager
def _patched_call_llm():
    """Patch ``acall_llm`` at every simulator-side binding site.

    ``acall_llm`` is imported by name into multiple modules at module-
    load time (``from usersim.engine.core.llm import acall_llm``),
    so a single patch on the source module doesn't propagate to
    callers — each binding has to be patched explicitly. The
    simulator-side callers are:

    - ``core/simulation.py`` — the unified ``ConversationLoop``.
    - ``core/judges.py`` — in-sim user-judge gate + assistant-quality
      judge + early-stop checker.
    - ``core/context.py`` — context-compression rolling summary
      generation.
    - ``probes/tool_calling/generator.py`` — probe-internal tool
      arg-generation calls (it doesn't go through ``ConversationLoop``
      for the tool-call branch).
    - ``probes/safety_agentic/generator.py`` — mock-environment
      tool-response synthesis.

    Evaluator-side scorer modules (``evaluator/scorers/*.py``) ALSO
    import ``acall_llm``, but they aren't exercised by the simulator
    pipeline this test covers. Leaving them unpatched keeps the
    surface area focused.
    """
    targets = (
        "usersim.engine.core.simulation.acall_llm",
        "usersim.engine.core.judges.acall_llm",
        "usersim.engine.core.context.acall_llm",
        "usersim.engine.probes.tool_calling.generator.acall_llm",
        "usersim.engine.probes.safety_agentic.generator.acall_llm",
    )
    patches = [patch(t, side_effect=_mock_call_llm_side_effect) for t in targets]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# Build the simulator config the same way the CLI does
# ---------------------------------------------------------------------------


def _cli_built_simulator_config(probe_type: str, locale: str) -> ConversationSimulatorConfig:
    """Build the ``ConversationSimulatorConfig`` for a single-probe
    mix the same way ``cli/simulate.py`` does.

    Goes through ``resolve_toolset_seed_for_mix`` +
    ``build_simulator_config_builder`` so a regression in either
    path-composition site fails this test.
    """
    from usersim.cli._models import default_models_path, load_models_config
    from usersim.cli._pipeline import (
        build_simulator_config_builder,
        resolve_toolset_seed_for_mix,
    )

    assets_dir = (packaged_assets_dir()).resolve()
    models = load_models_config(default_models_path())

    probe_mix = {probe_type: 1.0}
    toolset_seed = resolve_toolset_seed_for_mix(probe_mix, assets_dir)
    _, builder = build_simulator_config_builder(
        locale=locale,
        models=models,
        assets_dir=assets_dir,
        max_turns=1,
        probe_mix=probe_mix,
        toolset_seed_path=toolset_seed,
        random_seed=42,
    )
    cfg = builder.get_column_config("conversation_messages")
    assert isinstance(cfg, ConversationSimulatorConfig), (
        f"expected ConversationSimulatorConfig, got {type(cfg).__name__}"
    )
    return cfg


# ---------------------------------------------------------------------------
# Build the synthetic per-row data dict
# ---------------------------------------------------------------------------


def _synthetic_row_data(
    probe_type: str,
    locale: str,
    cfg: ConversationSimulatorConfig,
) -> dict[str, Any]:
    """Build the per-row ``data`` dict that
    ``ConversationSimulatorGenerator.generate(data)`` consumes.

    Includes the columns the simulator + the probes read:
    ``persona``, ``probe_type``, ``theme`` (always), and
    ``tools`` / ``toolset_name`` (when ``cfg.tools_column`` is set).
    """
    persona = _PERSONAS.get(locale) or _PERSONAS[_DEFAULT_LOCALE]
    data: dict[str, Any] = {
        cfg.persona_column: persona,
        cfg.probe_type_column: probe_type,
        cfg.theme_column: _theme_for(probe_type),
    }
    if cfg.tools_column is not None:
        data[cfg.tools_column] = _toolset_for_test(repo_root_assets())
        data[cfg.toolset_name_column] = "weather_tools"
    return data


def _theme_for(probe_type: str) -> str:
    """Theme-column value the SubcategorySampler would pick.

    For theme-driven probes the value is JSON; for others it is the
    sentinel placeholder ``__not_used__`` (matches the value
    ``cli/_pipeline.py`` uses for non-theme-driven probes)."""
    if probe_type == "tool_calling":
        return '{"type": "Weather & Location Lookup", "description": "Look up current weather", "tool_expected": true}'
    if probe_type == "general_open_ended":
        return '{"type": "general", "description": "casual chat about a topic"}'
    if probe_type == "general_educational":
        return '{"type": "math", "description": "elementary math help"}'
    return "__not_used__"


def repo_root_assets() -> Path:
    return packaged_assets_dir()


def _toolset_for_test(assets_dir: Path) -> list[dict[str, Any]]:
    """Read one row from the shipped ``toolsets_seed.parquet`` to use
    as the synthetic ``tools`` column value. Reads only the first
    row's ``tools`` field — keeps the test small and avoids
    re-implementing toolset authoring inline.
    """
    import pyarrow.parquet as pq

    toolset_path = assets_dir / "tool_calling" / "toolsets_seed.parquet"
    table = pq.read_table(toolset_path).slice(0, 1).to_pylist()
    raw = table[0]["tools"]
    if isinstance(raw, str):
        return json.loads(raw)
    return list(raw)


# ---------------------------------------------------------------------------
# Replicate the relevant slice of ConversationSimulatorGenerator.generate
# ---------------------------------------------------------------------------


async def _drive_one_probe(probe_type: str, locale: str) -> dict[str, Any]:
    """Construct + run one probe end-to-end against synthetic inputs
    and a mocked ``acall_llm``. Returns the result dict the probe
    produces (same shape ``ConversationSimulatorGenerator`` would
    return after ``data.update(result)``).
    """
    from usersim.engine.core.behavioral import (
        compute_behavioral_profile,
        get_conversation_language,
    )
    from usersim.engine.core.llm import set_current_outcome_builder

    cfg = _cli_built_simulator_config(probe_type, locale)
    data = _synthetic_row_data(probe_type, locale, cfg)
    persona = data[cfg.persona_column]
    profile = compute_behavioral_profile(persona, locale=locale)
    language = get_conversation_language(locale)

    probe_cls = resolve_probe(probe_type)
    provenance = Provenance(
        code_sha="test",
        scenario_prompt_version="v1.0",
    )
    outcome_builder = OutcomeBuilder(provenance=provenance)

    # Synthetic model registry: each value is a sentinel object the
    # mocked acall_llm doesn't actually inspect (we patch acall_llm
    # itself, so the model "client" is never used).
    models = {
        alias: object()
        for alias in (
            "user_model",
            "assistant_model",
            "judge_model",
            "summary_model",
            "api_response_model",
        )
    }

    set_current_outcome_builder(outcome_builder)
    try:
        with _patched_call_llm():
            probe = probe_cls(
                persona=persona,
                locale=locale,
                language=language,
                models=models,
                cfg=cfg,
                provenance=provenance,
                profile=profile,
                data=data,
                outcome_builder=outcome_builder,
            )
            result = await probe.run_dispatch(
                models=models,
                data=data,
                cfg=cfg,
            )
    finally:
        set_current_outcome_builder(None)
    return result


# ---------------------------------------------------------------------------
# The actual contract test
# ---------------------------------------------------------------------------


# Locale matrix: pick the locale that satisfies every per-locale
# bank's persona-tag matching. en_US covers most probes; pt_BR is
# the placeholder for sov_ai_facts / sov_ai_dynamic so use it for
# those families to ensure the persona matches the bank's
# ``persona_tags``. (For other probes en_US is fine.)
_PROBE_LOCALES: dict[str, str] = {
    "general_open_ended": "en_US",
    "general_educational": "en_US",
    "tool_calling": "en_US",
    "sov_ai_facts": "pt_BR",
    "sov_ai_dynamic": "pt_BR",
    "sov_ai_multilingual_parity": "en_US",
    "safety_chat_pressure": "en_US",
    "safety_agentic": "en_US",
}


@pytest.mark.parametrize("probe_type", sorted(known_probes()))
async def test_probe_constructs_and_dispatches_through_cli_wiring(
    probe_type: str,
) -> None:
    """For every shipped probe: the wiring
    ``cli._pipeline.build_simulator_config_builder`` →
    ``ConversationSimulatorConfig`` → probe construction →
    ``run_dispatch`` produces a non-aborted outcome with at least
    one assistant turn.

    The exact behavioural outcome (``ok`` vs.
    ``completed_with_warnings``) is not pinned — only the structural
    contract that the simulator successfully returned a trajectory
    with a sensible status that ISN'T ``failed/scenario_aborted``.
    Catches the bug class where the CLI's wiring produces a cfg the
    probe rejects at construction (the original ``cfg.tools_column
    not configured`` regression).
    """
    locale = _PROBE_LOCALES.get(probe_type, _DEFAULT_LOCALE)
    result = await _drive_one_probe(probe_type, locale)

    # Decode the simulation_outcome envelope.
    outcome = json.loads(result["simulation_outcome"])

    assert outcome["status"] != "failed" or outcome["failure_class"] != "scenario_aborted", (
        f"{probe_type}: probe construction or run_dispatch produced "
        f"scenario_aborted — likely a wiring issue between cli/_pipeline.py "
        f"and ConversationSimulatorConfig.\n"
        f"  failure_class:      {outcome.get('failure_class')!r}\n"
        f"  failure_attribution:{outcome.get('failure_attribution')!r}\n"
        f"  failure_detail:     {outcome.get('failure_detail')!r}\n"
        f"  status:             {outcome.get('status')!r}"
    )

    # Loop-level invariants every healthy run honours.
    assert result["num_turns"] >= 1, (
        f"{probe_type}: expected at least 1 user turn; got num_turns={result.get('num_turns')!r}"
    )
    assert "trajectory_id" not in result, (
        # The probe layer doesn't set trajectory_id — that lives in
        # the outer ConversationSimulatorGenerator. Pin to catch a
        # responsibility-leak regression.
        "probe set trajectory_id; that's the generator's job"
    )
