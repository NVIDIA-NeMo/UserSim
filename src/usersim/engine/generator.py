# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Unified ConversationSimulatorGenerator -- dispatches by probe_type."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
import time
from typing import Any

from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell,
    ColumnGeneratorWithModelRegistry,
)

from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core._assets import set_runtime_assets_dir
from usersim.engine.core.behavioral import (
    compute_behavioral_profile,
    compute_disclosure_style,
    compute_user_interaction_style,
    get_conversation_language,
)
from usersim.engine.core.identity import (
    persona_uuid as compute_persona_uuid,
)
from usersim.engine.core.identity import (
    resolve_model_name,
)
from usersim.engine.core.identity import (
    trajectory_id as compute_trajectory_id,
)
from usersim.engine.core.llm import (
    ContextWindowError,
    flush_debug_log,
    get_call_stats,
    get_record_stats,
    record_finished,
    set_conversation_id,
    set_current_outcome_builder,
)
from usersim.engine.core.locale import persona_dataset_locale
from usersim.engine.core.outcomes import OutcomeBuilder, Provenance
from usersim.engine.core.persona import religion_language_context
from usersim.engine.core.probes import BankLoadError, resolve_probe
from usersim.engine.core.provenance import (
    get_code_sha,
    get_nemotron_personas_version,
)

logger = logging.getLogger("usersim.engine")

MODEL_USER = "user_model"
MODEL_ASSISTANT = "assistant_model"
MODEL_API_RESPONSE = "api_response_model"
MODEL_JUDGE = "judge_model"
MODEL_SUMMARY = "summary_model"
MODEL_ALIASES = [
    MODEL_USER,
    MODEL_ASSISTANT,
    MODEL_API_RESPONSE,
    MODEL_JUDGE,
    MODEL_SUMMARY,
]


# ---------------------------------------------------------------------------
# Probe registry — class-based, populated via the substrate
# ---------------------------------------------------------------------------
#
# Each probe is a ``BaseProbe`` subclass registered in the substrate's
# ``_PROBE_REGISTRY`` (in ``core/probes.py``) via ``@register_probe``
# at import time. The bootstrap function below is the lazy-import
# trigger that ensures every shipped probe module is imported (which
# fires its ``@register_probe`` decorator) before the dispatcher
# starts resolving probe labels.


def _bootstrap_probes() -> None:
    """Trigger import-time ``@register_probe`` for every shipped probe.

    Each imported module's ``@register_probe`` decorator self-registers
    its probe class into ``_PROBE_REGISTRY`` and sets the per-module
    ``PROBE_FAMILY`` / ``PROMPT_VERSION`` / ``PROBE_VARIANTS`` constants.
    Idempotent: re-importing is a no-op for already-loaded modules.
    """
    # Imports trigger the decorators; the imports themselves go unused.
    import usersim.engine.probes.financial_services.generator  # noqa: F401
    import usersim.engine.probes.general_educational.generator  # noqa: F401
    import usersim.engine.probes.general_open_ended.generator  # noqa: F401
    import usersim.engine.probes.health_disclosure.decision_support  # noqa: F401
    import usersim.engine.probes.health_disclosure.general  # noqa: F401
    import usersim.engine.probes.health_disclosure.therapy  # noqa: F401
    import usersim.engine.probes.health_disclosure.triage  # noqa: F401
    import usersim.engine.probes.safety_agentic.generator  # noqa: F401
    import usersim.engine.probes.safety_chat_pressure.generator  # noqa: F401
    import usersim.engine.probes.sov_ai_dynamic.generator  # noqa: F401
    import usersim.engine.probes.sov_ai_facts.generator  # noqa: F401
    import usersim.engine.probes.sov_ai_multilingual_parity.generator  # noqa: F401
    import usersim.engine.probes.tool_calling.generator  # noqa: F401


_bootstrap_probes()


def _probe_variant(data: dict, probe_module: Any, probe_type: str) -> str:
    """Resolve the per-trajectory ``probe_variant`` label.

    Priority:
      1. Explicit ``probe_variant`` column on the row, if present.
      2. Probe-module default (first entry in ``PROBE_VARIANTS``).
      3. Fallback to ``"default"``.

    ``probe_variant`` is a probe-specific discriminator that the
    evaluator reads for stratification and the training-extraction
    workstream reads for matched-pair joins. Today the general probes
    declare a single variant (``"default"``); richer probes
    (``safety_chat_pressure``, ``safety_agentic``) declare
    multiple.
    """
    explicit = data.get("probe_variant")
    if isinstance(explicit, str) and explicit:
        return explicit
    variants = getattr(probe_module, "PROBE_VARIANTS", None)
    if variants and isinstance(variants, (list, tuple)) and variants:
        return str(variants[0])
    return "default"


def _build_provenance(probe_module: Any) -> Provenance:
    """Assemble per-trajectory provenance from module metadata and environment."""
    return Provenance(
        nemotron_personas_version=get_nemotron_personas_version(),
        scenario_prompt_version=getattr(probe_module, "PROMPT_VERSION", None),
        code_sha=get_code_sha(),
        bank_version={},  # populated by probes that consume curated fact banks
    )


class ConversationSimulatorGenerator(
    ColumnGeneratorCellByCell[ConversationSimulatorConfig],
    ColumnGeneratorWithModelRegistry[ConversationSimulatorConfig],
):
    """Unified generator that dispatches to probe-specific simulation modules."""

    def generate(self, data: dict) -> dict:
        cfg = self.config
        set_runtime_assets_dir(cfg.assets_dir)
        models = {}
        for alias in MODEL_ALIASES:
            try:
                models[alias] = self.get_model(alias)
            except Exception:
                if alias == MODEL_SUMMARY:
                    logger.warning(
                        "  |-- summary_model not found in model registry, falling back to user_model for summaries"
                    )
                    models[alias] = self.get_model(MODEL_USER)
                else:
                    raise

        # The embedding model is NOT a chat alias, so it's absent from
        # MODEL_ALIASES -- but the financial_services probe needs its facade in
        # the models dict to embed kb_search queries for dense retrieval. Add it
        # best-effort: if it isn't configured/resolvable, dense retrieval falls
        # back to lexical (the probe logs a loud warning), never a hard failure.
        embed_alias = getattr(self.config, "finance_embedding_model_alias", None)
        if embed_alias and embed_alias not in models:
            try:
                models[embed_alias] = self.get_model(embed_alias)
            except Exception:
                logger.debug(
                    "  |-- embedding model %r not in registry; finance dense retrieval will fall back to lexical",
                    embed_alias,
                )

        # Set per-call logging level based on verbosity
        if cfg.verbosity >= 2:
            logging.getLogger("usersim.engine").setLevel(logging.DEBUG)

        t_record_start = time.monotonic()

        # Parse persona
        raw_persona = data[cfg.persona_column]
        persona = raw_persona if isinstance(raw_persona, dict) else json.loads(raw_persona)
        # The raw persona column is dropped from trajectory output. Preserve the
        # normalized protected fields that now influence the user-agent prompt so
        # a stored row remains auditable without reloading the source dataset.
        data["persona_religion_language_context"] = json.dumps(
            religion_language_context(persona),
            ensure_ascii=False,
        )

        # Content-hashed persona identity. Stable across runs; observed
        # (not assigned). Replay from a prior trajectory parquet produces
        # the same UUID under unchanged content.
        persona_uuid_value = compute_persona_uuid(persona)
        data["persona_uuid"] = persona_uuid_value

        # Locale and language
        locale = cfg.locale

        # Compute behavioral profile (locale-aware for education/occupation
        # mapping). Persona attributes come from the persona *dataset* locale,
        # not the conversation locale, so India language-variants
        # (e.g. ta_Taml_IN drawing en_IN personas) resolve the education
        # ordinal against en_IN's exact vocabulary rather than the imperfect
        # keyword fallback. Non-variant locales are unchanged.
        profile = compute_behavioral_profile(persona, locale=persona_dataset_locale(locale))
        data["behavioral_profile"] = json.dumps(profile, ensure_ascii=False)

        # Stable hash for reproducible per-persona seeding (deterministic across
        # Python sessions, unlike the built-in hash() which is salted).
        persona_json = json.dumps(persona, sort_keys=True, default=str)
        persona_hash = int(hashlib.sha256(persona_json.encode()).hexdigest(), 16) & 0xFFFFFFFF
        disclosure_style = compute_disclosure_style(
            persona_seed=persona_hash,
            incremental_ratio=cfg.incremental_disclosure_ratio,
        )
        interaction_style = compute_user_interaction_style(profile)
        use_grounded_query = random.Random(persona_hash + 1).random() < cfg.persona_grounding_ratio
        data["disclosure_style"] = disclosure_style
        data["user_interaction_style"] = interaction_style
        data["persona_grounding"] = use_grounded_query
        language = get_conversation_language(locale)
        data["locale"] = locale
        data["conversation_language"] = language

        # Read probe type and dispatch
        probe_type = data[cfg.probe_type_column]
        persona_name = f"{persona.get('first_name', '?')} {persona.get('last_name', '?')}"
        logger.info(
            f"  |-- Starting {probe_type} sim for {persona_name} "
            f"[{interaction_style}, {disclosure_style}] (max_turns={cfg.max_turns})"
        )

        try:
            probe_cls = resolve_probe(probe_type)
        except KeyError as e:
            raise ValueError(str(e)) from e
        probe_module = sys.modules.get(probe_cls.__module__)

        # Build per-trajectory provenance + resolve probe_variant.
        # These land on the trajectory for replay correctness (persona
        # dataset / prompt version skew detection) and for stratified
        # reporting.
        provenance = _build_provenance(probe_module)
        probe_variant = _probe_variant(data, probe_module, probe_type)
        probe_family = getattr(probe_module, "PROBE_FAMILY", probe_type)
        data["probe_family"] = probe_family
        data["probe_variant"] = probe_variant

        # Deterministic trajectory_id. Composed from persona, probe
        # family + variant, probe seed (cfg.random_seed if set, else
        # a default), resolved model identities for each role (so
        # model-version drift produces a different id), and the
        # probe module's PROMPT_VERSION. Same inputs -> same id;
        # the simulator-driver uses this for resumable / idempotent
        # re-runs via core.idempotency.
        prompt_version = getattr(probe_module, "PROMPT_VERSION", "v1.0")
        user_model_name = resolve_model_name(models.get(MODEL_USER), MODEL_USER)
        assistant_model_name = resolve_model_name(models.get(MODEL_ASSISTANT), MODEL_ASSISTANT)
        # ``locale`` is a load-bearing content determinant: multiple
        # conversation locales can now draw from the SAME persona dataset
        # (India language-variants all sample en_IN personas), so without
        # the locale salt two languages would collide on persona_uuid +
        # probe + models + prompt_version and the idempotency layer would
        # drop the second as a duplicate. Salting keeps each locale's row
        # distinct. (One-time change to the id formula for all locales.)
        traj_id = compute_trajectory_id(
            persona_uuid=persona_uuid_value,
            probe_family=probe_family,
            probe_variant=probe_variant,
            scenario_seed=cfg.random_seed,
            user_model=user_model_name,
            assistant_model=assistant_model_name,
            prompt_version=prompt_version,
            extra_keys=[("locale", locale)],
        )
        data["trajectory_id"] = traj_id
        # Tag the conversation log with the trajectory_id too — much
        # easier to grep through debug_llm.jsonl when chasing a
        # specific replayed row.
        set_conversation_id(f"{traj_id} | {persona_name} | {probe_type}")

        # Install the thread-local outcome-builder hook so call_llm()
        # invocations inside this trajectory feed per-model tokens /
        # calls / latencies into simulation_outcome. Cleared in the
        # finally block so we never leak the builder across rows.
        row_outcome_builder = OutcomeBuilder(provenance=provenance)
        set_current_outcome_builder(row_outcome_builder)

        t_sim_start = time.monotonic()
        try:
            try:
                probe = probe_cls(
                    persona=persona,
                    locale=locale,
                    language=language,
                    models=models,
                    cfg=cfg,
                    provenance=provenance,
                    profile=profile,
                    data=data,
                    outcome_builder=row_outcome_builder,
                )
                result = probe.run_dispatch(
                    models=models,
                    data=data,
                    cfg=cfg,
                )
            except ContextWindowError as e:
                logger.warning(
                    "  |-- Context window failure for %s: %s",
                    persona_name,
                    e,
                )
                result = _make_failed_result(
                    str(e),
                    provenance=provenance,
                    model_alias=e.alias,
                )
            except (
                KeyError,
                ValueError,
                json.JSONDecodeError,
                BankLoadError,
            ) as e:
                # ValueError covers per-probe construction errors
                # (every <Probe>Error subclasses ValueError); BankLoadError
                # is the substrate's bank-load failure exception. Both
                # produce a structured SCENARIO_ABORTED outcome rather
                # than crashing the row.
                logger.warning(f"  |-- Simulation failed for {persona_name}: {e}")
                result = _make_aborted_result(str(e), provenance=provenance)
        finally:
            set_current_outcome_builder(None)
        t_sim_elapsed = time.monotonic() - t_sim_start

        data.update(result)
        status = "OK" if data.get("conversation_status") else "FAIL"
        n_turns = data.get("num_turns", 0)
        n_tools = data.get("num_tool_calls", 0)
        tools_tag = f", {n_tools} tool calls" if n_tools else ""
        logger.info(
            f"  |-- Finished sim for {persona_name}: {status}, {n_turns} turns{tools_tag} ({t_sim_elapsed:.1f}s)"
        )

        t_total = time.monotonic() - t_record_start
        logger.info(f"  |-- Total for {persona_name}: {t_total:.1f}s (sim={t_sim_elapsed:.1f}s)")
        record_finished(t_total)

        if cfg.verbosity >= 2:
            _log_running_stats()

        flush_debug_log()
        return data


def _log_running_stats() -> None:
    """Print running aggregate stats across all completed records."""
    rec = get_record_stats()
    if not rec:
        return
    model_stats = get_call_stats()
    if not model_stats:
        return

    lines = [
        f"  |-- 📈 Running stats ({rec['records']} records): "
        f"avg={rec['avg_s']:.1f}s, min={rec['min_s']:.1f}s, "
        f"max={rec['max_s']:.1f}s, total={rec['total_s']:.0f}s"
    ]
    for alias in MODEL_ALIASES:
        s = model_stats.get(alias)
        if not s or s["calls"] == 0:
            continue
        avg_t = s["total_s"] / s["calls"]
        avg_chars = s["total_chars"] / s["calls"]
        avg_words = s["total_words"] / s["calls"]
        lines.append(
            f"  |--   {alias}: {s['calls']} calls, avg {avg_t:.1f}s, avg {avg_chars:.0f} chars / {avg_words:.0f} words"
        )
    for line in lines:
        logger.info(line)


def _make_failed_result(
    reason: str,
    provenance: Provenance | None = None,
    *,
    model_alias: str | None = None,
) -> dict:
    """Produce side-effect columns for a failed simulation.

    If provenance is provided, the synthesized ``simulation_outcome``
    carries it so the downstream reporting layer can attribute even
    infrastructure failures correctly.
    """
    from usersim.engine.core.outcomes import (
        FailureAttribution,
        FailureClass,
        OutcomeBuilder,
        OutcomeStatus,
    )
    from usersim.engine.core.simulation import make_failed

    builder = OutcomeBuilder(provenance=provenance or Provenance())
    attribution = {
        MODEL_USER: FailureAttribution.USER_MODEL,
        MODEL_ASSISTANT: FailureAttribution.ASSISTANT_MODEL,
        MODEL_API_RESPONSE: FailureAttribution.API_RESPONSE_MODEL,
        MODEL_JUDGE: FailureAttribution.JUDGE_MODEL,
    }.get(model_alias, FailureAttribution.INFRASTRUCTURE)
    outcome = builder.finalize(
        status=OutcomeStatus.FAILED,
        failure_class=FailureClass.INFRASTRUCTURE_ERROR,
        failure_attribution=attribution,
        failure_detail=reason,
    )
    return make_failed(reason, outcome=outcome, traces=[])


def _make_aborted_result(
    reason: str,
    provenance: Provenance | None = None,
) -> dict:
    """Produce side-effect columns for a probe-aborted simulation.

    Distinct from ``_make_failed_result`` (which attributes to
    INFRASTRUCTURE): a probe abort is a SCENARIO_ABORTED outcome
    attributed to SCENARIO_LOGIC — the asset bank failed to load,
    no resolvable task for the persona, prompt pack missing for the
    locale, etc. The matching probe shim raises one of these errors
    at construction time; the dispatcher catches and surfaces a
    structured outcome rather than crashing the row.
    """
    from usersim.engine.core.outcomes import (
        FailureAttribution,
        FailureClass,
        OutcomeBuilder,
        OutcomeStatus,
    )
    from usersim.engine.core.simulation import make_failed

    builder = OutcomeBuilder(provenance=provenance or Provenance())
    outcome = builder.finalize(
        status=OutcomeStatus.FAILED,
        failure_class=FailureClass.SCENARIO_ABORTED,
        failure_attribution=FailureAttribution.SCENARIO_LOGIC,
        failure_detail=reason,
    )
    return make_failed(reason, outcome=outcome, traces=[])
