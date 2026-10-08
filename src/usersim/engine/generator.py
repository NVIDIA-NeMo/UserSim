# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified ConversationSimulatorGenerator -- dispatches by probe_type."""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace
from typing import Any, Mapping

from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell,
    ColumnGeneratorWithModelRegistry,
)

from usersim.engine.config import (
    MODEL_ALIASES,
    MODEL_API_RESPONSE,
    MODEL_ASSISTANT,
    MODEL_JUDGE,
    MODEL_SUMMARY,
    MODEL_USER,
    ConversationSimulatorConfig,
)
from usersim.engine.core._assets import reset_runtime_assets_dir, set_runtime_assets_dir
from usersim.engine.core.episode_input import EpisodeConstructionError, construct_probe_episode
from usersim.engine.core.llm import (
    ContextWindowError,
    flush_debug_log,
    get_call_stats,
    get_record_stats,
    record_finished,
    set_conversation_id,
    set_current_outcome_builder,
)
from usersim.engine.core.outcomes import Provenance
from usersim.engine.core.probes import BankLoadError, load_builtin_probes
from usersim.engine.core.provenance import get_code_sha

logger = logging.getLogger("usersim.engine")

_ABSENT = object()

__all__ = [
    "MODEL_ALIASES",
    "MODEL_API_RESPONSE",
    "MODEL_ASSISTANT",
    "MODEL_JUDGE",
    "MODEL_SUMMARY",
    "MODEL_USER",
    "ConversationSimulatorGenerator",
]


# ---------------------------------------------------------------------------
# Probe registry — class-based, populated via the substrate
# ---------------------------------------------------------------------------
#
# Each probe is a ``BaseProbe`` subclass registered in the substrate's
# ``_PROBE_REGISTRY`` (in ``core/probes.py``) via ``@register_probe``
# at import time. Lookups load the shipped probes on demand; loading
# them here as well keeps the registry complete for code that reads it
# directly.

load_builtin_probes()


class ConversationSimulatorGenerator(
    ColumnGeneratorCellByCell[ConversationSimulatorConfig],
    ColumnGeneratorWithModelRegistry[ConversationSimulatorConfig],
):
    """Unified generator that dispatches to probe-specific simulation modules."""

    @classmethod
    def with_models(
        cls,
        config: ConversationSimulatorConfig,
        models: Mapping[str, Any],
    ) -> ConversationSimulatorGenerator:
        """Build the simulator against supplied model facades, without Data Designer.

        The supported way to run one row outside a Data Designer pipeline — for
        a hosted-parity check, say. ``models`` maps a role alias
        (``user_model``, ``assistant_model``, ...) to anything exposing
        ``acompletion``.
        """
        registry = SimpleNamespace(get_model=lambda *, model_alias: dict(models)[model_alias])
        return cls(config, SimpleNamespace(model_registry=registry))

    def _initialize(self) -> None:
        """Prepare process-wide state once, before any row runs.

        The log level is process-wide, so it belongs to the run rather than to
        a row: setting it per row would let one column's verbosity decide how
        loudly everything else logs.

        Resolving the code SHA shells out to git. Doing it here keeps that
        subprocess out of the first row, where it would be charged to a
        trajectory rather than to setup.
        """
        if self.config.verbosity >= 2:
            logging.getLogger("usersim.engine").setLevel(logging.DEBUG)
        get_code_sha()

    def generate(self, data: dict) -> dict:
        """Not available: a conversation awaits model calls. Use ``agenerate``.

        Turns, judges and tool responses are awaited several levels down, so
        there is no synchronous path to fall back on. The engine calls
        ``agenerate`` directly; this exists because the base class declares
        it and a caller reaching here has taken a wrong turn.
        """
        raise NotImplementedError(
            f"{type(self).__name__} simulates conversations asynchronously. "
            f"Await agenerate(data) instead of calling generate(data)."
        )

    async def agenerate(self, data: dict) -> dict:
        """Run one row with the configured asset root installed for its duration.

        The root is scoped to the row rather than to the process, so a row that
        sets one cannot change how any later row resolves its banks.
        """
        token = set_runtime_assets_dir(self.config.assets_dir)
        try:
            return await self._generate_row(data)
        finally:
            reset_runtime_assets_dir(token)

    async def _generate_row(self, data: dict) -> dict:
        cfg = self.config
        input_columns = set(data)
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

        t_record_start = time.monotonic()

        construction_error: EpisodeConstructionError | None = None
        try:
            constructed = construct_probe_episode(data, config=cfg, models=models)
            preamble = constructed.preamble
        except EpisodeConstructionError as error:
            constructed = None
            preamble = error.preamble
            construction_error = error
        data.clear()
        data.update(preamble.data)
        before_dispatch = dict(data)
        persona = preamble.persona
        probe_type = preamble.probe_type
        persona_name = f"{persona.get('first_name', '?')} {persona.get('last_name', '?')}"
        logger.info(
            f"  |-- Starting {probe_type} sim for {persona_name} "
            f"[{preamble.user_interaction_style}, {preamble.disclosure_style}] (max_turns={cfg.max_turns})"
        )

        provenance = preamble.provenance
        traj_id = preamble.trajectory_id
        # Tag the conversation log with the trajectory_id too — much
        # easier to grep through debug_llm.jsonl when chasing a
        # specific replayed row.
        set_conversation_id(f"{traj_id} | {persona_name} | {probe_type}")

        # Install the per-conversation outcome-builder hook so await acall_llm()
        # invocations inside this trajectory feed per-model tokens /
        # calls / latencies into simulation_outcome. Cleared in the
        # finally block so we never leak the builder across rows.
        row_outcome_builder = constructed.outcome if constructed is not None else None
        set_current_outcome_builder(row_outcome_builder)

        t_sim_start = time.monotonic()
        try:
            try:
                if construction_error is not None:
                    raise construction_error.cause
                assert constructed is not None
                result = await constructed.probe.run_dispatch(models=models, data=data, cfg=cfg)
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
        # A run writes back only the configured column and its declared side
        # effects: an input column keeps its original value and any other key is
        # dropped. So the probe loses every undeclared key it adds, and every
        # change it makes to an undeclared input column, returned or made in place.
        lost = {
            column
            for column, value in data.items()
            if column not in input_columns or not _unchanged(value, before_dispatch.get(column, _ABSENT))
        }
        _warn_on_undeclared_columns(
            probe_type,
            lost - set(cfg.side_effect_columns) - {cfg.name},
            already_reported=self.__dict__.setdefault("_undeclared_reported", set()),
        )
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


def _unchanged(value: Any, before: Any) -> bool:
    """Whether a column still holds what it held before the probe ran.

    The same object counts as unchanged without a comparison, which covers NaN
    (never equal to itself) and arrays (whose ``==`` is elementwise).
    """
    if value is before:
        return True
    try:
        return bool(value == before)
    except (TypeError, ValueError):
        return False


def _warn_on_undeclared_columns(
    probe_type: str,
    columns: set[str],
    *,
    already_reported: set[tuple[str, str]],
) -> None:
    """Report columns a probe added or changed that a Data Designer run will not keep.

    A run writes back only the configured column and ``side_effect_columns``,
    without an error, so a probe's undeclared ``build_result_extras`` key would
    never reach the scorer that reads it. ``already_reported`` belongs to one
    generator, which a run builds once, so each pair is reported once per run
    and a standalone run does not use up the warning a later run should give.
    """
    new = sorted(column for column in columns if (probe_type, column) not in already_reported)
    if not new:
        return
    already_reported.update((probe_type, column) for column in new)
    logger.warning(
        "probe %r returned or changed column(s) %s that are not in ConversationSimulatorConfig.side_effect_columns, "
        "so `usersim simulate` drops them from the stored row (an input column keeps its original value). "
        "Keep per-row data in the probe's metadata "
        "(state.metadata), which is stored as conversation_metadata; see docs/engine/AUTHORING_A_PROBE.md.",
        probe_type,
        new,
    )


def _log_running_stats() -> None:
    """Print running aggregate stats across all completed records."""
    rec = get_record_stats()
    if not rec:
        return
    model_stats = get_call_stats()
    if not model_stats:
        return

    # ``combined`` is the sum of per-record durations, which is larger than
    # the elapsed time of the run: trajectories overlap, so the same second
    # of wall clock is counted once per trajectory running through it.
    lines = [
        f"  |-- 📈 Running stats ({rec['records']} records): "
        f"avg={rec['avg_s']:.1f}s, min={rec['min_s']:.1f}s, "
        f"max={rec['max_s']:.1f}s, combined={rec['total_s']:.0f}s"
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
