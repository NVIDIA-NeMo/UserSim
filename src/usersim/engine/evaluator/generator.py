# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TrajectoryEvaluator — DD column generator that scores assistant trajectories.

Reads one trajectory row (conversation messages + persona + probe
metadata + simulation outcome) and writes a single wide JSON cell
containing per-axis × per-judge scores plus deterministic scorer
outputs. The reporting layer (``reporting/``) handles
wide -> long via ``pd.melt`` for inter-judge agreement, intersectional
heatmaps, etc.

Cell envelope (the JSON written into ``self.config.name``):

    {
      "envelope": {
        "judge_aliases": ["..."],
        "judge_families": ["openai", "nvidia_nemotron", ...],
        "axes": ["helpfulness", "accuracy", ...],
        "scorers": ["tool_use", ...],
        "prompt_version": "v1.0",
        "evaluator_version": "v1.0",
      },
      "axes": {
        "helpfulness": {
          "<judge_alias>": {"score": 4, "reasoning": "..."},
          ...
        },
        ...
      },
      "scorers": {
        "<scorer_name>": {...arbitrary structured result...},
        ...
      },
      "skipped": false,
      "skipped_reason": null
    }

Partial-re-run skip: if the row already has a non-null cell whose
envelope matches the current config, the LLM judge calls are skipped
and ``skipped=true`` / ``skipped_reason="envelope_match"`` is returned
in place of fresh axes.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterable

from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell,
    ColumnGeneratorWithModelRegistry,
)
from data_designer.engine.column_generators.utils.judge_score_factory import (
    create_judge_response_model,
    create_judge_structured_output_model,
)

from usersim.engine.core.identity import resolve_model_name
from usersim.engine.core.llm import call_llm
from usersim.engine.core.messages import format_conversation_history_for_prompt
from usersim.engine.core.persona import format_persona_for_prompt
from usersim.engine.evaluator.axes import select_axes
from usersim.engine.evaluator.config import TrajectoryEvaluatorConfig
from usersim.engine.evaluator.judges import (
    JudgeSpec,
    validate_ensemble_diversity,
)
from usersim.engine.evaluator.prompts import (
    EVAL_SYSTEM_PROMPT,
    EVAL_USER_PROMPT,
    build_locale_rigor_instruction,
)
from usersim.engine.evaluator.scorers import get_scorer

logger = logging.getLogger("usersim.engine")

EVALUATOR_VERSION = "v1.0"

# Locales whose scripts need a higher token budget for judge responses.
_NON_ASCII_LOCALES = frozenset(
    {
        "ja_JP",
        "zh_CN",
        "zh_TW",
        "ko_KR",
        "hi_Deva_IN",
        "ar_SA",
        "th_TH",
        "he_IL",
    }
)


def _decode_json_field(raw: Any, default: Any) -> Any:
    """Best-effort JSON decode for fields that may arrive as str or dict/list."""
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return default
    return default


class TrajectoryEvaluatorGenerator(
    ColumnGeneratorCellByCell[TrajectoryEvaluatorConfig],
    ColumnGeneratorWithModelRegistry[TrajectoryEvaluatorConfig],
):
    """Score one trajectory row with the configured judge ensemble + scorers."""

    def generate(self, data: dict) -> dict:
        cfg = self.config
        col_name = cfg.name

        if cfg.verbosity >= 2:
            logging.getLogger("usersim.engine").setLevel(logging.DEBUG)

        # ── Resolve probe family from the row ─────────────────────
        probe_family = data.get(cfg.probe_family_column) or "general_open_ended"
        if not isinstance(probe_family, str):
            probe_family = str(probe_family)

        # ── Build the envelope describing this evaluator config ──
        judge_specs: list[JudgeSpec] = [j.to_spec() for j in cfg.judges]
        # Re-validate at runtime in case the alias-only families need
        # the resolved model id to disambiguate.
        resolved_ids: dict[str, str] = {}
        for spec in judge_specs:
            facade = self.get_model(spec.alias) if spec.alias else None
            resolved_ids[spec.alias] = resolve_model_name(facade, spec.alias)
        validate_ensemble_diversity(judge_specs, resolved_model_ids=resolved_ids)

        applicable_axes = select_axes(probe_family, cfg.axes)
        envelope = {
            "judge_aliases": [s.alias for s in judge_specs],
            "judge_families": [s.resolved_family(resolved_ids.get(s.alias)).value for s in judge_specs],
            "axes": [s.name for s in applicable_axes],
            "scorers": list(cfg.scorers),
            "prompt_version": cfg.prompt_version,
            "evaluator_version": EVALUATOR_VERSION,
        }

        simulation_outcome = _decode_json_field(data.get(cfg.simulation_outcome_column), default=None)
        if isinstance(simulation_outcome, dict) and simulation_outcome.get("status") == "failed":
            # Any simulator failure means the trajectory did not complete under
            # the declared harness contract. Keep it for reliability/coverage,
            # but never turn a partial row into assistant-quality evidence.
            result = {
                "envelope": envelope,
                "axes": {},
                "scorers": {},
                "skipped": True,
                "skipped_reason": "simulation_failed",
            }
            return {col_name: json.dumps(result, ensure_ascii=False)}

        # ── Partial-re-run skip ─────────────────────────────────
        if cfg.skip_if_existing:
            existing = _decode_json_field(data.get(col_name), default=None)
            if isinstance(existing, dict) and existing.get("envelope") == envelope:
                logger.debug(
                    f"  |-- evaluator: skip — envelope match for traj {data.get(cfg.trajectory_id_column, '?')}"
                )
                # Mark explicitly so the reporting layer can count skips.
                # Overwrite (not setdefault) — the prior cell may have
                # legitimately been marked `skipped=False` when it was
                # first written, and we want the most recent re-run's
                # outcome to be visible.
                existing["skipped"] = True
                existing["skipped_reason"] = "envelope_match"
                return {col_name: json.dumps(existing, ensure_ascii=False)}

        # ── Decode the trajectory ───────────────────────────────
        messages = _decode_json_field(data.get(cfg.conversation_messages_column), default=[])
        if not isinstance(messages, list) or not any(
            m.get("role") == "assistant" for m in messages if isinstance(m, dict)
        ):
            # No assistant content -> nothing for an out-of-sim judge
            # to score. Return a structurally-valid skipped record.
            result = {
                "envelope": envelope,
                "axes": {},
                "scorers": {},
                "skipped": True,
                "skipped_reason": "no_assistant_messages",
            }
            return {col_name: json.dumps(result, ensure_ascii=False)}

        persona = _decode_json_field(data.get(cfg.persona_column), default={})
        locale = data.get(cfg.locale_column) or "en_US"
        language = data.get(cfg.conversation_language_column) or "English"

        persona_text = format_persona_for_prompt(persona) if persona else ""
        conversation_text = format_conversation_history_for_prompt(messages)

        # ── Run each judge across all applicable axes ────────────
        models = self.get_models()
        axis_results: Dict[str, Dict[str, Dict[str, Any]]] = {s.name: {} for s in applicable_axes}

        prompt = EVAL_USER_PROMPT.format(
            persona=persona_text,
            language=language,
            locale=locale,
            locale_rigor_instruction=build_locale_rigor_instruction(language),
            conversation=conversation_text,
        )

        max_tokens = cfg.max_judge_tokens_non_ascii if locale in _NON_ASCII_LOCALES else cfg.max_judge_tokens
        schema_model = _build_schema_for(applicable_axes)

        for spec in judge_specs:
            t_judge = time.monotonic()
            parsed = self._call_judge(
                models=models,
                judge_alias=spec.alias,
                prompt=prompt,
                schema_model=schema_model,
                max_tokens=max_tokens,
            )
            judge_elapsed = time.monotonic() - t_judge
            logger.debug(
                f"  |-- evaluator: {spec.alias} scored "
                f"{sum(1 for v in parsed.values() if v is not None)}/"
                f"{len(applicable_axes)} axes in {judge_elapsed:.1f}s"
            )
            for s in applicable_axes:
                cell = parsed.get(s.name)
                if not isinstance(cell, dict):
                    continue
                axis_results[s.name][spec.alias] = {
                    "score": cell.get("score"),
                    "reasoning": cell.get("reasoning", ""),
                }

        # ── Run any registered deterministic scorers ─────────────
        scorer_results: Dict[str, Any] = {}
        if cfg.scorers:
            # Pass-through merge: the bag of probe-specific top-level
            # columns each probe writes (sovereign_facts_probed /
            # query_id / target_request_id / strategy_id /
            # reframings_used / action_request_id / sub_protocol /
            # attempted_actions / probing_categories_explored /
            # probing_subtopic_hints_used / tool_subset / ...) is an
            # open set — every new probe may add new keys and the
            # corresponding scorer reads them off the trajectory dict
            # directly. Forwarding the whole row is the contract that
            # keeps that extension story friction-free; the explicit
            # projection below wins for shared keys (decoded forms of
            # conversation_messages / persona / simulation_outcome,
            # plus the resolved locale / language defaults).
            #
            # An earlier projection enumerated only 7 keys, which was
            # a silent bug: production scorers read probe-specific
            # keys that never reached them and returned no-op skip
            # envelopes. The pass-through-then-override merge below
            # keeps the contract extension-friendly.
            traj_row = {
                **data,
                "conversation_messages": messages,
                "persona": persona,
                "probe_family": probe_family,
                "probe_variant": data.get(cfg.probe_variant_column),
                "simulation_outcome": _decode_json_field(data.get(cfg.simulation_outcome_column), default=None),
                "locale": locale,
                "language": language,
            }
            for name in cfg.scorers:
                try:
                    fn = get_scorer(name)
                except KeyError as e:
                    logger.warning(f"  |-- evaluator: {e}")
                    scorer_results[name] = {"error": str(e)}
                    continue
                try:
                    scorer_results[name] = fn(traj_row, models)
                except Exception as e:
                    logger.warning(f"  |-- evaluator: scorer {name!r} raised {type(e).__name__}: {e}")
                    scorer_results[name] = {"error": f"{type(e).__name__}: {e}"}

        result = {
            "envelope": envelope,
            "axes": axis_results,
            "scorers": scorer_results,
            "skipped": False,
            "skipped_reason": None,
        }
        return {col_name: json.dumps(result, ensure_ascii=False)}

    # ── Internals ───────────────────────────────────────────────

    def get_models(self) -> Dict[str, Any]:
        """Resolve every configured judge alias to its facade."""
        out: Dict[str, Any] = {}
        for j in self.config.judges:
            try:
                out[j.alias] = self.get_model(j.alias)
            except Exception as e:
                logger.warning(f"  |-- evaluator: judge alias {j.alias!r} not resolvable in model registry: {e}")
        return out

    def _call_judge(
        self,
        models: Dict[str, Any],
        judge_alias: str,
        prompt: str,
        schema_model: type,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """One judge call, structured-output enforced. Returns parsed dict."""
        msgs = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = call_llm(
                models,
                judge_alias,
                msgs,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "evaluation",
                        "schema": schema_model.model_json_schema(),
                    },
                },
            )
        except Exception as e:
            logger.warning(f"  |-- evaluator: judge {judge_alias!r} raised {type(e).__name__}: {e}")
            return {}
        content = resp.get("content", "") if isinstance(resp, dict) else ""
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"  |-- evaluator: judge {judge_alias!r} produced unparseable structured output")
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _build_schema_for(axes: Iterable) -> type:
    """Build a Pydantic schema for the given axis list using DD's factory."""
    response_models = [create_judge_response_model(s) for s in axes]
    return create_judge_structured_output_model(response_models)
