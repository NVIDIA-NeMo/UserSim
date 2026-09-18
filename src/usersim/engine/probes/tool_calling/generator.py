# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tool_calling probe — a `BaseProbe` subclass for the unified simulation engine.

Tests the assistant model's native tool-calling capabilities using
persona-grounded, realistic user requests. Key design choices:

- No majority voting -- single-shot response reveals true capability
- No correction loops -- invalid tool calls get API-like error responses
- Verifier kept as diagnostic tool -- validates structurally, returns
  realistic error messages instead of triggering retries

The 8-axis trajectory judge lives in ``evaluator/scorers/tool_use.py``
and is invoked by the ``trajectory-evaluator`` plugin. This module only
emits trajectories; sim-side guardrails (e.g. zero-tool-calls) live on
``ToolCallingProbe.should_succeed()``.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict

from usersim.engine.core.behavioral import (
    format_behavioral_profile_for_prompt,
    format_disclosure_instructions,
    format_interaction_style_instructions,
)
from usersim.engine.core.llm import call_llm
from usersim.engine.core.messages import (
    _parse_theme,
    format_conversation_history_for_prompt,
    format_tools_for_prompt,
)
from usersim.engine.core.persona import format_persona_for_prompt
from usersim.engine.core.outcomes import (
    SimulationTrace,
    TraceKind,
)
from usersim.engine.core.probes import (
    BaseProbe,
    ToolCallingMixin,
    ToolExecutionMixin,
    register_probe,
)
from usersim.engine.core.prompt_loader import render_prompt
from usersim.engine.core.simulation import (
    ConversationState,
    language_instruction as _language_instruction,
    make_failed,
)
from usersim.engine.probes.tool_calling.prompts import (
    assistant_system_prompt,
    tool_relevance_failure,
    tool_relevance_success,
    tool_simulator_prompt,
    user_agent_system_prompt,
    user_judge_prompt,
)
from usersim.engine.probes.tool_calling.verifiers import ToolCallVerifier

logger = logging.getLogger("usersim.engine")

# ── Scenario metadata ────────────────────────────────────────────────
# See general_educational/generator.py for the contract. `probe_variant` is
# carried on the trajectory; the evaluator reads it for stratification.

PROBE_FAMILY: str = "tool_calling"
PROBE_VARIANTS: list[str] = ["default"]  # tool-category via toolset_category column; single variant today
PROMPT_VERSION: str = "v1.1"

MODEL_USER = "user_model"
MODEL_ASSISTANT = "assistant_model"
MODEL_API_RESPONSE = "api_response_model"
MODEL_JUDGE = "judge_model"


def _derive_tool_theme(data: dict, cfg: Any) -> dict:
    category = data.get("toolset_category")
    if category:
        toolset_name = data.get("toolset_name") or ""
        label = toolset_name if toolset_name else category
        return {
            "type": category,
            "description": f"Help the user with a task that involves {label} capabilities.",
            "tool_expected": True,
        }
    return _parse_theme(data.get(cfg.theme_column, "{}"))


def _build_tool_context(tool_subset: list) -> str:
    lines = []
    for tool_data in tool_subset:
        spec = tool_data["tool"]["function"] if "tool" in tool_data else tool_data.get("function", tool_data)
        name = spec.get("name", "unknown")
        desc = spec.get("description", "No description available.")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines) if lines else "No tools available."


def _normalize_tool_list(tools_raw: Any) -> list[dict]:
    """Normalize parquet/JSONL tool cells to a Python list of mappings.

    Parquet stores ``tools`` as a JSON string. Data Designer reads a nested
    JSONL array through Arrow/Pandas as a NumPy-like array, whose ``tolist``
    result is the authored list. ``random.sample`` rejects that array directly,
    so normalize at the probe boundary before sampling.
    """
    if isinstance(tools_raw, str):
        parsed = json.loads(tools_raw)
    elif hasattr(tools_raw, "tolist"):
        parsed = tools_raw.tolist()
    else:
        parsed = tools_raw
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("tool_calling: tools must be a JSON array or JSON-encoded array")
    tools = list(parsed)
    if not all(isinstance(tool, dict) for tool in tools):
        raise ValueError("tool_calling: every tools entry must be an object")
    return tools


def _format_tools_for_api(tools: list) -> list:
    openai_tools = []
    for tool_data in tools:
        if "tool" in tool_data:
            openai_tools.append({"type": "function", "function": tool_data["tool"]["function"]})
        elif "function" in tool_data and "type" not in tool_data:
            openai_tools.append({"type": "function", "function": tool_data["function"]})
        else:
            openai_tools.append(tool_data)
    return openai_tools


def _find_tool_spec(tool_name: str, tools: list) -> dict | None:
    for t in tools:
        spec = t["tool"]["function"] if "tool" in t else t.get("function", t)
        if spec.get("name") == tool_name:
            return spec
    return None


def _collect_prior_tool_responses(conversation_messages: list) -> str:
    """Extract prior tool responses from the conversation for consistency context."""
    tool_responses = []
    for msg in conversation_messages:
        if msg.get("role") == "tool":
            tool_responses.append(msg.get("content", ""))
    if not tool_responses:
        return ""
    return (
        "\n\nPrior tool responses in this conversation (maintain consistency "
        "with any data already returned):\n" + "\n---\n".join(tool_responses[-5:])
    )


def _simulate_tool_response(
    models: dict,
    tool_spec: dict,
    tool_call: dict,
    conversation_messages: list,
) -> tuple[str, bool]:
    """Use the strong frontier model to simulate a realistic API response.

    Includes prior tool responses as context so the simulator maintains
    internal consistency across calls within a conversation.

    Returns ``(content, did_reroll)``. ``did_reroll`` is True if the
    first response was invalid JSON and a second call was needed; this
    is structured sim-health signal surfaced as ``n_api_response_rerolls``
    in ``simulation_outcome``.
    """
    prior_context = _collect_prior_tool_responses(conversation_messages)
    prompt = tool_simulator_prompt().format(
        tool_spec=json.dumps(tool_spec, indent=2, default=str),
        tool_args=json.dumps(tool_call.get("function", {}).get("arguments", "{}"), default=str),
        conversation_context=format_conversation_history_for_prompt(conversation_messages[-6:]),
    )
    if prior_context:
        prompt += prior_context
    msgs = [{"role": "user", "content": prompt}]
    resp = call_llm(models, MODEL_API_RESPONSE, msgs)
    content = resp.get("content", "{}") if isinstance(resp, dict) else "{}"

    did_reroll = False
    try:
        json.loads(content)
    except (json.JSONDecodeError, TypeError):
        did_reroll = True
        logger.debug("  |-- api_response_model: invalid JSON, retrying once")
        resp = call_llm(models, MODEL_API_RESPONSE, msgs)
        content = resp.get("content", "{}") if isinstance(resp, dict) else "{}"

    return content, did_reroll


# ---------------------------------------------------------------------------
# Tool-calling probe
# ---------------------------------------------------------------------------


@register_probe(family=PROBE_FAMILY, prompt_version=PROMPT_VERSION, variants=tuple(PROBE_VARIANTS))
class ToolCallingProbe(ToolExecutionMixin, ToolCallingMixin, BaseProbe):
    """Tool-calling probe.

    Tests the assistant model's native tool-calling capabilities using
    persona-grounded, realistic user requests. The 8-axis trajectory
    judge lives in ``evaluator/scorers/tool_use.py`` and is invoked
    by the out-of-sim trajectory evaluator. The simulator only emits
    trajectories; sim-side guardrails (e.g., zero-tool-calls) live on
    ``should_succeed()``. ``ToolCallingMixin`` blocks early-stop until
    at least one tool call has happened.

    ``ToolExecutionMixin`` (``tool_loop_mode="single"``, the default)
    supplies ``after_assistant_turn``: one round of tool calls then a single
    synthesis pass with no tools (single-shot capability, no correction loop).
    This probe supplies only the per-call behaviour (verify + simulate) via
    ``execute_tool_call`` and records ``tool_call_count_per_turn`` in
    ``on_tool_round_complete``.
    """

    label = "tool_calling"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        cfg = self._cfg
        if not getattr(cfg, "tools_column", None):
            raise ValueError("tool_calling: cfg.tools_column not configured")
        if cfg.tools_column not in self._data:
            raise ValueError(f"tool_calling: column {cfg.tools_column!r} missing in data")

        tools_raw = self._data[cfg.tools_column]
        all_tools = _normalize_tool_list(tools_raw)
        self.tool_subset = random.sample(
            all_tools,
            min(cfg.max_tools, len(all_tools)),
        )

        theme = _derive_tool_theme(self._data, cfg)
        logger.info(f"  |-- tool_calling: {len(self.tool_subset)} tools, theme={theme.get('type', '?')}")

        self.openai_tools = _format_tools_for_api(self.tool_subset)
        self.tools_for_judge = format_tools_for_prompt(self.tool_subset)
        self.verifier = ToolCallVerifier()
        self._theme = theme

    def get_user_system_prompt(self) -> str:
        return render_prompt(
            user_agent_system_prompt(),
            self._data,
            persona=format_persona_for_prompt(self._persona),
            tool_context=_build_tool_context(self.tool_subset),
            theme=json.dumps(self._theme, default=str),
            language_instruction=_language_instruction(self._language, self._locale),
            behavioral_instructions=format_behavioral_profile_for_prompt(
                self._profile,
                probe_type="tool_calling",
                language=self._language,
            ),
            disclosure_instructions=format_disclosure_instructions(
                self._data.get("disclosure_style", "upfront"),
            ),
            interaction_style_instructions=format_interaction_style_instructions(
                self._interaction_style,
            ),
        )

    def get_assistant_system_prompt(self) -> str:
        # Empty as shipped; see the accessor's docstring.
        return render_prompt(assistant_system_prompt(), self._data)

    def get_tools_for_assistant(self) -> list | None:
        return self.openai_tools

    def format_gate_prompt(self, user_query: str, conversation_history: str) -> str:
        if conversation_history == "N/A":
            extra_criteria = tool_relevance_success()
            extra_failure_criteria = tool_relevance_failure()
        else:
            extra_criteria = ""
            extra_failure_criteria = ""
        return render_prompt(
            user_judge_prompt(),
            self._data,
            tools=self.tools_for_judge,
            conversation_history=conversation_history,
            user_turn_to_evaluate=user_query,
            extra_criteria=extra_criteria,
            extra_failure_criteria=extra_failure_criteria,
        )

    def execute_tool_call(
        self,
        name: str,
        args: Dict[str, Any],
        tc: Dict[str, Any],
        state: ConversationState,
        models: dict,
        *,
        turn_idx: int,
        call_idx: int,
    ) -> str:
        """Verify one tool call and simulate a realistic API response.

        Owns the tool_calling-specific bookkeeping (verifier_results,
        TOOL_CALL_VERIFIER trace, api-response rerolls, tools_called). The
        ``ToolExecutionMixin`` drives the message plumbing + the single
        synthesis pass. ``name`` is the parsed function name; ``tc`` is the
        raw tool-call dict the verifier + simulator need.
        """
        tc_ok, tc_err = self.verifier.verify_single_tool_call(tc, self.tool_subset)
        state.metadata.setdefault("verifier_results", []).append(
            {
                "tool_name": name,
                "valid": tc_ok,
                "error": tc_err,
            }
        )
        state.outcome.add_trace(
            SimulationTrace(
                kind=TraceKind.TOOL_CALL_VERIFIER,
                turn_idx=turn_idx,
                call_idx=call_idx,
                rating="success" if tc_ok else "failure",
                detail=(tc_err if not tc_ok else f"{name} validated"),
                extra={"tool_name": name},
            )
        )

        if not tc_ok:
            return json.dumps(
                {
                    "error": f"Invalid tool call: {tc_err}",
                    "status": 400,
                }
            )

        tool_spec = _find_tool_spec(name, self.tool_subset)
        if tool_spec:
            simulated_response, did_reroll = _simulate_tool_response(
                models,
                tool_spec,
                tc,
                state.messages,
            )
            if did_reroll:
                state.outcome.inc_api_response_rerolls()
                state.outcome.add_trace(
                    SimulationTrace(
                        kind=TraceKind.API_RESPONSE_REROLL,
                        turn_idx=turn_idx,
                        call_idx=call_idx,
                        model_alias=MODEL_API_RESPONSE,
                        detail=f"invalid JSON on first attempt for {name}",
                    )
                )
        else:
            simulated_response = json.dumps({"error": f"Unknown tool: {name}"})

        state.metadata.setdefault("tools_called", []).append(name)
        return simulated_response

    def on_tool_round_complete(self, state: ConversationState, n_calls: int) -> None:
        state.metadata.setdefault("tool_call_count_per_turn", []).append(n_calls)

    def should_succeed(self, state: ConversationState) -> bool:
        """Sim-side guardrail: a tool-calling trajectory with zero tool calls
        is a sim-quality failure (the assistant never engaged the probe's
        defining behavior). The 8-axis assistant-quality scoring lives in
        the trajectory-evaluator's ``tool_use`` scorer."""
        num_tool_calls = len(state.metadata.get("tools_called", []))
        if num_tool_calls == 0:
            logger.info("  |-- tool_calling: sim_status=False — zero successful tool calls")
            return False
        return True

    def build_result_extras(self, state: ConversationState) -> dict:
        return {
            "num_tool_calls": len(state.metadata.get("tools_called", [])),
            "tool_subset": json.dumps(self.tool_subset, ensure_ascii=False, default=str),
        }


# ---------------------------------------------------------------------------
# Public entry point — thin shim around ToolCallingProbe
# ---------------------------------------------------------------------------


def simulate_tool_calling(
    models: dict,
    data: dict,
    persona: dict,
    profile: dict,
    locale: str,
    language: str,
    cfg: Any,
    **kwargs: Any,
) -> dict:
    """Thin shim for callers that import ``simulate_tool_calling`` directly.

    The dispatcher in ``usersim.engine.generator`` instantiates
    ``ToolCallingProbe`` and calls ``probe.run_dispatch`` directly;
    this function exists for backward-compatibility with per-probe
    tests that import the symbol. Returns a structured failure record
    on missing-tools-column rather than raising.
    """
    if not getattr(cfg, "tools_column", None):
        logger.warning("  |-- tool_calling: tools_column not configured")
        return make_failed("tools_column not configured")
    if cfg.tools_column not in data:
        logger.warning(f"  |-- tool_calling: column '{cfg.tools_column}' not found in data")
        return make_failed(f"tools_column '{cfg.tools_column}' missing in data")

    probe = ToolCallingProbe(
        persona=persona,
        locale=locale,
        language=language,
        models=models,
        cfg=cfg,
        provenance=kwargs.get("provenance"),
        profile=profile,
        data=data,
        outcome_builder=kwargs.get("outcome_builder"),
    )
    return probe.run_dispatch(models=models, data=data, cfg=cfg)
