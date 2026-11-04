# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tool_calling probe — a `BaseProbe` subclass for the unified simulation engine.

Tests the assistant model's native tool-calling capabilities using
persona-grounded, realistic user requests. Key design choices:

- No majority voting -- single-shot response reveals true capability
- No correction loops -- invalid tool calls get API-like error responses
- Verifier kept as diagnostic tool -- validates structurally, returns
  realistic error messages instead of triggering retries
- Credentials a tool declares are held by the simulated user, one random
  value per credential and conversation (``credentials.py``); a call that
  passes any other value gets a 401 response

The 8-axis trajectory judge lives in ``evaluator/scorers/tool_use.py``
and is invoked by the ``trajectory-evaluator`` plugin. This module only
emits trajectories. A conversation in which the assistant made no tool
calls is a completed trajectory, and the judge scores it like any other.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from usersim.engine.probes.tool_calling.credentials import (
    declared_credentials,
    draw_credentials,
    format_held_credentials,
    rejected_credential,
)
from usersim.engine.probes.tool_calling.prompts import (
    _SCENARIO,
    assistant_system_prompt,
    tool_relevance_failure,
    tool_relevance_success,
    tool_simulator_credentials,
    tool_simulator_prompt,
    user_agent_system_prompt,
    user_credentials,
    user_judge_prompt,
)
from usersim.engine.probes.tool_calling.verifiers import ToolCallVerifier
from usersim.extension import (
    BaseProbe,
    ChatRealism,
    ConversationState,
    PreviewField,
    SimulationTrace,
    ToolCallingMixin,
    ToolExecutionMixin,
    TraceKind,
    acall_llm,
    format_behavioral_profile_for_prompt,
    format_conversation_history_for_prompt,
    format_disclosure_instructions,
    format_interaction_style_instructions,
    format_persona_for_prompt,
    format_tools_for_prompt,
    load_prompt_file,
    register_probe,
    render_prompt,
)
from usersim.extension import language_instruction as _language_instruction

logger = logging.getLogger("usersim.engine")

# ── Scenario metadata ────────────────────────────────────────────────
# See general_educational/probe.py for the contract. `probe_variant` is
# carried on the trajectory; the evaluator reads it for stratification.

PROBE_FAMILY: str = "tool_calling"
PROBE_VARIANTS: list[str] = ["default"]  # tool-category via toolset_category column; single variant today
PROMPT_VERSION: str = "v1.2"

MODEL_USER = "user_model"
MODEL_ASSISTANT = "assistant_model"
MODEL_API_RESPONSE = "api_response_model"
MODEL_JUDGE = "judge_model"


def _derive_tool_theme(data: dict) -> dict:
    """The theme a row's toolset gives it: the toolset's category, and its name where it has one.

    A row from a catalog without categories takes its toolset's name, and one
    that names neither gets a theme of using the offered tools.
    """
    category = data.get("toolset_category") or ""
    toolset_name = data.get("toolset_name") or ""
    if not (category or toolset_name):
        return {
            "type": "Tool use",
            "description": "Help the user with a task the offered tools support.",
            "tool_expected": True,
        }
    return {
        "type": category or toolset_name,
        "description": f"Help the user with a task that involves {toolset_name or category} capabilities.",
        "tool_expected": True,
    }


def _function_spec(tool_data: dict) -> dict:
    """The function spec inside a tool entry, whichever envelope the entry uses."""
    return tool_data["tool"]["function"] if "tool" in tool_data else tool_data.get("function", tool_data)


def _build_tool_context(tool_subset: list) -> str:
    lines = []
    for tool_data in tool_subset:
        spec = _function_spec(tool_data)
        name = spec.get("name", "unknown")
        desc = spec.get("description", "No description available.")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines) if lines else "No tools available."


def _tools_column(cfg: Any) -> str:
    """The row column holding the toolset's tools: the run's ``tools_column``, else the catalog's ``tools``."""
    return getattr(cfg, "tools_column", None) or "tools"


def _row_identity(data: dict) -> str:
    """The row's identity, which seeds its tool draw and the credentials its user holds.

    Prefers the trajectory identity so two configurations over one persona
    draw independently, and falls back to the persona when a caller builds a
    probe without one.
    """
    return str(data.get("trajectory_id") or data.get("persona_uuid") or "")


def _tool_subset_seed(data: dict) -> int:
    """Derive a stable seed for a row's tool draw."""
    return int(hashlib.sha256(_row_identity(data).encode()).hexdigest(), 16) & 0xFFFFFFFF


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
    """The tools as the assistant's API receives them: each function spec unchanged.

    A tool's ``credentials`` declaration is for the simulation, so it stays out
    of the request.
    """
    openai_tools = []
    for tool_data in tools:
        if "tool" in tool_data:
            openai_tools.append({"type": "function", "function": tool_data["tool"]["function"]})
        elif "function" in tool_data and "type" not in tool_data:
            openai_tools.append({"type": "function", "function": tool_data["function"]})
        else:
            openai_tools.append({key: value for key, value in tool_data.items() if key != "credentials"})
    return openai_tools


def _find_tool_spec(tool_name: str, tools: list) -> dict | None:
    for t in tools:
        spec = _function_spec(t)
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


async def _simulate_tool_response(
    models: dict,
    tool_spec: dict,
    tool_call: dict,
    conversation_messages: list,
    *,
    credential_parameters: tuple[str, ...] = (),
) -> tuple[str, bool]:
    """Use the strong frontier model to simulate a realistic API response.

    Includes prior tool responses as context so the simulator maintains
    internal consistency across calls within a conversation. The probe
    checks a call's credentials before this runs, so when
    ``credential_parameters`` names any, the simulator is told they are
    valid rather than left to judge a random value.

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
    if credential_parameters:
        prompt += "\n\n" + tool_simulator_credentials().format(
            parameters=", ".join(f"`{parameter}`" for parameter in credential_parameters),
        )
    if prior_context:
        prompt += prior_context
    msgs = [{"role": "user", "content": prompt}]
    resp = await acall_llm(models, MODEL_API_RESPONSE, msgs)
    content = resp.get("content", "{}") if isinstance(resp, dict) else "{}"

    did_reroll = False
    try:
        json.loads(content)
    except (json.JSONDecodeError, TypeError):
        did_reroll = True
        logger.debug("  |-- api_response_model: invalid JSON, retrying once")
        resp = await acall_llm(models, MODEL_API_RESPONSE, msgs)
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
    trajectories. ``ToolCallingMixin`` blocks early-stop until at least
    one tool call has happened, so a conversation with no tool calls ran
    every turn; it stays scoreable, because not calling a tool is the
    assistant's behaviour, not a failure of the probe's protocol.

    ``ToolExecutionMixin`` (``tool_loop_mode="single"``, the default)
    supplies ``after_assistant_turn``: one round of tool calls then a single
    synthesis pass with no tools (single-shot capability, no correction loop).
    This probe supplies only the per-call behaviour (verify + simulate) via
    ``execute_tool_call`` and records ``tool_call_count_per_turn`` in
    ``on_tool_round_complete``.
    """

    label = "tool_calling"
    assets_root = Path(__file__).parent / "assets"
    # The toolset catalog: each row draws one toolset and reads its tools from
    # the ``tools`` column. The toolset is also the row's theme, so the probe
    # lists no themes and its rows' ``theme`` column holds the run's placeholder.
    seed_dataset = "toolsets_seed"
    description = (
        "Tests the assistant's ability to invoke tools correctly in response to user "
        "requests, including argument shaping, schema validation, and recovery from "
        "tool-call failures."
    )
    default_scorers = ("tool_use",)
    preview_fields = (
        PreviewField("toolset_name", "Toolset"),
        PreviewField("toolset_category", "Toolset category"),
        PreviewField("num_tools", "Tools available"),
    )

    @classmethod
    def asset_problems(cls) -> list[str]:
        """The prompts file the prompt defaults would mask."""
        if not load_prompt_file(_SCENARIO):
            return [f"{_SCENARIO}/prompts.yaml is missing or empty"]
        return []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        cfg = self._cfg
        tools_column = _tools_column(cfg)
        if tools_column not in self._data:
            raise ValueError(f"tool_calling: column {tools_column!r} missing in data")

        tools_raw = self._data[tools_column]
        all_tools = _normalize_tool_list(tools_raw)
        # Drawn from the row's own identity rather than the process RNG. The
        # subset shapes the entire conversation but is not part of the
        # trajectory identity, so an identity only means something if the same
        # row is always handed the same tools.
        self.tool_subset = random.Random(_tool_subset_seed(self._data)).sample(
            all_tools,
            min(cfg.max_tools, len(all_tools)),
        )

        theme = _derive_tool_theme(self._data)
        logger.info(f"  |-- tool_calling: {len(self.tool_subset)} tools, theme={theme.get('type', '?')}")

        # The credentials each offered tool declares, by tool name, and the one
        # value per credential that this conversation's user holds.
        self.declared_credentials = {
            _function_spec(entry).get("name"): declared
            for entry in self.tool_subset
            if (declared := declared_credentials(entry))
        }
        self.held_credentials = draw_credentials(self.tool_subset, _row_identity(self._data))

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
            credentials=(
                user_credentials().format(credential_list=format_held_credentials(self.held_credentials))
                if self.held_credentials
                else ""
            ),
            theme=json.dumps(self._theme, default=str),
            language_instruction=_language_instruction(self._language, self._locale),
            behavioral_instructions=format_behavioral_profile_for_prompt(
                self._profile,
                language=self._language,
                chat_realism=ChatRealism.SHARING_DETAILS,
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

    async def execute_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        tc: dict[str, Any],
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

        A call that fails the schema check gets a 400 response. One that
        passes it but does not carry the credential the user holds for a
        parameter the tool declares gets a 401, as the service would answer
        a missing or wrong key.
        """
        declared = self.declared_credentials.get(name, {})
        tc_ok, tc_err = self.verifier.verify_single_tool_call(tc, self.tool_subset)
        unauthorized = None
        if tc_ok:
            unauthorized = rejected_credential(declared, args, self.held_credentials)
            if unauthorized is not None:
                parameter, credential = unauthorized
                tc_ok, tc_err = False, f"`{parameter}` is not the {credential} the user holds."
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

        if unauthorized is not None:
            return json.dumps(
                {
                    "error": f"Unauthorized: the {unauthorized[1]} is missing or invalid.",
                    "status": 401,
                }
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
            simulated_response, did_reroll = await _simulate_tool_response(
                models,
                tool_spec,
                tc,
                state.messages,
                credential_parameters=tuple(declared),
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

    def build_result_extras(self, state: ConversationState) -> dict:
        return {
            "num_tool_calls": len(state.metadata.get("tools_called", [])),
            "tool_subset": json.dumps(self.tool_subset, ensure_ascii=False, default=str),
        }
