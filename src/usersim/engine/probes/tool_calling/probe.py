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
  passes any other value gets a 401 response, and an accepted call that sets
  a new value, such as a password change, leaves the user holding it

The trajectory judge lives in ``evaluator/scorers/tool_use.py``
and is invoked by the ``trajectory-evaluator`` plugin. This module only
emits trajectories. A conversation in which the assistant made no tool
calls is a completed trajectory, and the judge scores it like any other.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import Field

from usersim.engine.probes.tool_calling.credentials import (
    credentials_set_by,
    declared_credentials,
    draw_credentials,
    format_held_credentials,
    rejected_credential,
)
from usersim.engine.probes.tool_calling.prompts import (
    _SCENARIO,
    assistant_context,
    assistant_judge_prompt,
    assistant_system_prompt,
    tool_relevance_failure,
    tool_relevance_success,
    tool_simulator_caller,
    tool_simulator_credentials,
    tool_simulator_cut_off,
    tool_simulator_feedback,
    tool_simulator_judge_caller,
    tool_simulator_judge_credentials,
    tool_simulator_judge_prompt,
    tool_simulator_prompt,
    user_agent_system_prompt,
    user_credentials,
    user_judge_prompt,
)
from usersim.engine.probes.tool_calling.verifiers import ToolCallVerifier
from usersim.extension import (
    MODEL_JUDGE,
    BaseProbe,
    ChatRealism,
    ConversationState,
    PreviewField,
    SimulationTrace,
    ToolCallingMixin,
    ToolExecutionMixin,
    TraceKind,
    UserTurnPolicy,
    acall_llm,
    format_behavioral_profile_for_prompt,
    format_disclosure_instructions,
    format_interaction_style_instructions,
    format_persona_for_prompt,
    format_tools_for_prompt,
    load_prompt_file,
    persona_full_name,
    persona_location,
    register_probe,
    render_prompt,
    run_inline_judge_ex,
    tool_result_weight,
)
from usersim.extension import language_instruction as _language_instruction

logger = logging.getLogger("usersim.engine")

MODEL_API_RESPONSE = "api_response_model"


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


def _other_endpoints(tool_subset: list, tool_name: str) -> str:
    """The offered tools other than ``tool_name``, one per line as ``- name(parameter: type, ...)``, or ``None.``.

    The tool simulator and its judge read them as the service's other endpoints, so an ID or other value a
    response returns has the type and form the endpoints that take it accept.
    """
    lines = []
    for tool_data in tool_subset:
        spec = _function_spec(tool_data)
        name = spec.get("name")
        if not name or name == tool_name:
            continue
        properties = (spec.get("parameters") or {}).get("properties") or {}
        parameters = ", ".join(
            f"{parameter}: {(schema or {}).get('type', 'any')}" for parameter, schema in properties.items()
        )
        lines.append(f"- {name}({parameters})")
    return "\n".join(lines) if lines else "None."


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

    A tool's ``credentials`` and ``sets_credentials`` declarations are for the
    simulation, so they stay out of the request.
    """
    openai_tools = []
    for tool_data in tools:
        if "tool" in tool_data:
            openai_tools.append({"type": "function", "function": tool_data["tool"]["function"]})
        elif "function" in tool_data and "type" not in tool_data:
            openai_tools.append({"type": "function", "function": tool_data["function"]})
        else:
            openai_tools.append(
                {key: value for key, value in tool_data.items() if key not in ("credentials", "sets_credentials")}
            )
    return openai_tools


#: A parameter name, split into words, holding one of these takes a place: a city, an address, coordinates.
_PLACE_WORD = re.compile(
    r"(^|_)(location|city|town|address|place|region|state|province|country|countrycode|zip|zipcode|postal|"
    r"postcode|lat|latitude|lon|lng|longitude|coordinates|geo)(_|$)"
)


#: A parameter name, split into words, holding one of these refers to one of the person's own things: a parcel's
#: tracking number, an order or booking ID, an invoice.
_OWN_THING_WORD = re.compile(
    r"(^|_)(tracking|trackingnumber|shipment|parcel|waybill|awb|consignment|pnr|invoice|receipt)(_|$)"
    r"|(^|_)(order|booking|reservation|confirmation)_?(id|number|no|ref|reference|code)(_|$)"
)


def _parameter_names(tool_spec: dict) -> list[str]:
    """The tool's parameter names in lowercase snake case, so ``trackingNumber`` reads ``tracking_number``."""
    properties = (tool_spec.get("parameters") or {}).get("properties") or {}
    return [re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower() for name in properties]


def _takes_place(tool_spec: dict) -> bool:
    """Whether the tool takes a place among its arguments, so a call is where its place comes from."""
    return any(_PLACE_WORD.search(name) for name in _parameter_names(tool_spec))


def _names_own_thing(tool_spec: dict) -> bool:
    """Whether the tool takes the reference of one of the person's own things, such as a tracking or order number."""
    return any(_OWN_THING_WORD.search(name) for name in _parameter_names(tool_spec))


def _find_tool_spec(tool_name: str, tools: list) -> dict | None:
    for t in tools:
        spec = _function_spec(t)
        if spec.get("name") == tool_name:
            return spec
    return None


def _arguments_text(arguments: Any) -> str:
    """A call's arguments as text: a string as the assistant sent it, anything else as JSON, unescaped."""
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False, default=str)


#: The most of the earlier calls' responses that the simulator and its judge
#: read, weighed as ``tool_result_weight`` does: as much as the evaluator reads
#: of one tool result.
_EARLIER_CALLS_CHARS = 24_000


def _canonical_arguments(arguments: str) -> str:
    """Arguments as sorted compact JSON, so one call written two ways compares equal; other text as it is."""
    try:
        return json.dumps(json.loads(arguments), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return arguments


def _argument_values(arguments: str) -> list[str]:
    """The call's top-level string and number arguments of 3 or more characters, such as an ID or a name."""
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, dict):
        return []
    values = (str(v) for v in parsed.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool))
    return [v for v in values if len(v) >= 3]


def _earlier_calls(conversation_messages: list, tool_name: str, arguments: str) -> str:
    """The calls made earlier in the conversation, each with its arguments, time and response: the service's records.

    Every call is listed, in order, with the clock it was made at
    (``_clock``). Responses go in whole up to
    ``_EARLIER_CALLS_CHARS``: first the response to the same call
    (``tool_name`` with ``arguments``), then those naming one of its argument
    values, then the rest of ``tool_name``'s, then everything else, each
    newest first. A response past that shows as its call alone.
    """
    calls: dict[Any, tuple[str, str]] = {}
    records: list[tuple[str, str, str, str]] = []
    user_messages = 0
    for message in conversation_messages:
        if message.get("role") == "user":
            user_messages += 1
        elif message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                calls[call.get("id")] = (function.get("name", "?"), _arguments_text(function.get("arguments", "{}")))
        elif message.get("role") == "tool":
            name, args = calls.get(message.get("tool_call_id"), ("?", "{}"))
            call = f"{name}({args}) at {_clock(user_messages)} UTC"
            records.append((call, name, _canonical_arguments(args), message.get("content") or ""))
    if not records:
        return "None."
    same_call = (tool_name, _canonical_arguments(arguments))
    values = _argument_values(arguments)

    def rank(i: int) -> tuple[bool, bool, bool, int]:
        _call, name, args, response = records[i]
        return (name, args) != same_call, not any(v in response for v in values), name != tool_name, -i

    shown: set[int] = set()
    used = 0
    for i in sorted(range(len(records)), key=rank):
        weight = tool_result_weight(records[i][3])
        if used + weight <= _EARLIER_CALLS_CHARS:
            shown.add(i)
            used += weight
    return "\n\n".join(
        f"{call} returned:\n{response}" if i in shown else f"{call} returned a response too long to show here."
        for i, (call, _name, _args, response) in enumerate(records)
    )


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _format_day(day: date) -> str:
    """A date with its weekday, such as ``Friday, 2026-11-06``, independent of the process locale."""
    return f"{_WEEKDAYS[day.weekday()]}, {day.isoformat()}"


#: How far the simulated service's clock moves with each user message after the first: the time a person takes
#: to read a reply and write back.
_MINUTES_PER_USER_MESSAGE = 2


def _user_messages(conversation_messages: list) -> int:
    return sum(1 for message in conversation_messages if message.get("role") == "user")


def _clock(user_messages: int) -> str:
    """The simulated service's clock, as ``12:04``, once the user has sent ``user_messages`` messages.

    Noon UTC at the first, moving ``_MINUTES_PER_USER_MESSAGE`` with each one
    after it and stopping at 12:59, so the clock stays on the conversation's
    date in every time zone from UTC-11 to UTC+11, the date the user and the
    assistant work from.
    """
    return f"12:{min(_MINUTES_PER_USER_MESSAGE * max(user_messages - 1, 0), 59):02d}"


def _format_now(day: date, user_messages: int = 1) -> str:
    """The simulated service's date and clock, such as ``Friday, 2026-11-06, 12:02 UTC``."""
    return f"{_format_day(day)}, {_clock(user_messages)} UTC"


#: What a call answers when no attempt produced JSON.
_INVALID_RESPONSE = json.dumps({"error": "The service returned an invalid response.", "status": 502})


@dataclass
class SimulatedResponse:
    """One tool call's simulated response, and how it was reached.

    ``rerolls`` counts attempts regenerated because their output was not JSON;
    ``verdicts`` holds each judge verdict in order, as ``(passed, explanation)``.
    ``unverified`` says why the stored response stands without a passing
    verdict, or is ``None`` when it passed or was not judged.
    """

    content: str
    rerolls: int = 0
    verdicts: list[tuple[bool, str]] = field(default_factory=list)
    unverified: str | None = None


async def _judge_tool_response(
    models: dict,
    content: str,
    *,
    tool_spec: dict,
    tool_args: str,
    earlier_calls: str,
    now: str,
    credential_parameters: tuple[str, ...],
    caller: str = "",
    judge_alias: str = MODEL_JUDGE,
    other_endpoints: str = "None.",
) -> tuple[bool, str] | None:
    """The ``judge_alias`` model's verdict on one simulated response, ``(passed, explanation)``; None when it can't be read."""
    prompt = tool_simulator_judge_prompt().format(
        now=now,
        tool_spec=json.dumps(tool_spec, indent=2, default=str),
        tool_args=tool_args,
        other_endpoints=other_endpoints,
        credentials=(
            "\n"
            + tool_simulator_judge_credentials().format(
                parameters=", ".join(f"`{parameter}`" for parameter in credential_parameters)
            )
            + "\n"
            if credential_parameters
            else ""
        ),
        caller="\n" + tool_simulator_judge_caller().format(caller=caller) + "\n" if caller else "",
        earlier_calls=earlier_calls,
        response=content,
    )
    explanation, _rating, passed, readable = await run_inline_judge_ex(models, judge_alias, prompt)
    return (passed, explanation) if readable else None


async def _simulate_tool_response(
    models: dict,
    tool_spec: dict,
    tool_call: dict,
    conversation_messages: list,
    *,
    now: str,
    credential_parameters: tuple[str, ...] = (),
    caller: str = "",
    judge: bool = False,
    max_attempts: int = 2,
    judge_alias: str = MODEL_JUDGE,
    other_endpoints: str = "None.",
) -> SimulatedResponse:
    """Use the strong frontier model to simulate a realistic API response.

    The simulator sees what the service would: the call, the service's other
    endpoints (``other_endpoints``, from ``_other_endpoints``), whose
    parameters the IDs and values it returns fit, the calls made
    earlier in the conversation with their responses (``_earlier_calls``),
    which it keeps consistent with, ``now``, the service's clock, which moves
    as the conversation goes on, after which no event has a result yet though
    what is planned for later, such as a schedule, already exists,
    and ``caller``, the service's records on the person, which the user's
    own things carry and nothing else does. The conversation's
    messages reach neither it nor its judge. The probe checks a call's
    credentials before this runs, so when ``credential_parameters`` names
    any, the simulator is told they are valid rather than left to judge a
    random value.

    Each of up to ``max_attempts`` attempts must be JSON and, when ``judge`` is
    set, pass ``judge_model``'s check against the call; a rejected attempt is
    regenerated with the judge's reason as feedback, and a reply that is not
    JSON, which is output cut off at the token cap, with a request for a
    shorter one. A JSON reply is kept compact, as a service sends it. When every attempt is
    used, the last JSON response stands, marked ``unverified``; when none was
    JSON, a 502 error payload, so every tool result a conversation stores
    parses. A judge that fails to answer leaves the response standing,
    unverified, rather than stopping the conversation, as does one whose
    verdict can't be read even after it is asked to reformat. Nothing is recorded on
    the conversation here: the caller appends the result.
    """
    tool_args = _arguments_text(tool_call.get("function", {}).get("arguments", "{}"))
    earlier_calls = _earlier_calls(conversation_messages, tool_spec.get("name", ""), tool_args)
    prompt = tool_simulator_prompt().format(
        tool_spec=json.dumps(tool_spec, indent=2, default=str),
        tool_args=tool_args,
        other_endpoints=other_endpoints,
        earlier_calls=earlier_calls,
        now=now,
    )
    if credential_parameters:
        prompt += "\n\n" + tool_simulator_credentials().format(
            parameters=", ".join(f"`{parameter}`" for parameter in credential_parameters),
        )
    if caller:
        prompt += "\n\n" + tool_simulator_caller().format(caller=caller)

    result = SimulatedResponse(content=_INVALID_RESPONSE)
    accepted: str | None = None
    feedback = ""
    for attempt in range(max_attempts):
        resp = await acall_llm(models, MODEL_API_RESPONSE, [{"role": "user", "content": prompt + feedback}])
        content = resp.get("content", "{}") if isinstance(resp, dict) else "{}"
        try:
            content = json.dumps(json.loads(content), ensure_ascii=False, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            if attempt < max_attempts - 1:
                result.rerolls += 1
                logger.debug("  |-- api_response_model: invalid JSON, retrying")
            feedback = "\n\n" + tool_simulator_cut_off()
            continue
        accepted = content
        if not judge:
            break
        try:
            verdict = await _judge_tool_response(
                models,
                content,
                tool_spec=tool_spec,
                tool_args=tool_args,
                earlier_calls=earlier_calls,
                now=now,
                credential_parameters=credential_parameters,
                caller=caller,
                judge_alias=judge_alias,
                other_endpoints=other_endpoints,
            )
        except Exception as e:  # a judge outage leaves the response standing, flagged
            logger.warning("  |-- %s raised on a simulated response: %s: %s", judge_alias, type(e).__name__, e)
            result.unverified = f"the judge gave no verdict ({type(e).__name__})"
            break
        if verdict is None:
            result.unverified = "the judge's verdict could not be read"
            break
        passed, explanation = verdict
        result.verdicts.append((passed, explanation))
        if passed:
            break
        feedback = "\n\n" + tool_simulator_feedback().format(response=content, reason=explanation)
    if accepted is None:
        logger.warning(
            "  |-- api_response_model: no JSON in %d attempts for %s; answering with a 502",
            max_attempts,
            tool_call.get("function", {}).get("name", "?"),
        )
        return result
    result.content = accepted
    if result.verdicts and not result.verdicts[-1][0] and result.unverified is None:
        result.unverified = f"the judge rejected all {len(result.verdicts)} attempts it saw"
    return result


# ---------------------------------------------------------------------------
# Tool-calling probe
# ---------------------------------------------------------------------------


@register_probe
class ToolCallingProbe(ToolExecutionMixin, ToolCallingMixin, BaseProbe):
    """Tool-calling probe.

    Tests the assistant model's native tool-calling capabilities using
    persona-grounded, realistic user requests. The trajectory
    judge lives in ``evaluator/scorers/tool_use.py`` and is invoked
    by the out-of-sim trajectory evaluator. The simulator only emits
    trajectories. ``ToolCallingMixin`` blocks early-stop until at least
    one tool call has happened, so a conversation with no tool calls ran
    every turn; it stays scoreable, because not calling a tool is the
    assistant's behaviour, not a failure of the probe's protocol.

    ``ToolExecutionMixin`` (``tool_loop_mode="multi"``) supplies
    ``after_assistant_turn``: after each round of tool calls the assistant is
    offered the tools again, so it can chain calls within a turn, until it
    replies without calling one or the turn reaches the mixin's call cap.
    This probe supplies only the per-call behaviour (verify + simulate) via
    ``execute_tool_call`` and records ``tool_call_count_per_turn`` in
    ``on_tool_round_complete``.
    """

    label = "tool_calling"
    # Bump on any semantic change to the probe's prompts or task derivation.
    # One variant: the tool category is the row's ``toolset_category``.
    prompt_version = "v1.12"
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
    tool_loop_mode = "multi"
    # A turn changes the transcript, the metadata, the outcome record and
    # ``held_credentials``, so a rejected turn can be undone and regenerated.
    supports_assistant_resampling = True
    #: The warning a row carries for each simulated response that stands
    #: without a passing judge verdict.
    unverified_response_warning_kind = "unverified_tool_response"
    preview_fields = (
        PreviewField("toolset_name", "Toolset"),
        PreviewField("toolset_category", "Toolset category"),
        PreviewField("num_tools", "Tools available"),
    )

    class Options(BaseProbe.Options):
        """The run's settings for this probe, set as ``--probe-option tool_calling.<name>=<value>``."""

        #: The most tools a conversation offers. A toolset with no more tools
        #: than this is offered whole, in its catalog order; a larger one is
        #: drawn down to it.
        max_tools: int = Field(default=15, ge=1)
        #: The row column holding the toolset's tools.
        tools_column: str = "tools"
        #: The date each conversation takes place on. The simulated user, the API
        #: response simulator and, with ``share_user_context``, the assistant all
        #: work from it, so the dates in requests and in tool results agree.
        today: date = date(2026, 11, 6)
        #: Open the first message the assistant sees with the date and where the
        #: user is, as the assistant's app would attach them.
        share_user_context: bool = True
        #: Have a judge check each simulated tool response before the assistant
        #: sees it, regenerating a rejected one.
        judge_responses: bool = True
        #: The models-file alias of that judge, so the checks can run on another
        #: model than the per-turn judges'. A models file without it uses
        #: ``judge_model``.
        response_judge_model_alias: str = "api_response_judge_model"
        #: The attempts a simulated response gets: a reply that is not JSON, or
        #: that the judge rejects, uses one.
        max_response_attempts: int = Field(default=2, ge=1)

    @classmethod
    def model_aliases(cls, cfg: Any) -> tuple[str, ...]:
        """The model that judges simulated tool responses; without it, ``judge_model`` does."""
        return (cls.options_for(cfg).response_judge_model_alias,)

    @classmethod
    def asset_problems(cls) -> list[str]:
        """The prompts file the prompt defaults would mask."""
        if not load_prompt_file(_SCENARIO):
            return [f"{_SCENARIO}/prompts.yaml is missing or empty"]
        return []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        tools_column = self.options.tools_column
        if tools_column not in self._data:
            raise ValueError(f"tool_calling: column {tools_column!r} missing in data")

        tools_raw = self._data[tools_column]
        all_tools = _normalize_tool_list(tools_raw)
        if len(all_tools) <= self.options.max_tools:
            self.tool_subset = list(all_tools)
        else:
            # Drawn from the row's own identity rather than the process RNG. The
            # subset shapes the entire conversation but is not part of the
            # trajectory identity, so an identity only means something if the
            # same row is always handed the same tools.
            self.tool_subset = random.Random(_tool_subset_seed(self._data)).sample(all_tools, self.options.max_tools)

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
        # The credentials each offered tool sets a new value for, by tool name.
        self.credentials_set = {
            _function_spec(entry).get("name"): sets for entry in self.tool_subset if (sets := credentials_set_by(entry))
        }

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
            today=_format_day(self.options.today),
            assistant_context=assistant_context() if self.options.share_user_context else "",
        )

    def user_context(self) -> str:
        """The conversation's date and the persona's location, as ``[2026-11-06 · Austin, Texas, USA]``.

        The same place the persona's prompt states. A persona naming no place
        gives the date alone, and ``share_user_context`` off gives nothing.
        """
        if not self.options.share_user_context:
            return ""
        place = persona_location(self._persona)
        day = self.options.today.isoformat()
        return f"[{day} · {place}]" if place else f"[{day}]"

    def _caller_records(self, tool_spec: dict) -> str:
        """What the service holds on the person: their name and where they live, as ``Ana Souza, who lives in Austin``.

        Only a tool that takes the reference of one of the person's own
        things, such as a parcel's tracking number, gets them; a profile,
        a search or a lookup of anything public gets ``""``. A tool that takes
        a place gets the name alone, so its place comes only from the call.
        A persona naming neither gives ``""``.
        """
        if not _names_own_thing(tool_spec):
            return ""
        name = persona_full_name(self._persona)
        place = "" if _takes_place(tool_spec) else persona_location(self._persona)
        if name and place:
            return f"{name}, who lives in {place}"
        return name or (f"a person who lives in {place}" if place else "")

    def snapshot_turn_state(self) -> dict[str, str]:
        return dict(self.held_credentials)

    def restore_turn_state(self, snapshot: dict[str, str]) -> None:
        self.held_credentials = dict(snapshot)

    def seed_state_metadata(self, state: ConversationState) -> None:
        # The date the conversation is set on, so its dates can be read later.
        state.metadata.setdefault("today", self.options.today.isoformat())

    def get_assistant_system_prompt(self) -> str:
        # Empty as shipped; see the accessor's docstring.
        return render_prompt(assistant_system_prompt(), self._data)

    def get_tools_for_assistant(self) -> list | None:
        return self.openai_tools

    def user_turn_policy(self) -> UserTurnPolicy:
        """The user sees that the assistant used its tools, as a chat app shows it, and never their results."""
        return UserTurnPolicy(show_tool_use=True)

    def format_assistant_judge_prompt(self, assistant_response: str, conversation_history: str) -> str:
        """The per-turn judge's rubric, listing the tools the assistant was offered, as its gate's does."""
        return render_prompt(
            assistant_judge_prompt(),
            self._data,
            tools=self.tools_for_judge,
            assistant_response=assistant_response,
            conversation_history=conversation_history,
        )

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
        ``ToolExecutionMixin`` drives the message plumbing and the
        assistant's re-calls. ``name`` is the parsed function name; ``tc`` is the
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
            judge_alias = self._response_judge_alias(models)
            simulated = await _simulate_tool_response(
                models,
                tool_spec,
                tc,
                state.messages,
                now=_format_now(self.options.today, _user_messages(state.messages)),
                credential_parameters=tuple(declared),
                caller=self._caller_records(tool_spec),
                judge=self.options.judge_responses,
                max_attempts=self.options.max_response_attempts,
                judge_alias=judge_alias,
                other_endpoints=_other_endpoints(self.tool_subset, name),
            )
            simulated_response = simulated.content
            self._record_simulation(
                state, simulated, name, turn_idx=turn_idx, call_idx=call_idx, judge_alias=judge_alias
            )
        else:
            simulated_response = json.dumps({"error": f"Unknown tool: {name}"})

        # An accepted call that sets a credential, such as a password change,
        # leaves the user holding the value it passed.
        for parameter, credential in self.credentials_set.get(name, {}).items():
            if isinstance(args.get(parameter), str) and args[parameter]:
                self.held_credentials[credential] = args[parameter]

        state.metadata.setdefault("tools_called", []).append(name)
        return simulated_response

    def _response_judge_alias(self, models: dict) -> str:
        alias = self.options.response_judge_model_alias
        return alias if alias in models else MODEL_JUDGE

    def _record_simulation(
        self,
        state: ConversationState,
        simulated: SimulatedResponse,
        name: str,
        *,
        turn_idx: int,
        call_idx: int,
        judge_alias: str = MODEL_JUDGE,
    ) -> None:
        """Count and trace how one call's simulated response was reached."""
        for _ in range(simulated.rerolls):
            state.outcome.inc_api_response_rerolls()
            state.outcome.add_trace(
                SimulationTrace(
                    kind=TraceKind.API_RESPONSE_REROLL,
                    turn_idx=turn_idx,
                    call_idx=call_idx,
                    model_alias=MODEL_API_RESPONSE,
                    detail=f"invalid JSON for {name}",
                )
            )
        for passed, explanation in simulated.verdicts:
            if not passed:
                state.outcome.inc_api_response_rejections()
            state.outcome.add_trace(
                SimulationTrace(
                    kind=TraceKind.API_RESPONSE_JUDGE,
                    turn_idx=turn_idx,
                    call_idx=call_idx,
                    model_alias=judge_alias,
                    rating="success" if passed else "failure",
                    detail=(explanation or "")[:500],
                    extra={"tool_name": name},
                )
            )
        if simulated.unverified:
            state.outcome.add_warning(
                self.unverified_response_warning_kind,
                turn_idx=turn_idx,
                detail=f"{name}: {simulated.unverified}",
            )

    def on_tool_round_complete(self, state: ConversationState, n_calls: int) -> None:
        # A turn can hold several rounds, so each round adds to the total of
        # the turn it belongs to: the turn of the latest user message.
        counts = state.metadata.setdefault("tool_call_count_per_turn", [])
        turn = sum(1 for message in state.messages if message.get("role") == "user") - 1
        while len(counts) <= turn:
            counts.append(0)
        counts[turn] += n_calls

    def build_result_extras(self, state: ConversationState) -> dict:
        return {
            "num_tool_calls": len(state.metadata.get("tools_called", [])),
            "tool_subset": json.dumps(self.tool_subset, ensure_ascii=False, default=str),
        }
