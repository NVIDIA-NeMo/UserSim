# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conformance checks an extension author can run against their own code.

The registries accept anything advertised under the right entry-point group.
That is the point -- but it means a probe or scorer with a subtly wrong shape
registers successfully and then fails somewhere deep in a run, usually after
spending money. These checks move that failure to the author's own test suite.

Each check comes in two forms: a ``*_problems`` function returning a list of
human-readable findings, and an ``assert_*`` wrapper that raises with all of
them at once. The list form exists so a caller can report every problem in
one pass instead of fixing them one failure at a time.

Usage from an extension's own tests::

    from usersim.testing import assert_probe_conforms

    def test_my_probe_conforms():
        assert_probe_conforms("my_probe")

These checks are also run against the built-in probes and scorers, so the
suite cannot drift into describing a contract this project does not itself
keep.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "HostedParityReport",
    "ScriptedModel",
    "assert_hosted_parity",
    "assert_probe_conforms",
    "assert_scorer_conforms",
    "hosted_parity_report",
    "probe_conformance_problems",
    "scorer_conformance_problems",
]

#: Methods the simulation loop calls on every probe. A probe that inherits
#: the abstract versions without overriding them registers fine and then
#: raises once a run reaches it.
_REQUIRED_PROBE_METHODS = (
    "get_user_system_prompt",
    "get_assistant_system_prompt",
)

#: Hooks the loop awaits. Overriding one with a plain ``def`` registers
#: fine and then raises ``TypeError`` mid-conversation, once the loop tries
#: to await whatever it returned.
_AWAITED_PROBE_HOOKS = (
    "after_assistant_turn",
    "get_verbatim_first_user_turn",
    "format_followup_user_instructions",
    "is_capitulation_detected",
    "execute_tool_call",
    "run_dispatch",
)


def probe_conformance_problems(probe: str | type) -> list[str]:
    """Findings for one probe, by label or by class. Empty means conforming."""
    from usersim.engine.core._assets import probe_assets_dir
    from usersim.engine.core.probes import (
        _PROBE_REGISTRY,
        BaseProbe,
        known_probes,
    )

    problems: list[str] = []

    if isinstance(probe, str):
        if probe not in known_probes():
            return [
                f"{probe!r} is not registered. Advertise it under the "
                f"'usersim.probes' entry-point group, or import the module "
                f"defining it. Registered: {', '.join(known_probes()) or '(none)'}"
            ]
        cls = _PROBE_REGISTRY[probe]
    else:
        cls = probe

    label = getattr(cls, "label", None)
    if not isinstance(label, str) or not label or label == "<unset>":
        problems.append(f"{cls.__qualname__}.label must be a non-empty string; got {label!r}")
        return problems

    if not issubclass(cls, BaseProbe):
        problems.append(f"{cls.__qualname__} must subclass BaseProbe")

    if _PROBE_REGISTRY.get(label) is not cls:
        problems.append(
            f"{label!r} resolves to {_PROBE_REGISTRY.get(label)!r}, not "
            f"{cls.__qualname__}. Two probes cannot share a label."
        )

    # @register_probe sets these on the defining module; the dispatcher
    # reads them from there rather than from the class.
    module = getattr(cls, "__module__", None)
    if module:
        import sys

        mod = sys.modules.get(module)
        for const in ("PROBE_FAMILY", "PROMPT_VERSION", "PROBE_VARIANTS"):
            if getattr(mod, const, None) in (None, "", ()):
                problems.append(
                    f"{module}.{const} is unset. Use @register_probe rather than assigning to the registry directly."
                )

    for method in _REQUIRED_PROBE_METHODS:
        impl = getattr(cls, method, None)
        if impl is None:
            problems.append(f"{cls.__qualname__}.{method} is missing")
        elif getattr(BaseProbe, method, None) is impl:
            problems.append(
                f"{cls.__qualname__}.{method} is not overridden; the loop "
                f"calls it on every turn and the base version raises."
            )

    import inspect

    for hook in _AWAITED_PROBE_HOOKS:
        impl = getattr(cls, hook, None)
        if impl is None or getattr(BaseProbe, hook, None) is impl:
            continue
        if not inspect.iscoroutinefunction(impl):
            problems.append(
                f"{cls.__qualname__}.{hook} must be declared with 'async def'. "
                f"The loop awaits this hook, so a plain 'def' override raises "
                f"TypeError partway through a conversation."
            )

    # A probe that overrides run_dispatch must still be hostable: the external
    # episode runtime resumes it by passing the state it owns and suppressing
    # re-seeding. An override with the pre-hosting signature raises TypeError.
    dispatch = getattr(cls, "run_dispatch", None)
    if dispatch is not None and getattr(BaseProbe, "run_dispatch", None) is not dispatch:
        parameters = inspect.signature(dispatch).parameters
        for required in ("state", "seed_state"):
            if required not in parameters:
                problems.append(
                    f"{cls.__qualname__}.run_dispatch does not accept {required!r}. "
                    f"A custom dispatch must accept 'state' and 'seed_state' so an "
                    f"external host can resume the episode; see "
                    f"docs/engine/EXTERNAL_PROBE_RUNTIME.md."
                )

    assets = probe_assets_dir(label)
    if not assets.is_dir():
        problems.append(
            f"no assets for {label!r} at {assets}. Ship them in the package, or add their root to USERSIM_ASSET_PATH."
        )

    return problems


def assert_probe_conforms(probe: str | type) -> None:
    """Raise ``AssertionError`` listing every conformance problem at once."""
    problems = probe_conformance_problems(probe)
    if problems:
        name = probe if isinstance(probe, str) else probe.__qualname__
        raise AssertionError(f"probe {name!r} does not conform:\n  - " + "\n  - ".join(problems))


async def scorer_conformance_problems(
    name: str,
    *,
    sample_row: dict[str, Any] | None = None,
) -> list[str]:
    """Findings for one scorer. Empty means conforming.

    ``sample_row`` is passed to the scorer to check it returns the mapping
    the evaluator expects. The default is an empty row, which a scorer must
    tolerate: the evaluator calls it for trajectories that may be missing
    any given field.

    Awaitable, because a scorer is: the evaluator awaits it, so checking it
    means awaiting it here too.
    """
    import inspect

    from usersim.engine.evaluator.scorers import _REGISTRY, list_scorers

    if name not in _REGISTRY:
        return [
            f"{name!r} is not registered. Advertise it under the "
            f"'usersim.scorers' entry-point group. "
            f"Registered: {', '.join(sorted(list_scorers())) or '(none)'}"
        ]

    problems: list[str] = []
    fn: Callable[..., Any] = _REGISTRY[name]
    if not callable(fn):
        return [f"{name!r} is registered to a non-callable: {fn!r}"]

    if not inspect.iscoroutinefunction(fn):
        return [
            f"{name!r} is not an async function. The evaluator awaits every scorer, so declare it with 'async def'."
        ]

    try:
        result = await fn(dict(sample_row or {}), {})
    except Exception as exc:
        problems.append(
            f"raised {type(exc).__name__} on an empty row: {exc}. The "
            f"evaluator calls scorers on trajectories that may be missing "
            f"any given field, so this must degrade rather than raise."
        )
        return problems

    if not isinstance(result, dict):
        problems.append(
            f"returned {type(result).__name__}, expected a dict the evaluator can merge into the eval cell."
        )
    return problems


async def assert_scorer_conforms(
    name: str,
    *,
    sample_row: dict[str, Any] | None = None,
) -> None:
    """Raise ``AssertionError`` listing every conformance problem at once."""
    problems = await scorer_conformance_problems(name, sample_row=sample_row)
    if problems:
        raise AssertionError(f"scorer {name!r} does not conform:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# Hosted parity
# ---------------------------------------------------------------------------


class ScriptedModel:
    """A deterministic stand-in for a configured model.

    Returns the same reply for the same role, and a tool call whenever the
    assistant or the user is offered tools and no tool result is in view yet,
    which is enough to drive any probe's loop to completion without a provider.
    """

    #: A real model id: probes that grade the assistant's own identity refuse
    #: to run against a name no spec can attribute to a developer.
    model_name = "nvidia/llama-3.1-nemotron-70b-instruct"

    #: Replies by role. Override on a subclass for probe-specific wording.
    CONTENT = {
        "api": '{"ok": true}',
        "assistant": "Scripted assistant response.",
        "judge": "<explanation>valid scripted turn</explanation><rating>success</rating>",
        "summary": "yes",
        "user": "Could you explain that a little more?",
    }

    def __init__(self, role: str, prompts: list[dict[str, Any]] | None = None) -> None:
        self.role = role
        self.prompts: list[dict[str, Any]] = prompts if prompts is not None else []

    def _tool_calls(self, messages: Sequence[Any], tools: Any) -> list[dict[str, Any]] | None:
        # Either side of the conversation calls a tool it is offered, so a probe
        # whose simulated user acts through a tool is exercised as a model would.
        if self.role not in ("assistant", "user") or not tools:
            return None
        if any(str(_message_field(message, "role")) == "tool" for message in messages):
            return None
        function = tools[0]["function"]
        return [
            {
                "id": f"scripted-call-{len(self.prompts)}",
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": json.dumps(_example_arguments(function.get("parameters", {})), sort_keys=True),
                },
            }
        ]

    async def acompletion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        """Record the prompt and return the scripted reply."""
        tools = kwargs.get("tools")
        self.prompts.append(
            {
                "role": self.role,
                "messages": [_plain_message(message) for message in messages],
                "tools": copy.deepcopy(tools),
            }
        )
        tool_calls = self._tool_calls(messages, tools)
        return SimpleNamespace(
            message=SimpleNamespace(
                content="" if tool_calls else self.CONTENT[self.role],
                reasoning_content=f"{self.role} reasoning",
                tool_calls=tool_calls,
            ),
            usage=None,
        )


@dataclass(frozen=True)
class HostedParityReport:
    """Outcome of running one episode standalone and through a host."""

    direct_row: dict[str, Any]
    hosted_row: dict[str, Any]
    differences: list[str]
    direct_prompts: list[dict[str, Any]]
    hosted_prompts: list[dict[str, Any]]


async def hosted_parity_report(
    probe: str,
    *,
    config: Any,
    persona: Mapping[str, Any],
    data: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
    language: str = "English",
    model_factory: Callable[[str], Any] = ScriptedModel,
) -> HostedParityReport:
    """Run one episode twice -- standalone and hosted -- and compare.

    A probe is only hostable if an external host driving it through
    ``ProbeEpisodeRuntime`` reproduces what ``usersim simulate`` would have
    produced for the same inputs. That holds automatically for probes built on
    the shared loop; a probe with its own ``run_dispatch`` or its own tool
    execution has to be checked.
    """
    from usersim.engine.core.episode_runtime import (
        ActivationRequest,
        ActivationResult,
        EpisodeLifecycleComplete,
        ProbeEpisodeRuntime,
    )
    from usersim.engine.generator import ConversationSimulatorGenerator

    aliases = ("api_response_model", "assistant_model", "judge_model", "summary_model", "user_model")
    role_of = {
        "api_response_model": "api",
        "assistant_model": "assistant",
        "judge_model": "judge",
        "summary_model": "summary",
        "user_model": "user",
    }

    def _build(prompts: list[dict[str, Any]]) -> dict[str, Any]:
        models = {}
        for alias in aliases:
            model = model_factory(role_of[alias])
            model.prompts = prompts
            models[alias] = model
        return models

    def _runtime(models: Mapping[str, Any]) -> Any:
        return ProbeEpisodeRuntime(
            probe_type=probe,
            persona=dict(persona),
            locale=config.locale,
            language=language,
            models=dict(models),
            config=config,
            data=dict(data or {}),
            profile=dict(profile) if profile is not None else None,
        )

    direct_prompts: list[dict[str, Any]] = []
    direct_models = _build(direct_prompts)
    direct_runtime = _runtime(direct_models)
    generator = ConversationSimulatorGenerator.with_models(config, direct_models)
    direct_row = await generator.agenerate(direct_runtime.preamble.to_row())

    hosted_prompts: list[dict[str, Any]] = []
    hosted_models = _build(hosted_prompts)
    hosted_runtime = _runtime(hosted_models)
    event = await hosted_runtime.advance()
    while isinstance(event, ActivationRequest):
        kwargs = dict(event.parameters)
        if event.tools:
            kwargs["tools"] = [copy.deepcopy(tool) for tool in event.tools]
        model = model_factory(event.role)
        model.prompts = hosted_prompts
        reply = await model.acompletion(list(event.messages), **kwargs)
        response = {"role": "assistant", "content": reply.message.content or ""}
        if reply.message.reasoning_content is not None:
            response["reasoning_content"] = reply.message.reasoning_content
        if reply.message.tool_calls:
            response["tool_calls"] = copy.deepcopy(reply.message.tool_calls)
        event = await hosted_runtime.advance(ActivationResult(activation_id=event.activation_id, response=response))
    if not isinstance(event, EpisodeLifecycleComplete):  # pragma: no cover - defensive
        raise RuntimeError("hosted episode did not complete")
    hosted_row = event.result

    differences: list[str] = []
    # A probe that executes tool calls without notifying runs correctly but is
    # invisible to its host: the payloads never reach the agent's trajectory.
    observed = await hosted_runtime.executed_tool_calls()
    tool_messages = [
        message
        for message in _parity_value(hosted_row.get("conversation_messages")) or []
        if isinstance(message, Mapping) and message.get("role") == "tool"
    ]
    if tool_messages and not observed:
        differences.append(
            f"{len(tool_messages)} tool message(s) were produced but none were reported through "
            f"on_tool_call_executed; a probe that executes its own tool calls must notify"
        )
    elif len(tool_messages) != len(observed):
        differences.append(
            f"{len(tool_messages)} tool message(s) in the transcript but {len(observed)} reported "
            f"through on_tool_call_executed"
        )

    for name in sorted(set(direct_row) | set(hosted_row)):
        if name in _PARITY_IGNORED_COLUMNS:
            continue
        left = _parity_value(direct_row.get(name))
        right = _parity_value(hosted_row.get(name))
        if left != right:
            differences.append(f"{name}: standalone {left!r} != hosted {right!r}")
    if direct_prompts != hosted_prompts:
        for index, (left, right) in enumerate(zip(direct_prompts, hosted_prompts)):
            if left != right:
                differences.append(f"prompt {index} ({left['role']}) differs between standalone and hosted")
                break
        if len(direct_prompts) != len(hosted_prompts):
            differences.append(
                f"model was called {len(direct_prompts)} times standalone and {len(hosted_prompts)} times hosted"
            )

    return HostedParityReport(
        direct_row=direct_row,
        hosted_row=hosted_row,
        differences=differences,
        direct_prompts=direct_prompts,
        hosted_prompts=hosted_prompts,
    )


#: Wall-clock fields differ between any two runs and say nothing about parity.
_PARITY_IGNORED_COLUMNS = frozenset({"wall_clock_s", "wall_clock_s_by_alias"})


def _parity_value(value: Any) -> Any:
    """Normalize one column for comparison, dropping timing noise."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
        value = parsed
    if isinstance(value, dict):
        return {key: _parity_value(item) for key, item in value.items() if key not in _PARITY_IGNORED_COLUMNS}
    if isinstance(value, list):
        return [_parity_value(item) for item in value]
    return value


def _message_field(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def _plain_message(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return copy.deepcopy(dict(message))
    role = getattr(message, "role", "user")
    value: dict[str, Any] = {
        "role": str(getattr(role, "value", role)),
        "content": getattr(message, "content", "") or "",
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        value["tool_calls"] = [
            call.model_dump() if hasattr(call, "model_dump") else copy.deepcopy(call) for call in tool_calls
        ]
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        value["tool_call_id"] = str(tool_call_id)
    return value


def _example_arguments(schema: Mapping[str, Any]) -> Any:
    if enum := schema.get("enum"):
        return enum[0]
    schema_type = schema.get("type")
    if schema_type == "object":
        required = set(schema.get("required", []))
        return {
            name: _example_arguments(value) for name, value in schema.get("properties", {}).items() if name in required
        }
    return {"array": [], "boolean": True, "integer": 1, "number": 1, "string": "example"}.get(schema_type, "example")


async def assert_hosted_parity(probe: str, **kwargs: Any) -> None:
    """Raise ``AssertionError`` listing every hosted/standalone difference."""
    report = await hosted_parity_report(probe, **kwargs)
    if report.differences:
        joined = "\n  - ".join(report.differences)
        raise AssertionError(f"{probe} does not reproduce a standalone run when hosted:\n  - {joined}")
