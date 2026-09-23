# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

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

from typing import Any, Callable

__all__ = [
    "assert_probe_conforms",
    "assert_scorer_conforms",
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
