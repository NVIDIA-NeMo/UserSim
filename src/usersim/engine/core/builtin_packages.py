# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The built-in probe packages, in one ordered roster.

Each package under ``usersim.engine.probes`` keeps its probe in ``probe.py``,
and, when it has them, its scorer in ``scorer.py``, its capability rows in
``capabilities.py`` and its asset domain in ``asset_domain.py``. The probe
registry, the scorer registry, the capability report and ``usersim gen-assets``
each find the modules of the kind they read from this roster, by name, so a
built-in is registered by its folder and one line here.

The roster's order is the order of the capability report's rows. Packages are
named, not imported, because every probe module imports the probe substrate;
this module imports nothing from ``usersim``, so any layer can read it.
"""

from __future__ import annotations

import importlib.util

#: The package every built-in probe package lives under.
BUILTIN_ROOT = "usersim.engine.probes"

#: The built-in probe packages, in the order the capability report lists their rows.
BUILTIN_PACKAGES: tuple[str, ...] = (
    "general_open_ended",
    "general_educational",
    "sov_ai_facts",
    "sov_ai_dynamic",
    "sov_ai_multilingual_parity",
    "tool_calling",
    "safety_chat_pressure",
    "safety_agentic",
    "identity_disclosure",
    "financial_services",
    "health_disclosure",
)


def builtin_modules(kind: str) -> tuple[str, ...]:
    """The ``<package>.<kind>`` module of each built-in package that has one, in roster order.

    ``kind`` is a module name such as ``"probe"``, ``"scorer"``,
    ``"capabilities"`` or ``"asset_domain"``. Finding a module imports its
    package's ``__init__``, which holds a docstring only, and not the module.
    """
    modules = (f"{BUILTIN_ROOT}.{package}.{kind}" for package in BUILTIN_PACKAGES)
    return tuple(module for module in modules if importlib.util.find_spec(module) is not None)
