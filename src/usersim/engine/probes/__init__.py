# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The built-in probes, one package each.

Every probe here is a ``BaseProbe`` subclass that ``@register_probe`` adds to
the registry when its module is imported. The registry imports the ``probe``
module of each package in ``usersim.engine.core.builtin_packages.BUILTIN_PACKAGES``
the first time a probe is looked up, which runs the decorators. ``docs/probes.md`` is the
catalog: which probe catches which failure mode.

See ``usersim.engine.core.probes`` for the substrate
(``BaseProbe``, ``BankBackedProbe``, mixins, ``LocalePromptPack``,
``@register_probe``) every probe builds on.
"""
