# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe modules — the 8 measurement protocols the framework ships.

Every probe lives under this package as a ``BaseProbe`` subclass that
registers itself in the substrate's ``_PROBE_REGISTRY`` via
``@register_probe`` at import time. The bootstrap function in
``usersim.engine.generator._bootstrap_probes`` imports every
shipped probe module to fire the decorators before the dispatcher
starts resolving probe labels.

Three families:

- **general** — ``general_open_ended``, ``general_educational``,
  ``tool_calling``. Persona-grounded conversations with no curated
  asset bank. Quality judgment lives in the out-of-sim trajectory
  evaluator.
- **inclusion** — ``sov_ai_facts``, ``sov_ai_dynamic``,
  ``sov_ai_multilingual_parity``. Sovereign-AI readiness probes that
  surface where a model fails to serve users in non-en_US locales.
- **safety** — ``safety_chat_pressure``, ``safety_agentic``. Refusal
  robustness under conversational pressure and bounded action-space
  agentic behavior under untrusted input.

See ``usersim.engine.core.probes`` for the substrate
(``BaseProbe``, ``BankBackedProbe``, mixins, ``LocalePromptPack``,
``@register_probe``) every probe builds on.
"""
