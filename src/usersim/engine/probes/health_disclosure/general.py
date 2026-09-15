# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""health_general_disclosure — a general US health patient discloses to a general health assistant.

Thin client module: all behavior lives in ``HealthDisclosureProbe``; this only
binds the label + client config and registers the probe with its own family
(so it gets its own judge axes). The persona carries a hidden clinical profile
(ongoing conditions + medications); the gated content is a downplayed red-flag
symptom + a quiet medication lapse.
"""
from __future__ import annotations

from usersim.engine.core.clinical_profile_bank import (
    load_clinical_profile_bank_for_client,
)
from usersim.engine.core.probes import register_probe
from usersim.engine.probes.health_disclosure.base import HealthDisclosureProbe
from usersim.engine.probes.health_disclosure.clients import CLIENTS

_CFG = CLIENTS["health_general_disclosure"]
_LABEL = "health_general_disclosure"
PROBE_FAMILY: str = _CFG["family"]
PROMPT_VERSION: str = _CFG["prompt_version"]
# "default" = plain incremental-disclosure user; "guarded" = enable the auditable
# Guard-paced move-space (GuardedMoveMixin). Per-trajectory, set via probe_variant.
PROBE_VARIANTS: list[str] = ["default", "guarded"]

# Module-level alias so tests can monkeypatch bank loading (mirrors the other
# BankBackedProbe clients, e.g. therapy).
_load_bank = load_clinical_profile_bank_for_client


def _bank_loader():
    # Locale-independent, per-client (a customer overrides via profiles_env).
    return _load_bank(_LABEL, env_override=_CFG.get("profiles_env"))


@register_probe(family=PROBE_FAMILY, prompt_version=PROMPT_VERSION,
                variants=tuple(PROBE_VARIANTS))
class HealthGeneralDisclosureProbe(HealthDisclosureProbe):
    label = _LABEL
    CLIENT = _CFG
    bank_loader = staticmethod(_bank_loader)
    bank_version_key = _LABEL  # per-client (locale-independent) provenance slot
