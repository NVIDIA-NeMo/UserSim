# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four health_disclosure probes, one per client in ``clients.py``.

All behavior lives in ``HealthDisclosureProbe`` (``base.py``). Each class here
binds a label and its client's configuration, loads that client's clinical
profile bank, and registers under its own family, so it gets its own judge axes:

- ``health_therapy_disclosure``: a therapy patient discloses to a therapist.
- ``health_triage_disclosure``: a patient discloses to a care-navigation
  (triage) assistant.
- ``health_decision_support_disclosure``: role-reversed, the simulated user is
  a clinician presenting a case and the assistant under test is the
  decision-support tool.
- ``health_general_disclosure``: a patient with an ongoing condition discloses
  to a general health assistant; the persona carries a hidden clinical profile
  (conditions and medications), and the gated content is a downplayed red-flag
  symptom and a quiet medication lapse.

Each probe has two variants: ``default``, a plain incremental-disclosure user,
and ``guarded``, which turns on the auditable, Guard-paced move-space
(``GuardedMoveMixin``) for the trajectory through ``probe_variant``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from usersim.engine.probes.health_disclosure.bank import load_clinical_profile_bank_for_client
from usersim.engine.probes.health_disclosure.base import HealthDisclosureProbe
from usersim.engine.probes.health_disclosure.clients import CLIENTS
from usersim.extension import register_probe

_VARIANTS: tuple[str, ...] = ("default", "guarded")


def _registered(label: str) -> Callable[[type], type]:
    """``@register_probe`` with the family and prompt version the client declares for ``label``."""
    cfg = CLIENTS[label]
    return register_probe(family=cfg["family"], prompt_version=cfg["prompt_version"], variants=_VARIANTS)


def _bank_loader(label: str) -> Callable[[], Any]:
    """The client's clinical profile bank, the same for every locale (``profiles_env`` overrides it)."""
    return lambda: load_clinical_profile_bank_for_client(label, env_override=CLIENTS[label].get("profiles_env"))


@_registered("health_therapy_disclosure")
class TherapyDisclosureProbe(HealthDisclosureProbe):
    label = "health_therapy_disclosure"
    description = (
        "Tests whether a therapy assistant draws out what a guarded client discloses "
        "only gradually, including self-harm or crisis risk. In the guarded variant the "
        "user paces disclosure by plan, and the health_disclosure_concealment scorer "
        "measures how much the assistant drew out."
    )
    CLIENT = CLIENTS[label]
    bank_loader = staticmethod(_bank_loader(label))
    bank_version_key = label  # per-client (locale-independent) provenance slot


@_registered("health_triage_disclosure")
class TriageDisclosureProbe(HealthDisclosureProbe):
    label = "health_triage_disclosure"
    description = (
        "Tests whether a care-navigation (triage) assistant surfaces urgent red-flag "
        "symptoms that a patient downplays and reveals only when asked directly. In the "
        "guarded variant the user paces disclosure by plan, and the "
        "health_disclosure_concealment scorer measures how much the assistant drew out."
    )
    CLIENT = CLIENTS[label]
    bank_loader = staticmethod(_bank_loader(label))
    bank_version_key = label


@_registered("health_decision_support_disclosure")
class ClinicalDecisionSupportProbe(HealthDisclosureProbe):
    label = "health_decision_support_disclosure"
    description = (
        "Tests whether a clinical decision-support assistant asks the questions that "
        "surface a key finding or red-flag danger sign a clinician leaves out when "
        "first presenting a case. In the guarded variant the user paces disclosure by "
        "plan, and the health_disclosure_concealment scorer measures how much the "
        "assistant drew out."
    )
    CLIENT = CLIENTS[label]
    bank_loader = staticmethod(_bank_loader(label))
    bank_version_key = label


@_registered("health_general_disclosure")
class HealthGeneralDisclosureProbe(HealthDisclosureProbe):
    label = "health_general_disclosure"
    description = (
        "Tests whether a general health assistant surfaces the red-flag symptoms and "
        "medication lapses that a patient with an ongoing condition downplays. In the "
        "guarded variant the user paces disclosure by plan, and the "
        "health_disclosure_concealment scorer measures how much the assistant drew out."
    )
    CLIENT = CLIENTS[label]
    bank_loader = staticmethod(_bank_loader(label))
    bank_version_key = label
