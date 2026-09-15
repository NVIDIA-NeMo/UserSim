# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared base probe for the health_disclosure family.

``HealthDisclosureProbe`` holds ALL the behavior; each client is a thin
subclass that sets ``label`` + ``CLIENT`` (a config dict from ``clients.py``)
and carries its own ``@register_probe`` (so it gets its own ``PROBE_FAMILY`` →
its own judge axes). See therapy.py / triage.py / decision_support.py.

The auditable move + Guard layer is provided by the reusable
:class:`~usersim.engine.probes.health_disclosure.mixin.GuardedMoveMixin`
(a domain-agnostic capability, composed the same way the repo's other capability
mixins are). It is opt-in per trajectory via the ``guarded`` ``probe_variant``
(default OFF → a plain persona-grounded incremental-disclosure user, safe to
register and run by default). ``USERSIM_DISCLOSURE_MOVES`` remains as a
local-dev override only.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from usersim.engine.core.behavioral import (
    format_behavioral_profile_for_prompt,
    format_disclosure_instructions,
    format_interaction_style_instructions,
)
from usersim.engine.core.clinical_profile_bank import (
    DEFAULT_PROFILE_ID,
    ClinicalProfile,
)
from usersim.engine.core.outcomes import WarningKind
from usersim.engine.core.persona import format_persona_for_prompt
from usersim.engine.core.probes import BankBackedProbe
from usersim.engine.core.simulation import language_instruction
from usersim.engine.probes.health_disclosure.mixin import (
    GuardedMoveMixin,
    env_override,
)

# Local-dev override for the guarded move-space (primary switch is the ``guarded``
# probe_variant; see GuardedMoveMixin). Off/unset → driven by the variant.
_MOVES_ENV = "USERSIM_DISCLOSURE_MOVES"
# Env flag: inject a stand-in system-under-test prompt for the assistant (local
# runs only). Leave unset when the assistant is the client's real model.
_STANDIN_ENV = "USERSIM_DISCLOSURE_STANDIN"


class HealthDisclosureProbe(GuardedMoveMixin, BankBackedProbe):
    """A persona-grounded user who discloses gradually to an assistant/SUT.

    Client-shaped via the ``CLIENT`` class attribute (set by subclasses). The
    guarded move-space capability comes from ``GuardedMoveMixin``; the hidden
    clinical profile (gated topics + concealment/risk ground truth) comes from a
    versioned clinical-profile bank via ``BankBackedProbe``. Each client subclass
    sets ``label`` + ``CLIENT`` and its own ``bank_loader`` + ``bank_version_key``
    (one bank per client; see therapy.py / triage.py / decision_support.py).
    """

    #: Set by each client subclass to a config dict from ``clients.py``.
    CLIENT: dict = {}
    #: Local-dev override honoured by ``GuardedMoveMixin.guarded_moves_enabled``.
    GUARDED_MOVE_ENV = _MOVES_ENV
    #: Emitted (by BankBackedProbe) whenever the derived profile is a placeholder.
    placeholder_warning_kind = WarningKind.USED_PLACEHOLDER_CLINICAL_PROFILE

    # ── BankBackedProbe hook ────────────────────────────────────────
    def derive_task(
        self, persona: dict, bank: Any, *, cfg: Any,
    ) -> Optional[ClinicalProfile]:
        """Select this persona's hidden clinical profile from the bank.

        Only the guarded variant consumes the profile (the default variant is a
        plain incremental-disclosure user with no concealment ground truth), so
        the default variant derives no task — no bank_version pin churn, no
        placeholder warning for trajectories that never used the profile.
        Falls back to the bank's ``__default__`` entry when the persona uuid is
        not present (the usual case for a bundled synthetic bank).
        """
        if not self.guarded_moves_enabled():
            return None
        uuid = str(persona.get("uuid") or persona.get("persona_uuid") or "").strip()
        entry = bank.by_id(uuid) if uuid else None
        return entry or bank.by_id(DEFAULT_PROFILE_ID)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        client = self.CLIENT
        # The default (move-space OFF) variant has no Guard to pace the user, so an
        # "upfront" disclosure_style (set globally by the dispatcher) would let the
        # persona dump everything on turn 1 — degenerate for a disclosure probe.
        # Bias the baseline user toward incremental pacing. The guarded variant
        # keeps the dispatcher's style: it drives the patient archetype and the
        # Guard enforces pacing (see GuardedMoveMixin).
        disclosure_style = self._data.get("disclosure_style", "incremental")
        if not self.guarded_moves_enabled():
            disclosure_style = "incremental"
        # Scaffolds resolve against the ASSET locale (en_IN for an India
        # language-variant reusing the en_IN prompt pack), not the conversation
        # locale — en_IN is a shipped pack key so a variant never aborts. The
        # conversation OUTPUT language still comes from the interpolated
        # ``language_instruction`` below, which keys off ``self._locale``.
        self._user_system_prompt = client["user_prompt"].get(
            self._asset_locale,
            persona=format_persona_for_prompt(self._persona),
            language_instruction=language_instruction(self._language, self._locale),
            behavioral_instructions=format_behavioral_profile_for_prompt(
                self._profile, language=self._language,
            ),
            disclosure_instructions=format_disclosure_instructions(disclosure_style),
            interaction_style_instructions=format_interaction_style_instructions(
                self._interaction_style,
            ),
        )
        # Wire up the (opt-in) guarded move-space capability.
        max_turns = int(getattr(self._cfg, "max_turns", 6) or 6)
        self.init_guarded_moves(disclosure_style=disclosure_style, max_turns=max_turns)

    # ── GuardedMoveMixin config hook ────────────────────────────────
    def guarded_move_config(self) -> Mapping[str, Any]:
        return self.CLIENT

    # ── prompts ─────────────────────────────────────────────────────
    def get_user_system_prompt(self) -> str:
        return self._user_system_prompt

    def get_assistant_system_prompt(self) -> str:
        # Default: empty (the client's real model carries its own behavior).
        # Stand-in mode (local dev): the client's stand-in prompt behind an env flag.
        if env_override(_STANDIN_ENV):
            return self.CLIENT.get("standin_prompt", "")
        return ""

    def format_gate_prompt(self, user_query: str, conversation_history: str) -> str:
        return self.CLIENT["gate_prompt"].get(self._asset_locale).format(
            conversation_history=conversation_history,
            user_turn_to_evaluate=user_query,
        )
