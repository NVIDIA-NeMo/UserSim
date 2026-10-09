# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical construction of one fully resolved simulator episode input."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from usersim.engine.config import MODEL_ASSISTANT, MODEL_USER
from usersim.engine.core.behavioral import (
    compute_behavioral_profile,
    compute_disclosure_style,
    compute_user_interaction_style,
    get_conversation_language,
)
from usersim.engine.core.identity import persona_uuid, resolve_model_name, trajectory_id
from usersim.engine.core.locale import persona_dataset_locale
from usersim.engine.core.missing import none_for_missing
from usersim.engine.core.outcomes import OutcomeBuilder, Provenance
from usersim.engine.core.persona import religion_language_context
from usersim.engine.core.probes import BankLoadError, resolve_probe
from usersim.engine.core.provenance import get_code_sha, get_nemotron_personas_version

if TYPE_CHECKING:
    from usersim.engine.config import ConversationSimulatorConfig

#: A seed column that carries a stored episode row whole, as JSON. A seed table
#: types each of its columns, so an ``int`` column with a ``float`` in another
#: row would reach the probe as ``float``; inside one JSON cell every value
#: keeps the type it was stored with.
EPISODE_INPUT_COLUMN = "usersim_episode_input"


def unpack_stored_episode(data: dict[str, Any]) -> None:
    """Replace ``data``'s columns with the stored episode it carries, if any.

    The carrier column is removed, and the stored values overwrite the seed
    table's copies of the same columns.
    """
    cell = data.pop(EPISODE_INPUT_COLUMN, None)
    if cell is None:
        return
    stored = json.loads(cell) if isinstance(cell, str) else cell
    if not isinstance(stored, Mapping):
        raise ValueError(f"{EPISODE_INPUT_COLUMN} must hold a JSON object, got {type(stored).__name__}")
    data.update(stored)


@dataclass(frozen=True)
class EpisodePreamble:
    """Typed, model-call-free inputs shared by native and hosted execution."""

    probe_type: str
    probe_family: str
    probe_variant: str
    persona: dict[str, Any]
    persona_uuid: str
    locale: str
    language: str
    behavioral_profile: dict[str, Any]
    disclosure_style: str
    user_interaction_style: str
    persona_grounding: bool
    trajectory_id: str
    provenance: Provenance
    data: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        """Return the fully resolved JSON-safe simulator input row."""
        return deepcopy(self.data)


@dataclass(frozen=True)
class ConstructedEpisode:
    """A resolved preamble and its already-initialized native probe."""

    preamble: EpisodePreamble
    probe: Any
    outcome: OutcomeBuilder
    #: The resolved row as set-up left it, before the probe was constructed. The
    #: probe shares ``preamble.data``, so this is the baseline for what the probe
    #: itself adds or changes, in its constructor or its run. ``None`` when not
    #: recorded; the generator then compares with the row after construction.
    set_up_row: dict[str, Any] | None = None


class EpisodeConstructionError(ValueError):
    """Probe-construction failure carrying its already-resolved preamble.

    ``set_up_row`` is the row as set-up left it, before the failed constructor
    ran, like ``ConstructedEpisode.set_up_row``: the constructor may have written
    to ``preamble.data`` before it failed.
    """

    def __init__(
        self,
        preamble: EpisodePreamble,
        cause: Exception,
        *,
        set_up_row: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(str(cause))
        self.preamble = preamble
        self.cause = cause
        self.set_up_row = set_up_row


def construct_episode_preamble(
    data: Mapping[str, Any],
    *,
    config: ConversationSimulatorConfig,
    models: Mapping[str, Any],
) -> EpisodePreamble:
    """Resolve all per-episode inputs without invoking a model.

    This is the sole constructor for persona-derived settings, probe metadata,
    provenance, and trajectory identity. Both execution paths consume its
    returned ``data`` mapping unchanged. Every empty cell in that mapping and
    in the persona is None, however pandas held it, so persona hashes and
    falsy fallbacks do not depend on the pandas version.
    """
    resolved = none_for_missing(deepcopy(dict(data)))
    persona_column = config.persona_column
    probe_type_column = config.probe_type_column
    raw_persona = resolved[persona_column]
    persona = none_for_missing(deepcopy(raw_persona if isinstance(raw_persona, dict) else json.loads(raw_persona)))
    locale = config.locale
    raw_profile = resolved.get("behavioral_profile")
    if isinstance(raw_profile, str):
        raw_profile = json.loads(raw_profile)
    profile = (
        deepcopy(dict(raw_profile))
        if isinstance(raw_profile, Mapping)
        else compute_behavioral_profile(persona, locale=persona_dataset_locale(locale))
    )
    persona_json = json.dumps(persona, sort_keys=True, default=str)
    persona_seed = int(hashlib.sha256(persona_json.encode()).hexdigest(), 16) & 0xFFFFFFFF
    disclosure_style = str(
        resolved.get("disclosure_style")
        or compute_disclosure_style(
            persona_seed=persona_seed,
            incremental_ratio=config.incremental_disclosure_ratio,
        )
    )
    interaction_style = str(resolved.get("user_interaction_style") or compute_user_interaction_style(profile))
    grounding = (
        bool(resolved["persona_grounding"])
        if "persona_grounding" in resolved
        else random.Random(persona_seed + 1).random() < config.persona_grounding_ratio
    )
    probe_type = str(resolved[probe_type_column])
    probe_cls = resolve_probe(probe_type)
    probe_module = sys.modules.get(probe_cls.__module__)
    variants = getattr(probe_module, "PROBE_VARIANTS", None)
    explicit_variant = resolved.get("probe_variant")
    probe_variant = (
        explicit_variant
        if isinstance(explicit_variant, str) and explicit_variant
        else str(variants[0])
        if variants and isinstance(variants, (list, tuple))
        else "default"
    )
    probe_family = str(getattr(probe_module, "PROBE_FAMILY", probe_type))
    prompt_version = str(getattr(probe_module, "PROMPT_VERSION", "v1.0"))
    raw_provenance = resolved.get("usersim_provenance")
    if isinstance(raw_provenance, str):
        raw_provenance = json.loads(raw_provenance)
    provenance = (
        Provenance(
            nemotron_personas_version=raw_provenance.get("nemotron_personas_version"),
            scenario_prompt_version=raw_provenance.get("scenario_prompt_version"),
            code_sha=raw_provenance.get("code_sha"),
            bank_version=dict(raw_provenance.get("bank_version") or {}),
        )
        if isinstance(raw_provenance, Mapping)
        else Provenance(
            nemotron_personas_version=get_nemotron_personas_version(),
            scenario_prompt_version=prompt_version,
            code_sha=get_code_sha(),
            bank_version={},
        )
    )
    persona_id = persona_uuid(persona)
    explicit_trajectory_id = resolved.get("trajectory_id")
    episode_id = (
        explicit_trajectory_id
        if isinstance(explicit_trajectory_id, str) and explicit_trajectory_id
        else trajectory_id(
            persona_uuid=persona_id,
            probe_family=probe_family,
            probe_variant=probe_variant,
            scenario_seed=config.random_seed,
            user_model=resolve_model_name(models.get(MODEL_USER), MODEL_USER),
            assistant_model=resolve_model_name(models.get(MODEL_ASSISTANT), MODEL_ASSISTANT),
            prompt_version=prompt_version,
            extra_keys=[("locale", locale)],
        )
    )
    language = get_conversation_language(locale)
    resolved.update(
        {
            persona_column: persona,
            "persona_religion_language_context": json.dumps(
                religion_language_context(persona),
                ensure_ascii=False,
            ),
            "persona_uuid": persona_id,
            "behavioral_profile": json.dumps(profile, ensure_ascii=False),
            "disclosure_style": disclosure_style,
            "user_interaction_style": interaction_style,
            "persona_grounding": grounding,
            "locale": locale,
            "conversation_language": language,
            "probe_family": probe_family,
            "probe_variant": probe_variant,
            "trajectory_id": episode_id,
            "usersim_provenance": json.dumps(provenance.to_dict(), ensure_ascii=False),
        }
    )
    return EpisodePreamble(
        probe_type=probe_type,
        probe_family=probe_family,
        probe_variant=probe_variant,
        persona=persona,
        persona_uuid=persona_id,
        locale=locale,
        language=language,
        behavioral_profile=profile,
        disclosure_style=disclosure_style,
        user_interaction_style=interaction_style,
        persona_grounding=grounding,
        trajectory_id=episode_id,
        provenance=provenance,
        data=resolved,
    )


def construct_probe_episode(
    data: Mapping[str, Any],
    *,
    config: ConversationSimulatorConfig,
    models: Mapping[str, Any],
    provenance: Provenance | None = None,
) -> ConstructedEpisode:
    """Construct the canonical preamble and native probe exactly once."""
    preamble = construct_episode_preamble(data, config=config, models=models)
    resolved_provenance = provenance or preamble.provenance
    outcome = OutcomeBuilder(provenance=resolved_provenance)
    probe_cls = resolve_probe(preamble.probe_type)
    set_up_row = dict(preamble.data)
    try:
        probe = probe_cls(
            persona=preamble.persona,
            locale=preamble.locale,
            language=preamble.language,
            models=dict(models),
            cfg=config,
            provenance=resolved_provenance,
            profile=preamble.behavioral_profile,
            data=preamble.data,
            outcome_builder=outcome,
        )
    except (KeyError, ValueError, json.JSONDecodeError, BankLoadError) as error:
        raise EpisodeConstructionError(preamble, error, set_up_row=set_up_row) from error
    return ConstructedEpisode(preamble=preamble, probe=probe, outcome=outcome, set_up_row=set_up_row)
