# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public API for extending NeMo User Sim.

A probe, a scorer, a capability, an asset domain or an execution backend
imports what it needs from here, and its tests import from ``usersim.testing``.
``docs/extension_api.md`` lists every name with what it is for. Other
``usersim`` modules are internal, and code moves between them; the names here
are imported from this module wherever they are defined, and can change in any
release as well.

Each name is imported from the module that defines it the first time it is
used, so ``from usersim.extension import AssetDomain`` loads the asset-domain
registry and not the conversation loop. ``usersim --help`` imports every
installed asset domain and backend to list them, and stays fast.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from usersim.asset_gen.registry import AssetDomain
    from usersim.cli._models import require_aliases, to_data_designer_kwargs, to_model_configs
    from usersim.engine.config import MODEL_ASSISTANT, MODEL_JUDGE, MODEL_USER, ConversationSimulatorConfig
    from usersim.engine.core._assets import asset_roots_in_use, probe_asset_roots, probe_assets_dir
    from usersim.engine.core.backends import ExecutionBackend, JobHandle, JobSpec, JobStatus
    from usersim.engine.core.behavioral import (
        ChatRealism,
        education_ordinal,
        format_behavioral_profile_for_prompt,
        format_disclosure_instructions,
        format_interaction_style_instructions,
        get_conversation_language,
    )
    from usersim.engine.core.identity import persona_uuid, resolve_model_name
    from usersim.engine.core.llm import acall_llm, outcome_builder_scope
    from usersim.engine.core.locale import (
        SCRIPT_BUCKETS,
        SHIPPED_LOCALES,
        asset_locale,
        expected_language_display,
        expected_script_ranges,
        india_variant,
        persona_dataset_locale,
    )
    from usersim.engine.core.localized import LocalizedText, LocalizedTextError
    from usersim.engine.core.messages import format_conversation_history_for_prompt, format_tools_for_prompt
    from usersim.engine.core.missing import is_missing, none_for_missing
    from usersim.engine.core.outcomes import (
        FailureAttribution,
        FailureClass,
        OutcomeBuilder,
        OutcomeStatus,
        Provenance,
        SimulationTrace,
        TraceKind,
        WarningKind,
    )
    from usersim.engine.core.persona import format_persona_for_prompt, persona_full_name
    from usersim.engine.core.persona_tags import persona_to_tags
    from usersim.engine.core.probes import (
        AgenticMixin,
        BankBackedProbe,
        BankLoadError,
        BankReframingMixin,
        BankVerbatimMixin,
        BaseProbe,
        CustomTurn1InstructionMixin,
        LocalePromptPack,
        PreviewField,
        ProbeRegistrationError,
        ToolCallingMixin,
        ToolExecutionMixin,
        assistant_message,
        dispatch_episode,
        register_probe,
    )
    from usersim.engine.core.probing_taxonomy import (
        Category,
        ProbingTaxonomy,
        ProbingTaxonomyError,
        load_probing_taxonomy,
        load_probing_taxonomy_for,
        reset_probing_taxonomy_cache,
        select_category,
        taxonomy_path_for,
    )
    from usersim.engine.core.prompt_loader import get_prompt, load_prompt_file, render_prompt
    from usersim.engine.core.row_metadata import recorded_value
    from usersim.engine.core.seeds import load_seeds
    from usersim.engine.core.simulation import (
        RESERVED_METADATA_KEYS,
        ConversationState,
        finalize_completed,
        language_instruction,
        make_failed,
        make_result,
    )
    from usersim.engine.core.tool_calls import parse_tool_call, recover_tool_call
    from usersim.engine.core.translation import translate_search_query
    from usersim.engine.core.user_turn_policy import UserTurnPolicy
    from usersim.engine.evaluator.scorers import ScorerFn, ScorerRegistrationError, register_scorer
    from usersim.taxonomy.capabilities import ALL_PROBES, CapabilityDefinition, EvidenceSource

__all__ = [
    # Probes
    "AgenticMixin",
    "BankBackedProbe",
    "BankLoadError",
    "BankReframingMixin",
    "BankVerbatimMixin",
    "BaseProbe",
    "CustomTurn1InstructionMixin",
    "LocalePromptPack",
    "PreviewField",
    "ProbeRegistrationError",
    "ToolCallingMixin",
    "ToolExecutionMixin",
    "UserTurnPolicy",
    "register_probe",
    # Conversations and outcomes
    "ConversationState",
    "RESERVED_METADATA_KEYS",
    "assistant_message",
    "dispatch_episode",
    "FailureAttribution",
    "FailureClass",
    "OutcomeBuilder",
    "OutcomeStatus",
    "Provenance",
    "SimulationTrace",
    "TraceKind",
    "WarningKind",
    "finalize_completed",
    "make_failed",
    "make_result",
    "outcome_builder_scope",
    # Models and settings
    "MODEL_ASSISTANT",
    "MODEL_JUDGE",
    "MODEL_USER",
    "ConversationSimulatorConfig",
    "acall_llm",
    "resolve_model_name",
    # Prompts
    "ChatRealism",
    "format_behavioral_profile_for_prompt",
    "format_conversation_history_for_prompt",
    "format_disclosure_instructions",
    "format_interaction_style_instructions",
    "format_persona_for_prompt",
    "format_tools_for_prompt",
    "get_prompt",
    "load_prompt_file",
    "persona_full_name",
    "persona_uuid",
    "render_prompt",
    # Banks and assets
    "LocalizedText",
    "LocalizedTextError",
    "load_seeds",
    "asset_roots_in_use",
    "education_ordinal",
    "persona_to_tags",
    "probe_asset_roots",
    "probe_assets_dir",
    # Probing taxonomies
    "Category",
    "ProbingTaxonomy",
    "ProbingTaxonomyError",
    "load_probing_taxonomy",
    "load_probing_taxonomy_for",
    "reset_probing_taxonomy_cache",
    "select_category",
    "taxonomy_path_for",
    # Locales and languages
    "SCRIPT_BUCKETS",
    "SHIPPED_LOCALES",
    "asset_locale",
    "expected_language_display",
    "expected_script_ranges",
    "get_conversation_language",
    "india_variant",
    "language_instruction",
    "persona_dataset_locale",
    "translate_search_query",
    # Tool calls
    "parse_tool_call",
    "recover_tool_call",
    # Scorers
    "ScorerFn",
    "ScorerRegistrationError",
    "is_missing",
    "none_for_missing",
    "recorded_value",
    "register_scorer",
    # Capabilities, asset domains and backends
    "ALL_PROBES",
    "AssetDomain",
    "CapabilityDefinition",
    "EvidenceSource",
    "ExecutionBackend",
    "JobHandle",
    "JobSpec",
    "JobStatus",
    "require_aliases",
    "to_data_designer_kwargs",
    "to_model_configs",
]

#: The module each public name is defined in, by module.
_EXPORTS: dict[str, tuple[str, ...]] = {
    "usersim.asset_gen.registry": ("AssetDomain",),
    "usersim.cli._models": ("require_aliases", "to_data_designer_kwargs", "to_model_configs"),
    "usersim.engine.config": ("MODEL_ASSISTANT", "MODEL_JUDGE", "MODEL_USER", "ConversationSimulatorConfig"),
    "usersim.engine.core._assets": ("asset_roots_in_use", "probe_asset_roots", "probe_assets_dir"),
    "usersim.engine.core.backends": ("ExecutionBackend", "JobHandle", "JobSpec", "JobStatus"),
    "usersim.engine.core.behavioral": (
        "ChatRealism",
        "education_ordinal",
        "format_behavioral_profile_for_prompt",
        "format_disclosure_instructions",
        "format_interaction_style_instructions",
        "get_conversation_language",
    ),
    "usersim.engine.core.identity": ("persona_uuid", "resolve_model_name"),
    "usersim.engine.core.llm": ("acall_llm", "outcome_builder_scope"),
    "usersim.engine.core.locale": (
        "SCRIPT_BUCKETS",
        "SHIPPED_LOCALES",
        "asset_locale",
        "expected_language_display",
        "expected_script_ranges",
        "india_variant",
        "persona_dataset_locale",
    ),
    "usersim.engine.core.localized": ("LocalizedText", "LocalizedTextError"),
    "usersim.engine.core.messages": ("format_conversation_history_for_prompt", "format_tools_for_prompt"),
    "usersim.engine.core.missing": ("is_missing", "none_for_missing"),
    "usersim.engine.core.outcomes": (
        "FailureAttribution",
        "FailureClass",
        "OutcomeBuilder",
        "OutcomeStatus",
        "Provenance",
        "SimulationTrace",
        "TraceKind",
        "WarningKind",
    ),
    "usersim.engine.core.persona": ("format_persona_for_prompt", "persona_full_name"),
    "usersim.engine.core.persona_tags": ("persona_to_tags",),
    "usersim.engine.core.probes": (
        "AgenticMixin",
        "BankBackedProbe",
        "BankLoadError",
        "BankReframingMixin",
        "BankVerbatimMixin",
        "BaseProbe",
        "CustomTurn1InstructionMixin",
        "LocalePromptPack",
        "PreviewField",
        "ProbeRegistrationError",
        "ToolCallingMixin",
        "ToolExecutionMixin",
        "assistant_message",
        "dispatch_episode",
        "register_probe",
    ),
    "usersim.engine.core.probing_taxonomy": (
        "Category",
        "ProbingTaxonomy",
        "ProbingTaxonomyError",
        "load_probing_taxonomy",
        "load_probing_taxonomy_for",
        "reset_probing_taxonomy_cache",
        "select_category",
        "taxonomy_path_for",
    ),
    "usersim.engine.core.prompt_loader": ("get_prompt", "load_prompt_file", "render_prompt"),
    "usersim.engine.core.row_metadata": ("recorded_value",),
    "usersim.engine.core.seeds": ("load_seeds",),
    "usersim.engine.core.simulation": (
        "RESERVED_METADATA_KEYS",
        "ConversationState",
        "finalize_completed",
        "language_instruction",
        "make_failed",
        "make_result",
    ),
    "usersim.engine.core.tool_calls": ("parse_tool_call", "recover_tool_call"),
    "usersim.engine.core.translation": ("translate_search_query",),
    "usersim.engine.core.user_turn_policy": ("UserTurnPolicy",),
    "usersim.engine.evaluator.scorers": ("ScorerFn", "ScorerRegistrationError", "register_scorer"),
    "usersim.taxonomy.capabilities": ("ALL_PROBES", "CapabilityDefinition", "EvidenceSource"),
}

_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}


def __getattr__(name: str) -> Any:
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
