# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Unified ConversationSimulatorConfig for all probe types."""

from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig
from pydantic import Field

MODEL_USER = "user_model"
MODEL_ASSISTANT = "assistant_model"
MODEL_API_RESPONSE = "api_response_model"
MODEL_JUDGE = "judge_model"
MODEL_SUMMARY = "summary_model"

#: Every chat alias the simulator resolves from the model registry.
MODEL_ALIASES = [
    MODEL_USER,
    MODEL_ASSISTANT,
    MODEL_API_RESPONSE,
    MODEL_JUDGE,
    MODEL_SUMMARY,
]

#: The subset a run cannot start without. ``MODEL_SUMMARY`` is absent
#: because the engine falls back to ``MODEL_USER`` for summaries, and the
#: embedding alias because dense retrieval falls back to lexical.
REQUIRED_MODEL_ALIASES = [
    MODEL_USER,
    MODEL_ASSISTANT,
    MODEL_API_RESPONSE,
    MODEL_JUDGE,
]


class ConversationSimulatorConfig(SingleColumnConfig):
    """Configuration for the unified conversation simulator plugin.

    Dispatches to probe-specific generators based on the probe_type column.
    Tool-calling-specific fields are optional (None when not applicable).
    """

    column_type: Literal["conversation-simulator"] = "conversation-simulator"

    model_alias: str = "user_model"

    # Shared input columns
    persona_column: str = "persona"
    probe_type_column: str = "probe_type"
    theme_column: str = "theme"

    # Tool-calling-specific columns (optional, None for non-tool probes)
    tools_column: str | None = None
    toolset_name_column: str | None = None

    # Locale (set per pipeline run, not per row)
    locale: str = "en_US"
    assets_dir: str | None = None

    # Simulation parameters
    max_tools: int = 5
    max_turns: int = 5
    max_steps: int = 10
    max_query_attempts: int = 3

    # Assistant-turn resampling budget (judge-gated). ``1`` (default)
    # preserves the original behavior: one assistant attempt per turn,
    # with the per-turn quality judge acting as a flag only. Values > 1
    # discard a judge-rejected assistant candidate and regenerate it
    # within this budget; only the final candidate enters the
    # transcript. Intended for SDG-style runs where the assistant is a
    # data generator rather than a model under test — leave at 1 for
    # pure-capability evaluation. Probes whose ``after_assistant_turn``
    # has side effects (tool execution) opt out via
    # ``supports_assistant_resampling = False`` and are capped to one
    # attempt regardless of this setting. Gate semantics: ``warn``
    # counts as a pass (only ``failure`` triggers a resample), and a
    # judge reply with no parseable rating is re-asked once on the same
    # candidate instead of being charged against the budget. Failure
    # attribution is unchanged at any budget: only errors from the
    # assistant call itself are attributed to the assistant model.
    max_assistant_attempts: int = 1

    # User-model language enforcement (script-compliance gate).
    # The user-LLM is simulation infrastructure; when it emits a turn in
    # the wrong script (e.g. romanized/Latin Hindi in an hi_Deva_IN run)
    # the turn is regenerated within the existing ``max_query_attempts``
    # budget. The check is deterministic (Unicode script ranges) and a
    # no-op for Latin-script locales, so English / French / Portuguese
    # runs are unaffected.
    enforce_user_language: bool = True
    # Minimum fraction of letter codepoints that must fall in the
    # locale's expected script. 0.6 (not 1.0) tolerates Latin loanwords /
    # brand names / digits inside an otherwise-native turn while still
    # rejecting fully-transliterated output.
    user_language_min_script_compliance: float = 0.6
    # Skip the gate for very short turns (letter count below this), where
    # the script signal is too weak to act on (bare numbers, "OK", etc.).
    user_language_min_letters: int = 8

    # Sim2Real gap mitigations
    incremental_disclosure_ratio: float = 0.6
    # Default 1.0 = every trajectory's first user turn incorporates
    # persona-specific details. The whole point of this framework is
    # persona-driven simulation; running with generic queries
    # defeats it. Lower the ratio to introduce a generic-query
    # control arm (useful for ablation / matched-pair studies that
    # need a "what if the user query was persona-agnostic" baseline).
    persona_grounding_ratio: float = 1.0

    # Context compression: summarize older assistant responses to bound context growth
    context_compression: bool = True
    compression_window: int = Field(
        default=1,
        ge=1,
        description="Keep the last N summarized assistant responses verbatim.",
    )

    # Keep the assistant's thinking trace (``reasoning_content``) on the
    # assistant message when the model emits one. Default True: capturing
    # it is the point of running a reasoning model, and a trace that was
    # not stored cannot be recovered without re-simulating. Turn it off to
    # keep trajectories narrow: traces dominate the stored text rather than
    # merely adding to it, so an eval-only run that will never read them
    # pays a large size cost for nothing. Never affects what a model
    # receives: traces are stored, never replayed.
    store_reasoning: bool = True

    # Logging verbosity: 0=quiet, 1=normal (default), 2=detailed per-call timing
    verbosity: int = 1

    # financial_services probe knobs (ignored by other probes).
    # ``finance_tier_mix`` is the probability [0,1] a trajectory uses the DYNAMIC
    # tier (generated open-ended, judge-scored) vs the default VERIFIABLE tier;
    # ``finance_tier`` force-pins a tier ("verifiable"|"dynamic") when set.
    # ``finance_retrieval_mode`` selects the kb_search backend: "hybrid" (default,
    # the PRIMARY setting — normalised dense cosine blended with lexical overlap
    # over the institution's corpus; both tiers retrieve from the KB) | "dense"
    # (cosine ALONE — kept as an ablation, and measurably weak on a corpus this
    # homogeneous: 88.8% of its results were tool-reference docs against an 11.3%
    # corpus share) | "golden" (a reasoning-isolation ABLATION that injects the
    # verifiable task's gold docs so retrieval quality is removed as a variable;
    # the dynamic tier has no gold docs so it falls back to lexical). Every mode
    # degrades to lexical when no embedding endpoint is reachable. Retrieval
    # quality itself is measured separately by the financial_retrieval_recall
    # capability (document recall).
    # ``finance_embedding_model_alias`` is the query-embedding alias.
    finance_tier_mix: float = 0.0
    finance_tier: str | None = None
    finance_retrieval_mode: str = "hybrid"
    finance_embedding_model_alias: str = "embedding_model"

    # Reproducibility
    random_seed: int | None = None

    def get_model_aliases(self) -> list[str]:
        """The aliases a run cannot start without.

        A conversation needs all four: the simulated user, the assistant
        under test, tool-call responses, and the in-sim judge. Declaring
        them here means an unreachable endpoint or a typo surfaces in the
        startup health check rather than partway through a paid run.

        ``summary_model`` and ``finance_embedding_model_alias`` are left
        out on purpose. Both are optional at run time -- summaries fall
        back to the user model, dense retrieval falls back to lexical --
        and the health check treats every alias it is given as required.
        """
        return list(REQUIRED_MODEL_ALIASES)

    @property
    def required_columns(self) -> list[str]:
        return [
            self.persona_column,
            self.probe_type_column,
            self.theme_column,
        ]

    @property
    def side_effect_columns(self) -> list[str]:
        return [
            "conversation_messages",
            "user_query",
            "conversation_metadata",
            "conversation_status",
            "behavioral_profile",
            "locale",
            "conversation_language",
            "num_turns",
            "num_tool_calls",
            "tool_subset",
            "disclosure_style",
            "user_interaction_style",
            "persona_grounding",
            # Normalized protected fields rendered into the user-agent prompt.
            # The raw persona input is dropped, so this compact JSON audit view
            # preserves what influenced the simulated user.
            "persona_religion_language_context",
            # Structured per-trajectory outcome + per-turn telemetry sidecar.
            # See core/outcomes.py for the schema.
            "simulation_outcome",
            "simulation_traces",
            "probe_family",
            "probe_variant",
            # Stable identity for replay / matched-pair / canary /
            # idempotent re-runs. See core/identity.py.
            "persona_uuid",
            "trajectory_id",
            "usersim_provenance",
            # Probe-specific side channels. Each is written by one probe's
            # ``build_result_extras`` and read back by that probe's scorer,
            # which treats an absent value as "nothing to score" rather than
            # as an error. The engine writes the configured column plus the
            # names declared here and silently discards anything else, so an
            # undeclared column is computed and then dropped, and the probe
            # scores as unscoreable. They are declared for every run rather
            # than per probe because the declaration is read once, before
            # any row reveals which probe produced it; rows from other
            # probes simply leave them empty.
            "query_id",
            "sovereign_facts_probed",
            "probing_categories_explored",
            "probing_subtopic_hints_used",
            "action_request_id",
            "sub_protocol",
            "target_request_id",
            "strategy_id",
            "reframings_used",
            "capitulation_checks",
            "finance_task_id",
            "task_type",
            "task_tier",
            "task_contract_version",
            "taxonomy_version",
            "domain",
            "region",
            "institution_id",
            "institution_type",
            "dynamic_category_id",
            "dynamic_subtopic_hint",
            "expected_state_deltas",
            "gold_document_ids",
            "gold_tool_sequence",
            "retrieved_document_ids",
            "kb_search_queries",
            "num_kb_searches",
            "attempted_tool_names",
            "identity_tactic_id",
            "identity_tactic_kind",
            "identity_pair",
            "identity_competitor",
            "expected_identity",
            "identity_spec",
            # health_* guarded variant: the move/Guard ground truth written by
            # GuardedMoveMixin.build_result_extras and read back by
            # health_disclosure_concealment. The default variant writes none of
            # these, so rows from it leave them empty. A test checks that every
            # name in GUARDED_RESULT_COLUMNS (probes/health_disclosure/mixin.py)
            # is listed here.
            "moves_enabled",
            "moves_played",
            "moves_detail",
            "guard_veto_count",
            "disclosed_topics",
            "committed_disclosure_levels",
            "concealment_topics",
            "disclosure_coverage",
            "risk_present",
            "risk_noun",
            "risk_revealed",
            "risk_revealed_turn",
            "patient_archetype",
            "turn_budget",
            "risk_opportunity",
        ]
