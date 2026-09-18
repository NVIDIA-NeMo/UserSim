# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output schema projection for curated datasets.

By default the curated parquet carries every trajectory column except the
``run`` partition level and the persona fields copied verbatim from the source
dataset. That is still a lot of columns — most callers exporting a
training/eval dataset want a compact projection. A *schema* is either a named
preset or an explicit column list; logical group tokens (``conversation``,
``persona``, ``persona_verbatim``) expand to the underlying columns, and any
requested column that is absent from the frame is silently skipped (so
``num_turns`` "if available" just works).

Two sets of columns are opt-in rather than default: the persona fields copied
from the source dataset, so publishing a curated dataset does not reproduce
rows nobody asked to redistribute, and the protected attributes (religion,
linguistic background). Ask for either with ``--schema full``, the
``persona_verbatim`` / ``persona_protected`` group tokens, or by column name.

The partition columns ``locale`` / ``probe_family`` are always retained
(the partitioned writer needs them); ``run`` is included only when the schema
asks for it (it is otherwise encoded in the output path).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

# Logical group tokens that expand to several real columns. Missing columns
# are skipped at resolve time, so these can list optional columns freely.
LOGICAL_GROUPS: dict[str, tuple[str, ...]] = {
    "conversation": (
        "conversation_messages",
        "conversation_metadata",
        "conversation_status",
    ),
    # Fields UserSim computes about the simulated user: the bucketed age and
    # the behavioural axes derived from the OCEAN traits. These describe how
    # the user behaved, which is what a training or eval set is usually for.
    "persona": (
        "persona_age_bin",
        "behavioral_profile",
        "disclosure_style",
        "user_interaction_style",
        "persona_grounding",
        "conversation_language",
    ),
    # Fields copied from the persona dataset row, including ``persona``, which
    # is the whole sampled person. Excluded from the default export: a curated
    # dataset should carry the attributes a run turned on, not a reproduction
    # of the source rows. Ask for them by name or with this token.
    "persona_verbatim": (
        "persona",
        "persona_name",
        "persona_age",
        "persona_sex",
        "persona_education_level",
        "persona_occupation",
        "persona_location",
        "persona_country",
    ),
    # Protected attributes: religion, religious background, and spoken and
    # linguistic background. Also excluded by default, and separate from
    # ``persona_verbatim`` so that asking for one does not silently include
    # the other. Review before sharing anything exported with this token.
    "persona_protected": ("persona_religion_language_context",),
    # financial_services gold-metadata side channels the probe emits on every
    # trajectory (verifiable + dynamic). First-class here so curated
    # training/eval exports carry them for the goal-(b) flywheel: the verifiable
    # tier -> verifier-labeled tool trajectories (gold_tool_sequence +
    # expected_state_deltas), the dynamic tier -> grounded SFT / preference
    # pairs (dynamic.* judge). All optional -> skipped for non-finance rows.
    # SINGLE SOURCE OF TRUTH: financial_services.generator.FINANCE_TRAJECTORY_COLUMNS
    # (kept decoupled here to avoid a plugin import in generic selection code; a
    # schema-parity test in tests/test_selection.py binds the two).
    "finance": (
        "finance_task_id",
        "task_tier",
        "task_type",
        "task_contract_version",
        "institution_id",
        "institution_type",
        "domain",
        "dynamic_category_id",
        "dynamic_subtopic_hint",
        "taxonomy_version",
        "gold_document_ids",
        "gold_tool_sequence",
        "expected_state_deltas",
        "retrieved_document_ids",
        "attempted_tool_names",
        "num_tool_calls",
        "kb_search_queries",
        "num_kb_searches",
    ),
}

# Columns the default export leaves out. They still reach ``--schema full``,
# an explicit column list, or their group token, so nothing is unreachable:
# the point is that exporting them is a choice someone made.
OPT_IN_ONLY: frozenset[str] = frozenset(LOGICAL_GROUPS["persona_verbatim"] + LOGICAL_GROUPS["persona_protected"])

# Named presets. ``None`` is the sentinel for "all columns" (full).
NAMED_SCHEMAS: dict[str, Optional[tuple[str, ...]]] = {
    "full": None,
    # The default. Every column except the persona fields copied verbatim from
    # the source dataset. Handled as an exclusion rather than a token list so
    # it keeps picking up new columns the way ``full`` does.
    "standard": None,
    "compact": (
        "run",
        "locale",
        "probe_family",
        "conversation",
        "trajectory_id",
        "persona_uuid",
        "probe_type",
        "probe_variant",
        "theme",
        "persona",
        "finance",
        "tools",
        "num_tools",
        "num_turns",
    ),
}

# Partition columns the writer must keep regardless of the requested schema.
_PARTITION_REQUIRED: tuple[str, ...] = ("locale", "probe_family")

# Schema = a preset name, a comma-separated string, or an explicit column list.
Schema = Union[str, Sequence[str], None]


def available_schemas() -> tuple[str, ...]:
    return tuple(NAMED_SCHEMAS)


def _expand(tokens: Iterable[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        out.extend(LOGICAL_GROUPS.get(token, (token,)))
    return out


def resolve_columns(schema: Schema, available: Iterable[str]) -> list[str]:
    """Resolve a schema spec to a concrete, ordered, deduped column list.

    - ``None`` / ``"standard"`` → every available column except ``run`` (the
      partition-path level, not payload) and the verbatim persona fields.
    - ``"full"`` → every available column except ``run``.
    - a preset name → its token list, expanded.
    - a comma-separated string or a sequence → those tokens, expanded.

    Group tokens expand; unknown/absent columns are dropped; ``locale`` and
    ``probe_family`` are force-appended if present so partitioned writes work.
    """
    available = list(available)
    available_set = set(available)

    def _everything(*, drop_opt_in: bool) -> list[str]:
        return [c for c in available if c != "run" and not (drop_opt_in and c in OPT_IN_ONLY)]

    if schema is None or schema == "standard":
        return _everything(drop_opt_in=True)
    if schema == "full":
        return _everything(drop_opt_in=False)

    if isinstance(schema, str):
        if schema in NAMED_SCHEMAS:
            spec = NAMED_SCHEMAS[schema]
            if spec is None:  # "full" / "standard", handled above
                return _everything(drop_opt_in=schema != "full")
            tokens: list[str] = list(spec)
        else:
            tokens = [t.strip() for t in schema.split(",") if t.strip()]
    else:
        tokens = list(schema)

    resolved: list[str] = []
    for col in _expand(tokens):
        if col in available_set and col not in resolved:
            resolved.append(col)

    for col in _PARTITION_REQUIRED:
        if col in available_set and col not in resolved:
            resolved.append(col)
    return resolved


def project(df, schema: Schema):
    """Return ``df`` projected to the columns resolved from ``schema``."""
    return df[resolve_columns(schema, df.columns)]
