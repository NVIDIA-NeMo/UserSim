# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persona → probing-taxonomy task derivation.

Population probing differs structurally from sovereign-knowledge fact
selection in two ways:

- **Categories don't carry persona-tag filters.** Geography, history,
  culture, etc. are universal topic neighborhoods; every persona is in
  scope for every category. The point is *breadth* coverage. Persona
  attributes shape the user-agent's *voice* (vocabulary, register,
  pushback style) at prompt-render time, not category eligibility at
  derivation time.
- **There is no ground truth to defend with verbatim injection.** The
  user-agent generates the opening turn from the category invitation +
  a rotated sub-topic hint, in the persona's voice. So this module's
  job is just to pick a category and a hint deterministically; the
  probe then hands those to the user-agent prompt.

Determinism contract (mirrors :mod:`sov_ai_facts.task_derivation`):

- Picks are stable for ``(persona content, taxonomy_id + taxonomy_version,
  optional seed)`` tuples.
- ``taxonomy_version`` is folded into the salt so a taxonomy bump
  produces a different pick for the same persona — surfaces that the
  test set actually changed.
- Hint rotation: given the picked category, hints rotate per persona +
  ``hint_index`` so successive probes against the same category do not
  collapse onto the same sub-topic.
"""

from __future__ import annotations

from typing import Any, Iterable

from usersim.engine.core.probing_taxonomy import (
    Category,
    ProbingTaxonomy,
    select_category,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def persona_to_interest_tags(
    persona: dict[str, Any],
    locale: str,
) -> list[str]:
    """Return the structured interest-tag set this persona projects to.

    Re-uses :func:`sov_ai_facts.task_derivation.persona_to_tags`
    so both probes speak the same persona-tag dialect. The returned
    set is **not** used to filter categories (categories are universal in
    population probing) — it is exposed so the user-agent prompt can
    reflect what kind of person is asking, and so downstream stratification
    can join probing trajectories with sovereign-knowledge ones on the
    same persona-feature axis.
    """
    # Lazy import keeps a circular import from accidentally appearing if
    # the sovereign-knowledge package ever pulls anything from here.
    from usersim.engine.probes.sov_ai_facts.task_derivation import (
        persona_to_tags,
    )

    return persona_to_tags(persona, locale)


def derive_probe(
    persona: dict[str, Any],
    taxonomy: ProbingTaxonomy,
    *,
    seed: int | None = None,
    excluded_category_ids: Iterable[str] = (),
    hint_rotation_index: int = 0,
) -> tuple[Category, str] | None:
    """Pick a ``(category, subtopic_hint)`` pair for this persona.

    Returns ``None`` when every category in the taxonomy has been
    excluded (the caller should log and emit a structured failure —
    this should be impossible with the shipped 5-category taxonomies
    unless the caller is doing something exotic with exclusions).

    Determinism:
      - The category pick is a deterministic random choice over the
        non-excluded category list, seeded by
        ``mix(persona_content_hash, taxonomy_salt, seed)``.
      - The sub-topic hint is picked by indexing into the category's
        ``subtopic_hints`` tuple at
        ``(persona_hint_index + hint_rotation_index) % len(hints)``.
        ``hint_rotation_index`` lets the caller probe the same category
        multiple times for the same persona without collapse on the
        same sub-topic (e.g., a multi-turn rerun batch).
    """
    # Delegates to the shared core derivation so the hashing/rotation contract
    # is identical across every dynamic-tier probe (sov_ai_dynamic +
    # financial_services + future domains).
    return select_category(
        persona,
        taxonomy,
        seed=seed,
        excluded_category_ids=excluded_category_ids,
        hint_rotation_index=hint_rotation_index,
    )
