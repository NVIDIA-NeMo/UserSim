# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Persona -> tag mapping + template selection for financial_services.

Matching is deliberately PERMISSIVE: banking is near-universal, so we match on
region / age / occupation-family and never hard-filter on ``interest:finance``
(that would starve the sampler). Templates tagged ``*:any`` match everyone; the
resolver falls back to all templates when nothing matches.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from usersim.engine.core.finance_bank import FinanceBank
from usersim.engine.core.finance_tasks import (
    FinanceInstance,
    _instance_seed,
    build_dynamic_instance,
    build_instance,
    select_institution,
    select_institution_and_template,
)
from usersim.engine.core.probing_taxonomy import select_category


def _age_bin(age: Any) -> str | None:
    try:
        a = int(age)
    except (TypeError, ValueError):
        return None
    if a < 25:
        return "18-24"
    if a < 35:
        return "25-34"
    if a < 45:
        return "35-44"
    if a < 55:
        return "45-54"
    if a < 65:
        return "55-64"
    return "65+"


def _slug(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower().replace(" ", "_")


_FINANCE_OCCUPATION_KEYWORDS = (
    "financ",
    "account",
    "cpa",
    "auditor",
    "analyst",
    "econom",
    "actuar",
    "invest",
    "banker",
    "bank ",
    "trader",
    "trading",
    "portfolio",
    "tax ",
    "wealth",
    "treasur",
    "underwrit",
    "broker",
)


def derive_financial_literacy(persona: dict[str, Any], locale: str = "en_US") -> str:
    """Derive a coarse ``low``/``medium``/``high`` financial-literacy level.

    A DERIVED persona axis (not a raw Nemotron-Personas field): education
    ordinal (reusing the per-locale map in ``core.behavioral``) plus a boost
    for finance/analytical occupations. Feeds the ``financial-literacy:`` tag
    (template matching) and the user-simulator prompt (how the persona asks:
    basics vs. digging into specifics) — the population lever that makes the
    dynamic-tier scorecard ecologically weighted rather than uniform.
    """
    from usersim.engine.core.behavioral import _education_ordinal

    score = _education_ordinal(str(persona.get("education_level", "")), locale)
    occ = str(persona.get("occupation", "")).lower()
    if any(kw in occ for kw in _FINANCE_OCCUPATION_KEYWORDS):
        score = min(1.0, score + 0.3)
    if score >= 0.7:
        return "high"
    if score < 0.35:
        return "low"
    return "medium"


def _institution_type_weights(
    persona: dict[str, Any],
    literacy: str,
) -> dict[str, float]:
    """Soft prior over institution TYPES for this persona (base 1.0 each).

    Ecological biasing (retiree -> retirement/wealth; investor / high-literacy
    -> brokerage/wealth; younger -> first brokerage). Additive boosts only, so
    every type keeps its baseline weight and nothing is starved. Banks stay at
    baseline because everyday banking is near-universal.
    """
    weights: dict[str, float] = {}

    def boost(itype: str, amount: float) -> None:
        weights[itype] = weights.get(itype, 1.0) + amount

    age = _age_int(persona.get("age"))
    occ = str(persona.get("occupation", "")).lower()
    if age is not None and age >= 55:
        boost("retirement_provider", 1.5)
        boost("wealth_manager", 0.75)
    if age is not None and 35 <= age < 55:
        boost("retirement_provider", 0.5)
    if age is not None and age < 35:
        boost("brokerage", 0.5)
    if literacy == "high" or any(kw in occ for kw in _FINANCE_OCCUPATION_KEYWORDS):
        boost("brokerage", 1.0)
        boost("wealth_manager", 1.0)
    if literacy == "low":
        boost("bank", 0.5)  # basics-first users skew toward everyday banking
    return weights


def _category_weights(
    persona_tags: list[str],
    categories: "Sequence[Any]",
) -> dict[str, float]:
    """Soft prior over categories: +1 per persona tag a category is affine to.

    Data-driven via each category's authored ``persona_affinities`` (e.g.
    ``age:65+`` / ``financial-literacy:low``). Categories with no affinities, or
    no overlap, keep the 1.0 baseline — a soft prior, never a hard filter.
    """
    ptags = set(persona_tags)
    weights: dict[str, float] = {}
    for c in categories:
        w = 1.0
        for aff in getattr(c, "persona_affinities", ()) or ():
            if aff in ptags:
                w += 1.0
        weights[c.id] = w
    return weights


def _age_int(age: Any) -> int | None:
    try:
        return int(age)
    except (TypeError, ValueError):
        return None


def persona_to_tags(persona: dict[str, Any], locale: str = "en_US") -> list[str]:
    """Map a persona to `<prefix>:<value>` tags used for template matching."""
    tags: list[str] = []
    age_bin = _age_bin(persona.get("age"))
    if age_bin:
        tags.append(f"age:{age_bin}")
    region = _slug(persona.get("region") or persona.get("state"))
    if region:
        tags.append(f"region:{region}")
    occupation = _slug(persona.get("occupation"))
    if occupation:
        tags.append(f"occupation-family:{occupation}")
    tags.append(f"financial-literacy:{derive_financial_literacy(persona, locale)}")
    return tags


def derive_instance(
    persona: dict[str, Any],
    bank: FinanceBank,
    locale: str,
    *,
    persona_uuid: str,
    seed: int | None = None,
    tier_mix: float = 0.0,
    forced_tier: str | None = None,
    taxonomy: Any = None,
) -> FinanceInstance | None:
    """Instantiate a task for this persona, tier-branched.

    ``tier_mix`` is the probability of the ``dynamic`` tier (0.0 default =
    verifiable-only, preserving prior behavior); ``forced_tier`` overrides it.
    The ``dynamic`` tier requires a loaded ``taxonomy`` (a
    ``ProbingTaxonomy``); if absent or the picked institution's type has no
    applicable categories, we fall back to the verifiable tier.
    """
    tags = persona_to_tags(persona, locale)
    tier = _pick_tier(persona_uuid, seed, tier_mix, forced_tier, taxonomy)

    if tier == "dynamic" and taxonomy is not None:
        # Population soft-weighting: bias institution-type + category selection
        # toward ecologically-valid topics for this persona (life-stage /
        # occupation / financial-literacy), as a soft prior that never starves
        # a candidate (see _institution_type_weights / _category_weights).
        literacy = derive_financial_literacy(persona, locale)
        institution = select_institution(
            bank,
            tags,
            seed=seed,
            persona_uuid=persona_uuid,
            type_weights=_institution_type_weights(persona, literacy),
        )
        if institution is not None:
            # Scope categories to the institution's type by excluding the
            # non-matching ones, then reuse the shared core derivation.
            applicable = taxonomy.categories_for_type(institution.type)
            if applicable:
                applicable_ids = {c.id for c in applicable}
                excluded = [c.id for c in taxonomy.categories if c.id not in applicable_ids]
                picked = select_category(
                    persona,
                    taxonomy,
                    seed=seed,
                    excluded_category_ids=excluded,
                    category_weights=_category_weights(tags, applicable),
                )
                if picked is None:  # pragma: no cover — applicable is non-empty
                    return None
                category, hint = picked
                return build_dynamic_instance(
                    institution,
                    category_id=category.id,
                    invitation=category.invitation,
                    subtopic_hint=hint,
                    taxonomy_id=taxonomy.taxonomy_id,
                    taxonomy_version=taxonomy.taxonomy_version,
                    persona_uuid=persona_uuid,
                    locale=locale,
                    placeholder=bool(getattr(taxonomy, "placeholder", False) or institution.placeholder),
                )
        # No institution / no applicable categories -> fall back to verifiable.

    picked = select_institution_and_template(
        bank,
        tags,
        seed=seed,
        persona_uuid=persona_uuid,
        tier="verifiable",
    )
    if picked is None:
        return None
    institution, template = picked
    return build_instance(
        template,
        institution,
        persona_uuid=persona_uuid,
        locale=locale,
        seed=seed,
    )


def _pick_tier(
    persona_uuid: str,
    seed: int | None,
    tier_mix: float,
    forced_tier: str | None,
    taxonomy: Any,
) -> str:
    """Deterministically choose ``verifiable`` vs ``dynamic`` for this instance."""
    if forced_tier in ("verifiable", "dynamic"):
        return forced_tier
    if taxonomy is None or tier_mix <= 0.0:
        return "verifiable"
    if tier_mix >= 1.0:
        return "dynamic"
    rng = random.Random(_instance_seed(persona_uuid or "anon", "tier", seed))
    return "dynamic" if rng.random() < tier_mix else "verifiable"
