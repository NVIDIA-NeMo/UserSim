# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-time task instantiation + deterministic gold-derivation resolver.

This is the runtime contract shared by the ``financial_services`` probe (which
instantiates a template into a concrete conversation at sim time) and by
``asset_gen/validate.py`` (which reuses the same resolver to prove Tier-1 tasks
are well-posed). Keeping it in the plugin — importable by both — is what makes
gold a single source of truth with no sim-time/validate-time drift.

A "task" is a parameterized generator, not a fixed string:

    instance = template x persona x generated account-state x locale

Instantiation is deterministic given ``(persona_uuid, template_id, seed)`` so
re-runs are idempotent and reproducible. ``TASK_CONTRACT_VERSION`` pins this
CODE contract; bump it when the derivation logic changes so downstream
selection/reporting can detect drift independently of the data's ``bank_version``.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from usersim.engine.core.finance_bank import (
    FinanceBank,
    Institution,
    TaskTemplate,
)

# The gold-derivation code contract. Bump on any change to how slots are
# filled or gold is derived. Persisted into provenance + trajectory columns.
TASK_CONTRACT_VERSION = "v0.1.0"

_SLOT_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


@dataclass(frozen=True)
class GoldSpec:
    """The verifiable ground truth for one Tier-1 instance."""

    gold_document_ids: Tuple[str, ...]
    gold_tool_sequence: Tuple[str, ...]
    expected_state_deltas: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinanceInstance:
    """A concrete instantiation of a task for one conversation.

    Carries the grounding institution + domain so retrieval, tools, and the
    scorer stay scoped to a single institution. Two tiers:

    - ``verifiable`` — instantiated from a scripted ``TaskTemplate`` with a
      param-locked ``opening_user_message`` + deterministic ``gold``.
    - ``dynamic`` — instantiated from the authored dynamic-tier taxonomy: no
      scripted opening (the loop generates turn-1 from ``dynamic_invitation`` +
      ``dynamic_subtopic_hint``) and no gold (judge-scored, grounded in the KB).
    """

    template_id: str
    tier: str
    task_type: str
    placeholder: bool
    locale: str
    institution_id: str
    institution_type: str
    domain: str
    opening_user_message: str
    account_state: Dict[str, Any]
    params: Dict[str, Any]
    gold: Optional[GoldSpec]  # None for the dynamic tier (judge-scored)
    task_contract_version: str = TASK_CONTRACT_VERSION
    # Dynamic-tier fields (empty for verifiable instances).
    dynamic_category_id: str = ""
    dynamic_invitation: str = ""
    dynamic_subtopic_hint: str = ""
    taxonomy_id: str = ""
    taxonomy_version: str = ""

    @property
    def is_verifiable(self) -> bool:
        return self.tier == "verifiable" and self.gold is not None

    @property
    def is_dynamic(self) -> bool:
        return self.tier == "dynamic"


def _instance_seed(persona_uuid: str, template_id: str, seed: Optional[int]) -> int:
    """Deterministic per-instance seed (stable across Python sessions)."""
    key = f"{persona_uuid}|{template_id}|{seed if seed is not None else 0}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF


def _fill_slots(text: str, params: Dict[str, Any]) -> str:
    def repl(m: "re.Match[str]") -> str:
        key = m.group(1)
        return str(params.get(key, m.group(0)))

    return _SLOT_RE.sub(repl, text)


def _fill_obj(obj: Any, params: Dict[str, Any]) -> Any:
    """Recursively fill ``{{slot}}`` markers in strings within a nested object."""
    if isinstance(obj, str):
        return _fill_slots(obj, params)
    if isinstance(obj, dict):
        return {k: _fill_obj(v, params) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill_obj(v, params) for v in obj]
    return obj


def sample_params(template: TaskTemplate, rng: random.Random) -> Dict[str, Any]:
    """Pick one value per param-space slot, deterministically from ``rng``."""
    params: Dict[str, Any] = {}
    for slot, values in sorted(template.param_space.items()):
        if values:
            params[slot] = rng.choice(list(values))
    return params


def derive_gold(template: TaskTemplate, params: Dict[str, Any], account_state: Dict[str, Any]) -> Optional[GoldSpec]:
    """Deterministically derive the gold spec for a Tier-1 instance.

    For the current template shapes the gold is declared on the template and
    simply packaged here (slots filled). This is the single point that
    ``asset_gen/validate.py`` also calls, guaranteeing sim-time and
    validate-time gold agree. Returns ``None`` for open-ended (Tier-2) tasks.
    """
    if template.tier != "verifiable":
        return None
    return GoldSpec(
        gold_document_ids=template.gold_document_ids,
        gold_tool_sequence=template.gold_tool_sequence,
        expected_state_deltas=_fill_obj(dict(template.expected_state_deltas), params),
    )


def build_instance(
    template: TaskTemplate,
    institution: Institution,
    *,
    persona_uuid: str,
    locale: str,
    seed: Optional[int] = None,
) -> FinanceInstance:
    """Instantiate ``template`` (grounded in ``institution``) for one conversation."""
    rng = random.Random(_instance_seed(persona_uuid, f"{institution.institution_id}:{template.id}", seed))
    params = sample_params(template, rng)
    account_state = _fill_obj(dict(template.account_state), params)
    # Pick one of the template's opening phrasings (deterministic per instance)
    # so the assistant sees varied turn-1 wording while the gold stays fixed.
    opening = _fill_slots(rng.choice(list(template.opening_user_messages)), params)
    gold = derive_gold(template, params, account_state)
    return FinanceInstance(
        template_id=template.id,
        tier=template.tier,
        task_type=template.task_type,
        placeholder=template.placeholder,
        locale=locale,
        institution_id=institution.institution_id,
        institution_type=institution.type,
        domain=template.domain,
        opening_user_message=opening,
        account_state=account_state,
        params=params,
        gold=gold,
    )


def build_dynamic_instance(
    institution: Institution,
    *,
    category_id: str,
    invitation: str,
    subtopic_hint: str,
    taxonomy_id: str,
    taxonomy_version: str,
    persona_uuid: str,
    locale: str,
    placeholder: bool,
) -> FinanceInstance:
    """Instantiate a ``dynamic``-tier instance grounded in ``institution``.

    No scripted opening (the loop generates turn-1 from the invitation +
    subtopic hint) and no gold (grounded judge-scoring). Domain is the
    institution's primary domain, for stratified reporting.
    """
    domain = institution.domains[0] if institution.domains else ""
    return FinanceInstance(
        template_id=f"DYN-{institution.institution_id}-{category_id}",
        tier="dynamic",
        task_type="dynamic",
        placeholder=placeholder,
        locale=locale,
        institution_id=institution.institution_id,
        institution_type=institution.type,
        domain=domain,
        opening_user_message="",  # generated by the loop from invitation + hint
        account_state={},
        params={},
        gold=None,
        dynamic_category_id=category_id,
        dynamic_invitation=invitation,
        dynamic_subtopic_hint=subtopic_hint,
        taxonomy_id=taxonomy_id,
        taxonomy_version=taxonomy_version,
    )


def select_institution(
    bank: FinanceBank,
    persona_tags: List[str],
    *,
    seed: Optional[int] = None,
    persona_uuid: str = "",
    type_weights: Optional[Dict[str, float]] = None,
) -> Optional[Institution]:
    """Deterministically pick a persona-matching institution (permissive).

    Prefers institutions with at least one template whose persona_tags match,
    falling back to all institutions so the sampler is never starved. Used by
    the dynamic tier (which scopes categories to the picked institution's type).

    ``type_weights`` (institution_type -> positive weight) SOFT-WEIGHTS the
    pick toward ecologically-valid institution types for the persona (e.g. a
    retiree toward a retirement provider) while staying deterministic; ``None``
    keeps the uniform pick. A soft prior only — every candidate keeps a
    positive floor so no institution is starved.
    """
    insts = list(bank.institutions)
    if not insts:
        return None
    wanted = set(persona_tags)
    matched = [inst for inst in insts if any(_tags_match_pair(t.persona_tags, wanted) for t in inst.templates)]
    candidates = matched or insts
    rng = random.Random(_instance_seed(persona_uuid or "anon", "select_inst", seed))
    if type_weights:
        weights = [max(1e-6, float(type_weights.get(inst.type, 1.0))) for inst in candidates]
        return rng.choices(candidates, weights=weights, k=1)[0]
    return rng.choice(candidates)


def select_institution_and_template(
    bank: FinanceBank,
    persona_tags: List[str],
    *,
    seed: Optional[int] = None,
    persona_uuid: str = "",
    tier: Optional[str] = None,
) -> Optional[Tuple[Institution, TaskTemplate]]:
    """Deterministically pick a persona-matching (institution, template).

    Permissive matching (banking is universal): every ``(institution, template)``
    pair is a candidate; persona-tag matches are preferred but we fall back to the
    full set so the sampler is never starved. ``tier`` (when given) restricts to
    templates of that tier (e.g. ``"verifiable"``).
    """
    all_pairs: List[Tuple[Institution, TaskTemplate]] = [
        (inst, tpl) for inst in bank.institutions for tpl in inst.templates if tier is None or tpl.tier == tier
    ]
    if not all_pairs:
        return None
    matched: List[Tuple[Institution, TaskTemplate]] = [
        (inst, tpl) for inst, tpl in all_pairs if _tags_match_pair(tpl.persona_tags, set(persona_tags))
    ]
    candidates = matched or all_pairs
    rng = random.Random(_instance_seed(persona_uuid or "anon", "select", seed))
    return rng.choice(candidates)


def _tags_match_pair(tpl_tags: Tuple[str, ...], persona_tags: set) -> bool:
    if not tpl_tags:
        return True
    for tag in tpl_tags:
        if ":" not in tag:
            if tag not in persona_tags:
                return False
            continue
        _, value = tag.split(":", 1)
        if value == "any":
            continue
        if tag not in persona_tags:
            return False
    return True
