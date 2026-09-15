# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration for the financial_services probe: a trajectory flows
through eval -> select -> report for BOTH tiers and surfaces the finance
capabilities.

The *simulate* half (probe construction + ``run_dispatch``) is covered by
``tests/test_simulator_pipeline_e2e.py`` (which parametrizes over
``financial_services``) and the probe unit tests. This test starts from
faithful trajectory rows (the exact columns the probe emits) and exercises the
REAL pieces downstream of simulation:

- the tier-aware scorer (deterministic verifier for ``verifiable``; the
  reference-grounded judge for ``dynamic``, with the judge LLM mocked),
- the selection projection (finance gold-metadata retained in ``compact``),
- the capability report (aggregate + per-institution-type ``financial_task_success``
  and the cross-domain ``dynamic_grounding`` capability).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from usersim.engine.core.finance_bank import (
    load_finance_bank_for_locale,
    reset_finance_bank_cache,
)
from usersim.engine.evaluator.scorers import get_scorer, load_default_scorers
from usersim.engine.evaluator.scorers import financial_services as FS
from usersim.engine.probes.financial_services.generator import (
    FINANCE_TRAJECTORY_COLUMNS,
)
from usersim.reporting.capability_report import build_capability_report
from usersim.selection.schemas import resolve_columns


@pytest.fixture(autouse=True)
def _fresh():
    reset_finance_bank_cache()
    load_default_scorers()  # ensure the financial_services scorer is registered
    yield
    reset_finance_bank_cache()


def _outcome(status: str = "ok", n_tool_calls: int = 0) -> str:
    return json.dumps({
        "status": status,
        "failure_class": None,
        "failure_attribution": None,
        "failure_detail": "",
        "n_turns": 2,
        "n_tool_calls": n_tool_calls,
        "n_user_query_attempts": 1,
        "n_user_followup_retries": 0,
        "n_assistant_inline_failures": 0,
        "n_api_response_rerolls": 0,
        "n_fourth_wall_triggers": 0,
        "n_user_role_violations": 0,
        "warnings": [],
        "early_stop": False,
        "per_model_input_tokens": {"user_model": 80, "assistant_model": 90},
        "per_model_output_tokens": {"user_model": 30, "assistant_model": 70},
        "per_model_calls": {"user_model": 2, "assistant_model": 2},
        "wall_clock_s_by_alias": {"user_model": 2.0, "assistant_model": 3.0},
        "wall_clock_s": 6.0,
        "provenance": {
            "nemotron_personas_version": "2026-04",
            "scenario_prompt_version": "v1.0",
            "code_sha": "test",
            "bank_version": {},
        },
    })


def _messages(pairs) -> str:
    out = []
    for u, a in pairs:
        out.append({"role": "user", "content": u})
        out.append({"role": "assistant", "content": a})
    return json.dumps(out)


def _base_row(**overrides):
    """A finance trajectory row with every FINANCE_TRAJECTORY_COLUMN present."""
    row = {c: "" for c in FINANCE_TRAJECTORY_COLUMNS}
    row.update({
        "gold_document_ids": json.dumps([]),
        "gold_tool_sequence": json.dumps([]),
        "expected_state_deltas": json.dumps({}),
        "retrieved_document_ids": json.dumps([]),
        "attempted_tool_names": json.dumps([]),
        "num_tool_calls": 0,
        # standard trajectory columns the report reads.
        "persona_uuid": "p_fin_0001",
        "probe_family": "financial_services",
        "probe_variant": "default",
        "locale": "en_US",
        "conversation_language": "English",
        "user_interaction_style": "neutral",
        "disclosure_style": "upfront",
        "persona_grounding": True,
        "conversation_messages": _messages([("hi", "hello")]),
        "simulation_outcome": _outcome(),
    })
    row.update(overrides)
    return row


def _verifiable_row():
    bank = load_finance_bank_for_locale("en_US")
    inst = bank.institution("northwind_bank")
    tpl = inst.template_by_id("TPL-NB-DISPUTE-001")
    gold_docs = list(tpl.gold_document_ids)  # auto-derived at load
    return _base_row(
        trajectory_id="t_fin_verifiable_1",
        finance_task_id="TPL-NB-DISPUTE-001",
        task_tier="verifiable",
        task_type="transactional",
        institution_id="northwind_bank",
        institution_type="bank",
        domain="retail_banking",
        gold_tool_sequence=json.dumps(["file_transaction_dispute"]),
        gold_document_ids=json.dumps(gold_docs),
        retrieved_document_ids=json.dumps(gold_docs),
        attempted_tool_names=json.dumps(["file_transaction_dispute"]),
        num_tool_calls=1,
        simulation_outcome=_outcome(n_tool_calls=1),
        conversation_messages=_messages([
            ("There's a charge I don't recognize.", "Filed a dispute; it's under review."),
        ]),
    )


def _dynamic_row():
    bank = load_finance_bank_for_locale("en_US")
    inst = bank.institution("northwind_bank")
    return _base_row(
        trajectory_id="t_fin_dynamic_1",
        finance_task_id="DYN-northwind_bank-banking-account-fit",
        task_tier="dynamic",
        task_type="dynamic",
        institution_id="northwind_bank",
        institution_type="bank",
        domain="retail_banking",
        dynamic_category_id="banking-account-fit",
        dynamic_subtopic_hint="checking vs. savings fit",
        taxonomy_version="v0.1.0",
        retrieved_document_ids=json.dumps([inst.documents[0].id]),
        conversation_messages=_messages([
            ("Which everyday account fits me?", "Here are the options grounded in our KB..."),
        ]),
    )


def _dyn_judge_payload():
    axes = {
        "dynamic.grounding": 5, "dynamic.numeric_faithfulness": 5,
        "dynamic.boundary_adherence": 4, "dynamic.no_fabrication": 5,
        "dynamic.graceful_unknown": 5, "dynamic.self_consistency": 5,
    }
    return json.dumps({a: {"score": v, "reasoning": "ok"} for a, v in axes.items()})


def _eval_cell(scorer_block) -> str:
    return json.dumps({
        "envelope": {
            "judge_aliases": ["judge_model"],
            "axes": [],
            "scorers": ["financial_services"],
            "prompt_version": "v1.0",
            "evaluator_version": "v1.0",
        },
        "axes": {},
        "scorers": {"financial_services": scorer_block},
        "skipped": False,
        "skipped_reason": None,
    })


def test_financial_services_eval_select_report_both_tiers():
    pd = pytest.importorskip("pandas")

    ver = _verifiable_row()
    dyn = _dynamic_row()
    traj_df = pd.DataFrame([ver, dyn])

    # ── eval: run the REAL tier-aware scorer over each trajectory ──
    scorer = get_scorer("financial_services")
    ver_block = scorer(ver, {})  # verifiable -> deterministic, no LLM
    assert ver_block["scores"]["finance.tool_selection_rate"] == 1.0
    assert ver_block["status_proposal"] is True

    with patch.object(FS, "call_llm", return_value={"content": _dyn_judge_payload()}):
        dyn_block = scorer(dyn, {"judge_model": object()})
    assert dyn_block["task_tier"] == "dynamic"
    assert dyn_block["scores"]["dynamic.numeric_faithfulness"]["score"] == 5

    eval_df = pd.DataFrame({
        "trajectory_id": [ver["trajectory_id"], dyn["trajectory_id"]],
        "locale": ["en_US", "en_US"],
        "probe_family": ["financial_services", "financial_services"],
        "assistant_eval": [_eval_cell(ver_block), _eval_cell(dyn_block)],
    })

    # ── select: the compact projection keeps the finance gold-metadata ──
    cols = resolve_columns("compact", list(traj_df.columns))
    for c in ("finance_task_id", "task_tier", "institution_type",
              "gold_tool_sequence", "expected_state_deltas", "dynamic_category_id"):
        assert c in cols, f"compact projection dropped {c}"

    # ── report: finance capabilities surface, stratified by tier + type ──
    report = build_capability_report(
        traj_df, eval_df, eval_column="assistant_eval",
        run_id="fin-e2e", model_id="test-model", sample_mode="prototype", min_n=1,
    )
    by_key = {(c.capability, c.locale): c for c in report.capability_cells}

    # Verifiable transactional success (aggregate) scored from the verifier axes.
    agg = by_key[("financial_task_success", "en_US")]
    assert agg.score is not None and agg.score >= 0.9
    # Retrieval recall is surfaced as its OWN capability (decoupled from success),
    # scored from finance.document_recall_rate on the verifiable row (retrieved
    # == gold here, so recall is perfect).
    recall = by_key[("financial_retrieval_recall", "en_US")]
    assert recall.score is not None and recall.score >= 0.9
    # Per-institution-type breakdown: the bank cell is scored...
    bank_cell = by_key[("financial_task_success_bank", "en_US")]
    assert bank_cell.score is not None and bank_cell.score >= 0.9
    # ...while a type with no rows in this locale is present but unscored
    # (locale-aware no-data cell).
    assert by_key[("financial_task_success_brokerage", "en_US")].score is None

    # Cross-domain grounded-advice capability scored from the dynamic judge axes.
    grounded = by_key[("dynamic_grounding", "en_US")]
    assert grounded.score is not None
    assert grounded.source_probes == ["financial_services"]


def test_dynamic_row_feeds_no_verifiable_capability():
    """Tier isolation, checked against the capability registry rather than a
    naming convention.

    The property that matters is that a dynamic trajectory supplies no evidence to
    any VERIFIER-scored capability. Asserting "every axis starts with dynamic."
    was a proxy for that, and it broke once the tiers gained a legitimately shared
    axis (``finance.substantive_retrieval_rate``: retrieval quality needs no gold,
    so it is defined for both tiers, and the dynamic tier is where an ungrounded
    answer does the most damage). Deriving the forbidden set from the registry
    keeps the real guarantee and still fails if a verifier axis ever leaks here.
    """
    from usersim.taxonomy.capabilities import (
        FINANCE_CAPABILITY_INSTITUTION_TYPES,
        capability_definitions,
    )

    dyn = _dynamic_row()
    scorer = get_scorer("financial_services")
    with patch.object(FS, "call_llm", return_value={"content": _dyn_judge_payload()}):
        block = scorer(dyn, {"judge_model": object()})
    axes = set(block["scores"])

    verifiable_only = {
        "financial_task_success", "financial_retrieval_recall",
    } | {
        f"financial_task_success_{t}" for t in FINANCE_CAPABILITY_INSTITUTION_TYPES
    }
    forbidden = {
        axis
        for cap in capability_definitions() if cap.id in verifiable_only
        for src in cap.sources
        for axis in src.axes
    }
    assert forbidden, "registry lookup found no verifiable axes — test is vacuous"
    assert not (axes & forbidden), (
        f"dynamic row supplies verifier evidence: {sorted(axes & forbidden)}"
    )
    # The judged axes are all present, and the only finance axis is the shared
    # retrieval diagnostic.
    assert {a for a in axes if a.startswith("dynamic.")}
    assert {a for a in axes if a.startswith("finance.")} <= {
        "finance.substantive_retrieval_rate"
    }
