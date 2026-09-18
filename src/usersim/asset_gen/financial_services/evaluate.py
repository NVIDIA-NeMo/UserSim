# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DD-judge asset EVALUATION: a per-document quality scorecard over a committed bank.

The judge-ensemble layer on top of the deterministic ``validate.py`` gate (mirrors
how the evaluator's LLM judges sit on top of the mechanical scorers). Non-blocking
in v1: it produces a scorecard for review/selection, it does not gate.

Decoupled from generation: evaluates ANY committed bank, per-DOCUMENT, on richer
subjective axes than the per-cluster gen-time QC. Network-gated (LLM judge), so the
live path is not exercised in offline CI; the pure pieces (``build_eval_seed``,
``build_eval_columns``, ``shape_scorecard``) are tested offline.

Axes: realism, regulatory_soundness, clarity_readability, self_containment,
brief_consistency (judged by a family DISTINCT from doc-gen -- ``asset_judge_model``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from usersim.asset_gen.financial_services.prompts import EVAL_PROMPT
from usersim.asset_gen.financial_services.spec import RegionSpec

#: judge model alias (distinct from the doc-gen model to avoid self-preference).
EVAL_JUDGE_MODEL_ALIAS = "asset_judge_model"

#: per-axis score (1-5) below which a document is flagged for review.
LOW_SCORE_THRESHOLD = 3

#: (rubric name, flattened column name); the column order defines the scorecard.
_AXES: tuple[tuple[str, str], ...] = (
    ("Realism", "realism_score"),
    ("RegulatorySoundness", "regulatory_soundness_score"),
    ("ClarityReadability", "clarity_readability_score"),
    ("SelfContainment", "self_containment_score"),
    ("BriefConsistency", "brief_consistency_score"),
)


def evaluate_scores() -> list[Any]:
    """The per-document judge rubrics (``dd.Score``, 1-5)."""
    import data_designer.config as dd

    def score(name: str, description: str, high: str, low: str) -> Any:
        return dd.Score(
            name=name,
            description=description,
            options={
                5: f"Excellent. {high}",
                4: "Good, minor issues.",
                3: "Acceptable but noticeably rough.",
                2: "Weak; multiple problems.",
                1: f"Poor. {low}",
            },
        )

    return [
        score(
            "Realism",
            "Reads like a real document of its genre for this institution.",
            "Convincing, specific, plausible.",
            "Generic/templated or implausible.",
        ),
        score(
            "RegulatorySoundness",
            "Consistent with the applicable regulator context; no misstatements.",
            "Regulatorily sound and appropriately cautious.",
            "Misstates or ignores the rules.",
        ),
        score(
            "ClarityReadability",
            "Clear and appropriate for the institution's brand voice + reading level.",
            "Clear, well-structured, on-voice.",
            "Confusing or off-voice.",
        ),
        # Reading this axis: it runs ~2.7-2.9 corpus-wide and the low scorers cluster
        # on the TABULAR / reference genres -- discoverable_tool_doc 2.19,
        # fee_schedule 2.31, terms_conditions 2.47, eligibility_matrix and promo_notice
        # 2.52 -- while every prose genre sits at 2.9-3.0 (faq 3.02, product_sheet
        # 3.02, disclosure 3.03, security_advisory 3.00, troubleshooting 2.99).
        #
        # Two tempting explanations were measured and BOTH are wrong:
        #   - LENGTH. The pooled correlation(words, score) is +0.32, but the median
        #     WITHIN-genre correlation is +0.07, and it is NEGATIVE for fee_schedule
        #     (-0.19) and rate_sheet (-0.13). The pooled figure is Simpson's paradox:
        #     rich-tier genres are both longer and better. terms_conditions is already
        #     in the longest tier (258 median words) and still scores 2.47; faq is
        #     short (142) and scores 3.02. Raising the length budget would very likely
        #     make the tabular genres worse.
        #   - SUBSTANCE DEFERRAL via raw doc_key references. Real (those docs average
        #     2.38 vs 2.78) but it affects only ~0.7% of documents, so it cannot
        #     explain 18 of 27 fee_schedules scoring <=2. Now caught deterministically
        #     by validate.py's doc_key_leak check.
        #
        # What remains is a rubric-fit hypothesis: "usable on its own ... a policy is
        # actionable" is a prose standard, and a fee table or a tool signature cannot
        # satisfy it however well written. That is UNCONFIRMED. Treat a flat ~2.5 on
        # the tabular genres as un-diagnosed rather than as a known defect, and do not
        # re-derive a length theory from the pooled correlation.
        score(
            "SelfContainment",
            "Usable on its own (a FAQ answers its question; a policy is actionable).",
            "Self-contained and actionable.",
            "Depends on missing context.",
        ),
        score(
            "BriefConsistency",
            "Follows the institution brief's positioning/policy stance.",
            "Fully consistent with the brief.",
            "Contradicts the brief.",
        ),
    ]


def build_eval_columns() -> list[Any]:
    """DD columns: one LLM-judge over each doc + per-axis score flatteners."""
    import data_designer.config as dd

    cols: list[Any] = [
        dd.LLMJudgeColumnConfig(
            name="doc_eval",
            model_alias=EVAL_JUDGE_MODEL_ALIAS,
            prompt=EVAL_PROMPT,
            scores=evaluate_scores(),
        )
    ]
    for rubric, col in _AXES:
        cols.append(dd.ExpressionColumnConfig(name=col, expr="{{ doc_eval.%s.score }}" % rubric))
    return cols


# ── seed: one row per committed document ────────────────────────────────


def build_eval_seed(bank_dir: str | Path, region_spec: RegionSpec | None):
    """One DD seed row per committed document, carrying judge context (pure/offline).

    Joins each document with its institution's generated brief (from
    ``_generation_report.json``), the product's authoritative typed values (from the
    region_spec, keyed on ``product_category``), and the domain's regulator text.
    """
    import pandas as pd

    from usersim.asset_gen.financial_services.validate import _read_bank

    bank_dir = Path(bank_dir)
    region_meta, insts = _read_bank(bank_dir)

    # brief per institution (best-effort, from the generation report).
    briefs: dict[str, str] = {}
    report_path = bank_dir / "_generation_report.json"
    if report_path.exists():
        try:
            gr = json.loads(report_path.read_text(encoding="utf-8"))
            for iid, idata in (gr.get("institutions", {}) or {}).items():
                if idata.get("brief") is not None:
                    briefs[iid] = json.dumps(idata["brief"], ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    # authoritative typed values per (institution, product_category).
    values: dict[tuple[str, str], str] = {}
    if region_spec is not None:
        for inst in region_spec.institutions:
            for product in inst.products:
                values[(inst.id, product.id)] = json.dumps(
                    {k: v.value for k, v in product.variables.items()}, ensure_ascii=False
                )

    dr = region_meta.get("domain_regulators", {}) or {}
    language = region_spec.language if region_spec is not None else ""

    rows: list[dict[str, Any]] = []
    for iid, idata in insts.items():
        for d in idata.documents:
            domain = d.get("domain", "")
            reg = dr.get(domain, {}) if isinstance(dr, dict) else {}
            rows.append(
                {
                    "id": d.get("id"),
                    "institution_id": iid,
                    "document_type": d.get("document_type", ""),
                    "title": d.get("title", ""),
                    "body": d.get("body", ""),
                    "language": language,
                    "institution_brief": briefs.get(iid, "{}"),
                    "authoritative_values": values.get((iid, d.get("product_category")), "{}"),
                    "regulator_text": (reg.get("regulator_text", "") if isinstance(reg, dict) else ""),
                }
            )
    return pd.DataFrame(rows)


# ── scorecard shaping ────────────────────────────────────────────────────


@dataclass
class EvalScorecard:
    per_doc: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        lines = ["  aggregate (mean 1-5):"]
        for _, col in _AXES:
            lines.append(f"    {col}: {self.aggregate.get(col, float('nan')):.2f}")
        low = self.aggregate.get("low_docs", [])
        lines.append(f"  {len(low)} doc(s) below {LOW_SCORE_THRESHOLD} on some axis" + (f": {low[:10]}" if low else ""))
        return "\n".join(lines)


def shape_scorecard(result_df) -> EvalScorecard:
    """Flatten a judged dataframe into a per-doc scorecard + aggregate (pure)."""

    def _num(v: Any) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    per_doc: list[dict[str, Any]] = []
    low_docs: list[str] = []
    sums: dict[str, float] = {col: 0.0 for _, col in _AXES}
    counts: dict[str, int] = {col: 0 for _, col in _AXES}

    for _, row in result_df.iterrows():
        rec: dict[str, Any] = {
            "id": row.get("id"),
            "institution_id": row.get("institution_id"),
            "document_type": row.get("document_type"),
        }
        is_low = False
        for _, col in _AXES:
            val = _num(row.get(col))
            rec[col] = val
            if val is not None:
                sums[col] += val
                counts[col] += 1
                if val < LOW_SCORE_THRESHOLD:
                    is_low = True
        per_doc.append(rec)
        if is_low and rec["id"] is not None:
            low_docs.append(str(rec["id"]))

    aggregate: dict[str, Any] = {col: (sums[col] / counts[col] if counts[col] else float("nan")) for _, col in _AXES}
    aggregate["n_docs"] = len(per_doc)
    aggregate["low_docs"] = low_docs
    return EvalScorecard(per_doc=per_doc, aggregate=aggregate)


# ── entry point (network-gated) ─────────────────────────────────────────


def validate_eval_models(models: Any) -> None:
    from usersim.cli._models import require_aliases

    require_aliases(models, required=(EVAL_JUDGE_MODEL_ALIAS,), context="evaluate-assets --domain financial_services")


#: Where superseded scorecards go, relative to the bank dir.
SCORECARD_HISTORY_DIR = "_scorecards"


def _archive_previous_scorecard(bank_dir: Path) -> Path | None:
    """Move an existing ``_eval_scorecard.parquet`` into ``_scorecards/`` first.

    Regeneration used to overwrite the scorecard in place, so the previous run's
    numbers were gone the moment you produced new ones -- which makes "did that prompt
    change help?" unanswerable, and it repeatedly was. Archiving on write keeps the
    stable filename (the asset notebook reads that exact path) while preserving each
    superseded run, INCLUDING the one already on disk when this first ships.

    Stamped by the file's own mtime, so the name reflects when those scores were
    produced rather than when they were displaced. Note ``bank_dir`` is the scratch
    output dir, so this is for run-to-run comparison and is not durable provenance --
    it does not survive cleaning scratch and does not travel between machines.
    Returns the archived path, or ``None`` when there was nothing to archive.
    """
    current = bank_dir / "_eval_scorecard.parquet"
    if not current.is_file():
        return None
    stamp = datetime.fromtimestamp(current.stat().st_mtime, timezone.utc)
    dest_dir = bank_dir / SCORECARD_HISTORY_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stamp:%Y%m%dT%H%M%SZ}.parquet"
    if dest.exists():  # same-second re-run; keep both rather than clobber
        dest = dest_dir / f"{stamp:%Y%m%dT%H%M%SZ}-{current.stat().st_size}.parquet"
    current.replace(dest)
    return dest


def evaluate_bank(
    bank_dir: str | Path,
    region_spec: RegionSpec | None,
    models: Any,
) -> EvalScorecard:
    """Judge each committed document and write ``_eval_scorecard.parquet``.

    Live (needs the judge alias + network; validated up front). Returns the
    :class:`EvalScorecard`; the pure pieces are tested offline.
    """
    validate_eval_models(models)

    import data_designer.config as dd

    from usersim.asset_gen.financial_services.pipeline import _make_engine
    from usersim.cli._models import to_model_configs

    seed = build_eval_seed(bank_dir, region_spec)
    if seed.empty:
        return EvalScorecard()

    engine = _make_engine(models)
    builder = dd.DataDesignerConfigBuilder(model_configs=to_model_configs(models))
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed))
    for col in build_eval_columns():
        builder.add_column(col)
    result = engine.create(builder, num_records=len(seed)).load_dataset().reset_index(drop=True)

    scorecard = shape_scorecard(result)
    import pandas as pd

    _archive_previous_scorecard(Path(bank_dir))
    pd.DataFrame(scorecard.per_doc).to_parquet(Path(bank_dir) / "_eval_scorecard.parquet")
    return scorecard
