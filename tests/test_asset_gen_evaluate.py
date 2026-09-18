# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the DD-judge asset evaluation (asset_gen/financial_services/evaluate.py).

The live judge path is network-gated; these exercise the pure pieces:
build_eval_seed (reads a bank), build_eval_columns (config assembly),
shape_scorecard (from a mocked judge dataframe), and the endpoint-validation path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from usersim.asset_gen.financial_services.evaluate import (
    build_eval_columns,
    build_eval_seed,
    evaluate_scores,
    shape_scorecard,
    validate_eval_models,
)
from usersim.asset_gen.financial_services.pipeline import InstitutionBank, serialize_bank
from usersim.asset_gen.financial_services.spec import RegionSpec

_SPEC_DICT: dict[str, Any] = {
    "locale": "en_US",
    "region": "US",
    "currency": "USD",
    "currency_symbol": "$",
    "language": "English (US)",
    "document_types": ["product_sheet"],
    "domain_regulators": {"retail_banking": {"regulator_text": "Reg E applies."}},
    "institutions": [
        {
            "id": "testbank",
            "display_name": "Test Bank",
            "type": "bank",
            "brand_voice": "traditional",
            "domains": ["retail_banking"],
            "products": [
                {
                    "id": "checking",
                    "display_name": "Checking",
                    "domain": "retail_banking",
                    "variables": {"overdraft_fee": {"type": "money", "value": 34}},
                }
            ],
            "tool_taxonomy": [{"name": "kb_search", "discoverable": False, "side_effect_class": "read_only"}],
            "task_templates": [
                {"id": "T1", "tier": "dynamic", "task_type": "advisory_qa", "persona_tags": ["region:any"]}
            ],
        }
    ],
}


def _write_bank(tmp_path: Path) -> Path:
    region_meta = {"locale": "en_US", "domain_regulators": {"retail_banking": {"regulator_text": "Reg E applies."}}}
    inst = InstitutionBank(
        institution_meta={"institution_id": "testbank", "type": "bank", "domains": ["retail_banking"]},
        documents=[
            {
                "id": "D1",
                "title": "Checking overview",
                "body": "Overdraft fee $34.",
                "domain": "retail_banking",
                "product_category": "checking",
                "document_type": "product_sheet",
                "source_authority": "official",
                "placeholder": True,
                "mentions_tools": [],
            }
        ],
        tools=[
            {
                "name": "kb_search",
                "description": "",
                "discoverable": False,
                "side_effect_class": "read_only",
                "parameters": {},
            }
        ],
    )
    root = serialize_bank(tmp_path / "en_US", region_meta, [inst], write_tasks=False)
    (root / "_generation_report.json").write_text(
        json.dumps({"institutions": {"testbank": {"brief": {"positioning": "no-fee simplicity"}}}}), encoding="utf-8"
    )
    return root


def test_build_eval_seed_joins_context(tmp_path: Path):
    bank = _write_bank(tmp_path)
    seed = build_eval_seed(bank, RegionSpec.model_validate(_SPEC_DICT))
    assert len(seed) == 1
    row = seed.iloc[0]
    assert row["id"] == "D1" and row["language"] == "English (US)"
    assert "34" in row["authoritative_values"]  # from region_spec
    assert row["regulator_text"] == "Reg E applies."  # from region_meta
    assert "no-fee simplicity" in row["institution_brief"]  # from generation report


def test_build_eval_columns_axes():
    cols = build_eval_columns()
    by_type: dict[str, list[Any]] = {}
    for c in cols:
        by_type.setdefault(c.column_type, []).append(c)
    assert "llm-judge" in by_type
    expr_names = {c.name for c in by_type.get("expression", [])}
    assert expr_names == {
        "realism_score",
        "regulatory_soundness_score",
        "clarity_readability_score",
        "self_containment_score",
        "brief_consistency_score",
    }
    assert {s.name for s in evaluate_scores()} == {
        "Realism",
        "RegulatorySoundness",
        "ClarityReadability",
        "SelfContainment",
        "BriefConsistency",
    }


def test_shape_scorecard_aggregates_and_flags_low():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "id": "D1",
                "institution_id": "testbank",
                "document_type": "faq",
                "realism_score": 5,
                "regulatory_soundness_score": 5,
                "clarity_readability_score": 5,
                "self_containment_score": 5,
                "brief_consistency_score": 5,
            },
            {
                "id": "D2",
                "institution_id": "testbank",
                "document_type": "faq",
                "realism_score": 4,
                "regulatory_soundness_score": 2,
                "clarity_readability_score": 4,
                "self_containment_score": 4,
                "brief_consistency_score": 4,
            },
        ]
    )
    card = shape_scorecard(df)
    assert card.aggregate["n_docs"] == 2
    assert card.aggregate["regulatory_soundness_score"] == pytest.approx(3.5)
    assert card.aggregate["low_docs"] == ["D2"]  # has a 2 (< 3)


def test_validate_eval_models_requires_judge():
    from usersim.cli._models import ConfigError, ModelsConfig, ModelSpec

    cfg = ModelsConfig(providers=(), models=(ModelSpec(alias="doc_gen_model", model="m", provider="nvidia"),))
    with pytest.raises(ConfigError):
        validate_eval_models(cfg)


# ── scorecard history ────────────────────────────────────────────────────────


def test_previous_scorecard_is_archived_rather_than_overwritten(tmp_path):
    """Regeneration used to overwrite the scorecard in place, so the previous run's
    numbers vanished the moment you produced new ones -- which is why "did that prompt
    change help?" was repeatedly unanswerable. Archiving on write also rescues the
    scorecard already sitting on disk the first time this ships.
    """
    import pandas as pd

    from usersim.asset_gen.financial_services.evaluate import (
        SCORECARD_HISTORY_DIR,
        _archive_previous_scorecard,
    )

    stable = tmp_path / "_eval_scorecard.parquet"

    # Nothing to archive on a first run.
    assert _archive_previous_scorecard(tmp_path) is None

    pd.DataFrame({"id": ["a"], "realism_score": [3]}).to_parquet(stable)
    archived = _archive_previous_scorecard(tmp_path)

    assert archived is not None and archived.is_file()
    assert archived.parent.name == SCORECARD_HISTORY_DIR
    assert not stable.exists()  # moved, not copied
    assert pd.read_parquet(archived)["realism_score"].tolist() == [3]
    assert archived.name.endswith(".parquet") and archived.name[8] == "T"  # UTC stamp


def test_evaluate_bank_actually_calls_the_archiver():
    """The tests above exercise the archiver directly, so they stay green even if its
    CALL SITE is deleted -- which would silently restore the overwrite-in-place
    behaviour. ``evaluate_bank`` is network-gated (judge alias + live DD engine), so
    assert on its source: the call must precede the write it protects.
    """
    import inspect

    from usersim.asset_gen.financial_services import evaluate as E

    src = inspect.getsource(E.evaluate_bank)
    assert "_archive_previous_scorecard(" in src
    assert src.index("_archive_previous_scorecard(") < src.index("to_parquet(")


def test_two_runs_leave_two_history_files(tmp_path):
    import time

    import pandas as pd

    from usersim.asset_gen.financial_services.evaluate import (
        SCORECARD_HISTORY_DIR,
        _archive_previous_scorecard,
    )

    stable = tmp_path / "_eval_scorecard.parquet"
    for value in (3, 4):
        pd.DataFrame({"id": ["a"], "realism_score": [value]}).to_parquet(stable)
        time.sleep(1.05)  # the stamp has second resolution
        stable.touch()
        _archive_previous_scorecard(tmp_path)
    history = sorted((tmp_path / SCORECARD_HISTORY_DIR).glob("*.parquet"))
    assert len(history) == 2, [p.name for p in history]
