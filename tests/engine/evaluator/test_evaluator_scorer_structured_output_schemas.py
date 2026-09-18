# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for evaluator-scorer structured-output schemas.

These tests are the **enforcement guard** for the schema-design contract
in ``evaluator/scorers/__init__.py``: every enum-shaped string field must
be ``Literal[...]`` and every 1-5 judge score must be ``Literal[1..5]``
(or at minimum ``Field(ge=1, le=5)``). The contract pushes value
constraints into the JSON Schema so OpenAI strict structured-output mode
rejects invalid values *at decode time* — eliminating the silent
"judge invents a label, scorer drops the row" failure mode that left
heatmap cells as ``Inconclusive`` / ``Untested`` even when the underlying
probe ran successfully.

If a future scorer author types a vocabulary field as ``str`` instead of
``Literal[...]``, these tests fail loudly.
"""

from __future__ import annotations

import typing as _typing

from usersim.engine.evaluator.scorers.safety_agentic import (
    JudgeScore as AgenticJudgeScore,
)
from usersim.engine.evaluator.scorers.safety_agentic import (
    _AgenticJudgment,
)
from usersim.engine.evaluator.scorers.safety_chat_pressure import (
    _VALID_CLASSIFICATIONS,
    Classification,
    _PerTurnClassification,
    _PressureJudgment,
)
from usersim.engine.evaluator.scorers.safety_chat_pressure import (
    JudgeScore as PressureJudgeScore,
)

# ---------------------------------------------------------------------------
# safety_chat_pressure
# ---------------------------------------------------------------------------


class TestSafetyChatPressureSchema:
    def test_classification_field_is_literal_enum(self) -> None:
        """``classification`` MUST be a Literal so JSON Schema carries an enum.

        Without this, OpenAI strict structured-output cannot constrain the
        string value and the model is free to write ``"refusal"``,
        ``"full_refusal"``, ``"violation"`` — labels outside the documented
        vocabulary, which drop an entire trajectory from the capability cell.
        """
        schema = _PerTurnClassification.model_json_schema()
        classification_field = schema["properties"]["classification"]
        assert "enum" in classification_field, (
            "classification field must enforce its vocabulary at the JSON "
            "Schema level (Literal[...]). See evaluator/scorers/__init__.py "
            "for the schema-design contract."
        )
        assert set(classification_field["enum"]) == set(_typing.get_args(Classification))
        # The public _VALID_CLASSIFICATIONS tuple should mirror the Literal
        # so consumers can enumerate the valid values without re-deriving.
        assert set(_VALID_CLASSIFICATIONS) == set(_typing.get_args(Classification))

    def test_per_turn_judge_scores_are_literal_1_to_5(self) -> None:
        """Both 1-5 judge scores MUST be ``Literal[1, 2, 3, 4, 5]``."""
        schema = _PressureJudgment.model_json_schema()
        for field in ("refusal_held_score", "contextual_safety_persistence_score"):
            field_schema = schema["properties"][field]
            assert "enum" in field_schema, (
                f"{field} must be a Literal[1..5] so the strict-output "
                f"decoder rejects out-of-range values at decode time."
            )
            assert set(field_schema["enum"]) == {1, 2, 3, 4, 5}

    def test_judge_score_alias_is_one_to_five(self) -> None:
        """The exported ``JudgeScore`` alias should be ``Literal[1..5]``."""
        assert set(_typing.get_args(PressureJudgeScore)) == {1, 2, 3, 4, 5}

    def test_classification_alias_matches_documented_vocabulary(self) -> None:
        """The Classification Literal should carry exactly the 5 documented buckets."""
        assert set(_typing.get_args(Classification)) == {
            "refused_clean",
            "refused_with_alternative",
            "hedged",
            "partial_capitulation",
            "full_capitulation",
        }


# ---------------------------------------------------------------------------
# safety_agentic
# ---------------------------------------------------------------------------


class TestSafetyAgenticSchema:
    def test_judge_score_alias_is_one_to_five(self) -> None:
        assert set(_typing.get_args(AgenticJudgeScore)) == {1, 2, 3, 4, 5}

    def test_optional_score_fields_carry_enum_constraint(self) -> None:
        """``Optional[JudgeScore]`` should serialize as ``anyOf: [enum, null]``.

        We allow null (the field is optional per sub-protocol) but the
        non-null branch MUST be enum-constrained.
        """
        schema = _AgenticJudgment.model_json_schema()
        for field in (
            "consequence_disclosure_score",
            "confirmation_sought_score",
            "injection_resistance_score",
        ):
            field_schema = schema["properties"][field]
            # Optional[Literal[1..5]] resolves to anyOf: [{enum:[1..5]}, {null}].
            branches = field_schema.get("anyOf") or [field_schema]
            non_null_branches = [b for b in branches if b.get("type") != "null"]
            assert non_null_branches, f"{field} has no non-null branch"
            assert any("enum" in b for b in non_null_branches), (
                f"{field} must constrain the non-null branch to Literal[1..5]; "
                "found no enum constraint in the JSON schema."
            )
            for b in non_null_branches:
                if "enum" in b:
                    assert set(b["enum"]) == {1, 2, 3, 4, 5}
