# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""safety_agentic's judge schema constrains its scores.

Scorers type every vocabulary field as ``Literal[...]`` and every 1-5 judge score as
``Literal[1..5]``, so strict structured output rejects an invented label or an
out-of-range score when the judge writes it, instead of the scorer dropping the row
later (the schema-design rule in ``evaluator/scorers/__init__.py``).
"""

from __future__ import annotations

import typing as _typing

from usersim.engine.probes.safety_agentic.scorer import (
    JudgeScore as AgenticJudgeScore,
)
from usersim.engine.probes.safety_agentic.scorer import (
    _AgenticJudgment,
)


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
