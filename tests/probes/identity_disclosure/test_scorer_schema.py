# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""identity_disclosure's extraction schema constrains its vocabulary.

Scorers type every vocabulary field as ``Literal[...]`` and every 1-5 judge score as
``Literal[1..5]``, so strict structured output rejects an invented label or an
out-of-range score when the judge writes it, instead of the scorer dropping the row
later (the schema-design rule in ``evaluator/scorers/__init__.py``).
"""

from __future__ import annotations

from usersim.engine.probes.identity_disclosure.scorer import (
    CLAIM_TYPES,
    DETAIL_KINDS,
    PREMISE_STANCES,
    extraction_schema,
)


class TestIdentityDisclosureSchema:
    @staticmethod
    def _defs() -> dict:
        return extraction_schema(("nvidia", "openai")).model_json_schema()["$defs"]

    def test_every_vocabulary_field_is_an_enum(self) -> None:
        defs = self._defs()
        claim = defs["IdentityClaimExtraction"]["properties"]
        assert set(claim["claim_type"]["enum"]) == set(CLAIM_TYPES)
        assert set(claim["developer_guess"]["enum"]) == {"nvidia", "openai", "unknown"}
        assert set(defs["TurnExtraction"]["properties"]["premise_stance"]["enum"]) == set(PREMISE_STANCES)
        assert set(defs["SelfDetailExtraction"]["properties"]["kind"]["enum"]) == set(DETAIL_KINDS)

    def test_the_names_as_written_are_explicitly_nullable(self) -> None:
        claim = self._defs()["IdentityClaimExtraction"]
        for field in ("developer_text", "model_text"):
            assert {b.get("type") for b in claim["properties"][field]["anyOf"]} == {"string", "null"}
            assert field in claim["required"], f"{field} must be required-but-nullable, not implicitly optional"
