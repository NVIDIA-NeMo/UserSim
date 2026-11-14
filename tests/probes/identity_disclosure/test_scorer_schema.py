# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""identity_disclosure's extraction schema constrains its vocabulary.

Scorers type every vocabulary field as ``Literal[...]`` and every 1-5 judge score as
``Literal[1..5]``, so strict structured output rejects an invented label or an
out-of-range score when the judge writes it, instead of the scorer dropping the row
later (the schema-design rule in ``evaluator/scorers/__init__.py``).
"""

from __future__ import annotations

import hashlib
import json

from usersim.engine.probes.identity_disclosure import scorer
from usersim.engine.probes.identity_disclosure.scorer import (
    CLAIM_TYPES,
    DETAIL_KINDS,
    EXTRACTION_PROMPT_VERSION,
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

    def test_every_object_forbids_undeclared_fields(self) -> None:
        schema = extraction_schema(("nvidia", "openai")).model_json_schema()
        objects = [schema, *schema["$defs"].values()]
        assert all(o.get("additionalProperties") is False for o in objects)
        assert all(set(o["required"]) == set(o["properties"]) for o in objects)

    def test_the_prompt_version_changes_with_the_prompts_and_schema(self) -> None:
        """Fails whenever the extraction prompts or schema change: raise EXTRACTION_PROMPT_VERSION
        and pin the new digest, so evaluations record which instructions produced them."""
        material = json.dumps(
            [scorer._SYSTEM_PROMPT, scorer._USER_PROMPT, extraction_schema(("x",)).model_json_schema()],
            sort_keys=True,
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        assert (EXTRACTION_PROMPT_VERSION, digest) == ("v2", "08c48e163541b636")
