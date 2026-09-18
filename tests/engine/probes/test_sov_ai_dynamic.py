# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``sov_ai_dynamic`` probe.

Three layers (parallel to ``test_sov_ai_facts.py``):

1. **Task-derivation primitives** (``task_derivation.py``) — pure
   functions that pick a category + sub-topic hint deterministically
   from a probing taxonomy.
2. **Taxonomy caching** — process-local cache + env-override.
3. **End-to-end probe** — ``simulate_sov_ai_dynamic``
   with a mocked ``call_llm`` so no real network calls happen.

No live LLM. No real Nemotron-Personas dataset. Everything here runs
in under a second against fixtures or against the shipped placeholder
taxonomies.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import usersim.engine.generator  # noqa: F401 — triggers probe registration
from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.core.outcomes import (
    FailureClass,
    Provenance,
    WarningKind,
)
from usersim.engine.core.probes import _PROBE_REGISTRY, resolve_probe
from usersim.engine.core.probing_taxonomy import (
    Category,
    ProbingTaxonomy,
    load_probing_taxonomy,
)
from usersim.engine.probes.sov_ai_dynamic import (
    generator as probe_gen,
)
from usersim.engine.probes.sov_ai_dynamic import (
    prompts,
    task_derivation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pt_br_persona_southeast_teacher() -> dict[str, Any]:
    return {
        "first_name": "Mariana",
        "last_name": "Silva",
        "age": 37,
        "state": "SP",
        "state_abbrev": "SP",
        "city": "São Paulo",
        "education_level": "Ensino superior (graduação)",
        "occupation": "Professora de história em escola pública",
    }


@pytest.fixture
def pt_br_persona_northeast_retiree() -> dict[str, Any]:
    return {
        "first_name": "José",
        "last_name": "Santos",
        "age": 72,
        "state": "PE",
        "city": "Recife",
        "education_level": "Ensino fundamental incompleto",
        "occupation": "Aposentado",
    }


@pytest.fixture
def pt_br_taxonomy() -> ProbingTaxonomy:
    return load_probing_taxonomy((packaged_assets_dir() / "sov_ai_dynamic/pt_BR/categories.yaml"))


@pytest.fixture(autouse=True)
def _reset_taxonomy_cache_between_tests():
    """Keep the module-level taxonomy cache from leaking across tests."""
    probe_gen._reset_taxonomy_cache()
    yield
    probe_gen._reset_taxonomy_cache()


@pytest.fixture
def simulator_cfg():
    """Minimal cfg stub that ``simulate_sov_ai_dynamic`` reads from."""

    class _Cfg:
        random_seed = 42
        max_turns = 1

    return _Cfg()


# ---------------------------------------------------------------------------
# persona_to_interest_tags
# ---------------------------------------------------------------------------


class TestPersonaToInterestTags:
    """Population-probing reuses the same persona→tag vocabulary as
    sovereign-knowledge so cross-probe joins on persona-feature axes
    are direct. These tests guard the alias contract."""

    def test_delegates_to_sov_ai_facts_tagger(self, pt_br_persona_southeast_teacher: dict[str, Any]) -> None:
        from usersim.engine.probes.sov_ai_facts.task_derivation import (
            persona_to_tags as upstream,
        )

        ours = task_derivation.persona_to_interest_tags(pt_br_persona_southeast_teacher, "pt_BR")
        theirs = upstream(pt_br_persona_southeast_teacher, "pt_BR")
        assert ours == theirs

    def test_baseline_fallbacks_present(self) -> None:
        tags = task_derivation.persona_to_interest_tags({"age": 30}, "xx_YY")
        assert "region:any" in tags
        assert "age:any" in tags


# ---------------------------------------------------------------------------
# derive_probe
# ---------------------------------------------------------------------------


class TestDeriveProbe:
    def test_returns_a_category_and_a_hint(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        pt_br_taxonomy: ProbingTaxonomy,
    ) -> None:
        probe = task_derivation.derive_probe(
            pt_br_persona_southeast_teacher,
            pt_br_taxonomy,
            seed=42,
        )
        assert probe is not None
        category, hint = probe
        assert isinstance(category, Category)
        assert category in pt_br_taxonomy.categories
        assert isinstance(hint, str)
        assert hint in category.subtopic_hints

    def test_deterministic_on_persona_and_seed(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        pt_br_taxonomy: ProbingTaxonomy,
    ) -> None:
        a = task_derivation.derive_probe(
            pt_br_persona_southeast_teacher,
            pt_br_taxonomy,
            seed=42,
        )
        b = task_derivation.derive_probe(
            pt_br_persona_southeast_teacher,
            pt_br_taxonomy,
            seed=42,
        )
        assert a is not None and b is not None
        assert a[0].id == b[0].id
        assert a[1] == b[1]

    def test_different_seed_can_yield_different_category(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        pt_br_taxonomy: ProbingTaxonomy,
    ) -> None:
        # Over many seeds, expect at least one pair that disagrees on
        # category or hint. With 5 categories and 7-8 hints each, the
        # fixed-seed pool is small enough that 16 trials should land in
        # at least 2 distinct buckets with overwhelming probability.
        keys = {
            (
                task_derivation.derive_probe(
                    pt_br_persona_southeast_teacher,
                    pt_br_taxonomy,
                    seed=s,
                )[0].id
            )
            for s in range(16)
        }
        assert len(keys) > 1

    def test_different_personas_can_yield_different_categories(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        pt_br_persona_northeast_retiree: dict[str, Any],
        pt_br_taxonomy: ProbingTaxonomy,
    ) -> None:
        a_cat = task_derivation.derive_probe(
            pt_br_persona_southeast_teacher,
            pt_br_taxonomy,
            seed=42,
        )[0].id
        b_cat = task_derivation.derive_probe(
            pt_br_persona_northeast_retiree,
            pt_br_taxonomy,
            seed=42,
        )[0].id
        # If they happen to land on the same category at seed=42, an
        # alternate seed must separate them — otherwise the persona
        # part of the seed has no effect.
        if a_cat == b_cat:
            alt = task_derivation.derive_probe(
                pt_br_persona_northeast_retiree,
                pt_br_taxonomy,
                seed=43,
            )[0].id
            assert (
                alt != a_cat
                or task_derivation.derive_probe(
                    pt_br_persona_southeast_teacher,
                    pt_br_taxonomy,
                    seed=43,
                )[0].id
                != alt
            ), "persona content has no influence on category pick — salt is overwhelming the persona hash"

    def test_excluded_categories_skip_the_pool(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        pt_br_taxonomy: ProbingTaxonomy,
    ) -> None:
        first = task_derivation.derive_probe(
            pt_br_persona_southeast_teacher,
            pt_br_taxonomy,
            seed=7,
        )
        assert first is not None
        second = task_derivation.derive_probe(
            pt_br_persona_southeast_teacher,
            pt_br_taxonomy,
            seed=7,
            excluded_category_ids={first[0].id},
        )
        assert second is not None and second[0].id != first[0].id

    def test_all_categories_excluded_returns_none(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        pt_br_taxonomy: ProbingTaxonomy,
    ) -> None:
        all_ids = {c.id for c in pt_br_taxonomy.categories}
        result = task_derivation.derive_probe(
            pt_br_persona_southeast_teacher,
            pt_br_taxonomy,
            seed=1,
            excluded_category_ids=all_ids,
        )
        assert result is None

    def test_hint_rotation_index_shifts_hint(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        pt_br_taxonomy: ProbingTaxonomy,
    ) -> None:
        # Same (persona, seed) → same category. Different
        # hint_rotation_index → different hint within that category.
        base = task_derivation.derive_probe(
            pt_br_persona_southeast_teacher,
            pt_br_taxonomy,
            seed=42,
            hint_rotation_index=0,
        )
        rotated = task_derivation.derive_probe(
            pt_br_persona_southeast_teacher,
            pt_br_taxonomy,
            seed=42,
            hint_rotation_index=1,
        )
        assert base is not None and rotated is not None
        assert base[0].id == rotated[0].id, "category should not change with hint rotation"
        # If the category has at least 2 hints (the loader enforces
        # MIN_SUBTOPIC_HINTS=2), rotation by 1 must shift the hint.
        if len(base[0].subtopic_hints) >= 2:
            assert base[1] != rotated[1], "hint did not rotate"

    def test_taxonomy_version_salt_changes_pick(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        pt_br_persona_northeast_retiree: dict[str, Any],
        pt_br_taxonomy: ProbingTaxonomy,
    ) -> None:
        """Bumping taxonomy_version should change at least some picks."""
        # Build a sister taxonomy with a different version but same
        # categories. Even though we can't easily mutate a frozen dataclass
        # we can construct one fresh.
        bumped = ProbingTaxonomy(
            schema_version=pt_br_taxonomy.schema_version,
            locale=pt_br_taxonomy.locale,
            taxonomy_id=pt_br_taxonomy.taxonomy_id,
            taxonomy_version="v9.9.9",  # different
            placeholder=pt_br_taxonomy.placeholder,
            categories=pt_br_taxonomy.categories,
        )
        # Sweep across personas + seeds to surface any difference.
        diffs = 0
        for persona in (
            pt_br_persona_southeast_teacher,
            pt_br_persona_northeast_retiree,
        ):
            for seed in range(8):
                a = task_derivation.derive_probe(persona, pt_br_taxonomy, seed=seed)[0].id
                b = task_derivation.derive_probe(persona, bumped, seed=seed)[0].id
                if a != b:
                    diffs += 1
        assert diffs > 0, "taxonomy_version salt had zero effect on picks"


# ---------------------------------------------------------------------------
# Taxonomy caching
# ---------------------------------------------------------------------------


class TestTaxonomyCache:
    def test_load_once_cached(self, tmp_path: Path) -> None:
        path = _write_tiny_pt_br_taxonomy(tmp_path)
        with patch.dict(
            os.environ,
            {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(path)},
        ):
            t1 = probe_gen._load_taxonomy_for_locale("pt_BR")
            t2 = probe_gen._load_taxonomy_for_locale("pt_BR")
        assert t1 is t2

    def test_env_override_picks_up_different_taxonomy(self, tmp_path: Path) -> None:
        path = _write_tiny_pt_br_taxonomy(tmp_path, taxonomy_id="override_taxonomy")
        with patch.dict(
            os.environ,
            {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(path)},
        ):
            taxonomy = probe_gen._load_taxonomy_for_locale("pt_BR")
        assert taxonomy.taxonomy_id == "override_taxonomy"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    @pytest.mark.parametrize(
        "locale",
        SHIPPED_LOCALES,
    )
    def test_opening_prompt_available_for_all_shipped_locales(
        self,
        locale: str,
    ) -> None:
        text = prompts.get_opening_prompt(locale)
        # Template carries the three placeholders the probe fills.
        assert "{persona}" in text
        assert "{invitation}" in text
        assert "{subtopic_hint}" in text

    @pytest.mark.parametrize(
        "locale",
        SHIPPED_LOCALES,
    )
    def test_followup_instruction_available_for_all_shipped_locales(self, locale: str) -> None:
        assert prompts.get_followup_instruction(locale)

    def test_unknown_locale_raises_for_opening(self) -> None:
        with pytest.raises(prompts.PromptsUnavailableError):
            prompts.get_opening_prompt("xx_YY")

    def test_unknown_locale_raises_for_followup(self) -> None:
        with pytest.raises(prompts.PromptsUnavailableError):
            prompts.get_followup_instruction("xx_YY")

    def test_available_locales_lists_all_shipped(self) -> None:
        assert set(prompts.available_locales()) == set(SHIPPED_LOCALES)


# ---------------------------------------------------------------------------
# End-to-end simulate_sov_ai_dynamic (mocked LLM)
# ---------------------------------------------------------------------------


_PT_BR_TAXONOMY_PATH = packaged_assets_dir() / "sov_ai_dynamic/pt_BR/categories.yaml"


class TestSimulateSovAiDynamic:
    def test_registered_in_probe_registry(self) -> None:
        # Cross-probe invariants live in tests/probes/test_probe_invariants.py;
        # this is a fast smoke check that the SovAiDynamicProbe class
        # itself is the registered target.
        assert "sov_ai_dynamic" in _PROBE_REGISTRY
        assert resolve_probe("sov_ai_dynamic").__name__ == "SovAiDynamicProbe"

    def test_single_turn_runs_end_to_end(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        opening_msg = "Aproveitando, queria saber sobre o cerrado..."
        assistant_reply = "Boa pergunta! O cerrado é um bioma..."

        with (
            patch.dict(
                os.environ,
                {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(_PT_BR_TAXONOMY_PATH)},
            ),
            _patched_call_llm(
                _mock_call_llm(
                    assistant_responses=[
                        {"role": "assistant", "content": assistant_reply},
                    ],
                    user_responses=[
                        {"role": "assistant", "content": opening_msg},
                    ],
                )
            ),
        ):
            result = probe_gen.simulate_sov_ai_dynamic(
                models={
                    "user_model": object(),
                    "assistant_model": object(),
                    "judge_model": object(),
                    "summary_model": object(),
                    "api_response_model": object(),
                },
                data={},
                persona=pt_br_persona_southeast_teacher,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )

        assert result["conversation_status"] is True
        assert result["num_turns"] == 1
        assert result["num_tool_calls"] == 0
        # probe_variant carries the picked category id.
        assert result["probe_variant"].startswith("brazil-")
        assert len(result["probing_categories_explored"]) == 1
        assert len(result["probing_subtopic_hints_used"]) == 1

        # Conversation roles + content.
        messages = json.loads(result["conversation_messages"])
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == opening_msg
        assert messages[1]["content"] == assistant_reply

        # Provenance + warnings.
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] in {"ok", "completed_with_warnings"}
        # The shipped pt_BR taxonomy has been promoted to beta, so no
        # placeholder warning should be attached.
        assert not any(w["kind"] == WarningKind.USED_PLACEHOLDER_TAXONOMY.value for w in outcome["warnings"]), (
            f"unexpected USED_PLACEHOLDER_TAXONOMY warning: {outcome['warnings']}"
        )
        # taxonomy_version must be pinned on provenance.
        assert outcome["provenance"]["bank_version"].get("pt_BR") == "v0.5.1"

    def test_two_turn_mode_issues_followup(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        simulator_cfg.max_turns = 2
        opening_msg = "Olá, gostaria de entender melhor sobre..."
        assistant_reply_1 = "Resposta inicial do assistente."
        user_followup = "Hmm, e quanto a... ?"
        assistant_reply_2 = "Resposta refinada."

        with (
            patch.dict(
                os.environ,
                {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(_PT_BR_TAXONOMY_PATH)},
            ),
            _patched_call_llm(
                _mock_call_llm(
                    assistant_responses=[
                        {"role": "assistant", "content": assistant_reply_1},
                        {"role": "assistant", "content": assistant_reply_2},
                    ],
                    user_responses=[
                        {"role": "assistant", "content": opening_msg},
                        {"role": "assistant", "content": user_followup},
                    ],
                )
            ),
        ):
            result = probe_gen.simulate_sov_ai_dynamic(
                models={
                    "user_model": object(),
                    "assistant_model": object(),
                    "judge_model": object(),
                    "summary_model": object(),
                    "api_response_model": object(),
                },
                data={},
                persona=pt_br_persona_southeast_teacher,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )

        messages = json.loads(result["conversation_messages"])
        assert [m["role"] for m in messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert messages[0]["content"] == opening_msg
        assert messages[1]["content"] == assistant_reply_1
        assert messages[2]["content"] == user_followup
        assert messages[3]["content"] == assistant_reply_2
        assert result["num_turns"] == 2

    def test_empty_opening_fails_cleanly(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        """If the user-agent returns whitespace, the trajectory must
        fail with USER_QUERY_GATE_EXHAUSTED rather than producing a
        broken record with no user turn."""
        # ConversationLoop retries the user-LLM up to max_query_attempts
        # (default 3). Each whitespace-only response now triggers the
        # empty-content rejection branch and bumps role-violations.
        # After 3 retries, the loop produces a structured
        # USER_QUERY_GATE_EXHAUSTED outcome with the probe's
        # side-channel keys preserved (build_result_extras runs even
        # on the failure path).
        with (
            patch.dict(
                os.environ,
                {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(_PT_BR_TAXONOMY_PATH)},
            ),
            _patched_call_llm(
                _mock_call_llm(
                    assistant_responses=[],
                    user_responses=[
                        {"role": "assistant", "content": "   \n  "},
                        {"role": "assistant", "content": "   \n  "},
                        {"role": "assistant", "content": "   \n  "},
                    ],
                )
            ),
        ):
            result = probe_gen.simulate_sov_ai_dynamic(
                models={
                    "user_model": object(),
                    "assistant_model": object(),
                    "judge_model": object(),
                    "summary_model": object(),
                    "api_response_model": object(),
                },
                data={},
                persona=pt_br_persona_southeast_teacher,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )

        assert result["conversation_status"] is False
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == "failed"
        assert outcome["failure_class"] == FailureClass.USER_QUERY_GATE_EXHAUSTED.value
        # Even on failure, the picked-category side-channel must be set.
        assert result["probing_categories_explored"]
        assert result["probing_subtopic_hints_used"]
        assert result["probe_variant"].startswith("brazil-")

    def test_user_model_exception_fails_cleanly(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        # ConversationLoop's user-LLM call is wrapped in try/except;
        # the exception triggers a role-violation + retry loop. After
        # max_query_attempts retries → USER_QUERY_GATE_EXHAUSTED.
        with (
            patch.dict(
                os.environ,
                {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(_PT_BR_TAXONOMY_PATH)},
            ),
            _patched_call_llm(RuntimeError("rate limited")),
        ):
            result = probe_gen.simulate_sov_ai_dynamic(
                models={
                    "user_model": object(),
                    "assistant_model": object(),
                    "judge_model": object(),
                    "summary_model": object(),
                    "api_response_model": object(),
                },
                data={},
                persona=pt_br_persona_southeast_teacher,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert result["conversation_status"] is False
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == "failed"
        assert outcome["failure_class"] == FailureClass.USER_QUERY_GATE_EXHAUSTED.value

    def test_assistant_failure_after_opening_fails_cleanly(
        self,
        pt_br_persona_southeast_teacher: dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        """User-agent succeeds on turn 1; assistant_model raises. The
        record must carry FAILURE attribution = ASSISTANT_MODEL and
        the opening user turn must still be in the messages."""
        opening_msg = "Posso fazer uma pergunta sobre..."

        # Alias-dispatching side-effect: user_model + judge_model
        # succeed for the gate, then assistant_model raises. Matches
        # the unified loop's call sequence for turn-1.
        judge_payload = "<explanation>looks fine</explanation>\n<rating>success</rating>"

        def _side_effect(models, alias, msgs, **kwargs):
            if alias == "user_model":
                return {"role": "assistant", "content": opening_msg}
            if alias == "judge_model":
                return {"role": "assistant", "content": judge_payload}
            if alias == "assistant_model":
                raise RuntimeError("assistant down")
            if alias == "summary_model":
                return {"role": "assistant", "content": "no"}
            raise AssertionError(f"unexpected alias: {alias!r}")

        with (
            patch.dict(
                os.environ,
                {"USERSIM_SOV_AI_DYNAMIC_TAXONOMY_PT_BR": str(_PT_BR_TAXONOMY_PATH)},
            ),
            _patched_call_llm(_side_effect),
        ):
            result = probe_gen.simulate_sov_ai_dynamic(
                models={
                    "user_model": object(),
                    "assistant_model": object(),
                    "judge_model": object(),
                    "summary_model": object(),
                    "api_response_model": object(),
                },
                data={},
                persona=pt_br_persona_southeast_teacher,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert result["conversation_status"] is False
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == "failed"
        assert outcome["failure_class"] == FailureClass.INFRASTRUCTURE_ERROR.value
        # Opening user turn was captured before the assistant call failed.
        messages = json.loads(result["conversation_messages"])
        assert messages and messages[0]["role"] == "user"
        assert messages[0]["content"] == opening_msg


# ---------------------------------------------------------------------------
# should_succeed guardrail
# ---------------------------------------------------------------------------


class TestShouldSucceed:
    def test_returns_true_when_categories_explored(self) -> None:
        from usersim.engine.core.simulation import ConversationState

        state = ConversationState()
        state.metadata["probing_categories_explored"] = ["brazil-history"]
        assert probe_gen.should_succeed(state) is True

    def test_returns_false_when_no_categories_explored(self) -> None:
        from usersim.engine.core.simulation import ConversationState

        state = ConversationState()
        assert probe_gen.should_succeed(state) is False

    def test_returns_false_on_empty_list(self) -> None:
        from usersim.engine.core.simulation import ConversationState

        state = ConversationState()
        state.metadata["probing_categories_explored"] = []
        assert probe_gen.should_succeed(state) is False


# ---------------------------------------------------------------------------
# WarningKind.USED_PLACEHOLDER_TAXONOMY enum value
# ---------------------------------------------------------------------------


class TestPlaceholderTaxonomyWarning:
    def test_enum_value_stable(self) -> None:
        assert WarningKind.USED_PLACEHOLDER_TAXONOMY.value == "used_placeholder_taxonomy"

    def test_distinct_from_used_placeholder_fact(self) -> None:
        assert WarningKind.USED_PLACEHOLDER_TAXONOMY is not WarningKind.USED_PLACEHOLDER_FACT
        assert WarningKind.USED_PLACEHOLDER_TAXONOMY.value != WarningKind.USED_PLACEHOLDER_FACT.value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_call_llm(
    assistant_responses: list[dict[str, Any]],
    *,
    user_responses: list[dict[str, Any]] | None = None,
    judge_pass: bool = True,
):
    """Alias-dispatching ``call_llm`` mock for the unified loop.

    See test_sov_ai_facts._mock_call_llm for the full rationale.
    sov_ai_dynamic generates turn-1 via the user-LLM (no verbatim
    bank), so user_responses includes BOTH the opening message and
    any follow-ups.
    """
    iter_assist = iter(assistant_responses)
    iter_user = iter(user_responses or [])
    rating = "success" if judge_pass else "failure"
    judge_payload = f"<explanation>looks fine</explanation>\n<rating>{rating}</rating>"

    def _side_effect(models, alias, msgs, **kwargs):
        if alias == "assistant_model":
            try:
                return next(iter_assist)
            except StopIteration as e:
                raise AssertionError("test consumed more assistant_model calls than expected") from e
        if alias == "judge_model":
            return {"role": "assistant", "content": judge_payload}
        if alias == "user_model":
            try:
                return next(iter_user)
            except StopIteration as e:
                raise AssertionError("test consumed more user_model calls than expected") from e
        if alias == "summary_model":
            return {"role": "assistant", "content": "no"}
        raise AssertionError(f"unexpected alias: {alias!r}")

    return _side_effect


@contextmanager
def _patched_call_llm(side_effect):
    """Patch ``call_llm`` everywhere the unified loop binds it.

    Patches both ``core.simulation.call_llm`` (assistant + turn-1
    generation) AND ``core.judges.call_llm`` (in-sim judge) since
    each consumer module re-imports ``call_llm`` at module load.
    """
    with (
        patch(
            "usersim.engine.core.simulation.call_llm",
            side_effect=side_effect,
        ),
        patch(
            "usersim.engine.core.judges.call_llm",
            side_effect=side_effect,
        ),
    ):
        yield


def _write_tiny_pt_br_taxonomy(
    tmp_path: Path,
    *,
    taxonomy_id: str = "tiny_pt_br_taxonomy",
) -> Path:
    """Drop a minimal valid pt_BR probing taxonomy on disk for cache tests."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "categories.yaml"
    path.write_text(
        f'schema_version: "v0.1"\n'
        f"locale: pt_BR\n"
        f"taxonomy_id: {taxonomy_id}\n"
        f"taxonomy_version: v0.0.1\n"
        f"placeholder: true\n"
        f"categories:\n"
        f"  - id: brazil-tiny\n"
        f'    invitation: "Vamos falar sobre coisas brasileiras."\n'
        f"    subtopic_hints:\n"
        f'      - "uma coisa qualquer"\n'
        f'      - "outra coisa qualquer"\n',
        encoding="utf-8",
    )
    return path
