# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``sov_ai_facts`` probe.

Three layers:

1. **Task-derivation primitives** (``task_derivation.py``) — pure
   functions that map persona attributes to the bank's tag vocabulary
   and pick a fact deterministically.
2. **Fact-bank caching** — process-local cache + env-override.
3. **End-to-end probe** — ``simulate_sov_ai_facts``
   with a mocked ``call_llm`` so no real network calls happen.

No live LLM. No real Nemotron-Personas dataset. Everything here runs
in under a second against fixtures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from usersim.engine.core.fact_bank import (
    Fact,
    FactBank,
    FactProvenance,
    load_fact_bank,
)
from usersim.engine.core.locale import SHIPPED_LOCALES
from usersim.engine.core.outcomes import (
    FailureClass,
    Provenance,
)
import usersim.engine.generator  # noqa: F401 — triggers probe registration
from usersim.engine.core.probes import _PROBE_REGISTRY, resolve_probe
from usersim.engine.probes.sov_ai_facts import (
    generator as probe_gen,
    prompts,
    task_derivation,
)
from usersim.engine.core._assets import packaged_assets_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pt_br_persona_southeast_teacher() -> Dict[str, Any]:
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
def pt_br_persona_northeast_retiree() -> Dict[str, Any]:
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
def pt_br_sample_bank() -> FactBank:
    return load_fact_bank((packaged_assets_dir() / "sov_ai_facts/pt_BR/sample.yaml"))


@pytest.fixture(autouse=True)
def _reset_bank_cache_between_tests():
    """Keep the module-level bank cache from leaking across tests."""
    probe_gen._reset_bank_cache()
    yield
    probe_gen._reset_bank_cache()


@pytest.fixture
def simulator_cfg():
    """Minimal cfg stub that ``simulate_sov_ai_facts`` reads from."""
    class _Cfg:
        random_seed = 42
        max_turns = 1
    return _Cfg()


# ---------------------------------------------------------------------------
# Persona → tag mapping
# ---------------------------------------------------------------------------


class TestPersonaToTags:
    def test_brazilian_southeast_teacher_picks_right_tags(
        self, pt_br_persona_southeast_teacher: Dict[str, Any]
    ) -> None:
        tags = task_derivation.persona_to_tags(
            pt_br_persona_southeast_teacher, "pt_BR"
        )
        assert "region:southeast" in tags
        assert "region:any" in tags
        assert "age:25-44" in tags
        assert "age:any" in tags
        assert "education:tertiary" in tags
        assert "occupation-family:teacher" in tags
        assert "interest:literature" in tags or "interest:history" in tags

    def test_brazilian_northeast_retiree(
        self, pt_br_persona_northeast_retiree: Dict[str, Any]
    ) -> None:
        tags = task_derivation.persona_to_tags(
            pt_br_persona_northeast_retiree, "pt_BR"
        )
        assert "region:northeast" in tags
        assert "age:65+" in tags
        assert "occupation-family:retired" in tags
        assert "education:primary" in tags

    @pytest.mark.parametrize(
        "state_code, expected_region",
        [
            ("SP", "region:southeast"),
            ("RJ", "region:southeast"),
            ("BA", "region:northeast"),
            ("PE", "region:northeast"),
            ("RS", "region:south"),
            ("PR", "region:south"),
            ("AM", "region:north"),
            ("GO", "region:central-west"),
            ("DF", "region:central-west"),
        ],
    )
    def test_all_br_macro_regions(self, state_code: str, expected_region: str) -> None:
        persona = {"state_abbrev": state_code, "age": 30, "occupation": "Engineer"}
        tags = task_derivation.persona_to_tags(persona, "pt_BR")
        assert expected_region in tags

    def test_unknown_locale_still_returns_fallback_tags(self) -> None:
        persona = {"age": 40, "occupation": "Teacher"}
        tags = task_derivation.persona_to_tags(persona, "xx_YY")
        # Never empty: baseline fallbacks ensure there's always something
        # for the matcher to intersect.
        assert "region:any" in tags
        assert "age:any" in tags
        # Age bin still works (locale-agnostic).
        assert "age:25-44" in tags

    def test_missing_age_is_handled_gracefully(self) -> None:
        tags = task_derivation.persona_to_tags(
            {"state_abbrev": "SP", "occupation": "Teacher"}, "pt_BR"
        )
        # No age-bin tag (other than age:any fallback) — no crash.
        assert not any(t.startswith("age:") and t != "age:any" for t in tags)
        assert "age:any" in tags

    def test_education_substring_matching(self) -> None:
        for raw, expected in [
            ("Ensino fundamental completo", "education:primary"),
            ("High school diploma", "education:secondary"),
            ("Master's degree", "education:tertiary"),
            ("Pós-graduação lato sensu", "education:tertiary"),
        ]:
            tags = task_derivation.persona_to_tags(
                {"education_level": raw, "age": 30}, "pt_BR"
            )
            assert expected in tags, f"{raw!r} → expected {expected!r}, got {tags}"

    def test_occupation_substring_matching_multilingual(self) -> None:
        # English + Portuguese variants of the same concept should
        # produce the same occupation-family tag.
        for raw in ("Doctor of medicine", "Médico cardiologista"):
            tags = task_derivation.persona_to_tags(
                {"occupation": raw, "age": 40}, "pt_BR"
            )
            assert "occupation-family:healthcare" in tags
            assert "interest:health" in tags


# ---------------------------------------------------------------------------
# derive_task
# ---------------------------------------------------------------------------


class TestDeriveTask:
    def test_deterministic_on_persona_and_seed(
        self,
        pt_br_persona_southeast_teacher: Dict[str, Any],
        pt_br_sample_bank: FactBank,
    ) -> None:
        a = task_derivation.derive_task(
            pt_br_persona_southeast_teacher, pt_br_sample_bank, seed=42
        )
        b = task_derivation.derive_task(
            pt_br_persona_southeast_teacher, pt_br_sample_bank, seed=42
        )
        assert a is not None and b is not None
        assert a.id == b.id

    def test_different_seed_can_yield_different_fact(
        self,
        pt_br_persona_southeast_teacher: Dict[str, Any],
        pt_br_sample_bank: FactBank,
    ) -> None:
        # Over many seeds, we expect at least one pair that disagree.
        ids = {
            task_derivation.derive_task(
                pt_br_persona_southeast_teacher, pt_br_sample_bank, seed=s
            ).id
            for s in range(10)
        }
        assert len(ids) > 1

    def test_different_personas_can_yield_different_facts(
        self,
        pt_br_persona_southeast_teacher: Dict[str, Any],
        pt_br_persona_northeast_retiree: Dict[str, Any],
        pt_br_sample_bank: FactBank,
    ) -> None:
        # Same seed but different persona content → high probability of
        # a different choice. (Not guaranteed for every bank, but easy
        # to satisfy with the 25-entry sample bank.)
        a = task_derivation.derive_task(
            pt_br_persona_southeast_teacher, pt_br_sample_bank, seed=42
        )
        b = task_derivation.derive_task(
            pt_br_persona_northeast_retiree, pt_br_sample_bank, seed=42
        )
        # Either they differ, or (unlikely) both match the same entry —
        # in the latter case at least the matching pool differs enough
        # that bumping the seed separates them.
        if a.id == b.id:
            b_alt = task_derivation.derive_task(
                pt_br_persona_northeast_retiree, pt_br_sample_bank, seed=43
            )
            assert b_alt.id != a.id, (
                "persona-dependent variance is weaker than expected for the "
                "shipped sample bank"
            )

    def test_excluded_ids_skip_the_pool(
        self,
        pt_br_persona_southeast_teacher: Dict[str, Any],
        pt_br_sample_bank: FactBank,
    ) -> None:
        first = task_derivation.derive_task(
            pt_br_persona_southeast_teacher, pt_br_sample_bank, seed=7
        )
        assert first is not None
        second = task_derivation.derive_task(
            pt_br_persona_southeast_teacher, pt_br_sample_bank, seed=7,
            excluded_fact_ids={first.id},
        )
        assert second is not None and second.id != first.id

    def test_no_matching_fact_returns_none(self) -> None:
        # Build a bank whose every fact requires a persona tag our
        # input persona won't produce.
        empty_bank = _build_synthetic_bank(
            facts=[
                _fact("ZZ-001", persona_tags=("occupation-family:astronaut",)),
            ],
        )
        persona = {"age": 30, "occupation": "Shopkeeper", "state_abbrev": "SP"}
        result = task_derivation.derive_task(persona, empty_bank)
        assert result is None

    def test_seed_salt_includes_bank_version(self) -> None:
        """Bumping the bank's version should change the picked fact for at
        least some personas. A bank update that silently re-picks the
        same fact for every persona would be the bug we want to catch."""
        facts = [
            _fact(f"F-{i}", persona_tags=("interest:history", "region:any"))
            for i in range(8)
        ]
        bank_a = _build_synthetic_bank(facts=facts, bank_version="v0.1.0")
        bank_b = _build_synthetic_bank(facts=facts, bank_version="v0.2.0")

        per_persona_picks: List[tuple] = []
        for persona in _persona_variations():
            a = task_derivation.derive_task(persona, bank_a, seed=5)
            b = task_derivation.derive_task(persona, bank_b, seed=5)
            per_persona_picks.append((persona, a.id, b.id))

        diffs = [p for p in per_persona_picks if p[1] != p[2]]
        assert diffs, (
            "bank-version salt produced identical picks for every persona — "
            "a bank upgrade will silently re-pick the same old fact"
        )
        # Sanity check: the shift is not pathologically large either
        # (we shouldn't be permuting so aggressively that every pick
        # changes — that would mean the persona-content part of the
        # seed has been overwhelmed).
        assert len(diffs) < len(per_persona_picks), (
            "bank-version salt completely overwhelms persona identity — "
            "expected a partial shift, not a total permutation"
        )


# ---------------------------------------------------------------------------
# Fact-bank caching
# ---------------------------------------------------------------------------


class TestFactBankCache:
    def test_load_once_cached(self, tmp_path: Path) -> None:
        bank_path = _write_tiny_pt_br_bank(tmp_path)
        with patch.dict(os.environ, {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(bank_path)}):
            b1 = probe_gen._load_bank_for_locale("pt_BR")
            b2 = probe_gen._load_bank_for_locale("pt_BR")
        assert b1 is b2, "cache returned a different instance on second call"

    def test_env_override_picks_up_different_bank(self, tmp_path: Path) -> None:
        bank_path = _write_tiny_pt_br_bank(tmp_path, bank_id="override_bank")
        with patch.dict(os.environ, {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(bank_path)}):
            bank = probe_gen._load_bank_for_locale("pt_BR")
        assert bank.bank_id == "override_bank"

    def test_reset_drops_cache(self, tmp_path: Path) -> None:
        bank_path_a = _write_tiny_pt_br_bank(tmp_path / "a", bank_id="bank_a")
        bank_path_b = _write_tiny_pt_br_bank(tmp_path / "b", bank_id="bank_b")

        with patch.dict(os.environ, {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(bank_path_a)}):
            assert probe_gen._load_bank_for_locale("pt_BR").bank_id == "bank_a"
        probe_gen._reset_bank_cache()
        with patch.dict(os.environ, {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(bank_path_b)}):
            assert probe_gen._load_bank_for_locale("pt_BR").bank_id == "bank_b"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    # Every locale that ships a fact bank must also ship a matching
    # prompt pack. Running a sovereign-knowledge sim on a locale with no
    # Every shipped fact-bank locale must also have a prompt pack;
    # otherwise simulate-time prompt lookup fails after the bank itself
    # loads cleanly.

    @pytest.mark.parametrize("locale", SHIPPED_LOCALES)
    def test_factual_recall_prompt_available_for_all_shipped_locales(
        self, locale: str,
    ) -> None:
        text = prompts.get_system_prompt(locale, "factual_recall")
        # Required placeholders for the probe's .format call.
        assert "{persona}" in text
        assert "{category}" in text
        assert "{difficulty}" in text
        # Substantive content (not a stub).
        assert len(text) > 200

    @pytest.mark.parametrize("locale", SHIPPED_LOCALES)
    def test_completion_prompt_available_for_all_shipped_locales(
        self, locale: str,
    ) -> None:
        text = prompts.get_system_prompt(locale, "completion")
        assert "{persona}" in text
        assert "{category}" in text
        assert "{difficulty}" in text
        assert len(text) > 200

    @pytest.mark.parametrize("locale", SHIPPED_LOCALES)
    def test_followup_instruction_available_for_all_shipped_locales(
        self, locale: str,
    ) -> None:
        text = prompts.get_followup_instruction(locale)
        assert len(text) > 100

    def test_unknown_locale_raises(self) -> None:
        with pytest.raises(prompts.PromptsUnavailableError):
            prompts.get_system_prompt("xx_YY", "factual_recall")

    def test_unknown_question_type_raises(self) -> None:
        with pytest.raises(prompts.PromptsUnavailableError):
            prompts.get_system_prompt("pt_BR", "not_a_type")

    def test_available_locales_matches_shipped_set(self) -> None:
        # Round-trip: available_locales() must return every shipped locale.
        # SHIPPED_LOCALES lives in core/locale.py, so adding a new locale
        # automatically expands the per-locale prompt checks above.
        assert set(prompts.available_locales()) == set(SHIPPED_LOCALES)


# ---------------------------------------------------------------------------
# End-to-end simulate_sov_ai_facts (mocked LLM)
# ---------------------------------------------------------------------------


class TestSimulateSovAiFacts:
    def test_registered_in_probe_registry(self) -> None:
        # Cross-probe invariants (registry membership, module constants
        # well-formedness) live in tests/probes/test_probe_invariants.py
        # — this per-probe test exists as a fast smoke check that the
        # SovAiFactsProbe class itself is the registered target.
        assert "sov_ai_facts" in _PROBE_REGISTRY
        assert resolve_probe("sov_ai_facts").__name__ == "SovAiFactsProbe"

    def test_single_turn_runs_end_to_end(
        self,
        pt_br_persona_southeast_teacher: Dict[str, Any],
        pt_br_sample_bank: FactBank,
        simulator_cfg: Any,
        tmp_path: Path,
    ) -> None:
        with patch.dict(
            os.environ,
            {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(
                (packaged_assets_dir() / "sov_ai_facts/pt_BR/sample.yaml")
            )},
        ), _patched_call_llm(_mock_call_llm([
                {"role": "assistant", "content": "Essa é uma resposta de teste."},
            ])):
            result = probe_gen.simulate_sov_ai_facts(
                models={"user_model": object(), "assistant_model": object(), "judge_model": object(), "summary_model": object(), "api_response_model": object()},
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
        assert result["probe_variant"].startswith("brazil-")
        assert len(result["sovereign_facts_probed"]) == 1

        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] in {"ok", "completed_with_warnings"}
        # The shipped bank intentionally mixes reviewed entries with
        # honest placeholders for generated rows; either warning state
        # is valid depending on which fact the persona-derived picker chose.
        # Bank version must be pinned on the provenance block.
        assert outcome["provenance"]["bank_version"].get("pt_BR") == "v0.5.2"

    def test_verbatim_question_is_first_user_turn(
        self,
        pt_br_persona_southeast_teacher: Dict[str, Any],
        pt_br_sample_bank: FactBank,
        simulator_cfg: Any,
    ) -> None:
        # Pick the deterministic choice for this (persona, seed) pair so
        # we can assert on the exact question text that should end up in
        # message[0].
        expected_fact = task_derivation.derive_task(
            pt_br_persona_southeast_teacher,
            pt_br_sample_bank,
            seed=simulator_cfg.random_seed,
        )
        assert expected_fact is not None

        with patch.dict(
            os.environ,
            {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(
                (packaged_assets_dir() / "sov_ai_facts/pt_BR/sample.yaml")
            )},
        ), _patched_call_llm(_mock_call_llm([
                {"role": "assistant", "content": "ok"},
            ])):
            result = probe_gen.simulate_sov_ai_facts(
                models={"user_model": object(), "assistant_model": object(), "judge_model": object(), "summary_model": object(), "api_response_model": object()},
                data={},
                persona=pt_br_persona_southeast_teacher,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )

        messages = json.loads(result["conversation_messages"])
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == expected_fact.question, (
            "first user message was not the fact's verbatim question — the "
            "false-premise wording must survive intact"
        )
        assert result["sovereign_facts_probed"] == [expected_fact.id]
        assert result["probe_variant"] == expected_fact.category

    def test_two_turn_mode_issues_followup(
        self,
        pt_br_persona_southeast_teacher: Dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        simulator_cfg.max_turns = 2
        assistant_reply_1 = "Primeira resposta do assistente."
        user_followup = "Hmm, tem certeza? Pode explicar melhor?"
        assistant_reply_2 = "Resposta refinada do assistente."

        with patch.dict(
            os.environ,
            {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(
                (packaged_assets_dir() / "sov_ai_facts/pt_BR/sample.yaml")
            )},
        ), _patched_call_llm(_mock_call_llm(
            assistant_responses=[
                {"role": "assistant", "content": assistant_reply_1},
                {"role": "assistant", "content": assistant_reply_2},
            ],
            user_responses=[
                {"role": "assistant", "content": user_followup},
            ],
        )):
            result = probe_gen.simulate_sov_ai_facts(
                models={"user_model": object(), "assistant_model": object(), "judge_model": object(), "summary_model": object(), "api_response_model": object()},
                data={},
                persona=pt_br_persona_southeast_teacher,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )

        messages = json.loads(result["conversation_messages"])
        # user, assistant, user, assistant (no system msg per pure-capability policy)
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
        assert messages[1]["content"] == assistant_reply_1
        assert messages[2]["content"] == user_followup
        assert messages[3]["content"] == assistant_reply_2
        assert result["num_turns"] == 2

    def test_assistant_failure_produces_failed_outcome(
        self,
        pt_br_persona_southeast_teacher: Dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        with patch.dict(
            os.environ,
            {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(
                (packaged_assets_dir() / "sov_ai_facts/pt_BR/sample.yaml")
            )},
        ), _patched_call_llm(RuntimeError("rate-limited by provider")):
            result = probe_gen.simulate_sov_ai_facts(
                models={"user_model": object(), "assistant_model": object(), "judge_model": object(), "summary_model": object(), "api_response_model": object()},
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
        # facts_probed is still recorded — we DID pick a fact, even
        # though the assistant call failed — so the scorer can count
        # attempted-but-failed probes.
        assert result["sovereign_facts_probed"]

    def test_no_matching_fact_fails_cleanly(
        self, tmp_path: Path, simulator_cfg: Any,
    ) -> None:
        # A narrow bank whose facts require occupation-family:astronaut.
        narrow_bank_path = tmp_path / "narrow.yaml"
        narrow_bank_path.write_text(
            'schema_version: "v0.1"\n'
            "locale: pt_BR\n"
            "bank_id: narrow\n"
            "bank_version: v0.1.0\n"
            "categories: [rare]\n"
            "entries:\n"
            '  - id: N-001\n'
            '    category: rare\n'
            '    question_type: factual_recall\n'
            '    placeholder: true\n'
            '    question: "Uma pergunta rara?"\n'
            '    false_premises: []\n'
            '    ground_truth: "Uma resposta rara."\n'
            '    persona_tags: ["occupation-family:astronaut"]\n'
            '    difficulty: hard\n'
            '    provenance:\n'
            '      source: "placeholder"\n'
            '      last_reviewed: "2026-04-18"\n',
            encoding="utf-8",
        )
        persona = {
            "first_name": "Fernanda",
            "last_name": "Lima",
            "age": 30,
            "state_abbrev": "SP",
            "occupation": "Shopkeeper",
            "education_level": "Ensino médio",
        }

        with patch.dict(
            os.environ,
            {"USERSIM_SOV_AI_FACTS_BANK_PT_BR": str(narrow_bank_path)},
        ):
            result = probe_gen.simulate_sov_ai_facts(
                models={"user_model": object(), "assistant_model": object(), "judge_model": object(), "summary_model": object(), "api_response_model": object()},
                data={},
                persona=persona,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert result["conversation_status"] is False
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == "failed"
        assert outcome["failure_class"] == FailureClass.SCENARIO_ABORTED.value


# ---------------------------------------------------------------------------
# should_succeed guardrail
# ---------------------------------------------------------------------------


class TestShouldSucceed:
    def test_returns_true_when_facts_probed(self) -> None:
        from usersim.engine.core.simulation import ConversationState
        state = ConversationState()
        state.metadata["facts_probed"] = ["BR-GEO-001"]
        assert probe_gen.should_succeed(state) is True

    def test_returns_false_when_no_facts_probed(self) -> None:
        from usersim.engine.core.simulation import ConversationState
        state = ConversationState()
        assert probe_gen.should_succeed(state) is False

    def test_returns_false_on_empty_list(self) -> None:
        from usersim.engine.core.simulation import ConversationState
        state = ConversationState()
        state.metadata["facts_probed"] = []
        assert probe_gen.should_succeed(state) is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_call_llm(
    assistant_responses: List[Dict[str, Any]],
    *,
    user_responses: Optional[List[Dict[str, Any]]] = None,
    judge_pass: bool = True,
):
    """Alias-dispatching ``call_llm`` mock for the unified loop.

    The probe runs through ``ConversationLoop``, which calls multiple
    aliases per turn (assistant_model for the assistant under test,
    judge_model for in-sim assistant-quality judge + follow-up gate,
    user_model for follow-up generation, summary_model for early-stop
    checks).

    ``assistant_responses`` is a queue of payloads for the
    assistant_model alias. ``user_responses`` (optional) is a queue
    for user_model follow-ups; if absent, generic non-empty payload
    is returned. Judge calls return well-formed
    ``<explanation>`` / ``<rating>`` XML so the loop's parser doesn't
    burn through retries on the test fixture.
    """
    iter_assist = iter(assistant_responses)
    iter_user = iter(user_responses or [])
    rating = "success" if judge_pass else "failure"
    judge_payload = (
        f"<explanation>looks fine</explanation>\n"
        f"<rating>{rating}</rating>"
    )

    def _side_effect(models, alias, msgs, **kwargs):
        if alias == "assistant_model":
            try:
                return next(iter_assist)
            except StopIteration as e:
                raise AssertionError(
                    "test consumed more assistant_model calls than expected"
                ) from e
        if alias == "judge_model":
            return {"role": "assistant", "content": judge_payload}
        if alias == "user_model":
            try:
                return next(iter_user)
            except StopIteration:
                return {
                    "role": "assistant",
                    "content": "Got it, thanks for explaining.",
                }
        if alias == "summary_model":
            return {"role": "assistant", "content": "no"}
        raise AssertionError(f"unexpected alias: {alias!r}")

    return _side_effect


@contextmanager
def _patched_call_llm(side_effect):
    """Patch ``call_llm`` everywhere the unified loop binds it.

    See ``test_sov_ai_multilingual_parity._patched_call_llm`` for
    the rationale: ``core.simulation`` and ``core.judges`` each
    re-import ``call_llm`` from ``core.llm`` at module load, so a
    single patch at the source module doesn't propagate.
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


def _fact(id_: str, *, persona_tags: tuple = (), category: str = "cat-a") -> Fact:
    return Fact(
        id=id_,
        category=category,
        question_type="factual_recall",
        placeholder=True,
        question=f"Question for {id_}?",
        false_premises=(),
        ground_truth=f"Answer for {id_}.",
        graceful_unknown=None,
        persona_tags=persona_tags,
        difficulty="easy",
        provenance=FactProvenance(
            source="placeholder",
            last_reviewed="2026-04-18",
        ),
        locale="pt_BR",
        bank_id="synthetic",
        bank_version="v0.1.0",
    )


def _build_synthetic_bank(
    *,
    facts: List[Fact],
    bank_version: str = "v0.1.0",
) -> FactBank:
    return FactBank(
        schema_version="v0.1",
        locale="pt_BR",
        bank_id="synthetic",
        bank_version=bank_version,
        categories=("cat-a",),
        tags=(),
        facts=tuple(facts),
    )


def _persona_variations() -> List[Dict[str, Any]]:
    """A small population of synthetic personas for statistical assertions."""
    return [
        {"age": age, "state_abbrev": state, "occupation": occ}
        for age in (25, 35, 50, 68)
        for state in ("SP", "BA", "PR", "AM")
        for occ in ("Teacher", "Engineer", "Retired")
    ]


def _write_tiny_pt_br_bank(tmp_root: Path, *, bank_id: str = "tiny") -> Path:
    tmp_root.mkdir(parents=True, exist_ok=True)
    path = tmp_root / f"{bank_id}.yaml"
    path.write_text(
        'schema_version: "v0.1"\n'
        "locale: pt_BR\n"
        f"bank_id: {bank_id}\n"
        "bank_version: v0.1.0\n"
        "categories: [geo]\n"
        "entries:\n"
        "  - id: T-001\n"
        "    category: geo\n"
        "    question_type: factual_recall\n"
        "    placeholder: true\n"
        '    question: "Uma pergunta?"\n'
        "    false_premises: []\n"
        '    ground_truth: "Uma resposta."\n'
        '    persona_tags: ["region:any", "interest:geography"]\n'
        "    difficulty: easy\n"
        "    provenance:\n"
        '      source: "placeholder"\n'
        '      last_reviewed: "2026-04-18"\n',
        encoding="utf-8",
    )
    return path
