# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``sov_ai_multilingual_parity`` probe.

Layered the same way as ``test_sov_ai_facts.py`` and
``test_sov_ai_dynamic.py``:

1. **Task-derivation primitives** — ``persona_to_tags`` (re-exported
   from sovereign-knowledge) + ``derive_task`` (the new picker that
   filters by ``supports_locale`` and is deterministic in
   ``(persona_hash, bank_id, bank_version, seed)``).
2. **Query-bank cache** — process-local cache + env override.
3. **Prompts** — every supported locale has both system prompt + follow-up
   instruction; no half-shipped locales.
4. **End-to-end probe** — ``simulate_sov_ai_multilingual_parity``
   with mocked ``acall_llm``; covers single-turn, multi-turn, verbatim
   injection, placeholder warning, structured failure paths
   (bank-load / no-matching-query / locale-not-in-bank / assistant-error).
5. **Sim-side guardrail** — ``should_succeed`` invariant check.

No live LLM. No real Nemotron-Personas dataset. Everything runs in
under a second against fixtures.
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
    OutcomeStatus,
    Provenance,
    WarningKind,
)
from usersim.engine.core.probes import _PROBE_REGISTRY, resolve_probe
from usersim.engine.core.query_bank import (
    Query,
    QueryBank,
    QueryProvenance,
    load_query_bank,
)
from usersim.engine.probes.sov_ai_multilingual_parity import (
    generator as probe_gen,
)
from usersim.engine.probes.sov_ai_multilingual_parity import (
    prompts,
    task_derivation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_BANK_PATH = packaged_assets_dir() / "sov_ai_multilingual_parity/sample.yaml"


@pytest.fixture
def sample_bank() -> QueryBank:
    return load_query_bank(SAMPLE_BANK_PATH)


@pytest.fixture
def br_persona_southeast() -> dict[str, Any]:
    return {
        "first_name": "Mariana",
        "last_name": "Silva",
        "age": 37,
        "state": "SP",
        "state_abbrev": "SP",
        "city": "São Paulo",
        "education_level": "Ensino superior (graduação)",
        "occupation": "Enfermeira",
    }


@pytest.fixture
def in_persona_health_interest() -> dict[str, Any]:
    return {
        "first_name": "Priya",
        "last_name": "Sharma",
        "age": 32,
        "state": "Maharashtra",
        "city": "Mumbai",
        "education_level": "Bachelors",
        "occupation": "Nurse",
    }


@pytest.fixture(autouse=True)
def _reset_bank_cache_between_tests():
    """Keep the module-level query-bank cache from leaking across tests."""
    probe_gen._reset_bank_cache()
    yield
    probe_gen._reset_bank_cache()


@pytest.fixture
def simulator_cfg():
    """Minimal cfg stub the probe reads from."""

    class _Cfg:
        random_seed = 42
        max_turns = 1

    return _Cfg()


@pytest.fixture
def simulator_cfg_two_turns():
    class _Cfg:
        random_seed = 42
        max_turns = 2

    return _Cfg()


def _mock_call_llm(
    assistant_responses: list[dict[str, Any]],
    *,
    user_responses: list[dict[str, Any]] | None = None,
    judge_pass: bool = True,
):
    """Return a side_effect callable that dispatches by model alias.

    The probe runs through ``ConversationLoop``, which calls multiple
    aliases per turn (assistant_model for the assistant under test,
    judge_model for in-sim assistant-quality
    judge + follow-up gate, user_model for follow-up generation,
    summary_model for early-stop / compression).

    ``assistant_responses`` is a queue of payloads for the
    assistant_model alias. ``user_responses`` (optional) is a queue
    for user_model follow-ups; if absent, generic non-empty payload
    is returned. Judge / summary calls return a canned PASS / "yes"
    response so the loop's internal infrastructure does not block on
    test setup. ``judge_pass=False`` flips judge responses to FAIL.
    """
    iter_assist = iter(assistant_responses)
    iter_user = iter(user_responses or [])
    # ``run_inline_judge`` parses ``<explanation>`` / ``<rating>`` XML
    # tags. Anything that does not match the regex is treated as a
    # parse failure → reformat retry → ``failure`` rating, which
    # would burn through gate retries in the unified loop.
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
            except StopIteration:
                return {
                    "role": "assistant",
                    "content": "Got it, thanks for explaining.",
                }
        if alias == "summary_model":
            return {"role": "assistant", "content": "no"}
        raise AssertionError(f"unexpected alias: {alias!r}")

    return _side_effect


def _shipped_bank_env() -> dict[str, str]:
    """Env vars to point ``load_query_bank_default`` at the shipped sample."""
    return {"USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK": str(SAMPLE_BANK_PATH)}


@contextmanager
def _patched_call_llm(side_effect):
    """Patch ``acall_llm`` everywhere it's bound by the unified loop.

    ``ConversationLoop`` calls ``acall_llm`` from
    ``core.simulation`` (assistant turn), and ``run_inline_judge``
    calls it from ``core.judges`` (in-sim assistant-quality judge +
    user-judge gate). Both bindings are local re-imports of
    ``core.llm.acall_llm``, so a single patch at the source doesn't
    propagate. This helper patches both consumer modules at once.
    """
    with (
        patch(
            "usersim.engine.core.simulation.acall_llm",
            side_effect=side_effect,
        ),
        patch(
            "usersim.engine.core.judges.acall_llm",
            side_effect=side_effect,
        ),
    ):
        yield


def _query(
    qid: str,
    *,
    domain: str = "health",
    persona_tags: tuple = ("age:any", "education:any"),
    locales: tuple = ("en_US", "pt_BR", "ja_JP"),
    placeholder: bool = True,
    bank_id: str = "test_bank",
    bank_version: str = "v0.1.0",
    opt_out: tuple = (),
) -> Query:
    """Build a synthetic Query for the unit-test layer."""
    renderings = {loc: f"Question {qid} in {loc}" for loc in locales if loc not in opt_out}
    return Query(
        id=qid,
        domain=domain,
        placeholder=placeholder,
        difficulty="easy",
        persona_tags=persona_tags,
        concern=f"Test concern for {qid}",
        renderings=renderings,
        renderings_opted_out=opt_out,
        provenance=QueryProvenance(
            source="placeholder" if placeholder else "internal",
            last_reviewed="2026-04-20",
        ),
        bank_id=bank_id,
        bank_version=bank_version,
    )


def _make_bank(
    queries: tuple,
    *,
    locales: tuple = ("en_US", "pt_BR", "ja_JP"),
    bank_id: str = "test_bank",
    bank_version: str = "v0.1.0",
) -> QueryBank:
    return QueryBank(
        schema_version="v0.1",
        bank_id=bank_id,
        bank_version=bank_version,
        locales=locales,
        domains=("health", "civic", "tech"),
        tags=(),
        queries=queries,
        source_path="<synthetic>",
    )


# ---------------------------------------------------------------------------
# 1. persona_to_tags re-export
# ---------------------------------------------------------------------------


class TestPersonaToTagsReExport:
    """The module re-exports the canonical ``persona_to_tags`` from the
    sovereign-knowledge module so cross-probe joins on persona-feature
    axes use the same vocabulary. These tests assert the re-export is
    in place and behaves as expected for the multi-locale probe."""

    def test_re_exports_canonical_persona_to_tags(self) -> None:
        from usersim.engine.probes.sov_ai_facts.task_derivation import (
            persona_to_tags as canonical,
        )

        assert task_derivation.persona_to_tags is canonical

    def test_pt_br_persona_gets_brazilian_region(self, br_persona_southeast: dict[str, Any]) -> None:
        tags = task_derivation.persona_to_tags(br_persona_southeast, "pt_BR")
        assert "region:southeast" in tags
        assert "interest:health" in tags  # via "Enfermeira" → healthcare
        assert "education:tertiary" in tags

    def test_in_persona_gets_health_interest(self, in_persona_health_interest: dict[str, Any]) -> None:
        tags = task_derivation.persona_to_tags(in_persona_health_interest, "en_IN")
        assert "interest:health" in tags
        assert "education:tertiary" in tags
        assert "age:25-44" in tags


# ---------------------------------------------------------------------------
# 2. derive_task
# ---------------------------------------------------------------------------


class TestDeriveTask:
    def test_returns_none_when_pool_empty(self) -> None:
        bank = _make_bank((_query("Q1", persona_tags=("interest:music",)),))
        # A persona with no music interest will not match.
        persona = {"age": 30, "education_level": "high school"}
        assert task_derivation.derive_task(persona, bank, "en_US", seed=1) is None

    def test_deterministic_for_same_inputs(self) -> None:
        bank = _make_bank(tuple(_query(f"Q{i}", persona_tags=("age:any", "education:any")) for i in range(8)))
        persona = {
            "first_name": "X",
            "last_name": "Y",
            "age": 30,
            "education_level": "high school",
        }
        a = task_derivation.derive_task(persona, bank, "en_US", seed=42)
        b = task_derivation.derive_task(persona, bank, "en_US", seed=42)
        assert a is not None and a.id == b.id

    def test_different_seed_can_produce_different_pick(self) -> None:
        bank = _make_bank(tuple(_query(f"Q{i}", persona_tags=("age:any", "education:any")) for i in range(8)))
        persona = {
            "first_name": "X",
            "last_name": "Y",
            "age": 30,
            "education_level": "high school",
        }
        picks = {task_derivation.derive_task(persona, bank, "en_US", seed=s).id for s in range(20)}
        # Across 20 seeds we should see more than one distinct pick out
        # of 8 candidates (probability of collapsing to 1 is negligible).
        assert len(picks) > 1

    def test_bank_version_change_changes_pick(self) -> None:
        queries = tuple(_query(f"Q{i}", persona_tags=("age:any", "education:any")) for i in range(8))
        bank_a = _make_bank(queries, bank_version="v0.1.0")
        bank_b = _make_bank(queries, bank_version="v0.2.0")
        persona = {
            "first_name": "X",
            "last_name": "Y",
            "age": 30,
            "education_level": "high school",
        }
        # Same persona, same seed, two bank versions — collect the picks
        # across many seeds; at least one seed must produce different
        # picks (probability of all 20 collisions across 8 candidates
        # given different salts is vanishingly small).
        any_diff = False
        for s in range(20):
            a = task_derivation.derive_task(persona, bank_a, "en_US", seed=s)
            b = task_derivation.derive_task(persona, bank_b, "en_US", seed=s)
            if a.id != b.id:
                any_diff = True
                break
        assert any_diff, "bank_version change should perturb the deterministic pick"

    def test_locale_opt_out_filters_query(self) -> None:
        # Two queries: one supports both locales, one opts out of pt_BR.
        bank = _make_bank(
            (
                _query("Q-OK", persona_tags=("age:any", "education:any")),
                _query(
                    "Q-OPTOUT",
                    persona_tags=("age:any", "education:any"),
                    opt_out=("pt_BR",),
                ),
            )
        )
        persona = {"age": 30, "education_level": "high school"}
        # Pt_BR pool can only return Q-OK.
        for s in range(30):
            picked = task_derivation.derive_task(
                persona,
                bank,
                "pt_BR",
                seed=s,
            )
            assert picked is not None and picked.id == "Q-OK", (
                f"opted-out query should never appear for pt_BR, got {picked.id}"
            )

    def test_excluded_query_ids_filtered(self) -> None:
        bank = _make_bank(tuple(_query(f"Q{i}", persona_tags=("age:any", "education:any")) for i in range(8)))
        persona = {"age": 30, "education_level": "high school"}
        # Exclude all but one — the remaining must be picked.
        excluded = {f"Q{i}" for i in range(7)}
        picked = task_derivation.derive_task(
            persona,
            bank,
            "en_US",
            seed=42,
            excluded_query_ids=excluded,
        )
        assert picked is not None and picked.id == "Q7"

    def test_excluded_takes_pool_to_zero_returns_none(self) -> None:
        bank = _make_bank((_query("Q-ONLY", persona_tags=("age:any", "education:any")),))
        persona = {"age": 30, "education_level": "high school"}
        out = task_derivation.derive_task(
            persona,
            bank,
            "en_US",
            seed=42,
            excluded_query_ids={"Q-ONLY"},
        )
        assert out is None


# ---------------------------------------------------------------------------
# 3. Query-bank cache wiring on the probe module
# ---------------------------------------------------------------------------


class TestQueryBankCache:
    def test_load_bank_uses_default_path(self) -> None:
        with patch.dict(os.environ, _shipped_bank_env()):
            bank = probe_gen._load_bank()
        assert bank.bank_id == "sample_v1"

    def test_load_bank_picks_up_env_override(self, tmp_path: Path) -> None:
        # Write a tiny minimal valid bank to a temp path and point the
        # env var at it.
        text = (
            'schema_version: "v0.1"\n'
            'bank_id: "alt_bank"\n'
            'bank_version: "v9.9.9"\n'
            "locales:\n  - en_US\n"
            "domains:\n  - health\n"
            "entries:\n"
            "  - id: Q-ALT-001\n"
            "    domain: health\n"
            "    placeholder: true\n"
            "    difficulty: easy\n"
            '    persona_tags: ["age:any"]\n'
            "    concern: |\n"
            "      Alt bank concern.\n"
            "    renderings:\n"
            "      en_US: |\n"
            "        Alt question.\n"
            "    provenance:\n"
            '      source: "placeholder"\n'
            '      last_reviewed: "2026-04-20"\n'
        )
        path = tmp_path / "alt.yaml"
        path.write_text(text, encoding="utf-8")
        with patch.dict(os.environ, {"USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK": str(path)}):
            bank = probe_gen._load_bank()
        assert bank.bank_id == "alt_bank"
        assert bank.bank_version == "v9.9.9"


# ---------------------------------------------------------------------------
# 4. Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    @pytest.mark.parametrize(
        "locale",
        SHIPPED_LOCALES,
    )
    def test_every_supported_locale_has_system_prompt(self, locale: str) -> None:
        text = prompts.get_system_prompt(locale)
        assert isinstance(text, str) and len(text) > 200, (
            f"system prompt for {locale} suspiciously short ({len(text)} chars)"
        )
        # Both placeholders the generator's _run_followup_turn needs:
        assert "{persona}" in text
        assert "{domain}" in text
        assert "{difficulty}" in text

    @pytest.mark.parametrize(
        "locale",
        SHIPPED_LOCALES,
    )
    def test_every_supported_locale_has_followup_instruction(self, locale: str) -> None:
        text = prompts.get_followup_instruction(locale)
        assert isinstance(text, str) and len(text) > 100

    def test_unknown_locale_raises(self) -> None:
        with pytest.raises(prompts.PromptsUnavailableError, match="de_DE"):
            prompts.get_system_prompt("de_DE")

    def test_no_half_shipped_locales(self) -> None:
        # Both registries must have the same key set — a locale with a
        # system prompt but no follow-up (or vice versa) would crash at
        # simulate-time in a half-deterministic way.
        sys_locales = set(prompts.available_locales())
        # supported_locale_pairs returns pairs where BOTH prompts exist.
        paired_locales = {a for a, _ in prompts.supported_locale_pairs()}
        assert sys_locales == paired_locales

    def test_all_supported_locales_present(self) -> None:
        # The probe's load-bearing semantic is matched-pair across
        # locales — catching a regression that drops a locale early.
        assert set(prompts.available_locales()) == set(SHIPPED_LOCALES)


# ---------------------------------------------------------------------------
# 5. End-to-end probe
# ---------------------------------------------------------------------------


class TestSimulateSovAiMultilingualParity:
    def test_registered_in_probe_registry(self) -> None:
        # Cross-probe invariants live in tests/probes/test_probe_invariants.py;
        # this is a fast smoke check that the SovAiMultilingualParityProbe
        # class itself is the registered target.
        assert "sov_ai_multilingual_parity" in _PROBE_REGISTRY
        assert resolve_probe("sov_ai_multilingual_parity").__name__ == ("SovAiMultilingualParityProbe")

    async def test_single_turn_runs_end_to_end(
        self,
        br_persona_southeast: dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        with (
            patch.dict(os.environ, _shipped_bank_env()),
            _patched_call_llm(
                _mock_call_llm(
                    [
                        {"role": "assistant", "content": "Resposta de teste."},
                    ]
                )
            ),
        ):
            result = await probe_gen.simulate_sov_ai_multilingual_parity(
                models={
                    "user_model": object(),
                    "assistant_model": object(),
                    "judge_model": object(),
                    "summary_model": object(),
                    "api_response_model": object(),
                },
                data={},
                persona=br_persona_southeast,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )

        assert result["conversation_status"] is True
        assert result["num_turns"] == 1
        assert result["num_tool_calls"] == 0
        # probe_variant must be the picked query id (not "default").
        assert result["probe_variant"].startswith("Q-")
        assert result["query_id"] == result["probe_variant"]

        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] in {"ok", "completed_with_warnings"}
        # The shipped bank mixes reviewed queries with honest placeholders
        # for generated rows; either warning state is valid depending on
        # which query the persona-derived picker chose.
        # bank_version must be pinned on provenance for drift detection.
        assert outcome["provenance"]["bank_version"].get("pt_BR") == "v0.6.0"

    async def test_first_user_turn_is_locale_rendering_verbatim(
        self,
        br_persona_southeast: dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        # Pick the deterministic choice for this (persona, seed) so we
        # can assert on the exact text.
        with patch.dict(os.environ, _shipped_bank_env()):
            bank = probe_gen._load_bank()
            picked = task_derivation.derive_task(
                br_persona_southeast,
                bank,
                "pt_BR",
                seed=simulator_cfg.random_seed,
            )
            assert picked is not None
            expected_first_turn = picked.rendering_for("pt_BR")

        # Reset cache so the simulate call re-loads through the env path
        # rather than picking up the cached bank we just used.
        probe_gen._reset_bank_cache()

        with (
            patch.dict(os.environ, _shipped_bank_env()),
            _patched_call_llm(
                _mock_call_llm(
                    [
                        {"role": "assistant", "content": "ok"},
                    ]
                )
            ),
        ):
            result = await probe_gen.simulate_sov_ai_multilingual_parity(
                models={
                    "user_model": object(),
                    "assistant_model": object(),
                    "judge_model": object(),
                    "summary_model": object(),
                    "api_response_model": object(),
                },
                data={},
                persona=br_persona_southeast,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )

        # Post-Bucket-D: the unified ConversationLoop puts the
        # assistant_system_prompt as msgs[0] (system) and the
        # verbatim user turn as msgs[1] (user). The system message
        # is preserved in the conversation log for downstream
        # auditability.
        msgs = json.loads(result["conversation_messages"])
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert user_msgs, "expected at least one user turn"
        assert user_msgs[0]["content"] == expected_first_turn

    async def test_two_turn_runs_followup(
        self,
        br_persona_southeast: dict[str, Any],
        simulator_cfg_two_turns: Any,
    ) -> None:
        with (
            patch.dict(os.environ, _shipped_bank_env()),
            _patched_call_llm(
                _mock_call_llm(
                    assistant_responses=[
                        {"role": "assistant", "content": "Primeira resposta."},
                        {"role": "assistant", "content": "Segunda resposta."},
                    ],
                    user_responses=[
                        {
                            "role": "assistant",
                            "content": "Mas e quanto a outra parte?",
                        },
                    ],
                )
            ),
        ):
            result = await probe_gen.simulate_sov_ai_multilingual_parity(
                models={
                    "user_model": object(),
                    "assistant_model": object(),
                    "judge_model": object(),
                    "summary_model": object(),
                    "api_response_model": object(),
                },
                data={},
                persona=br_persona_southeast,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg_two_turns,
                provenance=Provenance(),
            )

        assert result["conversation_status"] is True
        assert result["num_turns"] == 2
        msgs = json.loads(result["conversation_messages"])
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[1]["content"] == "Mas e quanto a outra parte?"

    async def test_locale_not_in_bank_returns_structured_failure(
        self,
        br_persona_southeast: dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        with patch.dict(os.environ, _shipped_bank_env()):
            result = await probe_gen.simulate_sov_ai_multilingual_parity(
                models={},
                data={},
                persona=br_persona_southeast,
                profile={},
                locale="de_DE",  # not in shipped sample bank
                language="German",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert result["conversation_status"] is False
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == OutcomeStatus.FAILED.value
        assert outcome["failure_class"] == FailureClass.SCENARIO_ABORTED.value
        assert "de_DE" in outcome["failure_detail"]

    async def test_no_matching_query_returns_structured_failure(
        self,
        simulator_cfg: Any,
    ) -> None:
        # A persona whose tags do not match any shipped query (every
        # shipped query requires either age:any or some specific age
        # range; we craft a persona with no reachable tags).
        # The shipped sample uses interest:* and age:* tags, so a
        # persona whose occupation maps to no interest + whose age
        # falls outside any specific bin would still get the "interest:
        # geography" / "interest:history" wildcards from persona_to_tags.
        # To force no-match we patch a synthetic bank with a tag that
        # the persona cannot produce.
        synthetic_bank = _make_bank(
            (
                _query(
                    "Q-IMP",
                    persona_tags=("interest:nonexistent-interest",),
                    locales=("en_US", "pt_BR"),
                ),
            ),
            locales=("en_US", "pt_BR"),
        )
        mock_call = patch(
            "usersim.engine.core.simulation.acall_llm",
        )
        mock_judge_call = patch(
            "usersim.engine.core.judges.acall_llm",
        )
        with (
            patch.object(
                probe_gen,
                "_load_bank",
                return_value=synthetic_bank,
            ),
            mock_call as mc,
            mock_judge_call as mjc,
        ):
            result = await probe_gen.simulate_sov_ai_multilingual_parity(
                models={},
                data={},
                persona={"age": 30, "education_level": "high school"},
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        # No LLM call should have happened — we failed before turn 1.
        mc.assert_not_called()
        mjc.assert_not_called()
        assert result["conversation_status"] is False
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == OutcomeStatus.FAILED.value
        assert outcome["failure_class"] == FailureClass.SCENARIO_ABORTED.value
        assert "no query" in outcome["failure_detail"].lower()

    async def test_assistant_turn1_failure_is_attributed_to_assistant_model(
        self,
        br_persona_southeast: dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        async def _raises(*args, **kwargs):
            raise RuntimeError("assistant API exploded")

        with (
            patch.dict(os.environ, _shipped_bank_env()),
            _patched_call_llm(
                _raises,
            ),
        ):
            result = await probe_gen.simulate_sov_ai_multilingual_parity(
                models={
                    "user_model": object(),
                    "assistant_model": object(),
                    "judge_model": object(),
                    "summary_model": object(),
                    "api_response_model": object(),
                },
                data={},
                persona=br_persona_southeast,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert result["conversation_status"] is False
        # Even on failure, probe_variant + query_id must be pinned —
        # the matched-pair join needs them.
        assert result["probe_variant"].startswith("Q-")
        assert result["query_id"] == result["probe_variant"]
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == OutcomeStatus.FAILED.value
        assert outcome["failure_class"] == FailureClass.INFRASTRUCTURE_ERROR.value
        assert outcome["failure_attribution"] == "assistant_model"

    async def test_bank_load_failure_returns_structured_failure(
        self,
        br_persona_southeast: dict[str, Any],
        simulator_cfg: Any,
    ) -> None:
        # Point env at a path that does not exist.
        with patch.dict(
            os.environ,
            {"USERSIM_SOV_AI_MULTILINGUAL_PARITY_BANK": "/nonexistent/query_bank.yaml"},
        ):
            result = await probe_gen.simulate_sov_ai_multilingual_parity(
                models={},
                data={},
                persona=br_persona_southeast,
                profile={},
                locale="pt_BR",
                language="Portuguese",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        assert result["conversation_status"] is False
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == OutcomeStatus.FAILED.value
        assert "query-bank load failed" in outcome["failure_detail"]

    async def test_non_placeholder_query_does_not_emit_placeholder_warning(
        self,
        simulator_cfg: Any,
    ) -> None:
        # Synthetic non-placeholder bank to confirm the warning is
        # *only* attached when placeholder=True.
        synthetic_bank = _make_bank(
            (
                _query(
                    "Q-CLEAN",
                    persona_tags=("age:any", "education:any"),
                    locales=("en_US",),
                    placeholder=False,
                ),
            ),
            locales=("en_US",),
        )
        with (
            patch.object(
                probe_gen,
                "_load_bank",
                return_value=synthetic_bank,
            ),
            _patched_call_llm(
                _mock_call_llm(
                    [
                        {"role": "assistant", "content": "ok"},
                    ]
                )
            ),
        ):
            result = await probe_gen.simulate_sov_ai_multilingual_parity(
                models={
                    "user_model": object(),
                    "assistant_model": object(),
                    "judge_model": object(),
                    "summary_model": object(),
                    "api_response_model": object(),
                },
                data={},
                persona={"age": 30, "education_level": "high school"},
                profile={},
                locale="en_US",
                language="English",
                cfg=simulator_cfg,
                provenance=Provenance(),
            )
        outcome = json.loads(result["simulation_outcome"])
        assert not any(w["kind"] == WarningKind.USED_PLACEHOLDER_QUERY.value for w in outcome["warnings"])


# ---------------------------------------------------------------------------
# 6. should_succeed
# ---------------------------------------------------------------------------


class TestShouldSucceed:
    def test_returns_true_when_query_id_set(self) -> None:
        from usersim.engine.core.outcomes import OutcomeBuilder
        from usersim.engine.core.simulation import ConversationState

        state = ConversationState(outcome=OutcomeBuilder())
        state.metadata["query_id"] = "Q-MEDS-001"
        assert probe_gen.should_succeed(state) is True

    def test_returns_false_when_query_id_missing(self) -> None:
        from usersim.engine.core.outcomes import OutcomeBuilder
        from usersim.engine.core.simulation import ConversationState

        state = ConversationState(outcome=OutcomeBuilder())
        assert probe_gen.should_succeed(state) is False
