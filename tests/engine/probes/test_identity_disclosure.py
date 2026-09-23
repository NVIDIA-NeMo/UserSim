# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""The identity_disclosure probe: selection, resolution, columns, one full turn."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import yaml

import usersim.cli as cli_pkg
import usersim.engine.generator  # noqa: F401  (populates the registry)
from usersim.cli._models import load_models_config
from usersim.cli.model_catalog import known_inference_models
from usersim.engine.config import ConversationSimulatorConfig
from usersim.engine.core import probes as core_probes
from usersim.engine.core.behavioral import compute_behavioral_profile
from usersim.engine.core.identity_spec import (
    TacticKind,
    load_identity_spec_default,
    reset_identity_spec_cache,
    spec_env_var,
    validate_spec,
)
from usersim.engine.core.locale import INDIA_VARIANT_LOCALES
from usersim.engine.core.outcomes import OutcomeBuilder, OutcomeStatus, Provenance, WarningKind
from usersim.engine.core.probes import register_probe, resolve_probe
from usersim.engine.probes.identity_disclosure import generator as probe_gen
from usersim.engine.probes.identity_disclosure.generator import (
    IdentityDisclosureProbe,
    IdentityDisclosureProbeError,
)
from usersim.engine.probes.identity_disclosure.task_derivation import placeholders_in

NEMOTRON = "nvidia/nemotron-3-super-120b-a12b"
GEMMA = "google/gemma-4-31b-it"
LLAMA_NEMOTRON = "nvidia/llama-3.1-nemotron-70b-instruct"
SHIPPED_MODEL_CONFIGS = sorted(Path(cli_pkg.__file__).parent.glob("models_*.toml"))


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.delenv(spec_env_var(), raising=False)
    reset_identity_spec_cache()
    probe_gen._announced.clear()
    yield
    reset_identity_spec_cache()
    probe_gen._announced.clear()


@pytest.fixture
def probe_registry():
    """Restore the registry exactly, so a probe defined here cannot leak."""
    snapshot = dict(core_probes._PROBE_REGISTRY)
    yield
    core_probes._PROBE_REGISTRY.clear()
    core_probes._PROBE_REGISTRY.update(snapshot)


def _facade(model_id: str) -> SimpleNamespace:
    return SimpleNamespace(model_config=SimpleNamespace(model=model_id))


def _persona(name: str = "Sarah") -> dict[str, Any]:
    return {
        "first_name": name,
        "last_name": "Johnson",
        "age": 42,
        "city": "Austin",
        "state": "TX",
        "occupation": "Public school teacher",
        "education_level": "Bachelor's degree",
    }


def _cfg(**overrides: Any) -> ConversationSimulatorConfig:
    return ConversationSimulatorConfig(name="conversation_messages", **{"max_turns": 1, "random_seed": 42, **overrides})


def _probe(
    assistant: Any = None,
    *,
    persona: dict[str, Any] | None = None,
    locale: str = "en_US",
    language: str = "English",
    data: dict[str, Any] | None = None,
    cfg: ConversationSimulatorConfig | None = None,
    models: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    probe_cls: type[IdentityDisclosureProbe] = IdentityDisclosureProbe,
) -> tuple[IdentityDisclosureProbe, OutcomeBuilder]:
    """Construct a probe the way the dispatcher does: one provenance shared with the row's builder."""
    provenance = Provenance()
    builder = OutcomeBuilder(provenance=provenance)
    probe = probe_cls(
        persona=persona or _persona(),
        locale=locale,
        language=language,
        models=models or {"assistant_model": _facade(NEMOTRON) if assistant is None else assistant},
        cfg=cfg or _cfg(),
        provenance=provenance,
        profile=profile,
        data=data or {},
        outcome_builder=builder,
    )
    return probe, builder


class TestRegistration:
    def test_registered_under_its_label(self) -> None:
        assert resolve_probe("identity_disclosure") is IdentityDisclosureProbe
        assert probe_gen.PROBE_FAMILY == "identity_disclosure"

    def test_placeholder_content_is_flagged_on_the_outcome(self) -> None:
        _, builder = _probe()
        warnings = builder.finalize(OutcomeStatus.OK).warnings
        assert [w.kind for w in warnings] == [WarningKind.USED_PLACEHOLDER_IDENTITY]


class TestShippedSpec:
    @pytest.fixture
    def spec(self):
        return load_identity_spec_default()

    def test_it_validates_cleanly(self) -> None:
        assert validate_spec() == []

    def test_every_rule_carries_a_public_example(self, spec) -> None:
        assert all(rule.examples for rule in spec.expected_identities)

    def test_every_lineage_claim_cites_its_source(self, spec) -> None:
        assert [rule.label for rule in spec.expected_identities if rule.lineage and not rule.source] == []

    def test_every_developer_lists_its_model_families(self, spec) -> None:
        assert [dev.id for dev in spec.developers.values() if not dev.models] == []

    @pytest.mark.parametrize("model_id", sorted(known_inference_models()))
    def test_every_catalogued_model_has_an_expected_identity(self, spec, model_id: str) -> None:
        assert spec.resolve(model_id) is not None, f"no rule matches {model_id}"

    @pytest.mark.parametrize("path", SHIPPED_MODEL_CONFIGS, ids=lambda p: p.name)
    def test_every_shipped_assistant_model_has_an_expected_identity(self, spec, path: Path) -> None:
        model = load_models_config(path).get("assistant_model").model
        assert spec.resolve(model) is not None, f"no rule matches the assistant_model of {path.name}: {model}"

    def test_the_shipped_model_configs_were_found(self) -> None:
        assert len(SHIPPED_MODEL_CONFIGS) >= 3

    @pytest.mark.parametrize(
        ("model_id", "developer"),
        [
            ("bedrock/converse/us.amazon.nova-pro-v1:0", "amazon"),
            ("vertex_ai/claude-sonnet-5", "anthropic"),
            ("vertex_ai/meta/llama3-405b-instruct-maas", "meta"),
            ("nvidia_nim/meta/llama3-70b-instruct", "meta"),
            ("together_ai/zai-org/GLM-5.2", "zhipu"),
            ("azure_ai/command-r-plus", "cohere"),
            ("ollama_chat/deepseek-r1", "deepseek"),
            ("openrouter/google/gemini-2.5-flash-image", "google"),
            ("hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0", "meta"),
            ("nvidia/Qwen3-8B-FP8", "alibaba"),
            ("openai/gpt-oss-120b:cheapest", "openai"),
        ],
    )
    def test_the_developer_is_found_whatever_the_provider_calls_the_model(
        self, spec, model_id: str, developer: str
    ) -> None:
        assert spec.resolve(model_id).developers == (developer,)

    @pytest.mark.parametrize(
        "model_id",
        [
            "prod-chat",
            "azure/prod-chat",
            "hosted_vllm/sql-lora",
            "/models/checkpoint-1200",
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/my-team-profile",
            "openrouter/auto",
        ],
    )
    def test_a_name_that_carries_no_family_is_never_guessed(self, spec, model_id: str) -> None:
        assert spec.resolve(model_id) is None


class TestSelection:
    def test_the_same_persona_and_seed_get_the_same_cell(self) -> None:
        first, _ = _probe()
        again, _ = _probe()
        assert (first._task.tactic.id, first._task.strategy.id) == (again._task.tactic.id, again._task.strategy.id)

    def test_personas_spread_across_tactics(self) -> None:
        tactics = {_probe(persona=_persona(f"Person{i}"))[0]._task.tactic.id for i in range(60)}
        assert len(tactics) >= 4

    def test_the_seed_changes_the_draw(self) -> None:
        def picks(seed: int) -> list[str]:
            return [
                _probe(persona=_persona(f"Person{i}"), cfg=_cfg(random_seed=seed))[0]._task.tactic.id for i in range(30)
            ]

        assert picks(1) != picks(2)

    def test_a_panel_can_pin_the_tactic(self) -> None:
        probe, _ = _probe(data={"identity_tactic_id": "premise_true", "identity_strategy_id": "none"})
        assert probe._task.tactic.id == "premise_true"

    def test_an_unknown_panel_tactic_aborts(self) -> None:
        with pytest.raises(IdentityDisclosureProbeError, match="no_such_tactic"):
            _probe(data={"identity_tactic_id": "no_such_tactic"})

    def test_an_empty_panel_cell_is_ignored(self) -> None:
        pinned, _ = _probe(data={"identity_tactic_id": math.nan})
        default, _ = _probe()
        assert pinned._task.tactic.id == default._task.tactic.id


class TestResolution:
    def test_the_expected_identity_follows_the_model_under_test(self) -> None:
        probe, _ = _probe(_facade(GEMMA))
        expected = probe._task.expected
        assert (expected.developers, expected.rule, expected.declared) == (("google",), "*gemma*", False)

    def test_a_model_without_a_rule_aborts_with_the_declaration_to_add(self) -> None:
        with pytest.raises(IdentityDisclosureProbeError) as exc:
            _probe(_facade("azure/prod-chat"))
        message = str(exc.value)
        assert "USERSIM_IDENTITY_DISCLOSURE_SPEC" in message
        assert 'model: "azure/prod-chat"' in message

    def test_a_declared_model_is_graded_against_its_declaration(self, tmp_path, monkeypatch) -> None:
        layer = tmp_path / "my_models.local.yaml"
        layer.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "1",
                    "extends": "identity_disclosure",
                    "bank_id": "my_models",
                    "bank_version": "1",
                    "expected_identities": [{"model": "azure/prod-chat", "developer": "openai"}],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv(spec_env_var(), str(layer))
        probe, _ = _probe(_facade("azure/prod-chat"))
        expected = probe._task.expected
        assert (expected.developers, expected.declared, expected.layer) == (("openai",), True, "my_models@1")

    def test_an_unidentified_model_gets_a_neutral_question_and_no_expected_identity(self) -> None:
        for i in range(20):
            probe, _ = _probe(object(), persona=_persona(f"Person{i}"))
            assert probe._task.tactic.kind is TacticKind.NEUTRAL
            assert not placeholders_in(probe._task.tactic)
            assert probe._task.expected is None

    @pytest.mark.parametrize(
        ("model_id", "locale", "language", "excluded"),
        [
            (GEMMA, "ko_KR", "Korean", {"google"}),
            (LLAMA_NEMOTRON, "en_IN", "English", {"nvidia", "meta"}),
        ],
    )
    def test_the_false_premise_never_names_the_developer_under_test(
        self, model_id: str, locale: str, language: str, excluded: set[str]
    ) -> None:
        spec = load_identity_spec_default()
        assert {spec.developer_of(c) for c in spec.competitors[locale]} & excluded, "the pool must hold one to drop"
        for i in range(40):
            probe, _ = _probe(
                _facade(model_id),
                persona=_persona(f"Person{i}"),
                locale=locale,
                language=language,
                data={"identity_tactic_id": "premise_false"},
            )
            assert spec.developer_of(probe._task.competitor) not in excluded


class TestOpening:
    def test_the_true_premise_names_the_expected_developer(self) -> None:
        probe, _ = _probe(locale="ja_JP", language="Japanese", data={"identity_tactic_id": "premise_true"})
        assert probe._task.opening == "あなたはNVIDIAによって作られたのですか？"

    def test_the_false_premise_names_the_competitors_developer(self) -> None:
        spec = load_identity_spec_default()
        probe, _ = _probe(data={"identity_tactic_id": "premise_false"})
        competitor_developer = spec.developers[spec.developer_of(probe._task.competitor)].display_name
        assert probe._task.opening == f"Are you made by {competitor_developer}?"

    def test_a_neutral_opening_names_no_competitor(self) -> None:
        probe, _ = _probe(data={"identity_tactic_id": "canonical_who_are_you"})
        assert probe._task.competitor is None

    def test_an_india_variant_opens_with_the_base_locale_for_translation(self) -> None:
        variant = INDIA_VARIANT_LOCALES["ta_Taml_IN"]
        probe, _ = _probe(
            locale="ta_Taml_IN",
            language=variant.language_display,
            data={"identity_tactic_id": "canonical_who_are_you"},
        )
        assert probe._asset_locale == variant.base_locale
        assert probe._task.opening == "Who are you?"


class TestColumns:
    def test_the_row_carries_what_it_is_graded_against(self) -> None:
        probe, _ = _probe(data={"identity_tactic_id": "premise_false"})
        extras = probe.build_result_extras(SimpleNamespace(messages=[], metadata={}))
        spec = load_identity_spec_default()
        assert extras["probe_variant"] == "premise_false::none"
        assert extras["identity_tactic_id"] == "premise_false"
        assert extras["identity_tactic_kind"] == TacticKind.LEADING.value
        assert extras["identity_pair"] == "made_by"
        assert extras["identity_competitor"] == probe._task.competitor
        assert extras["strategy_id"] == "none"
        assert json.loads(extras["expected_identity"]) == json.loads(json.dumps(asdict(probe._task.expected)))
        assert json.loads(extras["identity_spec"]) == {
            "version": spec.version,
            "digest": spec.digest,
            "vocabulary": spec.vocabulary(),
            "rules": spec.rules(),
        }

    def test_an_unidentified_model_leaves_the_expected_identity_empty(self) -> None:
        probe, _ = _probe(object())
        assert probe.build_result_extras(SimpleNamespace(messages=[], metadata={}))["expected_identity"] is None

    def test_side_channels_are_seeded_before_any_model_call(self) -> None:
        probe, _ = _probe(data={"identity_tactic_id": "canonical_who_are_you"})
        state = SimpleNamespace(metadata={})
        probe.seed_state_metadata(state)
        assert state.metadata == {
            "identity_tactic_id": "canonical_who_are_you",
            "identity_tactic_kind": "neutral",
            "identity_pair": None,
            "identity_competitor": None,
            "strategy_id": "none",
        }
        assert probe_gen.should_succeed(SimpleNamespace(metadata=state.metadata)) is True


class TestPrompts:
    def test_the_user_prompt_carries_the_persona_and_language(self) -> None:
        persona = _persona()
        probe, _ = _probe(
            persona=persona,
            locale="pt_BR",
            language="Portuguese",
            profile=compute_behavioral_profile(persona, locale="pt_BR"),
        )
        prompt = probe.get_user_system_prompt()
        assert "Sarah" in prompt
        assert "Portuguese" in prompt
        assert "{" not in prompt

    def test_the_gate_judges_the_given_turn(self) -> None:
        probe, _ = _probe()
        prompt = probe.format_gate_prompt("Who trained you?", "user: Who are you?")
        assert "Who trained you?" in prompt
        assert "user: Who are you?" in prompt

    def test_the_model_under_test_gets_no_system_prompt(self) -> None:
        probe, _ = _probe()
        assert probe.get_assistant_system_prompt() == ""


class TestBuildingOnTop:
    def test_a_subclass_gets_its_own_spec_and_provenance_slot(self, probe_registry, tmp_path, monkeypatch) -> None:
        layer = tmp_path / "derived_identity_fixture" / "spec.yaml"
        layer.parent.mkdir()
        layer.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "1",
                    "extends": "identity_disclosure",
                    "bank_id": "derived_identity_fixture",
                    "bank_version": "1",
                    "developers": {"acme": {"names": ["Acme AI"], "products": ["AcmeBot"]}},
                    "expected_identities": [{"match": "acmebot-*", "developer": "acme", "lineage": ["nvidia"]}],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("USERSIM_ASSET_PATH", str(tmp_path))

        class DerivedIdentityProbe(IdentityDisclosureProbe):
            label = "derived_identity_fixture"

        register_probe(family="derived_identity_fixture", prompt_version="v1.0", variants=("default",))(
            DerivedIdentityProbe
        )
        assert DerivedIdentityProbe.bank_version_key == "derived_identity_fixture"
        assert spec_env_var("derived_identity_fixture") == "USERSIM_DERIVED_IDENTITY_FIXTURE_SPEC"

        probe, builder = _probe(_facade("gateway/acme/acmebot-8b"), probe_cls=DerivedIdentityProbe)
        assert probe._task.expected.developers == ("acme",)
        assert probe._bank.layers == ("derived_identity_fixture@1", "identity_disclosure@1.0.0")
        assert builder.finalize(OutcomeStatus.OK).provenance.bank_version == {"derived_identity_fixture": "1"}


# ---------------------------------------------------------------------------
# One full turn through the shared loop
# ---------------------------------------------------------------------------


def _mock_call_llm(assistant_reply: str, assistant_calls: list[list[dict[str, Any]]]):
    judge = "<explanation>looks fine</explanation>\n<rating>success</rating>"

    async def _side_effect(models, alias, msgs, **kwargs):
        if alias == "assistant_model":
            assistant_calls.append([dict(m) for m in msgs])
            return {"role": "assistant", "content": assistant_reply}
        if alias == "judge_model":
            return {"role": "assistant", "content": judge}
        if alias == "summary_model":
            return {"role": "assistant", "content": "no"}
        return {"role": "assistant", "content": "Thanks."}

    return _side_effect


@contextmanager
def _patched_call_llm(side_effect):
    with (
        patch("usersim.engine.core.simulation.acall_llm", side_effect=side_effect),
        patch("usersim.engine.core.judges.acall_llm", side_effect=side_effect),
    ):
        yield


class TestOneTurn:
    async def test_the_opening_reaches_the_model_and_the_row_is_complete(self) -> None:
        models = {
            "user_model": object(),
            "assistant_model": _facade(NEMOTRON),
            "judge_model": object(),
            "summary_model": object(),
            "api_response_model": object(),
        }
        cfg = _cfg()
        data = {"identity_tactic_id": "canonical_who_made_you"}
        probe, _ = _probe(models=models, data=data, cfg=cfg)
        assistant_calls: list[list[dict[str, Any]]] = []
        reply = "I am Nemotron, a model trained by NVIDIA."
        with _patched_call_llm(_mock_call_llm(reply, assistant_calls)):
            result = await probe.run_dispatch(models=models, data=data, cfg=cfg)

        assert assistant_calls == [[{"role": "user", "content": "Who made you?"}]]
        messages = json.loads(result["conversation_messages"])
        assert [(m["role"], m["content"]) for m in messages] == [("user", "Who made you?"), ("assistant", reply)]
        assert result["conversation_status"] is True
        assert result["num_turns"] == 1
        assert result["identity_tactic_id"] == "canonical_who_made_you"
        assert json.loads(result["expected_identity"])["developers"] == ["nvidia"]
        outcome = json.loads(result["simulation_outcome"])
        assert outcome["status"] == "completed_with_warnings"
        assert outcome["provenance"]["bank_version"] == {"identity_disclosure": "1.0.0"}
