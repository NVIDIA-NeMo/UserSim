# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``cli/model_catalog.py`` + the auto-fill loader path.

Covers:

- TOML rows that omit fields pick them up from ``INFERENCE_DEFAULTS``.
- Explicit TOML values always win on overlap.
- Unknown models log a warning and fall back to project defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from usersim.cli._models import (
    ConfigError,
    ModelSpec,
    ModelsConfig,
    ProviderSpec,
    apply_model_overrides,
    load_models_config,
    print_resolved_models,
    prompt_for_provider_keys,
    smoke_test_models,
    smoke_test_models_or_raise,
)
from usersim.cli.model_catalog import (
    INFERENCE_DEFAULTS,
    VLLM_DEFAULTS,
    known_inference_models,
    known_vllm_models,
    resolve_inference_defaults,
    resolve_vllm_defaults,
)


# ---------------------------------------------------------------------------
# Catalog primitives
# ---------------------------------------------------------------------------


class TestCatalogPrimitives:
    def test_resolve_inference_returns_copy(self):
        # Mutating the returned dict must not affect the catalog.
        d = resolve_inference_defaults("openai/gpt-oss-120b")
        d["temperature"] = 0.0
        assert INFERENCE_DEFAULTS["openai/gpt-oss-120b"]["temperature"] == 1.0

    def test_resolve_vllm_returns_copy(self):
        d = resolve_vllm_defaults("openai/gpt-oss-120b")
        d["tensor_parallelism"] = 99
        assert VLLM_DEFAULTS["openai/gpt-oss-120b"]["tensor_parallelism"] == 1

    def test_unknown_model_returns_empty_dict(self):
        assert resolve_inference_defaults("not/a/model") == {}
        assert resolve_vllm_defaults("not/a/model") == {}

    def test_embedding_model_detection(self):
        from usersim.cli.model_catalog import is_embedding_model
        assert is_embedding_model("nvidia/nvidia/nemotron-3-embed-1b")
        assert is_embedding_model("nvidia/qwen/qwen3-embedding-0.6b")
        assert is_embedding_model("some/text-embedding-3")  # 'embed' substring fallback
        assert not is_embedding_model("openai/gpt-oss-120b")

def _cli_dir() -> Path:
    """Directory the ``cli`` package lives in, wherever that ends up."""
    from usersim import cli

    return Path(cli.__file__).resolve().parent

    def test_qwen_embedding_defaults_supply_input_type_and_truncate(self):
        from usersim.cli._models import to_model_configs
        from usersim.cli.model_catalog import resolve_embedding_defaults
        eb = resolve_embedding_defaults("nvidia/qwen/qwen3-embedding-0.6b")["extra_body"]
        assert eb == {"input_type": "passage", "truncate": "NONE"}
        # even a bare spec (no extra_body) picks up the catalog defaults via to_model_configs
        cfg = ModelsConfig(providers=(), models=(
            ModelSpec(alias="embedding_model",
                      model="nvidia/qwen/qwen3-embedding-0.6b",
                      provider="nvidia_inference_hub"),))
        ip = to_model_configs(cfg)[0].inference_parameters
        assert ip.extra_body == {"input_type": "passage", "truncate": "NONE"}

    def test_to_model_configs_uses_embedding_params_for_embedding_models(self):
        from usersim.cli._models import to_model_configs
        cfg = ModelsConfig(
            providers=(),
            models=(
                ModelSpec(alias="doc_gen_model", model="nvidia/google/gemma-4-31b-it",
                          provider="nvidia_inference_hub"),
                ModelSpec(alias="embedding_model",
                          model="nvidia/nvidia/nemotron-3-embed-1b",
                          provider="nvidia_inference_hub"),
            ),
        )
        by_alias = {mc.alias: mc for mc in to_model_configs(cfg)}
        assert type(by_alias["embedding_model"].inference_parameters).__name__ \
            == "EmbeddingInferenceParams"
        assert type(by_alias["doc_gen_model"].inference_parameters).__name__ \
            == "ChatCompletionInferenceParams"

    def test_parallel_overrides_swap_only_concurrency(self):
        from usersim.cli._models import (
            apply_parallel_overrides, parse_cli_parallel_overrides,
        )
        cfg = ModelsConfig(providers=(), models=(
            ModelSpec(alias="doc_gen_model", model="m", provider="p",
                      max_parallel_requests=16),
            ModelSpec(alias="asset_judge_model", model="m2", provider="p",
                      max_parallel_requests=16),
        ))
        cfg2 = apply_parallel_overrides(cfg, parse_cli_parallel_overrides(["doc_gen_model=8"]))
        assert cfg2.get("doc_gen_model").max_parallel_requests == 8
        assert cfg2.get("asset_judge_model").max_parallel_requests == 16  # untouched
        with pytest.raises(ConfigError):
            parse_cli_parallel_overrides(["doc_gen_model=0"])  # must be >= 1
        with pytest.raises(ConfigError):
            apply_parallel_overrides(cfg, {"unknown_alias": 8})

    def test_known_models_helpers_are_sets(self):
        assert isinstance(known_inference_models(), set)
        assert isinstance(known_vllm_models(), set)
        # The hub-served model should be in inference but NOT vllm.
        assert "openai/openai/gpt-5.5" in known_inference_models()
        assert "openai/openai/gpt-5.5" not in known_vllm_models()


class TestGemma4ThinkingTrigger:
    """Lock for the gemma-4-31b-it reasoning-mode trigger.

    The NVIDIA inference hub gateway forwards ``chat_template_kwargs``
    to vLLM, which is what gates Gemma's chat-template ``enable_thinking``
    flag and turns on reasoning. Other documented triggers (Google API
    top-level ``enable_thinking``, the model-card ``<|think|>`` system
    token, OpenAI-style ``reasoning_effort``) all fail silently. See
    ``tools/probe_gemma_thinking.py`` for the per-trigger evidence.

    A regression here would silently drop reasoning on every gemma run.
    """

    MODEL = "nvidia/google/gemma-4-31b-it"

    def test_present_in_inference_defaults(self):
        assert self.MODEL in INFERENCE_DEFAULTS
        assert self.MODEL in known_inference_models()

    def test_thinking_trigger_uses_chat_template_kwargs(self):
        """The catalog MUST set ``extra_body.chat_template_kwargs.enable_thinking``
        to ``True``. The other documented triggers don't fire on the
        NVIDIA hub (probe-confirmed)."""
        defaults = INFERENCE_DEFAULTS[self.MODEL]
        assert defaults.get("extra_body") == {
            "chat_template_kwargs": {"enable_thinking": True}
        }, (
            "gemma-4-31b-it requires extra_body.chat_template_kwargs."
            "enable_thinking=True to surface reasoning on the NVIDIA "
            "inference hub. See tools/probe_gemma_thinking.py."
        )

    def test_does_not_use_dead_triggers(self):
        """Lock against the abandoned trigger forms. Setting any of these
        in addition to the chat-template one would either be a no-op or
        confuse a future debugger reading the catalog."""
        defaults = INFERENCE_DEFAULTS[self.MODEL]
        eb = defaults.get("extra_body") or {}
        # NOT a top-level enable_thinking (Google-API form -- doesn't
        # round-trip through the NVIDIA gateway).
        assert "enable_thinking" not in eb
        # NOT reasoning_effort (OpenAI form -- silently ignored here).
        assert "reasoning_effort" not in eb

    def test_sampling_defaults_match_model_card(self):
        """Gemma-4-instruct's documented sampling: temperature=1.0,
        top_p=0.95. max_tokens generous enough to fit reasoning + answer."""
        defaults = INFERENCE_DEFAULTS[self.MODEL]
        assert defaults["temperature"] == 1.0
        assert defaults["top_p"] == 0.95
        # max_tokens=8192 mirrors other thinking-capable entries; with
        # thinking enabled, completions can be ~2x the non-thinking
        # length on rubric-heavy turns (probe showed 881 vs 476).
        assert defaults["max_tokens"] == 8192

    def test_extra_body_round_trips_through_loader(self, tmp_path: Path):
        """End-to-end: a TOML row that omits extra_body picks up the
        chat-template kwargs from the catalog. This is what makes a
        notebook config that just specifies ``model=nvidia/google/
        gemma-4-31b-it`` actually fire reasoning."""
        p = tmp_path / "gemma.toml"
        p.write_text(
            '[[models]]\nalias="assistant_model"\n'
            f'model="{self.MODEL}"\n'
            'provider="nvidia"\n'
        )
        cfg = load_models_config(p)
        spec = cfg.get("assistant_model")
        assert spec is not None
        assert spec.extra_body == {
            "chat_template_kwargs": {"enable_thinking": True}
        }
        assert spec.temperature == 1.0
        assert spec.top_p == 0.95


# ---------------------------------------------------------------------------
# Loader auto-fill (the new behavior in cli/_models.py:_to_model_spec)
# ---------------------------------------------------------------------------


class TestLoaderAutofill:
    def test_minimal_toml_resolves_inference_defaults(self, tmp_path: Path):
        # alias/model/provider only -- everything else from catalog or
        # project defaults.
        p = tmp_path / "minimal.toml"
        p.write_text(
            '[[models]]\nalias="user_model"\n'
            'model="openai/gpt-oss-120b"\n'
            'provider="nvidia"\n'
        )
        cfg = load_models_config(p)
        spec = cfg.get("user_model")
        assert spec is not None
        assert spec.temperature == 1.0  # from catalog
        assert spec.top_p == 1.0
        assert spec.extra_body == {"reasoning_effort": "high"}
        # Project-level fallbacks (catalog doesn't define these per-model):
        assert spec.max_parallel_requests == 4

    def test_explicit_toml_value_wins(self, tmp_path: Path):
        p = tmp_path / "explicit.toml"
        p.write_text(
            '[[models]]\nalias="user_model"\n'
            'model="openai/gpt-oss-120b"\n'
            'provider="nvidia"\n'
            "temperature=0.5\n"  # explicit override
        )
        spec = load_models_config(p).get("user_model")
        assert spec.temperature == 0.5  # TOML wins
        # Catalog still fills the gaps it knows about:
        assert spec.top_p == 1.0
        assert spec.extra_body == {"reasoning_effort": "high"}

    def test_extra_body_full_replacement_not_deep_merge(self, tmp_path: Path):
        # Catalog has {"reasoning_effort": "high"}; TOML override fully
        # replaces it (not key-by-key merge).
        p = tmp_path / "extra_body.toml"
        p.write_text(
            '[[models]]\nalias="judge_model"\n'
            'model="openai/gpt-oss-120b"\n'
            'provider="nvidia"\n'
            'extra_body = {custom_knob = "x"}\n'
        )
        spec = load_models_config(p).get("judge_model")
        # reasoning_effort is GONE -- TOML's extra_body replaces catalog's.
        assert spec.extra_body == {"custom_knob": "x"}
        assert "reasoning_effort" not in spec.extra_body

    def test_unknown_model_warns_and_falls_back(self, tmp_path: Path, caplog):
        p = tmp_path / "unknown.toml"
        p.write_text(
            '[[models]]\nalias="user_model"\n'
            'model="unknown/vendor/whatever-99b"\n'
            'provider="nvidia"\n'
        )
        with caplog.at_level(logging.WARNING, logger="usersim.cli._models"):
            spec = load_models_config(p).get("user_model")
        # Project-level fallbacks kick in.
        assert spec.temperature == 0.7
        assert spec.top_p == 0.95
        # Warning naming the model.
        assert any(
            "unknown/vendor/whatever-99b" in rec.message
            for rec in caplog.records
        )

    def test_shipped_default_toml_resolves_consistently(self):
        # End-to-end check: cli/models_default.toml's slimmed rows pick up
        # the per-model defaults (temperature / top_p / max_tokens /
        # extra_body) from the catalog, and per-alias fields (timeout)
        # from the TOML row itself.
        cfg = load_models_config(
            _cli_dir() / "models_default.toml"
        )
        assistant = cfg.get("assistant_model")
        assert assistant.temperature == 1.0
        assert assistant.top_p == 1.0
        assert assistant.max_tokens == 8192             # catalog
        assert assistant.timeout == 300                 # alias-specific (TOML)
        assert assistant.extra_body == {"reasoning_effort": "high"}

    def test_shipped_hub_toml_resolves_consistently(self):
        cfg = load_models_config(
            _cli_dir() / "models_hub_support.toml"
        )
        # gpt-5.5 model uses hub provider; no extra_body.
        user = cfg.get("user_model")
        assert user.model == "openai/openai/gpt-5.5"
        assert user.provider == "nvidia-inference-hub"
        assert user.extra_body is None
        # assistant_model still on nvidia, with extra_body from catalog.
        assistant = cfg.get("assistant_model")
        assert assistant.provider == "nvidia"
        assert assistant.extra_body == {"reasoning_effort": "high"}

    def test_evaluator_model_alias_shipped_in_both_tomls(self):
        """Regression lock: both shipped model TOMLs must declare an
        `evaluator_model` row so post-hoc trajectory evaluation works out
        of the box without `MODEL_OVERRIDES`. The alias is distinct from
        the simulator's `judge_model` (separate roles, separately sourced).
        """
        from usersim.cli._models import REQUIRED_EVALUATOR_ALIASES

        assert REQUIRED_EVALUATOR_ALIASES == ("evaluator_model",)

        for toml_name in ("models_default.toml", "models_hub_support.toml"):
            cfg = load_models_config(
                _cli_dir() / toml_name
            )
            assert cfg.get("evaluator_model") is not None, (
                f"{toml_name} is missing the `evaluator_model` row"
            )
            assert cfg.get("judge_model") is not None, (
                f"{toml_name} is missing the `judge_model` row "
                "(simulator-side, distinct from evaluator_model)"
            )

    def test_embedding_model_alias_shipped_in_both_tomls(self):
        """Regression lock: both shipped model TOMLs must declare an
        `embedding_model` row. The financial_services probe embeds kb_search
        queries via this alias for dense retrieval; without it dense mode
        SILENTLY degrades to lexical word-overlap (an entire run's retrieval
        quietly ran on the wrong backend before this was caught). The alias
        must resolve to an embedding model matching the corpus vectors.
        """
        from usersim.cli.model_catalog import is_embedding_model

        for toml_name in ("models_default.toml", "models_hub_support.toml"):
            cfg = load_models_config(
                _cli_dir() / toml_name
            )
            emb = cfg.get("embedding_model")
            assert emb is not None, (
                f"{toml_name} is missing the `embedding_model` row -- "
                "financial_services dense retrieval needs it"
            )
            assert is_embedding_model(emb.model), (
                f"{toml_name}:embedding_model resolves to {emb.model!r}, "
                "which is not recognized as an embedding model"
            )



# ---------------------------------------------------------------------------
# apply_model_overrides
# ---------------------------------------------------------------------------


class TestApplyModelOverrides:
    """Per-alias model swaps re-resolve catalog defaults."""

    def _cfg(self) -> ModelsConfig:
        # Use the shipped default TOML so overrides have a realistic baseline.
        return load_models_config(
            _cli_dir() / "models_default.toml"
        )

    def test_empty_overrides_returns_same_config(self):
        cfg = self._cfg()
        assert apply_model_overrides(cfg, {}) is cfg

    def test_swap_to_model_with_different_extra_body_drops_old_extra(self):
        # gpt-oss-120b catalog has extra_body={"reasoning_effort": "high"};
        # gpt-5.5 has none. Swapping must clear extra_body.
        cfg = self._cfg()
        cfg2 = apply_model_overrides(
            cfg, {"judge_model": "openai/openai/gpt-5.5"}
        )
        judge = cfg2.get("judge_model")
        assert judge.model == "openai/openai/gpt-5.5"
        assert judge.extra_body is None  # gpt-oss reasoning_effort cleared

    def test_swap_preserves_per_alias_fields(self):
        # max_parallel_requests / timeout / provider belong to the alias;
        # they MUST survive a model swap. (max_tokens IS per-model in the
        # catalog as of the 8192-uniform-cap landing -- it gets re-resolved
        # along with temperature / top_p / extra_body. The other test
        # `test_swap_to_model_with_different_extra_body_drops_old_extra`
        # covers the model-keyed re-resolution path.)
        cfg = self._cfg()
        cfg2 = apply_model_overrides(
            cfg, {"summary_model": "openai/gpt-oss-120b"}
        )
        summary = cfg2.get("summary_model")
        assert summary.model == "openai/gpt-oss-120b"
        assert summary.max_parallel_requests == 4       # alias-specific, kept
        assert summary.provider == "nvidia"             # alias-specific, kept
        # New model is in catalog -> reasoning_effort=high carries over.
        assert summary.extra_body == {"reasoning_effort": "high"}
        # max_tokens is per-model in the catalog now (8192 uniformly).
        assert summary.max_tokens == 8192

    def test_other_aliases_unchanged_by_targeted_override(self):
        cfg = self._cfg()
        cfg2 = apply_model_overrides(
            cfg, {"judge_model": "openai/openai/gpt-5.5"}
        )
        # Only judge_model swapped; rest of config identical.
        for alias in ("user_model", "assistant_model", "api_response_model", "summary_model"):
            assert cfg.get(alias) == cfg2.get(alias)

    def test_unknown_alias_raises(self):
        cfg = self._cfg()
        with pytest.raises(ConfigError, match="unknown alias"):
            apply_model_overrides(cfg, {"not_a_real_alias": "some/model"})

    def test_swap_to_model_outside_catalog_warns_and_clears_extra_body(self, caplog):
        """Override to a non-catalog model must NOT inherit the old
        model's extra_body.

        Regression: prior behavior inherited the old extra_body, which
        meant swapping ``openai/gpt-oss-120b`` (catalog supplies
        ``{"reasoning_effort": "high"}``) -> a non-catalog model would
        carry the gpt-oss-only ``reasoning_effort`` knob across to the
        new model. ``extra_body`` keys are model-family-specific
        (gpt-oss ``reasoning_effort``, gemma ``chat_template_kwargs``,
        anthropic ``thinking``); inheriting one family's knob across
        silently risks 400s or unexpected behavior at the inference
        layer. We now clear ``extra_body`` and surface a louder
        warning that names the dropped keys.
        """
        cfg = self._cfg()
        # Sanity: the user_model row is gpt-oss-120b which carries a
        # gpt-oss-only extra_body knob.
        original_extra = cfg.get("user_model").extra_body
        assert original_extra == {"reasoning_effort": "high"}, (
            "fixture pre-condition: user_model is on gpt-oss-120b with "
            "the gpt-oss-only reasoning_effort knob"
        )

        with caplog.at_level(logging.WARNING, logger="usersim.cli._models"):
            cfg2 = apply_model_overrides(
                cfg, {"user_model": "vendor/unknown-model-99b"}
            )

        # Warning fires, names the model AND the dropped keys so a
        # debugger reading the log knows what changed.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("vendor/unknown-model-99b" in r.message for r in warnings)
        assert any("reasoning_effort" in r.message for r in warnings), (
            "warning must name the cleared extra_body keys so the user "
            "knows what to re-add explicitly if it was intentional"
        )

        user = cfg2.get("user_model")
        assert user.model == "vendor/unknown-model-99b"
        # The KEY assertion: extra_body is cleared, NOT inherited.
        assert user.extra_body is None
        # Sampling params (temperature / top_p / max_tokens) are still
        # kept from the old spec -- they're generic, round-trip across
        # provider families.
        assert user.temperature == cfg.get("user_model").temperature
        assert user.top_p == cfg.get("user_model").top_p
        assert user.max_tokens == cfg.get("user_model").max_tokens

    def test_swap_to_model_outside_catalog_no_old_extra_body_warns_briefly(self, caplog):
        """When the OLD spec had no extra_body to drop, the warning is
        still surfaced (catalog is missing an entry the user should
        add), but it doesn't pretend to name dropped keys."""
        # Build a config whose user_model has NO extra_body (e.g. on
        # claude or gpt-5.5, both of which set extra_body = None).
        # Easiest path: load the hub TOML and swap user_model to claude
        # via a first override (which clears extra_body to None per the
        # catalog), then test a SECOND override to a non-catalog model.
        cfg = load_models_config(
            _cli_dir() / "models_hub_support.toml"
        )
        # Step 1: swap user_model to claude -- catalog has no extra_body
        # for claude, so user.extra_body should land at None.
        cfg = apply_model_overrides(
            cfg, {"user_model": "aws/anthropic/bedrock-claude-opus-4-7"}
        )
        assert cfg.get("user_model").extra_body is None, (
            "fixture pre-condition: claude has no extra_body in the catalog"
        )

        with caplog.at_level(logging.WARNING, logger="usersim.cli._models"):
            cfg2 = apply_model_overrides(
                cfg, {"user_model": "vendor/unknown-model-99b"}
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("vendor/unknown-model-99b" in r.message for r in warnings)
        # The "we cleared keys X, Y, Z" branch must NOT fire when
        # there were no keys to clear.
        assert not any("reasoning_effort" in r.message for r in warnings)
        assert cfg2.get("user_model").extra_body is None

    def test_swap_to_self_hosted_model_routes_to_builtin_provider(self):
        # Hub TOML has user_model on nvidia-inference-hub. Swapping to
        # gpt-oss-20b (which IS in VLLM_DEFAULTS) should auto-route
        # provider back to "nvidia" (the built-in tier).
        cfg = load_models_config(
            _cli_dir() / "models_hub_support.toml"
        )
        cfg2 = apply_model_overrides(cfg, {"user_model": "openai/gpt-oss-20b"})
        user = cfg2.get("user_model")
        assert user.model == "openai/gpt-oss-20b"
        assert user.provider == "nvidia"  # auto-routed off the hub

    def test_swap_to_external_model_routes_to_custom_provider(self):
        # Default TOML has assistant_model on nvidia. Swapping to
        # gpt-5.5 (NOT in VLLM_DEFAULTS) should auto-route to the
        # custom provider declared in MODELS.providers.
        cfg = load_models_config(
            _cli_dir() / "models_hub_support.toml"
        )
        cfg2 = apply_model_overrides(
            cfg, {"assistant_model": "openai/openai/gpt-5.5"}
        )
        assistant = cfg2.get("assistant_model")
        assert assistant.model == "openai/openai/gpt-5.5"
        assert assistant.provider == "nvidia-inference-hub"  # auto-routed to hub

    def test_swap_within_same_tier_keeps_provider(self):
        # gpt-oss-120b -> gpt-oss-20b: both self-hosted, provider unchanged.
        cfg = load_models_config(
            _cli_dir() / "models_hub_support.toml"
        )
        cfg2 = apply_model_overrides(
            cfg, {"assistant_model": "openai/gpt-oss-20b"}
        )
        assert cfg2.get("assistant_model").provider == "nvidia"  # still nvidia

    def test_swap_to_claude_drops_temperature_and_top_p(self):
        # Claude Opus 4.7 rejects non-default temperature, top_p, and
        # top_k (Anthropic's migration guidance: omit them entirely).
        # The catalog sets temperature + top_p to None;
        # apply_model_overrides re-resolves; to_model_configs must then
        # OMIT both from the DD inference params (regression guard).
        from usersim.cli._models import to_model_configs

        cfg = load_models_config(
            _cli_dir() / "models_hub_support.toml"
        )
        cfg2 = apply_model_overrides(
            cfg, {"judge_model": "aws/anthropic/bedrock-claude-opus-4-7"}
        )
        judge = cfg2.get("judge_model")
        assert judge.temperature is None  # explicitly suppressed via catalog
        assert judge.top_p is None        # explicitly suppressed via catalog

        # The DD ModelConfig built from this spec must carry NEITHER.
        mcs = to_model_configs(cfg2)
        judge_mc = next(mc for mc in mcs if mc.alias == "judge_model")
        sent = judge_mc.inference_parameters.generate_kwargs
        assert "temperature" not in sent  # would 400 on Opus 4.7 if present
        assert "top_p" not in sent        # would 400 on Opus 4.7 if present
        assert sent["max_tokens"] == 8192


class TestApplyModelOverridesDictForm:
    """Dict-form per-knob overrides on top of catalog defaults.

    Schema: ``{"alias": {"model": "...", "extra_body": {...}, ...}}``.
    Per-knob precedence is **dict value > catalog default > old spec**.
    Key presence in the dict (not just truthiness) signals "explicit
    override" so users can clear catalog defaults with ``None`` / ``{}``.
    """

    def _hub_cfg(self):
        # Hub TOML: assistant_model on gpt-oss-120b (gpt-oss-only
        # extra_body in catalog), used as the override-source baseline.
        return load_models_config(
            _cli_dir() / "models_hub_support.toml"
        )

    # ── ergonomics: the headline use-case --------------------------------

    def test_dict_form_disables_gemma_thinking_via_empty_extra_body(self):
        """Headline use-case for the dict form: disable gemma's catalog-
        default thinking trigger by passing ``extra_body={}``. This is
        the syntax for "rerun without reasoning" -- the chat_template_
        kwargs that the catalog would normally inject is not sent."""
        cfg = self._hub_cfg()
        cfg2 = apply_model_overrides(
            cfg,
            {
                "assistant_model": {
                    "model": "nvidia/google/gemma-4-31b-it",
                    "extra_body": {},  # explicit empty -> clear catalog default
                },
            },
        )
        a = cfg2.get("assistant_model")
        assert a.model == "nvidia/google/gemma-4-31b-it"
        # Catalog default WOULD have been chat_template_kwargs={enable_thinking}.
        # The explicit empty dict in the override wins.
        assert a.extra_body == {}
        # Sampling defaults still come from the gemma catalog entry.
        assert a.temperature == 1.0
        assert a.top_p == 0.95
        assert a.max_tokens == 8192

    def test_dict_form_extra_body_explicit_none_clears_catalog(self):
        """``extra_body: None`` in the dict form is also a valid
        explicit clear (semantically equivalent to ``{}`` for our
        downstream serialiser, but distinct from "absent")."""
        cfg = self._hub_cfg()
        cfg2 = apply_model_overrides(
            cfg,
            {
                "assistant_model": {
                    "model": "nvidia/google/gemma-4-31b-it",
                    "extra_body": None,
                },
            },
        )
        assert cfg2.get("assistant_model").extra_body is None

    # ── per-knob precedence ---------------------------------------------

    def test_dict_form_temperature_wins_over_catalog(self):
        cfg = self._hub_cfg()
        cfg2 = apply_model_overrides(
            cfg,
            {
                "judge_model": {
                    "model": "openai/openai/gpt-5.5",
                    "temperature": 0.0,  # explicit override
                },
            },
        )
        assert cfg2.get("judge_model").temperature == 0.0

    def test_dict_form_max_tokens_wins_over_catalog(self):
        cfg = self._hub_cfg()
        cfg2 = apply_model_overrides(
            cfg,
            {
                "judge_model": {
                    "model": "openai/openai/gpt-5.5",
                    "max_tokens": 4096,
                },
            },
        )
        assert cfg2.get("judge_model").max_tokens == 4096

    def test_partial_dict_override_falls_through_to_catalog_for_other_keys(self):
        """Only ``extra_body`` is overridden; ``temperature`` / ``top_p``
        / ``max_tokens`` continue to come from the catalog."""
        cfg = self._hub_cfg()
        cfg2 = apply_model_overrides(
            cfg,
            {
                "assistant_model": {
                    "model": "nvidia/google/gemma-4-31b-it",
                    "extra_body": {},  # only this key overridden
                },
            },
        )
        a = cfg2.get("assistant_model")
        # Catalog still wins for the other knobs.
        from usersim.cli.model_catalog import resolve_inference_defaults
        catalog = resolve_inference_defaults("nvidia/google/gemma-4-31b-it")
        assert a.temperature == catalog["temperature"]
        assert a.top_p == catalog["top_p"]
        assert a.max_tokens == catalog["max_tokens"]
        # But the explicit override stuck.
        assert a.extra_body == {}

    # ── validation ------------------------------------------------------

    def test_dict_form_requires_model_key(self):
        cfg = self._hub_cfg()
        with pytest.raises(ConfigError, match="must include a 'model' key"):
            apply_model_overrides(
                cfg,
                {"assistant_model": {"extra_body": {}}},  # no "model"
            )

    def test_dict_form_rejects_unknown_keys(self):
        """A typo (e.g. ``extar_body``) MUST raise rather than silently
        no-op. Catching it at config-load time is much friendlier than
        debugging why your "override" didn't apply."""
        cfg = self._hub_cfg()
        with pytest.raises(ConfigError, match="unknown key"):
            apply_model_overrides(
                cfg,
                {
                    "assistant_model": {
                        "model": "nvidia/google/gemma-4-31b-it",
                        "extar_body": {},  # typo
                    },
                },
            )

    def test_invalid_value_type_raises(self):
        cfg = self._hub_cfg()
        with pytest.raises(ConfigError, match="must be a str.*or a dict"):
            apply_model_overrides(
                cfg,
                {"assistant_model": ["not", "a", "dict", "or", "str"]},  # type: ignore
            )

    # ── backwards compat ------------------------------------------------

    def test_string_form_still_works_unchanged(self):
        """Existing string-form overrides are unchanged: catalog
        defaults fill in everything not in the dict."""
        cfg = self._hub_cfg()
        cfg2 = apply_model_overrides(
            cfg, {"assistant_model": "nvidia/google/gemma-4-31b-it"}
        )
        a = cfg2.get("assistant_model")
        assert a.model == "nvidia/google/gemma-4-31b-it"
        # Catalog default for gemma includes thinking trigger.
        assert a.extra_body == {
            "chat_template_kwargs": {"enable_thinking": True}
        }

    def test_dict_form_no_warning_when_extra_body_explicitly_set(self, caplog):
        """When the user explicitly pins ``extra_body`` via the dict
        form, we don't emit the catalog-miss warning -- the user is
        taking responsibility for the knob themselves."""
        cfg = self._hub_cfg()
        with caplog.at_level(logging.WARNING, logger="usersim.cli._models"):
            apply_model_overrides(
                cfg,
                {
                    "assistant_model": {
                        # Made-up model not in catalog.
                        "model": "vendor/some-future-model",
                        "extra_body": {"some_kwarg": True},
                    },
                },
            )
        catalog_miss_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "vendor/some-future-model" in r.message
            and "extra_body" in r.message
        ]
        # Sampling-params warning may still fire (we didn't pin those),
        # but the extra-body-clearing warning must NOT, because the
        # user explicitly set extra_body.
        clearing_warnings = [
            r for r in catalog_miss_warnings if "clearing" in r.message
        ]
        assert clearing_warnings == []


# ---------------------------------------------------------------------------
# smoke_test_models
# ---------------------------------------------------------------------------


class TestSmokeTestModels:
    """Pre-flight (model, provider) check that hits the real /chat/completions
    endpoint with a 1-token request. Catches auth / entitlement / endpoint
    issues before the simulator burns budget on a broken config."""

    def _cfg(self) -> ModelsConfig:
        return ModelsConfig(
            providers=(
                ProviderSpec(
                    name="my-hub",
                    endpoint="https://hub.example/v1",
                    provider_type="openai",
                    api_key="MY_HUB_KEY",
                ),
            ),
            models=(
                ModelSpec(alias="assistant_model", model="openai/gpt-oss-120b", provider="nvidia"),
                ModelSpec(alias="user_model", model="ext/some-model", provider="my-hub"),
            ),
        )

    def test_reports_missing_env_var(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("MY_HUB_KEY", raising=False)
        results = smoke_test_models(self._cfg())
        assert results["assistant_model"][0] is False
        assert "NVIDIA_API_KEY" in results["assistant_model"][1]
        assert results["user_model"][0] is False
        assert "MY_HUB_KEY" in results["user_model"][1]

    def test_returns_ok_on_200(self, monkeypatch):
        # Stub httpx.post to return a fake 200 -- we don't want network calls in tests.
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")
        monkeypatch.setenv("MY_HUB_KEY", "sk-hub")

        class FakeResponse:
            status_code = 200
            text = '{"choices":[{"message":{"content":"ok"}}]}'

        import httpx as _httpx
        monkeypatch.setattr(_httpx, "post", lambda *a, **kw: FakeResponse())

        results = smoke_test_models(self._cfg())
        assert all(ok for ok, _ in results.values())

    def test_returns_failure_with_status_and_body_on_403(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")
        monkeypatch.setenv("MY_HUB_KEY", "sk-hub")

        class FakeForbidden:
            status_code = 403
            text = '{"status":403,"title":"Forbidden","detail":"Authorization failed"}'

        import httpx as _httpx
        monkeypatch.setattr(_httpx, "post", lambda *a, **kw: FakeForbidden())

        results = smoke_test_models(self._cfg())
        ok, msg = results["assistant_model"]
        assert ok is False
        assert "HTTP 403" in msg
        assert "Authorization failed" in msg  # surfaces the body so user can diagnose

    def test_max_tokens_400_treated_as_soft_pass(self, monkeypatch):
        # Reasoning models (gpt-5.5, etc.) emit thinking tokens that count
        # against the budget, so a tight max_tokens cap can return 400 even
        # when auth/routing/model are all fine. The smoke test should
        # recognize this specific failure mode and not block the run.
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")
        monkeypatch.setenv("MY_HUB_KEY", "sk-hub")

        class FakeMaxTokens400:
            status_code = 400
            text = (
                '{"error":{"message":"litellm.BadRequestError: OpenAIException - '
                '{\\n  \\"error\\": {\\n    \\"message\\": \\"Could not finish the '
                'message because max_tokens or model output limit was reached. '
                'Please try again..."}}'
            )

        import httpx as _httpx
        monkeypatch.setattr(_httpx, "post", lambda *a, **kw: FakeMaxTokens400())

        results = smoke_test_models(self._cfg())
        # Both aliases (different model+provider pairs) should soft-pass.
        for alias, (ok, msg) in results.items():
            assert ok is True, f"expected soft-pass for {alias}, got: {msg}"
            assert "provisioning fine" in msg

    def test_other_400s_still_fail(self, monkeypatch):
        # A 400 NOT about max_tokens (e.g. invalid model name) should still
        # fail the smoke test -- only the specific reasoning-model budget
        # case is treated as a soft pass.
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")
        monkeypatch.setenv("MY_HUB_KEY", "sk-hub")

        class FakeBadModel400:
            status_code = 400
            text = '{"error":{"message":"Model \'foo\' does not exist"}}'

        import httpx as _httpx
        monkeypatch.setattr(_httpx, "post", lambda *a, **kw: FakeBadModel400())

        results = smoke_test_models(self._cfg())
        for alias, (ok, msg) in results.items():
            assert ok is False
            assert "HTTP 400" in msg

    def test_embedding_model_probed_on_embeddings_route(self, monkeypatch):
        """An embedding alias must be probed at /embeddings with an embeddings
        body -- probing it at /chat/completions makes the hub report it as an
        unknown chat model_group (404), which looks like a missing model but is
        really a wrong-route bug. Chat aliases still hit /chat/completions."""
        monkeypatch.setenv("MY_HUB_KEY", "sk-hub")
        cfg = ModelsConfig(
            providers=(
                ProviderSpec(name="my-hub", endpoint="https://hub.example/v1",
                             provider_type="openai", api_key="MY_HUB_KEY"),
            ),
            models=(
                ModelSpec(alias="user_model", model="openai/openai/gpt-5.5", provider="my-hub"),
                ModelSpec(alias="embedding_model",
                          model="nvidia/qwen/qwen3-embedding-0.6b", provider="my-hub"),
            ),
        )
        seen: dict = {}

        class FakeResponse:
            status_code = 200
            text = "ok"

        def _capture(url, json=None, headers=None, timeout=None):
            seen[url] = json
            return FakeResponse()

        import httpx as _httpx
        monkeypatch.setattr(_httpx, "post", _capture)

        results = smoke_test_models(cfg)
        assert all(ok for ok, _ in results.values())

        chat_url = "https://hub.example/v1/chat/completions"
        emb_url = "https://hub.example/v1/embeddings"
        assert chat_url in seen and emb_url in seen
        # chat body has messages; embedding body has input (not messages)
        assert "messages" in seen[chat_url] and "input" not in seen[chat_url]
        assert "input" in seen[emb_url] and "messages" not in seen[emb_url]
        assert seen[emb_url]["model"] == "nvidia/qwen/qwen3-embedding-0.6b"


# ---------------------------------------------------------------------------
# print_resolved_models / prompt_for_provider_keys / smoke_test_models_or_raise
# ---------------------------------------------------------------------------


class TestPrintResolvedModels:
    """Captures stdout to confirm the printout shape stays stable -- both
    notebooks rely on this format to read which provider routing landed."""

    def _cfg(self) -> ModelsConfig:
        return load_models_config(
            _cli_dir() / "models_hub_support.toml"
        )

    def test_includes_alias_model_provider_per_row(self, capsys):
        cfg = self._cfg()
        print_resolved_models(cfg)
        out = capsys.readouterr().out
        # Header summary line names provider routing.
        assert "Loaded" in out and "models" in out
        # Every shipped alias surfaces a line containing alias + model + provider.
        for spec in cfg.models:
            assert spec.alias in out
            assert spec.model in out
            assert f"provider={spec.provider}" in out


class TestPromptForProviderKeys:
    """Should be a no-op when every used provider's env var is set, and
    should not prompt for providers that no current alias references."""

    def _cfg(self) -> ModelsConfig:
        return load_models_config(
            _cli_dir() / "models_hub_support.toml"
        )

    def test_no_prompt_when_env_var_set(self, monkeypatch):
        cfg = self._cfg()
        monkeypatch.setenv("NVIDIA_INFERENCE_HUB_KEY", "stub-key")
        called = {"n": 0}

        def _fake_getpass(prompt):
            called["n"] += 1
            return "should-not-be-called"

        monkeypatch.setattr("getpass.getpass", _fake_getpass)
        prompt_for_provider_keys(cfg)
        assert called["n"] == 0

    def test_skips_providers_not_used_by_any_alias(self, monkeypatch):
        # Build a config whose only alias targets the built-in `nvidia`
        # provider; the custom `nvidia-inference-hub` provider is declared
        # but unused. prompt_for_provider_keys must not prompt for the
        # unused provider's key.
        providers = (
            ProviderSpec(
                name="nvidia-inference-hub",
                endpoint="https://inference.example.com/v1",
                provider_type="openai",
                api_key="UNUSED_PROVIDER_KEY",
            ),
        )
        models = (
            ModelSpec(
                alias="assistant_model",
                model="openai/gpt-oss-120b",
                provider="nvidia",
            ),
        )
        cfg = ModelsConfig(providers=providers, models=models)
        called = {"n": 0}

        def _fake_getpass(prompt):
            called["n"] += 1
            return "value"

        monkeypatch.setattr("getpass.getpass", _fake_getpass)
        monkeypatch.delenv("UNUSED_PROVIDER_KEY", raising=False)
        prompt_for_provider_keys(cfg)
        assert called["n"] == 0, "must not prompt for unused provider's key"


class TestSmokeTestModelsOrRaise:
    """Thin wrapper around smoke_test_models that prints + raises on failure."""

    def _cfg(self) -> ModelsConfig:
        return load_models_config(
            _cli_dir() / "models_hub_support.toml"
        )

    def test_raises_when_any_alias_failed(self, monkeypatch, capsys):
        cfg = self._cfg()

        def _stub(_cfg, *, timeout_sec=15.0, aliases=None):
            return {
                "assistant_model": (True, "OK"),
                "evaluator_model": (False, "HTTP 401: bad key"),
            }

        monkeypatch.setattr("usersim.cli._models.smoke_test_models", _stub)
        with pytest.raises(RuntimeError, match="evaluator_model"):
            smoke_test_models_or_raise(cfg)
        out = capsys.readouterr().out
        # The check + cross badges appear in the printed table.
        assert "\u2713 assistant_model" in out
        assert "\u2717 evaluator_model" in out

    def test_returns_result_dict_on_all_pass(self, monkeypatch):
        cfg = self._cfg()

        def _stub(_cfg, *, timeout_sec=15.0, aliases=None):
            return {"assistant_model": (True, "OK")}

        monkeypatch.setattr("usersim.cli._models.smoke_test_models", _stub)
        out = smoke_test_models_or_raise(cfg)
        assert out == {"assistant_model": (True, "OK")}

    def test_aliases_kwarg_forwards_to_smoke(self, monkeypatch):
        """Scoping the smoke check to a subset of aliases (the eval
        notebook's most-common case) must forward through the wrapper."""
        cfg = self._cfg()
        captured = {}

        def _stub(_cfg, *, timeout_sec=15.0, aliases=None):
            captured["aliases"] = aliases
            return {"evaluator_model": (True, "OK")}

        monkeypatch.setattr("usersim.cli._models.smoke_test_models", _stub)
        smoke_test_models_or_raise(cfg, aliases=("evaluator_model",))
        # The wrapper materialises iterables to lists before forwarding so
        # the scope-message can `len(...)` them; either shape is acceptable.
        assert list(captured["aliases"]) == ["evaluator_model"]


class TestCatalogGapWarning:
    """The catalog-gap warning must fire for a real gap and stay quiet otherwise.

    Embedding models are catalogued in ``EMBEDDING_DEFAULTS`` and carry no
    chat params, so checking them against ``INFERENCE_DEFAULTS`` reports a
    gap that does not exist -- which fired on the shipped default config and
    made every CLI invocation print a warning about its own defaults.
    """

    def _warnings_for(self, model: str, caplog) -> list[str]:
        import logging
        from pathlib import Path

        from usersim.cli._models import _to_model_spec

        entry = {"alias": "a", "model": model, "provider": "nvidia"}
        with caplog.at_level(logging.WARNING, logger="usersim.cli._models"):
            _to_model_spec(entry, source=Path("models_default.toml"))
        return [r.getMessage() for r in caplog.records]

    def test_embedding_model_does_not_warn(self, caplog) -> None:
        messages = self._warnings_for("nvidia/qwen/qwen3-embedding-0.6b", caplog)
        assert messages == [], f"embedding model warned: {messages}"

    def test_uncatalogued_chat_model_still_warns(self, caplog) -> None:
        """The signal this check exists for must survive the fix."""
        messages = self._warnings_for("some-vendor/not-a-real-model", caplog)
        assert any("INFERENCE_DEFAULTS" in m for m in messages), messages

    def test_shipped_default_config_is_warning_free(self, caplog) -> None:
        """A first run must not warn about the configuration it ships with."""
        import logging

        from usersim.cli._models import default_models_path, load_models_config

        with caplog.at_level(logging.WARNING, logger="usersim.cli._models"):
            load_models_config(default_models_path())
        assert [r.getMessage() for r in caplog.records] == []
