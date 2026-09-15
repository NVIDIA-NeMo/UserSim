# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``usersim`` CLI surface.

Scope: argparse parsing, dry-run paths, models-config loader, probe
mix parser, judges parser, and the report subcommand against the
``trajectory_df`` / ``evaluator_df`` fixtures (which exercise the full
report path without any LLM calls). LLM-driven panel/simulate/eval runs
are out of scope here — they are integration-tested via the notebooks
and ad-hoc by the engineer running them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from usersim import cli
from usersim.cli._models import (
    ConfigError,
    ModelSpec,
    ModelsConfig,
    ProviderSpec,
    REQUIRED_SIMULATOR_ALIASES,
    default_models_path,
    load_models_config,
    require_aliases,
    warn_if_missing_api_key,
)
from usersim.cli._pipeline import (
    DEFAULT_PROBE_MIX,
    THEME_DRIVEN_PROBES,
    build_simulator_config_builder,
    known_probes,
)
from usersim.cli.evaluate import _parse_csv_or_none, _parse_judges
from usersim.cli.simulate import _parse_probe_mix
from usersim.engine.core._assets import packaged_assets_dir


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------


class TestTopLevel:
    def test_help_prints_and_exits(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--help"])
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        assert "usersim" in captured.out
        for sub in ("panel", "simulate", "eval", "report"):
            assert sub in captured.out

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--version"])
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "usersim" in out
        assert cli.__version__ in out

    def test_no_subcommand_errors(self):
        # argparse with required=True on subparsers errors before func is set
        with pytest.raises(SystemExit):
            cli.main([])


# ---------------------------------------------------------------------------
# Models config loader
# ---------------------------------------------------------------------------


class TestModelsConfig:
    def test_default_loads(self):
        cfg = load_models_config(default_models_path())
        aliases = set(cfg.aliases())
        for required in REQUIRED_SIMULATOR_ALIASES:
            assert required in aliases

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError):
            load_models_config(tmp_path / "nonexistent.toml")

    def test_no_models_entries(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text("# empty\n")
        with pytest.raises(ConfigError, match="no \\[\\[models\\]\\] entries"):
            load_models_config(p)

    def test_duplicate_aliases(self, tmp_path):
        p = tmp_path / "dup.toml"
        p.write_text(
            '[[models]]\nalias="user_model"\nmodel="x"\nprovider="nvidia"\n'
            '[[models]]\nalias="user_model"\nmodel="y"\nprovider="nvidia"\n'
        )
        with pytest.raises(ConfigError, match="duplicate model aliases"):
            load_models_config(p)

    def test_require_aliases_missing(self, tmp_path):
        p = tmp_path / "minimal.toml"
        p.write_text(
            '[[models]]\nalias="user_model"\nmodel="x"\nprovider="nvidia"\n'
        )
        cfg = load_models_config(p)
        with pytest.raises(ConfigError, match="missing required aliases"):
            require_aliases(
                cfg, required=("user_model", "assistant_model", "judge_model"),
                context="test",
            )


class TestWarnIfMissingApiKey:
    """Backwards-compat single-key check + multi-provider enumeration."""

    def _make_cfg(self, providers, models):
        return ModelsConfig(
            providers=tuple(providers),
            models=tuple(models),
        )

    def test_no_config_legacy_path_warns_when_unset(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        msg = warn_if_missing_api_key()
        assert msg is not None and "NVIDIA_API_KEY" in msg

    def test_no_config_legacy_path_silent_when_set(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")
        assert warn_if_missing_api_key() is None

    def test_with_config_enumerates_custom_provider_env(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")
        monkeypatch.delenv("NVIDIA_INFERENCE_HUB_KEY", raising=False)
        cfg = self._make_cfg(
            providers=[
                ProviderSpec(
                    name="nvidia-inference-hub",
                    endpoint="https://example/v1",
                    provider_type="openai",
                    api_key="NVIDIA_INFERENCE_HUB_KEY",
                ),
            ],
            models=[
                ModelSpec(alias="assistant_model", model="x", provider="nvidia"),
                ModelSpec(alias="user_model", model="y", provider="nvidia-inference-hub"),
                ModelSpec(alias="judge_model", model="z", provider="nvidia-inference-hub"),
            ],
        )
        msg = warn_if_missing_api_key(cfg)
        assert msg is not None
        assert "NVIDIA_INFERENCE_HUB_KEY" in msg
        assert "NVIDIA_API_KEY" not in msg  # NVIDIA_API_KEY is set; only the hub is missing

    def test_with_config_silent_when_all_set(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")
        monkeypatch.setenv("NVIDIA_INFERENCE_HUB_KEY", "sk-hub")
        cfg = self._make_cfg(
            providers=[
                ProviderSpec(
                    name="nvidia-inference-hub",
                    endpoint="https://example/v1",
                    provider_type="openai",
                    api_key="NVIDIA_INFERENCE_HUB_KEY",
                ),
            ],
            models=[
                ModelSpec(alias="assistant_model", model="x", provider="nvidia"),
                ModelSpec(alias="user_model", model="y", provider="nvidia-inference-hub"),
            ],
        )
        assert warn_if_missing_api_key(cfg) is None

    def test_with_config_warns_on_builtin_nvidia_when_unset(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        cfg = self._make_cfg(
            providers=[],
            models=[ModelSpec(alias="assistant_model", model="x", provider="nvidia")],
        )
        msg = warn_if_missing_api_key(cfg)
        assert msg is not None and "NVIDIA_API_KEY" in msg

    def test_with_config_unknown_provider_silently_skipped(self, monkeypatch):
        # An undeclared, non-builtin provider name shouldn't crash the
        # warning path — DD will surface its own error later. The helper
        # just enumerates what it knows about.
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")
        cfg = self._make_cfg(
            providers=[],
            models=[
                ModelSpec(alias="user_model", model="x", provider="some-undeclared-thing"),
            ],
        )
        assert warn_if_missing_api_key(cfg) is None


class TestToDataDesignerKwargs:
    """Verify the DD provider list merges built-ins with custom providers.

    Regression guard: when a config declares a custom provider AND any
    model alias still references a built-in provider name (e.g.
    ``assistant_model`` on ``provider="nvidia"``), the kwargs must
    include both — DD's ``DataDesigner(model_providers=...)`` treats
    a non-empty list as a complete replacement of the defaults, so
    omitting ``nvidia`` raises ``UnknownProviderError`` mid-run.
    """

    def test_no_custom_providers_returns_empty_kwargs(self):
        from usersim.cli._models import to_data_designer_kwargs

        cfg = ModelsConfig(
            providers=(),
            models=(ModelSpec(alias="assistant_model", model="x", provider="nvidia"),),
        )
        assert to_data_designer_kwargs(cfg) == {}

    def test_custom_provider_merges_with_builtin_nvidia(self):
        from usersim.cli._models import to_data_designer_kwargs

        cfg = ModelsConfig(
            providers=(
                ProviderSpec(
                    name="nvidia-inference-hub",
                    endpoint="https://inference.example.com/v1",
                    provider_type="openai",
                    api_key="NVIDIA_INFERENCE_HUB_KEY",
                ),
            ),
            models=(
                ModelSpec(alias="assistant_model", model="x", provider="nvidia"),
                ModelSpec(alias="user_model", model="y", provider="nvidia-inference-hub"),
            ),
        )
        kwargs = to_data_designer_kwargs(cfg)
        names = {p.name for p in kwargs["model_providers"]}
        assert "nvidia-inference-hub" in names  # custom provider preserved
        assert "nvidia" in names                # built-in merged in for assistant_model

    def test_custom_provider_overrides_builtin_on_name_conflict(self):
        from usersim.cli._models import to_data_designer_kwargs

        cfg = ModelsConfig(
            providers=(
                ProviderSpec(
                    name="nvidia",  # shadow the built-in
                    endpoint="https://my-custom-nvidia/v1",
                    provider_type="openai",
                    api_key="MY_KEY",
                ),
            ),
            models=(ModelSpec(alias="assistant_model", model="x", provider="nvidia"),),
        )
        kwargs = to_data_designer_kwargs(cfg)
        nvidia_entries = [p for p in kwargs["model_providers"] if p.name == "nvidia"]
        assert len(nvidia_entries) == 1
        assert nvidia_entries[0].endpoint == "https://my-custom-nvidia/v1"


# ---------------------------------------------------------------------------
# Argument parsers (sub-helpers)
# ---------------------------------------------------------------------------


class TestScenarioMixParser:
    def test_default_normalizes(self):
        m = _parse_probe_mix(
            "tool_calling=1,general_open_ended=1,general_educational=2",
        )
        assert pytest.approx(sum(m.values())) == 1.0
        assert m["general_educational"] > m["tool_calling"]

    def test_unbalanced_renormalizes(self):
        m = _parse_probe_mix("tool_calling=0.5,general_open_ended=0.5,general_educational=0.5")
        assert pytest.approx(sum(m.values())) == 1.0
        assert all(v == pytest.approx(1 / 3) for v in m.values())

    def test_sovereign_ai_probe_accepted(self):
        m = _parse_probe_mix("sov_ai_facts=1.0")
        assert pytest.approx(sum(m.values())) == 1.0
        assert "sov_ai_facts" in m

    def test_safety_probe_accepted(self):
        m = _parse_probe_mix(
            "safety_chat_pressure=0.5,safety_agentic=0.5",
        )
        assert set(m) == {"safety_chat_pressure", "safety_agentic"}

    def test_mixed_general_and_sovereign_ai_accepted(self):
        m = _parse_probe_mix(
            "tool_calling=0.4,sov_ai_dynamic=0.6",
        )
        assert pytest.approx(sum(m.values())) == 1.0
        assert pytest.approx(m["sov_ai_dynamic"]) == 0.6

    def test_empty_raises(self):
        with pytest.raises(SystemExit):
            _parse_probe_mix("")

    def test_zero_weights_raises(self):
        with pytest.raises(SystemExit):
            _parse_probe_mix("tool_calling=0,general_open_ended=0")

    def test_bad_format_raises(self):
        with pytest.raises(SystemExit, match="k=v"):
            _parse_probe_mix("nope")

    def test_unknown_probe_raises_with_helpful_message(self):
        with pytest.raises(SystemExit, match="unknown probe_type"):
            _parse_probe_mix("not_a_real_probe=1.0")

    def test_partial_unknown_lists_only_offenders(self):
        with pytest.raises(SystemExit, match=r"\['typo_probe'\]"):
            _parse_probe_mix("tool_calling=0.5,typo_probe=0.5")


class TestPipelineFactory:
    """Exercise ``build_simulator_config_builder`` against the registry.

    Doesn't run the simulator — just confirms the factory accepts the
    full set of registered probes and rejects unknown ones with a
    clear message. The DD config object it produces is opaque to us;
    we just need it to construct without raising.
    """

    def test_known_probes_matches_registry(self):
        names = set(known_probes())
        # Sanity-check the canonical probes are present.
        assert names >= {
            "tool_calling", "general_open_ended", "general_educational",
            "sov_ai_facts",
            "sov_ai_dynamic",
            "sov_ai_multilingual_parity",
            "safety_chat_pressure",
            "safety_agentic",
        }

    def test_default_probe_mix_is_general_family_only(self):
        # Default-surface contract: when no explicit mix is provided
        # the simulator runs the three ``general`` family probes
        # (tool_calling / general_open_ended / general_educational — the
        # general-purpose conversational simulators). Sovereign-AI / safety
        # probes are opt-in via --probe-mix because they each
        # touch a curated asset bank that ships with placeholder
        # content the user may not want to run against by default.
        assert set(DEFAULT_PROBE_MIX) == {
            "tool_calling", "general_open_ended", "general_educational",
        }

    def test_theme_driven_set_matches_subcategory_branches(self):
        # The pipeline's theme-column dispatcher only has explicit
        # branches for these three; everything else uses the placeholder.
        assert THEME_DRIVEN_PROBES == frozenset(
            {"tool_calling", "general_open_ended", "general_educational"}
        )

    @pytest.mark.parametrize(
        "probe",
        sorted([
            "tool_calling", "general_open_ended", "general_educational",
            "sov_ai_facts",
            "sov_ai_dynamic",
            "sov_ai_multilingual_parity",
            "safety_chat_pressure",
            "safety_agentic",
        ]),
    )
    def test_build_simulator_accepts_each_probe_singly(self, probe):
        models = load_models_config(default_models_path())
        # Should not raise for any registered probe.
        data_designer, builder = build_simulator_config_builder(
            locale="en_US",
            models=models,
            assets_dir=packaged_assets_dir(),
            probe_mix={probe: 1.0},
        )
        assert data_designer is not None
        assert builder is not None

    def test_build_simulator_rejects_unknown_probe(self):
        models = load_models_config(default_models_path())
        with pytest.raises(ValueError, match="Unknown probe_type"):
            build_simulator_config_builder(
                locale="en_US",
                models=models,
                assets_dir=packaged_assets_dir(),
                probe_mix={"made_up": 1.0},
            )

    def test_build_simulator_accepts_mixed_general_and_inclusion(self):
        models = load_models_config(default_models_path())
        data_designer, builder = build_simulator_config_builder(
            locale="en_US",
            models=models,
            assets_dir=packaged_assets_dir(),
            probe_mix={
                "tool_calling": 0.4,
                "safety_agentic": 0.3,
                "sov_ai_dynamic": 0.3,
            },
        )
        assert data_designer is not None
        assert builder is not None


class TestSampleTrajectories:
    """Coverage-aware sampler used by 02_evaluate_simulation.ipynb."""

    def _df(self):

        # 4 locales x 2 probes x 3 rows -- 24 total.
        rows = []
        for locale in ("en_US", "en_IN", "fr_FR", "ja_JP"):
            for probe in ("tool_calling", "safety_chat_pressure"):
                for i in range(3):
                    rows.append({
                        "locale": locale, "probe_family": probe,
                        "trajectory_id": f"{locale}_{probe}_{i}",
                        "query_id": f"q_{i}" if probe == "sov_ai_multilingual_parity" else None,
                    })
        # Add a few multilingual-parity rows for matched_pair coverage.
        for locale in ("en_US", "fr_FR"):
            for q in ("q_a", "q_b", "q_c"):
                rows.append({
                    "locale": locale,
                    "probe_family": "sov_ai_multilingual_parity",
                    "trajectory_id": f"{locale}_parity_{q}",
                    "query_id": q,
                })
        return pd.DataFrame(rows)

    def test_full_mode_passes_through(self):
        from usersim.cli._pipeline import sample_trajectories
        df = self._df()
        out = sample_trajectories(df, mode="full")
        assert len(out) == len(df)

    def test_per_locale_samples_n_per_locale(self):
        from usersim.cli._pipeline import sample_trajectories
        df = self._df()
        out = sample_trajectories(df, mode="per_locale", n=2, seed=42)
        # 4 locales -> at most 4*2 = 8 rows (each locale has >= 2 rows).
        per_locale = out.groupby("locale").size()
        assert (per_locale == 2).all()

    def test_per_locale_probe_samples_per_pair(self):
        from usersim.cli._pipeline import sample_trajectories
        df = self._df()
        out = sample_trajectories(df, mode="per_locale_probe", n=1, seed=42)
        per_pair = out.groupby(["locale", "probe_family"]).size()
        assert (per_pair == 1).all()

    def test_matched_pair_restricts_to_parity_probe(self):
        from usersim.cli._pipeline import sample_trajectories
        df = self._df()
        out = sample_trajectories(df, mode="matched_pair", n=2)
        assert (out["probe_family"] == "sov_ai_multilingual_parity").all()
        # First 2 sorted query_ids (q_a, q_b) -> 2 locales x 2 ids = 4 rows.
        assert set(out["query_id"]) == {"q_a", "q_b"}
        assert len(out) == 4

    def test_unknown_mode_raises(self):
        from usersim.cli._pipeline import sample_trajectories
        with pytest.raises(ValueError, match="Unknown sample mode"):
            sample_trajectories(self._df(), mode="bogus")

    def test_stable_across_runs_at_fixed_seed(self):
        from usersim.cli._pipeline import sample_trajectories
        df = self._df()
        a = sample_trajectories(df, mode="per_locale_probe", n=2, seed=42)
        b = sample_trajectories(df, mode="per_locale_probe", n=2, seed=42)
        assert list(a["trajectory_id"]) == list(b["trajectory_id"])

    def test_n_none_means_no_cap_per_locale(self):
        """``n=None`` for per_locale should pass every row through -- no
        sampling. Functionally equivalent to mode='full'."""
        from usersim.cli._pipeline import sample_trajectories
        df = self._df()
        out = sample_trajectories(df, mode="per_locale", n=None)
        # Per-locale unfiltered means we keep ALL rows whose locale is one
        # of the locales present in df (i.e. every row).
        assert len(out) == len(df)
        per_locale = out.groupby("locale").size()
        # Each locale's count survives unchanged.
        expected = df.groupby("locale").size()
        assert (per_locale == expected).all()

    def test_n_none_means_no_cap_matched_pair(self):
        """``n=None`` for matched_pair should keep every parity query_id,
        not just the first N."""
        from usersim.cli._pipeline import sample_trajectories
        df = self._df()
        out = sample_trajectories(df, mode="matched_pair", n=None)
        # All parity rows present (df has 2 locales x 3 query_ids = 6 rows).
        assert len(out) == 6
        assert set(out["query_id"]) == {"q_a", "q_b", "q_c"}


class TestEvaluateDataframe:
    """Cover the tempfile + key-col-backfill quirks of the eval-pipeline wrapper.

    Mocks ``build_evaluator_config_builder`` so we don't touch DataDesigner;
    the test asserts on the side-effects the helper is responsible for
    (temp-parquet write + delete; key-col backfill on partial DD output).
    """

    def _source_df(self):
        return pd.DataFrame([
            {"trajectory_id": "T1", "locale": "en_US", "probe_family": "x",
             "conversation_messages": "[]"},
            {"trajectory_id": "T2", "locale": "fr_FR", "probe_family": "y",
             "conversation_messages": "[]"},
        ])

    def test_backfills_key_cols_when_dd_drops_them(self, monkeypatch, tmp_path):

        from usersim.cli import _pipeline

        captured_seed_path = {}

        class _Result:
            def load_dataset(self):
                # DataDesigner output: just the eval column, no keys.
                return pd.DataFrame({"assistant_eval": ["{}", "{}"]})

        class _DataDesigner:
            def create(self, builder, num_records):
                return _Result()

        def _stub_builder(*, trajectory_parquet, **kwargs):
            captured_seed_path["path"] = trajectory_parquet
            # The helper writes the df to this path BEFORE calling us;
            # confirm the temp parquet is there + has the right prefix.
            assert Path(trajectory_parquet).exists()
            assert Path(trajectory_parquet).name.startswith("usersim_eval_seed_")
            return _DataDesigner(), object()

        monkeypatch.setattr(_pipeline, "build_evaluator_config_builder", _stub_builder)

        out = _pipeline.evaluate_dataframe(
            self._source_df(),
            models=object(),  # not consumed by the stub
            scorers=["x"],
            judges=[{"alias": "evaluator_model"}],
        )

        # Key cols backfilled from source.
        assert list(out["trajectory_id"]) == ["T1", "T2"]
        assert list(out["locale"]) == ["en_US", "fr_FR"]
        assert list(out["probe_family"]) == ["x", "y"]
        # Eval column preserved.
        assert list(out["assistant_eval"]) == ["{}", "{}"]
        # Temp parquet cleaned up.
        assert not Path(captured_seed_path["path"]).exists()

    def test_preserves_key_cols_already_in_dd_output(self, monkeypatch):

        from usersim.cli import _pipeline

        class _Result:
            def load_dataset(self):
                # DataDesigner output already has the key cols.
                return pd.DataFrame({
                    "trajectory_id": ["DD1", "DD2"],
                    "locale": ["DD_LOCALE_1", "DD_LOCALE_2"],
                    "probe_family": ["DD_PROBE_1", "DD_PROBE_2"],
                    "assistant_eval": ["{}", "{}"],
                })

        class _DataDesigner:
            def create(self, builder, num_records):
                return _Result()

        def _stub_builder(*, trajectory_parquet, **kwargs):
            return _DataDesigner(), object()

        monkeypatch.setattr(_pipeline, "build_evaluator_config_builder", _stub_builder)

        out = _pipeline.evaluate_dataframe(
            self._source_df(),
            models=object(),
            scorers=[],
            judges=[{"alias": "evaluator_model"}],
        )
        # When DD provides them, do NOT clobber with the seed values.
        assert list(out["trajectory_id"]) == ["DD1", "DD2"]

    def test_default_judges_uses_evaluator_model_alias(self, monkeypatch):
        """Regression lock: omitting ``judges=`` should produce a single-judge
        ensemble keyed on ``evaluator_model`` (the canonical eval alias).
        Notebook callers shouldn't have to specify this redundantly when
        they already declared ``evaluator_model`` via MODEL_OVERRIDES."""
        from usersim.cli import _pipeline

        captured = {}

        def _stub_builder(*, trajectory_parquet, judges, **kwargs):
            captured["judges"] = judges
            class _Result:
                def load_dataset(self):
                    return pd.DataFrame({"assistant_eval": ["{}", "{}"]})
            class _DataDesigner:
                def create(self, builder, num_records):
                    return _Result()
            return _DataDesigner(), object()

        monkeypatch.setattr(_pipeline, "build_evaluator_config_builder", _stub_builder)
        _pipeline.evaluate_dataframe(
            self._source_df(),
            models=object(),
            scorers=[],
            # judges omitted -- should default to [{"alias": "evaluator_model"}]
        )
        assert captured["judges"] == [{"alias": "evaluator_model"}]


class TestJudgesParser:
    def test_alias_only(self):
        out = _parse_judges("evaluator_model")
        assert out == [{"alias": "evaluator_model"}]

    def test_alias_with_family(self):
        out = _parse_judges("evaluator_model:OPENAI,backup:ANTHROPIC")
        assert out == [
            {"alias": "evaluator_model", "family": "OPENAI"},
            {"alias": "backup", "family": "ANTHROPIC"},
        ]

    def test_strips_whitespace(self):
        out = _parse_judges(" a , b:OPENAI ")
        assert out == [{"alias": "a"}, {"alias": "b", "family": "OPENAI"}]

    def test_empty_raises(self):
        with pytest.raises(SystemExit):
            _parse_judges("")

    def test_csv_or_none(self):
        assert _parse_csv_or_none(None) is None
        assert _parse_csv_or_none("") is None
        assert _parse_csv_or_none("a,b") == ["a", "b"]
        assert _parse_csv_or_none(" a , , b ") == ["a", "b"]


class TestUndispatchedScorerWarning:
    """A capability whose scorer was not dispatched reads Untested (n=0), which is
    indistinguishable from an untested model. That is not hypothetical: the
    financial_services scorer was missing from the evaluation notebook's SCORERS
    list, so all 11 finance capabilities came back empty across a 13-locale run
    while the scorer sat registered, correctly wired to its axes, and never called.
    Nothing connected the capability registry to the run config, so nothing said so.
    """

    def _warn(self, scorers, caplog):
        import logging

        from usersim.cli._pipeline import _warn_on_undispatched_scorers

        with caplog.at_level(logging.WARNING, logger="usersim.cli.pipeline"):
            _warn_on_undispatched_scorers(scorers)
        return caplog.text

    def test_partial_list_names_the_scorer_and_its_capabilities(self, caplog):
        from usersim.taxonomy.capabilities import scorers_required_by_capabilities

        partial = [
            s for s in scorers_required_by_capabilities() if s != "financial_services"
        ]
        text = self._warn(partial, caplog)
        assert "financial_services" in text
        assert "11 capabilities" in text
        assert "financial_task_success" in text
        assert "Untested" in text

    def test_complete_list_is_silent(self, caplog):
        from usersim.taxonomy.capabilities import scorers_required_by_capabilities

        assert self._warn(list(scorers_required_by_capabilities()), caplog) == ""

    def test_dispatching_nothing_warns_once_not_once_per_scorer(self, caplog):
        """Running with no scorers is a deliberate, documented choice (the CLI
        default). One line per missing scorer there is noise people learn to
        scroll past, which is how the load-bearing warning gets missed."""
        text = self._warn([], caplog)
        assert text.count("WARNING") == 1
        assert "no scorers dispatched" in text

    def test_the_warning_is_wired_into_the_config_builder(self):
        """The tests above call the warner directly, so they stay green even if the
        CALL SITE is deleted. ``build_evaluator_config_builder`` constructs a live
        DataDesigner, so assert on its source: the warning must precede the column
        that freezes the scorer list into the run.
        """
        import inspect

        from usersim.cli import _pipeline

        src = inspect.getsource(_pipeline.build_evaluator_config_builder)
        assert "_warn_on_undispatched_scorers(" in src
        assert src.index("_warn_on_undispatched_scorers(") < src.index("add_column(")


class TestEvalCliJudgesDefault:
    """Regression lock for the post-rename `usersim eval --judges` default.

    Anyone changing the default without updating this assertion is making
    a CLI behaviour change that affects scripted callers — the failure
    should be loud.
    """

    def test_judges_flag_default_is_evaluator_model(self):
        import argparse

        from usersim.cli import evaluate as eval_mod

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        eval_mod.register(sub)
        ns = parser.parse_args(
            ["eval", "--trajectories", "tmp.parquet", "--out", "tmp_out.parquet"]
        )
        assert ns.judges == "evaluator_model"


# ---------------------------------------------------------------------------
# Dry-run subcommands
# ---------------------------------------------------------------------------


class TestDryRuns:
    def test_panel_dry_run(self, tmp_path, capsys):
        rc = cli.main([
            "panel",
            "--locale", "en_US",
            "--num-personas", "5",
            "--out", str(tmp_path / "panel.parquet"),
            "--dry-run",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "dry run" in out
        assert "en_US" in out

    def test_simulate_dry_run_with_locale(self, tmp_path, capsys):
        rc = cli.main([
            "simulate",
            "--locale", "en_US",
            "--num-rows", "3",
            "--out", str(tmp_path / "traj.parquet"),
            "--dry-run",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "probe mix" in out
        assert "max_turns" in out
        assert "store_reasoning: True" in out

    def test_simulate_dry_run_can_disable_reasoning_storage(
        self, tmp_path, capsys,
    ):
        rc = cli.main([
            "simulate",
            "--locale", "en_US",
            "--num-rows", "3",
            "--out", str(tmp_path / "traj.parquet"),
            "--no-store-reasoning",
            "--dry-run",
        ])
        assert rc == 0
        assert "store_reasoning: False" in capsys.readouterr().out

    def test_simulate_locale_without_num_rows_errors(self, tmp_path):
        with pytest.raises(SystemExit, match="--num-rows"):
            cli.main([
                "simulate",
                "--locale", "en_US",
                "--out", str(tmp_path / "traj.parquet"),
            ])

    def test_simulate_panel_or_locale_required(self, tmp_path):
        with pytest.raises(SystemExit):
            cli.main([
                "simulate",
                "--out", str(tmp_path / "traj.parquet"),
            ])

    def test_eval_dry_run(self, tmp_path, capsys):
        traj = tmp_path / "trajs.parquet"
        # The dry-run path doesn't open the file, so an empty placeholder is fine.
        traj.write_bytes(b"")
        rc = cli.main([
            "eval",
            "--trajectories", str(traj),
            "--out", str(tmp_path / "eval.parquet"),
            "--judges", "judge_model",
            "--dry-run",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "judge_model" in out

    def test_eval_unknown_judge_alias_errors(self, tmp_path):
        traj = tmp_path / "trajs.parquet"
        traj.write_bytes(b"")
        with pytest.raises(SystemExit, match="aliases not in"):
            cli.main([
                "eval",
                "--trajectories", str(traj),
                "--out", str(tmp_path / "eval.parquet"),
                "--judges", "nonexistent_judge",
                "--dry-run",
            ])


# ---------------------------------------------------------------------------
# Report subcommand (full path; no LLM)
# ---------------------------------------------------------------------------



class TestReportSubcommand:
    """End-to-end tests for ``usersim report`` against the
    capability-dashboard artifacts. Single-file inputs skip the
    ``run=`` partition resolution and write artifacts directly into
    ``--out``; the ``--run`` test exercises a Hive-partitioned layout.
    """

    _ARTIFACTS = (
        "index.html",
        "report_manifest.json",
        "capability_matrix.json",
        "coverage_summary.json",
        "triage_queue.json",
    )

    def test_happy_path_writes_five_artifacts(
        self, tmp_path, trajectory_df, evaluator_df,
    ):
        """Trajectories + evaluations → all five artifacts land in ``--out``."""
        traj_path = tmp_path / "trajs.parquet"
        trajectory_df.to_parquet(traj_path, index=False)
        eval_path = tmp_path / "evals.parquet"
        evaluator_df.to_parquet(eval_path, index=False)
        out = tmp_path / "out"
        rc = cli.main([
            "report",
            "--trajectories", str(traj_path),
            "--evaluations", str(eval_path),
            "--out", str(out),
        ])
        assert rc == 0
        for name in self._ARTIFACTS:
            assert (out / name).exists(), f"missing artifact: {name}"
            assert (out / name).stat().st_size > 0, f"empty artifact: {name}"
        # Each JSON parses.
        for name in self._ARTIFACTS:
            if name.endswith(".json"):
                json.loads((out / name).read_text())
        # Manifest carries the run-level metadata keys the dashboard reads.
        manifest = json.loads((out / "report_manifest.json").read_text())
        assert "run_id" in manifest
        assert "model_id" in manifest
        assert "metadata_source" in manifest
        assert manifest["n_trajectories"] == len(trajectory_df)

    def test_missing_evaluations_writes_partial_report(
        self, tmp_path, trajectory_df,
    ):
        """``--evaluations`` is optional; capability cells render as
        ``not measured`` but the manifest + coverage still write."""
        traj_path = tmp_path / "trajs.parquet"
        trajectory_df.to_parquet(traj_path, index=False)
        out = tmp_path / "out"
        rc = cli.main([
            "report",
            "--trajectories", str(traj_path),
            "--out", str(out),
        ])
        assert rc == 0
        for name in self._ARTIFACTS:
            assert (out / name).exists(), f"missing artifact: {name}"
        manifest = json.loads((out / "report_manifest.json").read_text())
        assert manifest["n_trajectories"] == len(trajectory_df)
        # No eval frame → no rows survived the eval cell filter.
        assert manifest["n_evaluated"] == 0

    def test_capability_cells_span_multiple_locales(
        self, tmp_path, trajectory_df, evaluator_df,
    ):
        """Capability matrix has at least one cell per observed locale."""
        traj_path = tmp_path / "trajs.parquet"
        trajectory_df.to_parquet(traj_path, index=False)
        eval_path = tmp_path / "evals.parquet"
        evaluator_df.to_parquet(eval_path, index=False)
        out = tmp_path / "out"
        rc = cli.main([
            "report",
            "--trajectories", str(traj_path),
            "--evaluations", str(eval_path),
            "--out", str(out),
        ])
        assert rc == 0
        cells = json.loads((out / "capability_matrix.json").read_text())
        assert isinstance(cells, list) and cells
        observed_locales = {cell["locale"] for cell in cells}
        # The fixture spans en_US, pt_BR, ja_JP.
        assert {"en_US", "pt_BR", "ja_JP"}.issubset(observed_locales)

    def test_run_resolution_picks_latest_run(
        self, tmp_path, trajectory_df, evaluator_df,
    ):
        """Hive-partitioned layout: with multiple ``run=*`` partitions
        present, ``usersim report`` (no ``--run``) defaults to the
        latest run id; the artifacts land in ``--out/run=<id>/``."""
        from usersim.engine.core.storage import (
            run_subroot,
            write_partitioned_dataset,
        )

        traj_root = tmp_path / "trajectories"
        eval_root = tmp_path / "evaluations"
        # Two runs: an older "1000000000" and a newer "2000000000".
        for run_id in ("1000000000", "2000000000"):
            write_partitioned_dataset(
                trajectory_df, run_subroot(traj_root, run_id),
            )
            write_partitioned_dataset(
                evaluator_df, run_subroot(eval_root, run_id),
            )

        out = tmp_path / "out"
        rc = cli.main([
            "report",
            "--trajectories", str(traj_root),
            "--evaluations", str(eval_root),
            "--out", str(out),
        ])
        assert rc == 0
        # Artifacts land under the latest run subroot.
        run_dir = out / "run=2000000000"
        for name in self._ARTIFACTS:
            assert (run_dir / name).exists(), f"missing artifact: {name}"
        manifest = json.loads((run_dir / "report_manifest.json").read_text())
        assert manifest["run_id"] == "2000000000"


class TestConfigErrorPresentation:
    """``ConfigError`` reaches the user as a message, not a traceback.

    The entry points under ``cli/`` report user errors with
    ``SystemExit``, which bypasses ``main``'s ``try`` and prints without
    a traceback. ``cli/_pipeline.py`` cannot do that — the notebooks and
    the notebooks import it as a library, and a ``SystemExit`` there
    would kill the caller — so it raises instead, and ``main`` is the
    other half of the arrangement.
    """

    def test_main_presents_config_error_without_a_traceback(self, caplog):
        from usersim.cli._errors import ConfigError

        def _raises(args):
            raise ConfigError("toolset seed not found: /nope.parquet")

        with patch("usersim.cli._build_parser") as build:
            build.return_value.parse_args.return_value = SimpleNamespace(
                func=_raises, verbose=0,
            )
            with caplog.at_level(logging.ERROR):
                code = cli.main([])

        assert code == 2
        assert "toolset seed not found: /nope.parquet" in caplog.text

    def test_missing_toolset_seed_raises_config_error(self, tmp_path):
        """The real path: an explicit ``--toolset-seed-path`` that is not
        there is a user error, not a crash."""
        from usersim.cli._errors import ConfigError
        from usersim.cli._pipeline import resolve_toolset_seed_path

        with pytest.raises(ConfigError, match="toolset seed not found"):
            resolve_toolset_seed_path(
                toolset_seed_path=tmp_path / "absent.parquet",
            )


class TestReportErgonomics:
    """A rendered report must be findable without guessing.

    ``report`` used to print five paths identically -- four JSON cells and
    the one HTML file a person actually opens -- and neither ``simulate``
    nor ``eval`` said what to run next.
    """

    def _report_dir(self, tmp_path: Path) -> Path:
        import pandas as pd

        traj = tmp_path / "traj.parquet"
        pd.DataFrame([{
            "trajectory_id": "t1",
            "locale": "en_US",
            "probe_type": "general_open_ended",
            "conversation_messages": "[]",
            "simulation_outcome": "{}",
        }]).to_parquet(traj)
        return traj

    def test_index_html_is_distinguished_from_the_json_cells(
        self, tmp_path, capsys,
    ) -> None:
        traj = self._report_dir(tmp_path)
        out = tmp_path / "report"
        assert cli.main([
            "report", "--trajectories", str(traj), "--out", str(out),
        ]) == 0
        printed = capsys.readouterr().out
        assert "open this:" in printed
        # Clickable in most terminals, and unambiguous about which file.
        assert "file://" in printed and "index.html" in printed

    def test_open_is_opt_in(self, tmp_path) -> None:
        """CI and headless runs must never spawn a browser."""
        import argparse

        parser = cli._build_parser()
        report = parser._subparsers._group_actions[0].choices["report"]
        action = next(a for a in report._actions if a.dest == "open")
        assert action.default is False
        assert isinstance(action, argparse._StoreTrueAction)

    def test_open_flag_launches_the_browser(self, tmp_path, monkeypatch) -> None:
        opened: list[str] = []
        monkeypatch.setattr(
            "webbrowser.open", lambda url: opened.append(url) or True,
        )
        traj = self._report_dir(tmp_path)
        out = tmp_path / "report"
        cli.main([
            "report", "--trajectories", str(traj), "--out", str(out), "--open",
        ])
        assert opened and opened[0].startswith("file://")
        assert opened[0].endswith("index.html")

    def test_browser_failure_does_not_fail_the_run(
        self, tmp_path, monkeypatch,
    ) -> None:
        """The report is already written; a missing browser is not an error."""
        def boom(url):
            raise RuntimeError("no display")

        monkeypatch.setattr("webbrowser.open", boom)
        traj = self._report_dir(tmp_path)
        assert cli.main([
            "report", "--trajectories", str(traj),
            "--out", str(tmp_path / "r"), "--open",
        ]) == 0

    def test_list_runs_reports_nothing_to_render(self, tmp_path, capsys) -> None:
        empty = tmp_path / "trajectories"
        empty.mkdir()
        rc = cli.main([
            "report", "--trajectories", str(empty),
            "--out", str(tmp_path / "r"), "--list-runs",
        ])
        assert rc == 1
        assert "usersim simulate" in capsys.readouterr().out

    def test_list_runs_does_not_write_an_output_dir(self, tmp_path) -> None:
        """Asking what exists must not create anything."""
        empty = tmp_path / "trajectories"
        empty.mkdir()
        out = tmp_path / "should_not_exist"
        cli.main([
            "report", "--trajectories", str(empty),
            "--out", str(out), "--list-runs",
        ])
        assert not out.exists()
