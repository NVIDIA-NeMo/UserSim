# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `--toolset-seed-path` and the toolset-seed path resolver.

Scope: `cli/_pipeline.py`'s resolver, the CLI flag that overrides it, and
Data Designer's JSONL contract. No LLM calls; path-only cases use empty temp
files, while one valid JSONL fixture is read through Data Designer.

The resolver is worth pinning because its failure mode is quiet: a path
that stops existing surfaces as every `tool_calling` row aborting with
`cfg.tools_column not configured` — a probe-looking error with an
asset-layout cause.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from usersim import cli
from usersim.cli._errors import ConfigError
from usersim.cli._pipeline import (
    TOOLSET_SEED_SUFFIXES,
    resolve_toolset_seed_path,
)
from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.probes.tool_calling.generator import _normalize_tool_list


def _assets_root(tmp_path: Path, *seed_names: str) -> Path:
    """An `assets/` root containing `tool_calling/` and the named seeds."""
    probe_dir = tmp_path / "tool_calling"
    probe_dir.mkdir(parents=True)
    # A real tree carries each probe's bank, not just a seed file; the
    # pre-flight rejects one that holds only the seed. sov_ai_facts is
    # locale-scoped, so it needs the run's locale subtree.
    (probe_dir / "prompts.yaml").write_text("prompts: []\n", encoding="utf-8")
    facts = tmp_path / "sov_ai_facts" / "en_US"
    facts.mkdir(parents=True)
    (facts / "facts.yaml").write_text("facts: []\n", encoding="utf-8")
    for name in seed_names:
        (probe_dir / name).touch()
    return tmp_path


class TestResolveToolsetSeedPath:
    def test_parquet_is_found(self, tmp_path):
        root = _assets_root(tmp_path, "toolsets_seed.parquet")
        assert resolve_toolset_seed_path(root).name == "toolsets_seed.parquet"

    def test_jsonl_is_found(self, tmp_path):
        """jsonl is the hand-editable alternative to the shipped parquet."""
        root = _assets_root(tmp_path, "toolsets_seed.jsonl")
        assert resolve_toolset_seed_path(root).name == "toolsets_seed.jsonl"

    def test_both_present_raises(self, tmp_path):
        """Not a precedence question: silently preferring one would make an
        edit to the other look like it had no effect."""
        root = _assets_root(
            tmp_path,
            "toolsets_seed.parquet",
            "toolsets_seed.jsonl",
        )
        with pytest.raises(ConfigError) as e:
            resolve_toolset_seed_path(root)
        msg = str(e.value)
        assert "toolsets_seed.parquet" in msg and "toolsets_seed.jsonl" in msg

    def test_neither_present_returns_none(self, tmp_path):
        """Absence is a resolution result, not an error.

        It only matters for a mix containing tool_calling, which this
        low-level resolver cannot see; the mix-aware wrapper and wiring layer
        enforce that requirement.
        """
        assert resolve_toolset_seed_path(_assets_root(tmp_path)) is None

    def test_resolves_under_the_probe_asset_dir(self, tmp_path):
        """Composed via `probe_assets_dir`, not a string literal — the bug
        class that broke this before."""
        root = _assets_root(tmp_path, "toolsets_seed.parquet")
        assert resolve_toolset_seed_path(root).parent.name == "tool_calling"

    def test_explicit_path_wins(self, tmp_path):
        root = _assets_root(tmp_path, "toolsets_seed.parquet")
        custom = tmp_path / "custom.jsonl"
        custom.touch()
        assert resolve_toolset_seed_path(root, custom) == custom.resolve()

    def test_explicit_missing_path_raises(self, tmp_path):
        """You named a path; a run that silently ignores it is worse."""
        root = _assets_root(tmp_path, "toolsets_seed.parquet")
        with pytest.raises(ConfigError) as e:
            resolve_toolset_seed_path(root, tmp_path / "nope.jsonl")
        assert "toolset seed not found" in str(e.value)

    def test_explicit_unsupported_suffix_raises(self, tmp_path):
        seed = tmp_path / "toolsets.csv"
        seed.touch()
        with pytest.raises(ConfigError, match="unsupported toolset seed suffix"):
            resolve_toolset_seed_path(toolset_seed_path=seed)

    def test_default_candidate_must_be_a_file(self, tmp_path):
        root = _assets_root(tmp_path)
        (root / "tool_calling" / "toolsets_seed.parquet").mkdir()
        assert resolve_toolset_seed_path(root) is None

    def test_neither_argument_searches_the_asset_path(self):
        """Passing no root means "resolve normally", not "nothing to do".

        An unpinned run has no single root to hand down, so the seed is
        looked for along the same search path the loaders use.
        """
        resolved = resolve_toolset_seed_path()
        assert resolved is not None
        assert resolved.name.startswith("toolsets_seed")
        assert resolved.parent.name == "tool_calling"

    def test_supported_suffixes_are_declared(self):
        assert TOOLSET_SEED_SUFFIXES == (".parquet", ".jsonl")


class TestJsonlSeedContract:
    def test_data_designer_loads_nested_tools(self, tmp_path):
        """Exercise Data Designer's real JSONL reader, including its nested
        array representation at the probe boundary."""
        import data_designer.config as dd
        from data_designer.engine.resources.seed_reader import LocalFileSeedReader

        row = {
            "toolset_source": "custom",
            "toolset_category": "Weather",
            "toolset_name": "demo",
            "num_tools": 1,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                },
            ],
        }
        seed = tmp_path / "toolsets_seed.jsonl"
        seed.write_text(json.dumps(row) + "\n", encoding="utf-8")

        class _NoSecrets:
            def resolve(self, secret):
                return secret

        reader = LocalFileSeedReader()
        reader.attach(dd.LocalFileSeedSource(path=str(seed)), _NoSecrets())
        assert reader.get_seed_dataset_size() == 1
        frame = (
            reader.create_batch_reader(
                batch_size=1,
                index_range=None,
                shuffle=False,
            )
            .read_next_batch()
            .to_pandas()
        )

        assert frame.iloc[0]["toolset_name"] == "demo"
        tools = _normalize_tool_list(frame.iloc[0]["tools"])
        assert tools[0]["function"]["name"] == "get_weather"


class TestToolsetSeedFlag:
    def _parse(self, *extra):
        return cli._build_parser().parse_args(
            ["simulate", "--locale", "en_US", "--num-rows", "1", "--out", "/tmp/x", *extra]
        )

    def test_defaults_to_none(self):
        """None means 'resolve from --assets-dir', not 'no toolset'."""
        assert self._parse().toolset_seed_path is None

    def test_flag_is_parsed_as_a_path(self, tmp_path):
        seed = tmp_path / "custom.jsonl"
        args = self._parse("--toolset-seed-path", str(seed))
        assert args.toolset_seed_path == Path(str(seed))

    def test_help_renders(self, capsys):
        with pytest.raises(SystemExit):
            cli._build_parser().parse_args(["simulate", "--help"])
        assert "--toolset-seed-path" in capsys.readouterr().out


class TestDryRunResolvesTheSeed:
    """`--dry-run` must reach the same verdict as the real run.

    It is the plan-inspection command, so a dry run that prints a plan
    should be a dry run whose plan will execute. Both paths call the
    same `resolve_toolset_seed_for_mix` for exactly this reason — the dry
    run at `cli/simulate.py::run`, the real run at
    `_simulate_to_parquet` — and the dry run resolves before printing
    anything, so a bad config fails cleanly rather than interleaving
    with a half-written plan.
    """

    def _run(self, assets_dir, *extra, out="/tmp/x"):
        args = cli._build_parser().parse_args(
            [
                "simulate",
                "--locale",
                "en_US",
                "--num-rows",
                "1",
                "--out",
                out,
                "--assets-dir",
                str(assets_dir),
                "--dry-run",
                *extra,
            ]
            + ([] if "--probe-mix" in extra else ["--probe-mix", "tool_calling=1.0"])
        )
        return args.func(args)

    def test_prints_the_resolved_path(self, tmp_path, capsys):
        """Not '(default)' — which file a run will read is the question
        --dry-run exists to answer."""
        root = _assets_root(tmp_path, "toolsets_seed.parquet")
        assert self._run(root) == 0
        assert str(root / "tool_calling" / "toolsets_seed.parquet") in capsys.readouterr().out

    def test_no_seed_is_not_fatal(self, tmp_path, capsys):
        """A missing default must not block a run that needs no tools."""
        assert self._run(_assets_root(tmp_path), "--probe-mix", "sov_ai_facts=1.0") == 0
        assert "(not used)" in capsys.readouterr().out

    def test_non_tool_mix_does_not_inspect_ambiguous_seeds(self, tmp_path, capsys):
        root = _assets_root(
            tmp_path,
            "toolsets_seed.parquet",
            "toolsets_seed.jsonl",
        )
        assert self._run(root, "--probe-mix", "sov_ai_facts=1.0") == 0
        assert "(not used)" in capsys.readouterr().out

    def test_tool_mix_without_seed_fails_before_printing(self, tmp_path, capsys):
        with pytest.raises(ConfigError, match="no toolset seed"):
            self._run(_assets_root(tmp_path))
        assert capsys.readouterr().out == ""

    def test_non_tool_mix_rejects_explicit_override(self, tmp_path):
        root = _assets_root(tmp_path)
        custom = tmp_path / "custom.jsonl"
        custom.touch()
        with pytest.raises(ConfigError, match="does not include tool_calling"):
            self._run(
                root,
                "--probe-mix",
                "sov_ai_facts=1.0",
                "--toolset-seed-path",
                str(custom),
            )

    def test_raises_when_seeds_are_ambiguous(self, tmp_path):
        root = _assets_root(
            tmp_path,
            "toolsets_seed.parquet",
            "toolsets_seed.jsonl",
        )
        with pytest.raises(ConfigError) as e:
            self._run(root)
        assert "ambiguous" in str(e.value)

    def test_raises_when_explicit_seed_is_missing(self, tmp_path):
        root = _assets_root(tmp_path, "toolsets_seed.parquet")
        with pytest.raises(ConfigError) as e:
            self._run(root, "--toolset-seed-path", str(tmp_path / "nope.jsonl"))
        assert "toolset seed not found" in str(e.value)

    def test_explicit_seed_overrides_the_default(self, tmp_path, capsys):
        root = _assets_root(tmp_path, "toolsets_seed.parquet")
        custom = tmp_path / "custom.jsonl"
        custom.touch()
        assert self._run(root, "--toolset-seed-path", str(custom)) == 0
        assert "custom.jsonl" in capsys.readouterr().out


class TestPipelineRejectsToolCallingWithoutASeed:
    """The builder owns mix-dependent resolution and toolset wiring.

    It guards every caller —
    including the notebooks, which reach
    `build_simulator_config_builder` without touching `cli/simulate.py`.
    """

    def _build(self, mix, seed, assets_dir=None):
        from usersim.cli._models import default_models_path, load_models_config
        from usersim.cli._pipeline import build_simulator_config_builder

        return build_simulator_config_builder(
            locale="en_US",
            models=load_models_config(default_models_path()),
            assets_dir=Path(assets_dir).resolve() if assets_dir else packaged_assets_dir(),
            probe_mix=mix,
            toolset_seed_path=seed,
        )

    def test_raises_when_tool_calling_has_no_seed(self, tmp_path):
        """assets_dir with no seed in it — the builder resolves, finds
        nothing, and refuses rather than building a config whose every row
        will abort at probe construction."""
        (tmp_path / "tool_calling").mkdir()
        with pytest.raises(ConfigError) as e:
            self._build({"tool_calling": 1.0}, None, assets_dir=tmp_path)
        assert "tool_calling" in str(e.value)

    def test_resolves_from_assets_dir_when_no_seed_is_passed(self):
        """The point of the move: a caller that names an asset root and
        nothing else gets the shipped toolset found for it."""
        self._build({"tool_calling": 1.0}, None)

    def test_raises_when_the_seed_path_does_not_exist(self, tmp_path):
        with pytest.raises(ConfigError):
            self._build({"tool_calling": 1.0}, tmp_path / "nope.parquet")

    def test_other_probes_are_unaffected(self):
        """A mix that needs no tools must build without a toolset seed."""
        self._build({"sov_ai_facts": 1.0}, None)

    def test_other_probes_do_not_inspect_ambiguous_seeds(self, tmp_path):
        root = _assets_root(
            tmp_path,
            "toolsets_seed.parquet",
            "toolsets_seed.jsonl",
        )
        self._build(
            {"sov_ai_facts": 1.0},
            None,
            assets_dir=root,
        )

    def test_other_probes_reject_explicit_seed(self, tmp_path):
        seed = tmp_path / "custom.jsonl"
        seed.touch()
        with pytest.raises(ConfigError, match="does not include tool_calling"):
            self._build({"sov_ai_facts": 1.0}, seed)
