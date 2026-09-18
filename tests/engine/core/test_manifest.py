# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``usersim.engine.core.manifest``.

Covers:

- Round-trip ``RunManifest`` through ``to_dict`` / ``from_dict`` / JSON.
- ``write_run_manifest`` writes the canonical path with the v1 schema
  version stamped.
- Atomic-write contract: a second concurrent writer noops; the first
  writer's content is preserved.
- ``load_run_manifest`` returns ``None`` for runs without a manifest
  (back-compat for old-run handling).
- ``_resolve_models`` extracts provider, endpoint, and api_key_env_var
  for built-in providers (without ever pulling the resolved key value).
- ``_aggregate_from_trajectories`` rolls up per-row provenance into the
  manifest's ``asset_versions`` and ``outcome_summary`` blocks.
- Manifest is credential-free — defensive grep for ``sk-...`` /
  ``nvapi-...``-shaped strings in the serialized form.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from usersim.engine.core.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    AssetVersionsInfo,
    CodeInfo,
    ModelIdentity,
    OutcomeSummary,
    ReplayInfo,
    RunManifest,
    ScopeInfo,
    SimulationConfigSnapshot,
    SourceInfo,
    _aggregate_from_trajectories,
    _file_content_hash,
    _resolve_models,
    build_run_manifest,
    load_run_manifest,
    manifest_path,
    write_run_manifest,
)


# ─── Test fixtures: lightweight stand-ins for cli/_models.py types ────────


@dataclass(frozen=True)
class _StubModelSpec:
    """Mirror of ``cli._models.ModelSpec`` with the fields the manifest reads."""

    alias: str
    model: str
    provider: str
    max_tokens: Optional[int] = None
    max_parallel_requests: int = 4
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    timeout: Optional[float] = None
    extra_body: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class _StubProviderSpec:
    """Mirror of ``cli._models.ProviderSpec``."""

    name: str
    endpoint: str
    provider_type: str
    api_key: str  # env var NAME, never the resolved key


@dataclass(frozen=True)
class _StubModelsConfig:
    providers: tuple = ()
    models: tuple = ()


@dataclass
class _StubSimConfig:
    """Mirror of ``ConversationSimulatorConfig`` fields the snapshot reads."""

    max_turns: int = 5
    max_steps: int = 10
    max_tools: int = 5
    max_query_attempts: int = 3
    incremental_disclosure_ratio: float = 0.6
    persona_grounding_ratio: float = 1.0
    context_compression: bool = True
    compression_window: int = 1
    random_seed: Optional[int] = None
    store_reasoning: bool = True


# ─── Round-trip ──────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_manifest_to_dict_and_from_dict_roundtrip(self) -> None:
        m = RunManifest(
            manifest_schema_version=MANIFEST_SCHEMA_VERSION,
            run_id="1700000000",
            started_at_epoch=1700000000,
            finished_at_epoch=1700001000,
            total_runtime_s=1000.0,
            source=SourceInfo(
                kind="cli",
                invocation="usersim simulate --locale en_US --num-rows 4",
                hostname="host.local",
                platform="darwin-25.2.0",
                python_version="3.13.5",
            ),
            code=CodeInfo(
                code_sha="abc123def",
                conversation_plugin_version="0.4.2",
                data_designer_version="0.5.7",
            ),
            models_config_path="cli/models_default.toml",
            models={
                "assistant_model": ModelIdentity(
                    alias="assistant_model",
                    model="nvidia/nemotron-3-super-v3",
                    provider="nvidia",
                    endpoint="https://integrate.api.nvidia.com/v1",
                    api_key_env_var="NVIDIA_API_KEY",
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=8192,
                    max_parallel_requests=8,
                    timeout=300.0,
                    extra_body={"reasoning_effort": "high"},
                ),
            },
            simulation_config=SimulationConfigSnapshot(
                max_turns=5,
                max_steps=10,
                max_tools=5,
                max_query_attempts=3,
                max_assistant_attempts=1,
                incremental_disclosure_ratio=0.6,
                persona_grounding_ratio=1.0,
                context_compression=True,
                compression_window=1,
                random_seed=42,
                store_reasoning=False,
            ),
            scope=ScopeInfo(
                locales=["en_US", "ja_JP"],
                probes=["tool_calling", "general_open_ended"],
                probe_mix={"tool_calling": 0.5, "general_open_ended": 0.5},
                tool_calling_themes=["weather", "calendar"],
                trajectories_requested={"en_US": 2, "ja_JP": 2},
                trajectories_completed={"en_US": 2, "ja_JP": 2},
            ),
            asset_versions=AssetVersionsInfo(
                assets_dir="/repo/assets",
                nemotron_personas_version="v1.0",
                bank_versions={"sov_ai_facts:en_US": "2025.04.01"},
                prompt_versions={"tool_calling": "v1.0"},
                probe_variants_exercised={"safety_chat_pressure": ["default"]},
            ),
            outcome_summary=OutcomeSummary(
                trajectories_total=4,
                status_counts={"ok": 4},
                failure_class_histogram={},
                coverage_per_locale_probe={
                    "en_US:tool_calling": 1,
                    "en_US:general_open_ended": 1,
                    "ja_JP:tool_calling": 1,
                    "ja_JP:general_open_ended": 1,
                },
            ),
            replay=ReplayInfo(),
        )

        d = m.to_dict()
        # Must be JSON-serializable.
        json.dumps(d, ensure_ascii=False)
        m2 = RunManifest.from_dict(d)
        assert m2 == m
        assert m2.simulation_config.store_reasoning is False

    def test_from_dict_tolerates_missing_optional_blocks(self) -> None:
        partial = {
            "manifest_schema_version": "v1",
            "run_id": "1700000000",
            "started_at_epoch": 1700000000,
            "models": {},
        }
        m = RunManifest.from_dict(partial)
        assert m.run_id == "1700000000"
        assert m.models == {}
        assert m.replay == ReplayInfo()
        assert m.simulation_config.max_turns == 5  # default
        assert m.simulation_config.store_reasoning is True

    def test_schema_version_is_v1(self) -> None:
        assert MANIFEST_SCHEMA_VERSION == "v1"


# ─── Writer + reader ─────────────────────────────────────────────────────


def _minimal_manifest(run_id: str = "1700000000") -> RunManifest:
    return RunManifest(
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        started_at_epoch=int(run_id) if run_id.isdigit() else 0,
        finished_at_epoch=None,
        total_runtime_s=None,
        source=SourceInfo(
            kind="cli", invocation="x", hostname="h", platform="p", python_version="3.13"
        ),
        code=CodeInfo(),
        models_config_path=None,
        models={},
        simulation_config=SimulationConfigSnapshot(
            max_turns=5,
            max_steps=10,
            max_tools=5,
            max_query_attempts=3,
            max_assistant_attempts=1,
            incremental_disclosure_ratio=0.6,
            persona_grounding_ratio=1.0,
            context_compression=True,
            compression_window=1,
        ),
        scope=ScopeInfo(locales=[], probes=[], probe_mix={}),
        asset_versions=AssetVersionsInfo(),
        outcome_summary=OutcomeSummary(),
        replay=ReplayInfo(),
    )


class TestWriter:
    def test_write_creates_canonical_path(self, tmp_path: Path) -> None:
        m = _minimal_manifest("1700000000")
        path = write_run_manifest(m, tmp_path)
        assert path == tmp_path / "run=1700000000" / MANIFEST_FILENAME
        assert path.exists()

    def test_written_payload_contains_schema_version(self, tmp_path: Path) -> None:
        m = _minimal_manifest("1700000000")
        path = write_run_manifest(m, tmp_path)
        loaded = json.loads(path.read_text())
        assert loaded["manifest_schema_version"] == "v1"
        assert loaded["run_id"] == "1700000000"

    def test_second_write_is_noop(self, tmp_path: Path) -> None:
        """First-writer-wins: a second call must not clobber existing content."""
        m1 = _minimal_manifest("1700000000")
        m2 = _minimal_manifest("1700000000")
        # Make m2's content different so we can verify m1's content survives.
        # We can't mutate frozen dataclasses; instead, write m1, manually
        # rewrite the file with sentinel content, then call write again
        # with m2 and confirm sentinel is preserved.
        path = write_run_manifest(m1, tmp_path)
        sentinel = '{"manifest_schema_version": "v1", "run_id": "SENTINEL"}'
        path.write_text(sentinel)
        write_run_manifest(m2, tmp_path)
        assert path.read_text() == sentinel

    def test_no_temp_files_left_behind(self, tmp_path: Path) -> None:
        """The temp staging file must be cleaned up regardless of outcome."""
        m = _minimal_manifest("1700000000")
        write_run_manifest(m, tmp_path)
        run_dir = tmp_path / "run=1700000000"
        leftover = [p for p in run_dir.iterdir() if p.name.startswith(".")]
        assert leftover == []


class TestLoader:
    def test_loads_what_writer_wrote(self, tmp_path: Path) -> None:
        m = _minimal_manifest("1700000000")
        write_run_manifest(m, tmp_path)
        loaded = load_run_manifest("1700000000", tmp_path)
        assert loaded == m

    def test_returns_none_for_run_without_manifest(self, tmp_path: Path) -> None:
        # No manifest written. Old-run handling depends on this contract.
        assert load_run_manifest("1700000000", tmp_path) is None

    def test_returns_none_for_malformed_json(self, tmp_path: Path) -> None:
        path = manifest_path(tmp_path, "1700000000")
        path.parent.mkdir(parents=True)
        path.write_text("{ this is not valid json")
        assert load_run_manifest("1700000000", tmp_path) is None

    def test_returns_none_for_non_object_json(self, tmp_path: Path) -> None:
        path = manifest_path(tmp_path, "1700000000")
        path.parent.mkdir(parents=True)
        path.write_text("[1, 2, 3]")
        assert load_run_manifest("1700000000", tmp_path) is None


# ─── Model resolution ────────────────────────────────────────────────────


class TestResolveModels:
    def test_resolves_builtin_provider_endpoint_and_env_var(self) -> None:
        specs = (
            _StubModelSpec(
                alias="assistant_model",
                model="openai/gpt-oss-120b",
                provider="nvidia",
                temperature=1.0,
                top_p=1.0,
                max_tokens=8192,
                max_parallel_requests=8,
                timeout=300.0,
                extra_body={"reasoning_effort": "high"},
            ),
        )
        out = _resolve_models(model_specs=specs, provider_specs=())
        identity = out["assistant_model"]
        assert identity.model == "openai/gpt-oss-120b"
        assert identity.provider == "nvidia"
        assert identity.endpoint == "https://integrate.api.nvidia.com/v1"
        assert identity.api_key_env_var == "NVIDIA_API_KEY"
        assert identity.extra_body == {"reasoning_effort": "high"}

    def test_resolves_custom_provider(self) -> None:
        provider = _StubProviderSpec(
            name="gateway",
            endpoint="https://gateway.example/v1",
            provider_type="openai",
            api_key="GATEWAY_API_KEY",
        )
        spec = _StubModelSpec(
            alias="user_model",
            model="openai/openai/gpt-5.5",
            provider="gateway",
        )
        out = _resolve_models(model_specs=(spec,), provider_specs=(provider,))
        identity = out["user_model"]
        assert identity.provider == "gateway"
        assert identity.endpoint == "https://gateway.example/v1"
        assert identity.api_key_env_var == "GATEWAY_API_KEY"

    def test_unknown_provider_yields_none_endpoint(self) -> None:
        spec = _StubModelSpec(
            alias="x",
            model="m",
            provider="some-unregistered-provider",
        )
        out = _resolve_models(model_specs=(spec,), provider_specs=())
        identity = out["x"]
        assert identity.endpoint is None
        assert identity.api_key_env_var is None


# ─── Trajectory aggregation ──────────────────────────────────────────────


class TestAggregateTrajectories:
    def _row(
        self,
        *,
        locale: str,
        probe_family: str,
        probe_variant: str = "default",
        status: str = "ok",
        failure_class: Optional[str] = None,
        scenario_prompt_version: str = "v1.0",
        bank_version: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        outcome = {
            "status": status,
            "failure_class": failure_class,
            "provenance": {
                "scenario_prompt_version": scenario_prompt_version,
                "bank_version": bank_version or {},
            },
        }
        return {
            "locale": locale,
            "probe_family": probe_family,
            "probe_variant": probe_variant,
            "simulation_outcome": json.dumps(outcome),
        }

    def test_empty_frame_returns_zero_aggregates(self) -> None:
        df = pd.DataFrame()
        out = _aggregate_from_trajectories(df)
        assert out["trajectories_total"] == 0
        assert out["status_counts"] == {}
        assert out["bank_versions"] == {}

    def test_aggregates_status_and_failure_class(self) -> None:
        df = pd.DataFrame([
            self._row(locale="en_US", probe_family="tool_calling", status="ok"),
            self._row(
                locale="en_US",
                probe_family="tool_calling",
                status="failed",
                failure_class="user_query_gate_exhausted",
            ),
        ])
        out = _aggregate_from_trajectories(df)
        assert out["trajectories_total"] == 2
        assert out["status_counts"] == {"failed": 1, "ok": 1}
        assert out["failure_class_histogram"] == {"user_query_gate_exhausted": 1}

    def test_aggregates_per_probe_prompt_versions(self) -> None:
        df = pd.DataFrame([
            self._row(
                locale="en_US",
                probe_family="tool_calling",
                scenario_prompt_version="v1.0",
            ),
            self._row(
                locale="ja_JP",
                probe_family="general_open_ended",
                scenario_prompt_version="v1.2",
            ),
        ])
        out = _aggregate_from_trajectories(df)
        assert out["prompt_versions"] == {
            "general_open_ended": "v1.2",
            "tool_calling": "v1.0",
        }

    def test_aggregates_bank_versions_across_locales(self) -> None:
        df = pd.DataFrame([
            self._row(
                locale="en_US",
                probe_family="sov_ai_facts",
                bank_version={"sov_ai_facts:en_US": "2025.04.01"},
            ),
            self._row(
                locale="ja_JP",
                probe_family="sov_ai_facts",
                bank_version={"sov_ai_facts:ja_JP": "2025.04.10"},
            ),
        ])
        out = _aggregate_from_trajectories(df)
        assert out["bank_versions"] == {
            "sov_ai_facts:en_US": "2025.04.01",
            "sov_ai_facts:ja_JP": "2025.04.10",
        }

    def test_aggregates_probe_variants_exercised(self) -> None:
        df = pd.DataFrame([
            self._row(
                locale="en_US",
                probe_family="safety_chat_pressure",
                probe_variant="default",
            ),
            self._row(
                locale="en_US",
                probe_family="safety_chat_pressure",
                probe_variant="consequence_framing",
            ),
        ])
        out = _aggregate_from_trajectories(df)
        assert out["probe_variants_exercised"] == {
            "safety_chat_pressure": ["consequence_framing", "default"]
        }

    def test_coverage_per_locale_probe(self) -> None:
        df = pd.DataFrame([
            self._row(locale="en_US", probe_family="tool_calling"),
            self._row(locale="en_US", probe_family="tool_calling"),
            self._row(locale="ja_JP", probe_family="tool_calling"),
        ])
        out = _aggregate_from_trajectories(df)
        assert out["coverage_per_locale_probe"] == {
            "en_US:tool_calling": 2,
            "ja_JP:tool_calling": 1,
        }
        assert out["trajectories_completed"] == {"en_US": 2, "ja_JP": 1}

    def test_decodes_outcome_when_already_dict(self) -> None:
        # simulation_outcome may arrive as dict (notebook prototyping) or
        # as JSON string (post-DD). Both must aggregate.
        df = pd.DataFrame([
            {
                "locale": "en_US",
                "probe_family": "tool_calling",
                "probe_variant": "default",
                "simulation_outcome": {"status": "ok", "provenance": {}},
            },
        ])
        out = _aggregate_from_trajectories(df)
        assert out["status_counts"] == {"ok": 1}


# ─── build_run_manifest end-to-end ───────────────────────────────────────


class TestBuildRunManifest:
    def test_assembles_all_blocks(self) -> None:
        models_config = _StubModelsConfig(
            providers=(
                _StubProviderSpec(
                    name="gateway",
                    endpoint="https://gateway.example/v1",
                    provider_type="openai",
                    api_key="GATEWAY_API_KEY",
                ),
            ),
            models=(
                _StubModelSpec(
                    alias="assistant_model",
                    model="nvidia/nemotron-3-super-v3",
                    provider="gateway",
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=8192,
                    max_parallel_requests=8,
                    timeout=300.0,
                ),
            ),
        )
        traj = pd.DataFrame([
            {
                "locale": "en_US",
                "probe_family": "tool_calling",
                "probe_variant": "default",
                "simulation_outcome": json.dumps({
                    "status": "ok",
                    "provenance": {
                        "scenario_prompt_version": "v1.0",
                        "bank_version": {},
                    },
                }),
            },
        ])
        manifest = build_run_manifest(
            run_id="1700000001",
            started_at_epoch=1700000001,
            finished_at_epoch=1700000050,
            models_config=models_config,
            models_config_path=Path("cli/models_default.toml"),
            sim_config=_StubSimConfig(random_seed=42),
            locales_requested=["en_US"],
            probe_mix={"tool_calling": 1.0},
            tool_calling_themes=("weather",),
            assets_dir=Path("/repo/assets"),
            trajectories_requested={"en_US": 1},
            trajectory_df=traj,
            source_kind="cli",
            source_invocation="usersim simulate --locale en_US --num-rows 1",
        )
        assert manifest.manifest_schema_version == "v1"
        assert manifest.run_id == "1700000001"
        assert manifest.total_runtime_s == 49.0
        assert manifest.simulation_config.random_seed == 42
        assert manifest.scope.probe_mix == {"tool_calling": 1.0}
        assert manifest.scope.trajectories_completed == {"en_US": 1}
        assert manifest.outcome_summary.status_counts == {"ok": 1}
        assert manifest.asset_versions.prompt_versions == {"tool_calling": "v1.0"}
        identity = manifest.models["assistant_model"]
        assert identity.model == "nvidia/nemotron-3-super-v3"
        assert identity.endpoint == "https://gateway.example/v1"
        assert identity.api_key_env_var == "GATEWAY_API_KEY"
        assert manifest.replay == ReplayInfo()  # reserved fields default null

    def test_empty_trajectory_df_renders_zeros(self) -> None:
        manifest = build_run_manifest(
            run_id="1700000002",
            started_at_epoch=1700000002,
            finished_at_epoch=1700000003,
            models_config=_StubModelsConfig(),
            sim_config=_StubSimConfig(),
            locales_requested=["en_US"],
            probe_mix={},
            trajectories_requested={"en_US": 0},
            trajectory_df=None,
        )
        assert manifest.outcome_summary.trajectories_total == 0
        assert manifest.scope.trajectories_completed == {}


# ─── No-credential-leak guard ────────────────────────────────────────────


class TestNoCredentialLeak:
    def test_serialized_manifest_has_no_credential_shaped_strings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defensive grep — fail if any field's serialized form looks like
        a resolved API key (sk-... / nvapi-...).

        Since the writer takes only a typed ``RunManifest``, this is
        belt-and-suspenders: if a future change ever piped a credential
        through the builder, it would land in ``api_key_env_var`` or
        ``extra_body`` and this test would catch it.
        """
        # Set a credential-shaped env var to make sure it doesn't sneak in
        # via env detection elsewhere.
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-FAKE_TEST_KEY_DO_NOT_LEAK")

        models_config = _StubModelsConfig(
            providers=(),
            models=(
                _StubModelSpec(
                    alias="assistant_model",
                    model="openai/gpt-oss-120b",
                    provider="nvidia",
                ),
            ),
        )
        manifest = build_run_manifest(
            run_id="1700000003",
            started_at_epoch=1700000003,
            finished_at_epoch=1700000004,
            models_config=models_config,
            sim_config=_StubSimConfig(),
            locales_requested=["en_US"],
            probe_mix={"tool_calling": 1.0},
            trajectories_requested={"en_US": 0},
        )
        write_run_manifest(manifest, tmp_path)
        path = manifest_path(tmp_path, "1700000003")
        content = path.read_text()

        # Patterns mirror common provider key prefixes.
        forbidden = [
            re.compile(r"sk-[a-zA-Z0-9]{20,}"),
            re.compile(r"nvapi-[a-zA-Z0-9_-]{10,}"),
        ]
        for pattern in forbidden:
            match = pattern.search(content)
            assert match is None, (
                f"manifest leaked credential-looking string: {match.group(0)!r} "
                f"in {path}"
            )
        # The ENV VAR NAME, however, should be present.
        assert "NVIDIA_API_KEY" in content


# ─── File content hash ───────────────────────────────────────────────────


class TestFileContentHash:
    def test_returns_none_for_missing_path(self) -> None:
        assert _file_content_hash(None) is None

    def test_returns_none_for_nonexistent_file(self, tmp_path: Path) -> None:
        assert _file_content_hash(tmp_path / "nope.parquet") is None

    def test_hashes_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "a.bin"
        p.write_bytes(b"hello")
        h = _file_content_hash(p)
        assert h is not None
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64

    def test_hash_is_deterministic_for_same_content(self, tmp_path: Path) -> None:
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        assert _file_content_hash(a) == _file_content_hash(b)
