# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guards for the security-sensitive defaults.

These assert the *default*, not the configurable behaviour: for each of these
settings the unsafe value is the one a caller gets without acting.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Generated dashboards: model-controlled text must not become script ──


def _render_dashboard(tmp_path: Path, traj_df, transcript: str | None = None) -> str:
    """Render the real capability dashboard through the production build path.

    When ``transcript`` is given it replaces the first row's assistant turn, so
    hostile input travels the same route a model-generated turn would.
    """
    from usersim.reporting import build_capability_report, write_capability_dashboard_artifacts

    if transcript is not None:
        # conversation_messages is a JSON-encoded list of role/content dicts.
        traj_df = traj_df.copy()
        msgs = json.loads(traj_df.at[0, "conversation_messages"])
        msgs[-1]["content"] = transcript
        traj_df.at[0, "conversation_messages"] = json.dumps(msgs)

    report = build_capability_report(
        traj_df,
        None,
        eval_column="assistant_eval",
        run_id="sec-test",
    )
    write_capability_dashboard_artifacts(report, tmp_path)
    return (tmp_path / "index.html").read_text()


class TestDashboardSanitisation:
    """Transcript text is model- and user-controlled. `marked` permits inline
    HTML by default, and the rendered result goes through
    dangerouslySetInnerHTML, so an unsanitised path makes a crafted turn
    executable script in a report someone opens or shares."""

    def test_sanitizer_is_wired_into_the_render_path(self, tmp_path, trajectory_df) -> None:
        html = _render_dashboard(tmp_path, trajectory_df)
        assert "DOMPurify" in html, "DOMPurify import missing"
        assert re.search(r"DOMPurify\.sanitize\(\s*marked\.parse\(", html), (
            "marked output must be sanitised before it reaches the DOM"
        )

    def test_no_unsanitised_marked_parse_reaches_the_dom(self, tmp_path, trajectory_df) -> None:
        html = _render_dashboard(tmp_path, trajectory_df)
        # Every marked.parse( must be wrapped by DOMPurify.sanitize(.
        for match in re.finditer(r"marked\.parse\(", html):
            prefix = html[max(0, match.start() - 40) : match.start()]
            assert "DOMPurify.sanitize(" in prefix, f"unwrapped marked.parse() at offset {match.start()}"

    def test_parse_failure_does_not_fall_back_to_raw_text(self, tmp_path, trajectory_df) -> None:
        # The original catch returned the raw string, which bypassed the
        # sanitizer entirely on any input that made marked throw.
        html = _render_dashboard(tmp_path, trajectory_df)
        assert "catch (_e) { return text; }" not in html.replace("  ", " ")

    def test_csp_present_and_restrictive(self, tmp_path, trajectory_df) -> None:
        html = _render_dashboard(tmp_path, trajectory_df)
        assert "Content-Security-Policy" in html
        for directive in ("object-src 'none'", "base-uri 'none'", "frame-ancestors 'none'"):
            assert directive in html, f"CSP missing {directive!r}"

    def test_hostile_payloads_are_embedded_inertly(self, tmp_path, trajectory_df) -> None:
        # The JSON payload must not be able to close its own <script> tag,
        # which is the one breakout the sanitizer cannot help with.
        payload = "</script><script>alert(1)</script>"
        html = _render_dashboard(tmp_path, trajectory_df, payload)
        body = html.split('type="application/json">', 1)[1]
        embedded = body.split("</script>", 1)[0]
        assert "</script" not in embedded


class TestFrontendPinning:
    """A generated report is a long-lived artifact. Floating major versions
    mean a hijacked or broken upstream release changes what an
    already-published report executes when someone opens it."""

    def test_esm_imports_are_exact_versions(self) -> None:
        from usersim.reporting import dashboard

        pinned = [
            dashboard._ESM_REACT,
            dashboard._ESM_REACT_DOM,
            dashboard._ESM_MARKED,
            dashboard._ESM_DOMPURIFY,
            dashboard._ESM_VEGA_EMBED,
        ]
        for url in pinned:
            assert re.search(r"@\d+\.\d+\.\d+", url), f"not an exact pin: {url}"

    def test_no_floating_pins_in_generated_html(self, tmp_path, trajectory_df) -> None:
        html = _render_dashboard(tmp_path, trajectory_df)
        floating = re.findall(r'https://esm\.sh/[a-z0-9-]+@\d+(?:/[a-z]+)?"', html)
        assert not floating, f"floating esm.sh pins: {floating}"


# ── LLM debug transcripts: opt-in, and never committable ──


class TestDebugTranscriptDefaults:
    """The debug log captures full prompts, completions, reasoning content,
    and tool calls. It defaulted to ON, writing to the caller's working
    directory."""

    @staticmethod
    def _fresh_import(expr: str) -> str:
        """Evaluate ``expr`` against a freshly imported llm module.

        A subprocess rather than importlib.reload(): the module owns
        threading state and an atexit flush, and reloading it in-process
        hangs the suite.
        """
        env = {k: v for k, v in os.environ.items() if k != "USERSIM_DEBUG_LOG"}
        proc = subprocess.run(
            [sys.executable, "-c", f"from usersim.engine.core import llm; print({expr})"],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def test_disabled_unless_explicitly_enabled(self) -> None:
        assert self._fresh_import("llm._DEBUG_LOG_ENABLED") == "False"

    def test_default_path_is_inside_the_ignored_output_tree(self) -> None:
        assert self._fresh_import("llm._DEFAULT_DEBUG_LOG_PATH.parts[0]") == "output"

    def test_enabling_via_env_var_works(self) -> None:
        env = dict(os.environ, USERSIM_DEBUG_LOG="/tmp/x.jsonl")
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from usersim.engine.core import llm; print(llm._DEBUG_LOG_ENABLED, llm._DEBUG_LOG_PATH)",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "True /tmp/x.jsonl"

    def test_transcript_names_are_gitignored_anywhere(self) -> None:
        patterns = (_REPO_ROOT / ".gitignore").read_text().splitlines()
        for name in ("debug_llm_responses.jsonl", "debug_litellm_requests.jsonl"):
            assert name in patterns, f"{name} must be ignored at any depth"

    def test_env_files_are_gitignored(self) -> None:
        patterns = (_REPO_ROOT / ".gitignore").read_text().splitlines()
        assert ".env" in patterns and ".env.*" in patterns


# ── Notebook credential handling ──


class TestNotebookCredentials:
    """NVIDIA_API_KEY (inference) is unrelated to NGC auth. The simulate
    notebook copied one into the other, which tests/test_setup_ngc.py already
    forbids in the library path -- but notebooks had no lint or test gate."""

    def test_simulate_notebook_does_not_mirror_the_inference_key(self) -> None:
        nb = _REPO_ROOT / "notebooks" / "01_simulate.ipynb"
        if not nb.is_file():
            pytest.fail(f"{nb} is missing; this guard must not skip silently")
        source = json.dumps(json.load(nb.open()))
        for forbidden in (
            'os.environ.setdefault(\\"NGC_CLI_API_KEY\\", nvidia_key)',
            'os.environ.setdefault(\\"NGC_API_KEY\\", nvidia_key)',
        ):
            assert forbidden not in source, "the notebook must not populate NGC credentials from NVIDIA_API_KEY"

    def test_no_notebook_mirrors_the_inference_key(self) -> None:
        for nb in sorted((_REPO_ROOT / "notebooks").glob("*.ipynb")):
            source = json.dumps(json.load(nb.open()))
            assert 'NGC_CLI_API_KEY\\", nvidia_key' not in source, nb.name


class TestProductNaming:
    """The display name in emitted artifacts must not drift.

    These strings ship inside a partner's report HTML and dataset cards,
    so a wrong one is invisible here and obvious there. Nothing asserted
    on them before, which is how a brand sweep could quietly rewrite a
    title and pass.

    The split is deliberate: ``NeMo UserSim`` is the product name a reader
    sees; ``usersim`` is the technical identifier (CLI, imports, env vars,
    entry-point groups) and must stay lowercase and unprefixed.
    """

    DISPLAY = "NeMo UserSim"

    def _source(self, dotted: str) -> str:
        import importlib
        import inspect

        return inspect.getsource(importlib.import_module(dotted))

    def test_report_titles_carry_the_display_name(self) -> None:
        for module in (
            "usersim.reporting.dashboard",
            "usersim.reporting.capability_report",
            "usersim.reporting.comparison_dashboard",
        ):
            source = self._source(module)
            titles = [line for line in source.splitlines() if "<title>" in line or "<h1>" in line]
            assert titles, f"{module} renders no title or heading"
            assert any(self.DISPLAY in line for line in titles), (
                f"{module} renders a title without {self.DISPLAY!r}: {titles}"
            )

    def test_dataset_card_attributes_the_product(self) -> None:
        assert self.DISPLAY in self._source("usersim.selection.io")

    def test_cli_banner_carries_the_display_name(self) -> None:
        import usersim.cli

        assert self.DISPLAY in usersim.cli._build_parser().description

    def test_no_prior_brand_survives_in_emitted_strings(self) -> None:
        for module in (
            "usersim.reporting.dashboard",
            "usersim.reporting.capability_report",
            "usersim.reporting.comparison_dashboard",
            "usersim.selection.io",
            "usersim.cli",
        ):
            source = self._source(module)
            for stale in ("NeMo-Sim", "nemo-sim", "NEMO-SIM"):
                assert stale not in source, f"{module} still contains {stale!r}"

    def test_technical_identifiers_stay_unprefixed(self) -> None:
        """The CLI, import root, and env prefix are not the product name."""
        import usersim.cli
        from usersim.engine.core._extensions import (
            DISABLE_ENV_VAR,
            ENTRY_POINT_GROUPS,
        )

        assert usersim.cli._build_parser().prog == "usersim"
        assert DISABLE_ENV_VAR == "USERSIM_DISABLE_EXTENSIONS"
        assert all(g.startswith("usersim.") for g in ENTRY_POINT_GROUPS)


class TestDocumentedCounts:
    """README counts must match the registries they describe.

    These had all drifted: the README advertised 12 probes and 10 scorers
    against 13 and 11 actually registered. A reader checks the headline
    number, not the code, so a stale count is a false claim about what
    ships -- and nothing was watching it.
    """

    @staticmethod
    def _readme() -> str:
        """README plus the guides it delegates to.

        The counts are split across files: the headline lives in the README,
        the prose that repeats it lives in the guide for that topic. Both
        have to agree with the registry, so both are searched.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        parts = [root / "README.md", root / "docs" / "probes.md", root / "docs" / "outputs.md"]
        joined = "\n".join(p.read_text() for p in parts if p.is_file())
        # Collapse whitespace: these assertions are about the count being
        # right, not about where prose happens to wrap.
        return " ".join(joined.split())

    @staticmethod
    def _registries() -> tuple[int, int, int]:
        import usersim.engine.generator  # noqa: F401  (populates the registry)
        from usersim.engine.core.probes import known_probes
        from usersim.engine.evaluator.scorers import list_scorers

        probes = known_probes()
        non_general = [p for p in probes if not p.startswith("general")]
        return len(probes), len(list_scorers()), len(non_general)

    #: Spelled-out forms the prose uses.
    WORDS = {
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        9: "nine",
        10: "ten",
    }

    def test_probe_count_matches_the_registry(self) -> None:
        probes, _, _ = self._registries()
        readme = self._readme()
        assert f"**{probes} probes ·" in readme, f"README headline does not say {probes} probes"
        word = self.WORDS[probes].capitalize()
        assert f"{word} probes ship today" in readme, f"README prose does not say '{word} probes ship today'"

    def test_scorer_count_matches_the_registry(self) -> None:
        _, scorers, _ = self._registries()
        readme = self._readme()
        assert f"· {scorers} evaluator scorers ·" in readme
        assert f"{self.WORDS[scorers]} ship today" in readme

    def test_non_general_probe_count_matches(self) -> None:
        _, _, non_general = self._registries()
        assert f"The {self.WORDS[non_general]} non-`general` probes" in self._readme()

    def test_no_superseded_counts_remain(self) -> None:
        """The specific wrong numbers this test was written to catch."""
        readme = self._readme()
        for stale in ("**12 probes ·", "· 10 evaluator scorers ·", "Twelve probes ship today", "2500+ tests"):
            assert stale not in readme, f"stale count survives: {stale!r}"
