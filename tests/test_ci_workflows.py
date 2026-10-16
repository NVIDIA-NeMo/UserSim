# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invariants that keep CI safe for pull requests from forks.

None of these are visible in a passing workflow run: a file that leaks a
credential to fork-authored code, or an action pinned to a mutable tag, works
exactly as well as a correct one right up until it is exploited. They are
also easy to undo by accident, because the fix for "my workflow cannot see
the API key" is always to add the key.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.specifiers import SpecifierSet

_WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

#: Workflows that run automatically against code the author controls. Anyone
#: can open a pull request, so whatever these can reach, anyone can reach.
_UNTRUSTED_TRIGGERS = {"pull_request", "pull_request_target"}


def _workflows():
    return sorted(_WORKFLOWS.glob("*.yml"))


def _load(path: Path) -> dict:
    # `on:` is the YAML 1.1 boolean True, so it does not survive as a string.
    return yaml.safe_load(path.read_text())


def _triggers(doc: dict) -> set[str]:
    on = doc.get("on", doc.get(True))
    if isinstance(on, str):
        return {on}
    return set(on)


def _uses_clauses(doc: dict) -> list[str]:
    found = []
    for job in doc.get("jobs", {}).values():
        if "uses" in job:
            found.append(job["uses"])
        for step in job.get("steps") or []:
            if "uses" in step:
                found.append(step["uses"])
    return found


def test_there_are_workflows_to_check() -> None:
    """Guard against the whole suite below passing vacuously."""
    assert _workflows()


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
class TestForkSafety:
    def test_actions_are_pinned_to_a_full_sha(self, path: Path) -> None:
        """A tag is mutable, so a compromised upstream can rewrite it to point
        at anything. Only a full commit SHA is an actual pin."""
        for clause in _uses_clauses(_load(path)):
            ref = clause.rsplit("@", 1)[-1]
            assert len(ref) == 40 and all(c in "0123456789abcdef" for c in ref), (
                f"{path.name} uses {clause!r}, which is pinned to a mutable ref. Pin the full commit SHA and note the version in a comment."
            )

    def test_no_secret_is_readable_by_fork_authored_code(self, path: Path) -> None:
        """Credentials and untrusted code must not meet.

        `pull_request` runs the fork's code but gets no secrets, which is
        safe. `pull_request_target` gets secrets but must never check the
        fork's code out, which is safe only as long as it does not.
        """
        doc = _load(path)
        triggers = _triggers(doc)
        body = path.read_text()

        if "pull_request" in triggers:
            assert "secrets." not in body.replace("secrets.GITHUB_TOKEN", ""), (
                f"{path.name} runs on pull_request and references a secret. "
                f"Anyone can open a pull request, so that secret is public. "
                f"Move the step to a workflow triggered by schedule or "
                f"workflow_dispatch."
            )

        if "pull_request_target" in triggers:
            assert "actions/checkout" not in body, (
                f"{path.name} runs on pull_request_target, which holds a "
                f"writable token, and also checks code out. If that checkout "
                f"is the pull request head, the fork's code runs with that "
                f"token."
            )

    def test_piped_commands_cannot_swallow_a_failure(self, path: Path) -> None:
        """Bash reports the exit status of the *last* command in a pipeline,
        and Actions does not set pipefail by default. A check piped into
        `tee` therefore reports success however it went, which is worse than
        having no check at all."""
        doc = _load(path)
        default_shell = doc.get("defaults", {}).get("run", {}).get("shell", "")
        if "pipefail" in default_shell:
            return
        for job_name, job in doc.get("jobs", {}).items():
            for step in job.get("steps") or []:
                run = step.get("run", "")
                piped = [
                    line
                    for line in run.splitlines()
                    if "|" in line and not line.strip().startswith("#") and "||" not in line
                ]
                assert not piped or "pipefail" in run, (
                    f"{path.name} job {job_name!r} pipes a command without pipefail, so a failure in it is discarded. Set `defaults.run.shell` to `bash -euo pipefail {{0}}`."
                )

    def test_permissions_are_declared(self, path: Path) -> None:
        """Without an explicit block a workflow inherits the repository
        default, which may be write-all."""
        doc = _load(path)
        assert "permissions" in doc, (
            f"{path.name} declares no permissions, so it inherits the repository default. Declare the minimum it needs, or `{{}}` if it needs none."
        )


class TestCodeowners:
    """GitHub does not fail a push for naming a team that does not exist; it
    just requests review from nobody, which looks identical to a repository
    with no CODEOWNERS at all."""

    #: The teams that exist on the repository. Extend this when a team is
    #: created, not to make a test pass.
    KNOWN_TEAMS = {
        "@NVIDIA-NeMo/user_sim_maintainers",  # admin
        "@NVIDIA-NeMo/user_sim_team",  # write
    }

    def _rules(self) -> list[tuple[str, list[str]]]:
        path = _WORKFLOWS.parent / "CODEOWNERS"
        rules = []
        for line in path.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                pattern, *owners = line.split()
                rules.append((pattern, owners))
        return rules

    def test_every_owner_is_a_team_that_exists(self) -> None:
        for pattern, owners in self._rules():
            assert owners, f"CODEOWNERS pattern {pattern!r} names no owner."
            for owner in owners:
                assert owner in self.KNOWN_TEAMS, (
                    f"CODEOWNERS assigns {pattern!r} to {owner}, which is not a known team. GitHub would request review from nobody."
                )

    def test_there_is_a_default_owner(self) -> None:
        assert any(pattern == "*" for pattern, _ in self._rules()), (
            "CODEOWNERS has no `*` rule, so files outside the listed paths have no reviewer."
        )


class TestLiveChecksStayOutOfPullRequests:
    """The only workflow holding provider credentials."""

    def test_health_checks_is_not_triggered_by_pull_requests(self) -> None:
        triggers = _triggers(_load(_WORKFLOWS / "health-checks.yml"))
        assert not (triggers & _UNTRUSTED_TRIGGERS), (
            f"health-checks.yml holds the provider API keys and is triggered by {sorted(triggers & _UNTRUSTED_TRIGGERS)}."
        )
        assert "workflow_dispatch" in triggers, (
            "health-checks.yml must stay runnable on demand, or nobody will run it before a release."
        )


class TestCiWrapsMakeTargets:
    """CI that restates commands instead of calling the Makefile drifts from
    what contributors run locally, and the drift only shows up as a failure
    nobody can reproduce."""

    #: The jobs held to this, by workflow (None for every job). release.yml's
    #: release job is left out: it creates the GitHub release with a token only
    #: that job holds, which nothing local could reproduce.
    SCOPE = {"ci.yml": None, "release.yml": {"build"}}

    def _steps(self):
        for name, jobs in self.SCOPE.items():
            doc = _load(_WORKFLOWS / name)
            missing = (jobs or set()) - set(doc["jobs"])
            assert not missing, f"{name} has no job {sorted(missing)}, so nothing in it is checked."
            for job_name, job in doc["jobs"].items():
                if jobs is None or job_name in jobs:
                    for step in job["steps"]:
                        yield name, job_name, step

    def test_every_ci_step_is_a_make_target(self) -> None:
        for name, job_name, step in self._steps():
            if "uses" in step:
                continue
            run = step["run"].strip()
            assert run.startswith("make "), (
                f"{name} job {job_name!r} runs {run!r} directly. Add a make target and call that instead."
            )

    def test_the_targets_it_calls_exist(self) -> None:
        makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
        declared = {
            line.split(":")[0] for line in makefile.splitlines() if line and not line[0].isspace() and ":" in line
        }
        for name, _, step in self._steps():
            if "run" not in step:
                continue
            target = step["run"].strip().split()[1]
            assert target in declared, f"{name} calls `make {target}`, which the Makefile does not define."


class TestReleaseWorkflow:
    """The workflow that creates the release whose files are published. A
    mistake here does not fail a run; it publishes something nobody checked."""

    def _doc(self) -> dict:
        return _load(_WORKFLOWS / "release.yml")

    def test_only_a_version_tag_starts_it(self) -> None:
        doc = self._doc()
        on = doc.get("on", doc.get(True))
        assert on == {"push": {"tags": ["v*"]}}, (
            f"release.yml must run only when a v* tag is pushed, not on {on!r}: anything else could release a commit nobody tagged."
        )

    def test_only_the_job_behind_the_release_environment_can_write(self) -> None:
        doc = self._doc()
        assert doc["permissions"] == {"contents": "read"}, "release.yml must default to read-only."
        writers = []
        for name, job in doc["jobs"].items():
            permissions = job.get("permissions") or {}
            if permissions == "write-all" or (isinstance(permissions, dict) and "write" in permissions.values()):
                writers.append(name)
        assert writers, "No job in release.yml can write, so none can create the release."
        for name in writers:
            assert doc["jobs"][name].get("environment") == "release", (
                f"release.yml job {name!r} can write without a maintainer approving the release environment."
            )

    def test_the_tag_name_is_never_parsed_as_code(self) -> None:
        """A tag name may contain `$(shell ...)`, backticks or quotes. make expands a
        variable given on its command line, and a recipe that writes $(TAG) hands the
        name to the shell; only the environment passes it through untouched."""
        steps = [step for step in self._doc()["jobs"]["build"]["steps"] if "check-release-tag" in step.get("run", "")]
        assert len(steps) == 1, "release.yml's build job should run make check-release-tag exactly once."
        assert steps[0]["run"].strip() == "make check-release-tag", (
            "Pass TAG in the step's env, not on make's command line, which make expands."
        )
        assert steps[0].get("env", {}).get("TAG") == "${{ github.ref_name }}"
        makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
        recipe = makefile.split("\ncheck-release-tag:", 1)[1].split("\n\n", 1)[0]
        assert "$(TAG)" not in recipe, (
            "The check-release-tag recipe must read $$TAG; $(TAG) puts the tag name in the shell command."
        )

    def test_the_build_restores_no_cache(self) -> None:
        steps = self._doc()["jobs"]["build"]["steps"]
        setups = [step for step in steps if step.get("uses", "").startswith("astral-sh/setup-uv@")]
        assert setups and all(step.get("with", {}).get("enable-cache") is False for step in setups), (
            "release.yml's build must not restore a cache: one written by another run could reach the published files."
        )


class TestDcoAssistant:
    """A contributor signs by pasting a phrase, and nothing reports it when the
    paste does not count: the step is skipped and the job still passes."""

    def _step(self) -> dict:
        doc = _load(_WORKFLOWS / "dco-assistant.yml")
        steps = [
            step
            for job in doc["jobs"].values()
            for step in job.get("steps") or []
            if "contributor-assistant/github-action" in step.get("uses", "")
        ]
        assert len(steps) == 1, "dco-assistant.yml should run the DCO action in exactly one step."
        return steps[0]

    def test_the_sign_phrase_runs_the_step_with_surrounding_whitespace(self) -> None:
        """A phrase copied from the browser usually ends in a newline, which an
        exact comparison rejects. The action trims the comment before its own
        match, so the condition only has to let the comment through."""
        step = self._step()
        phrase = step["with"]["custom-pr-sign-comment"]
        assert f"contains(github.event.comment.body, '{phrase}')" in step["if"], (
            "The step condition must let the sign phrase through with surrounding whitespace."
        )
        assert "startsWith(github.event.comment.body, 'recheck')" in step["if"]
        assert "github.event.comment.body ==" not in step["if"], (
            "An exact comparison skips the step for a comment ending in a newline, and the job still passes."
        )

    def test_the_signed_commit_message_uses_only_substituted_placeholders(self) -> None:
        """The pinned action replaces $contributorName, $owner and $repo in this
        message and leaves any other placeholder in the commit subject."""
        message = self._step()["with"]["signed-commit-message"]
        unsubstituted = set(re.findall(r"\$\w+", message)) - {"$contributorName", "$owner", "$repo"}
        assert not unsubstituted, (
            f"signed-commit-message uses {sorted(unsubstituted)}, which the action never replaces."
        )


def test_native_matrix_matches_package_python_support() -> None:
    project = tomllib.loads((_WORKFLOWS.parents[1] / "pyproject.toml").read_text())["project"]
    supported = SpecifierSet(project["requires-python"])
    expected = [f"3.{minor}" for minor in range(30) if f"3.{minor}" in supported]
    matrix = _load(_WORKFLOWS / "ci.yml")["jobs"]["test"]["strategy"]["matrix"]
    assert matrix["python-version"] == expected


class TestMandatoryCiAggregate:
    def test_aggregate_waits_for_every_job_and_checks_explicit_results(self) -> None:
        import importlib.util

        script = _WORKFLOWS.parents[1] / "scripts" / "check_ci_results.py"
        spec = importlib.util.spec_from_file_location("check_ci_results", script)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        jobs = _load(_WORKFLOWS / "ci.yml")["jobs"]
        status = jobs["status"]
        assert set(jobs) - {"status"} == checker.REQUIRED_JOBS
        assert set(status["needs"]) == checker.REQUIRED_JOBS
        assert status["if"] == "always()"
        assert status["name"] == "CI Status"
        assert status["env"]["CI_JOB_RESULTS"] == "${{ toJSON(needs) }}"
        assert any(step.get("run") == "make check-ci-status" for step in status["steps"])

    def test_mandatory_jobs_cannot_be_optional_or_ignore_failures(self) -> None:
        jobs = _load(_WORKFLOWS / "ci.yml")["jobs"]
        for name, job in jobs.items():
            assert job.get("timeout-minutes", 0) > 0, name
            assert not job.get("continue-on-error"), name
            if name != "status":
                assert "if" not in job, name
            for step in job["steps"]:
                assert not step.get("continue-on-error"), (name, step)

    def test_native_matrix_workers_and_diagnostics_are_bounded(self) -> None:
        doc = _load(_WORKFLOWS / "ci.yml")
        assert doc["env"]["PYTEST_WORKERS"] == "2"
        assert doc["env"]["NEMO_TELEMETRY_ENABLED"] == "false"
        for name in ["test", "e2e", "container"]:
            assert any(
                step.get("if") in {"failure()", "always()"}
                and step.get("uses", "").startswith("actions/upload-artifact@")
                for step in doc["jobs"][name]["steps"]
            ), name
