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

from pathlib import Path

import pytest
import yaml

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

    def test_every_ci_step_is_a_make_target(self) -> None:
        doc = _load(_WORKFLOWS / "ci.yml")
        for job_name, job in doc["jobs"].items():
            for step in job["steps"]:
                if "uses" in step:
                    continue
                run = step["run"].strip()
                assert run.startswith("make "), (
                    f"ci.yml job {job_name!r} runs {run!r} directly. Add a make target and call that instead."
                )

    def test_the_targets_it_calls_exist(self) -> None:
        makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
        declared = {
            line.split(":")[0] for line in makefile.splitlines() if line and not line[0].isspace() and ":" in line
        }
        doc = _load(_WORKFLOWS / "ci.yml")
        for job in doc["jobs"].values():
            for step in job["steps"]:
                if "run" not in step:
                    continue
                target = step["run"].strip().split()[1]
                assert target in declared, f"ci.yml calls `make {target}`, which the Makefile does not define."
