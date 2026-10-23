# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The linked-issue check, run step by step against a fake `gh`.

The workflow closes pull requests, and a mistake in it only shows on someone
else's: it closes one it should have kept, or keeps one it should have closed.
Each test runs a step's script the way the runner does, with the workflow's
default shell, then reads what the step wrote to GITHUB_OUTPUT and which `gh`
calls it made.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "pr-linked-issue.yml"
_REPO = "NVIDIA-NeMo/UserSim"
_CLOSE = "Close new or reopened PR without an open, triaged issue"

#: Fake `gh` responses, as shell snippets.
_NOT_FOUND = 'echo "gh: Not Found (HTTP 404)" >&2; exit 1'
_DELETED = 'echo "gh: This issue was deleted (HTTP 410)" >&2; exit 1'
_REDIRECTED = 'echo "gh: Moved Permanently (HTTP 301)" >&2; exit 1'
_UNAVAILABLE = 'echo "gh: Service Unavailable (HTTP 503)" >&2; exit 1'


def _issue(state: str = "open", labels: tuple[str, ...] = ("triaged",), repo: str = _REPO, **fields: object) -> str:
    """A fake `gh api` answer for an issue in `repo`."""
    issue = {
        "state": state,
        "labels": [{"name": label} for label in labels],
        "repository_url": f"https://api.github.com/repos/{repo}",
    }
    return f"echo '{json.dumps(issue | fields)}'"


_TRIAGED = _issue()
_UNTRIAGED = _issue(labels=("bug",))
_CLOSED = _issue(state="closed")
_A_PULL_REQUEST = _issue(pull_request={})
#: What `gh api` returns after following the redirect for an issue that moved.
_MOVED = _issue(repo="example-org/another-repo")


def _doc() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text())


def _step(name: str, job: str = "check") -> dict:
    steps = [step for step in _doc()["jobs"][job]["steps"] if step["name"] == name]
    assert len(steps) == 1, f"pr-linked-issue.yml job {job!r} should have exactly one step named {name!r}."
    return steps[0]


@dataclass
class _Run:
    returncode: int
    stdout: str
    stderr: str
    outputs: dict[str, str]
    calls: list[str]
    temp: Path


def _run(tmp_path: Path, name: str, gh: str, job: str = "check", **env: str) -> _Run:
    """Run one step's script as the runner does: with the workflow's default
    shell, every variable the step declares set (empty unless given), and a
    fake `gh` first on PATH that logs its arguments, then runs `gh`."""
    step = _step(name, job)
    declared = set(step.get("env", {}))
    assert set(env) <= declared, f"{name!r} does not declare {sorted(set(env) - declared)}."

    bin_dir, temp = tmp_path / "bin", tmp_path / "runner_temp"
    bin_dir.mkdir(parents=True)
    temp.mkdir()
    log, output, script = tmp_path / "gh.log", tmp_path / "output", tmp_path / "step.sh"
    fake = bin_dir / "gh"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{log}"\n{gh}\n')
    fake.chmod(0o755)
    output.write_text("")
    script.write_text(step["run"])

    shell = [str(script) if part == "{0}" else part for part in _doc()["defaults"]["run"]["shell"].split()]
    runner = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(output),
        "RUNNER_TEMP": str(temp),
    }
    result = subprocess.run(
        shell, env={key: "" for key in declared} | env | runner, capture_output=True, text=True, check=False
    )
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    calls = log.read_text().splitlines() if log.exists() else []
    return _Run(result.returncode, result.stdout, result.stderr, outputs, calls, temp)


class TestAuthor:
    @pytest.mark.parametrize(
        ("gh", "expected"),
        [
            ("echo admin", "true"),
            ("echo write", "true"),  # the Maintain role reads as write
            ("echo read", "false"),  # and Triage as read
            (_NOT_FOUND, "false"),
        ],
        ids=["admin", "write", "read", "no-access"],
    )
    def test_authors_with_write_access_skip_the_check(self, tmp_path: Path, gh: str, expected: str) -> None:
        run = _run(tmp_path, "Check author permissions", gh, PR_AUTHOR="example-user", REPO=_REPO)
        assert run.returncode == 0, run.stderr
        assert run.outputs["is_collaborator"] == expected
        assert run.calls == [f"api repos/{_REPO}/collaborators/example-user/permission --jq .permission"]

    def test_a_failed_lookup_stops_the_run(self, tmp_path: Path) -> None:
        """Reading an outage as "no access" would close a maintainer's pull request."""
        run = _run(tmp_path, "Check author permissions", _UNAVAILABLE, PR_AUTHOR="example-user", REPO=_REPO)
        assert run.returncode != 0
        assert run.outputs == {}

    def test_dependabot_is_the_only_bot_that_skips_the_check(self, tmp_path: Path) -> None:
        assert re.findall(r"[\w-]+\[bot\]", _step("Check author permissions")["run"]) == ["dependabot[bot]"]
        dependabot = _run(tmp_path / "dependabot", "Check author permissions", "exit 1", PR_AUTHOR="dependabot[bot]")
        assert dependabot.outputs == {"is_collaborator": "true"}
        assert dependabot.calls == []
        other = _run(tmp_path / "other", "Check author permissions", _NOT_FOUND, PR_AUTHOR="renovate[bot]", REPO=_REPO)
        assert other.outputs["is_collaborator"] == "false"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Fixes #123", "123"),
        ("closes #7", "7"),
        ("RESOLVES   #42", "42"),
        ("Fixes #12, then Closes #34", "12"),
        ("Refs #5", ""),
        ("", ""),
    ],
    ids=["fixes", "lower-case", "upper-case", "first-only", "other-keyword", "empty"],
)
def test_the_first_closing_reference_in_the_current_body_counts(tmp_path: Path, body: str, expected: str) -> None:
    (tmp_path / "body.txt").write_text(body)
    run = _run(
        tmp_path, "Parse issue reference from PR body", f'cat "{tmp_path / "body.txt"}"', PR_NUMBER="42", REPO=_REPO
    )
    assert run.returncode == 0, run.stderr
    assert run.outputs == {"issue_num": expected}
    assert run.calls[0].startswith(f"pr view 42 --repo {_REPO} --json body")


class TestIssue:
    @pytest.mark.parametrize(
        ("gh", "exists", "is_open", "triaged"),
        [
            (_TRIAGED, "true", "true", "true"),
            (_UNTRIAGED, "true", "true", "false"),
            (_CLOSED, "true", "false", "false"),
            (_A_PULL_REQUEST, "false", "false", "false"),
            (_NOT_FOUND, "false", "false", "false"),
            (_DELETED, "false", "false", "false"),
            (_REDIRECTED, "false", "false", "false"),
            (_MOVED, "false", "false", "false"),
        ],
        ids=["triaged", "untriaged", "closed", "pull-request", "missing", "deleted", "redirected", "moved"],
    )
    def test_only_an_open_triaged_issue_validates(
        self, tmp_path: Path, gh: str, exists: str, is_open: str, triaged: str
    ) -> None:
        """A deleted or moved issue counts as missing. Read as an outage, it
        would keep the pull request open, and a moved one would be judged by
        another repository's labels."""
        run = _run(tmp_path, "Validate issue is open and triaged", gh, ISSUE_NUM="123", REPO=_REPO)
        assert run.returncode == 0, run.stderr
        assert run.outputs == {"issue_exists": exists, "is_open": is_open, "is_triaged": triaged}
        assert run.calls == [f"api repos/{_REPO}/issues/123"]

    def test_a_failed_lookup_stops_the_run(self, tmp_path: Path) -> None:
        """Only an answer that the issue is not here may count against the pull
        request; an outage must not."""
        run = _run(tmp_path, "Validate issue is open and triaged", _UNAVAILABLE, ISSUE_NUM="123", REPO=_REPO)
        assert run.returncode != 0
        assert run.outputs == {}


class TestComment:
    STEP = "Build comment body and post result"
    FAILURES = {
        "no-issue": {"ISSUE_NUM": ""},
        "missing": {"ISSUE_NUM": "123", "ISSUE_EXISTS": "false", "ISSUE_OPEN": "false", "IS_TRIAGED": "false"},
        "closed": {"ISSUE_NUM": "123", "ISSUE_EXISTS": "true", "ISSUE_OPEN": "false", "IS_TRIAGED": "false"},
        "untriaged": {"ISSUE_NUM": "123", "ISSUE_EXISTS": "true", "ISSUE_OPEN": "true", "IS_TRIAGED": "false"},
    }

    @staticmethod
    def _gh(earlier: str) -> str:
        """Lists `earlier` as this check's earlier comment, and gives a new comment id 88."""
        return f'case "$*" in\n  "api --paginate"*) {earlier} ;;\n  *"/comments -f body="*) echo 88 ;;\nesac'

    @pytest.mark.parametrize("reason", sorted(FAILURES))
    def test_a_failure_posts_one_comment_that_explains_it(self, tmp_path: Path, reason: str) -> None:
        failure = self.FAILURES[reason]
        run = _run(
            tmp_path, self.STEP, self._gh("true"), IS_COLLABORATOR="false", PR_NUMBER="42", REPO=_REPO, **failure
        )
        assert run.returncode == 0, run.stderr
        assert run.outputs == {"status": "fail", "comment_id": "88"}
        assert (
            len([call for call in run.calls if call.startswith(f"api repos/{_REPO}/issues/42/comments -f body=")]) == 1
        )

        body = (run.temp / "comment-body.md").read_text()
        assert body.startswith("<!-- linked-issue-check -->\n")
        assert "`triaged`" in body
        assert "${" not in body
        assert "reopen this pull request" not in body.lower(), (
            "Its author cannot reopen a pull request this check closed."
        )
        if reason == "no-issue":
            assert all(f"`{keyword} #<issue-number>`" in body for keyword in ("Fixes", "Closes", "Resolves"))
            assert f"https://github.com/{_REPO}/issues/new/choose" in body
        else:
            assert "#123 " in body

    def test_a_later_failure_updates_the_earlier_comment(self, tmp_path: Path) -> None:
        failure = self.FAILURES["untriaged"]
        run = _run(
            tmp_path, self.STEP, self._gh("echo 77"), IS_COLLABORATOR="false", PR_NUMBER="42", REPO=_REPO, **failure
        )
        assert run.returncode == 0, run.stderr
        assert run.outputs == {"status": "fail", "comment_id": "77"}
        assert any(
            call.startswith(f"api --silent -X PATCH repos/{_REPO}/issues/comments/77 -f body=") for call in run.calls
        )
        assert not any("/issues/42/comments -f body=" in call for call in run.calls)

    @pytest.mark.parametrize(
        "env",
        [
            {"IS_COLLABORATOR": "true"},
            {
                "IS_COLLABORATOR": "false",
                "ISSUE_NUM": "123",
                "ISSUE_EXISTS": "true",
                "ISSUE_OPEN": "true",
                "IS_TRIAGED": "true",
            },
        ],
        ids=["write-access", "triaged-issue"],
    )
    def test_a_pass_removes_the_earlier_comment(self, tmp_path: Path, env: dict[str, str]) -> None:
        run = _run(tmp_path, self.STEP, self._gh("echo 77"), PR_NUMBER="42", REPO=_REPO, **env)
        assert run.returncode == 0, run.stderr
        assert run.outputs == {"status": "pass"}
        assert f"api -X DELETE repos/{_REPO}/issues/comments/77" in run.calls


class TestClose:
    @staticmethod
    def _gh(tmp_path: Path, state: str, labels: list[str], body: str, issue: str) -> str:
        """Answers `gh pr view` with the pull request's current state, and an issue lookup with `issue`."""
        pr = tmp_path / "pr.json"
        pr.write_text(json.dumps({"state": state, "labels": [{"name": label} for label in labels], "body": body}))
        return f'case "$1 $2" in\n  "pr view") cat "{pr}" ;;\n  "api repos/"*) {issue} ;;\nesac'

    def test_only_new_or_reopened_pull_requests_are_closed(self) -> None:
        """An edit or a push only fails the check. A pull request opened before
        the check existed, or kept open by a maintainer, stays open."""
        condition = _step(_CLOSE)["if"]
        assert "steps.comment.outputs.status == 'fail'" in condition
        assert set(re.findall(r"github\.event\.action == '(\w+)'", condition)) == {"opened", "reopened"}
        assert "github.run_attempt == '1' || github.triggering_actor != 'github-actions[bot]'" in condition
        closers = [
            step["name"]
            for job in _doc()["jobs"].values()
            for step in job["steps"]
            if "pr close" in step.get("run", "")
        ]
        assert closers == [_CLOSE]
        assert "steps.close.outputs.current_valid != 'true'" in _step("Set check result")["if"]

    @pytest.mark.parametrize(
        ("state", "labels", "body", "issue", "closes"),
        [
            ("OPEN", [], "", "true", True),
            ("OPEN", [], "Fixes #123", _UNTRIAGED, True),
            ("OPEN", [], "Fixes #123", _NOT_FOUND, True),
            ("OPEN", [], "Fixes #123", _DELETED, True),
            ("OPEN", [], "Fixes #123", _MOVED, True),
            ("OPEN", [], "Fixes #123", _A_PULL_REQUEST, True),
            ("OPEN", ["bug", "keep-open"], "", "true", False),
            ("CLOSED", [], "", "true", False),
        ],
        ids=["no-issue", "untriaged", "missing", "deleted", "moved", "pull-request", "keep-open", "already-closed"],
    )
    def test_it_decides_on_the_current_pull_request(
        self, tmp_path: Path, state: str, labels: list[str], body: str, issue: str, closes: bool
    ) -> None:
        gh = self._gh(tmp_path, state, labels, body, issue)
        run = _run(tmp_path, _CLOSE, gh, PR_NUMBER="42", REPO=_REPO, COMMENT_ID="77")
        assert run.returncode == 0, run.stderr
        assert (f"pr close 42 --repo {_REPO}" in run.calls) == closes

    def test_a_body_corrected_while_queued_passes(self, tmp_path: Path) -> None:
        """The opened event can carry a body the author has since fixed."""
        gh = self._gh(tmp_path, "OPEN", [], "Fixes #123", _TRIAGED)
        run = _run(tmp_path, _CLOSE, gh, PR_NUMBER="42", REPO=_REPO, COMMENT_ID="77")
        assert run.returncode == 0, run.stderr
        assert run.outputs == {"current_valid": "true"}
        assert f"api -X DELETE repos/{_REPO}/issues/comments/77" in run.calls
        assert not any(call.startswith("pr close") for call in run.calls)

    def test_a_failed_issue_lookup_never_closes(self, tmp_path: Path) -> None:
        gh = self._gh(tmp_path, "OPEN", [], "Fixes #123", _UNAVAILABLE)
        run = _run(tmp_path, _CLOSE, gh, PR_NUMBER="42", REPO=_REPO, COMMENT_ID="77")
        assert run.returncode != 0
        assert not any(call.startswith("pr close") for call in run.calls)


class TestRetrigger:
    def test_it_finds_pull_requests_whose_first_reference_is_the_issue(self, tmp_path: Path) -> None:
        """It must read a body the way the check does, or an issue change
        re-checks the wrong pull requests."""
        prs = tmp_path / "prs.json"
        prs.write_text(
            json.dumps(
                [
                    {"number": 1, "body": "Fixes #12"},
                    {"number": 2, "body": "Closes #34, then fixes #12"},
                    {"number": 3, "body": "Resolves #123"},
                    {"number": 4, "body": None},
                    {"number": 5, "body": "RESOLVES #12"},
                ]
            )
        )
        run = _run(
            tmp_path, "Find PRs referencing this issue", f'cat "{prs}"', "retrigger", ISSUE_NUMBER="12", REPO=_REPO
        )
        assert run.returncode == 0, run.stderr
        assert run.outputs["prs"].split() == ["1", "5"]

    @pytest.mark.parametrize(
        "body",
        [
            "Fixes #12",
            "fixes\t#12",
            "Fixes #012",
            "Fixes #0",
            "Fixes\n#12",
            "Fixes\n#3, then Fixes #12",
            "Closes #3, then fixes #12",
        ],
        ids=["plain", "tab", "zero-padded", "zero", "line-break", "line-break-first", "later-reference"],
    )
    def test_it_reads_a_body_exactly_as_the_check_does(self, tmp_path: Path, body: str) -> None:
        """An issue change re-checks a pull request exactly when the check reads
        that issue from its description."""
        (tmp_path / "body.txt").write_text(body)
        check = _run(
            tmp_path / "check",
            "Parse issue reference from PR body",
            f'cat "{tmp_path / "body.txt"}"',
            PR_NUMBER="42",
            REPO=_REPO,
        )
        prs = tmp_path / "prs.json"
        prs.write_text(json.dumps([{"number": 42, "body": body}]))
        retrigger = _run(
            tmp_path / "retrigger",
            "Find PRs referencing this issue",
            f'cat "{prs}"',
            "retrigger",
            ISSUE_NUMBER="12",
            REPO=_REPO,
        )
        assert check.returncode == 0, check.stderr
        assert retrigger.returncode == 0, retrigger.stderr
        reads_issue_12 = check.outputs["issue_num"] != "" and int(check.outputs["issue_num"]) == 12
        assert (retrigger.outputs["prs"].split() == ["42"]) == reads_issue_12

    @pytest.mark.parametrize(
        ("runs", "rerun", "says"),
        [
            ("echo 555", "true", ", so the linked issue check is running again."),
            ("echo 555", "exit 1", ". Push a commit or edit the description to run the linked issue check again."),
            ("true", "true", ". Push a commit or edit the description to run the linked issue check again."),
        ],
        ids=["re-run", "re-run-refused", "no-earlier-run"],
    )
    def test_it_reruns_the_latest_check_and_says_why(self, tmp_path: Path, runs: str, rerun: str, says: str) -> None:
        """A run GitHub will not repeat, such as one started over 30 days ago,
        must not stop the pull requests after it from being re-checked."""
        step = "Re-run the linked issue check"
        assert f"--workflow {_WORKFLOW.name}" in _step(step, "retrigger")["run"]
        gh = f'case "$1 $2" in\n  "pr view") echo abc123 ;;\n  "run list") {runs} ;;\n  "run rerun") {rerun} ;;\nesac'
        run = _run(
            tmp_path, step, gh, "retrigger", ISSUE_ACTION="labeled", ISSUE_NUMBER="12", PR_NUMBERS="7 8 ", REPO=_REPO
        )
        assert run.returncode == 0, run.stderr
        assert run.calls.count(f"run rerun 555 --repo {_REPO}") == (2 if runs == "echo 555" else 0)
        assert [call for call in run.calls if call.startswith("pr comment")] == [
            f"pr comment {number} --repo {_REPO} --body Issue #12 received the `triaged` label{says}"
            for number in (7, 8)
        ]
        assert ("::warning::" in run.stdout) == (rerun != "true" or runs == "true")


class TestWorkflow:
    def test_the_required_check_keeps_its_name(self) -> None:
        """A ruleset requires a check by name. Renamed, it never reports, and
        every pull request waits for it."""
        assert _doc()["jobs"]["check"]["name"] == "Linked issue"

    def test_only_the_job_that_reruns_checks_can_write_to_actions(self) -> None:
        """The check handles what a contributor wrote, so its token gets only
        what commenting on and closing a pull request needs."""
        doc = _doc()
        assert doc["permissions"] == {}, "Each job declares its own permissions."
        assert "actions" not in doc["jobs"]["check"]["permissions"]
        assert doc["jobs"]["retrigger"]["permissions"]["actions"] == "write"

    def test_pull_requests_into_every_branch_are_checked(self) -> None:
        doc = _doc()
        trigger = doc.get("on", doc.get(True))["pull_request_target"]
        assert not {"branches", "branches-ignore"} & set(trigger), (
            "A branch filter lets a pull request into any other branch skip the check."
        )
        assert {"opened", "reopened", "edited", "synchronize"} <= set(trigger["types"])

    def test_it_reads_no_secret(self) -> None:
        assert "secrets." not in _WORKFLOW.read_text()

    def test_scripts_read_event_fields_only_through_env(self) -> None:
        """An expression inside `run:` is pasted into the script before it runs,
        so a crafted title or branch name would run with the write token."""
        for job_name, job in _doc()["jobs"].items():
            for step in job["steps"]:
                assert "${{" not in step.get("run", ""), (
                    f"{job_name} step {step['name']!r} puts an expression in its script."
                )

    def test_the_pull_request_template_links_no_issue(self) -> None:
        """The template lands in every description, hidden comments included, so
        an example number in it would link that issue for everyone who keeps it."""
        template = (_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text()
        assert not re.search(r"(fixes|closes|resolves)\s+#[0-9]", template, re.IGNORECASE)

    def test_no_issue_form_applies_triaged(self) -> None:
        """A form applies its labels for whoever opens the issue, so one that
        applied `triaged` would let any pull request through."""
        forms = [
            path for path in sorted((_ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.y*ml")) if path.name != "config.yml"
        ]
        assert forms
        for form in forms:
            labels = yaml.safe_load(form.read_text()).get("labels") or []
            if isinstance(labels, str):
                labels = [label.strip() for label in labels.split(",")]
            assert "triaged" not in labels, f"{form.name} applies `triaged` to every issue opened with it."
