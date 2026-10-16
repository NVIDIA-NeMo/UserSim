# Local and CI validation

CI calls repository Mise tasks (Make targets remain compatibility wrappers) so failures can be reproduced locally. After
`make install-dev`, run `make validate-ci` to execute the native validation
bundle, including distribution checks and installed CLI tests.

| Target | Checks |
| --- | --- |
| `make check-all` | Static checks, including the configured type-check scope. |
| `make check-dependency-licenses` | Dependency license policy. |
| `make coverage` | Ordinary test suite with an 84% minimum total coverage. |
| `make smoke` | Offline configuration, registry, asset, and reporting self-check. |
| `make check-extensions` | Out-of-tree installed extension behavior. |
| `make validate-native` | The five targets above. |
| `make check-dist` | Build and validate archives, then check a clean wheel installation. |
| `make test-e2e` | Installed CLI workflow against a loopback provider in isolated environments. |
| `make validate-ci` | `validate-native` and `check-dist`, followed by e2e against the resulting wheel. |

The distribution archives are produced by `check-dist`; extension checks build
a separate temporary wheel. CI also runs the test suite across Python 3.11
through 3.14; a local bundle checks the active
Python environment. Provider health checks run separately and need credentials.

The separate `Installed CLI e2e` job runs the [installed workflow](e2e.md) on
Python 3.12 through `scripts/mise run test:e2e`. It retains failure diagnostics under
`test-results/e2e/` as `e2e-failure` for 14 days. The local bundle passes
`--wheel-dir dist` after archive validation so the e2e test installs that same
wheel without rebuilding.

## Type checking

`make typecheck` uses the locked `ty==0.0.69` against `src/usersim`, `scripts`,
and `tests_e2e`, targeting Python 3.11 with `src` as the module root. The
dedicated CI job runs `scripts/mise run check:type` after locked setup; hooks,
static validation, and release validation inherit it through the `check` task.
Ordinary unit-test fixtures are outside this initial scope.

Keep annotations aligned with runtime behavior. Rule-specific suppressions
cover documented runtime schema construction or upstream annotation limits;
do not disable a rule across the repository.

## Coverage

The Python 3.12 CI matrix job runs `scripts/mise run test:ci`; the other Python
versions run `scripts/mise run test`. Both test paths use the same ordinary suite. Coverage failures
fail the existing `Tests (Python 3.12)` check. The 84% floor lives in the shared pytest coordinator
and applies to local runs as well as CI.

```bash
make coverage PYTEST_WORKERS=2
make coverage PYTEST_ARGS="--durations=25 --junitxml=junit.xml"
make coverage COV_XML=coverage-custom.xml
```

`coverage.xml` is the default XML output. `COV_XML` changes its location, and
`PYTEST_ARGS` passes additional pytest options through to the coverage command.
Create a custom output directory before writing there. A focused pytest
selection also limits measured execution; run the complete suite to reproduce
the CI coverage gate. `COV_FAIL_UNDER` is configurable for experiments, but
maintainers must not lower the committed floor to make a change pass.

CI uploads `coverage.xml` as `coverage-python-3.12` for 14 days, even when the
coverage command fails. Missing XML fails the upload step, and uploading a
report does not suppress a preceding test or coverage failure. If collection or
setup stops before XML is produced, inspect the failed step's logs.

## Mandatory jobs and aggregate status

`CI Status` waits for static checks, the dedicated `Type checking` job, the complete Python test matrix, packaging,
installed CLI e2e, and CPU container tests. Every job must report `success`.
Failures, cancellation, missing results, malformed results, and skipped jobs fail
the aggregate; there are no intentional job skips. Conditional coverage steps
inside the test matrix do not skip their enclosing job.

The aggregate runs with `always()` and invokes `python3 scripts/check_ci_results.py` with
`CI_JOB_RESULTS` set to GitHub's `toJSON(needs)` object. Its result checker uses
only Python's standard library, so it can report dependency or tool-bootstrap
failures in other jobs. The DCO job likewise invokes its shared Python checker
directly. Both checkers remain available as local Mise tasks. When adding a mandatory job, update both the workflow's
`needs` list and the explicit registry in `scripts/check_ci_results.py`; workflow
tests reject an omitted dependency or unregistered job.

CI bounds native pytest workers to two and sets job timeouts. Failed native
suites upload JUnit XML; e2e failures upload their available artifacts; the container job always uploads
its available diagnostics, including successful reports. If setup fails before an artifact exists, the failed step's GitHub
logs remain the diagnostic record. Offline test jobs disable telemetry and use
no provider credentials. The container job runs tests without external network
access; its dependency download occurs during image construction.

A local native bundle runs in one Python environment; GitHub additionally
orchestrates the supported-version matrix and the CPU container. Run
`make test-container` separately to reproduce the container gate.

## Reruns, cancellation, and stacked PRs

For a transient infrastructure failure on the current revision, open its `CI`
run in GitHub Actions and choose **Re-run all jobs**. **Re-run failed jobs** is
also suitable when the successful jobs remain valid, but confirm that the failed
mandatory jobs and `CI Status` execute again and finish successfully. Rerunning
only `CI Status` cannot repair a failed dependency. If a partial rerun leaves the
aggregate stale or fails to rerun a dependency, rerun all jobs. See GitHub's
[rerun instructions](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

PR runs share a concurrency group by PR number. A newer PR run cancels an older
in-progress run; use the newest run for the current revision and leave the
superseded one cancelled. If the current run is cancelled without a replacement,
rerun all its jobs after confirming no newer run is pending. Cancellation may
prevent even the `always()` aggregate from completing; a cancelled or absent
aggregate is not passing evidence.

Each stacked PR targets its immediate parent branch. When a parent changes or
merges, check the child's base and incremental diff, update its branch as needed,
and rerun affected local Make targets. Recheck the base after GitHub retargets a
child automatically. A base edit alone is not a trigger for this workflow's
default `pull_request` events. If no fresh PR run appears after retargeting,
close and reopen the unmerged PR to trigger one. Verify that the resulting checks
cover the current head and base before merging. An old run's
[rerun uses its original SHA and ref](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs);
it does not validate a new head or merge base. A manual branch dispatch also
does not replace validation of the PR's test merge commit.

Recheck required policies and approvals after every stack update. Approval
dismissal and approval of the latest push depend on the repository's active
[review protection settings](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
Request renewed review when approvals are dismissed or the reviewed diff changes;
do not assume that passing CI preserves an earlier approval. This guide does not
verify or change those settings.

## Required-check migration for maintainers

This change does not edit repository rules. First verify that all mandatory jobs
and `CI Status` complete successfully on representative PRs. Keep current
required checks until that evidence exists. Then prepare a ruleset change that
replaces only the status contexts produced by `ci.yml` with the exact observed
`CI Status` context. Check the actual check-run name in GitHub before saving the
rule; do not guess a workflow/name prefix.

Preserve the separately required `dco` and
`semantic-pull-request / semantic-pull-request` policy contexts. Trailer
validation introduced by the contribution-policy change must also remain
required according to that change's checklist. These independent policies are
not members of this workflow's aggregate. Retest a successful PR and a PR with
a deliberately failing mandatory check after the authorized ruleset update.

See [Mise tools and tasks](mise.md) for canonical commands, tool ownership,
bootstrap prerequisites, and the continuing Make compatibility interface.
