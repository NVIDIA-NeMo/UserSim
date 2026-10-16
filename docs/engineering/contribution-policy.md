# Contribution checks and merge policy

Every commit submitted in a pull request needs a `Signed-off-by: Name <email>`
trailer matching its author name and email exactly. Each `Co-authored-by`
identity also needs its own matching sign-off. Do not add another person's
certification without their agreement. Bots follow the same trailer rule;
there is no bot allowlist in the trailer checker.

The DCO Assistant separately records one-time contributor agreement. Its
existing `dependabot[bot]` agreement exemption does not exempt bot commits from
trailers. A comment to the Assistant cannot repair an unsigned commit.
Cryptographic signatures are independent of DCO trailers and are not required.

## Run checks locally

`make install-dev` installs both Git hooks, including when only the static hook
was installed previously. To reinstall them directly:

```bash
scripts/mise exec -- uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Normal commits use `git commit -s`. The commit-message hook invokes
`scripts/check_commit_signoffs.py` with Git's author identity. The PR workflow
uses the same checker against stored commit authors, independently of whether
local hooks were installed. Git parses the final trailer block; text mentioning
a sign-off in an earlier paragraph does not count.

To reproduce the PR check, supply the actual parent base and head SHAs:

```bash
python3 scripts/check_commit_signoffs.py --base "$(git rev-parse origin/main)" --head "$(git rev-parse HEAD)"
```

For a stacked PR, replace `origin/main` with its parent branch. The checker
validates every commit reachable from the head but not the base, including
merge commits. An empty range or incomplete history fails. Shallow repositories
are rejected even when both endpoint SHAs are present; run `git fetch --unshallow`
before checking a range. Fetch the parent first if it is missing locally.
Retargeting a stacked PR reruns the trailer workflow against its new base, even
when the head commit is unchanged.

If the latest commit is yours and its author matches your Git identity:

```bash
git commit --amend --no-edit -s
```

For several commits authored by you, rebase onto the actual PR parent:

```bash
git rebase --signoff origin/main
```

Review the rewritten commits with the checker before updating a published
branch. Coordinate history changes with collaborators. If author and committer
differ, `-s` adds the committer's identity; obtain the author's certification
and add that exact trailer as well. Preserve co-author trailers and their
certifications during rewrites.

## Maintainer settings checklist

The new workflow publishes the separate `DCO trailers` job status. Adding this
file does not configure repository rules. Before treating trailer validation
as enforced, maintainers must:

- Add `DCO trailers` as a required check for `main`, using a successful workflow
  run to identify the exact status context.
- Retain the existing DCO Assistant and PR-title requirements. Migrating checks
  from `ci.yml` to its aggregate status must not remove these separate checks.
- Review bot-generated PRs and configure their authorship/sign-offs to satisfy
  the same rule before making the new status required.
- Verify CODEOWNERS review requirements and the required checks in repository
  rules after each status migration.

Squash-only merging is the proposed repository policy so the validated PR
title becomes the subject on `main`. This change does not alter GitHub merge
settings. Maintainers must inspect current settings, enable squash merging,
and disable merge commits and rebase merging to adopt that policy. Until
then, the PR-title check validates the PR title without guaranteeing the final
commit subject. If merge or rebase remains allowed, document that choice and
add commit-subject validation before claiming conventional main-branch history
is enforced. When squash merging, preserve author and co-author certifications
in the final commit message and ensure its author has a matching sign-off.
