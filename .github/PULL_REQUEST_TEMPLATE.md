<!--
The PR title must follow Conventional Commits (feat:, fix:, docs:, ci:,
chore:, refactor:, test:, style:, perf:). CI checks this.

Every commit author and declared co-author needs a matching DCO sign-off.
Use `git commit -s`; bots follow the same rule. See CONTRIBUTING.md.
-->

## Related issue

<!--
Required if you do not have write access to this repository: a line such as
`Fixes #<issue-number>` (or Closes, or Resolves) for an open issue that a
maintainer has labelled `triaged`. Only the first one counts, and a new pull
request without one is closed. See "Issues before pull requests" in
CONTRIBUTING.md.
-->

## What changed and why

<!-- What a reviewer needs to understand the change. -->

## How it was verified

<!--
Which commands you ran. `make check-all` and `make test` cover most changes.

If the change can affect a live run (models, providers, prompts, scorers, the
trajectory store), say whether you ran `make health-check` and against which
provider. The offline suite cannot catch a payload a provider rejects.
-->

## Results impact

<!--
Delete if not applicable. Changes to probes, assets, scorers or prompts move
published numbers even when every test still passes. Say what moves, so the
shift is a decision rather than a surprise.
-->
