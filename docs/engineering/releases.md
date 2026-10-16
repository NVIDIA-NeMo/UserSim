# Releasing UserSim

Release validation builds the sdist and wheel once and tests that wheel before
publication. GitHub and PyPI must receive those same archive bytes. Publishing
the GitHub release and uploading to PyPI are separate maintainer operations.

## Prepare and validate a version

1. Merge a PR setting `project.version` in `pyproject.toml` to `X.Y.Z` or
   `X.Y.ZrcN`. Development, alpha, beta, post, and local versions are not
   accepted by the release-tag gate.
2. Check out the resulting commit on an up-to-date `main`, bootstrap the
   repository's declared uv version, and install locked development tools.
3. Run `make validate-native`, `make check-dist`,
   `TAG=vX.Y.Z make check-release-tag`, and `make test-e2e WHEEL_DIR=dist`.
   Replace `X.Y.Z` with the actual version, including an `rcN` suffix when
   appropriate. These commands run the native checks, coverage,
   extension checks, archive checks, and installed CLI e2e against the newly
   built release wheel. These gates do not replace the container or Python-matrix
   checks required on the PR.
4. After reviewing the successful checks, create an annotated tag on that
   commit and push that specific tag. The tag workflow repeats validation on
   the tagged commit; passing CI on an earlier PR is insufficient.

```bash
git switch main
git pull --ff-only
make install-ci
make validate-native
make check-dist
TAG=vX.Y.Z make check-release-tag
make test-e2e WHEEL_DIR=dist
make release-digests
git tag -a vX.Y.Z -m 'usersim X.Y.Z'
git push origin vX.Y.Z
```

The release build has read-only repository access and restores no build cache.
Its native and installed-wheel checks must finish before it uploads the
original archives to the workflow artifact. The read-only build records their
SHA-256 digests in its summary after installed-wheel validation and before the
publication approval gate. The publication job waits for approval through the
existing `release` environment, then downloads the archives, records their
digests again, creates a draft GitHub release, attaches both archives, and
publishes it. Release candidates are marked as prereleases.

`make release-digests` prints those hashes locally and appends them to the CI
summary when `GITHUB_STEP_SUMMARY` is set. Review the run and SHA-256 digest
summary before approving publication. This
change does not create environment reviewers, change repository rules, or add
PyPI credentials. Maintainers must verify the existing environment protection
and release permissions independently.

## Publish the original assets to PyPI

The publishing maintainer needs GitHub release access and a PyPI account with
permission to publish the `usersim` project. A project-scoped PyPI API token is
the local authentication method below. Store it in an approved credential
manager; supply it as `UV_PUBLISH_TOKEN` without putting it in a command line,
tracked file, workflow log, or shell history. GitHub authentication does not
grant PyPI access. Local publication does not use GitHub OIDC trusted publishing;
an automated trusted publisher would require a separately configured PyPI
publisher and workflow permissions.

Use the repository's pinned uv version (`make check-tools`) and a current
GitHub CLI that supports `gh release verify-asset`. Download into a fresh
directory so an old wheel cannot be included accidentally:

```bash
RELEASE_TAG=vX.Y.Z
RELEASE_ASSETS=$(mktemp -d)
gh release download "$RELEASE_TAG" --repo NVIDIA-NeMo/UserSim \
  --pattern '*.whl' --pattern '*.tar.gz' --dir "$RELEASE_ASSETS"
```

Confirm that the directory contains exactly one wheel and one sdist for the
intended version. Verify each against the GitHub release:

```bash
for archive in "$RELEASE_ASSETS"/*; do
  gh release verify-asset "$RELEASE_TAG" "$archive" --repo NVIDIA-NeMo/UserSim
done
sha256sum "$RELEASE_ASSETS"/*
```

`verify-asset` relies on GitHub's release attestation. If it is unavailable or
verification fails, stop publication and resolve it; do not substitute a newly
built archive. Compare the local SHA-256 values with the release run summary
as a separate check of the handoff.

After loading the project-scoped token into `UV_PUBLISH_TOKEN`, inspect the
upload selection with uv's dry run. The production command is intentionally a
separate step requiring the publishing maintainer's decision:

```bash
uv publish --dry-run --trusted-publishing never "$RELEASE_ASSETS"/*
uv publish --trusted-publishing never "$RELEASE_ASSETS"/*
```

For a rehearsal, use a separate TestPyPI account/token and upload endpoint:

```bash
uv publish --trusted-publishing never \
  --publish-url https://test.pypi.org/legacy/ "$RELEASE_ASSETS"/*
```

Set `UV_PUBLISH_TOKEN` to the TestPyPI token for that command, then restore the
production credential deliberately before a production upload. TestPyPI is a
separate service; a successful rehearsal does not establish PyPI permissions.
Neither command builds distributions. Do not run `make dist`, `uv build`, or a
publishing command that rebuilds after downloading the release assets.

## Verify publication and close the release

Compare PyPI metadata and downloaded bytes with the original GitHub assets:

```bash
python3 - "$RELEASE_TAG" "$RELEASE_ASSETS" <<'PY'
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

version = sys.argv[1].removeprefix("v")
assets = Path(sys.argv[2])
expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.iterdir() if p.is_file()}
with urllib.request.urlopen(f"https://pypi.org/pypi/usersim/{version}/json", timeout=30) as response:
    published = json.load(response)["urls"]
if len(expected) != 2 or len(published) != 2 or {p["filename"] for p in published} != set(expected):
    raise SystemExit("Expected exactly the original wheel and sdist on PyPI")
for asset in published:
    name = asset["filename"]
    if asset["digests"]["sha256"] != expected[name]:
        raise SystemExit(f"PyPI metadata digest differs: {name}")
    with urllib.request.urlopen(asset["url"], timeout=30) as response:
        actual = hashlib.sha256(response.read()).hexdigest()
    if actual != expected[name]:
        raise SystemExit(f"Downloaded PyPI digest differs: {name}")
    print(f"{name}: {actual}")
PY
```

A missing file, unexpected filename, or differing digest means publication has
not been verified. Compare these values with the release run summary too.
Allow index propagation before retrying a missing version; do not rebuild or
overwrite archives.

Merge a follow-up PR setting the next development version, such as
`X.Y.(Z+1).dev0`. Retain the workflow run, tag, asset filenames, and verified
digests in the release record.

Before GitHub publication, a failed attempt may be repaired by deleting the
unpublished tag and any draft release, fixing the cause, and tagging again.
Deletion is an explicit maintainer operation. Once a release is published,
its tag and files stay fixed; corrections ship as a new version. If only one
PyPI archive uploaded, verify the already published file and upload only the
missing original archive. Do not use duplicate-skipping options to hide a
digest mismatch.
