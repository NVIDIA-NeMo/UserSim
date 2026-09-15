# Assets

Probe content (fact banks, query banks, pressure banks, clinical profiles,
finance corpora, tool catalogues, action taxonomies) ships as **package
data** under `usersim/engine/assets/<probe>/`.

They ship inside the package so a plain `pip install` gives you working
probes with no extra download step, and `make check-wheel` fails if that ever
stops being true.

## Resolution

Assets resolve along an ordered **search path**, highest priority first:

1. the per-context override set by `set_runtime_assets_dir`
2. each entry of `$USERSIM_ASSET_PATH` (`os.pathsep`-separated)
3. `$USERSIM_ASSETS_DIR`
4. the packaged `usersim/engine/assets/` directory

`probe_assets_dir()` walks that path and takes the first root holding the
probe. The packaged directory is always last, so prepending a root never
hides the shipped banks, which is what lets a package contribute one
probe's assets without restating the rest.

An explicit `--assets-dir` is **not** part of the search. It pins resolution
to exactly that directory, because a user who named a location should be told
when the assets are not there rather than quietly served the packaged copies.

There is no fallback to the working directory. Resolving against wherever the
process happened to start makes a broken install look healthy from a source
checkout.

## Pre-flight

Before any model call, a run verifies that every selected probe's assets
exist, including the locale subtree for locale-scoped probes like
`financial_services` and the `sov_ai_*` family. Every missing path is
reported at once, with the three remedies: point `--assets-dir` somewhere
else, generate the bank, or drop the probe from the mix.

Without this, a missing bank surfaces from inside a loader partway through a
run, after the other probes in the mix have already been paid for, while
`--dry-run` would report the plan and exit 0.

## Generating a bank

`usersim gen-assets --domain <d>` generates one; `validate-assets` vets it
offline and `evaluate-assets` scores it with a judge. Generation and
evaluation are **live, paid** steps and are not needed to use the shipped
banks.

Domains are discovered through the `usersim.asset_domains` entry point, so a
new domain does not require modifying this project. See
[plugins.md](plugins.md).

## Every entry declares its provenance

Bank entries carry a review status. Content authored by a model rather than
reviewed by a subject-matter expert is marked `placeholder: true`, and that
marking propagates: a capability cell computed from unreviewed content is
labelled accordingly rather than presented as a finished result.

This is deliberate machinery, not a caveat. Building a locale bank means
generating a first draft and then getting it reviewed, and the framework
tracks which stage each entry is at so a scorecard never overstates its own
evidence. When a native reviewer signs off on an entry, flip the flag and the
cell upgrades itself.

Today the `general` probes run on reviewed content; the sovereign-AI, safety,
health and finance banks ship as reviewed-pending first drafts.
