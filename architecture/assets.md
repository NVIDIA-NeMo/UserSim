# Assets

Probe content (fact banks, query banks, pressure banks, clinical profiles,
finance corpora, tool catalogs, action taxonomies) ships as **package
data** under `usersim/engine/assets/<probe>/`.

They ship inside the package so a plain `pip install` gives you working
probes with no extra download step. `make check-dist` fails if the wheel
misses a file git tracks under `src/usersim/`, apart from notebooks and
Markdown outside the banks.

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

A probe class can also declare `assets_root`, the directory that holds its
`<label>/` folder. A probe shipped in its own package points it inside that
package, such as `Path(__file__).parent / "assets"`, so the probe finds its
banks wherever the package is installed, with no environment variable to
set. Registration refuses a relative root. That root is searched for that
probe only, after the first three entries and before the packaged directory,
so a deliberate override wins over it. The built-in probes declare none. A
probe's bank loader reads under its class's `assets_dir()`, the folder this
search finds for the probe.

An explicit `--assets-dir` is **not** part of the search. It pins resolution
to exactly that directory, for every probe in the run, including one that
declares `assets_root`, because a user who named a location should be told
when the assets are not there rather than quietly served another copy.

There is no fallback to the working directory. Resolving against wherever the
process happened to start makes a broken install look healthy from a source
checkout.

## Pre-flight

Before any model call, a run verifies that every selected probe's assets
exist, including the locale subtree for the locale-scoped probes:
`financial_services`, `sov_ai_facts` and `sov_ai_dynamic`. A locale that runs
on another locale's bank is checked against that bank, the one the probe loads
in its place: a registered India variant with no subtree of its own runs on
`en_IN`'s. For `financial_services` a subtree counts once it has tasks, and
`hi_Deva_IN` and `hi_Latn_IN` run on `en_IN`'s until theirs does. Every
missing path is reported at once, with what to do: drop the probe or the
locale from the run, or point `--assets-dir` somewhere else. When a probe
can't run one of the run's locales, the refusal also names the locales it can
run and links the feature-request form, and for `financial_services` the guide
to authoring a bank.

Without this, a missing bank surfaces from inside a loader partway through a
run, after the other probes in the mix have already been paid for, while
`--dry-run` would report the plan and exit 0.

## Generating a bank

`usersim gen-assets --domain <d>` generates one; `validate-assets` vets it
offline and `evaluate-assets` scores it with a judge. Generation and
evaluation are **live, paid** steps and are not needed to use the shipped
banks.

Each domain's region specs (`<locale>.yaml`, one per locale it can generate)
ship as package data, so all three commands find them from an installed
package without `--region-spec`.

Domains are discovered through the `usersim.asset_domains` entry point, so a
new domain does not require modifying this project. See
[plugins.md](plugins.md).

## Every entry declares its provenance

Bank entries carry a review status. Content authored by a model rather than
reviewed by a subject-matter expert is marked `placeholder: true`, and that
marking propagates: every trajectory that draws on such an entry records a
`used_placeholder_*` warning on its outcome (`<prefix>_used_placeholder_*` for
a probe shipped in its own package), which is written to the trajectory
parquet alongside the scores.

Building a locale bank is a two-stage process, generate then review, and the
framework tracks which stage each entry is at so a scorecard's evidence base
stays traceable. Once a native speaker or domain expert has verified an entry,
set `placeholder: false` and every later run reflects it.

Review status is per entry rather than per bank. The `sov_ai_dynamic`
taxonomies are reviewed for nine locales, with the `en_GB` and `de_DE`
taxonomies flagged; `safety_agentic`, `safety_chat_pressure` and
`sov_ai_multilingual_parity` mix reviewed and flagged entries; the finance,
sovereign-AI fact and health banks and the identity spec are flagged. A bank's
own YAML is the authority on which is which.
