# Assets

Probe content (fact banks, query banks, pressure banks, clinical profiles,
finance corpora, tool catalogs, action taxonomies) ships as **package
data** in each built-in probe's own folder, under
`usersim/engine/probes/<folder>/assets/<probe>/`.

They ship inside the package so a plain `pip install` gives you working
probes with no extra download step. `make check-dist` fails if the wheel
misses a file git tracks under `src/usersim/`, apart from notebooks and
Markdown outside the banks.

## Resolution

Each probe keeps its banks in the `<label>/` folder of the root its class
declares as `assets_root`. A built-in probe declares its own folder in the
package, `engine/probes/<folder>/assets/`. A probe shipped in its own package
points it inside that package, such as `Path(__file__).parent / "assets"`, so
the probe finds its banks wherever the package is installed, with no
environment variable to set. Registration refuses a relative root. The lookup
finds a built-in's folder before the probe's module is imported: a prompt read
while the module loads, or a scorer in a process that never imports the probe,
reaches the probe's assets.

Ahead of a probe's own root, assets resolve along an ordered **search path**
of deliberate overrides, highest priority first:

1. the per-context override set by `set_runtime_assets_dir`
2. each entry of `$USERSIM_ASSET_PATH` (`os.pathsep`-separated)

`probe_assets_dir()` walks those roots, then the probe's own, and takes the
first that holds the probe. An override without a probe's folder leaves that
probe's own banks in reach, which is what lets a package contribute one
probe's assets without restating the rest. A probe's bank loader reads under
its class's `assets_dir()`, the folder this search finds for the probe, and
its `prompts.yaml` is read from that folder too.

An explicit `--assets-dir` is **not** part of the search. It pins resolution
to exactly that directory, for every probe in the run, including one that
declares `assets_root`, because a user who named a location should be told
when the assets are not there rather than quietly served another copy. A run
without it records no asset folder. `simulate --inputs` reads stored rows
from the folder they were stored with, or from each probe's own assets when
they name none.

There is no fallback to the working directory. Resolving against wherever the
process happened to start makes a broken install look healthy from a source
checkout.

## Pre-flight

Before any model call, a run verifies that every selected probe's assets
exist. A probe whose banks are split by locale declares a `locale_bank_check`
on its class, and the run loads the bank each run locale runs on with it:
`financial_services`, `sov_ai_facts` and `sov_ai_dynamic` declare one. The
class's `bank_locale` names that bank, the one the probe loads in the locale's
place when it has none of its own: a registered India variant with no subtree
of its own runs on `en_IN`'s. For `financial_services` a subtree counts once it
has tasks, and `hi_Deva_IN` and `hi_Latn_IN` run on `en_IN`'s until theirs
does. Every missing path is reported at once, with what to do: drop the probe
or the locale from the run, or point `--assets-dir` somewhere else. When a
probe can't run one of the run's locales, the refusal also names the locales it
can run and the guide the class's `authoring_guide` links, if any, and for a
probe that ships with User Sim, the feature-request form.

Without this, a missing bank surfaces from inside a loader partway through a
run, after the other probes in the mix have already been paid for, while
`--dry-run` would report the plan and exit 0.

## Install check

`usersim smoke` runs each registered probe's `asset_problems` classmethod,
which checks the assets the probe ships without a model call. For example,
`sov_ai_facts` and `sov_ai_dynamic` look for a bank in every shipped locale,
`financial_services` loads its `en_US` bank and that bank's dynamic-tier
taxonomy, and `identity_disclosure` loads its spec through every layer. A probe
from another package is checked the same way, and `assert_probe_conforms`
reports what its check finds.

## Generating a bank

`usersim gen-assets --domain <d>` generates one; `validate-assets` vets it
offline and `evaluate-assets` scores it with a judge. Generation and
evaluation are **live, paid** steps and are not needed to use the shipped
banks.

Each domain's region specs (`<locale>.yaml`, one per locale it can generate)
ship as package data, so all three commands find them from an installed
package without `--region-spec`.

A domain builds the bank of the probe that shares its name. Without `--out`
(`gen-assets`) or `--dir` (`validate-assets`, `evaluate-assets`), the commands
use that probe's own asset folder, `<assets_root>/<domain>/<locale>/`. A domain
with no probe of its name has no such folder, and the commands ask for one.

A built-in domain's generator lives beside the probe whose banks it builds:
`financial_services` keeps its generator, region specs and asset notebook in
its folder's `asset_gen/`. Other domains are discovered through the
`usersim.asset_domains` entry point, so a new domain does not require
modifying this project. See [plugins.md](plugins.md).

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
