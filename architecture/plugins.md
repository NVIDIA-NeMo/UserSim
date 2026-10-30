# Extensions

Every registry in this project is populated the same way: the built-ins it
defines itself, merged with whatever installed distributions advertise under
the matching entry-point group. **You do not need to modify this repository
to add a probe, a scorer, a domain, a backend, a capability, or a
subcommand.**

## A note on the word

Data Designer has its own entry-point system for column generators, which it
calls *plugins* and gates with `DISABLE_DATA_DESIGNER_PLUGINS`. This project
registers two of them (`conversation-simulator`, `trajectory-evaluator`).

What is described on this page is unrelated, and is called an *extension*
throughout, variable names included. Anyone running both can tell from a name
which system they are looking at.

## The seven groups

| Group | Advertise | Merged into |
|---|---|---|
| `usersim.probes` | a `BaseProbe` subclass or its module | the probe registry |
| `usersim.scorers` | a module calling `register_scorer` | the scorer registry |
| `usersim.asset_domains` | an `AssetDomain` or a zero-arg factory | `gen-assets` domains |
| `usersim.backends` | an `ExecutionBackend` implementation | `--backend` choices |
| `usersim.capabilities` | a `CapabilityDefinition` or an iterable | the capability matrix |
| `usersim.commands` | a `register(subparsers)` callable | `usersim --help` |
| `usersim.toolset_sources` | *(reserved)* | n/a |

```toml
[project.entry-points."usersim.probes"]
my_probe = "my_package.probe:MyProbe"
```

**Every name an extension adds carries its prefix.** Probe labels, scorer
names, capability ids, asset domains, command names, output columns, warning
kinds and metadata keys all start with the package's own prefix, such as
`acme_recipes` or `acme_used_placeholder_recipe`. Unprefixed names belong to
the built-ins. The registries refuse a name that is already taken, so the
prefix is what keeps two packages, or a package and a later built-in, from
claiming the same one.

[`docs/extensions.md`](../docs/extensions.md) builds a whole extension
package: a probe with its own bank, a scorer, their tests and a CI recipe.

An extension probe declares the columns it writes in its class's
`output_columns`, and a run keeps them alongside the core columns. Other
per-row data goes in its metadata (`state.metadata`), stored with the row as
`conversation_metadata`. See [Probes](probes.md).

An extension's warnings carry its prefix too: a bank-backed probe names its
placeholder warning `<prefix>_used_placeholder_<thing>`, which the capability
report counts as placeholder content, and any other warning it records is a
prefixed name such as `acme_slow_tool`.

## How discovery behaves

**Probes and scorers are discovered by import.** `@register_probe` and
`register_scorer` already run at import, so an out-of-tree probe travels the
identical code path as a built-in and inherits its validation. There is no
parallel registration path to drift.

**Collisions.** Commands, backends, asset domains and capabilities keep the
built-in: an extension under a taken name is reported and dropped. A probe
label or scorer name belongs to the first class or function registered under
it. Registering a different one under a taken name raises
`ProbeRegistrationError` or `ScorerRegistrationError`, naming both
definitions; for an extension, that is a load failure, so the extension is
skipped with a warning and the shipped probe or scorer stays. Registering the
same definition again, as a reloaded module does, updates the entry. Give an
extension probe or scorer a name of its own.

**An extension that fails to load is skipped, not fatal.** An entry point
that fails to import is logged with the exception and omitted, so one bad
package cannot make `--help` unusable. The log is a warning rather than a
debug line, because a probe that silently fails to appear is far harder to
diagnose than a noisy startup.

**Ordering is by entry-point name**, not install order, so registry contents
do not depend on the sequence of `pip install` commands.

**A name two packages advertise in a group is loaded from neither.** Choosing
one would make what a run uses depend on which package sorts first, so both
are skipped with a warning naming each package. Every discovery warning names
the package an entry point comes from.

**`USERSIM_DISABLE_EXTENSIONS` suppresses all of it.** The test suite sets
this, because an extension installed in a developer's virtualenv would
otherwise change what the suite sees and make results depend on the machine.

## Capabilities: the seam that is easy to miss

A probe with a registered class, a registered scorer, and resolvable assets
will simulate and evaluate correctly, and then be **absent from the
report**, because the capability matrix is built from capability definitions
rather than from trajectories. The rows exist; nothing asks about them.

`usersim.capabilities` closes that. If you contribute a probe, contribute the
capability it provides evidence for.

## Validating an extension

`usersim.testing` publishes the conformance checks that run against every
built-in, so they cannot drift into describing a contract this project does
not itself keep:

```python
from usersim.testing import assert_probe_conforms, assert_scorer_conforms

def test_my_probe_conforms():
    assert_probe_conforms("my_probe")

async def test_my_scorer_conforms():
    await assert_scorer_conforms("my_scorer")
```

The scorer check is awaitable because a scorer is. The evaluator awaits every
scorer, so `register_scorer` expects an async function:

```python
from usersim.engine.evaluator.scorers import register_scorer

async def score(row, models):
    return {"my_axis": 1.0}

register_scorer("my_scorer", score)
```

A scorer declared with plain `def` is reported as non-conforming rather than
failing later inside a run.

The row a scorer receives holds every column of the trajectory, and an empty
cell arrives as `None` whichever way pandas held it. The evaluator fills
`conversation_messages` (a list), `persona` (a dict), `probe_family`, `locale`
and `language` itself, so those are never `None`; `probe_family` is an empty
string for a row that names no probe. A default passed to
`row.get(key, default)` covers only an absent key, so pass a row shaped like
your probe's trajectories to the check as well:
`assert_scorer_conforms("my_scorer", sample_row=row)` also scores that row with
every other field set to `None`.

Each check has a `*_problems` form that returns every finding at once, so you
fix one thing rather than rediscovering the next on each run.

`usersim smoke` reports what discovery found. Its `extensions` line lists each
installed package with the number of entry points it contributes, and fails
when an entry point contributes nothing: it failed to load, another package
advertises the same name, or a built-in holds it.

```
[OK]   extensions            my-probes 0.3.0: 3 entry point(s)
```

## Asset paths

An extension that ships its own banks keeps them under `assets/<label>/` in
its package and names that `assets` directory on the probe class:

```python
from pathlib import Path

from usersim.engine.core.probes import BaseProbe

class MyProbe(BaseProbe):
    label = "my_probe"
    assets_root = Path(__file__).parent / "assets"
```

The probe then finds its banks wherever the package is installed. That root
is searched after `$USERSIM_ASSET_PATH` and before the packaged banks, so a
deployment can override it and the shipped banks remain reachable.
`@register_probe` refuses a relative `assets_root`, which would resolve against
the working directory. The probe's bank loader reads under `MyProbe.assets_dir()`,
its folder on that search, so an override reaches the loader too.
`$USERSIM_ASSET_PATH` stays the way to layer content over a probe you don't
ship, such as a patch to the shipped `identity_disclosure` spec. See
[assets.md](assets.md).

## Testing an extension

`usersim.testing.pytest_plugin` is the harness the built-in tests run under.
An extension's suite loads it from its top-level `conftest.py`:

```python
pytest_plugins = ["usersim.testing.pytest_plugin"]
```

With the plugin loaded, a test run:

- fails any outbound connection with `BlockedNetworkCall`. It is a
  `BaseException`, so the retry and fallback paths that catch `Exception`
  cannot absorb it and pass the test down a fallback;
- stops before collecting unless `USERSIM_DISABLE_EXTENSIONS` is set, so other
  packages installed in the environment cannot register themselves and change
  what the tests see;
- stops unless `pytest-asyncio` is installed, because the awaited probe hooks
  and `assert_scorer_conforms` are async;
- stubs the tokenizer behind the reasoning-token estimate, which otherwise
  downloads its vocabulary on first use;
- points Data Designer's persona sampler at four synthetic `en_US` personas.
  The `synthetic_persona_dataset` fixture yields the directory they are in.

Set the variable for the run with `pytest-env`:
`env = ["USERSIM_DISABLE_EXTENSIONS=1"]` under `[tool.pytest.ini_options]`.
With discovery off, the tests import the extension's own package to register
its probes and scorers.

`run_episode` runs one episode of a probe offline, through the same generator
as `usersim simulate`, with a `ScriptedModel` answering for every model alias.
It returns the row the generator produces and every prompt the models
received, so a probe's tests need neither a provider nor a patch of the
engine's internals.

Installing `usersim` does not load the plugin. It has no `pytest11` entry
point, because it blocks outbound connections, and a project that installs
`usersim` for another reason keeps its tests as they are.

## What the community can bring

Beyond the mechanics above, four contributions the design is specifically
built to absorb:

- **New persona datasets and locales.** A national persona dataset, grounded
  in local census data and written in the local language, is the foundation
  of a new locale: once Data Designer serves it, the locale gains the whole
  simulation, evaluation and reporting pipeline, and probe content follows
  bank by bank. See [`docs/locales.md`](../docs/locales.md) for how a preview
  locale graduates to a shipped one.
- **Agent benchmark integrations.** Task suites define the *task*; this
  defines the *user*. Wrapping one adds behavioral diversity and demographic
  grounding that task-only benchmarks lack.
- **Custom evaluation axes.** The multi-axis schema is extensible, so a
  domain-specific quality dimension does not require touching the core.
- **Behavioral-realism mitigations.** The realism layer (incremental
  disclosure, OCEAN-derived interaction style, reactive frustration) is
  modular. Typo injection, device switching, session abandonment and
  non-native language patterns are all additive.

## Verifying the whole surface

`make check-extensions` installs a fixture package into a throwaway
virtualenv and asks the installed CLI what it can see: all six live seams,
against real packaging metadata rather than a mock. The fixture also ships a
probe built on `identity_disclosure`, whose spec layer must extend the spec
inside the installed wheel and resolve both its own model and an inherited one.
