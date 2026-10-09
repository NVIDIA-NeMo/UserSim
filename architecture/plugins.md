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
| `usersim.toolset_sources` | *(declared; consumed by future work)* | n/a |

```toml
[project.entry-points."usersim.probes"]
my_probe = "my_package.probe:MyProbe"
```

An extension probe's per-row data goes in its metadata (`state.metadata`),
stored with the row as `conversation_metadata`: the list of extra columns a
run keeps is part of the core config. See [Probes](probes.md).

## How discovery behaves

**Probes and scorers are discovered by import.** `@register_probe` and
`register_scorer` already run at import, so an out-of-tree probe travels the
identical code path as a built-in and inherits its validation. There is no
parallel registration path to drift.

**A built-in always wins a name collision.** An extension that silently
shadowed a shipped probe would change what a run means while every config
file and report still named the original. The collision is reported and the
extension dropped.

**A broken extension is skipped, not fatal.** An entry point that fails to
import is logged with the exception and omitted, so one bad package cannot
make `--help` unusable. The log is a warning rather than a debug line,
because a probe that silently fails to appear is far harder to diagnose than
a noisy startup.

**Ordering is by entry-point name**, not install order, so registry contents
do not depend on the sequence of `pip install` commands.

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
and `language` itself, so those are never `None`. A default passed to
`row.get(key, default)` covers only an absent key, so pass a row shaped like
your probe's trajectories to the check as well:
`assert_scorer_conforms("my_scorer", sample_row=row)` also scores that row with
every other field set to `None`.

Each check has a `*_problems` form that returns every finding at once, so you
fix one thing rather than rediscovering the next on each run.

## Asset paths

An extension that ships its own banks adds them to `$USERSIM_ASSET_PATH`
rather than replacing the tree. The packaged directory stays last in the
search order, so the shipped banks remain reachable. See
[assets.md](assets.md).

## What the community can bring

Beyond the mechanics above, four contributions the design is specifically
built to absorb:

- **New persona datasets and locales.** Any census-grounded or
  domain-specific persona dataset can slot in as a population source. A
  sovereign-AI team contributing a national dataset (grounded in local
  census data, written in the local language) gains the whole simulation,
  evaluation and reporting pipeline with it. See
  [`docs/locales.md`](../docs/locales.md) for how a preview locale graduates
  to a shipped one.
- **Agent benchmark integrations.** Task suites define the *task*; this
  defines the *user*. Wrapping one adds behavioural diversity and demographic
  grounding that task-only benchmarks lack.
- **Custom evaluation axes.** The multi-axis schema is extensible, so a
  domain-specific quality dimension does not require touching the core.
- **Behavioural-realism mitigations.** The realism layer (incremental
  disclosure, OCEAN-derived interaction style, reactive frustration) is
  modular. Typo injection, device switching, session abandonment and
  non-native language patterns are all additive.

## Verifying the whole surface

`make check-extensions` installs a fixture package into a throwaway
virtualenv and asks the installed CLI what it can see: all six live seams,
against real packaging metadata rather than a mock. The fixture also ships a
probe built on `identity_disclosure`, whose spec layer must extend the spec
inside the installed wheel and resolve both its own model and an inherited one.
