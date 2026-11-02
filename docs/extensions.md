# Writing an extension package

A probe, a scorer or another addition can ship in its own Python package,
installed next to `usersim`. It registers through entry points, so nothing in
this repository changes. This guide builds one such package, `acme-recipes`: a
probe that asks a cooking assistant questions from its own bank, and a scorer
for the rows it produces.

[`architecture/plugins.md`](../architecture/plugins.md) describes the entry
point groups and how discovery behaves, and
[`docs/engine/AUTHORING_A_PROBE.md`](engine/AUTHORING_A_PROBE.md) covers every
hook a probe can implement.

## Layout

```
acme-recipes/
├── pyproject.toml
├── src/acme_recipes/
│   ├── __init__.py
│   ├── probe.py
│   ├── scorer.py
│   └── assets/acme_recipes/en_US/recipes.yaml
└── tests/
    ├── conftest.py
    └── test_acme_recipes.py
```

The banks sit under `assets/<label>/`, the layout the asset search path reads.

## Names

Every name the package adds starts with its prefix, `acme_`, because unprefixed
names belong to the built-ins:

| Name | In `acme-recipes` |
|---|---|
| Probe label and family | `acme_recipes` |
| Scorer | `acme_recipes` |
| Result column | `acme_recipe_id` |
| Placeholder warning | `acme_used_placeholder_recipe` |
| Metadata keys, if any | `acme_...` |
| Entry point names | the names they register |

The conversation loop keeps its own record in `state.metadata` under the keys
in `RESERVED_METADATA_KEYS` (`usersim/engine/core/simulation.py`); a probe
reads them and never writes them.

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "acme-recipes"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["usersim>=X.Y,<X.Z", "pyyaml"]

[project.entry-points."usersim.probes"]
acme_recipes = "acme_recipes.probe:RecipeProbe"

[project.entry-points."usersim.scorers"]
acme_recipes = "acme_recipes.scorer"

[tool.setuptools.package-data]
acme_recipes = ["assets/**/*.yaml"]

[dependency-groups]
dev = ["pytest", "pytest-asyncio", "pytest-env"]

[tool.pytest.ini_options]
env = ["USERSIM_DISABLE_EXTENSIONS=1"]
asyncio_mode = "auto"
```

- **`dependencies`** names a range of User Sim releases: `X.Y` is the first
  release with every hook and attribute the package uses, and `X.Z` the next
  minor release. A lock file pins the exact build an environment runs.
- **The entry points** name the probe class and the scorer's module.
- **`package-data`** ships the banks in the wheel; without it, an installed
  package has no assets.
- **The pytest settings** turn extension discovery off under test and run
  `async def` tests without a marker (see [Tests](#tests)).

## The probe

```python
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from usersim.engine.core.identity import persona_uuid
from usersim.engine.core.probes import BankBackedProbe, register_probe


@dataclass(frozen=True)
class Recipe:
    id: str
    question: str
    placeholder: bool


@dataclass(frozen=True)
class RecipeBank:
    bank_id: str
    bank_version: str
    recipes: tuple[Recipe, ...]


@cache
def load_recipe_bank(locale: str) -> RecipeBank:
    path = RecipeProbe.assets_dir() / locale / "recipes.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RecipeBank(doc["bank_id"], doc["bank_version"], tuple(Recipe(**r) for r in doc["recipes"]))


@register_probe(family="acme_recipes", prompt_version="v1.0", variants=("default",))
class RecipeProbe(BankBackedProbe):
    label = "acme_recipes"
    assets_root = Path(__file__).parent / "assets"
    bank_loader = staticmethod(load_recipe_bank)
    placeholder_warning_kind = "acme_used_placeholder_recipe"
    output_columns = ("acme_recipe_id",)

    def derive_task(self, persona, bank, *, cfg):
        rng = random.Random(f"{persona_uuid(persona)}:{bank.bank_version}:{cfg.random_seed}")
        return rng.choice(bank.recipes)

    def get_user_system_prompt(self) -> str:
        return f"You are a home cook. Ask a cooking assistant: {self._task.question}"

    def get_assistant_system_prompt(self) -> str:
        return ""

    def build_result_extras(self, state) -> dict:
        return {**super().build_result_extras(state), "acme_recipe_id": self._task.id}
```

- **`assets_root`** is the absolute directory holding `acme_recipes/`, so the
  probe finds its banks wherever the package is installed.
- **`load_recipe_bank`** reads under `RecipeProbe.assets_dir()`, the probe's
  folder on the asset search path, so a folder `--assets-dir` or
  `USERSIM_ASSET_PATH` names reaches it too.
- **`placeholder_warning_kind`** is the warning a row records when its recipe
  is marked `placeholder: true`. Its `<prefix>_used_placeholder_<thing>` form is
  what the capability report counts as placeholder content.
- **`output_columns`** declares the column `build_result_extras` adds, so a run
  keeps it.
- **`derive_task`** picks the row's recipe with a generator seeded from the
  persona, the bank's version and the run's seed, never the global `random`, so
  a row draws the same recipe every time it runs.
- **The assistant's system prompt is empty:** the assistant under test answers
  as it would for any user.

The bank, `assets/acme_recipes/en_US/recipes.yaml`:

```yaml
bank_id: acme_recipes
bank_version: "1.0.0"
recipes:
  - id: pantry_curry
    question: How do I make a chickpea curry with what is in my pantry?
    placeholder: true
```

`placeholder: true` marks content no subject-matter expert has reviewed; see
[`architecture/assets.md`](../architecture/assets.md).

## The scorer

```python
from usersim.engine.evaluator.scorers import register_scorer


async def score(row, models):
    """1.0 when the conversation ran against a recipe from the bank."""
    return {"acme_recipe_grounded": 1.0 if row.get("acme_recipe_id") else 0.0}


register_scorer("acme_recipes", score)
```

The evaluator awaits a scorer once per trajectory, with the row and the
models, and merges the mapping it returns into the row's evaluation. A field
the row holds empty arrives as `None`.

## Tests

`tests/conftest.py` loads the hermetic harness the built-in tests run under:

```python
pytest_plugins = ["usersim.testing.pytest_plugin"]
```

It fails any outbound connection, stubs the tokenizer and the persona dataset,
and stops the run unless extension discovery is off and `pytest-asyncio` is
installed. With discovery off, the tests import the package to register it:

```python
import json

import acme_recipes.probe  # noqa: F401  registers the probe; discovery is off under test
import acme_recipes.scorer  # noqa: F401  registers the scorer
from usersim.engine.config import ConversationSimulatorConfig
from usersim.testing import assert_hosted_parity, assert_probe_conforms, assert_scorer_conforms, run_episode

PERSONA = {"first_name": "Sarah", "last_name": "Johnson", "age": 35, "city": "Seattle"}
CONFIG = ConversationSimulatorConfig(name="conversation_messages", max_turns=2, random_seed=7)


def test_the_probe_conforms():
    assert_probe_conforms("acme_recipes")


async def test_the_scorer_conforms():
    await assert_scorer_conforms("acme_recipes")


async def test_an_episode_runs_against_a_recipe():
    run = await run_episode("acme_recipes", config=CONFIG, persona=PERSONA)

    assert run.row["conversation_status"] is True
    assert run.row["acme_recipe_id"] == "pantry_curry"
    warnings = json.loads(run.row["simulation_outcome"])["warnings"]
    assert [w["kind"] for w in warnings] == ["acme_used_placeholder_recipe"]


async def test_a_host_reproduces_a_standalone_run():
    await assert_hosted_parity("acme_recipes", config=CONFIG, persona=PERSONA)
```

- **`assert_probe_conforms`** checks the probe's shape: its registration, its
  required methods, its class attributes and its assets.
- **`assert_scorer_conforms`** checks that the scorer is async and tolerates
  empty fields.
- **`run_episode`** runs a whole episode with scripted models, through the same
  generator as `usersim simulate`, with nothing patched and no provider.
- **`assert_hosted_parity`** checks that a host driving the probe through
  `ProbeEpisodeRuntime` reproduces a standalone run.

## In CI

```bash
uv sync
uv run pytest
uv run usersim smoke
```

Outside the tests, discovery is on, so `usersim smoke` lists each installed
extension package with its entry points, in a line like this one:

```
[OK]   extensions            acme-recipes 0.1.0: 2 entry point(s)
```

It fails when an entry point contributes nothing: it failed to load, another
package advertises the same name, or a built-in holds it.

## Running it

Once installed, the probe joins a run by its label. This makes live model calls
and is billed by the provider:

```bash
usersim simulate --locale en_US --num-rows 5 \
  --probe-mix "acme_recipes=1.0" --out output/trajectories
```

`--assets-dir` pins every probe in a run to the folder it names, this one
included, so a run that passes it needs `acme_recipes/` there too.
