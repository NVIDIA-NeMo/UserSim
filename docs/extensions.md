# Writing an extension package

A probe, a scorer or another addition can ship in its own Python package,
installed next to `usersim`. It registers through entry points, so nothing in
this repository changes. This guide builds one such package, `my-bookings`: a
probe whose simulated users ask an assistant to book an appointment, writing the
date the way people write dates in their own language, and a scorer that checks
the booking the assistant makes.

[`architecture/plugins.md`](../architecture/plugins.md) describes the entry
point groups and how discovery behaves,
[`docs/engine/AUTHORING_A_PROBE.md`](engine/AUTHORING_A_PROBE.md) covers every
hook a probe can implement, and [`extension_api.md`](extension_api.md) lists
every name the package can import from `usersim`. When a package needs a hook
User Sim does not have,
[Adding a hook](../CONTRIBUTING.md#adding-a-hook) says how to ask for one.

## Layout

```
my-bookings/
├── pyproject.toml
├── src/my_bookings/
│   ├── __init__.py
│   ├── probe.py
│   ├── scorer.py
│   └── assets/my_bookings/
│       ├── en_US/bookings.yaml
│       └── pt_BR/bookings.yaml
└── tests/
    ├── conftest.py
    └── test_my_bookings.py
```

The banks sit under `assets/<label>/<locale>/`, the layout the asset search path
reads.

## Names

Every name the package adds starts with its prefix, `my_`, because unprefixed
names belong to the built-ins:

| Name | In `my-bookings` |
|---|---|
| Probe label and family | `my_bookings` |
| Scorer | `my_bookings` |
| Result columns | `my_booking_id`, `my_booking` |
| Placeholder warning | `my_used_placeholder_booking` |
| Metadata key | `my_booking` |
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
name = "my-bookings"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["usersim>=X.Y,<X.Z", "pyyaml"]

[project.entry-points."usersim.probes"]
my_bookings = "my_bookings.probe:BookingProbe"

[project.entry-points."usersim.scorers"]
my_bookings = "my_bookings.scorer"

[tool.setuptools.package-data]
my_bookings = ["assets/**/*.yaml"]

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
import json
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml
from usersim.extension import (
    BankBackedProbe,
    BankVerbatimMixin,
    PreviewField,
    ToolExecutionMixin,
    format_persona_for_prompt,
    language_instruction,
    persona_uuid,
    register_probe,
)

#: The locales the package ships a bank for.
LOCALES = ("en_US", "pt_BR")

BOOK_APPOINTMENT = {
    "type": "function",
    "function": {
        "name": "book_appointment",
        "description": "Book an appointment.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:MM, 24-hour"},
                "people": {"type": "integer"},
            },
            "required": ["date", "time", "people"],
        },
    },
}


@dataclass(frozen=True)
class Booking:
    id: str
    opening: str
    date: str
    time: str
    people: int
    placeholder: bool


@dataclass(frozen=True)
class BookingBank:
    bank_id: str
    bank_version: str
    bookings: tuple[Booking, ...]


def read_booking_bank(locale_dir: Path) -> BookingBank:
    doc = yaml.safe_load((locale_dir / "bookings.yaml").read_text(encoding="utf-8"))
    return BookingBank(doc["bank_id"], doc["bank_version"], tuple(Booking(**b) for b in doc["bookings"]))


@cache
def load_booking_bank(locale: str) -> BookingBank:
    return read_booking_bank(BookingProbe.assets_dir() / locale)


@register_probe(family="my_bookings", prompt_version="v1.0", variants=("default",))
class BookingProbe(ToolExecutionMixin, BankVerbatimMixin, BankBackedProbe):
    label = "my_bookings"
    description = (
        "Tests whether an assistant books what the user means: a date, a time and a number of people, "
        "written the way people write them in the user's language."
    )
    preview_fields = (PreviewField("my_booking_id", "Booking id"),)
    default_scorers = ("my_bookings",)
    assets_root = Path(__file__).parent / "assets"
    bank_loader = staticmethod(load_booking_bank)
    locale_bank_check = staticmethod(read_booking_bank)
    placeholder_warning_kind = "my_used_placeholder_booking"
    verbatim_field = "opening"
    output_columns = ("my_booking_id", "my_booking")

    @classmethod
    def asset_problems(cls) -> list[str]:
        problems = []
        for locale in LOCALES:
            try:
                read_booking_bank(cls.assets_dir() / locale)
            except Exception as error:
                problems.append(f"{locale}: {type(error).__name__}: {error}")
        return problems

    def derive_task(self, persona, bank, *, cfg):
        rng = random.Random(f"{persona_uuid(persona)}:{bank.bank_version}:{cfg.random_seed}")
        return rng.choice(bank.bookings)

    def get_user_system_prompt(self) -> str:
        return "\n\n".join(
            [
                format_persona_for_prompt(self._persona),
                "You asked an assistant to book an appointment, giving the date and time. "
                f"If it asks how many people are coming, the answer is {self._task.people}; "
                "say it the way you would. Once it confirms the booking, thank it.",
                language_instruction(self._language, self._locale),
            ]
        )

    def get_assistant_system_prompt(self) -> str:
        return ""

    def get_tools_for_assistant(self) -> list:
        return [BOOK_APPOINTMENT]

    def allow_early_stop_at_turn(self, turn_idx, state) -> bool:
        return "my_booking" in state.metadata

    async def execute_tool_call(self, name, args, tc, state, models, *, turn_idx, call_idx) -> str:
        if "my_booking" not in state.metadata:
            user_turns = sum(1 for message in state.messages if message.get("role") == "user")
            state.metadata["my_booking"] = {"arguments": args, "after_user_turns": user_turns}
        return json.dumps({"status": "booked"})

    def build_result_extras(self, state) -> dict:
        task = self._task
        booking = {
            "expected": {"date": task.date, "time": task.time, "people": task.people},
            "made": state.metadata.get("my_booking"),
        }
        return {
            **super().build_result_extras(state),
            "my_booking_id": task.id,
            "my_booking": json.dumps(booking, ensure_ascii=False),
        }
```

- **`ToolExecutionMixin`** runs the assistant's tool calls. The probe offers
  `book_appointment` in `get_tools_for_assistant`, and `execute_tool_call`
  answers each call. It records the first booking in `state.metadata`, with
  how many user turns came before it.
- **`BankVerbatimMixin`** sends the booking's `opening`, the field
  `verbatim_field` names, as turn 1, so the date reaches the assistant written
  exactly as the user wrote it.
- **`get_user_system_prompt`** sets up the user model for the turns after the
  first: the persona, the number of people it gives only when the assistant
  asks, and `language_instruction`, which keeps it writing in the run's
  language.
- **`allow_early_stop_at_turn`** keeps the conversation going until the
  assistant books, so a user who adds a thank-you to the number of people does
  not end it before the booking is made.
- **The assistant's system prompt is empty:** the tool's description is all the
  assistant under test is told, so it answers as it would for any user.
- **`description`** is what the per-row preview and the capability report
  show about the probe.
- **`preview_fields`** puts the row's booking id in the per-row preview's Probe
  Inputs panel.
- **`default_scorers`** names the package's scorer, so
  `usersim eval --scorers auto` runs it whenever a run holds the probe's rows,
  beside `language_compliance`, which checks that the assistant replied in the
  user's language.
- **`assets_root`** is the absolute directory holding `my_bookings/`, so the
  probe finds its banks wherever the package is installed.
- **`load_booking_bank`** reads under `BookingProbe.assets_dir()`, the probe's
  folder on the asset search path, so a folder `--assets-dir` or
  `USERSIM_ASSET_PATH` names reaches it too.
- **`locale_bank_check`** reads one locale's bank, so a run in a locale with no
  bank that loads stops before any model call rather than failing every row.
- **`asset_problems`** reads each bank the package ships, so `usersim smoke` and
  `assert_probe_conforms` report an install that is missing one.
- **`placeholder_warning_kind`** is the warning a row records when its booking
  is marked `placeholder: true`. Its `<prefix>_used_placeholder_<thing>` form is
  what the capability report counts as placeholder content.
- **`output_columns`** declares the columns `build_result_extras` adds, so a run
  keeps them. `my_booking` holds, as JSON, what the user meant and what the
  assistant booked.
- **`derive_task`** picks the row's booking with a generator seeded from the
  persona, the bank's version and the run's seed, never the global `random`, so
  a row draws the same booking every time it runs.

The banks, `assets/my_bookings/en_US/bookings.yaml`:

```yaml
bank_id: my_bookings_en_US
bank_version: "1.0.0"
bookings:
  - id: museum_tour
    opening: Hi! Can you book the guided museum tour for 04/03/2027 at 3:30 pm?
    date: "2027-04-03"
    time: "15:30"
    people: 3
    placeholder: true
```

and `assets/my_bookings/pt_BR/bookings.yaml`:

```yaml
bank_id: my_bookings_pt_BR
bank_version: "1.0.0"
bookings:
  - id: museum_tour
    opening: Oi! Você pode reservar a visita guiada ao museu para 04/03/2027 às 15h30?
    date: "2027-03-04"
    time: "15:30"
    people: 3
    placeholder: true
```

Both open with the same date, `04/03/2027`. A user writing American English
means 3 April and a user writing Brazilian Portuguese means 4 March. An
assistant that reads every date month first books the Brazilian user's tour a
month late, and one that reads every date day first books the American user's
a month early. Neither opening says how many people are coming, so an assistant
that books on its first reply has to guess.

`placeholder: true` marks content no subject-matter expert has reviewed, which
for these banks means a native speaker of the bank's language; see
[`architecture/assets.md`](../architecture/assets.md).

## The scorer

```python
import json

from usersim.extension import register_scorer


async def score(row, models):
    """Whether the assistant booked what the user meant, after asking how many people."""
    if not row.get("my_booking"):
        return {}
    booking = json.loads(row["my_booking"])
    expected, made = booking["expected"], booking["made"]
    if made is None:
        return {"my_booked": 0.0}
    arguments = made["arguments"]
    return {
        "my_booked": 1.0,
        "my_date_correct": float(arguments.get("date") == expected["date"]),
        "my_time_correct": float(arguments.get("time") == expected["time"]),
        "my_people_correct": float(arguments.get("people") == expected["people"]),
        "my_asked_before_booking": float(made["after_user_turns"] > 1),
    }


register_scorer("my_bookings", score, deterministic=True)
```

The evaluator awaits a scorer once per trajectory, with the row and the
models, and merges the mapping it returns into the row's evaluation. It calls
the scorer on every row it evaluates, so the scorer returns nothing for a row
without a `my_booking` column, such as one from another probe. A field the
row holds empty arrives as `None`. Each argument must match what the user meant
exactly, in the form the tool's description asks for, and
`my_asked_before_booking` is 1.0 when the booking came after the user's
second turn, the first that can say how many people are coming.
`deterministic=True` says the scorer never calls a model, so `usersim rescore`
can recompute it on a stored evaluation.

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

from usersim.extension import ConversationSimulatorConfig
from usersim.testing import assert_hosted_parity, assert_probe_conforms, assert_scorer_conforms, run_episode

import my_bookings.probe  # noqa: F401  registers the probe; discovery is off under test
from my_bookings.scorer import score

PERSONA = {"first_name": "Ana", "last_name": "Souza", "age": 35, "city": "Recife"}
CONFIG = ConversationSimulatorConfig(name="conversation_messages", max_turns=3, random_seed=7, locale="pt_BR")


def test_the_probe_conforms():
    assert_probe_conforms("my_bookings")


async def test_the_scorer_conforms():
    await assert_scorer_conforms("my_bookings")


async def test_an_episode_opens_in_portuguese_and_records_the_booking():
    run = await run_episode("my_bookings", config=CONFIG, persona=PERSONA, language="Portuguese")

    assert run.row["conversation_status"] is True
    opening = json.loads(run.row["conversation_messages"])[0]
    assert opening["content"] == "Oi! Você pode reservar a visita guiada ao museu para 04/03/2027 às 15h30?"
    booking = json.loads(run.row["my_booking"])
    assert booking["expected"] == {"date": "2027-03-04", "time": "15:30", "people": 3}
    assert booking["made"]["after_user_turns"] == 1  # the scripted assistant books on its first reply
    warnings = json.loads(run.row["simulation_outcome"])["warnings"]
    assert [w["kind"] for w in warnings] == ["my_used_placeholder_booking"]


async def test_a_month_first_reading_of_a_day_first_date_is_wrong():
    made = {"arguments": {"date": "2027-04-03", "time": "15:30", "people": 3}, "after_user_turns": 2}
    expected = {"date": "2027-03-04", "time": "15:30", "people": 3}
    result = await score({"my_booking": json.dumps({"expected": expected, "made": made})}, models={})

    assert result == {
        "my_booked": 1.0,
        "my_date_correct": 0.0,
        "my_time_correct": 1.0,
        "my_people_correct": 1.0,
        "my_asked_before_booking": 1.0,
    }


async def test_a_host_reproduces_a_standalone_run():
    await assert_hosted_parity("my_bookings", config=CONFIG, persona=PERSONA, language="Portuguese")
```

- **`assert_probe_conforms`** checks the probe's shape: its registration, its
  required methods, its class attributes and its assets.
- **`assert_scorer_conforms`** checks that the scorer is async and tolerates
  empty fields.
- **`run_episode`** runs a whole episode with scripted models, through the same
  generator as `usersim simulate`, with nothing patched and no provider. Here it
  runs in `pt_BR`: the opening reaches the assistant as written, and the
  scripted assistant, which calls a tool as soon as one is offered, books on its
  first reply.
- **The scorer** is tested on a row built by hand, in which the assistant read
  the Brazilian date month first.
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
[OK]   extensions            my-bookings 0.1.0: 2 entry point(s)
```

It fails when an entry point contributes nothing: it failed to load, another
package advertises the same name, or a built-in holds it.

## Running it

Once installed, the probe joins a run by its label. This makes live model calls
and is billed by the provider:

```bash
usersim simulate --locale pt_BR --num-rows 5 \
  --probe-mix "my_bookings=1.0" --out output/trajectories
```

The package has banks for `en_US` and `pt_BR` only, so a run with
`--locale fr_FR` stops before any model call and says the probe has no bank for
it. A new locale is a folder holding a `bookings.yaml` written in its language,
plus an entry in `LOCALES`. `--assets-dir` pins every probe in a run to the
folder it names, this one included, so a run that passes it needs
`my_bookings/` there too.
