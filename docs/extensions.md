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
[Adding a hook](../CONTRIBUTING.md#adding-a-hook) says how to ask for one, and
[When the package needs a change in User Sim](#when-the-package-needs-a-change-in-user-sim)
how to build against it before a release has it.

## Layout

```
my-bookings/
├── pyproject.toml
├── src/my_bookings/
│   ├── __init__.py
│   ├── probe.py
│   ├── scorer.py
│   ├── capabilities.py
│   └── assets/my_bookings/
│       ├── prompts.yaml
│       ├── en_US/bookings.yaml
│       └── pt_BR/bookings.yaml
└── tests/
    ├── conftest.py
    └── test_my_bookings.py
```

The banks sit under `assets/<label>/<locale>/`, the layout the asset search path
reads.

The package is in the repository, in
[`docs/examples/my-bookings/`](examples/my-bookings/), and every file this guide
shows is that file as it is, apart from its license header.
[`templates/extension_package/`](../templates/extension_package/) holds a
package with the same layout to copy, every name in it a placeholder.
`make check-extensions` installs both and runs their tests against an installed
User Sim.

## Names

Every name the package adds starts with its prefix, `my_`, because unprefixed
names belong to the built-ins:

| Name | In `my-bookings` |
|---|---|
| Probe label | `my_bookings` |
| Scorer | `my_bookings` |
| Result columns | `my_booking_id`, `my_booking` |
| Placeholder warning | `my_used_placeholder_booking` |
| Metadata key | `my_booking` |
| Entry point names | the names they register |

The conversation loop keeps its own record in `state.metadata` under the keys
in `RESERVED_METADATA_KEYS` (in `usersim.extension`); a probe reads them and
never writes them.

## `pyproject.toml`

<!-- example: pyproject.toml -->
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "my-bookings"
version = "0.1.0"
requires-python = ">=3.10"
# The User Sim releases the package works with: from the first release with
# every hook it uses, up to the next minor release.
dependencies = ["usersim>=0.3,<0.4", "pyyaml"]

[project.entry-points."usersim.probes"]
my_bookings = "my_bookings.probe:BookingProbe"

[project.entry-points."usersim.scorers"]
my_bookings = "my_bookings.scorer"

[project.entry-points."usersim.capabilities"]
my_bookings = "my_bookings.capabilities"

[tool.setuptools.package-data]
my_bookings = ["assets/**/*.yaml"]

[dependency-groups]
dev = ["pytest", "pytest-asyncio", "pytest-env"]

[tool.pytest.ini_options]
env = ["USERSIM_DISABLE_EXTENSIONS=1"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

- **`dependencies`** should name a range of User Sim releases, as the comment
  says: the first release with every hook and attribute the package uses, up
  to the next minor release. A lock file pins the exact build an environment
  runs. The copy of this package in the repository names no range, so its
  checks install it beside the User Sim they build.
- **The entry points** name the probe class, the scorer's module and the
  capabilities module.
- **`package-data`** ships the banks in the wheel; without it, an installed
  package has no assets.
- **The pytest settings** turn extension discovery off under test, run
  `async def` tests without a marker, and give each async fixture its test's
  event loop (see [Tests](#tests)).

## The probe

A probe sets the area its conversations explore and the bounds a scorer
checks, and the simulated user explores that area in the persona's own words.
Here the area is booking an appointment. The bank names what can be booked,
the probe draws a date, a time and a number of people for each row, and the
user model asks for the booking the way its persona would, so rows that draw
the same appointment still open differently. A bank's text goes to the
assistant verbatim only when its exact wording is what a probe tests, such as
a false premise.

<!-- example: src/my_bookings/probe.py -->
```python
import json
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from usersim.extension import (
    BankBackedProbe,
    PreviewField,
    ToolExecutionMixin,
    format_persona_for_prompt,
    get_prompt,
    language_instruction,
    load_prompt_file,
    persona_uuid,
    register_probe,
    render_prompt,
)

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
class Appointment:
    id: str
    what: str
    placeholder: bool


@dataclass(frozen=True)
class BookingBank:
    bank_id: str
    bank_version: str
    appointments: tuple[Appointment, ...]


@dataclass(frozen=True)
class Booking:
    id: str
    what: str
    date: str
    time: str
    people: int
    placeholder: bool


def read_booking_bank(locale_dir: Path) -> BookingBank:
    doc = yaml.safe_load((locale_dir / "bookings.yaml").read_text(encoding="utf-8"))
    return BookingBank(doc["bank_id"], doc["bank_version"], tuple(Appointment(**a) for a in doc["appointments"]))


@cache
def load_booking_bank(locale: str) -> BookingBank:
    return read_booking_bank(BookingProbe.assets_dir() / locale)


def draw_booking(appointment: Appointment, rng: random.Random) -> Booking:
    """A date, a time and a number of people for ``appointment``.

    The day and the month are different numbers no greater than 12, so the date
    reads as two different dates, day first and month first.
    """
    month = rng.randint(1, 12)
    day = rng.choice([d for d in range(1, 13) if d != month])
    return Booking(
        id=appointment.id,
        what=appointment.what,
        date=f"2027-{month:02d}-{day:02d}",
        time=f"{rng.randint(9, 17):02d}:{rng.choice(('00', '30'))}",
        people=rng.randint(2, 6),
        placeholder=appointment.placeholder,
    )


@register_probe
class BookingProbe(ToolExecutionMixin, BankBackedProbe):
    label = "my_bookings"
    description = (
        "Tests whether an assistant books what the user means: a date, a time and a number of people, "
        "written the way people write them in the user's language."
    )
    preview_fields = (PreviewField("my_booking_id", "Appointment"),)
    default_scorers = ("my_bookings",)
    assets_root = Path(__file__).parent / "assets"
    bank_loader = staticmethod(load_booking_bank)
    placeholder_warning_kind = "my_used_placeholder_booking"
    output_columns = ("my_booking_id", "my_booking")

    @classmethod
    def check_locale(cls, locale_dir: Path) -> None:
        read_booking_bank(locale_dir)

    @classmethod
    def asset_problems(cls) -> list[str]:
        problems = super().asset_problems()
        if "user_system_prompt" not in load_prompt_file(cls.label):
            problems.append("prompts.yaml has no user_system_prompt")
        return problems

    def derive_task(self, persona, bank, *, cfg):
        rng = random.Random(f"{persona_uuid(persona)}:{bank.bank_version}:{cfg.random_seed}")
        return draw_booking(rng.choice(bank.appointments), rng)

    def get_user_system_prompt(self) -> str:
        task = self._task
        prompt = render_prompt(
            get_prompt(self.label, "user_system_prompt"),
            self._data,
            persona=format_persona_for_prompt(self._persona),
            what=task.what,
            date=task.date,
            time=task.time,
            people=task.people,
        )
        return f"{prompt}\n\n{language_instruction(self._language, self._locale)}"

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
- **`get_user_system_prompt`** fills the instruction in `prompts.yaml` with the
  persona, what to book, the date and time in the tool's form, and the number
  of people, to give only if the assistant asks. The instruction is written
  once, in English, and `language_instruction` has the user write in the row's
  language and, for a locale with a script of its own, in that script; the loop
  checks each user turn's script and has a turn written again when it is wrong.
  The user model writes every user turn, the first included, in the persona's
  own words, which is what lets rows that draw the same appointment open
  differently; the gate checks each one.
  [Step 4 of the authoring guide](engine/AUTHORING_A_PROBE.md#step-4-write-the-user-models-instructions)
  says when to write the instructions in each language instead.
- **`allow_early_stop_at_turn`** keeps the conversation going until the
  assistant books, so a user who adds a thank-you to the number of people does
  not end it before the booking is made.
- **The assistant's system prompt is empty:** the tool's description is all the
  assistant under test is told, so it answers as it would for any user.
- **`description`** is what the per-row preview and the capability report
  show about the probe.
- **`preview_fields`** puts the row's appointment in the per-row preview's
  Probe Inputs panel.
- **`default_scorers`** names the package's scorer, so
  `usersim eval --scorers auto` runs it whenever a run holds the probe's rows,
  beside `language_compliance`, which checks that the assistant replied in the
  user's language.
- **`assets_root`** is the absolute directory holding `my_bookings/`, so the
  probe finds its banks wherever the package is installed.
- **`load_booking_bank`** reads under `BookingProbe.assets_dir()`, the probe's
  folder on the asset search path, so a folder `--assets-dir` or
  `USERSIM_ASSET_PATH` names reaches it too.
- **`check_locale`** reads one locale's bank, so a run in a locale with no
  bank that loads stops before any model call rather than failing every row.
  `asset_problems`, which `usersim smoke` and `assert_probe_conforms` run,
  calls it on each locale folder the package ships, so they report a bank
  that does not load, and the override adds a check that `prompts.yaml` holds
  the instruction.
- **`placeholder_warning_kind`** is the warning a row records when its
  appointment is marked `placeholder: true`. Its
  `<prefix>_used_placeholder_<thing>` form is what the capability report counts
  as placeholder content.
- **`output_columns`** declares the columns `build_result_extras` adds, so a run
  keeps them. `my_booking` holds, as JSON, what the user meant and what the
  assistant booked.
- **`derive_task`** picks the row's appointment, and `draw_booking` its date,
  time and number of people, with a generator seeded from the persona, the
  bank's version and the run's seed, never the global `random`, so a row draws
  the same booking every time it runs.

The instruction, `assets/my_bookings/prompts.yaml`:

<!-- example: src/my_bookings/assets/my_bookings/prompts.yaml -->
```yaml
user_system_prompt: |
  {persona}

  You want an assistant to book {what} on {date} at {time}. Ask for it in your
  own words, and write the date and time the way you would write them yourself,
  not in the form they are given here. Say how many people are coming only if it
  asks; the answer is {people}. Once it confirms the booking, thank it.
```

The banks, `assets/my_bookings/en_US/bookings.yaml`:

<!-- example: src/my_bookings/assets/my_bookings/en_US/bookings.yaml -->
```yaml
bank_id: my_bookings_en_US
bank_version: "1.0.0"
appointments:
  - id: museum_tour
    what: the guided tour of the city museum
    placeholder: true
  - id: restaurant
    what: a table at a restaurant
    placeholder: true
  - id: pottery_class
    what: a pottery class
    placeholder: true
```

and `assets/my_bookings/pt_BR/bookings.yaml`:

<!-- example: src/my_bookings/assets/my_bookings/pt_BR/bookings.yaml -->
```yaml
bank_id: my_bookings_pt_BR
bank_version: "1.0.0"
appointments:
  - id: museum_tour
    what: a visita guiada ao museu da cidade
    placeholder: true
  - id: restaurant
    what: uma mesa num restaurante
    placeholder: true
  - id: pottery_class
    what: uma aula de cerâmica
    placeholder: true
```

Both banks name the same three appointments, each in its bank's language.
Every date `draw_booking` picks reads as two dates. For `2027-03-06`, 6 March, a
user who writes the day first, as people in Brazil do, writes `06/03/2027`,
which an assistant that reads every date month first books for 3 June. A user
who writes the month first, as people in the United States do, writes
`03/06/2027`, which an assistant that reads every date day first gets wrong the
other way. Whether the user writes the date in numbers or in words is the user
model's choice, as it would be a person's. The user says how many people are
coming only when asked, so an assistant that books on its first reply has to
guess.

`placeholder: true` marks content no subject-matter expert has reviewed, which
for these banks means a native speaker of the bank's language; see
[`architecture/assets.md`](../architecture/assets.md).

## The scorer

<!-- example: src/my_bookings/scorer.py -->
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
        return {"scores": {"my_bookings.booked_rate": {"score": 0.0}}}
    arguments = made["arguments"]
    rates = {
        "my_bookings.booked_rate": 1.0,
        "my_bookings.date_match_rate": float(arguments.get("date") == expected["date"]),
        "my_bookings.time_match_rate": float(arguments.get("time") == expected["time"]),
        "my_bookings.people_match_rate": float(arguments.get("people") == expected["people"]),
        "my_bookings.asked_first_rate": float(made["after_user_turns"] > 1),
    }
    return {"scores": {axis: {"score": value} for axis, value in rates.items()}}


register_scorer("my_bookings", score, deterministic=True)
```

The evaluator awaits a scorer once per trajectory, with the row and the
models, and stores the mapping it returns in the row's evaluation. Its numbers
go under `scores`, one `{"score": ...}` per axis, which is where the capability
report reads them; an axis whose name ends in `_rate` is read as a share from 0
to 1. The evaluator calls the scorer on every row it evaluates, so the scorer
returns nothing for a row without a `my_booking` column, such as one from
another probe. A field the row holds empty arrives as `None`. Each argument
must match what the user meant exactly, in the form the tool's description asks
for, and `my_bookings.asked_first_rate` is 1.0 when the booking came after the
user's second turn, the first that can say how many people are coming.
`deterministic=True` says the scorer never calls a model, so `usersim rescore`
can recompute it on a stored evaluation.

## The capabilities

The capability report is built from capability definitions, not from
trajectories, so a probe without one simulates and evaluates and is then absent
from the report. `capabilities.py` names the rows the probe's evidence feeds:

<!-- example: src/my_bookings/capabilities.py -->
```python
from usersim.extension import CapabilityDefinition, EvidenceSource

CAPABILITIES = (
    CapabilityDefinition(
        id="my_booking_arguments",
        label="Bookings as users write them",
        description="Does the assistant book the date, time and party size the user meant, however they wrote them?",
        sources=(
            EvidenceSource(
                probe="my_bookings",
                scorer="my_bookings",
                axes=(
                    "my_bookings.date_match_rate",
                    "my_bookings.time_match_rate",
                    "my_bookings.people_match_rate",
                ),
            ),
        ),
        threshold=0.9,
        scale="rate",
        aggregation_policy="mean",
        next_action="Read the failing rows and compare the date the user wrote with the one the assistant booked.",
    ),
    CapabilityDefinition(
        id="my_booking_asks_first",
        label="Asking before booking",
        description="Does the assistant ask how many people are coming before it books?",
        sources=(EvidenceSource(probe="my_bookings", scorer="my_bookings", axes=("my_bookings.asked_first_rate",)),),
        threshold=0.8,
        scale="rate",
        aggregation_policy="mean",
        next_action="Read the rows booked on the first reply: the assistant guessed the party size.",
    ),
)

AXIS_DESCRIPTIONS = {
    "my_bookings.date_match_rate": "Share of bookings made for the date the user meant, however they wrote it.",
    "my_bookings.time_match_rate": "Share of bookings made for the time the user meant.",
    "my_bookings.people_match_rate": "Share of bookings made for the number of people the user gave.",
    "my_bookings.asked_first_rate": "Share of bookings the assistant made only after asking how many people.",
}
```

- **`sources`** names the probe, the scorer and the axes of the scorer's
  `scores` each row reads, and `threshold` the share a model needs to pass.
- **`AXIS_DESCRIPTIONS`** says what each rate measures, in a reviewer's terms,
  on the report's evidence cards.
- **The `usersim.capabilities` entry point** names the module, the same shape a
  probe in this repository keeps in its package's `capabilities.py`.
  [`architecture/plugins.md`](../architecture/plugins.md#capabilities-the-seam-that-is-easy-to-miss)
  describes how the report merges an extension's rows.

## Tests

`tests/conftest.py` loads the hermetic harness the built-in tests run under:

<!-- example: tests/conftest.py -->
```python
pytest_plugins = ["usersim.testing.pytest_plugin"]
```

It fails any outbound connection, stubs the tokenizer and the persona dataset,
and stops the run unless extension discovery is off and `pytest-asyncio` is
installed. With discovery off, the tests import the package to register it:

<!-- example: tests/test_my_bookings.py -->
```python
import json
import random

import my_bookings.probe  # noqa: F401  registers the probe; discovery is off under test
from my_bookings.capabilities import AXIS_DESCRIPTIONS, CAPABILITIES
from my_bookings.probe import Appointment, BookingProbe, draw_booking
from my_bookings.scorer import score

from usersim.extension import ConversationSimulatorConfig
from usersim.testing import (
    ScriptedModel,
    assert_hosted_parity,
    assert_probe_conforms,
    assert_scorer_conforms,
    run_episode,
)

PERSONA = {"first_name": "Ana", "last_name": "Souza", "age": 35, "city": "Recife"}
CONFIG = ConversationSimulatorConfig(name="conversation_messages", max_turns=3, random_seed=7, locale="pt_BR")


def test_the_probe_conforms():
    assert_probe_conforms("my_bookings")


async def test_the_scorer_conforms():
    await assert_scorer_conforms("my_bookings")


async def test_the_user_asks_for_the_drawn_booking_in_their_own_words():
    run = await run_episode("my_bookings", config=CONFIG, persona=PERSONA, language="Portuguese")

    assert run.row["conversation_status"] is True
    booking = json.loads(run.row["my_booking"])
    assert booking["expected"] == {"date": "2027-03-06", "time": "11:00", "people": 6}
    instructions = run.prompts[0]["messages"][0]["content"]
    assert "a visita guiada ao museu da cidade on 2027-03-06 at 11:00" in instructions
    opening = json.loads(run.row["conversation_messages"])[0]
    assert opening == {"role": "user", "content": ScriptedModel.CONTENT["user"]}  # the user model's own words
    assert booking["made"]["after_user_turns"] == 1  # the scripted assistant books on its first reply
    warnings = json.loads(run.row["simulation_outcome"])["warnings"]
    assert [w["kind"] for w in warnings] == [BookingProbe.placeholder_warning_kind]


def test_every_drawn_date_reads_two_ways():
    appointment = Appointment(id="museum_tour", what="the guided tour of the city museum", placeholder=True)
    for seed in range(200):
        _, month, day = draw_booking(appointment, random.Random(seed)).date.split("-")
        assert month != day and int(month) <= 12 and int(day) <= 12


async def test_a_month_first_reading_of_a_day_first_date_is_wrong():
    made = {"arguments": {"date": "2027-06-03", "time": "11:00", "people": 6}, "after_user_turns": 2}
    expected = {"date": "2027-03-06", "time": "11:00", "people": 6}
    result = await score({"my_booking": json.dumps({"expected": expected, "made": made})}, models={})

    assert {axis: cell["score"] for axis, cell in result["scores"].items()} == {
        "my_bookings.booked_rate": 1.0,
        "my_bookings.date_match_rate": 0.0,
        "my_bookings.time_match_rate": 1.0,
        "my_bookings.people_match_rate": 1.0,
        "my_bookings.asked_first_rate": 1.0,
    }


async def test_each_capability_reads_axes_the_scorer_writes_and_describes():
    made = {"arguments": {"date": "2027-03-06", "time": "11:00", "people": 6}, "after_user_turns": 2}
    expected = {"date": "2027-03-06", "time": "11:00", "people": 6}
    written = (await score({"my_booking": json.dumps({"expected": expected, "made": made})}, models={}))["scores"]

    read = {axis for capability in CAPABILITIES for source in capability.sources for axis in source.axes}
    assert read <= set(written)
    assert read <= set(AXIS_DESCRIPTIONS)


async def test_a_host_reproduces_a_standalone_run():
    await assert_hosted_parity("my_bookings", config=CONFIG, persona=PERSONA, language="Portuguese")
```

- **`assert_probe_conforms`** checks the probe's shape: its registration, its
  required methods, its class attributes and its assets.
- **`assert_scorer_conforms`** checks that the scorer is async and tolerates
  empty fields.
- **`run_episode`** runs a whole episode with scripted models, through the same
  generator as `usersim simulate`, with nothing patched and no provider. Here it
  runs in `pt_BR`: the user model's instructions, the first prompt the episode
  sends, hold the drawn booking, and the opening is what the user model
  replied, the scripted user's stock line. The scripted assistant, which calls a
  tool as soon as one is offered, books on its first reply.
- **`draw_booking`** is tested on its own over many seeds, since one episode
  draws one date.
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

In a workspace that holds several packages, run each package's tests from
inside its folder. Its `tests/conftest.py` loads the harness through
`pytest_plugins`, which pytest accepts only in the conftest at the root of the
run, so one `pytest` over the whole workspace stops before running a test.

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
it. A new locale is a folder holding a `bookings.yaml` written in its language.
`--assets-dir` pins every probe in a run to the
folder it names, this one included, so a run that passes it needs
`my_bookings/` there too.

A probe that a run can tune declares its settings in an `Options` class, and
a run gives them as `--probe-option my_bookings.<setting>=<value>`; see
[Settings a run can give the probe](engine/AUTHORING_A_PROBE.md#settings-a-run-can-give-the-probe).
When stored rows run again (`usersim simulate --inputs`), the probe's assets,
prompt version, settings and the package's release are its own part of the
setup: a release of the package gives new ids to its rows only.

## When the package needs a change in User Sim

Ask for the hook, attribute or fix in general terms first, through the
[Extension hook form](https://github.com/NVIDIA-NeMo/UserSim/issues/new?template=extension-hook.yml)
or an issue, so the change serves other packages too.

To build against the change before it is released, run the package's
commands against a User Sim checkout, without recording it:

```bash
uv run --with-editable ../UserSim pytest
```

`uv run` reinstalls the locked `usersim` before each command, so a checkout
installed once with `uv pip install -e` is replaced by the next `uv run`;
`--with-editable` adds it for that command. Keep the checkout out of
`pyproject.toml` and the lock file: a path dependency installs only on your
machine. Once the change is merged, the package can depend on that
commit of `main` until a release includes it, then on a range that starts at
that release:

```toml
dependencies = ["usersim @ git+https://github.com/NVIDIA-NeMo/UserSim@<commit>"]
```

While you wait, work around the gap only with what `usersim.extension` offers:
subclass or wrap a public class, or keep per-row data in `state.metadata` under
your prefix. Mark each workaround with the issue it waits on, so it goes when
the change lands. A copied or edited User Sim file, a change to `usersim` at
import time, or a fork of the repository breaks with the next release and
hides the need from everyone else.
