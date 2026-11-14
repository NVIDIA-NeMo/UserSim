# Probe substrate

A probe owns *what* a simulated user does. This page is the substrate: the
base class, the hooks, the mixins. For the catalog of what ships, see
[`docs/probes.md`](../docs/probes.md); for a worked example,
[`docs/engine/AUTHORING_A_PROBE.md`](../docs/engine/AUTHORING_A_PROBE.md).

## Registration

`@register_probe` on a `BaseProbe` subclass does everything: it checks the
class's declarations and adds it to the registry under its `label`, which
its rows store as `probe_family`. Everything else about the probe, its
`prompt_version` and `variants` included, is a class attribute.

```python
@register_probe
class OpenEndedProbe(BaseProbe):
    label = "general_open_ended"
```

The class *is* the registration; there is no separate label table to edit. A
second class registered under a taken label raises `ProbeRegistrationError`,
naming both classes, so give each probe a label of its own.

Shipped probes are listed by package in `BUILTIN_PACKAGES`
(`core/builtin_packages.py`); the registry imports each package's `probe`
module the first time a probe is looked up, which is when their decorators
run, and the scorer registry, the capability report, the evaluator and the
asset commands read the package's `scorer`, `capabilities`, `axes` and
`asset_domain` modules from the same list. Out-of-tree probes are
imported through the `usersim.probes` entry-point group instead.

## Hooks

The conversation loop in `core/simulation.py` calls most of these on the probe;
the table says where the others are called from. Only the first two are
required, plus `derive_task` on a `BankBackedProbe` and `execute_tool_call`
with `ToolExecutionMixin`; the rest have working defaults on `BaseProbe` or the
mixin that adds them.

| Hook | When |
|---|---|
| `get_user_system_prompt` | every user turn |
| `get_assistant_system_prompt` | every assistant turn |
| `get_verbatim_first_user_turn` | turn 1, when a bank supplies it literally |
| `get_user_query_instruction` | turn 1, when the user-LLM writes it from the probe's own instruction |
| `user_context` | turn 1: a line the assistant's app attaches, such as the date and where the user is, opening the first user message the assistant sees |
| `format_gate_prompt` | judging a generated user turn before accepting it |
| `format_followup_user_instructions` | turns after the first |
| `on_followup_failure` | a follow-up turn fails its gate |
| `get_tools_for_assistant` | tool-using probes |
| `transform_assistant_history` | every assistant call: the copy of the history the assistant sees, which may differ from the stored transcript |
| `after_assistant_turn` | after each assistant reply; may execute tools |
| `snapshot_turn_state` | before each assistant attempt that may be resampled: a copy of the probe's own state that a turn can change |
| `restore_turn_state` | the judge rejects an attempt that will be resampled: puts back what `snapshot_turn_state` returned |
| `format_assistant_judge_prompt` | in-sim assessment of the assistant turn, unless the run switches the assistant judge off |
| `is_capitulation_detected` | pressure probes checking for a flip |
| `allow_early_stop_at_turn` | deciding whether the conversation may end |
| `user_turn_policy` | once per trajectory: whether earlier replies may be summarized for context management, whether the last follow-ups may wrap up, which filter phrases the user may write, which names the script check skips, and a check each generated opening must pass before the gate |
| `seed_state_metadata` | recording bank/seed provenance on the row |
| `build_result_extras` | adding probe-specific columns to the trajectory, each declared in the class's `output_columns` |
| `should_succeed` | the end of the conversation: whether the probe completed its protocol; `False` fails the row |
| `derive_task` | constructing a `BankBackedProbe`: the bank entry the row runs on |
| `run_dispatch` | once per row, from the dispatcher: runs the conversation, by default through the loop |
| `execute_tool_call` | `ToolExecutionMixin`: each tool call the assistant makes |
| `parse_tool_call` | `ToolExecutionMixin`: reading a tool call's name and arguments |
| `on_tool_round_complete` | `ToolExecutionMixin`: after each round of tool calls |
| `on_tool_call_executed` | after each tool call runs: `ToolExecutionMixin` calls it, and a probe that runs tool calls itself must |
| `set_tool_call_observer` | an external host, to watch the tool calls the probe runs |

Three helpers are for the probe to call:

| Helper | Gives |
|---|---|
| `theme` | the row's theme, one of those `themes` lists; `{}` for a probe that lists none |
| `assets_dir` | the probe's asset folder, `<root>/<label>/`, where its bank loaders read |
| `options_for` | the probe's settings in a run's config, validated against its `Options`, for a classmethod such as `model_aliases`; its defaults when the config gives none |

Anything a probe does not override inherits a default, so a new probe declares
only what is specific to it; [What a probe declares](#what-a-probe-declares)
lists each declaration.

A probe's own columns are the one part of this that is declared twice. Whatever
`build_result_extras` returns is written only if the name is a core column or
appears in the `output_columns` of a probe in the run's mix; anything else is
discarded, and the scorer that reads it back finds nothing to score. The
generator logs a warning naming the probe and the column the first time it
happens. A run declares the core columns plus the union of the `output_columns`
of the probes in its mix (the config's `probes`) as
`ConversationSimulatorConfig.side_effect_columns`, because Data Designer reads
the declaration once, before any row says which probe produced it. A config
that names no probes, built outside a run, declares every registered probe's.

Per-row data that is not a column of its own goes in `state.metadata`, which is
stored with the row as `conversation_metadata` and read from there by the
probe's scorer. Write it during the run, for example in `after_assistant_turn`:
the metadata is serialized before `build_result_extras` runs, so anything added
there is lost.

The conversation loop writes eight keys of its own there, listed with what each
holds in `RESERVED_METADATA_KEYS` (`core/simulation.py`): `user_judge_ratings`,
`assistant_judge_ratings`, `frustration_events`, `capitulation_checks`,
`stopped_on_capitulation`, `capitulated_at_turn`, `user_followup_retries` and
`early_stop`. A probe reads them and writes keys of its own;
`assert_probe_conforms` reads a probe's classes and reports any that write a
reserved key.

Six hooks can reach a model, and the loop awaits them: `get_verbatim_first_user_turn`,
`format_followup_user_instructions`, `after_assistant_turn`, `is_capitulation_detected`,
`execute_tool_call` and `run_dispatch`. Declare an override of any of these with
`async def`. `usersim.testing.assert_probe_conforms` reports a plain `def` override,
which would otherwise raise partway through a conversation.

## What a probe declares

Everything else the core needs from a probe it reads from the class: an
attribute, or a classmethod where the answer depends on the locale or the run's
settings. Code outside `engine/probes/` writes a probe's label only in the
files
[`tests/test_probe_names_in_shared_code.py`](../tests/test_probe_names_in_shared_code.py)
pins it to, in a string, an identifier or a `.md` or `.yaml` file alike, so
what a new probe needs from the core goes here rather than into a branch on
its label.

| Declaration | Default | Read by |
|---|---|---|
| `label` | none: required | the registry, every stage that names the probe, and the row's `trajectory_id` and `probe_family` column |
| `prompt_version` | `"v1.0"` | the row's `trajectory_id` and provenance |
| `variants` | `("default",)` | the row's `probe_variant` when the row names none, and so its `trajectory_id` |
| `Options` | no settings | the run's pre-flight, which checks `probe_options[<label>]` against it, and the probe, as `self.options` |
| `description` | `""` | the per-row preview and the capability report, beside the probe's rows |
| `preview_fields` | `()` | the per-row preview's Probe Inputs panel |
| `default_scorers` | `()` | `usersim eval --scorers auto`, for a run that holds the probe's rows |
| `assets_root` | `None`: the override roots only | `assets_dir()`, and through it the probe's loaders, the pre-flight and smoke |
| `check_locale` | checks nothing: the probe keeps no bank per locale, and the pre-flight checks only that its asset folder holds bank files | the pre-flight, for each run locale, and the default `asset_problems`, for each locale folder the probe ships |
| `bank_locale` | an India variant's own bank when it has one, else its base locale's | the pre-flight |
| `authoring_guide` | `None` | the pre-flight, when a run asks for a locale the probe has no bank for |
| `asset_problems` | `check_locale` on each locale folder the probe ships, when it defines one; else no problems | `usersim smoke` and `assert_probe_conforms` |
| `themes` | `None`: no theme | run setup, which draws one per row into the `theme` column |
| `seed_dataset` | `None`: no seed table | run setup, which attaches `<seed_dataset>.parquet` or `.jsonl` from the probe's asset folder and gives each of the probe's rows the columns of one of its rows, and the mix's other probes' rows those columns empty (the run's config lists them in `seed_dataset_columns`); a mix with two such probes is refused; smoke and `assert_probe_conforms` check the table is there |
| `model_aliases` | `()` | run setup, which adds each alias the models file defines to every row's models, for the probes in the run's mix |
| `asset_labels` | its own label | the setup fingerprint of stored rows, which counts the asset folders of each label as the probe's part, so an edit there gives only its rows new ids |
| `output_columns` | `()` | the run's `side_effect_columns`; a column left out is dropped |
| `supports_assistant_resampling` | `True`; `False` with the tool mixins, `True` again on `tool_calling` | judge-gated assistant resampling (`max_assistant_attempts > 1`) |
| `bank_loader` | none: required on a `BankBackedProbe` | constructing a `BankBackedProbe`, for the row's locale |
| `placeholder_warning_kind` | none: required on a `BankBackedProbe` | the warning a row records when its task is a placeholder, which the capability report counts |
| `bank_version_key` | `None`: one bank per locale | where provenance records the bank's version |
| `verbatim_field` | `"verbatim_text"` | `BankVerbatimMixin`: the task field turn 1 reads |
| `turn1_instruction_template` | `""` | `CustomTurn1InstructionMixin`: the instruction turn 1 is written from |
| `tool_loop_mode` | `"single"` | `ToolExecutionMixin`: one round of tool calls per turn, or with `"multi"` rounds until the assistant stops calling tools |
| `tool_max_calls_per_turn` | `8` | `ToolExecutionMixin`: the most tool calls it runs in one turn. A call past it gets a reply saying it was not run; after such a reply, the turn ends with a reply made without tools |

## Mixins

Six mixins cover the shapes that recur, all in `core/probes.py`:

- **`BankVerbatimMixin`**: turn 1 comes from a bank, used literally.
- **`BankReframingMixin`**: turn 1 from a bank, then reframed under pressure.
- **`CustomTurn1InstructionMixin`**: turn 1 generated from an instruction
  rather than a bank.
- **`ToolCallingMixin`**: the assistant is offered tools, and the conversation
  cannot stop early until it has used one.
- **`ToolExecutionMixin`**: tool calls are actually executed against a mock
  environment.
- **`AgenticMixin`**: bank-seeded, tool-executing, multi-step.

Mixins that offer or execute tools (`AgenticMixin`, `ToolCallingMixin`,
`ToolExecutionMixin`) set `supports_assistant_resampling = False`. A resample
undoes the rejected attempt and re-calls the assistant: the loop restores the
transcript, `state.metadata` and the outcome record's counters, warnings and
traces, keeping only the judge's verdict, and the probe restores its own state
through `snapshot_turn_state` and `restore_turn_state`. That is only safe when
nothing else changed, so a probe that changes a mock environment, as
`financial_services` and `safety_agentic` do, keeps it off. `tool_calling`
turns it on: its turn changes nothing else but the credentials the user holds,
which its hooks restore. Model usage stays counted, since the calls were made.

## Assets

A built-in probe's content lives in its own folder, at
`engine/probes/<folder>/assets/<label>/`, where the folder is named after the
probe or, as `health_disclosure` is, shared by several. It
is resolved through `probe_assets_dir()`. A run fails pre-flight, before any
model call, if a selected probe's bank is missing or, for the probes whose banks
are split by locale, doesn't load for a run locale (a finance bank without
tasks, for instance). Those probes define a `check_locale` classmethod, which
the pre-flight calls on the bank the class's `bank_locale` names. The refusal names
the probe, the locale and the locales that can run, rather than failing from
inside a loader mid-run. `simulate` and its `--dry-run`, `--materialize-inputs`,
`--inputs` and notebook 01 all run the check. A probe's `asset_problems`
classmethod checks the assets it ships without a model call: `usersim smoke`
runs it for each registered probe, and `assert_probe_conforms` reports what it
finds. See [assets.md](assets.md).

A probe whose conversations vary by topic lists its themes in a `themes`
classmethod, read under its asset folder. A run draws one per row into the
`theme` column, and the probe reads it back with `theme()`. A probe that lists
none gets the placeholder `NO_THEME` in that column, which `theme()` reads as
`{}`. `tool_calling` lists none: a row's theme is the toolset it draws from its
seed table.
