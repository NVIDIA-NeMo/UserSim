# Extension API

An extension imports from two modules: `usersim.extension`, for writing probes,
scorers, capabilities, asset domains and execution backends, and
`usersim.testing`, for testing them. This page lists every name the two export.
A probe's hooks and declarations are part of the same contract;
[`architecture/probes.md`](../architecture/probes.md) lists them, and
[`architecture/plugins.md`](../architecture/plugins.md) lists the entry-point
groups an extension registers through.

A host that runs probes from its own loop uses `usersim.engine.external`,
described in [`docs/engine/EXTERNAL_PROBE_RUNTIME.md`](engine/EXTERNAL_PROBE_RUNTIME.md).
Other `usersim` modules are internal, and code moves between them; import a
name on this page from `usersim.extension`, wherever it is defined. These names
can change in any release as well.

```python
from usersim.extension import BaseProbe, register_probe, register_scorer
from usersim.testing import assert_probe_conforms
```

`usersim.extension` imports each name from the module that defines it the
first time it is used, so an asset domain or a backend that imports from it
does not load the conversation loop.

## `usersim.extension`

### Probes

| Name | What it is |
|---|---|
| `BaseProbe` | The class every probe subclasses, with a default for each optional hook |
| `BankBackedProbe` | A probe whose rows each run on an entry from a bank it loads per locale |
| `BankVerbatimMixin` | Turn 1 is a bank entry's text, sent as written |
| `BankReframingMixin` | Turn 1 from the bank, and follow-ups that reframe it under pressure |
| `CustomTurn1InstructionMixin` | Turn 1 written by the user model from the probe's own instruction |
| `ToolCallingMixin` | The assistant is offered tools, and the conversation cannot end early before it uses one |
| `ToolExecutionMixin` | The assistant's tool calls run against the probe's mock environment |
| `AgenticMixin` | One user turn from the bank, then the assistant's tool loop |
| `register_probe` | The class decorator that checks a probe's declarations and adds it to the registry under its label |
| `ProbeRegistrationError` | Raised when `register_probe` refuses a class, such as one under a label another class holds |
| `PreviewField` | One line of the per-row preview's Probe Inputs panel, for a probe's `preview_fields` |
| `LocalePromptPack` | A probe's prompts by locale; constructing one fails when a locale it is given has no prompt |
| `BankLoadError` | Raised when a `BankBackedProbe` cannot load its bank |
| `UserTurnPolicy` | What a probe changes about how the loop writes and checks user turns, returned by its `user_turn_policy` hook |
| `probe_description` | The `description` of the probe registered under a label, or empty when no installed probe holds it, so a report can describe a stored row's probe |

### Conversations and outcomes

| Name | What it is |
|---|---|
| `ConversationState` | The conversation so far, as the hooks receive it, with its messages, metadata and `OutcomeBuilder` |
| `RESERVED_METADATA_KEYS` | The `ConversationState.metadata` keys the conversation loop writes, with what each holds; a probe reads them and writes keys of its own |
| `dispatch_episode` | Runs one episode through a probe's `run_dispatch` with the run's per-episode settings installed, as `usersim simulate` runs it |
| `make_result` | Builds the row a conversation produces, for a probe that writes its own `run_dispatch` |
| `make_failed` | Builds the row for a conversation that failed |
| `assistant_message` | An assistant message for the conversation, carrying the reasoning of the model call that wrote it, for a probe that appends its own |
| `finalize_completed` | Finishes a conversation that ran to its end, failing it when the probe did not complete its protocol |
| `OutcomeBuilder` | Collects the row's structured outcome during a conversation |
| `outcome_builder_scope` | Records the model calls made inside a `with` block on an `OutcomeBuilder`, for a probe whose `run_dispatch` runs its own loop |
| `OutcomeStatus` | Whether a row completed, completed with warnings, or failed |
| `FailureClass` | How a row failed |
| `FailureAttribution` | What caused a failure: one of the models, the probe's own logic, or infrastructure |
| `Provenance` | What a row was built from: persona dataset, prompt and bank versions, code |
| `WarningKind` | The warning kinds the engine itself records; a probe records its own, such as its `placeholder_warning_kind`, as a string |
| `SimulationTrace` | One check or model call during the conversation: its kind, turn, alias, tokens and latency |
| `TraceKind` | Which check or call a `SimulationTrace` records |

### Models and settings

| Name | What it is |
|---|---|
| `MODEL_USER` | The alias of the model playing the user |
| `MODEL_ASSISTANT` | The alias of the assistant under test |
| `MODEL_JUDGE` | The alias of the model that checks user turns and judges assistant turns during the conversation, and checks tool_calling's simulated tool responses when the models file has no `api_response_judge_model` |
| `ConversationSimulatorConfig` | The run's settings, which hooks receive as `cfg` |
| `acall_llm` | Calls the model behind an alias and returns its reply as a message |
| `run_inline_judge` | Asks a judge alias for an `<explanation>` and a `<rating>`, reformatting once if needed; returns the explanation, the rating and whether it passed (`success` or `warn`) |
| `run_inline_judge_ex` | `run_inline_judge` plus whether the reply could be read: `False` when even the reformatted reply has no valid `<rating>`, so a caller can tell an unreadable verdict from a rejection |
| `resolve_model_name` | The model id behind an alias, or the alias when its model does not expose one |

### Prompts

| Name | What it is |
|---|---|
| `format_persona_for_prompt` | A persona as a text block for the user model's prompt |
| `persona_full_name` | A persona's name in the order its own script writes it |
| `persona_uuid` | A persona's content-hashed identifier |
| `format_behavioral_profile_for_prompt` | A persona's behavioral profile as a prompt block |
| `ChatRealism` | Which chat-writing rules `format_behavioral_profile_for_prompt` gives the user |
| `format_interaction_style_instructions` | The prompt block for a user's interaction style |
| `format_disclosure_instructions` | The prompt block for a user's disclosure style |
| `format_conversation_history_for_prompt` | A conversation as a text block for a judge prompt |
| `format_tools_for_prompt` | Tool definitions as a readable prompt block |
| `render_prompt` | Fills a template's placeholders from given values, then from the row |
| `load_prompt_file` | A probe's `prompts.yaml` as a mapping, read from the folder its banks resolve to; empty when there is none |
| `get_prompt` | One prompt from a probe's `prompts.yaml`, or the default it is given when the file or the key is missing |

### Banks and assets

| Name | What it is |
|---|---|
| `probe_assets_dir` | The folder a probe's assets resolve to: the first override root that holds them, else the root its class declares |
| `probe_asset_roots` | The roots searched for a probe's assets, highest priority first, for assets layered across them |
| `asset_roots_in_use` | Every root an asset lookup can read, for keying a cache by the roots a run uses |
| `load_seeds` | A probe's seeds of one kind for a locale, such as its topics, from `<probe>/<locale>/<kind>.json`, or the `en_US` file when the locale has none |
| `theme_problems` | One line for each of the given locales whose themes a probe's `themes` cannot read, for its `asset_problems` |
| `LocalizedText` | Bank text with a rendering per locale, read with `for_locale`, which falls back to `en_US` |
| `LocalizedTextError` | Raised when a bank value cannot be read as `LocalizedText` |
| `persona_to_tags` | The tags a persona matches bank entries on, such as its region, age and occupation, for a locale |
| `education_ordinal` | A persona's education level as a score from 0 to 1, read for its locale |

### Probing taxonomies

A probing taxonomy is a bank of topic categories, one file per locale, that a
probe draws a persona's opening from. The file format is in the
[`sov_ai_dynamic` schema](../src/usersim/engine/probes/sov_ai_dynamic/assets/sov_ai_dynamic/SCHEMA.md).

| Name | What it is |
|---|---|
| `ProbingTaxonomy` | A loaded taxonomy: its categories, id and version, which it also gives as `bank_id` and `bank_version` |
| `Category` | One category of a taxonomy: its invitation, its sub-topic hints, and the entity types and persona tags it suits |
| `ProbingTaxonomyError` | Raised when a taxonomy file is missing or breaks the schema |
| `load_probing_taxonomy` | Reads and validates one taxonomy file |
| `load_probing_taxonomy_for` | A probe's taxonomy for a locale, read once and cached, so a probe and its scorer read the same version |
| `taxonomy_path_for` | The file `load_probing_taxonomy_for` reads: the one an environment variable names, else the one in the probe's asset folder |
| `select_category` | A category and sub-topic hint for a persona, the same for the same persona, taxonomy version and seed |
| `reset_probing_taxonomy_cache` | Empties the cache `load_probing_taxonomy_for` reads |

### Locales and languages

| Name | What it is |
|---|---|
| `SHIPPED_LOCALES` | The locales User Sim ships |
| `asset_locale` | The locale whose banks a conversation locale reads: an India language variant reads its base locale's unless it has its own |
| `india_variant` | The India language variant a locale is, with its base locale, language and script, or None for any other locale |
| `get_conversation_language` | The language a locale's conversations are in |
| `expected_language_display` | A locale's language as a label for prompts |
| `language_instruction` | The prompt block telling a model which language, and for some locales which script, to write in |
| `expected_script_ranges` | The Unicode ranges of the scripts a locale's text is written in; empty for a locale User Sim does not know |
| `SCRIPT_BUCKETS` | The Unicode ranges of each script User Sim tells apart, by name, such as `han` and `hangul` |
| `persona_dataset_locale` | The locale whose persona dataset a conversation locale samples: an India language variant samples its base locale's unless its own dataset is in `datasets_dir` |
| `translate_search_query` | A search query translated into the language of the documents it searches, or the query unchanged when translation fails |

### Tool calls

| Name | What it is |
|---|---|
| `parse_tool_call` | A tool call's name and arguments |
| `recover_tool_call` | The arguments of a tool call from a completion, falling back to JSON in the reply's text |

### Scorers

| Name | What it is |
|---|---|
| `register_scorer` | Adds an async scorer function to the registry under a name |
| `ScorerFn` | The type of a scorer: an async function of the row and the models |
| `ScorerRegistrationError` | Raised when `register_scorer` refuses a name |
| `is_missing` | Whether a table cell is empty: None, NaN, `pd.NA` or `pd.NaT` |
| `none_for_missing` | A copy of a row with every empty cell as None |
| `recorded_value` | A value a probe recorded for a row, from the row or from its `conversation_metadata` |

### Capabilities, asset domains and backends

| Name | What it is |
|---|---|
| `CapabilityDefinition` | A row of the capability report and the evidence it is scored from |
| `EvidenceSource` | One probe, scorer and set of axes a capability reads |
| `ALL_PROBES` | The probe an `EvidenceSource` names to read the rows of every probe |
| `AssetDomain` | A domain `usersim gen-assets` can build banks for |
| `to_model_configs` | The Data Designer model configs for the models an asset domain receives |
| `to_data_designer_kwargs` | The `DataDesigner` keyword arguments for those models' providers |
| `require_aliases` | Raises a configuration error naming the aliases a domain needs that the models lack |
| `ExecutionBackend` | Somewhere a run can be submitted, for `--backend` |
| `JobSpec` | What to run, independent of where |
| `JobHandle` | A backend's reference to work it accepted |
| `JobStatus` | Where a submitted job is, and why it stopped if it did |

### Runs

For a tool that reads what `usersim simulate` or `usersim eval` wrote, such as a report or a dashboard.

| Name | What it is |
|---|---|
| `list_runs` | The run ids stored under an output folder, oldest first |
| `latest_run_id` | The most recent run id stored under an output folder, or None when there is none |
| `read_partitioned_dataset` | The rows stored under an output folder, or in one parquet file, as one table; `run=` reads one run and `columns=` only the columns named |

## `usersim.testing`

These checks run against every built-in probe and scorer as well.
[`architecture/plugins.md`](../architecture/plugins.md#testing-an-extension)
describes `usersim.testing.pytest_plugin`, the test harness an extension's
suite can load.

| Name | What it is |
|---|---|
| `assert_probe_conforms` | Fails with every problem found in a probe's class and declarations |
| `probe_conformance_problems` | The same findings, as a list |
| `assert_scorer_conforms` | Fails with every problem found in a scorer, optionally scoring a sample row |
| `scorer_conformance_problems` | The same findings, as a list |
| `run_episode` | Runs one episode of a probe offline, as `usersim simulate` runs it, with scripted models or ones you give by alias |
| `EpisodeRun` | What `run_episode` returns: the row and every prompt sent |
| `ScriptedModel` | A model stand-in that answers from a script |
| `assert_hosted_parity` | Fails when a probe run through a host differs from the same probe run alone |
| `hosted_parity_report` | Runs one episode alone and through a host, and compares the two |
| `HostedParityReport` | What `hosted_parity_report` returns |
| `registry_snapshot` | A context manager that puts the probe and scorer registries back as they were, for a test that registers its own |
