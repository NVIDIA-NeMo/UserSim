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
| `register_probe` | The class decorator that sets a probe's family, prompt version and variants and adds it to the registry |
| `ProbeRegistrationError` | Raised when `register_probe` refuses a class, such as one under a label another class holds |
| `PreviewField` | One line of the per-row preview's Probe Inputs panel, for a probe's `preview_fields` |
| `LocalePromptPack` | A probe's prompts by locale; constructing one fails when a locale it is given has no prompt |
| `BankLoadError` | Raised when a `BankBackedProbe` cannot load its bank |
| `UserTurnPolicy` | What a probe changes about how the loop writes and checks user turns, returned by its `user_turn_policy` hook |

### Conversations and outcomes

| Name | What it is |
|---|---|
| `ConversationState` | The conversation so far, as the hooks receive it, with its messages, metadata and `OutcomeBuilder` |
| `make_result` | Builds the row a conversation produces, for a probe that writes its own `run_dispatch` |
| `make_failed` | Builds the row for a conversation that failed |
| `finalize_completed` | Finishes a conversation that ran to its end, failing it when the probe did not complete its protocol |
| `OutcomeBuilder` | Collects the row's structured outcome during a conversation |
| `OutcomeStatus` | Whether a row completed, completed with warnings, or failed |
| `FailureClass` | How a row failed |
| `FailureAttribution` | What caused a failure: one of the models, the probe's own logic, or infrastructure |
| `Provenance` | What a row was built from: persona dataset, prompt and bank versions, code |
| `WarningKind` | The built-in warning kinds a row can record |
| `SimulationTrace` | One check or model call during the conversation: its kind, turn, alias, tokens and latency |
| `TraceKind` | Which check or call a `SimulationTrace` records |

### Models and settings

| Name | What it is |
|---|---|
| `MODEL_USER` | The alias of the model playing the user |
| `MODEL_ASSISTANT` | The alias of the assistant under test |
| `MODEL_JUDGE` | The alias of the model that checks user turns and judges assistant turns during the conversation |
| `ConversationSimulatorConfig` | The run's settings, which hooks receive as `cfg` |
| `acall_llm` | Calls the model behind an alias and returns its reply as a message |

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

### Locales and languages

| Name | What it is |
|---|---|
| `SHIPPED_LOCALES` | The locales User Sim ships |
| `get_conversation_language` | The language a locale's conversations are in |
| `expected_language_display` | A locale's language as a label for prompts |
| `language_instruction` | The prompt block telling a model which language, and for some locales which script, to write in |

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

### Capabilities, asset domains and backends

| Name | What it is |
|---|---|
| `CapabilityDefinition` | A row of the capability report and the evidence it is scored from |
| `EvidenceSource` | One probe, scorer and set of axes a capability reads |
| `AssetDomain` | A domain `usersim gen-assets` can build banks for |
| `ExecutionBackend` | Somewhere a run can be submitted, for `--backend` |
| `JobSpec` | What to run, independent of where |
| `JobHandle` | A backend's reference to work it accepted |
| `JobStatus` | Where a submitted job is, and why it stopped if it did |

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
| `run_episode` | Runs one episode of a probe offline, as `usersim simulate` runs it |
| `EpisodeRun` | What `run_episode` returns: the row and every prompt sent |
| `ScriptedModel` | A model stand-in that answers from a script |
| `assert_hosted_parity` | Fails when a probe run through a host differs from the same probe run alone |
| `hosted_parity_report` | Runs one episode alone and through a host, and compares the two |
| `HostedParityReport` | What `hosted_parity_report` returns |
