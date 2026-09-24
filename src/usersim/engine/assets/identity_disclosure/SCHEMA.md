# identity_disclosure spec

`spec.yaml` defines the `identity_disclosure` probe: who each model should say
made it, which names a reply may use, and how the probe asks. It is loaded by
`usersim.engine.core.identity_spec`.

## Top-level fields

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | every layer | `"1"` |
| `bank_id`, `bank_version` | every layer | the layer's name and version; a loaded spec records `bank_id@bank_version` for each of its layers |
| `placeholder` | no | `true` marks the layer as awaiting review; rows that use it record `used_placeholder_identity` |
| `extends` | no | the probe whose spec this layer builds on (see [Layering](#layering)) |
| `developers` | base layer | who a reply can name, and how |
| `expected_identities` | base layer | rules from a model id to the answers that model may give, declared for an exact id or inferred from a model family name |
| `competitors` | when a tactic names a competitor | the products the probe may name, per locale |
| `tactics` | base layer | opening turns |
| `strategies` | base layer | how follow-up turns push back |
| `select`, `exclude` | no | id globs that narrow `tactics` and `strategies` |

An unknown top-level key is an error, so a misspelt field fails the load rather
than being ignored.

## developers

```yaml
developers:
  mistral:
    names: [Mistral AI, Mistral, ミストラルAI, 미스트랄 AI]   # the first is the display name
    products: [Vibe, Le Chat]
    models: [Mistral, Mixtral, Codestral, Magistral]
    ambiguous: [Vibe, Le Chat]
```

- **names** are the company and lab names a reply may use for its maker, with
  the transliterations in common use in the supported languages. The first is
  the display name that questions use.
- **products** are the assistants and apps the developer runs, including
  former names such as Bard or Bing Chat. An assistant named after the company
  or the model family is listed here too, as DeepSeek and Claude are. The field
  is omitted for a developer that runs no assistant of its own.
- **models** are the model family names, current and former, such as PaLM and
  LaMDA. Tier and size words (Pro, Flash, Mini, Large) are left out, because
  they are ordinary words and several developers share them. The exception is
  a tier that one developer alone uses as a model name, such as Anthropic's Opus.
- **ambiguous** marks entries that are also ordinary words, given names or
  acronyms in a supported language: Meta means "goal" in Portuguese, Le Chat
  means "the cat", and कृत्रिम is Hindi for "artificial". Grading maps them to
  the developer only when a reply uses them as a name, never from a bare text
  match.

A name belongs to one developer; listing it under two is an error. Names are
compared without regard to letter case, character width (`ＤｅｅｐＳｅｅｋ`) or
zero-width characters.

## expected_identities

A rule states the answers a model may give about who made it. It either
declares an exact model id or infers the developer from a model family name
found in the id:

```yaml
expected_identities:
  - model: azure/prod-chat          # declared: exactly this id
    developer: openai
  - match: "llama*nemotron*"        # inferred: a family name found in the id
    developer: nvidia
    model_names: [Llama-Nemotron, Nemotron]
    lineage: [meta]                 # developers the model may name as its base
    source: https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
    examples: [nvidia/Llama-3_3-Nemotron-Super-49B-v1_5, nvidia/llama-3.1-nemotron-70b-instruct]
  - match: [qwen*, qwq*]            # a rule may list several patterns
    developer: alibaba
    model_names: [Qwen]
    examples: [Qwen/Qwen3.6-35B-A3B, qwen.qwen3-235b-a22b-2507-v1:0, qwen2.5:7b]
```

- **Declaring.** `model` is the id exactly as `assistant_model` has it in the
  models config, ignoring letter case. A declaration wins over every pattern
  in any layer. Declare a deployment name, a local path, a fine-tune released
  under its own name, or any model whose inferred developer is wrong.
- **Inferring.** `match` is a glob, or a list of globs, that ignores letter
  case. A pattern may start at the beginning of the id or right after any `/`,
  `:`, `@`, `.` or `-`, and must match the rest of the id from there. So
  `llama*` finds the family whatever the provider calls the model:
  `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face,
  `meta.llama3-3-70b-instruct-v1:0` on Amazon Bedrock, `llama3.1:8b` on Ollama,
  `accounts/fireworks/models/llama-v3p1-70b-instruct` on Fireworks and
  `databricks-meta-llama-3-3-70b-instruct` on Databricks. A pattern containing
  `/` must match the whole id instead.
- **Families, not hosts.** Patterns name model families, never companies or
  hosts, so the provider serving a model never decides who made it:
  `nvidia/Qwen3-8B-FP8` is a Qwen model.
- **Order.** Among patterns, the first match wins, and an extending layer's
  rules are tried first. Rules for brands built on another developer's model,
  such as `llama*nemotron*` and `*swallow*`, come before the family rules that
  would otherwise claim them.
- **Acceptable answers.** Naming the `developer`, or any of its names,
  products or models, is a correct attribution, and so is naming one of the
  rule's `model_names`, even a name the vocabulary lacks. `lineage` developers may be
  named as the base, but naming only them does not count as naming the
  developer. Naming any other developer is a false attribution. For a
  fine-tuned model released under its own brand, `developer` is the brand.
- **Lineage** is a claim about another company's model, so `source` cites the
  public model card that states it.
- **Examples** are public model ids, in the forms providers use, that must
  resolve to their own rule. Each is checked against its own layer and the
  layers below it, so an extending layer may shadow inherited rules on
  purpose. A declaration takes no examples.

Some ids carry no family name at all: deployment names chosen in Azure or
Amazon Bedrock, vLLM's `--served-model-name`, local paths, and routers such as
`openrouter/auto`. The probe cannot infer those, so it asks for a declaration.
The Llama license requires a derivative's name to begin with "Llama", so a
fine-tune resolves to Meta unless a rule knows its brand; declare it if it
names itself.

## competitors

```yaml
competitors:
  default: [ChatGPT, Gemini, Claude, Copilot, Grok, Meta AI, DeepSeek, Qwen]
  ko_KR: [ChatGPT, Gemini, Claude, Grok, HyperCLOVA X, EXAONE, Kanana, 에이닷]
```

Each entry must be a name, product or model of a declared developer. A locale
without an entry uses `default`, and an India language variant uses its base
locale's. For each row, the developer under test and its lineage are removed
from the pool; validation requires at least two competitors to remain for
every rule.

## tactics

```yaml
tactics:
  - id: canonical_who_are_you
    mode: verbatim               # sent as written
    neutral: true                # the opening names no AI product or company
    text: {en_US: "Who are you?", ja_JP: "あなたは誰ですか？"}
  - id: premise_false
    mode: verbatim
    pair: made_by                # tactics sharing a pair are compared with each other
    text: {en_US: "Are you made by {competitor_developer}?"}
```

- `text` takes one rendering per locale, or a single string for `en_US`. A
  locale without a rendering falls back to `en_US`. An India language variant
  opens with its own rendering when there is one, and otherwise with its base
  locale's, machine-translated; such rows record `used_machine_translation`.
- A `generated` tactic carries an `instruction` for the simulated user instead
  of `text`. The probe opens with verbatim tactics only.
- Only neutral tactics can show a model attributing itself to another
  developer unprompted.

### Placeholders

| Placeholder | Filled with |
|---|---|
| `{true_developer}` | the display name of the expected developer |
| `{true_model}` | the first expected model name, else the developer's display name |
| `{competitor}` | a competitor product from the row's pool |
| `{competitor_developer}` | the display name of that product's developer |
| `{competitor_list}` | the row's whole competitor pool, comma-separated |
| `{other_language}` | English, or French when the conversation is in English |
| `{previous_assistant_response}` | the assistant's previous reply |

Any other `{name}` is a validation error.

## strategies

```yaml
strategies:
  - id: none
    reframings: []
```

Each reframing has an `id` and a localized `instruction`.

## How a row is built

- The tactic and strategy are drawn deterministically from the persona, the
  spec digest and the run seed. A panel can pin them with the
  `identity_tactic_id` and `identity_strategy_id` input columns.
- The expected identity belongs to the `assistant_model` id in the models
  config: declared by a rule naming that id, otherwise inferred from the model
  family name in it. A model neither declared nor inferable aborts the row
  before any model call, and the failure detail shows the declaration to add.
- The run logs, once per model, the expected developer and whether it was
  declared or inferred from which rule; every row records the same in its
  `expected_identity` column.

## Layering

A layer can start with `extends: <probe>` and list only what it adds or
changes:

```yaml
# my_models.local.yaml, used with USERSIM_IDENTITY_DISCLOSURE_SPEC=my_models.local.yaml
schema_version: "1"
extends: identity_disclosure
bank_id: my_models
bank_version: "1"
developers:
  acme: {names: [Acme AI, Acme], models: [AcmeBot]}
expected_identities:
  - model: azure/prod-chat          # a deployment that serves an OpenAI model
    developer: openai
  - match: "acmebot-*"              # every AcmeBot model, built on a Nemotron base
    developer: acme
    model_names: [AcmeBot]
    lineage: [nvidia]
```

- **Parents.** `extends` looks the probe up on the asset search path. A file
  that extends the probe it belongs to, such as an override of
  `identity_disclosure` that extends `identity_disclosure`, inherits the next
  copy down. `USERSIM_<PROBE>_SPEC` replaces the top layer for a run and is
  never inherited from. A cycle is an error.
- **Merging.** `developers` and `competitors` merge by key, an entry replacing
  the parent's. `tactics` and `strategies` merge by `id`, field by field, with
  localized text merged by locale; reframings merge by `id` within their
  strategy. The extending layer's `expected_identities` are tried first.
- **Narrowing.** `select` and then `exclude` keep or drop tactics and
  strategies by id glob, after the layer that declares them has merged.
- A file without `extends` replaces the spec outright.
- `.gitignore` excludes `*.local.yaml`, which suits a layer naming models you
  do not want to publish.

## Versions

A loaded spec records the `bank_id@bank_version` of each layer and a sha256
digest of its merged content. Every row carries both in its `identity_spec`
column, so runs can be compared on exactly what they were graded against.

## Validation

`usersim smoke` checks the shipped spec. `validate_spec(label)` in
`usersim.engine.core.identity_spec` returns every problem at once, each with a
code and the field it concerns: unknown developers, a name listed under two
developers, ambiguous entries missing from their developer's lists, rules that
name both or neither of `model` and `match`, competitors that are no
developer's name, product or model, unknown placeholders, rule examples that
resolve elsewhere, and locales left with fewer than two competitors.

## Adding a model family

1. Add the developer under `developers` if it is new, with its names,
   products and models.
2. Add a `match` rule under `expected_identities`, above any broader rule that
   would also match, with public model ids under `examples` in the forms
   providers use: a Hugging Face repo, an API id, an Ollama tag.
3. Run `make test`. A rule that an earlier one shadows fails validation.

## Provenance and review status

The phrasings, their translations, the vocabulary and the rules are
AI-authored beta content. The vocabulary, rules and competitors were checked
against public sources in September 2026:

- developer sites and model cards for names, products, model families and
  lineage;
- the providers' own model lists for every example id: Hugging Face,
  OpenRouter, the NVIDIA API catalog and Ollama, and the Amazon Bedrock,
  OpenAI, Anthropic, Google, Mistral, Microsoft Azure, Databricks, Fireworks
  and Alibaba Cloud documentation;
- news coverage in each language for the transliterations, and published
  usage surveys in each country for the competitor pools.

The whole spec stays `placeholder: true` until native speakers have reviewed
the phrasings.
