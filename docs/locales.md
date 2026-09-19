# Locales and native-language evaluation

Language is a first-class axis here, not a translation layer bolted on
afterwards. Each locale draws from its own Nemotron-Personas dataset, and the
conversation happens in the persona's actual language against
locale-specific content.

That means a 67-year-old retired schoolteacher in rural Akita (Japan)
asking in Japanese about pension regulations, a 25-year-old software engineer
in Bangalore asking in English about local healthcare authentication, and a
factory worker in Minas Gerais asking in Brazilian Portuguese about SUS
referral policy all go through the simulator with persona-grounded
demographics and probe-appropriate seeds, and the results stratify cleanly
by locale, education, age cohort, and occupation in the capability dashboard.

| Locale | Language | Dataset |
|--------|----------|---------|
| `en_US` | American English | Nemotron-Personas-USA |
| `en_IN` | English (Indian) | Nemotron-Personas-India |
| `en_SG` | English (Singapore) | Nemotron-Personas-Singapore |
| `hi_Deva_IN` | Hindi (Devanagari) | Nemotron-Personas-India |
| `hi_Latn_IN` | Hindi (romanized / Latin) | Nemotron-Personas-India |
| `pt_BR` | Brazilian Portuguese | Nemotron-Personas-Brazil |
| `fr_FR` | French | Nemotron-Personas-France |
| `ja_JP` | Japanese | Nemotron-Personas-Japan |
| `ko_KR` | Korean | Nemotron-Personas-Korea |

> `hi_Latn_IN` (romanized Hindi) is a full shipped locale. Script and lingua
> checks cannot identify romanized Hindi, so its user-turn language is enforced by
> an LLM-judge clause folded into the existing per-turn gate, costing no extra model
> calls, and the deterministic language scorers skip it
> (`expected_language_name = None`). Its probe assets are transliterated from
> `hi_Deva_IN` and carry `placeholder: true` until a native reviewer signs off.

### India persona-language matching

Conversation language and persona-cohort selection are separate controls.
`--locale en_IN` conducts the simulation in English while sampling the full
India persona corpus by default. To run in English across all available
`en_IN` personas, **do not pass `--match-persona-language`**:

```bash
usersim simulate --locale en_IN --num-rows 1000 \
  --out output/trajectories
```

`--match-persona-language` deliberately narrows the cohort using the language
field declared by the sampled corpus. For the current India assets that field
is `first_language`, so combining the flag with `en_IN` selects only records
whose first language is recorded as English. It does not include every persona
who may speak English as a second or third language and is not an estimate of
India's real English-speaking population. The CLI logs the exact eligible
synthetic-persona count when the managed corpus is already available locally,
or an explicit "count unavailable" diagnostic before its first download.

Language matching also changes the cohort's correlated demographics. Use it
for an explicitly primary-language-matched population, not as a neutral way to
change only the language of an otherwise identical evaluation cohort.

### Eleven more Indian languages

India supports **11 additional languages** beyond the three locales above,
each available in native script and romanized form (22 locale codes in
total).

These work through translation because the persona itself does not change.
Census demographics, OCEAN traits, occupation and name all come from the
Indian population regardless of which Indian language the conversation is
conducted in, so the `en_IN` persona corpus is already the right population
for a Tamil or Bengali conversation. Only the language of the exchange
differs.

These are opt-in and kept separate from the shipped set. Pass the codes
straight to `--locale`, or use the `INDIA_NATIVE_LOCALES` and
`INDIA_LATIN_LOCALES` lists in
[`notebooks/01_simulate.ipynb`](../notebooks/01_simulate.ipynb) to enable the
native-script set, the romanized set, or both.

| Language | Native locale | Romanized locale | Native lingua support |
|----------|---------------|------------------|:---:|
| Bengali | `bn_Beng_IN` | `bn_Latn_IN` | ✅ |
| Gujarati | `gu_Gujr_IN` | `gu_Latn_IN` | ✅ |
| Kannada | `kn_Knda_IN` | `kn_Latn_IN` | ❌ |
| Malayalam | `ml_Mlym_IN` | `ml_Latn_IN` | ❌ |
| Marathi | `mr_Deva_IN` | `mr_Latn_IN` | ❌ (Devanagari-shared with Hindi; script-only) |
| Nepali | `ne_Deva_IN` | `ne_Latn_IN` | ❌ |
| Odia | `or_Orya_IN` | `or_Latn_IN` | ❌ |
| Punjabi | `pa_Guru_IN` | `pa_Latn_IN` | ✅ |
| Tamil | `ta_Taml_IN` | `ta_Latn_IN` | ✅ |
| Telugu | `te_Telu_IN` | `te_Latn_IN` | ✅ |
| Urdu | `ur_Arab_IN` | `ur_Latn_IN` | ✅ |

How it works, all handled for you:

- **Personas** are sampled from the `en_IN` Nemotron-Personas dataset (no new
  dataset download); the conversation is conducted in the target language.
- **Content reuses `en_IN`.** Every probe wired into these locales works: the
  verbatim turn-1 of asset-driven probes (`sov_ai_facts` /
  `sov_ai_multilingual_parity` / `safety_chat_pressure` / `safety_agentic`) is
  **machine-translated** from the `en_IN` bank into the target language (a
  single cached `summary_model` call, with false premises preserved);
  generate-and-gate probes write in the target language directly.
  `financial_services` needs a per-locale bank with its own task set, so it
  stays on the two markets that have one: `en_US` and `en_IN`.
- **Language enforcement** mirrors the shipped locales: native scripts are
  enforced by Unicode-range checks (with a real lingua scorer where the ✅
  column allows); romanized twins ride the LLM-judge language clause like
  `hi_Latn_IN`.
- **Translation is visible in the data.** Rows whose opening turn was
  machine-translated carry a `USED_MACHINE_TRANSLATION` marker, and any
  capability cell built from them inherits it, so a translated result never
  reads as a native one.
- **Persona region and language are independent.** Personas are sampled
  pan-India and the language is set explicitly, so a Telangana persona may
  converse in Bengali. That is the right trade for measuring language
  coverage; region-matched evaluation needs per-language persona data, which
  is what graduation provides.

**Graduation is automatic.** Every fallback is presence-aware: drop in a real per-language asset (a `ta_Taml_IN` fact bank, a
native persona parquet, a native prompt pack) and that path is used
automatically with **no code change**: no translation, no `USED_MACHINE_TRANSLATION`
flag. Full graduation (native assets for every probe + a native persona
dataset) promotes the language into the shipped-locale set above. See
[`docs/engine/README.md`](engine/README.md#india-language-variant-resolution)
for the `asset_locale` / `persona_dataset_locale` resolution details.

Persona datasets are pulled lazily on first use from NGC; `usersim setup-ngc`
installs the NGC CLI into your venv, so no administrator access is needed.
See [`docs/personas.md`](personas.md) for setup and licensing.
