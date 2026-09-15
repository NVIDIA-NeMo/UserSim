# financial_services asset schema

Region-grounded, hybrid-agentic financial-services probe. A locale is a SET of
**institutions** (brands); each spans one or more **domains**. Layout:

```
assets/financial_services/<locale>/
  region_meta.yaml                     # locale-level: identity + per-domain regulators
  <institution_type>/<institution_id>/ # grouped by type (a stable taxonomy), not brand
    institution_meta.yaml              # brand, type, domains
    corpus.yaml                        # whole, short, atomic documents
    tools.yaml                         # tool taxonomy (permanent + discoverable)
    tasks.yaml                         # declarative task templates + gold contract
```

Institutions are grouped under an **institution-type** directory (`bank/`,
`brokerage/`, `retirement_provider/`, `wealth_manager/`, ...) so the layout stays
stable when a brand is renamed and multiple brands can share a vertical. The loader
discovers institutions by finding each `institution_meta.yaml` (layout-agnostic).

All files share `schema_version: v0.1`. All AI-authored beta content carries
`placeholder: true`; any trajectory that touches a placeholder document or
template emits `WarningKind.USED_PLACEHOLDER_FINANCE` (scorecards stay preview-only).

Every document / tool / template is scoped to its institution (`institution_id`
comes from `institution_meta.yaml`; the leaf directory is named for it) and carries a
`domain`. The runtime (probe + scorer) reads everything from the committed bank and
never imports `asset_gen`.

## region_meta.yaml (locale-level)

`schema_version`, `locale`, `bank_id`, `bank_version`, `task_contract_version`
(pins the gold-derivation CODE contract in `core/finance_tasks.py`, independent
of `bank_version`), `region`, `currency`, `currency_symbol`, `number_format`,
`privacy_regime`, `placeholder`, `provenance`, and `domain_regulators`:

```yaml
domain_regulators:
  <domain>:
    regulators: [<name>, ...]
    regulator_text: >-   # rubric text the scorer's finance.regulatory_compliance uses
```

Every domain any institution offers MUST have a `domain_regulators` entry
(enforced at load time). Regulation is domain-specific (retail_banking =
CFPB/Fed/FDIC, brokerage_investing = SEC/FINRA, retirement = DOL/IRS, ...).

### Region vs. language (`extends`)

A `region_spec` is keyed by LOCALE, but institutions, products, typed values,
tools and regulators are properties of the REGION: identical across every
language spoken there. India alone has three surfaces (`en_IN`, `hi_Deva_IN`,
`hi_Latn_IN`), so a language variant inherits the region model instead of copying
it (which would drift the first time a fee changed in only one file):

```yaml
extends: en_IN                    # the India region model
locale: hi_Deva_IN
language: Hindi (Devanagari)
institutions:
- id: chandrika_bank              # merged onto the base entry BY id
  display_name_local: चंद्रिका बैंक
```

Merge rules: mappings deep-merge; a list whose every entry is a mapping with an
`id` (`institutions`, `products`, `task_templates`) merges per `id`, so an overlay
can patch one entry without restating the rest; every other list replaces
wholesale (`document_types` is a choice, not a patch, and `tool_taxonomy` is keyed
by `name`). An overlay **cannot delete** an inherited entry or null out an
inherited field: a variant that needs a different institution set should be
authored standalone rather than as an overlay. `extends` is resolved by
`load_region_spec` (which knows the directory); passing a mapping carrying
`extends` to `validate_region_spec` is rejected rather than silently
un-inherited. Single-language regions (`fr_FR`, `pt_BR`) need none of this: one
spec, no `extends`.

`display_name` stays in the LATIN script even for non-Latin locales, with
`display_name_local` carrying the in-language rendering. Two reasons: the Latin
form survives transliteration untouched, so a derived romanized locale shows the
same brand rather than a re-romanized respelling; and authoring the local name
(instead of leaving it to the generator) keeps one product named consistently
across a whole corpus instead of re-translated per document.

An institution's `display_name_local` is carried through into the generated
`institution_meta.yaml` (omitted when unset) and read back at simulation time via
`Institution.brand_name()`, so the conversation refers to the brand exactly as the
retrieved documents do. Without that round-trip the customer would say
"Chandrika Bank" while the knowledge base said "चंद्रिका बैंक".

```yaml
doc_titles:                       # optional; in-language title vocabulary
  topic_separator: ' — '
  product_genre_angles:           # per document_type; overrides one genre at a time
    product_sheet: [सिंहावलोकन, शुरुआत कैसे करें]
  faq_variable_template: '{v} की व्याख्या'   # {v} slot is required
  regulatory_note_topic: '{domain} के लिए नियामक नोट्स'
  tool_policy_topic: '{tool} का उपयोग कब करें'    # {tool} stays VERBATIM
  variable_labels: {below_mab_charge: MAB से कम शेष होने पर शुल्क}
  domain_labels: {retail_banking: रिटेल बैंकिंग}
```

A document title is minted deterministically by the doc plan as
`<product><topic_separator><angle>`, and that exact string is the cross-reference
handle every sibling is told to cite, so it must be knowable before generation,
which means the angle words cannot be left to the generator to translate. They
are reader-facing prose, so like `display_name_local` they belong to the LANGUAGE
overlay. Skipping them is what made the first `hi_Deva_IN` corpus 93% Devanagari
in its bodies but only 56% in its customer-facing titles: the plan handed the
model an English title while the prompt simultaneously required it be reproduced
exactly, and the title+body script check could not see the split because a
40-character English title inside a 2,000-character Hindi body still scores ~0.93.

Every field is optional and falls back to the English default in `pipeline.py`, so
`en_US` / `en_IN` omit the block entirely. An unmapped `variable_labels` /
`domain_labels` entry falls back to the machine field name with underscores
stripped, which is why `en_IN` ships titles like `— below mab charge explained`
and `Regulatory notes for retail_banking`. Not serialized into `region_meta.yaml`;
it only shapes generation.

```yaml
concept_guidance:                 # optional; per-institution-TYPE brief emphasis
  retirement_provider: >-
    Emphasize contribution limits, RMDs, early-withdrawal penalties, ...
```

Market-specific framing lives here, in the `region_spec` (generation input), not
in the shared doc-gen prompt: US instruments (RMDs, Regulation Best Interest,
Advisers Act fiduciary duty) would otherwise be injected into every locale's
corpus. Types you omit fall back to a region-neutral default
(`prompts.DEFAULT_TYPE_CONCEPT_GUIDANCE`), so a new locale never silently
inherits another market's concepts. Not serialized into `region_meta.yaml`: it
only shapes generation.

```yaml
identity_verification:            # optional; the KYC contract for this region
  required_fields: [full_name, date_of_birth, pan]
```

What an agent must collect before acting on an account differs by market, so it
is regional DATA rather than code: the US stops at `full_name` + `date_of_birth`
(the default when the block is omitted), while India also expects a `pan`.
Known fields: `full_name`, `date_of_birth`, `pan`, `aadhaar_last4`,
`national_id_last4`.

The required set is both **advertised** in the offered `verify_identity` schema
and **enforced** by the simulator's gate: an agent that was never told a PAN is
required could not collect one, so gating on a hidden requirement would score a
failure it had no way to avoid. The simulated customer is grounded with a value
for each required field (a deterministic, fictional PAN in India's case).

```yaml
account_labels:                   # optional; local product vocabulary per domain
  retail_banking: savings account
  payments: wallet account
```

What this market calls the account the simulator's read tools report back, keyed
by domain. Regional DATA rather than code, because naming an account after a
product the market does not have reads as the wrong institution to the customer:
"checking account" exists in the US and essentially nowhere else, while India
holds savings / current accounts and a UPI-first neobank holds a wallet.

Resolution order for the reported label, most specific first: a task's authored
`account_state.account_type`, then this block, then a region-NEUTRAL per-domain
default (`deposit account`, `payment account`, `loan account`, `policy`, ...). A
locale that omits the block is therefore generic rather than silently
US-flavoured. Keys must be known domains.

## <institution_id>/institution_meta.yaml

`schema_version`, `institution_id`, `display_name`, optional
`display_name_local` (the in-language brand for non-Latin locales; see
"Region vs. language" above), `type` (bank | neobank |
credit_union | brokerage | wealth_manager | robo_advisor | retirement_provider |
insurer | nbfc), `brand_voice` (traditional | neobank | boutique - one per
institution), `domains` (list), `placeholder`.

`nbfc` is a deposit-less lender (consumer/EMI finance, gold loans, vehicle
loans): a dominant retail-credit channel in markets like India. It may offer
`lending_mortgage` / `payments` but NOT `retail_banking`, and its injected
common operations are the loan lifecycle (`apply_loan` / `pay_emi` /
`foreclose_loan` / `update_repayment_mandate` / `get_loan_details`) rather than
the deposit-servicing set.

## corpus.yaml

`documents`: each `id`, `title`, `body`, `domain`, `product_category`,
`document_type` (see below), `source_authority` (official | marketing |
community), `placeholder`, `mentions_tools` (the discoverable-tool mechanic: a
tool is only callable once a doc mentioning it has been retrieved).

Genres come in three scopes, decided by the doc plan in `pipeline.py` (NOT by the
region spec's `document_types`, which is a declared inventory: a test binds the
two so they cannot drift):

| scope | genres |
| --- | --- |
| **product**: one set per product | product_sheet, fee_schedule, rate_sheet, faq, how_to_guide, troubleshooting, comparison, eligibility_matrix, terms_conditions, disclosure, promo_notice, security_advisory |
| **tool**: one pair per discoverable tool | policy_procedure, discoverable_tool_doc |
| **institution**: one each, per institution | regulatory_note (per domain), fee_schedule (institution-wide overview), faq (general) |

Relationship-level servicing questions: what identity and address evidence is
accepted, what it takes to open and what must settle before closing, who inherits,
why a transfer has not arrived, how to escalate: are answered **per product**, not
by institution-wide documents, because in this domain they genuinely differ per
product: a 401(k) requires an employer sponsorship letter where a checking account
does not, and transfer limits are product *variables*. Four product angles carry
them, and `build_product_doc_plan` guarantees all four regardless of a product's
variable count:

| question | angle |
| --- | --- |
| what evidence do I need? | `eligibility_matrix — required documentation` |
| how do I open this? | `how_to_guide — set up and activate` |
| how do I close it, and what settles first? | `how_to_guide — cancel, close, or downgrade` |
| my transfer/action did not complete | `troubleshooting — a failed or pending action` |

These were previously reachable only by accident: they sat behind per-variable
angles that consumed every slot a genre received, so in en_US exactly the seven
products with two variables got them and the other eleven did not. See
`build_product_doc_plan` for the ordering that fixes it.

## tools.yaml

`tools`: each `name`, `description`, `discoverable` (bool; non-discoverable tools
are offered up front, discoverable ones only via the `call_tool` gate),
`side_effect_class` (read_only | state_changing | irreversible), `parameters`
(OpenAI-style JSON schema).

Two layers land here automatically at generation time and need not be listed in a
region spec: (1) the framework primitives `kb_search` + `verify_identity` (schema
backfilled from `core.finance_bank.FRAMEWORK_PRIMITIVE_TOOLS`); (2) the
**common-operations** layer: account-lifecycle + servicing tools every
institution offers (`open_account`, `close_account`, `update_contact_info`,
`get_statements`, `update_beneficiary`, `set_alerts`) plus a fuller per-type set
(deposit institutions: `transfer_funds`, `order_replacement_card`,
`stop_payment`, `pay_bill`, `activate_card`, `dispute_status`,
`manage_authorized_users`; brokerages: `cancel_order`, `get_order_status`,
`deposit_funds`, `withdraw_funds`; retirement: `request_withdrawal`,
`update_investment_election`, `plan_loan`; wealth: `fund_account`,
`generate_financial_plan`; insurer: `file_claim`, `get_claim_status`,
`update_policy`), injected by `asset_gen/financial_services/spec.py`. Most are
never gold: they are realistic, gold-ADJACENT tool docs that make retrieval
(and the discoverable-tool gate) non-trivial, and they stay consistent across
locales.

## tasks.yaml

A task is a **parameterized generator**, not a fixed string. `tasks.yaml` holds
`verifiable`-tier templates only; the open-ended `dynamic` tier is taxonomy-driven
(see `dynamic.yaml` below), not scripted here. `templates`: each `id`, `tier`
(verifiable), `task_type`, `domain`, `placeholder`, `persona_tags` (see below),
`opening_user_messages` (a LIST of `{{slot}}` templates; author >=3),
`account_state` (seed state the tools mutate), `param_space` (per-instance
slot values), `gold_tool_sequence`, `ordered` (OPTIONAL, default `false`),
`gold_document_ids` (OPTIONAL -- see below), `expected_state_deltas`,
`allowed_tools` (OPTIONAL). Verifiable templates carry a
derivable gold spec whose docs/tools MUST live within the SAME institution (scope
integrity, enforced at load time). `allowed_tools` lists state-changing tools that
are PERMITTED but not required (e.g. `freeze_card` during a fraud dispute): the
verifier requires `gold_tool_sequence` but does not flag a state change in
`gold_tool_sequence ∪ allowed_tools` as unauthorized; they must also be in-scope.

`ordered` controls whether the verifier enforces the ORDER of `gold_tool_sequence`.
Default `false`. Set it `true` only when EVERY consecutive pair is a genuine
dependency (redeem before withdraw; issue a card before activating it). A chain
whose tail is interchangeable (open an account, then add a nominee and enable
alerts in either order) must stay unordered, or a correct agent fails on order
alone.

**`persona_tags` must come from the vocabulary `persona_to_tags` actually emits**,
which is exactly four prefixes:

| Prefix | Values |
| --- | --- |
| `age:` | `18-24`, `25-34`, `35-44`, `45-54`, `55-64`, `65+` (the `_age_bin` splits) |
| `financial-literacy:` | `high`, `medium`, `low` (derived from education + occupation) |
| `region:` | free-form slug of the persona's region/state |
| `occupation-family:` | free-form slug of the persona's occupation |

`<prefix>:any` is a wildcard that always matches. **Anything else silently never
matches**: template matching requires every tag to be literally present on the
persona, and the unmatched-fallback pool only engages when NO template matches, so
a task carrying an unknown tag is simply never selected. This is not hypothetical,
11 `solstice_invest` tasks were authored against an `interest:finance` tag that this
document used to list but the code never emitted, and they never ran. The
`TestAuthoredPersonaTagsAreReachable` guard now fails CI on any unreachable tag.

## dynamic.yaml (locale-level; the `dynamic` tier taxonomy)

The open-ended, judge-scored `dynamic` tier generates turn-1 from an authored
per-locale taxonomy instead of scripted templates. One `dynamic.yaml` per locale sits
beside `region_meta.yaml`. Fields: `schema_version`, `locale`, `taxonomy_id`,
`taxonomy_version`, `placeholder`, and `categories[]` with `id`, `applies_to_types`
(institution TYPE(s) the category applies to: `bank` / `neobank` / `brokerage` /
`retirement_provider` / `wealth_manager` / `insurer` / `nbfc`), `invitation` (a
topic boundary in the persona's voice -- NOT a question), `subtopic_hints` (>=2
neighborhoods -- NOT questions; the loader rejects <2), and an optional
`persona_affinities` (persona tags like `age:65+` / `financial-literacy:low`,
drawn from the SAME reachable vocabulary as `persona_tags` above) that
SOFT-WEIGHT selection toward
ecologically-valid personas (a soft prior, never a hard filter). At runtime the probe picks the institution, scopes
categories to `institution.type`, and the loop generates turn-1 from the invitation +
a rotated subtopic hint. Loaded by the shared `core/probing_taxonomy.py` (the
same substrate as `sov_ai_dynamic`, with per-category `applies_to_types`
scoping; env override `USERSIM_FINANCIAL_SERVICES_DYNAMIC_TAXONOMY_<LOCALE>`).

**Gold documents are auto-derived, not hand-typed.** For a verifiable task, omit
`gold_document_ids`; the loader derives them from `gold_tool_sequence`: the corpus
docs (`policy_procedure` + `discoverable_tool_doc`) whose `mentions_tools` include a
gold tool. This keeps tasks free of brittle concrete ids that drift when the corpus is
regenerated. An explicit `gold_document_ids` may be supplied to override (must resolve
within the institution's corpus). Document `id`s are deterministic UUIDs
(`uuid5(locale:institution:doc_key)`), stable across regenerations.
