# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Document-generation prompts for financial_services (two-phase).

Phase 1 (`BRIEF_PROMPT`): generate one `InstitutionBrief` per institution -- a
generic finserv brand+operating playbook grounded in the seeded product lineup +
regulators, with Jinja branching on institution_type / brand_voice.

Phase 2 (`DOCSET_PROMPT`): generate one bounded document CLUSTER per row (a
product cluster, or the institution's shared cross-cutting cluster), CONDITIONED
on the generated brief so every cluster echoes the same voice/positioning/policy
stance. `DOCSET_QC_PROMPT` scores each cluster before it is committed.

All prompts are Jinja templates rendered per seed row; the seed columns are
produced by ``pipeline.build_institution_seed`` (phase 1) and
``pipeline.explode_to_cluster_rows`` (phase 2).

Market-specific framing is DATA, not template text: per-institution-type emphasis
comes from ``concept_guidance_for`` (region_spec override over a region-neutral
default), so a non-US locale does not inherit US instruments or statutes.
"""

from __future__ import annotations

# Phase 1 seed columns: institution_id, display_name, institution_type,
# brand_voice, language, currency_symbol, products_json, regulators_json,
# concept_guidance.
BRIEF_PROMPT = """\
You are the brand and operations lead for a FICTIONAL {{ institution_type }} called
"{{ display_name }}"{% if display_name_local %} (written "{{ display_name_local }}" in {{ language }} — use that
form in {{ language }} prose, and keep it identical everywhere){% endif %} (id: {{ institution_id }}). Author a concise internal BRIEF that
every downstream document will follow, so the institution's voice and policies stay
consistent. Write ALL prose in {{ language }}; keep product ids and tool/function
names verbatim. Use the {{ currency_symbol }} symbol for money.

Products offered (authoritative typed values -- reference exactly, never invent):
{{ products_json }}

Applicable regulators / frameworks by domain (respect; do not restate verbatim):
{{ regulators_json }}

Write each field as 1-3 sentences, generic enough to guide many documents:
- positioning: market position, target segments, and what the institution competes on.
- brand_voice_guide: tone, person, reading level, jargon tolerance, formatting + disclaimer style.
- service_model: support channels, hours/timezone, escalation path.
- fee_philosophy: pricing stance and WHY fees exist / how they are waived.
- risk_and_compliance_posture: how risk, disclosures, fair treatment and data privacy are framed; reference the applicable framework + any deposit/investor-protection scheme.
- security_and_verification: identity/authentication stance BEFORE any state-changing action.
- dispute_and_fraud_handling: complaint/dispute/fraud philosophy and general timeline posture.
- advice_boundaries: what the institution advises vs defers (e.g. education, not individualized tax/legal advice); the standard of care it holds itself to when recommending products.
- product_positionings: for EACH product id above, a one-line positioning relative to the others.
- key_terms_glossary: a few canonical terms + definitions reused across documents.
{{ concept_guidance }}
Return the brief in the required structured format.
"""

#: Per-institution-type emphasis for the brief, deliberately REGION-NEUTRAL: it
#: names what the institution type is about, not the instruments or statutes of
#: any one market. A region_spec supplies market-specific framing via
#: ``concept_guidance`` (e.g. the US names RMDs and best-interest/fiduciary duty;
#: India would name NPS/PPF lock-ins, AMFI market-risk wording and gold-loan LTV).
#: Baking US concepts in here leaked "RMDs" and "APY" into every locale's corpus.
DEFAULT_TYPE_CONCEPT_GUIDANCE: dict[str, str] = {
    "bank": (
        "Emphasize deposits, everyday banking, fees/overdraft posture, and unauthorized-transaction dispute rights."
    ),
    "neobank": (
        "Emphasize digital-first onboarding, everyday payments, fees, and unauthorized-transaction dispute rights."
    ),
    "credit_union": (
        "Emphasize member ownership, deposits, everyday banking, and unauthorized-transaction dispute rights."
    ),
    "brokerage": (
        "Emphasize suitability of recommendations, settlement timing, market "
        "risk, and that returns are never guaranteed."
    ),
    "wealth_manager": ("Emphasize advisory duty of care, advisory fees, and risk-profiling."),
    "robo_advisor": ("Emphasize automated advice limits, advisory fees, and risk-profiling."),
    "retirement_provider": (
        "Emphasize contribution limits, withdrawal restrictions and penalties, "
        "and flagging tax consequences without giving individualized tax advice."
    ),
    "insurer": ("Emphasize underwriting, claims handling, coverage terms, and exclusions."),
    "nbfc": (
        "Emphasize that this lender takes no deposits, loan eligibility and "
        "pricing transparency (interest, processing fees, prepayment/foreclosure "
        "charges), collateral valuation where secured, repayment mandates, and "
        "fair, non-coercive collections."
    ),
}


def concept_guidance_for(institution_type: str, overrides: dict[str, str] | None = None) -> str:
    """Brief emphasis for ``institution_type``: region override else neutral default.

    Returns ``""`` for an unknown type with no override, so the prompt simply
    omits the emphasis line rather than injecting another market's concepts.
    """
    if overrides:
        text = overrides.get(institution_type)
        if text and text.strip():
            return text.strip()
    return DEFAULT_TYPE_CONCEPT_GUIDANCE.get(institution_type, "")


# Phase 2 seed columns: institution_id, display_name, institution_type, brand_voice,
# language, currency_symbol, institution_brief, cluster_kind, focus_json, doc_plan_json
# (each entry carries doc_key/document_type/topic/angle/audience/length), sibling_doc_index,
# target_doc_count.
DOCSET_PROMPT = """\
You are authoring part of the internal knowledge base for the FICTIONAL institution
"{{ display_name }}"{% if display_name_local %} (in {{ language }}: "{{ display_name_local }}"){% endif %} (id: {{ institution_id }}), a {{ brand_voice }}-voice {{ institution_type }}.
Follow this institution BRIEF exactly (voice, positioning, fee/dispute/advice posture):
{{ institution_brief }}

Write ALL body prose in {{ language }}. Keep tool/function identifiers, product ids, and
doc_keys VERBATIM. Use the {{ currency_symbol }} symbol for money.
TITLES ARE GIVEN, NOT AUTHORED: copy each entry's `topic` into its `title` EXACTLY as
provided -- already in {{ language }} -- and do not translate, shorten, re-word, or
re-punctuate it. The same string is what every sibling document is told to cite when it
cross-references this one, so a title that drifts breaks those references.
NAMING CONSISTENCY: when a product carries a `display_name_local`, use exactly that name in
{{ language }} prose -- never re-translate it and never vary it between documents, because
readers (and retrieval) rely on one product having one name across the knowledge base.
NUMERIC FIDELITY IS THE #1 REQUIREMENT: reproduce every fee, rate, yield, balance, limit,
threshold, duration, date, and percentage from the focus values EXACTLY as given (same digits,
symbol, %, and units) -- never round, approximate, paraphrase, bucket, omit, or invent a number.
The VALUE is verbatim; its NAME is not. The focus keys are machine field names, so write what
each one means in ordinary prose and NEVER print the key itself: say "an annual interest rate
of 0%", not "an interest_rate_annual of 0%"; "a one-time processing fee of {{ currency_symbol }}499",
not "a processing_fee of {{ currency_symbol }}499". A document that shows a snake_case field name
reads like an unfilled template and is a failure even if the number is right.

{% if cluster_kind == 'shared' %}
This is the institution's SHARED, cross-cutting document set (applies across all products).
Focus context (tools + product lineup):
{% else %}
This is the document set for ONE product. Focus product and its EXACT typed values:
{% endif %}
{{ focus_json }}

Produce EXACTLY these {{ target_doc_count }} documents (one per entry); use the given
`doc_key`, `document_type` and `topic` (verbatim, as the `title`), cover that topic, write
each from its `angle` (its unique
lens/sub-topic) FOR its `audience` (the intended reader), at roughly its `length`. The
angle + audience are what make same-genre siblings different -- honor them so no two
documents overlap in scope:
{{ doc_plan_json }}

Sibling documents you may cross-reference by their exact title (doc_key -> title):
{{ sibling_doc_index }}

Requirements:
- Faithfulness (HIGHEST PRIORITY -- overrides brevity): every numeric value present in the
  focus values MUST appear VERBATIM in the document(s) where it belongs. If stating all exact
  values conflicts with the length target, keep the exact values and cut prose instead -- a
  document that rounds, buckets, or omits a value is a failure even if it reads well.
- Length: aim for each body's per-entry `length` target -- a SOFT target that yields to
  faithfulness. Narrative genres (product_sheet, disclosure, policy_procedure, regulatory_note,
  how_to_guide, troubleshooting, terms_conditions, security_advisory) have a larger budget:
  use it to be genuinely thorough and specific, not padded. Structured genres (fee_schedule,
  eligibility_matrix, promo_notice, faq, rate_sheet, comparison) stay tight. Value-bearing docs
  MAY exceed their target if needed to state every applicable value exactly.
- Consistency: names, numbers, and fees must match the focus values and the brief exactly.
- Distinctness: every document must be materially different in scope, framing, and detail --
  driven by its `angle` and `audience`. Two documents of the same document_type MUST NOT
  restate the same content; if two angles feel close, deepen one and keep the other high-level.
- Self-containment first: each document must fully answer its own topic on its own. Only
  AFTER it stands alone, optionally add ONE cross-reference to a sibling by its exact title
  (list its doc_key in `references`); never replace substance with "see the other doc".
  In the BODY, name the sibling only by that exact title -- a `doc_key` is an internal
  slug, so prose like "see 1-Year Fixed Deposit - faq_5" is meaningless to a reader and
  defers the very substance this document owes them. Keys belong in `references` only.
- Genre conventions (only the genres THIS cluster emits are listed):
{% if cluster_kind == 'shared' %}
    policy_procedure: numbered agent steps referencing the exact tool name(s). The
      documented tool is the agent's AUTHORITATIVE mechanism to complete the action
      IN THE CONVERSATION after identity is verified -- the final step is calling the
      tool, NOT redirecting the customer to an external portal, app, branch, website,
      or form. (Use get_accounts to resolve the customer's account_id from a last-4.)
    discoverable_tool_doc: the tool's exact signature, args, and effect. State plainly
      that the agent performs this action by calling the tool directly (never by
      telling the customer to self-serve it elsewhere).
      EVERY argument must be something the agent can actually obtain in the
      conversation: a value the customer states, or one another tool returns. Never
      invent an internal identifier with no source -- no `customer_id` for the person
      the agent is already talking to and has verified, and no `user_id` for a spouse
      or nominee the customer is naming for the first time. Those tools take the
      PERSON'S DETAILS (full name, date of birth, contact) exactly as the customer
      gives them. A signature demanding an id nobody can supply reads authoritative
      and stops the action dead: the agent believes it, asks for the id, the customer
      does not have one, and a completable task fails on a documentation artifact.
    regulatory_note: the applicable rule and the agent's obligation.
    fee_schedule: the INSTITUTION-WIDE charges that are not tied to one product; say
      which apply to every account and which vary by product, without restating any
      product's own numbers.
    faq: 2-4 Q&A on questions that span products (getting started, contacting
      support, escalating). Point to the relevant policy by title.
{% else %}
    Every genre below is CUSTOMER-facing, so none of them may name an internal tool or
      function (``get_accounts``, ``file_transaction_dispute``, ...). A customer-facing
      document says what the customer can ask for and what they get back; which tool
      serves it is an internal detail, documented separately in the agent-facing
      policy_procedure / discoverable_tool_doc set. Naming one here reads like an
      internal runbook leaked to the public.
    product_sheet: highlights + EVERY applicable fee/rate/limit, each stated with its exact value.
    fee_schedule: list EVERY fee item -> its exact amount (no ranges, no "varies").
    rate_sheet: the rate/yield figures you were given, when each applies, and what
      changes them. Do NOT invent a tier structure: if the product has a single flat
      rate, say plainly that one rate applies to all balances and spend the document on
      when it is credited, what it is quoted against, and what would change it.
    faq: 2-4 Q&A; point to the relevant policy/product by title; quote exact values where relevant.
    how_to_guide: numbered CUSTOMER-facing steps for one task end to end -- what they
      need before starting, each step, and how they know it worked. Write every step
      from the CUSTOMER's side: what they say or provide, and what they get back. A
      step the institution can do for them is phrased as "ask us to ..." / "we will
      ..." and completed in the conversation -- do NOT send them to a portal, app,
      branch, website, phone line, or form for it, and do NOT narrate the mechanism
      ("the agent then calls X"). A step that is genuinely physical in this market
      (presenting an original document, having collateral valued) may of course be
      described as such.
      Reaching us is NOT a step: the reader is already talking to us. Never open with
      "contact us via phone / secure chat / email" and never list our channels -- step 1
      is the first thing that actually moves the task forward (what they tell us, what
      they hand over, what we confirm back). A contact step costs the document its whole
      first move and makes every guide read from the same template.
    troubleshooting: the symptom as the customer would describe it, the likely causes,
      what to check in what order, the fix, and when to escalate -- with any timeline
      or reference the customer should expect. Written from the CUSTOMER's side: it is
      resolved in the conversation, so "escalate" means what WE do next (raise a case,
      apply a provisional credit, route to a specialist team) and what the customer
      receives, never sending them to a portal, app, branch, website, phone line, or
      form to fix something we can fix, and never narrating the mechanism ("the agent
      then calls X"). A step that is genuinely physical in this market may of course be
      described as such.
    comparison: compare against the named alternative on the few dimensions that
      actually decide it (cost, access, risk, lock-in), with this product's exact
      figures on its side. Only state a figure for the alternative when it is a SIBLING
      product whose values you were given; for an outside offering compare on structure
      and trade-offs rather than inventing its numbers. Then say who each suits. Never
      disparage the alternative; state where it is the better choice.
    eligibility_matrix: conditions -> outcome rows, with exact thresholds/limits.
    terms_conditions: the binding terms in formal register -- obligations on each side,
      what triggers a change, notice periods, and how the agreement ends. Anchor every
      clause in THIS product's actual fees, limits and thresholds rather than generic
      boilerplate: a term the reader cannot tie to a stated figure or a named
      obligation does not belong in the document.
    disclosure: formal regulatory / T&C language; exact figures where cited.
    promo_notice: a time-bound offer with explicit start/end dates + exact rate/amount.
    security_advisory: the specific fraud patterns that target THIS product, what the
      institution will never ask for, and the first action to take when something looks
      wrong; include any reporting deadline that affects liability.
{% endif %}
- Do not fabricate tools; only reference tools present in the focus/brief.
Return the documents in the required structured format.
"""

# Phase 2 QC seed columns: focus_json + the generated `doc_set`.
DOCSET_QC_PROMPT = """\
You are auditing a generated internal knowledge-base document set before it is committed.
Score it against the rubric.

Authoritative product values -- the ONLY numbers to check for NumericFaithfulness. Any
other numbers in the documents (dates, illustrative examples, regulatory citations) are
EXPECTED and must NOT lower the faithfulness score. If this context lists no product
values (e.g. an institution-wide shared set), there is nothing to contradict -> score 4.
{{ focus_json }}

Generated document set:
{{ doc_set }}
"""

# ── asset EVALUATION (DD-judge scorecard over the committed corpus) ──────
# Seed columns: institution_brief, authoritative_values, regulator_text,
# document_type, language, title, body.
EVAL_PROMPT = """\
You are a senior reviewer auditing ONE document from a fictional financial
institution's internal knowledge base. Judge its QUALITY against the rubric; be
strict and specific. The document should read like a real {{ document_type }} for
this institution.

Institution brief (the intended voice / positioning / policy the doc should follow):
{{ institution_brief }}

Authoritative product values the document must not contradict:
{{ authoritative_values }}

Applicable regulator / framework context (the doc must be consistent with this):
{{ regulator_text }}

Document (type: {{ document_type }}; expected language: {{ language }}):
Title: {{ title }}
{{ body }}
"""
