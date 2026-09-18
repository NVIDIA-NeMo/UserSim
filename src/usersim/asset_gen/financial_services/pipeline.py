# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services document-generation pipeline (Data Designer, two-phase).

Turns a validated ``RegionSpec`` (see ``spec.py``) into a per-institution asset
bank. The file reads top-to-bottom as the pipeline:

    schemas -> doc-plan partition -> Phase 1 (brief) -> bridge (explode)
    -> Phase 2 (clusters) -> collect -> embed -> serialize.

Cohesion is two-phase (DD runs each seed row independently, so cross-row cohesion
needs a GENERATED shared artifact):
  - Phase 1 generates one ``InstitutionBrief`` per institution (positioning, voice,
    fee/risk/security/dispute/advice posture, per-product positioning), grounded in
    the seeded product lineup + regulators.
  - A pandas ``explode_to_cluster_rows`` fans each institution into per-product rows
    (batched by ``max_docs_per_cluster``) + one shared-docs row, carrying the brief.
  - Phase 2 generates each bounded document cluster CONDITIONED on the brief, so
    clusters are consistent within a product and across products.

The doc-gen / judge / embedding model aliases are declared in
``cli/models_default.toml``. ``generate_corpus`` is a real implementation (needs
network + credentials to RUN, so it is not exercised in offline CI); the pure
pieces (doc-plan builders, ``build_institution_seed``, ``explode_to_cluster_rows``,
column builders, ``explode_docset``, ``serialize_bank``) + the endpoint-validation
path are tested offline.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from usersim.asset_gen.financial_services.prompts import (
    BRIEF_PROMPT,
    DOCSET_PROMPT,
    DOCSET_QC_PROMPT,
    concept_guidance_for,
)
from usersim.asset_gen.financial_services.spec import (
    DocTitleVocabulary,
    Institution,
    Product,
    RegionSpec,
    Tool,
)

logger = logging.getLogger("usersim.asset_gen.financial_services.pipeline")

# Model aliases (resolved from cli/models_default.toml at generation time).
DOC_GEN_MODEL_ALIAS = "doc_gen_model"
DOCSET_JUDGE_MODEL_ALIAS = "asset_judge_model"
EMBEDDING_MODEL_ALIAS = "embedding_model"

#: Minimum DocSet QC faithfulness score (0-4); below this a cluster is kept but
#: flagged + logged (it is already ``placeholder: true``), never silently dropped.
MIN_FAITHFULNESS_SCORE = 3

#: Default document-count knobs (overridable via CLI / generate_corpus params).
DEFAULT_DOCS_PER_PRODUCT = 40  # ~220 docs/institution, so retrieval is a real
# bottleneck (see probe README / SCHEMA). 40 is
# chosen, not arbitrary: it is where every
# expandable genre gets 5 slots, so how_to_guide
# and troubleshooting receive all their intents
# (open / close / failed action) WHILE faq keeps
# a per-variable explainer for every value. It is
# also the last multiple of 10 below the smallest
# product's distinct-angle supply (46), so no
# cluster is padded. ~10 for fast test regens.
DEFAULT_MAX_DOCS_PER_CLUSTER = 10  # per-call cap; larger targets fan into batches.
# 10, not 12: each doc_set call returns that many
# documents as ONE structured object, and larger
# clusters made the hub 5xx / time out, dropping
# whole clusters. Coherence does not need big
# clusters -- the shared brief, the deterministic
# angle plan and the whole-institution sibling
# index carry it across batches.

#: Target per-document body size, in TOKENS, by genre TIER. Backed out to a
#: word/character target per script (see ``doc_length_hint``) so docs stay ~this size
#: in the TARGET language. Two tiers: reference/structured genres stay tight (readable,
#: fast, and unbounded bodies are the main driver of slow/timing-out ``doc_set`` calls);
#: narrative genres get more room to read like real, fleshed-out documents (realism).
TARGET_DOC_TOKENS_TIGHT = 250  # ~150-210 words (Latin): fee tables, matrices, FAQs, tool docs
TARGET_DOC_TOKENS_RICH = 520  # ~300-450 words (Latin): sheets, disclosures, policies, reg notes

#: Genres that warrant longer, narrative bodies. Everything else defaults to tight.
_RICH_GENRES = frozenset(
    {
        "product_sheet",
        "disclosure",
        "policy_procedure",
        "regulatory_note",
        "how_to_guide",
        "troubleshooting",
        "terms_conditions",
        "security_advisory",
    }
)

_SCHEMA_VERSION = "v0.1"


# ── Structured output schemas ───────────────────────────────────────────


class GeneratedDoc(BaseModel):
    """One generated document within a cluster."""

    doc_key: str = Field(description="Stable slug the model assigns, for cross-refs.")
    title: str
    document_type: str
    body: str
    mentions_tools: list[str] = Field(default_factory=list)
    #: Deliberately NOT carried into ``corpus.yaml`` (see ``explode_docset``): this field
    #: exists to give the model somewhere to put a sibling's slug, so the slug stays OUT
    #: of the prose. The prompt asks for a cross-reference by TITLE in the body and the
    #: key here; before that split, 0.7% of documents cited a sibling as "... - faq_5",
    #: which means nothing to a reader. Keeping the sink is what makes that 0%.
    references: list[str] = Field(
        default_factory=list,
        description="doc_keys of sibling documents this one links to.",
    )


class DocSet(BaseModel):
    """A cohesive, cross-referencing set of documents for one cluster."""

    documents: list[GeneratedDoc]


class ProductPositioning(BaseModel):
    product_id: str
    one_liner: str = Field(description="One-line positioning relative to siblings.")


class GlossaryTerm(BaseModel):
    term: str
    definition: str


class InstitutionBrief(BaseModel):
    """Phase-1 generated brand+operating playbook that all clusters condition on."""

    positioning: str
    brand_voice_guide: str
    service_model: str
    fee_philosophy: str
    risk_and_compliance_posture: str
    security_and_verification: str
    dispute_and_fraud_handling: str
    advice_boundaries: str
    product_positionings: list[ProductPositioning] = Field(default_factory=list)
    key_terms_glossary: list[GlossaryTerm] = Field(default_factory=list)


# ── Doc-plan partition (deterministic; product-specific vs cross-cutting) ─


def _slot(
    doc_key: str,
    document_type: str,
    topic: str,
    domain: str,
    product_category: str,
    mentions_tools: list[str] | None = None,
    angle: str = "",
    audience: str = "",
) -> dict[str, Any]:
    return {
        "doc_key": doc_key,
        "document_type": document_type,
        "topic": topic,
        # Diversity axes -> distinctness. ``angle`` is the doc's unique lens/sub-topic
        # (so repeated genres don't overlap); ``audience`` is who it's written for.
        "angle": angle,
        "audience": audience,
        "domain": domain,
        "product_category": product_category,
        "mentions_tools": list(mentions_tools or []),
    }


#: Reader lenses cycled across a product's doc set so docs address different readers.
_PRODUCT_AUDIENCES = [
    "a first-time / prospective customer",
    "an existing customer already using the product",
    "a customer comparing this against alternatives",
    "a support agent assisting a customer",
]

#: DEFAULT (English) per-genre distinct angles; a locale overrides any subset via
#: ``RegionSpec.doc_titles.product_genre_angles``. A genre stops emitting once its
#: angles are exhausted, so the FAQ tail -- not duplicate product sheets -- absorbs
#: large targets. Keep these SHORT and title-appropriate: they become the doc title
#: (``{title}{sep}{angle}``) and are quoted verbatim in sibling cross-references, so
#: instruction-y phrasing here leaks into the corpus. Because they are reader-facing
#: prose they must be TRANSLATED for a non-English locale, not left to the generator:
#: see ``DocTitleVocabulary``. Genre conventions ("list every fee") live in the prompt.
_PRODUCT_GENRE_ANGLES: dict[str, list[str]] = {
    "product_sheet": [
        "overview",
        "who it is best for and common use cases",
        "getting started walkthrough",
        "limits, restrictions, and edge cases",
        "comparison vs. alternatives and when not to choose it",
    ],
    "fee_schedule": ["fee schedule"],
    # ``required documentation`` is 2nd, not 3rd: it is what answers "what evidence
    # do I need to open this?", and a genre only gets 2-3 slots at shipped depth. At
    # 3rd it landed on slot 28, so en_IN at docs_per_product=25 never produced it.
    "eligibility_matrix": ["eligibility requirements", "required documentation", "qualification tiers"],
    "disclosure": ["regulatory disclosures", "risk disclosures", "privacy and data handling"],
    "promo_notice": ["current offer", "referral and loyalty offer"],
    "rate_sheet": ["interest rate / APY schedule", "how rates are tiered and applied"],
    "terms_conditions": ["account agreement", "change-of-terms policy"],
    "security_advisory": [
        "fraud and unauthorized-activity protection",
        "account security best practices",
        "recognizing scams targeting this product",
    ],
}

#: FAQ intents cycled after per-variable FAQs, so extra FAQs stay distinct.
_FAQ_INTENTS = [
    "cost and charges",
    "eligibility and setup",
    "how-to / step-by-step",
    "troubleshooting a problem",
    "comparison and alternatives",
]

#: Intents for the OTHER expandable genres (besides FAQ) that share the tail so a
#: large ``docs_per_product`` yields a balanced mix instead of all-FAQ. Each is a
#: distinct customer-facing lens; cycled after any per-variable angles.
#: Lifecycle open/close lead, because those are the two the customer actually asks
#: about ("how do I open this?", "how do I close it and what settles first?") and a
#: genre only gets a few slots.
_HOW_TO_INTENTS = [
    "set up and activate",
    "cancel, close, or downgrade",
    "make a change or update",
    "check status or history",
    "resolve a common request end-to-end",
]
_TROUBLESHOOTING_INTENTS = [
    "a failed or pending action",
    "an unexpected charge or amount",
    "access or login problems",
    "a declined or blocked action",
    "a discrepancy in your records or statement",
]
_COMPARISON_INTENTS = [
    "vs. a similar product here",
    "vs. a typical competitor offering",
    "choosing between tiers or options",
    "when to switch or upgrade",
]


#: Default topic separator between a product name and its angle.
_TOPIC_SEPARATOR = " — "

#: Default (English) shared-cluster topics; overridable per locale via
#: ``DocTitleVocabulary``. ``{tool}`` stays verbatim in every language -- it is the
#: identifier the agent calls, not prose.
_FEE_OVERVIEW_TOPIC = "Institution-wide fees and charges"
_GENERAL_FAQ_TOPIC = "General frequently asked questions"
_REGULATORY_NOTE_TOPIC = "Regulatory notes for {domain}"
_TOOL_POLICY_TOPIC = "When to use {tool}"
_TOOL_REFERENCE_TOPIC = "{tool} tool reference"


def _vocab(spec_or_vocab: Any) -> DocTitleVocabulary:
    """Coerce a ``RegionSpec`` / ``DocTitleVocabulary`` / None into a vocabulary."""
    if spec_or_vocab is None:
        return DocTitleVocabulary()
    return getattr(spec_or_vocab, "doc_titles", spec_or_vocab)


def build_product_doc_plan(
    product: Product,
    target_docs: int,
    vocab: Any = None,
) -> list[dict[str, Any]]:
    """PRODUCT-specific doc slots, BALANCED across genres (not FAQ-dominated).

    Round-robins one slot at a time across the product-appropriate genres (an
    overview leads), so angle *n* of the genre at list position *p* lands at slot
    ``p + 12(n-1)``. Cross-cutting docs (policy_procedure / discoverable_tool_doc /
    regulatory_note / institution fee overview) are NOT here -- they live in the
    shared cluster.

    Two ordering properties are load-bearing, because a genre only receives 2-5 slots
    at realistic ``target_docs`` and whichever angles come first take all of them:

    - ``how_to_guide`` / ``troubleshooting`` yield their INTENTS first
      (``_cycle_before_variables``), so every product gets the customer-journey
      documents the dynamic tier asks about -- opening, closing, a failed action --
      rather than only products with few enough variables to exhaust.
    - ``faq`` keeps per-variable angles first (``_cycle_after_variables``), so each
      authoritative value still gets an explainer document.

    Every ``(genre, angle)`` pair is emitted at most once. Past a product's distinct-
    angle supply the plan stops short of ``target_docs`` and logs, rather than padding
    with repeats that differ only by audience and an index suffix.
    """
    pid = product.id
    domain = str(product.domain)
    voc = _vocab(vocab)
    # The IN-LANGUAGE product name when the locale authored one. The overlay
    # authors ``display_name_local`` precisely so a product is named identically
    # across the corpus, but the plan used to read only the Latin
    # ``display_name`` -- so every title it minted was English no matter what the
    # locale's language was, and the generator was left to translate a string it
    # had also been told to reproduce exactly.
    title = product.display_name_local or product.display_name or pid
    sep = voc.topic_separator or _TOPIC_SEPARATOR
    # A variable name is a machine field; its label is reader-facing and belongs
    # to the language. Unmapped names keep the historical underscores-to-spaces
    # rendering (which is why en_IN ships "below mab charge explained").
    var_names = [voc.variable_labels.get(v) or v.replace("_", " ") for v in product.variables]
    genre_angles = {**_PRODUCT_GENRE_ANGLES, **voc.product_genre_angles}
    faq_intents = voc.faq_intents or _FAQ_INTENTS
    how_to_intents = voc.how_to_intents or _HOW_TO_INTENTS
    troubleshooting_intents = voc.troubleshooting_intents or _TROUBLESHOOTING_INTENTS
    comparison_intents = voc.comparison_intents or _COMPARISON_INTENTS
    faq_var_fmt = voc.faq_variable_template or "{v} explained"
    how_to_var_fmt = voc.how_to_variable_template or "how to manage {v}"
    trouble_var_fmt = voc.troubleshooting_variable_template or "issues with {v}"

    def _cycle_after_variables(per_variable_fmt: str, intents: list[str]):
        # Per-variable angles FIRST. Used by ``faq``, whose "{v} explained" angles are
        # the value explainers that carry each authoritative number into prose -- the
        # backbone of the numeric-faithfulness check.
        for v in var_names:
            yield per_variable_fmt.format(v=v)
        yield from intents

    def _cycle_before_variables(per_variable_fmt: str, intents: list[str]):
        # Intents FIRST. Used by ``how_to_guide`` / ``troubleshooting``, whose intents
        # are the customer-journey documents (open, close, a failed action) that the
        # dynamic tier asks about.
        #
        # Order matters far more than it looks: a genre gets only 2-3 slots at shipped
        # depth, so whichever kind of angle comes first takes all of them. With
        # per-variable first, a product's intents were reached only if it had few
        # enough variables to exhaust -- in en_US exactly the 7 two-variable products
        # got them and the other 11 did not, at any depth.
        yield from intents
        for v in var_names:
            yield per_variable_fmt.format(v=v)

    def _cycle(intents: list[str]):
        yield from intents

    def finite(genre: str):
        yield from genre_angles[genre]

    # Round 1 yields one of each genre (product_sheet leads). Finite genres stop
    # once their angles are exhausted; the FOUR expandable genres (faq /
    # how_to_guide / troubleshooting / comparison) SHARE the tail, so a large
    # ``target_docs`` produces a balanced, realistic mix instead of all-FAQ.
    genres: list[tuple[str, Any]] = [
        ("product_sheet", finite("product_sheet")),
        ("faq", _cycle_after_variables(faq_var_fmt, faq_intents)),
        ("how_to_guide", _cycle_before_variables(how_to_var_fmt, how_to_intents)),
        ("fee_schedule", finite("fee_schedule")),
        ("troubleshooting", _cycle_before_variables(trouble_var_fmt, troubleshooting_intents)),
        ("eligibility_matrix", finite("eligibility_matrix")),
        ("comparison", _cycle(comparison_intents)),
        ("disclosure", finite("disclosure")),
        ("rate_sheet", finite("rate_sheet")),
        ("promo_notice", finite("promo_notice")),
        ("security_advisory", finite("security_advisory")),
        ("terms_conditions", finite("terms_conditions")),
    ]
    counts: dict[str, int] = {g: 0 for g, _ in genres}
    slots: list[dict[str, Any]] = []
    seen: set = set()
    target = max(target_docs, 1)
    while len(slots) < target:
        progressed = False
        for genre, gen in genres:
            if len(slots) >= target:
                break
            angle = next(gen, None)
            if angle is None or (genre, angle) in seen:
                # Exhausted, or a repeat. Either way this genre makes no progress
                # this round and its slot passes to one that still has a new angle.
                continue
            seen.add((genre, angle))
            counts[genre] += 1
            n = counts[genre]
            key = f"{pid}_{genre}" if n == 1 else f"{pid}_{genre}_{n}"
            audience = _PRODUCT_AUDIENCES[len(slots) % len(_PRODUCT_AUDIENCES)]
            topic = f"{title}{sep}{angle}"
            slots.append(_slot(key, genre, topic, domain, pid, angle=angle, audience=audience))
            progressed = True
        if not progressed:
            # Every genre is out of NEW angles: the product's distinct-angle supply
            # is spent. Stop rather than pad with repeats differentiated only by
            # audience and an index suffix, which is where near-duplicate documents
            # came from. Supply is 40 + 3 * len(variables), i.e. 46-58 for the
            # shipped products.
            logger.info(
                "%s: angle supply exhausted at %d/%d requested docs "
                "(distinct angles only; raise variables or angle lists to go higher)",
                pid,
                len(slots),
                target,
            )
            break
    return slots[:target]


def build_shared_doc_plan(
    inst: Institution,
    vocab: Any = None,
) -> list[dict[str, Any]]:
    """CROSS-CUTTING doc slots owned by the institution's shared cluster.

    Generated once (never per product) so they cannot drift: institution fee
    overview, general FAQ, a regulatory note per domain, and a policy_procedure +
    discoverable_tool_doc for every discoverable tool.

    Relationship-level servicing questions (accepted ID evidence, opening/closing,
    beneficiaries, transfer limits and delays, escalation) are deliberately NOT here:
    they are product-specific in this domain -- what a brokerage needs to open a
    demat account differs from a savings account, and transfer limits are product
    variables -- so they are answered by the product plan's `required documentation`
    / `set up and activate` / `cancel, close, or downgrade` / `a failed or pending
    action` angles, which ``build_product_doc_plan`` guarantees.
    """
    primary = inst.domains[0]
    voc = _vocab(vocab)
    fee_topic = voc.fee_overview_topic or _FEE_OVERVIEW_TOPIC
    faq_topic = voc.general_faq_topic or _GENERAL_FAQ_TOPIC
    reg_topic = voc.regulatory_note_topic or _REGULATORY_NOTE_TOPIC
    policy_topic = voc.tool_policy_topic or _TOOL_POLICY_TOPIC
    tooldoc_topic = voc.tool_reference_topic or _TOOL_REFERENCE_TOPIC
    slots = [
        _slot("fee_overview", "fee_schedule", fee_topic, primary, "general"),
        _slot("faq_general", "faq", faq_topic, primary, "general"),
    ]
    for d in inst.domains:
        # An unmapped domain keeps the raw machine name, which is how en_IN came
        # to ship "Regulatory notes for retail_banking".
        label = voc.domain_labels.get(str(d)) or str(d)
        slots.append(_slot(f"regulatory_note_{d}", "regulatory_note", reg_topic.format(domain=label), d, "compliance"))
    for tool in inst.tool_taxonomy:
        if not tool.discoverable:
            continue
        slots.append(
            _slot(
                f"{tool.name}_policy",
                "policy_procedure",
                policy_topic.format(tool=tool.name),
                primary,
                "procedure",
                [tool.name],
            )
        )
        slots.append(
            _slot(
                f"{tool.name}_tooldoc",
                "discoverable_tool_doc",
                tooldoc_topic.format(tool=tool.name),
                primary,
                "procedure",
                [tool.name],
            )
        )
    return slots


def doc_length_hint(locale: str, language: str = "", genre: str = "") -> str:
    """Script- AND genre-aware per-document length instruction, sized so each body
    lands near its tier's token budget IN THE TARGET LANGUAGE.

    Token counts are not comparable across scripts (non-Latin scripts tokenize much
    heavier), so a fixed word count would make CJK/Indic docs balloon in tokens and
    time out. We instead back the token budget out to a script-appropriate WORD (or
    CHARACTER, for CJK where word boundaries are ill-defined) range, keeping doc SIZE
    -- and thus latency / cost / retrieval-chunk parity -- roughly constant.

    ``genre`` selects the tier: narrative genres (product_sheet, disclosure,
    policy_procedure, regulatory_note) get the RICH budget so they read like real,
    fleshed-out documents; everything else stays TIGHT. Detection prefers the
    structured ``locale`` (e.g. ``ja_JP``, ``hi_Deva_IN``); ranges are ~+/-15% slack.
    """
    tag = f"{locale} {language}".lower()
    budget = TARGET_DOC_TOKENS_RICH if genre in _RICH_GENRES else TARGET_DOC_TOKENS_TIGHT

    def _band(lo_ratio: float, hi_ratio: float) -> tuple:
        r = lambda x: max(10, int(round(budget * x / 10.0)) * 10)
        return r(lo_ratio), r(hi_ratio)

    # Romanized/transliterated locales (``hi_Latn_IN``, ``ta_Latn_IN``, ...) are
    # written in the LATIN alphabet, so they tokenize like Latin text and must be
    # checked FIRST: their tag carries the source language too ("hi_latn_in hindi
    # latin"), which would otherwise match the Indic branch below on "hindi" and
    # hand a romanized doc the much tighter Indic word budget.
    if "latn" in tag or "latin" in tag:
        lo, hi = _band(0.6, 0.85)
        return f"roughly {lo}-{hi} words"
    # CJK: instruct in CHARACTERS (~1 token/char; "words" undefined).
    if any(k in tag for k in ("ja_", "japanese", "ko_", "korean", "zh_", "chinese", "hant", "hans")):
        lo, hi = _band(0.6, 0.9)
        return f"roughly {lo}-{hi} characters"
    # Indic scripts (Devanagari etc.): heavy subword tokenization -> ~0.4 words/token.
    if any(
        k in tag
        for k in (
            "deva",
            "hindi",
            "bengali",
            "beng",
            "tamil",
            "taml",
            "telugu",
            "telu",
            "marathi",
            "gujarati",
            "gujr",
            "kannada",
        )
    ):
        lo, hi = _band(0.3, 0.45)
        return f"roughly {lo}-{hi} words"
    # Latin scripts (English, Portuguese, French, ...): ~0.75 words/token.
    lo, hi = _band(0.6, 0.85)
    return f"roughly {lo}-{hi} words"


def _batches(items: list[Any], cap: int) -> Iterator[list[Any]]:
    cap = max(cap, 1)
    for i in range(0, len(items), cap):
        yield items[i : i + cap]


# ── Phase 1: institution seed + brief config ────────────────────────────


def build_institution_seed(spec: RegionSpec):
    """One seed row per institution for the Phase-1 brief (pure)."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for inst in spec.institutions:
        regulators = {d: spec.domain_regulators[d].model_dump() for d in inst.domains if d in spec.domain_regulators}
        rows.append(
            {
                "institution_id": inst.id,
                "display_name": inst.display_name or inst.id,
                # Empty for locales that need no separate in-language rendering; the
                # prompts branch on it.
                "display_name_local": inst.display_name_local,
                "institution_type": inst.type,
                "brand_voice": inst.brand_voice,
                "language": spec.language,
                "currency_symbol": spec.currency_symbol,
                "products_json": json.dumps([p.model_dump() for p in inst.products], ensure_ascii=False),
                "regulators_json": json.dumps(regulators, ensure_ascii=False),
                # Market-specific emphasis: region override else region-neutral
                # default, resolved here so the prompt stays free of any one
                # market's instruments.
                "concept_guidance": concept_guidance_for(inst.type, spec.concept_guidance),
            }
        )
    return pd.DataFrame(rows)


def build_brief_columns() -> list[Any]:
    """Phase-1 DD columns (introspectable): the structured InstitutionBrief."""
    import data_designer.config as dd

    return [
        dd.LLMStructuredColumnConfig(
            name="institution_brief",
            model_alias=DOC_GEN_MODEL_ALIAS,
            prompt=BRIEF_PROMPT,
            output_format=InstitutionBrief,
        )
    ]


def build_brief_config(models: Any, inst_seed) -> Any:
    """Assemble the Phase-1 (engine, builder). One place, explicit add_column."""
    import data_designer.config as dd

    from usersim.cli._models import to_model_configs

    engine = _make_engine(models)
    builder = dd.DataDesignerConfigBuilder(model_configs=to_model_configs(models))
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=inst_seed))
    for col in build_brief_columns():
        builder.add_column(col)
    return engine, builder


# ── Bridge: explode institutions -> cluster rows carrying the brief ─────


def explode_to_cluster_rows(
    brief_df,
    spec: RegionSpec,
    *,
    docs_per_product: int = DEFAULT_DOCS_PER_PRODUCT,
    max_docs_per_cluster: int = DEFAULT_MAX_DOCS_PER_CLUSTER,
):
    """Pure pandas fan-out: institution rows -> per-product (batched) + shared rows.

    Each emitted row carries the generated ``institution_brief`` plus its own
    ``cluster_kind`` / ``focus_json`` / ``doc_plan_json`` / ``sibling_doc_index`` /
    ``target_doc_count`` so Phase 2 can generate the cluster conditioned on the brief.
    """
    import pandas as pd

    spec_by_id = {i.id: i for i in spec.institutions}
    rows: list[dict[str, Any]] = []

    for _, brow in brief_df.iterrows():
        iid = str(brow["institution_id"])
        inst = spec_by_id[iid]
        brief_json = _as_json(brow.get("institution_brief"))

        product_plans = {p.id: build_product_doc_plan(p, docs_per_product, spec) for p in inst.products}
        shared_plan = build_shared_doc_plan(inst, spec)

        # Per-doc, genre-aware length target (script-robust): narrative genres get a
        # longer band, structured genres stay tight. Threaded into each plan slot so
        # the single cluster call can size each doc independently.
        for pl in (*product_plans.values(), shared_plan):
            for s in pl:
                s["length"] = doc_length_hint(spec.locale, spec.language, s["document_type"])
                # Stable, deterministic doc id minted from the PLAN key (not the
                # model's echoed doc_key) -> survives regeneration.
                s["uuid"] = doc_uuid(spec.locale, iid, s["doc_key"])

        # Deterministic sibling index (doc_key -> intended title/topic) for by-title
        # cross-references across the whole institution.
        sibling_index: dict[str, str] = {}
        for pl in product_plans.values():
            for s in pl:
                sibling_index[s["doc_key"]] = s["topic"]
        for s in shared_plan:
            sibling_index[s["doc_key"]] = s["topic"]

        base = {
            "institution_id": iid,
            "display_name": inst.display_name or inst.id,
            "display_name_local": inst.display_name_local,
            "institution_type": inst.type,
            "brand_voice": inst.brand_voice,
            "language": spec.language,
            "currency_symbol": spec.currency_symbol,
            "institution_brief": brief_json,
            "sibling_doc_index": json.dumps(sibling_index, ensure_ascii=False),
        }

        for product in inst.products:
            focus = json.dumps(product.model_dump(), ensure_ascii=False)
            for batch in _batches(product_plans[product.id], max_docs_per_cluster):
                rows.append(
                    {
                        **base,
                        "cluster_kind": "product",
                        "focus_json": focus,
                        "doc_plan_json": json.dumps(batch, ensure_ascii=False),
                        "target_doc_count": len(batch),
                        "cluster_id": _cluster_id(iid, "product", batch),
                    }
                )

        shared_focus = json.dumps(
            {
                "tools": [t.model_dump() for t in inst.tool_taxonomy],
                "products": [
                    {"id": p.id, "display_name": p.display_name or p.id, "display_name_local": p.display_name_local}
                    for p in inst.products
                ],
            },
            ensure_ascii=False,
        )
        for batch in _batches(shared_plan, max_docs_per_cluster):
            rows.append(
                {
                    **base,
                    "cluster_kind": "shared",
                    "focus_json": shared_focus,
                    "doc_plan_json": json.dumps(batch, ensure_ascii=False),
                    "target_doc_count": len(batch),
                    "cluster_id": _cluster_id(iid, "shared", batch),
                }
            )

    return pd.DataFrame(rows)


# ── Phase 2: cluster generation config ──────────────────────────────────


def docset_qc_scores() -> list[Any]:
    """Generation-time LLM-judge rubrics for a cluster (``dd.Score``)."""
    import data_designer.config as dd

    return [
        dd.Score(
            name="NumericFaithfulness",
            description=(
                "Among the PROVIDED authoritative values, are the ones a document "
                "states reproduced exactly? Judge ONLY the provided values; other "
                "numbers (dates, illustrative examples, regulatory citations) are "
                "expected and must NOT lower the score. If there are no provided "
                "values to check, score 4."
            ),
            options={
                4: "Provided values that appear are stated exactly; none contradicted.",
                3: "A single minor rounding/format slip on a provided value.",
                2: "A provided value is rounded, paraphrased, or vague.",
                1: "A provided value is stated with the wrong digits.",
                0: "Multiple provided values are wrong or contradicted.",
            },
        ),
        dd.Score(
            name="Cohesion",
            description="Are names, numbers, and facts consistent across the set + brief?",
            options={
                4: "Fully consistent; no contradictions.",
                3: "Minor inconsistency that does not mislead.",
                2: "A noticeable contradiction between two documents.",
                1: "Several contradictions.",
                0: "Documents read as unrelated / mutually inconsistent.",
            },
        ),
        dd.Score(
            name="Distinctness",
            description="Is every document materially different (no near-duplicates)?",
            options={
                4: "All documents are distinct in content and purpose.",
                3: "Largely distinct; slight overlap between two.",
                2: "A pair is near-duplicate.",
                1: "Several near-duplicates.",
                0: "Many documents restate the same content.",
            },
        ),
    ]


def build_docset_columns() -> list[Any]:
    """Phase-2 DD columns (introspectable): DocSet -> judge QC -> score flatteners."""
    import data_designer.config as dd

    cols: list[Any] = [
        dd.LLMStructuredColumnConfig(
            name="doc_set",
            model_alias=DOC_GEN_MODEL_ALIAS,
            prompt=DOCSET_PROMPT,
            output_format=DocSet,
        ),
        dd.LLMJudgeColumnConfig(
            name="doc_set_qc",
            model_alias=DOCSET_JUDGE_MODEL_ALIAS,
            prompt=DOCSET_QC_PROMPT,
            scores=docset_qc_scores(),
        ),
    ]
    # Flatten each rubric score into its own column (DD-idiomatic; cf. product_info_qa).
    for name, rubric in (
        ("numeric_faithfulness_score", "NumericFaithfulness"),
        ("cohesion_score", "Cohesion"),
        ("distinctness_score", "Distinctness"),
    ):
        cols.append(dd.ExpressionColumnConfig(name=name, expr="{{ doc_set_qc.%s.score }}" % rubric))
    return cols


def build_docset_config(models: Any, cluster_seed) -> Any:
    """Assemble the Phase-2 (engine, builder). One place, explicit add_column."""
    import data_designer.config as dd

    from usersim.cli._models import to_model_configs

    engine = _make_engine(models)
    builder = dd.DataDesignerConfigBuilder(model_configs=to_model_configs(models))
    builder.with_seed_dataset(dd.DataFrameSeedSource(df=cluster_seed))
    for col in build_docset_columns():
        builder.add_column(col)
    return engine, builder


def build_embedding_columns() -> list[Any]:
    """Embedding pass columns (run on the exploded per-doc frame): one vector/body."""
    import data_designer.config as dd

    return [
        dd.EmbeddingColumnConfig(
            name="body_embedding",
            target_column="body",
            model_alias=EMBEDDING_MODEL_ALIAS,
        )
    ]


# ── Explode a generated DocSet into loader-compatible document dicts ────


def explode_docset(
    docset: "DocSet | dict[str, Any]",
    *,
    id_prefix: str,
    institution_id: str,
    spec_by_key: dict[str, dict[str, Any]],
    default_domain: str,
    placeholder: bool = True,
) -> list[dict[str, Any]]:
    """Explode a ``DocSet`` into loader-compatible document dicts.

    Each generated doc is joined (by ``doc_key``) with the requested plan slot
    (``spec_by_key`` -> domain / product_category) so the emitted docs carry the
    right scope. Docs beyond the plan fall back to ``default_domain``.
    """
    if isinstance(docset, DocSet):
        docs = docset.documents
    else:
        docs = [GeneratedDoc(**d) for d in (docset.get("documents") or [])]
    out: list[dict[str, Any]] = []
    seen: set = set()
    off_plan: list[str] = []
    for gd in docs:
        slot = spec_by_key.get(gd.doc_key, {})
        if not slot and spec_by_key:
            # The model was told to echo doc_key VERBATIM. When it doesn't (a
            # non-English run may try to localize the slug), the doc loses its
            # product_category and lands in "general" — which then reads as a
            # numeric-faithfulness failure ("typed values absent from prose") far
            # from the real cause. Surface it here instead.
            off_plan.append(gd.doc_key)
        # Prefer the PLAN's deterministic uuid (stable across regens); fall back to a
        # readable derived id for any off-plan doc the model invented.
        doc_id = slot.get("uuid") or f"{id_prefix}-{gd.doc_key.upper().replace(' ', '-')}"
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(
            {
                "id": doc_id,
                # The PLAN owns the title, NOT the model. The plan's topic is already
                # the institution-wide cross-reference handle every sibling is told to
                # cite verbatim, so letting the model re-author it makes that contract
                # a promise the corpus cannot keep. In English the model echoed the
                # topic exactly (1066/1066 en_IN docs), so this is a no-op there; in
                # Hindi it echoed only 40% and translated the rest, which is how a
                # 93%-Devanagari corpus ended up with 44% English customer-facing
                # titles. An off-plan doc_key has no topic, so it keeps the model's.
                "title": slot.get("topic") or gd.title,
                "body": gd.body,
                "domain": slot.get("domain", default_domain),
                "product_category": slot.get("product_category", "general"),
                "document_type": gd.document_type,
                "source_authority": slot.get("source_authority", "official"),
                "placeholder": placeholder,
                # The PLAN decides which tools a document documents, NOT the model.
                # ``mentions_tools`` drives the sim-time discoverable-tool gate (a tool is
                # callable only once a doc naming it has been retrieved), and it is a field
                # on ``GeneratedDoc``, which the model could otherwise set on any
                # document. Left to the model it does: 7.8% of en_IN customer-facing docs
                # claimed tools they merely mentioned in passing, so retrieving a generic
                # how-to unlocked tools whose documentation was never read -- silently
                # weakening the one mechanic the gate exists to enforce. The plan knows
                # (``build_shared_doc_plan`` sets it on each tool's policy/tool doc, and
                # product slots carry none), so read it from there and let the model's
                # value be advisory. An off-plan doc_key therefore unlocks nothing, which
                # is the safe direction and is already reported above.
                "mentions_tools": list(slot.get("mentions_tools") or []),
                "institution_id": institution_id,
                # ``gd.references`` is intentionally absent: it is a prompt-side sink that
                # keeps sibling slugs out of the prose, not corpus data. See GeneratedDoc.
            }
        )
    if off_plan:
        logger.warning(
            "%s: %d/%d generated doc(s) echoed a doc_key that is not in the plan, "
            "so they carry no product scope (product_category='general') and will "
            "not satisfy that product's numeric-faithfulness check. Expected keys "
            "e.g. %s; got e.g. %s",
            institution_id,
            len(off_plan),
            len(docs),
            sorted(spec_by_key)[:3],
            off_plan[:3],
        )
    return out


# ── Serialize a generated bank into the committed YAML layout ──────────
#
#   assets/financial_services/<locale>/
#     region_meta.yaml
#     <institution_id>/{institution_meta,corpus,tools,tasks}.yaml (+ embeddings.parquet)


@dataclass
class InstitutionBank:
    """One institution's serializable content."""

    institution_meta: dict[str, Any]
    documents: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    templates: list[dict[str, Any]] = field(default_factory=list)
    #: doc_id -> dense embedding vector; written as an ``embeddings.parquet`` sidecar
    #: (not into corpus.yaml -- vectors don't belong in human-readable YAML).
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    @property
    def institution_id(self) -> str:
        return str(self.institution_meta["institution_id"])


#: Prepended to every generated tools.yaml so a reader isn't misled by the empty
#: description/parameters on discoverable tools (they LOOK like placeholders).
_TOOLS_YAML_HEADER = (
    "# Discoverable domain tools intentionally carry an EMPTY description /\n"
    "# parameters here. Their real schema (signature + args) lives in each tool's\n"
    "# discoverable_tool_doc in corpus.yaml -- the assistant learns a tool + its\n"
    "# arguments by RETRIEVING that doc via kb_search, then calls it through the\n"
    "# call_tool gate (the tau-Banking discoverable-tool mechanic). Putting schemas\n"
    "# here would defeat that discovery challenge. Only the always-offered\n"
    "# primitives (kb_search / verify_identity) carry a schema; side_effect_class\n"
    "# is always set (the scorer uses it).\n"
)


def _dump(path: Path, doc: dict[str, Any], *, header: str = "") -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        if header:
            fh.write(header)
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)


#: Fixed namespace for DETERMINISTIC document UUIDs. uuid5 over a stable logical key
#: (locale + institution + doc_key) -> globally unique, brand-independent, and IDENTICAL
#: across regenerations (unlike a random uuid4, which would re-dangle every reference).
_DOC_UUID_NAMESPACE = uuid.UUID("6f9d3c2e-1a4b-5c6d-8e7f-0a1b2c3d4e5f")


def doc_uuid(locale: str, institution_id: str, doc_key: str) -> str:
    """Stable document id: ``uuid5(namespace, "<locale>:<institution>:<doc_key>")``.

    Deterministic, so regenerating the same logical doc yields the same id -- the
    property that keeps cross-references (tasks, sibling links, embeddings) from
    dangling. Assigned from the PLAN's doc_key, not the model's echoed key.
    """
    key = f"{locale}:{institution_id}:{doc_key}"
    return str(uuid.uuid5(_DOC_UUID_NAMESPACE, key))


def _cluster_id(institution_id: str, cluster_kind: str, batch: list[dict[str, Any]]) -> str:
    """Stable id for one generation cluster, minted from its batch's deterministic
    doc uuids. Used to match a seed row to its generated output so dropped clusters
    (transient hub failures) can be detected and regenerated on their own -- and, being
    deterministic, it is identical across regenerations."""
    keys = "|".join(sorted(str(s.get("uuid") or s.get("doc_key")) for s in batch))
    return str(uuid.uuid5(_DOC_UUID_NAMESPACE, f"cluster:{institution_id}:{cluster_kind}:{keys}"))


def institution_rel_dir(institution_type: Any, institution_id: str) -> Path:
    """Bank-relative dir for an institution: ``<institution_type>/<institution_id>``.

    Grouping by TYPE (a stable taxonomy) rather than brand keeps the repo layout
    stable across brand renames and lets multiple brands share a vertical. Falls
    back to a flat ``<institution_id>`` if the type is missing.
    """
    itype = str(institution_type).strip() if institution_type else ""
    return Path(itype) / institution_id if itype else Path(institution_id)


def _dump_embeddings(path: Path, embeddings: dict[str, list[float]]) -> None:
    """Write a doc_id -> vector sidecar the dense retriever loads (not in YAML)."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"id": doc_id, "embedding": list(vec)} for doc_id, vec in embeddings.items()]
    pd.DataFrame(rows).to_parquet(path)


def serialize_bank(
    out_dir: str | Path,
    region_meta: dict[str, Any],
    institutions: list[InstitutionBank],
    *,
    write_tasks: bool = True,
) -> Path:
    """Write a full locale bank; returns the locale directory.

    ``write_tasks=False`` preserves any existing ``tasks.yaml`` (corpus
    regeneration should not clobber curated task templates).
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    region = dict(region_meta)
    region.setdefault("schema_version", _SCHEMA_VERSION)
    _dump(root / "region_meta.yaml", region)

    for inst in institutions:
        meta = dict(inst.institution_meta)
        meta.setdefault("schema_version", _SCHEMA_VERSION)
        idir = root / institution_rel_dir(meta.get("type"), inst.institution_id)
        _dump(idir / "institution_meta.yaml", meta)
        _dump(idir / "corpus.yaml", {"schema_version": _SCHEMA_VERSION, "documents": inst.documents})
        _dump(
            idir / "tools.yaml",
            {"schema_version": _SCHEMA_VERSION, "tools": _backfill_primitive_tools(inst.tools)},
            header=_TOOLS_YAML_HEADER,
        )
        if write_tasks:
            _dump(idir / "tasks.yaml", {"schema_version": _SCHEMA_VERSION, "templates": inst.templates})
        if inst.embeddings:
            _dump_embeddings(idir / "embeddings.parquet", inst.embeddings)
    return root


# ── Metadata + validation helpers ───────────────────────────────────────


def validate_generation_models(models: Any) -> None:
    """Fail fast if the doc-gen / judge / embedding aliases aren't configured."""
    from usersim.cli._models import require_aliases

    require_aliases(
        models,
        required=(DOC_GEN_MODEL_ALIAS, DOCSET_JUDGE_MODEL_ALIAS, EMBEDDING_MODEL_ALIAS),
        context="gen-assets --domain financial_services",
    )


def _make_engine(models: Any) -> Any:
    """A DataDesigner engine with the NATIVE Jinja engine (matches cli/_pipeline.py)."""
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    from usersim.cli._models import to_data_designer_kwargs

    engine = DataDesigner(**to_data_designer_kwargs(models))
    engine.set_run_config(dd.RunConfig(jinja_rendering_engine=dd.JinjaRenderingEngine.NATIVE))
    return engine


def _institution_meta(inst: Institution) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "institution_id": inst.id,
        "display_name": inst.display_name or inst.id,
        "type": inst.type,
        "brand_voice": inst.brand_voice,
        "domains": list(inst.domains),
        "placeholder": True,
    }
    # Carried into the asset so the RUNTIME can refer to the brand the way this
    # locale's own documents do. Omitted when unset, so a Latin-only locale's
    # committed bank is unchanged by this field existing.
    if inst.display_name_local:
        meta["display_name_local"] = inst.display_name_local
    return meta


def _tool_dict(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "discoverable": tool.discoverable,
        "side_effect_class": tool.side_effect_class,
        "parameters": tool.parameters,
    }


def _backfill_primitive_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill framework-primitive (kb_search / verify_identity) description +
    parameters from the runtime bank contract when the source spec left them
    empty. Region specs list these by name only, so without this the generated
    ``tools.yaml`` would ship a primitive with an empty schema — leaving the
    assistant to call ``kb_search({})`` (no query), which retrieves nothing and
    never unlocks the discoverable gold tools. Single source of truth:
    ``core.finance_bank.FRAMEWORK_PRIMITIVE_TOOLS``.
    """
    from usersim.engine.core.finance_bank import framework_primitive_tool

    out: list[dict[str, Any]] = []
    for t in tools:
        t = dict(t)
        canonical = framework_primitive_tool(t.get("name"))
        if canonical is not None:
            if not t.get("description"):
                t["description"] = canonical["description"]
            if not t.get("parameters"):
                t["parameters"] = canonical["parameters"]
        out.append(t)
    return out


def _region_meta_from_spec(spec: RegionSpec) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "locale": spec.locale,
        "region": spec.region,
        "currency": spec.currency,
        "currency_symbol": spec.currency_symbol,
        "bank_id": spec.bank_id or f"{spec.locale}_financial_services",
        "bank_version": spec.bank_version or "v0.1.0",
        "task_contract_version": spec.task_contract_version or "v0.1.0",
        "domain_regulators": {d: rc.model_dump() for d, rc in spec.domain_regulators.items()},
        # Always emitted (even at the default) so the sim-side gate reads the
        # region's KYC contract from the bank rather than hardcoding one.
        "identity_verification": spec.identity_verification.model_dump(),
        "placeholder": True,
    }
    if spec.account_labels:
        meta["account_labels"] = dict(spec.account_labels)
    if spec.number_format is not None:
        meta["number_format"] = spec.number_format
    if spec.privacy_regime is not None:
        meta["privacy_regime"] = spec.privacy_regime
    return meta


def _as_json(val: Any) -> str:
    """Coerce a (possibly structured) DD column value to a JSON string for seeding."""
    if val is None:
        return "{}"
    if isinstance(val, str):
        return val
    if hasattr(val, "model_dump"):
        return json.dumps(val.model_dump(), ensure_ascii=False)
    return json.dumps(val, ensure_ascii=False, default=str)


def _to_py(obj: Any) -> Any:
    """Recursively coerce numpy containers/scalars to plain Python.

    DD hands structured columns back through pandas, so a nested ``List[...]`` can
    surface as a ``numpy.ndarray`` -- truth-testing one (``not docs`` / ``docs or []``)
    raises "ambiguous truth value". Normalize to lists/dicts/scalars once, up front.
    """
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    tolist = getattr(obj, "tolist", None)  # numpy ndarray / scalar
    if callable(tolist) and not isinstance(obj, (str, bytes)):
        return _to_py(tolist())
    return obj


def _as_docset(val: Any) -> dict[str, Any]:
    """Coerce a DD structured column value into a ``{"documents": [...]}`` dict,
    with ``documents`` guaranteed to be a plain Python list."""
    if val is None:
        return {"documents": []}
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return {"documents": []}
    elif hasattr(val, "model_dump"):
        val = val.model_dump()
    val = _to_py(val)
    if not isinstance(val, dict):
        return {"documents": []}
    docs = val.get("documents")
    val["documents"] = list(docs) if isinstance(docs, (list, tuple)) else []
    return val


# ── Collect Phase-2 results into per-institution banks ──────────────────


def collect_institutions(result_df, spec: RegionSpec) -> list[InstitutionBank]:
    """Group cluster rows by institution, QC-gate, explode, and de-dupe doc ids.

    QC-drop policy: an unparseable/empty cluster is dropped (with a warning);
    a low-faithfulness cluster is KEPT (already ``placeholder: true``) and logged,
    so we never leave silent gaps that would break future task gold links.
    """
    spec_by_id = {i.id: i for i in spec.institutions}
    docs_by_inst: dict[str, dict[str, dict[str, Any]]] = {}

    for _, row in result_df.iterrows():
        iid = str(row["institution_id"])
        inst = spec_by_id.get(iid)
        if inst is None:
            continue
        docset = _as_docset(row.get("doc_set"))
        if not docset.get("documents"):
            logger.warning("empty/unparseable doc_set for %s (%s) — dropping cluster", iid, row.get("cluster_kind"))
            continue
        faith = float(row.get("numeric_faithfulness_score") or 0)
        if faith < MIN_FAITHFULNESS_SCORE:
            logger.warning(
                "low faithfulness (%.1f) for %s %s cluster; keeping flagged", faith, iid, row.get("cluster_kind")
            )
        plan_raw = row.get("doc_plan_json")
        plan = json.loads(plan_raw) if isinstance(plan_raw, str) else (plan_raw or [])
        spec_by_key = {s["doc_key"]: s for s in plan}
        docs = explode_docset(
            docset,
            id_prefix=iid.upper().replace("_", "-"),
            institution_id=iid,
            spec_by_key=spec_by_key,
            default_domain=inst.domains[0],
        )
        bucket = docs_by_inst.setdefault(iid, {})
        for d in docs:
            bucket.setdefault(d["id"], d)  # de-dupe ids across the institution's rows

    institutions: list[InstitutionBank] = []
    for inst in spec.institutions:
        institutions.append(
            InstitutionBank(
                institution_meta=_institution_meta(inst),
                documents=list(docs_by_inst.get(inst.id, {}).values()),
                tools=[_tool_dict(t) for t in inst.tool_taxonomy],
                templates=[],  # tasks authored separately
            )
        )
    return institutions


#: Doc-set (Phase 2) backfill policy. The doc-gen hub can transiently 5xx an
#: individual cluster; DD drops that record instead of retrying, so we re-run
#: generation for ONLY the dropped clusters (matched by ``cluster_id``) up to a few
#: attempts. This makes an unattended, many-region run robust to flaky per-record
#: failures without regenerating the whole (expensive) corpus.
DOCSET_MAX_ATTEMPTS = 3
DOCSET_RETRY_BASE_DELAY_S = 10.0


def _usable_docset_rows(df):
    """The subset of ``df`` whose ``doc_set`` parsed into at least one document.

    This is the success predicate for the backfill loop, and it has to be about
    CONTENT rather than about a row coming back. A truncated, empty or refused
    response still carries its ``cluster_id``, so a presence test counts it as a
    success and retries nothing.

    Deliberately does NOT filter on faithfulness. A low-faithfulness cluster is
    kept and flagged by ``collect_institutions``; that is a judgement about
    quality, not about whether the call produced anything to judge.
    """
    if "doc_set" not in df.columns:
        # Nothing to inspect (a shape we do not produce, but the loop must not
        # discard every row if it ever happens) -- fall back to presence.
        return df
    return df[df["doc_set"].map(lambda v: bool(_as_docset(v)["documents"]))]


def _generate_docsets_with_backfill(
    models: Any,
    cluster_seed,
    *,
    max_attempts: int = DOCSET_MAX_ATTEMPTS,
    base_delay: float = DOCSET_RETRY_BASE_DELAY_S,
):
    """Run Phase-2 doc_set generation, auto-regenerating any cluster that did not
    come back usable.

    Two things can go wrong, and both are equally recoverable:

    - **The record is dropped.** DD omits a record on a generation error (a
      transient hub 5xx), so the result frame is short of the seed.
    - **The record arrives unusable.** A response can be truncated mid-body,
      schema-valid but empty, or refused, and still carry its ``cluster_id``.

    We diff by ``cluster_id`` against the rows that produced at least one
    document (see :func:`_usable_docset_rows`) and re-run generation for the
    rest, up to ``max_attempts`` with linear backoff. Judging on content rather
    than row presence matters: on a presence test the second failure marked the
    cluster done, and the only thing that noticed was ``collect_institutions``
    dropping it with a warning and no path back to regeneration.

    Returns the combined result frame (usable clusters only); logs a clear
    warning if any clusters remain missing after the final attempt (the corpus
    is still written without them, and a later run can fill the gap since ids
    are deterministic).
    """
    import pandas as pd

    if "cluster_id" not in cluster_seed.columns:  # defensive: no keys -> single pass
        engine, builder = build_docset_config(models, cluster_seed)
        return engine.create(builder, num_records=len(cluster_seed)).load_dataset().reset_index(drop=True)

    total = len(cluster_seed)
    frames: list[Any] = []
    done: set = set()
    remaining = cluster_seed
    for attempt in range(1, max_attempts + 1):
        batch_seed = remaining.reset_index(drop=True)
        engine, builder = build_docset_config(models, batch_seed)
        df = engine.create(builder, num_records=len(batch_seed)).load_dataset().reset_index(drop=True)
        usable = _usable_docset_rows(df)
        frames.append(usable)
        if "cluster_id" in usable.columns:
            done |= set(usable["cluster_id"].dropna())
        remaining = cluster_seed[~cluster_seed["cluster_id"].isin(done)]
        n_missing = len(remaining)
        if n_missing == 0:
            if attempt > 1:
                logger.info("doc_set backfill: all %d clusters generated after %d attempt(s)", total, attempt)
            break
        if attempt < max_attempts:
            delay = base_delay * attempt
            logger.warning(
                "doc_set backfill: %d/%d clusters dropped or came back unusable "
                "(transient hub error, or a truncated/empty response); "
                "regenerating just those in %.0fs (attempt %d/%d)",
                n_missing,
                total,
                delay,
                attempt + 1,
                max_attempts,
            )
            time.sleep(delay)
        else:
            logger.warning(
                "doc_set backfill exhausted: %d/%d clusters still missing after %d "
                "attempts; writing the corpus WITHOUT them -- re-run gen-assets to "
                "fill the gap (doc ids are deterministic)",
                n_missing,
                total,
                max_attempts,
            )
    result_df = pd.concat(frames, ignore_index=True)
    # We only retry missing clusters, so duplicates shouldn't occur -- guard anyway.
    if "cluster_id" in result_df.columns:
        result_df = result_df.drop_duplicates(subset="cluster_id", keep="first").reset_index(drop=True)
    return result_df


#: Embedding-pass retry policy. The hub's /embeddings route can throw transient
#: 5xx / timeouts on the health-check probe; embeddings are idempotent + cheap, so
#: retry with linear backoff before giving up (the caller keeps the corpus either way).
EMBED_MAX_ATTEMPTS = 4
EMBED_RETRY_BASE_DELAY_S = 6.0


def _embed_documents(
    institutions: list[InstitutionBank],
    models: Any,
    engine: Any,
    *,
    max_attempts: int = EMBED_MAX_ATTEMPTS,
    base_delay: float = EMBED_RETRY_BASE_DELAY_S,
) -> None:
    """Embedding pass: dense-embed each document body, attach per-institution.

    Retries transient DD generation errors (the embeddings endpoint is flaky under
    load); on final failure it re-raises so the caller can persist the corpus without
    embeddings rather than discarding it.
    """
    import data_designer.config as dd
    import pandas as pd
    from data_designer.interface.errors import DataDesignerGenerationError

    rows = [{"id": d["id"], "body": d["body"]} for ib in institutions for d in ib.documents]
    if not rows:
        return
    edf = pd.DataFrame(rows)

    from usersim.cli._models import to_model_configs

    result = None
    for attempt in range(1, max_attempts + 1):
        builder = dd.DataDesignerConfigBuilder(model_configs=to_model_configs(models))
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=edf[["body"]]))
        for col in build_embedding_columns():
            builder.add_column(col)
        try:
            result = engine.create(builder, num_records=len(edf)).load_dataset().reset_index(drop=True)
            break
        except DataDesignerGenerationError as e:
            if attempt == max_attempts:
                raise
            delay = base_delay * attempt
            logger.warning(
                "embedding pass attempt %d/%d failed (%s); retrying in %.0fs",
                attempt,
                max_attempts,
                str(e).splitlines()[0][:160],
                delay,
            )
            time.sleep(delay)

    # ORDERED seed + reset_index preserves row order, so ids align by position.
    id_to_vec = {doc_id: r["body_embedding"]["embeddings"][0] for doc_id, (_, r) in zip(edf["id"], result.iterrows())}
    for ib in institutions:
        ib.embeddings = {d["id"]: id_to_vec[d["id"]] for d in ib.documents if d["id"] in id_to_vec}


# ── Orchestrator ─────────────────────────────────────────────────────────


def generate_corpus(
    spec: RegionSpec,
    models: Any,
    out_dir: Any,
    *,
    docs_per_product: int = DEFAULT_DOCS_PER_PRODUCT,
    max_docs_per_cluster: int = DEFAULT_MAX_DOCS_PER_CLUSTER,
) -> Any:
    """Two-phase generation: brief -> explode -> clusters -> collect -> embed -> serialize.

    Needs the gen model aliases + network (validated up front). Generates the CORPUS
    + embeddings; task templates are authored separately, so existing ``tasks.yaml``
    is preserved (``write_tasks=False``).
    """
    validate_generation_models(models)  # validate endpoints/aliases first

    # Phase 1: one InstitutionBrief per institution.
    inst_seed = build_institution_seed(spec)
    brief_engine, brief_builder = build_brief_config(models, inst_seed)
    brief_df = brief_engine.create(brief_builder, num_records=len(inst_seed)).load_dataset().reset_index(drop=True)

    # Bridge: fan out to bounded per-product + shared cluster rows carrying the brief.
    cluster_seed = explode_to_cluster_rows(
        brief_df,
        spec,
        docs_per_product=docs_per_product,
        max_docs_per_cluster=max_docs_per_cluster,
    )

    # Phase 2: generate each cluster conditioned on the brief. Auto-backfills any
    # clusters the hub transiently drops so a full corpus doesn't need a manual re-run.
    result_df = _generate_docsets_with_backfill(models, cluster_seed)

    institutions = collect_institutions(result_df, spec)

    # Persist the (expensive) corpus BEFORE embedding, so a flaky /embeddings
    # endpoint can't discard a full generation run. Embeddings are a cheap,
    # resumable sidecar written afterward.
    root = serialize_bank(
        out_dir,
        _region_meta_from_spec(spec),
        institutions,
        write_tasks=False,
    )
    briefs = {str(r["institution_id"]): _as_json(r.get("institution_brief")) for _, r in brief_df.iterrows()}
    _write_generation_report(root, result_df, institutions, models, docs_per_product, max_docs_per_cluster, briefs)

    # Embedding pass (retried; non-fatal). On success, write per-institution
    # sidecars; on failure the corpus is already on disk -- re-run to add them.
    try:
        _embed_documents(institutions, models, _make_engine(models))
    except Exception as e:  # noqa: BLE001  -- never lose the corpus over embeddings
        logger.warning(
            "embedding pass failed after retries (%s); corpus written to %s WITHOUT "
            "embeddings.parquet -- re-run gen-assets (or an embed-only step) to add them.",
            str(e).splitlines()[0][:160],
            root,
        )
    else:
        for inst in institutions:
            if inst.embeddings:
                idir = root / institution_rel_dir(inst.institution_meta.get("type"), inst.institution_id)
                _dump_embeddings(idir / "embeddings.parquet", inst.embeddings)
    return root


def _write_generation_report(
    root: Path,
    result_df,
    institutions: list[InstitutionBank],
    models: Any,
    docs_per_product: int,
    max_docs_per_cluster: int,
    briefs: dict[str, str] | None = None,
) -> None:
    """Write ``_generation_report.json`` (provenance + per-cluster QC) for the vet gate."""
    from datetime import datetime, timezone

    def _model(alias: str) -> str:
        spec = getattr(models, "get", lambda _a: None)(alias)
        return getattr(spec, "model", alias) if spec is not None else alias

    def _num(v: Any) -> Any:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _brief(iid: str) -> Any:
        raw = (briefs or {}).get(iid)
        if not raw:
            return None
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    by_inst: dict[str, list[dict[str, Any]]] = {}
    for _, row in result_df.iterrows():
        iid = str(row["institution_id"])
        docset = _as_docset(row.get("doc_set"))
        by_inst.setdefault(iid, []).append(
            {
                "cluster_kind": row.get("cluster_kind"),
                "target": int(row["target_doc_count"]) if "target_doc_count" in row else None,
                "produced": len(docset.get("documents", []) or []),
                "numeric_faithfulness": _num(row.get("numeric_faithfulness_score")),
                "cohesion": _num(row.get("cohesion_score")),
                "distinctness": _num(row.get("distinctness_score")),
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "doc_gen": _model(DOC_GEN_MODEL_ALIAS),
            "judge": _model(DOCSET_JUDGE_MODEL_ALIAS),
            "embedding": _model(EMBEDDING_MODEL_ALIAS),
        },
        "docs_per_product": docs_per_product,
        "max_docs_per_cluster": max_docs_per_cluster,
        "institutions": {
            ib.institution_id: {
                "n_documents": len(ib.documents),
                "clusters": by_inst.get(ib.institution_id, []),
                "brief": _brief(ib.institution_id),
            }
            for ib in institutions
        },
    }
    (root / "_generation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
