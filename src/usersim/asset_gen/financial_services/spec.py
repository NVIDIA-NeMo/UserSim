# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Region-agnostic generation spec + validation, declared as Pydantic models.

The ``region_spec`` is the generation INPUT (data, not code): a locale is a SET
of institutions, each spanning one or more domains. The models here ARE the
contract — Literal enums fix the shared vocabulary (document genres, institution
types, domains, tool side-effect classes, task types/tiers) and a couple of
``model_validator``s enforce the cross-field rules (per-institution gold-tool
scope integrity + per-domain regulator coverage). Replicating to a new region is
"author one region_spec YAML + regenerate" with zero code change.

Public surface (kept stable for callers/tests):
``RegionSpec``/``Institution``/``Product``/``Tool``/``TaskTemplate`` models,
``load_region_spec`` (returns a ``RegionSpec``), ``validate_region_spec`` (dict ->
raises ``RegionSpecError``), ``core_ontology`` and the ``DOMAINS`` /
``INSTITUTION_TYPES`` vocabulary sets (derived from the Literals).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, get_args

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

# ── Shared vocabulary (region-agnostic) — the enums ARE the contract ───

InstitutionType = Literal[
    "bank", "neobank", "credit_union", "brokerage", "wealth_manager",
    "robo_advisor", "retirement_provider", "insurer", "nbfc",
]
Domain = Literal[
    "retail_banking", "cards", "lending_mortgage", "wealth_management",
    "brokerage_investing", "retirement", "insurance", "payments", "crypto", "tax",
]
BrandVoice = Literal["traditional", "neobank", "boutique"]
DocumentType = Literal[
    "product_sheet", "fee_schedule", "policy_procedure", "discoverable_tool_doc",
    "faq", "promo_notice", "disclosure", "eligibility_matrix", "regulatory_note",
    # Added to diversify the corpus (so it isn't FAQ-dominated at scale) and to
    # create realistic near-neighbor distractors for retrieval:
    "how_to_guide", "troubleshooting", "security_advisory", "rate_sheet",
    "terms_conditions", "comparison",
]
SideEffectClass = Literal["read_only", "state_changing", "irreversible"]
TaskType = Literal[
    "advisory_qa", "product_comparison", "eligibility", "numerical",
    "transactional", "troubleshooting", "complaint_dispute", "onboarding_kyc",
    "fraud_triage", "cross_sell_resistance",
]
TaskTier = Literal["verifiable", "dynamic"]

# Vocabulary sets derived from the Literals (no hand-maintained frozensets).
INSTITUTION_TYPES: frozenset = frozenset(get_args(InstitutionType))
DOMAINS: frozenset = frozenset(get_args(Domain))
BRAND_VOICES: frozenset = frozenset(get_args(BrandVoice))
DOCUMENT_TYPES: frozenset = frozenset(get_args(DocumentType))
SIDE_EFFECT_CLASSES: frozenset = frozenset(get_args(SideEffectClass))
TASK_TYPES: frozenset = frozenset(get_args(TaskType))
TASK_TIERS: frozenset = frozenset(get_args(TaskTier))

# Which domains each institution type may operate in. This is the conditional
# constraint that keeps ``type`` and ``domains`` coherent: a locale authors
# institutions by hand (they are a grounded seed, not independently sampled), so
# the DD-idiomatic conditional sampler does not apply — the authoring-time
# equivalent is this compatibility map, enforced by an ``Institution`` validator.
# (e.g. an ``insurer`` may not offer ``retail_banking``.)
INSTITUTION_DOMAINS: Dict[str, frozenset] = {
    "bank": frozenset({"retail_banking", "cards", "lending_mortgage", "payments"}),
    "neobank": frozenset({"retail_banking", "cards", "payments", "crypto"}),
    "credit_union": frozenset({"retail_banking", "cards", "lending_mortgage", "payments"}),
    "brokerage": frozenset({"brokerage_investing"}),
    "wealth_manager": frozenset({"wealth_management", "brokerage_investing"}),
    "robo_advisor": frozenset({"wealth_management", "brokerage_investing", "retirement"}),
    "retirement_provider": frozenset({"retirement"}),
    "insurer": frozenset({"insurance"}),
    # Non-banking financial company: a deposit-less lender (consumer/EMI
    # finance, gold loans, vehicle loans). A dominant retail-credit channel in
    # several markets (e.g. India) with no deposit-taking licence, so it offers
    # lending + repayment rails but NOT retail_banking.
    "nbfc": frozenset({"lending_mortgage", "payments"}),
}

# Not tied to a model field, but part of the introspectable ontology.
INFORMATION_FORMATS: frozenset = frozenset({
    "prose", "table", "bulleted", "qa", "stepwise", "decision_tree", "form",
})
SOURCE_AUTHORITIES: frozenset = frozenset({"official", "marketing", "community"})
PERSONA_TAG_PREFIXES: frozenset = frozenset({
    "region", "age", "occupation-family", "education", "interest",
    "financial-literacy",
})
_PERSONA_TAG_RE = re.compile(
    r"^(" + "|".join(sorted(PERSONA_TAG_PREFIXES)) + r"):[a-zA-Z0-9_\-+]+$"
)


class RegionSpecError(ValueError):
    """A region_spec that violates the ontology contract."""


# ── Models ─────────────────────────────────────────────────────────────

class ProductVariable(BaseModel):
    """A typed product value (e.g. a fee or rate) the generator must state exactly."""

    type: str
    value: Any = None


class Product(BaseModel):
    id: str
    #: Canonical name. Kept in the LATIN script even for non-Latin locales: it is
    #: the stable label a derived romanized locale inherits unchanged, and Indian
    #: financial prose genuinely code-switches product names.
    display_name: str = ""
    #: Optional in-language rendering (e.g. Devanagari for hi_Deva_IN). Authored
    #: rather than left to the generator so a product is named CONSISTENTLY
    #: across a whole corpus instead of being re-translated per document.
    display_name_local: str = ""
    domain: Domain
    variables: Dict[str, ProductVariable] = Field(default_factory=dict)


class Tool(BaseModel):
    name: str
    discoverable: bool = False
    side_effect_class: SideEffectClass = "read_only"
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Common operations layer
# ---------------------------------------------------------------------------
# Account-lifecycle + servicing tools that EVERY institution should offer, plus
# a few high-value per-type operations. Injected into each institution's
# tool_taxonomy at spec load (see ``Institution._inject_common_operations``) so
# the generator documents them and they replicate consistently across locales
# WITHOUT being hand-repeated in every region spec. All are discoverable domain
# tools (reached via the ``call_tool`` gate; the pipeline generates their docs).
# A region spec that explicitly declares a tool of the same name wins (no dup).
COMMON_UNIVERSAL_OPERATIONS: Tuple[Tuple[str, SideEffectClass], ...] = (
    # Read primitive that lists a VERIFIED customer's accounts (id, type, last4,
    # balance). Analogous to tau-Banking's get_user_details: it lets the agent
    # resolve a last-4 the customer gives into the account_id that every
    # state-changing tool requires, instead of fabricating an id or deflecting.
    ("get_accounts", "read_only"),
    ("open_account", "state_changing"),
    ("close_account", "irreversible"),
    ("update_contact_info", "state_changing"),
    ("get_statements", "read_only"),
    ("update_beneficiary", "state_changing"),
    ("set_alerts", "state_changing"),
)

# Deposit-taking institutions (bank / neobank / credit_union) share a servicing
# set; brokerages / robo-advisors share a trading + cash set. Most of these are
# never gold for a task — they exist as realistic, gold-ADJACENT tool docs so
# retrieval must discriminate the right procedure among many similar ones (the
# tau-Knowledge difficulty), and so the discoverable-tool gate is non-trivial.
_DEPOSIT_OPS: Tuple[Tuple[str, SideEffectClass], ...] = (
    ("transfer_funds", "state_changing"),
    ("order_replacement_card", "state_changing"),
    ("stop_payment", "state_changing"),
    ("pay_bill", "state_changing"),
    ("activate_card", "state_changing"),
    ("dispute_status", "read_only"),
    ("manage_authorized_users", "state_changing"),
)
_BROKERAGE_OPS: Tuple[Tuple[str, SideEffectClass], ...] = (
    ("transfer_funds", "state_changing"),
    ("cancel_order", "state_changing"),
    ("get_order_status", "read_only"),
    ("deposit_funds", "state_changing"),
    ("withdraw_funds", "irreversible"),
    ("manage_authorized_users", "state_changing"),
)
COMMON_TYPE_OPERATIONS: Dict[str, Tuple[Tuple[str, SideEffectClass], ...]] = {
    "bank": _DEPOSIT_OPS,
    "neobank": _DEPOSIT_OPS,
    "credit_union": _DEPOSIT_OPS,
    "brokerage": _BROKERAGE_OPS,
    "robo_advisor": _BROKERAGE_OPS,
    "retirement_provider": (
        ("request_withdrawal", "irreversible"),
        ("update_investment_election", "state_changing"),
        ("plan_loan", "state_changing"),
    ),
    "wealth_manager": (
        ("fund_account", "state_changing"),
        ("generate_financial_plan", "read_only"),
        ("manage_authorized_users", "state_changing"),
    ),
    "insurer": (
        ("file_claim", "state_changing"),
        ("get_claim_status", "read_only"),
        ("update_policy", "state_changing"),
    ),
    # Lenders deliberately do NOT reuse ``_DEPOSIT_OPS``: an NBFC holds no
    # deposits, so cheque stop-payments / card reissues / bill-pay make no
    # sense there. Servicing revolves around the loan lifecycle instead.
    "nbfc": (
        ("get_loan_details", "read_only"),
        ("apply_loan", "state_changing"),
        ("pay_emi", "state_changing"),
        ("foreclose_loan", "irreversible"),
        ("update_repayment_mandate", "state_changing"),
    ),
}


def common_operations_for_type(
    institution_type: str,
) -> Tuple[Tuple[str, SideEffectClass], ...]:
    """(name, side_effect_class) pairs every institution of ``institution_type``
    offers: the universal set plus any type-specific additions."""
    return COMMON_UNIVERSAL_OPERATIONS + COMMON_TYPE_OPERATIONS.get(
        institution_type, ()
    )


class TaskTemplate(BaseModel):
    id: str
    tier: TaskTier
    task_type: TaskType
    persona_tags: List[str] = Field(default_factory=list)
    gold_tool_sequence: List[str] = Field(default_factory=list)

    @field_validator("persona_tags")
    @classmethod
    def _valid_persona_tags(cls, v: List[str]) -> List[str]:
        for tag in v:
            if not isinstance(tag, str) or not _PERSONA_TAG_RE.match(tag):
                raise ValueError(
                    f"persona_tag {tag!r} violates ontology prefixes "
                    f"{sorted(PERSONA_TAG_PREFIXES)}"
                )
        return v

    @model_validator(mode="after")
    def _verifiable_needs_gold(self) -> "TaskTemplate":
        if self.tier == "verifiable" and not self.gold_tool_sequence:
            raise ValueError(
                f"{self.id}: verifiable template needs a gold_tool_sequence"
            )
        return self


class Institution(BaseModel):
    id: str
    #: Canonical brand name, kept in the LATIN script (see ``Product.display_name``):
    #: a brand should read identically in a native-script locale and in its derived
    #: romanized twin, which only holds if it survives transliteration untouched.
    display_name: str = ""
    #: Optional in-language rendering of the brand for native-script locales.
    display_name_local: str = ""
    type: InstitutionType
    brand_voice: BrandVoice = "traditional"
    domains: List[Domain] = Field(min_length=1)
    products: List[Product] = Field(default_factory=list)
    tool_taxonomy: List[Tool] = Field(min_length=1)
    task_templates: List[TaskTemplate] = Field(min_length=1)

    @model_validator(mode="after")
    def _inject_common_operations(self) -> "Institution":
        """Merge the common-operations layer (account lifecycle + servicing +
        per-type ops) into the tool taxonomy. Runs FIRST so injected tools count
        as in-scope for the gold-tool check below; an explicitly-declared tool of
        the same name is left untouched (no duplicate)."""
        existing = {t.name for t in self.tool_taxonomy}
        for name, side_effect in common_operations_for_type(self.type):
            if name not in existing:
                self.tool_taxonomy.append(
                    Tool(name=name, discoverable=True, side_effect_class=side_effect)
                )
                existing.add(name)
        return self

    @model_validator(mode="after")
    def _domains_match_type(self) -> "Institution":
        allowed = INSTITUTION_DOMAINS.get(self.type, frozenset())
        bad = [d for d in self.domains if d not in allowed]
        if bad:
            raise ValueError(
                f"{self.id}: a {self.type!r} may not operate in domains {bad}; "
                f"allowed for this type: {sorted(allowed)}"
            )
        return self

    @model_validator(mode="after")
    def _products_within_domains(self) -> "Institution":
        offered = set(self.domains)
        bad = [(p.id, p.domain) for p in self.products if p.domain not in offered]
        if bad:
            raise ValueError(
                f"{self.id}: products {bad} declare a domain the institution does not "
                f"offer; institution domains: {sorted(offered)}"
            )
        return self

    @model_validator(mode="after")
    def _gold_tools_in_scope(self) -> "Institution":
        names = {t.name for t in self.tool_taxonomy}
        for tpl in self.task_templates:
            if tpl.tier != "verifiable":
                continue
            bad = [g for g in tpl.gold_tool_sequence if g not in names]
            if bad:
                raise ValueError(
                    f"{self.id}::{tpl.id}: gold_tool_sequence references tools {bad} "
                    f"not in this institution's taxonomy (cross-institution gold is invalid)"
                )
        return self


class RegulatorContext(BaseModel):
    regulators: List[str] = Field(default_factory=list)
    regulator_text: str

    @field_validator("regulator_text")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("needs a non-empty regulator_text")
        return v


#: Identity fields a region's agents may demand before acting on an account.
#: ``full_name`` / ``date_of_birth`` are universal; the rest are region-specific
#: national identifiers (e.g. India's PAN, issued by the income-tax department
#: and near-universally used for financial KYC).
IDENTITY_FIELDS: frozenset = frozenset({
    "full_name", "date_of_birth", "pan", "aadhaar_last4", "national_id_last4",
})

#: Region-agnostic default: name + date of birth, which every market accepts.
DEFAULT_IDENTITY_REQUIRED_FIELDS: Tuple[str, ...] = ("full_name", "date_of_birth")


class IdentityVerification(BaseModel):
    """Per-region KYC contract for the ``verify_identity`` framework primitive.

    Regions differ in what counts as proof of identity, so the required set is
    DATA rather than code: India expects a PAN alongside name + DOB, while the
    US stops at name + DOB. The simulator gates its mock ``verify_identity`` on
    this list and grounds the persona with matching values, so an agent that
    skips a required field fails verification instead of waving the customer
    through.
    """

    required_fields: List[str] = Field(
        default_factory=lambda: list(DEFAULT_IDENTITY_REQUIRED_FIELDS),
        min_length=1,
    )

    @field_validator("required_fields")
    @classmethod
    def _known_fields(cls, v: List[str]) -> List[str]:
        unknown = sorted(set(v) - IDENTITY_FIELDS)
        if unknown:
            raise ValueError(
                f"identity_verification.required_fields: unknown field(s) {unknown} "
                f"(known: {sorted(IDENTITY_FIELDS)})"
            )
        return v


#: Region-NEUTRAL fallback label for the account a listing tool reports, keyed by
#: domain. Deliberately generic: retail deposit products are named differently in
#: every market ("checking account" exists in the US and nowhere else, India says
#: savings / current), so the default must be true everywhere and regions override
#: it with local vocabulary via ``RegionSpec.account_labels``.
DEFAULT_DOMAIN_ACCOUNT_LABELS: Dict[str, str] = {
    "retail_banking": "deposit account",
    "payments": "payment account",
    "cards": "credit card account",
    "lending_mortgage": "loan account",
    "brokerage_investing": "brokerage account",
    "retirement": "retirement account",
    "wealth_management": "managed investment account",
    "insurance": "policy",
    "crypto": "crypto account",
}


class DocTitleVocabulary(BaseModel):
    """In-language overrides for the vocabulary the doc PLAN builds titles from.

    Document titles are minted deterministically by ``build_product_doc_plan`` /
    ``build_shared_doc_plan`` as ``<product> <sep> <angle>``, and that string is
    also the sibling cross-reference handle the whole institution shares. So the
    angle words are reader-facing prose, and in a non-English locale they have to
    be authored in that language: leaving them to the generator produced a corpus
    whose bodies were 93% Devanagari while 44% of customer-facing TITLES were
    still English, because the model was simultaneously told to write titles in
    Hindi and to reproduce each sibling's title EXACTLY. It cannot do both, and
    the concatenated title+body script check could not see the split (a 40-char
    English title inside a 2,000-char Hindi body still scores ~0.93).

    Every field is optional and an empty value falls back to the English default
    in ``pipeline.py``, so a locale that omits the block — en_US and en_IN today —
    plans byte-for-byte as before.
    """

    #: Per-genre angle lists, keyed by document_type; overrides one genre at a
    #: time, so a locale can translate ``fee_schedule`` and inherit the rest.
    product_genre_angles: Dict[str, List[str]] = Field(default_factory=dict)
    faq_intents: List[str] = Field(default_factory=list)
    how_to_intents: List[str] = Field(default_factory=list)
    troubleshooting_intents: List[str] = Field(default_factory=list)
    comparison_intents: List[str] = Field(default_factory=list)
    #: Per-variable angle templates; must contain a ``{v}`` slot for the label.
    faq_variable_template: str = ""
    how_to_variable_template: str = ""
    troubleshooting_variable_template: str = ""
    #: Machine variable name -> reader-facing label. Unmapped names keep the
    #: default underscores-to-spaces rendering, which is why en_IN currently
    #: ships titles like "below mab charge explained".
    variable_labels: Dict[str, str] = Field(default_factory=dict)
    #: Machine domain -> reader-facing label, for the regulatory-note topic.
    domain_labels: Dict[str, str] = Field(default_factory=dict)
    #: Separator between the product name and the angle.
    topic_separator: str = ""
    #: Shared-cluster topics. ``{domain}`` / ``{tool}`` are substituted; a tool
    #: name stays VERBATIM because it is an identifier the agent must call.
    fee_overview_topic: str = ""
    general_faq_topic: str = ""
    regulatory_note_topic: str = ""
    tool_policy_topic: str = ""
    tool_reference_topic: str = ""

    @model_validator(mode="after")
    def _templates_keep_their_slots(self) -> "DocTitleVocabulary":
        """A template that loses its placeholder silently collapses every doc of
        that genre onto one title, which then de-duplicates down to a single
        document. Fail at spec load instead."""
        for field, slot in (
            ("faq_variable_template", "{v}"),
            ("how_to_variable_template", "{v}"),
            ("troubleshooting_variable_template", "{v}"),
            ("regulatory_note_topic", "{domain}"),
            ("tool_policy_topic", "{tool}"),
            ("tool_reference_topic", "{tool}"),
        ):
            value = getattr(self, field)
            if value and slot not in value:
                raise ValueError(
                    f"doc_titles.{field} = {value!r} must contain {slot}"
                )
        return self


class RegionSpec(BaseModel):
    schema_version: str = "v0.1"
    #: Optional locale of a BASE spec in the same directory to inherit from.
    #: A ``region_spec`` is keyed by LOCALE, but institutions / products /
    #: regulators are properties of the REGION — identical across every language
    #: spoken there. India alone has three surfaces (en_IN, hi_Deva_IN,
    #: hi_Latn_IN), so without inheritance the same 400-line model would be
    #: copied per language and drift the first time a fee changed in only one.
    #: A language variant therefore overlays just what differs (locale, language,
    #: in-language names). Resolved at load time; never serialized.
    extends: Optional[str] = None
    region: str
    locale: str
    currency: str
    currency_symbol: str = "$"
    #: Human-readable language for generated document CONTENT (e.g. "English (US)",
    #: "Japanese", "Brazilian Portuguese"). Threaded into the doc-gen prompt so a
    #: non-English locale produces native-language docs. Region-specific data, not
    #: derived from ``locale``, so authors control script/register explicitly.
    language: str
    number_format: Optional[str] = None
    privacy_regime: Optional[str] = None
    #: Optional; omitted means the region-agnostic name + DOB default.
    identity_verification: IdentityVerification = Field(
        default_factory=IdentityVerification
    )
    #: Optional per-domain label for the account the simulator's read tools report,
    #: in this region's own product vocabulary (US: "checking account"; India:
    #: "savings account"). Region data rather than code, because naming an account
    #: after a product the market does not have reads as the wrong institution to
    #: the customer. Omitted domains fall back to
    #: ``DEFAULT_DOMAIN_ACCOUNT_LABELS``; a task's authored ``account_type`` always
    #: wins over both.
    account_labels: Dict[str, str] = Field(default_factory=dict)
    #: Optional in-language overrides for the words the doc plan builds titles
    #: from. A LANGUAGE surface, not a region property, so it belongs in the
    #: language overlay next to ``display_name_local`` rather than in the region
    #: model. Omitted entirely => the English defaults in ``pipeline.py``.
    doc_titles: DocTitleVocabulary = Field(default_factory=DocTitleVocabulary)
    bank_id: Optional[str] = None
    bank_version: Optional[str] = None
    task_contract_version: Optional[str] = None
    document_types: List[DocumentType] = Field(min_length=1)
    domain_regulators: Dict[str, RegulatorContext] = Field(default_factory=dict)
    #: Optional per-institution-type emphasis for the generated brief, keyed by
    #: institution type. This is where MARKET-SPECIFIC framing belongs (US: RMDs,
    #: best-interest/fiduciary; India: NPS/PPF lock-ins, AMFI market-risk wording,
    #: gold-loan LTV). Unset types fall back to a region-neutral default, so a new
    #: locale never silently inherits another market's instruments or statutes.
    concept_guidance: Dict[str, str] = Field(default_factory=dict)
    institutions: List[Institution] = Field(min_length=1)

    @field_validator("concept_guidance")
    @classmethod
    def _known_types(cls, v: Dict[str, str]) -> Dict[str, str]:
        unknown = sorted(set(v) - INSTITUTION_TYPES)
        if unknown:
            raise ValueError(
                f"concept_guidance: unknown institution type(s) {unknown} "
                f"(known: {sorted(INSTITUTION_TYPES)})"
            )
        return v

    @field_validator("account_labels")
    @classmethod
    def _known_label_domains(cls, v: Dict[str, str]) -> Dict[str, str]:
        unknown = sorted(set(v) - DOMAINS)
        if unknown:
            raise ValueError(
                f"account_labels: unknown domain(s) {unknown} "
                f"(known: {sorted(DOMAINS)})"
            )
        return v

    @field_validator("region", "locale", "currency")
    @classmethod
    def _nonempty(cls, v: str, info: ValidationInfo) -> str:
        if not v or not v.strip():
            raise ValueError(f"missing/empty required field {info.field_name!r}")
        return v

    @model_validator(mode="after")
    def _domain_regulator_coverage(self) -> "RegionSpec":
        seen: set = set()
        for inst in self.institutions:
            if inst.id in seen:
                raise ValueError(f"{inst.id}: duplicate institution id")
            seen.add(inst.id)
        declared = {d for inst in self.institutions for d in inst.domains}
        missing = sorted(declared - set(self.domain_regulators))
        if missing:
            raise ValueError(
                f"domain_regulators: missing regulator context for domains {missing} "
                f"(every offered domain needs a regulator_text)"
            )
        return self


# ── Loading / validation entry points (stable public surface) ──────────

def _validate(raw: Any, *, src: str) -> RegionSpec:
    try:
        return RegionSpec.model_validate(raw)
    except ValidationError as e:
        raise RegionSpecError(f"{src}: {e}") from e


def _is_id_keyed(items: Any) -> bool:
    """True for a non-empty list whose every entry is a mapping carrying an ``id``.

    Structural rather than an allowlist of field names: one rule to remember
    ("entries with an ``id`` are patched by id, everything else is replaced"), and
    a new id-bearing list needs no change here. ``institutions``, ``products`` and
    ``task_templates`` all qualify; ``document_types`` / ``domains`` /
    ``gold_tool_sequence`` (plain strings) and ``tool_taxonomy`` (keyed by
    ``name``, not ``id``) do not, and so replace wholesale.
    """
    return bool(items) and all(
        isinstance(e, dict) and "id" in e for e in items
    )


def _merge_overlay(base: Any, overlay: Any, *, path: str = "") -> Any:
    """Deep-merge ``overlay`` onto ``base`` for region_spec inheritance.

    Mappings merge key-by-key. Lists of ``id``-bearing entries merge per id — an
    overlay entry patches the matching base entry, an unmatched one is appended,
    and order follows the base — so a language variant can rename a single
    product without repeating the lineup. Every other list replaces wholesale
    (``document_types`` is a set-like choice, not a patch).

    An overlay cannot DELETE an inherited entry or null out an inherited field;
    a variant that needs to drop institutions should be authored standalone
    rather than as an overlay.
    """
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = dict(base)
        for k, v in overlay.items():
            out[k] = _merge_overlay(base.get(k), v, path=f"{path}.{k}" if path else k)
        return out
    if (
        isinstance(base, list)
        and isinstance(overlay, list)
        and _is_id_keyed(base)
    ):
        if not _is_id_keyed(overlay):
            raise RegionSpecError(
                f"{path or '<list>'}: the base list is keyed by 'id', so every "
                f"overlay entry must be a mapping carrying an 'id' to patch "
                f"(got {overlay!r})"
            )
        by_id = {e["id"]: dict(e) for e in base}
        order = [e["id"] for e in base]
        for entry in overlay:
            eid = entry["id"]
            if eid in by_id:
                by_id[eid] = _merge_overlay(by_id[eid], entry, path=path)
            else:
                by_id[eid] = dict(entry)
                order.append(eid)
        return [by_id[i] for i in order]
    return overlay if overlay is not None else base


def load_region_spec(path: str | Path) -> RegionSpec:
    """Load + validate a region_spec YAML, resolving ``extends``.

    Raises :class:`RegionSpecError`.
    """
    raw, p = _read_spec_yaml(path)
    seen: List[str] = []
    while raw.get("extends"):
        base_locale = str(raw.pop("extends"))
        if base_locale in seen:
            raise RegionSpecError(
                f"{p}: circular extends chain via {base_locale!r} ({' -> '.join(seen)})"
            )
        seen.append(base_locale)
        base_path = p.parent / f"{base_locale}.yaml"
        if not base_path.exists():
            raise RegionSpecError(
                f"{p}: extends {base_locale!r} but {base_path} does not exist"
            )
        base_raw, _ = _read_spec_yaml(base_path)
        # The overlay wins; the base may itself extend another spec.
        raw = _merge_overlay(base_raw, raw)
    return _validate(raw, src=str(p))


def _read_spec_yaml(path: str | Path) -> Tuple[Dict[str, Any], Path]:
    import yaml

    p = Path(path)
    if not p.exists():
        raise RegionSpecError(f"{path}: file not found")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise RegionSpecError(f"{path}: top-level must be a mapping")
    return raw, p


def validate_region_spec(spec: Any, *, src: str = "<region_spec>") -> RegionSpec:
    """Validate a region_spec mapping. Raises :class:`RegionSpecError`.

    Rejects a mapping carrying ``extends``: resolving inheritance needs the spec
    DIRECTORY to find the base, which this signature does not take. Validating
    such a mapping would otherwise succeed while silently ignoring everything the
    base was meant to contribute.
    """
    if isinstance(spec, dict) and spec.get("extends"):
        raise RegionSpecError(
            f"{src}: 'extends' can only be resolved by load_region_spec(path) "
            f"(the base spec is looked up next to the file). Either call "
            f"load_region_spec, or inline the base's fields into this mapping."
        )
    return _validate(spec, src=src)


def core_ontology() -> Dict[str, Any]:
    """The full ontology vocabulary as a dict (for docs / introspection)."""
    return {
        "institution_types": sorted(INSTITUTION_TYPES),
        "domains": sorted(DOMAINS),
        "institution_domains": {k: sorted(v) for k, v in INSTITUTION_DOMAINS.items()},
        "brand_voices": sorted(BRAND_VOICES),
        "document_types": sorted(DOCUMENT_TYPES),
        "information_formats": sorted(INFORMATION_FORMATS),
        "source_authorities": sorted(SOURCE_AUTHORITIES),
        "side_effect_classes": sorted(SIDE_EFFECT_CLASSES),
        "task_types": sorted(TASK_TYPES),
        "task_tiers": sorted(TASK_TIERS),
        "persona_tag_prefixes": sorted(PERSONA_TAG_PREFIXES),
    }
