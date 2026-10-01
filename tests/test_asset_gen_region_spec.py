# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the asset_gen spec contract + the committed en_US region_spec."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from usersim.asset_gen.financial_services.spec import (
    DOMAINS,
    INSTITUTION_TYPES,
    RegionSpecError,
    core_ontology,
    load_region_spec,
    validate_region_spec,
)


def _asset_gen_dir() -> Path:
    """Directory the ``asset_gen`` package lives in, wherever that ends up."""
    import usersim.asset_gen

    return Path(usersim.asset_gen.__file__).resolve().parent


_REGION_SPEC_DIR = _asset_gen_dir() / "financial_services" / "region_spec"
_EN_US = _REGION_SPEC_DIR / "en_US.yaml"


def _raw(path: Path) -> dict:
    """Load the region_spec as a plain dict (for mutation-then-validate tests)."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_core_ontology_shape():
    onto = core_ontology()
    assert "brokerage" in onto["institution_types"]
    assert "retirement" in onto["domains"]
    assert "transactional" in onto["task_types"]
    assert "state_changing" in onto["side_effect_classes"]


def test_en_us_region_spec_valid_multi_institution():
    spec = load_region_spec(_EN_US)
    assert spec.locale == "en_US"
    institutions = spec.institutions
    # ~4 institutions spanning >=4 domains.
    assert len(institutions) >= 4
    types = {i.type for i in institutions}
    assert {"bank", "brokerage", "retirement_provider", "wealth_manager"} <= types
    all_domains = {d for i in institutions for d in i.domains}
    assert {"retail_banking", "brokerage_investing", "retirement", "wealth_management"} <= all_domains
    assert all_domains <= DOMAINS
    # Every offered domain has regulator context.
    assert all_domains <= set(spec.domain_regulators)
    # Every verifiable template's gold tools are within its OWN institution.
    for inst in institutions:
        tool_names = {t.name for t in inst.tool_taxonomy}
        for tpl in inst.task_templates:
            if tpl.tier == "verifiable":
                assert set(tpl.gold_tool_sequence) <= tool_names


def test_all_institution_types_in_vocabulary():
    spec = load_region_spec(_EN_US)
    assert {i.type for i in spec.institutions} <= INSTITUTION_TYPES


def test_common_operations_injected_into_every_institution():
    """Every institution gets the universal account-lifecycle + servicing tools
    without the region spec listing them, plus its per-type operations."""
    spec = load_region_spec(_EN_US)
    universal = {
        "open_account",
        "close_account",
        "update_contact_info",
        "get_statements",
        "update_beneficiary",
        "set_alerts",
    }
    for inst in spec.institutions:
        names = {t.name for t in inst.tool_taxonomy}
        assert universal <= names, f"{inst.id} missing universal ops: {universal - names}"
    by_type = {i.type: {t.name for t in i.tool_taxonomy} for i in spec.institutions}
    assert {"transfer_funds", "order_replacement_card", "stop_payment", "pay_bill"} <= by_type["bank"]
    assert {"cancel_order", "get_order_status", "deposit_funds"} <= by_type["brokerage"]
    assert {"request_withdrawal", "plan_loan"} <= by_type["retirement_provider"]
    assert "generate_financial_plan" in by_type["wealth_manager"]
    # Injection dedupes: no tool name appears twice in a taxonomy.
    for inst in spec.institutions:
        names = [t.name for t in inst.tool_taxonomy]
        assert len(names) == len(set(names)), f"{inst.id} has duplicate tools"


def test_common_operations_do_not_override_explicit_tools():
    """An explicitly-declared tool of a common-ops name keeps its own schema."""
    from usersim.asset_gen.financial_services.spec import Institution

    inst = Institution.model_validate(
        {
            "id": "x",
            "display_name": "X",
            "type": "bank",
            "brand_voice": "traditional",
            "domains": ["retail_banking"],
            "products": [{"id": "p", "domain": "retail_banking", "variables": {"fee": {"type": "money", "value": 1}}}],
            "tool_taxonomy": [
                {"name": "kb_search", "discoverable": False, "side_effect_class": "read_only"},
                {
                    "name": "open_account",
                    "discoverable": True,
                    "side_effect_class": "read_only",
                    "description": "custom",
                },
            ],
            "task_templates": [
                {"id": "T", "tier": "dynamic", "task_type": "advisory_qa", "persona_tags": ["region:any"]}
            ],
        }
    )
    open_tools = [t for t in inst.tool_taxonomy if t.name == "open_account"]
    assert len(open_tools) == 1  # not duplicated by injection
    assert open_tools[0].description == "custom"  # explicit declaration wins


def test_no_rho_bank_contamination():
    """Contamination policy: no reuse of tau-Banking's fictional institution."""
    raw = _EN_US.read_text().lower()
    assert "rho-bank" not in raw and "rho bank" not in raw


def test_validator_rejects_unknown_institution_type():
    bad = _raw(_EN_US)
    bad["institutions"][0]["type"] = "not_a_type"
    with pytest.raises(RegionSpecError):
        validate_region_spec(bad)


def test_validator_rejects_cross_institution_gold_tool():
    """Scope integrity: a template's gold tool must be in its own institution."""
    bad = _raw(_EN_US)
    for inst in bad["institutions"]:
        for tpl in inst["task_templates"]:
            if tpl["tier"] == "verifiable":
                tpl["gold_tool_sequence"] = ["some_other_institutions_tool"]
                break
    with pytest.raises(RegionSpecError):
        validate_region_spec(bad)


def test_validator_rejects_missing_domain_regulator():
    bad = _raw(_EN_US)
    bad["domain_regulators"].pop("retail_banking", None)
    with pytest.raises(RegionSpecError):
        validate_region_spec(bad)


def test_validator_rejects_bad_persona_tag():
    bad = _raw(_EN_US)
    bad["institutions"][0]["task_templates"][0]["persona_tags"] = ["not_a_prefix:x"]
    with pytest.raises(RegionSpecError):
        validate_region_spec(bad)


def test_validator_rejects_type_incompatible_domain():
    """Conditional coherence: an insurer may not operate in retail_banking."""
    bad = _raw(_EN_US)
    bad["institutions"][0]["type"] = "insurer"  # still declares retail_banking/cards
    with pytest.raises(RegionSpecError):
        validate_region_spec(bad)


def test_validator_rejects_product_domain_outside_institution():
    """A product may not declare a domain the institution does not offer."""
    bad = _raw(_EN_US)
    bad["institutions"][0]["products"][0]["domain"] = "crypto"
    with pytest.raises(RegionSpecError):
        validate_region_spec(bad)


def test_nbfc_is_a_deposit_less_lender_in_the_ontology():
    """``nbfc`` models a lender with no deposit-taking licence.

    Region-agnostic (any market with non-bank lenders can use it), but the
    motivating case is India, where NBFCs (consumer/EMI finance, gold loans) are
    a dominant retail-credit channel with no equivalent in the US institution
    set. It must be able to lend, and must NOT offer retail banking.
    """
    from usersim.asset_gen.financial_services.spec import (
        COMMON_TYPE_OPERATIONS,
        INSTITUTION_DOMAINS,
        common_operations_for_type,
    )

    assert "nbfc" in INSTITUTION_TYPES
    assert "nbfc" in core_ontology()["institution_types"]

    domains = INSTITUTION_DOMAINS["nbfc"]
    assert "lending_mortgage" in domains
    assert "retail_banking" not in domains

    # Loan-lifecycle servicing, and explicitly NOT the deposit servicing set
    # (no cheque stop-payment / card reissue / bill-pay at a deposit-less lender).
    ops = dict(common_operations_for_type("nbfc"))
    assert {"apply_loan", "pay_emi", "foreclose_loan"} <= set(ops)
    assert ops["foreclose_loan"] == "irreversible"
    assert ops["get_loan_details"] == "read_only"
    assert not ({"stop_payment", "order_replacement_card", "pay_bill"} & set(ops))
    assert COMMON_TYPE_OPERATIONS["nbfc"] is not COMMON_TYPE_OPERATIONS["bank"]


def test_identity_verification_defaults_to_name_and_dob():
    spec = load_region_spec(_EN_US)
    assert spec.identity_verification.required_fields == [
        "full_name",
        "date_of_birth",
    ]


def test_identity_verification_accepts_a_region_specific_field():
    raw = _raw(_EN_US)
    raw["identity_verification"] = {
        "required_fields": ["full_name", "date_of_birth", "pan"],
    }
    spec = validate_region_spec(raw)
    assert spec.identity_verification.required_fields[-1] == "pan"


def test_identity_verification_rejects_unknown_field():
    raw = _raw(_EN_US)
    raw["identity_verification"] = {"required_fields": ["full_name", "favourite_pet"]}
    with pytest.raises((RegionSpecError, ValueError)):
        validate_region_spec(raw)


def test_identity_verification_is_serialized_into_region_meta():
    """The sim reads the KYC contract off region_meta, so it must be emitted
    even at the default (otherwise the probe silently hardcodes name + DOB)."""
    from usersim.asset_gen.financial_services.pipeline import _region_meta_from_spec

    raw = _raw(_EN_US)
    raw["identity_verification"] = {
        "required_fields": ["full_name", "date_of_birth", "pan"],
    }
    meta = _region_meta_from_spec(validate_region_spec(raw))
    assert meta["identity_verification"]["required_fields"] == [
        "full_name",
        "date_of_birth",
        "pan",
    ]
    # Present at the default too.
    default_meta = _region_meta_from_spec(load_region_spec(_EN_US))
    assert default_meta["identity_verification"]["required_fields"] == [
        "full_name",
        "date_of_birth",
    ]


def test_concept_guidance_defaults_are_region_neutral():
    """The SHARED prompt must not carry any one market's instruments.

    US concepts (RMDs, Regulation Best Interest, Advisers Act fiduciary duty)
    live in the US region_spec as data. Hardcoding them in BRIEF_PROMPT's
    per-type branches would make every non-US locale inherit them.
    """
    from usersim.asset_gen.financial_services.prompts import (
        BRIEF_PROMPT,
        DEFAULT_TYPE_CONCEPT_GUIDANCE,
        DOCSET_PROMPT,
    )

    blob = " ".join(DEFAULT_TYPE_CONCEPT_GUIDANCE.values()).lower()
    for us_term in (
        "rmd",
        "regulation e",
        "regulation best interest",
        "advisers act",
        "fiduciary",
        "apy",
        "401(k)",
        "fdic",
    ):
        assert us_term not in blob, f"US concept {us_term!r} leaked into the default"
    for us_term in ("rmd", "regulation best interest", "advisers act", "apy"):
        assert us_term not in BRIEF_PROMPT.lower()
        assert us_term not in DOCSET_PROMPT.lower()
    # Every ontology type has an entry, so no type silently loses its emphasis.
    assert INSTITUTION_TYPES <= set(DEFAULT_TYPE_CONCEPT_GUIDANCE)


def test_concept_guidance_override_beats_the_default():
    from usersim.asset_gen.financial_services.prompts import concept_guidance_for

    spec = load_region_spec(_EN_US)
    # en_US supplies its own US framing...
    us = concept_guidance_for("retirement_provider", spec.concept_guidance)
    assert "RMDs" in us
    # ...while a type it does NOT override falls back to the neutral default.
    assert "no deposits" in concept_guidance_for("nbfc", spec.concept_guidance)
    # Unknown type with no override contributes nothing (rather than another
    # market's concepts).
    assert concept_guidance_for("made_up_type") == ""


def test_concept_guidance_rejects_unknown_institution_type():
    raw = _raw(_EN_US)
    raw["concept_guidance"] = {"not_an_institution": "..."}
    with pytest.raises((RegionSpecError, ValueError)):
        validate_region_spec(raw)


def test_institution_seed_carries_resolved_concept_guidance():
    from usersim.asset_gen.financial_services.pipeline import build_institution_seed

    spec = load_region_spec(_EN_US)
    seed = build_institution_seed(spec)
    assert "concept_guidance" in seed.columns
    by_type = dict(zip(seed["institution_type"], seed["concept_guidance"]))
    assert "RMDs" in by_type["retirement_provider"]
    assert all(v.strip() for v in by_type.values())


# ---------------------------------------------------------------------------
# hi_Deva_IN (India) region_spec
# ---------------------------------------------------------------------------

_EN_IN = _REGION_SPEC_DIR / "en_IN.yaml"  # the India REGION model
_HI_DEVA_IN = _REGION_SPEC_DIR / "hi_Deva_IN.yaml"  # a language overlay on it


def test_india_region_spec_valid_and_locally_grounded():
    spec = load_region_spec(_HI_DEVA_IN)
    assert spec.region == "IN"
    assert spec.locale == "hi_Deva_IN"
    assert spec.currency == "INR"
    assert spec.currency_symbol == "₹"
    assert "Hindi" in spec.language
    # India's KYC leans on PAN in addition to the universal pair.
    assert spec.identity_verification.required_fields == [
        "full_name",
        "date_of_birth",
        "pan",
    ]


def test_india_institution_set_is_prevalence_chosen_not_copied_from_us():
    """The five types are Indian retail touchpoints, not the US set.

    A UPI-first neobank and an NBFC lender are dominant in India and absent from
    en_US; conversely a standalone retirement provider / wealth manager is a US
    shape (in India those products are reached through a bank or a broker, so
    they live in the dynamic tier instead).
    """
    spec = load_region_spec(_HI_DEVA_IN)
    types = {i.type for i in spec.institutions}
    assert types == {"bank", "neobank", "brokerage", "insurer", "nbfc"}
    assert not ({"retirement_provider", "wealth_manager", "credit_union"} & types)
    # The deposit-less lender must not claim retail banking.
    nbfc = next(i for i in spec.institutions if i.type == "nbfc")
    assert "retail_banking" not in nbfc.domains
    assert "lending_mortgage" in nbfc.domains


def test_india_spec_matches_en_us_coverage_scale():
    """Volume parity with en_US (composition differs, scale should not)."""
    india = load_region_spec(_HI_DEVA_IN)
    us = load_region_spec(_EN_US)
    assert len(india.institutions) >= len(us.institutions)
    india_products = sum(len(i.products) for i in india.institutions)
    us_products = sum(len(i.products) for i in us.institutions)
    assert india_products >= us_products
    # Every institution seeds at least one verifiable and one dynamic template.
    for inst in india.institutions:
        tiers = {t.tier for t in inst.task_templates}
        assert "verifiable" in tiers, inst.id
        assert "dynamic" in tiers, inst.id


def test_india_spec_covers_every_offered_domain_and_type():
    spec = load_region_spec(_HI_DEVA_IN)
    offered = {d for i in spec.institutions for d in i.domains}
    # Load-time validation already enforces this; assert it so a future edit that
    # adds a domain without a regulator fails here with a clear message.
    assert offered <= set(spec.domain_regulators)
    assert {"payments", "lending_mortgage", "insurance"} <= offered
    # India framing is supplied for every type it ships (no silent fallback to
    # the region-neutral default for a type this locale actually authors).
    assert {i.type for i in spec.institutions} <= set(spec.concept_guidance)


def test_india_spec_carries_no_us_instruments():
    """US concepts must not leak into India's framing or regulator text."""
    spec = load_region_spec(_HI_DEVA_IN)
    blob = " ".join(
        [*spec.concept_guidance.values(), *(rc.regulator_text for rc in spec.domain_regulators.values())]
    ).lower()
    for us_term in ("rmd", "401(k)", "fdic", "regulation best interest", "advisers act", "finra", "cfpb", "apy"):
        assert us_term not in blob, f"US concept {us_term!r} leaked into hi_Deva_IN"
    # ...and the India regulators ARE named.
    for want in ("rbi", "sebi", "irdai", "npci", "dicgc"):
        assert want in blob, want


def test_india_gold_tools_resolve_including_injected_common_ops():
    """Seed gold sequences may name common-ops tools (file_claim, foreclose_loan).

    Those are injected at spec load rather than listed per institution, so this
    guards the injection actually reaching the taxonomy the verifier checks.
    """
    spec = load_region_spec(_HI_DEVA_IN)
    for inst in spec.institutions:
        names = {t.name for t in inst.tool_taxonomy}
        for tpl in inst.task_templates:
            for gold in tpl.gold_tool_sequence or []:
                assert gold in names, f"{inst.id}/{tpl.id}: {gold} not in taxonomy"
    insurer = next(i for i in spec.institutions if i.type == "insurer")
    assert "file_claim" in {t.name for t in insurer.tool_taxonomy}
    nbfc = next(i for i in spec.institutions if i.type == "nbfc")
    nbfc_tools = {t.name for t in nbfc.tool_taxonomy}
    assert {"apply_loan", "pay_emi", "foreclose_loan"} <= nbfc_tools
    # The deposit servicing set must NOT be injected into a deposit-less lender.
    assert not ({"stop_payment", "pay_bill", "order_replacement_card"} & nbfc_tools)


# ---------------------------------------------------------------------------
# Region-vs-language: `extends` inheritance
# ---------------------------------------------------------------------------


def test_language_overlay_inherits_the_whole_region_model():
    """India is ONE region with several language surfaces.

    Institutions / products / typed values / tools / regulators / KYC belong to
    the region, so the Hindi overlay must inherit them rather than restate them —
    copying a 400-line model per language drifts the first time a fee changes in
    only one file.
    """
    base = load_region_spec(_EN_IN)
    deva = load_region_spec(_HI_DEVA_IN)

    assert base.region == deva.region == "IN"
    assert base.locale == "en_IN" and deva.locale == "hi_Deva_IN"
    assert "English" in base.language and "Hindi" in deva.language

    def shape(spec):
        return [
            (
                i.id,
                i.type,
                tuple(i.domains),
                tuple(
                    (p.id, p.domain, tuple(sorted((k, v.value) for k, v in p.variables.items()))) for p in i.products
                ),
                tuple(t.name for t in i.tool_taxonomy),
                tuple(t.id for t in i.task_templates),
            )
            for i in spec.institutions
        ]

    assert shape(base) == shape(deva)
    assert base.domain_regulators.keys() == deva.domain_regulators.keys()
    assert (
        base.identity_verification.required_fields
        == deva.identity_verification.required_fields
        == ["full_name", "date_of_birth", "pan"]
    )
    assert base.concept_guidance == deva.concept_guidance


def test_overlay_adds_in_language_names_without_touching_canonical_ones():
    """Latin names stay canonical; the overlay only ADDS the local rendering.

    The Latin form is what a derived romanized locale inherits unchanged, so a
    brand reads the same in hi_Deva_IN and its hi_Latn_IN twin instead of being
    re-romanized into a different spelling.
    """
    base = load_region_spec(_EN_IN)
    deva = load_region_spec(_HI_DEVA_IN)

    for b, d in zip(base.institutions, deva.institutions):
        assert b.display_name == d.display_name  # canonical unchanged
        assert not b.display_name_local  # base needs none
        assert d.display_name_local  # overlay supplies one
        assert d.display_name.isascii()
        assert not d.display_name_local.isascii()  # actually Devanagari
        for bp, dp in zip(b.products, d.products):
            assert bp.display_name == dp.display_name
            assert dp.display_name_local, f"{dp.id} missing a local name"


def test_every_product_has_an_authored_local_name():
    """Left unauthored, the generator would re-translate a product per document
    and the corpus would name one product several ways (hurting retrieval)."""
    deva = load_region_spec(_HI_DEVA_IN)
    missing = [p.id for i in deva.institutions for p in i.products if not p.display_name_local]
    assert missing == []


def test_extends_rejects_a_missing_base(tmp_path):
    raw = _raw(_HI_DEVA_IN)
    raw["extends"] = "no_such_locale"
    p = tmp_path / "xx_YY.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(RegionSpecError):
        load_region_spec(p)


def test_extends_rejects_a_circular_chain(tmp_path):
    (tmp_path / "aa_AA.yaml").write_text("extends: bb_BB\n", encoding="utf-8")
    (tmp_path / "bb_BB.yaml").write_text("extends: aa_AA\n", encoding="utf-8")
    with pytest.raises(RegionSpecError):
        load_region_spec(tmp_path / "aa_AA.yaml")


def test_overlay_merges_any_id_keyed_list_and_replaces_the_rest():
    """One structural rule, not an allowlist of field names.

    task_templates carries ids too, so it must patch by id like institutions and
    products; string lists and tool_taxonomy (keyed by `name`) replace wholesale.
    """
    from usersim.asset_gen.financial_services.spec import _is_id_keyed, _merge_overlay

    assert _is_id_keyed([{"id": "a"}]) is True
    assert _is_id_keyed(["faq", "product_sheet"]) is False
    assert _is_id_keyed([{"name": "kb_search"}]) is False
    assert _is_id_keyed([]) is False

    patched = _merge_overlay(
        [{"id": "T1", "tier": "verifiable"}, {"id": "T2", "tier": "dynamic"}],
        [{"id": "T2", "task_type": "eligibility"}],
        path="task_templates",
    )
    assert patched == [
        {"id": "T1", "tier": "verifiable"},
        {"id": "T2", "tier": "dynamic", "task_type": "eligibility"},
    ]
    # A string list is a choice, not a patch.
    assert _merge_overlay(["faq", "disclosure"], ["faq"], path="document_types") == ["faq"]


def test_overlay_onto_an_id_keyed_list_must_carry_ids():
    from usersim.asset_gen.financial_services.spec import _merge_overlay

    with pytest.raises(RegionSpecError):
        _merge_overlay([{"id": "a"}], [{"display_name": "no id here"}], path="institutions")


def test_validate_region_spec_refuses_to_silently_drop_inheritance():
    """`extends` needs the spec directory, which this entry point lacks.

    Validating such a mapping used to succeed while ignoring everything the base
    was meant to contribute — a silent wrong answer.
    """
    raw = _raw(_EN_US)
    raw["extends"] = "en_IN"
    with pytest.raises(RegionSpecError, match="load_region_spec"):
        validate_region_spec(raw)


# ── product doc plan: coverage, ordering, and no duplicates ─────────────────
#: The four angles that answer the RELATIONSHIP questions a customer asks as often
#: as product questions: what evidence do I need, how do I open this, how do I close
#: it, why did my action not complete. They are product-scoped because they genuinely
#: differ per product (a 401(k) needs an employer sponsorship letter; a checking
#: account does not, and transfer limits are product variables).
_SERVICING_ANGLES = frozenset(
    {
        "required documentation",
        "set up and activate",
        "cancel, close, or downgrade",
        "a failed or pending action",
    }
)

#: The floor derived from the round-robin: with 12 genres, the latest of the four
#: lands at slot 18 (`eligibility_matrix` position 6, second angle).
_SERVICING_FLOOR = 18


def _all_products(path):
    spec = load_region_spec(path)
    return [(inst.id, p) for inst in spec.institutions for p in inst.products]


@pytest.mark.parametrize("path", [_EN_US, _EN_IN])
def test_declared_document_types_match_what_the_plan_emits(path):
    """``document_types`` is DECLARATIVE — the pipeline never reads it to gate
    generation, the doc plan decides. So it drifts silently: en_US declared 7
    genres while its committed corpus contained 15, which misleads anyone reading
    the spec to learn what a locale's corpus holds. Bind the two.
    """
    from usersim.asset_gen.financial_services.pipeline import (
        build_product_doc_plan,
        build_shared_doc_plan,
    )

    spec = load_region_spec(path)
    emitted = set()
    for inst in spec.institutions:
        emitted |= {s["document_type"] for s in build_shared_doc_plan(inst)}
        for product in inst.products:
            emitted |= {s["document_type"] for s in build_product_doc_plan(product, 40)}
    declared = set(spec.document_types)
    assert not (emitted - declared), f"{path.name}: plan emits undeclared genre(s) {sorted(emitted - declared)}"
    assert not (declared - emitted), f"{path.name}: declares genre(s) the plan never emits {sorted(declared - emitted)}"


@pytest.mark.parametrize("path", [_EN_US, _EN_IN])
@pytest.mark.parametrize("target", [_SERVICING_FLOOR, 40])
def test_every_product_gets_every_servicing_angle(path, target):
    """EVERY product, not most of them.

    Regression, and the reason this is parameterised per product rather than
    spot-checked: these angles used to sit behind per-variable angles that consumed
    every slot a genre received, so coverage depended on a product's variable count.
    In en_US exactly the 7 products with 2 variables got them and the other 11 did
    not — at any depth. A single-product assertion would have passed throughout.
    """
    from usersim.asset_gen.financial_services.pipeline import build_product_doc_plan

    for inst_id, product in _all_products(path):
        angles = {s["angle"] for s in build_product_doc_plan(product, target)}
        missing = _SERVICING_ANGLES - angles
        assert not missing, (
            f"{inst_id}/{product.id} ({len(product.variables)} variables) missing "
            f"{sorted(missing)} at docs_per_product={target}"
        )


@pytest.mark.parametrize("path", [_EN_US, _EN_IN])
@pytest.mark.parametrize("target", [18, 40, 45, 60])
def test_no_product_plan_ever_repeats_an_angle(path, target):
    """Duplicate ``(genre, angle)`` pairs differ only by audience and an index
    suffix, which is how near-duplicate documents got into the corpus.

    This replaces a hardcoded "docs_per_product must be <= N" rule: it stays correct
    as angle lists grow, where a fixed ceiling would go stale on the next addition.
    Past the supply the plan is expected to return FEWER slots than requested.
    """
    from usersim.asset_gen.financial_services.pipeline import build_product_doc_plan

    for inst_id, product in _all_products(path):
        slots = build_product_doc_plan(product, target)
        pairs = [(s["document_type"], s["angle"]) for s in slots]
        assert len(pairs) == len(set(pairs)), f"{inst_id}/{product.id} repeats an angle at docs_per_product={target}"
        assert len(slots) <= target
        # doc_keys must stay unique too -- they mint the deterministic doc uuids.
        keys = [s["doc_key"] for s in slots]
        assert len(keys) == len(set(keys)), f"{inst_id}/{product.id} duplicate doc_key"


def test_plan_under_delivers_rather_than_padding_past_the_angle_supply():
    """Supply is ``40 + 3 * n_variables``; asking for more must shrink, not repeat."""
    from usersim.asset_gen.financial_services.pipeline import build_product_doc_plan

    _, product = _all_products(_EN_IN)[0]
    supply = 40 + 3 * len(product.variables)
    slots = build_product_doc_plan(product, supply + 50)
    assert len(slots) < supply + 50
    pairs = [(s["document_type"], s["angle"]) for s in slots]
    assert len(pairs) == len(set(pairs))


def test_intent_first_and_variable_first_generators_differ_as_intended():
    """``how_to_guide`` / ``troubleshooting`` lead with intents (the customer-journey
    docs); ``faq`` leads with per-variable explainers (the value-bearing docs that
    back numeric faithfulness). Getting these backwards is the original bug."""
    from usersim.asset_gen.financial_services.pipeline import build_product_doc_plan

    _, product = next((i, p) for i, p in _all_products(_EN_IN) if len(p.variables) >= 4)
    slots = build_product_doc_plan(product, 40)

    def angles(genre):
        return [s["angle"] for s in slots if s["document_type"] == genre]

    # Intent-led genres: first angle is an intent, not "how to manage <variable>".
    assert angles("how_to_guide")[0] == "set up and activate"
    assert angles("troubleshooting")[0] == "a failed or pending action"
    # Variable-led genre: every value still gets an explainer.
    faq = angles("faq")
    for v in product.variables:
        assert f"{v.replace('_', ' ')} explained" in faq, v


@pytest.mark.parametrize("cluster_kind", ["shared", "product"])
def test_genre_conventions_are_scoped_to_the_cluster(cluster_kind):
    """A cluster must receive a convention for every genre it emits, and none for a
    genre it cannot. The flat block sent product clusters instructions for
    institution-only genres and vice versa, diluting both."""
    import re

    from jinja2 import Template

    from usersim.asset_gen.financial_services.pipeline import (
        build_product_doc_plan,
        build_shared_doc_plan,
    )
    from usersim.asset_gen.financial_services.prompts import DOCSET_PROMPT

    spec = load_region_spec(_EN_IN)
    inst = spec.institutions[0]
    if cluster_kind == "shared":
        emitted = {s["document_type"] for s in build_shared_doc_plan(inst)}
    else:
        emitted = set()
        for product in inst.products:
            emitted |= {s["document_type"] for s in build_product_doc_plan(product, 40)}

    rendered = Template(DOCSET_PROMPT).render(
        display_name="X",
        display_name_local="",
        institution_id="x",
        brand_voice="traditional",
        institution_type="bank",
        language="English",
        currency_symbol="$",
        institution_brief="{}",
        focus_json="{}",
        doc_plan_json="[]",
        sibling_doc_index="{}",
        target_doc_count=1,
        cluster_kind=cluster_kind,
    )
    documented = set(re.findall(r"^    (\w+):", rendered, re.M))
    assert not (emitted - documented), (
        f"{cluster_kind} cluster emits genre(s) with no convention: {sorted(emitted - documented)}"
    )
    assert not (documented - emitted), (
        f"{cluster_kind} cluster is given conventions for genre(s) it cannot emit: {sorted(documented - emitted)}"
    )


# ── the retrieved documents must not send the customer away ──────────────────
#: Genres whose documents describe how a task gets DONE, and which the assistant
#: retrieves mid-conversation. If one of them says "visit a branch" the assistant
#: relays that instead of calling the tool, which is a deflection failure scored
#: against the model for something the corpus told it.
_ACTION_GENRES = ("policy_procedure", "how_to_guide", "troubleshooting")


@pytest.mark.parametrize("genre", _ACTION_GENRES)
def test_action_genres_require_the_agent_to_act_in_conversation(genre):
    """Regression, region-agnostic because DOCSET_PROMPT is shared by every locale.

    `policy_procedure` always carried this constraint. `how_to_guide` and
    `troubleshooting` did not when their conventions were first written, and the
    en_US corpus generated from them told the customer to self-serve elsewhere in
    58% and 28% of those documents respectively (against 15% and 2% before) --
    e.g. "visit a branch with your government ID". Pin the constraint on all three.
    """
    import re

    from usersim.asset_gen.financial_services.prompts import DOCSET_PROMPT

    block = re.search(rf"^    {genre}:(.*?)(?=^    \w+:|^-|\Z)", DOCSET_PROMPT, re.M | re.S)
    assert block, f"no convention for {genre}"
    # Collapse the prompt's line wrapping: this asserts on wording, not layout.
    text = re.sub(r"\s+", " ", block.group(1)).lower()
    # It must name the agent-acts-in-conversation rule...
    assert "in the conversation" in text, genre
    # ...and explicitly rule out the self-serve destinations.
    for destination in ("portal", "branch", "form"):
        assert destination in text, f"{genre} does not rule out '{destination}'"


def test_how_to_guide_may_not_make_reaching_us_a_step():
    """Regression on the fix for the deflection fix.

    Barring the self-serve destinations left the model needing a first step, and it
    reached for "Contact us via phone, secure chat, or email" -- step 1 went from 29.5%
    to 82.2% of en_US how-to guides, and the genre paid for it: realism -0.30,
    self-containment -0.37, and triage 15% -> 32%, the worst move of any genre in that
    regeneration. The document spends its opening move telling a reader who is already
    mid-conversation to start one, and every guide then reads off the same template.
    """
    import re

    from usersim.asset_gen.financial_services.prompts import DOCSET_PROMPT

    block = re.search(r"^    how_to_guide:(.*?)(?=^    \w+:|^-|\Z)", DOCSET_PROMPT, re.M | re.S)
    assert block, "no convention for how_to_guide"
    text = re.sub(r"\s+", " ", block.group(1)).lower()
    assert "reaching us is not a step" in text
    assert "already talking to us" in text


def test_only_the_how_to_convention_bans_a_contact_step():
    """Scoped on purpose. `troubleshooting` sat at ~1% contact-steps before AND after,
    and its numbers IMPROVED in the same run (triage 27% -> 16%), so it needs no such
    rule; `policy_procedure` is agent-facing, where naming the mechanism is the point.
    A blanket ban would spend prompt budget on genres that never had the problem."""
    import re

    from usersim.asset_gen.financial_services.prompts import DOCSET_PROMPT

    for genre in ("troubleshooting", "policy_procedure"):
        block = re.search(rf"^    {genre}:(.*?)(?=^    \w+:|^-|\Z)", DOCSET_PROMPT, re.M | re.S)
        assert block, f"no convention for {genre}"
        assert "reaching us is not a step" not in re.sub(r"\s+", " ", block.group(1)).lower(), genre


def test_the_constraint_is_region_neutral():
    """It ships in the shared prompt, so it must not assume one market's channels --
    and it must still allow steps that really are physical (presenting an original
    document, having collateral valued), which India's KYC and gold loans require."""
    from usersim.asset_gen.financial_services.prompts import DOCSET_PROMPT

    # No market-specific institutions or instruments in the constraint text.
    for us_ism in ("FDIC", "IRS", "401(k)", "Regulation E", "RBI", "IRDAI", "UPI"):
        assert us_ism not in DOCSET_PROMPT, us_ism
    assert "genuinely physical" in DOCSET_PROMPT


def test_customer_facing_genres_are_told_not_to_name_tools():
    """Regression: ~10% of how_to_guide + troubleshooting docs named an internal tool
    in BOTH locales, e.g. "the agent will then use the internal activate_card tool"
    inside a document whose own convention calls it CUSTOMER-facing. The rule goes on
    the PRODUCT branch so it covers every customer-facing genre at once, and must NOT
    reach the shared branch, where naming the tool is the whole point."""
    import re

    from jinja2 import Template

    from usersim.asset_gen.financial_services.prompts import DOCSET_PROMPT

    base = dict(
        display_name="X",
        display_name_local="",
        institution_id="x",
        brand_voice="traditional",
        institution_type="bank",
        language="English",
        currency_symbol="$",
        institution_brief="{}",
        focus_json="{}",
        doc_plan_json="[]",
        sibling_doc_index="{}",
        target_doc_count=1,
    )
    product = re.sub(r"\s+", " ", Template(DOCSET_PROMPT).render(**base, cluster_kind="product")).lower()
    shared = re.sub(r"\s+", " ", Template(DOCSET_PROMPT).render(**base, cluster_kind="shared")).lower()

    rule = "none of them may name an internal tool"
    assert rule in product
    assert rule not in shared


def test_cross_references_must_use_titles_not_doc_keys():
    """Docs citing "1-Year Fixed Deposit - faq_5" hand the reader an internal slug and
    defer their own substance (self-containment 2.38 vs 2.78 for clean docs)."""
    import re

    from usersim.asset_gen.financial_services.prompts import DOCSET_PROMPT

    text = re.sub(r"\s+", " ", DOCSET_PROMPT).lower()
    assert "doc_key` is an internal slug" in text or "internal slug" in text
    assert "keys belong in `references` only" in text


def test_tool_docs_must_not_invent_unobtainable_arguments():
    """The doc IS the schema: tools.yaml ships empty parameters on purpose so the agent
    has to retrieve the discoverable_tool_doc to learn a signature. An invented argument
    is therefore authoritative and stops the action dead -- solstice_invest's
    manage_authorized_users doc asked for a `user_id` for the spouse being added, so the
    agent gathered her name, DOB and phone and then refused for want of an id, while the
    same tool at northwind and meridian asked for `user_details` and worked.
    """
    import re

    from jinja2 import Template

    from usersim.asset_gen.financial_services.prompts import DOCSET_PROMPT

    base = dict(
        display_name="X",
        display_name_local="",
        institution_id="x",
        brand_voice="traditional",
        institution_type="bank",
        language="English",
        currency_symbol="$",
        institution_brief="{}",
        focus_json="{}",
        doc_plan_json="[]",
        sibling_doc_index="{}",
        target_doc_count=1,
    )
    shared = re.sub(r"\s+", " ", Template(DOCSET_PROMPT).render(**base, cluster_kind="shared"))

    # Arguments must be obtainable in the conversation...
    assert "must be something the agent can actually obtain" in shared
    # ...and the two invented-id shapes are named explicitly, because the model
    # produced both: customer_id for the already-verified caller, user_id for a
    # third party the customer is naming for the first time.
    assert "`customer_id`" in shared and "`user_id`" in shared
    # The working alternative is spelled out rather than left implicit.
    assert "PERSON'S DETAILS" in shared

    # It belongs to the agent-facing branch only; product clusters never document a
    # signature, so spending their budget on this would be noise.
    product = re.sub(r"\s+", " ", Template(DOCSET_PROMPT).render(**base, cluster_kind="product"))
    assert "must be something the agent can actually obtain" not in product
