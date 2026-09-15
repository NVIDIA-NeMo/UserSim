# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generate+vet gate (asset_gen/financial_services/validate.py), offline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from usersim.asset_gen.financial_services.pipeline import InstitutionBank, serialize_bank
from usersim.asset_gen.financial_services.spec import RegionSpec
from usersim.asset_gen.financial_services.validate import FAIL, PASS, WARN, validate_bank

_SPEC_DICT: Dict[str, Any] = {
    "locale": "en_US", "region": "US", "currency": "USD", "currency_symbol": "$",
    "language": "English (US)", "document_types": ["product_sheet"],
    "domain_regulators": {"retail_banking": {"regulator_text": "x"}},
    "institutions": [{
        "id": "testbank", "display_name": "Test Bank", "type": "bank",
        "brand_voice": "traditional", "domains": ["retail_banking"],
        "products": [{
            "id": "checking", "display_name": "Checking", "domain": "retail_banking",
            "variables": {"overdraft_fee": {"type": "money", "value": 34}},
        }],
        "tool_taxonomy": [{"name": "kb_search", "discoverable": False,
                           "side_effect_class": "read_only"}],
        "task_templates": [{"id": "T1", "tier": "dynamic", "task_type": "advisory_qa",
                            "persona_tags": ["region:any"]}],
    }],
}


def _spec() -> RegionSpec:
    return RegionSpec.model_validate(_SPEC_DICT)


def _doc(doc_id: str, body: str) -> Dict[str, Any]:
    return {"id": doc_id, "title": "Checking overview", "body": body,
            "domain": "retail_banking", "product_category": "checking",
            "document_type": "product_sheet", "source_authority": "official",
            "placeholder": True, "mentions_tools": []}


def _write_bank(tmp_path: Path, locale: str, docs: List[Dict[str, Any]],
                embeddings: Optional[Dict[str, List[float]]] = None) -> Path:
    region_meta = {"locale": locale,
                   "domain_regulators": {"retail_banking": {"regulator_text": "x"}}}
    inst = InstitutionBank(
        institution_meta={"institution_id": "testbank", "type": "bank",
                          "domains": ["retail_banking"]},
        documents=docs,
        tools=[{"name": "kb_search", "description": "", "discoverable": False,
                "side_effect_class": "read_only", "parameters": {}}],
        embeddings=embeddings or {},
    )
    return serialize_bank(tmp_path / locale, region_meta, [inst], write_tasks=False)


def _status(report, check: str, scope: Optional[str] = None) -> Optional[str]:
    for r in report.results:
        if r.check == check and (scope is None or r.scope == scope):
            return r.status
    return None


def test_good_bank_passes_numeric_faithfulness(tmp_path: Path):
    bank = _write_bank(tmp_path, "en_US", [_doc("D1", "The overdraft fee is $34 per item.")])
    rep = validate_bank(bank, _spec())
    assert not rep.failed
    assert _status(rep, "numeric_faithfulness", "testbank") == PASS
    assert _status(rep, "language_script", "testbank") == PASS


def test_serialize_backfills_primitive_tool_schema(tmp_path: Path):
    """The bank fixture hands kb_search in with an EMPTY schema; serialize_bank
    backfills it, so the written tools.yaml has a real `query` param and the
    tool_schemas check passes."""
    bank = _write_bank(tmp_path, "en_US", [_doc("D1", "The overdraft fee is $34 per item.")])
    rep = validate_bank(bank, _spec())
    assert _status(rep, "tool_schemas", "testbank") == PASS
    tools = yaml.safe_load(
        (bank / "bank" / "testbank" / "tools.yaml").read_text()
    )["tools"]
    kb = next(t for t in tools if t["name"] == "kb_search")
    assert kb["parameters"]["properties"]["query"]["type"] == "string"


def test_empty_primitive_schema_fails_tool_schemas(tmp_path: Path):
    """A bank that predates the backfill (empty kb_search on disk) is caught."""
    bank = _write_bank(tmp_path, "en_US", [_doc("D1", "The overdraft fee is $34 per item.")])
    tools_path = bank / "bank" / "testbank" / "tools.yaml"
    tools_path.write_text(yaml.safe_dump({
        "schema_version": "v0.1",
        "tools": [{"name": "kb_search", "description": "", "discoverable": False,
                   "side_effect_class": "read_only", "parameters": {}}],
    }), encoding="utf-8")
    rep = validate_bank(bank, _spec())
    assert _status(rep, "tool_schemas", "testbank") == FAIL
    assert rep.failed


def test_wrong_number_fails_numeric_faithfulness(tmp_path: Path):
    bank = _write_bank(tmp_path, "en_US", [_doc("D1", "The overdraft fee is $35 per item.")])
    rep = validate_bank(bank, _spec())
    assert rep.failed
    assert _status(rep, "numeric_faithfulness", "testbank") == FAIL


def test_non_latin_locale_fails_language_script(tmp_path: Path):
    # ja_JP bank with an English body -> expected-script fraction ~0 -> FAIL.
    bank = _write_bank(tmp_path, "ja_JP",
                       [_doc("D1", "The overdraft fee is thirty four dollars.")])
    rep = validate_bank(bank)  # no region_spec -> numeric skipped
    assert _status(rep, "language_script", "testbank") == FAIL


def test_duplicate_embeddings_warn(tmp_path: Path):
    docs = [_doc("D1", "Overdraft fee is $34."), _doc("D2", "Overdraft fee is $34.")]
    emb = {"D1": [1.0, 0.0, 0.0], "D2": [1.0, 0.0, 0.0]}
    bank = _write_bank(tmp_path, "en_US", docs, emb)
    rep = validate_bank(bank, _spec())
    assert _status(rep, "near_duplicate", "testbank") == WARN


def test_cli_validate_assets(tmp_path: Path):
    from usersim.cli import main
    bank = _write_bank(tmp_path, "en_US", [_doc("D1", "The overdraft fee is $34 per item.")])
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_SPEC_DICT), encoding="utf-8")
    rc = main(["validate-assets", "--domain", "financial_services", "--locale", "en_US",
               "--dir", str(bank), "--region-spec", str(spec_path)])
    assert rc == 0


# ---------------------------------------------------------------------------
# numeric_faithfulness across number-formatting conventions
# ---------------------------------------------------------------------------


def test_numeric_faithfulness_accepts_indian_digit_grouping():
    """A typed value rendered in Indian grouping is still faithful.

    India groups the last three digits then in pairs, so 250000 reads
    "2,50,000" — a form the Western ``{:,}`` grouping never produces. Without
    this the validator would report every INR value as "absent from prose" and
    fail an otherwise perfect Hindi corpus.
    """
    from usersim.asset_gen.financial_services.validate import _indian_grouped, _value_present

    assert _indian_grouped(250_000) == "2,50,000"
    assert _indian_grouped(1_500_000) == "15,00,000"
    assert _indian_grouped(10_000_000) == "1,00,00,000"
    assert _indian_grouped(999) == "999"
    assert _indian_grouped(-250_000) == "-2,50,000"

    assert _value_present(250_000, "minimum balance of Rs 2,50,000 applies")
    # The Western rendering still counts, so en_US is unaffected.
    assert _value_present(250_000, "minimum balance of $250,000 applies")
    assert _value_present(250_000, "minimum balance of 250000 applies")


def test_numeric_faithfulness_accepts_lakh_and_crore_phrasing():
    from usersim.asset_gen.financial_services.validate import _value_present

    assert _value_present(250_000, "a ceiling of Rs 2.5 lakh per year")
    assert _value_present(150_000, "the 1.5 lakh limit under the tax section")
    assert _value_present(10_000_000, "cover up to Rs 1 crore")


def test_numeric_faithfulness_still_rejects_a_wrong_number():
    """Tolerating locale formats must not tolerate a DIFFERENT value."""
    from usersim.asset_gen.financial_services.validate import _value_present

    assert not _value_present(250_000, "minimum balance of Rs 2,60,000 applies")
    assert not _value_present(250_000, "a ceiling of Rs 3 lakh per year")
    assert not _value_present(250_000, "no figure at all here")


def test_variable_name_leak_flags_field_keys_in_prose(tmp_path: Path):
    """A doc that prints a spec FIELD KEY reads like an unfilled template.

    The numeric check cannot catch it (the number IS right) and the per-doc judge
    only registers it as slightly lower realism, so it needs a deterministic
    check. Observed at ~8-9% of documents in real runs, e.g. "we make ownership
    affordable with an interest_rate_annual of 0%".
    """
    leaky = _write_bank(tmp_path / "leaky", "en_US",
                        [_doc("D1", "Our overdraft_fee is $34 per item.")])
    rep = validate_bank(leaky, _spec())
    assert _status(rep, "variable_name_leak", "testbank") == WARN
    assert not rep.failed          # a quality flag, not a gate failure

    clean = _write_bank(tmp_path / "clean", "en_US",
                        [_doc("D1", "Our overdraft fee is $34 per item.")])
    rep = validate_bank(clean, _spec())
    assert _status(rep, "variable_name_leak", "testbank") == PASS


# ── tool names and doc_keys must not appear in customer-facing prose ─────────

def _bank_with(tmp_path: Path, docs: List[Dict[str, Any]],
               tools: Optional[List[Dict[str, Any]]] = None) -> Path:
    """A bank whose tool taxonomy includes a real state-changing tool to leak."""
    region_meta = {"locale": "en_US",
                   "domain_regulators": {"retail_banking": {"regulator_text": "x"}}}
    inst = InstitutionBank(
        institution_meta={"institution_id": "testbank", "type": "bank",
                          "domains": ["retail_banking"]},
        documents=docs,
        tools=tools or [
            {"name": "kb_search", "description": "", "discoverable": False,
             "side_effect_class": "read_only", "parameters": {}},
            {"name": "activate_card", "description": "", "discoverable": True,
             "side_effect_class": "state_changing", "parameters": {}},
        ],
    )
    return serialize_bank(tmp_path / "en_US", region_meta, [inst], write_tasks=False)


def _typed(doc_id: str, genre: str, body: str) -> Dict[str, Any]:
    d = _doc(doc_id, body)
    d["document_type"] = genre
    return d


def test_tool_name_in_customer_facing_prose_warns(tmp_path: Path):
    """Regression: ~10% of how_to/troubleshooting docs named an internal tool in BOTH
    locales, e.g. "the agent will use the internal activate_card tool" inside a
    document whose own convention calls it CUSTOMER-facing."""
    bank = _bank_with(tmp_path, [
        _typed("D1", "how_to_guide",
               "The agent will then use activate_card to switch your card on."),
    ])
    rep = validate_bank(bank, None)
    assert _status(rep, "tool_name_leak", "testbank") == WARN


def test_agent_facing_genres_may_name_tools(tmp_path: Path):
    """policy_procedure / discoverable_tool_doc / regulatory_note exist to document
    the tool; regulatory_note's convention is literally "the AGENT's obligation"."""
    for genre in ("policy_procedure", "discoverable_tool_doc", "regulatory_note"):
        bank = _bank_with(tmp_path / genre, [
            _typed("D1", genre, "Call activate_card once identity is verified."),
        ])
        rep = validate_bank(bank, None)
        assert _status(rep, "tool_name_leak", "testbank") == PASS, genre


def test_ordinary_prose_does_not_false_positive_as_a_tool_name(tmp_path: Path):
    """Tool names are snake_case, so the English phrase must not trip the check."""
    bank = _bank_with(tmp_path, [
        _typed("D1", "how_to_guide", "Ask us to activate card services for you."),
    ])
    rep = validate_bank(bank, None)
    assert _status(rep, "tool_name_leak", "testbank") == PASS


def test_raw_doc_key_reference_warns(tmp_path: Path):
    """Docs citing "1-Year Fixed Deposit - faq_5" hand the reader an internal slug and
    defer their own substance; they average 2.38 on self-containment vs 2.78."""
    bank = _bank_with(tmp_path, [
        _typed("D1", "fee_schedule",
               "For penalty calculations refer to 1-Year Fixed Deposit - faq_5."),
    ])
    rep = validate_bank(bank, None)
    assert _status(rep, "doc_key_leak", "testbank") == WARN


def test_title_cross_reference_is_fine(tmp_path: Path):
    bank = _bank_with(tmp_path, [
        _typed("D1", "fee_schedule",
               "See 1-Year Fixed Deposit - fee schedule for the penalty detail."),
    ])
    rep = validate_bank(bank, None)
    assert _status(rep, "doc_key_leak", "testbank") == PASS


def test_a_tool_named_like_a_doc_key_is_not_double_reported(tmp_path: Path):
    """A tool whose own name ends in _policy (renew_policy, update_policy) matches the
    ``<tool>_policy`` doc_key shape. It is a tool-name leak, not a doc_key leak, and
    must be blamed under exactly one check."""
    tools = [
        {"name": "kb_search", "description": "", "discoverable": False,
         "side_effect_class": "read_only", "parameters": {}},
        {"name": "renew_policy", "description": "", "discoverable": True,
         "side_effect_class": "state_changing", "parameters": {}},
    ]
    bank = _bank_with(tmp_path, [
        _typed("D1", "how_to_guide", "We will call renew_policy for you."),
    ], tools=tools)
    rep = validate_bank(bank, None)
    assert _status(rep, "tool_name_leak", "testbank") == WARN
    assert _status(rep, "doc_key_leak", "testbank") == PASS


# ── provenance has to survive the freeze ─────────────────────────────────────

def test_freeze_promotes_the_generation_report():
    """``_generation_report.json`` is written to the scratch --out dir, and the freeze
    cell promotes a fixed list of files into ``assets/``. The report was not on that
    list, so every frozen bank lost its provenance: ``qc_scores`` could only report
    "provenance/QC unknown", and ``build_eval_seed`` reads each institution's GENERATED
    brief out of the same file -- so re-judging a frozen bank scored brief-consistency
    against an empty brief. Pin it, since the symptom is a WARN and a silently weaker
    judge rather than a failure.
    """
    import json as _json

    from usersim import asset_gen

    # Derived from the package, not the working directory: a relative path
    # here turns a real regression into a skip that nobody reads.
    nb_path = (
        Path(asset_gen.__file__).resolve().parent
        / "notebooks" / "finance_assets.ipynb"
    )
    assert nb_path.is_file(), f"asset notebook is missing from the package: {nb_path}"
    nb = _json.loads(nb_path.read_text())
    freeze = [s for s in ("".join(c.get("source", [])) for c in nb["cells"])
              if "FREEZE" in s and "shutil" in s]
    assert freeze, "no freeze cell found"
    assert any("_generation_report.json" in s for s in freeze)


# ── a tool doc must not require an id nobody can supply ──────────────────────
#
# tools.yaml carries EMPTY parameters on purpose: the agent learns a tool's signature
# by retrieving its discoverable_tool_doc (the discovery mechanic). That makes the doc
# authoritative, so an invented argument stops the action dead. Seen for real:
# solstice_invest's manage_authorized_users doc specified `user_id: string` for the
# spouse being added, so the agent collected the spouse's name, DOB and phone and then
# refused to act for want of an id -- while northwind's and meridian's versions of the
# same tool asked for `user_details` and worked.

def test_tool_doc_requiring_an_unsourced_id_warns(tmp_path: Path):
    bank = _bank_with(tmp_path, [
        _typed("D1", "discoverable_tool_doc",
               "manage_authorized_users(account_id: string, user_id: string). "
               "user_id: the unique identifier of the person being added."),
    ])
    assert _status(validate_bank(bank, None), "unsourced_tool_argument", "testbank") == WARN


def test_taking_the_person_s_details_is_fine(tmp_path: Path):
    """The working form: what the customer can actually tell the agent."""
    bank = _bank_with(tmp_path, [
        _typed("D1", "discoverable_tool_doc",
               "manage_authorized_users(account_id: string, user_details: object). "
               "user_details: the person's full name, date of birth and contact."),
    ])
    assert _status(validate_bank(bank, None), "unsourced_tool_argument", "testbank") == PASS


def test_account_id_is_not_flagged(tmp_path: Path):
    """account_id and card_id HAVE a source -- get_accounts resolves them from a
    last-4 -- so requiring one is correct and must not be reported."""
    bank = _bank_with(tmp_path, [
        _typed("D1", "discoverable_tool_doc",
               "freeze_card(account_id: string, card_id: string). Use get_accounts "
               "to resolve the account_id from the customer's last-4."),
    ])
    assert _status(validate_bank(bank, None), "unsourced_tool_argument", "testbank") == PASS


def test_only_tool_docs_are_scanned(tmp_path: Path):
    """A customer-facing genre mentioning an id in passing is not a signature."""
    bank = _bank_with(tmp_path, [
        _typed("D1", "faq", "Your customer_id appears on your statement header."),
    ])
    assert _status(validate_bank(bank, None), "unsourced_tool_argument", "testbank") == PASS


def test_the_check_actually_reads_the_documents(tmp_path: Path):
    """Guard against the vacuous pass this check shipped with for one commit.

    ``insts`` is a Dict[str, _InstData], so ``for inst in insts`` walks the KEYS --
    strings -- and ``getattr("testbank", "documents", [])`` is empty, which made every
    institution PASS no matter what its docs said. A check that cannot fail is worse
    than no check: it reports safety it never verified. Two offenders in one bank must
    be counted, which is only possible if the documents were really read.
    """
    bank = _bank_with(tmp_path, [
        _typed("D1", "discoverable_tool_doc", "add_user(user_id: string)"),
        _typed("D2", "policy_procedure", "Look up the client_id, then call the tool."),
    ])
    rep = validate_bank(bank, None)
    assert _status(rep, "unsourced_tool_argument", "testbank") == WARN
    detail = next(
        r.detail for r in rep.results
        if r.check == "unsourced_tool_argument" and r.scope == "testbank"
    )
    assert "2 tool doc(s)" in detail
