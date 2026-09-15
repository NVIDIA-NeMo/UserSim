# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the two-phase financial_services generation pipeline (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from usersim.asset_gen.financial_services.pipeline import (
    DOC_GEN_MODEL_ALIAS,
    EMBEDDING_MODEL_ALIAS,
    DocSet,
    InstitutionBank,
    _backfill_primitive_tools,
    build_brief_columns,
    build_docset_columns,
    build_embedding_columns,
    build_institution_seed,
    build_product_doc_plan,
    doc_length_hint,
    docset_qc_scores,
    explode_docset,
    explode_to_cluster_rows,
    serialize_bank,
)
from usersim.asset_gen.financial_services.spec import load_region_spec
from usersim.engine.core._assets import packaged_assets_dir

def _asset_gen_dir() -> Path:
    """Directory the ``asset_gen`` package lives in, wherever that ends up."""
    import usersim.asset_gen

    return Path(usersim.asset_gen.__file__).resolve().parent

_REGION_SPEC = (
    _asset_gen_dir() / "financial_services" / "region_spec" / "en_US.yaml"
)


def test_backfill_primitive_tools_fills_empty_primitive_schemas():
    """Region specs list kb_search / verify_identity by name only; serialization
    must backfill their description + parameters (with a real `query` on
    kb_search) so no tools.yaml ships an empty primitive schema."""
    tools = [
        {"name": "kb_search", "description": "", "discoverable": False,
         "side_effect_class": "read_only", "parameters": {}},
        {"name": "verify_identity", "description": "", "discoverable": False,
         "side_effect_class": "read_only", "parameters": {}},
        {"name": "file_transaction_dispute", "description": "", "discoverable": True,
         "side_effect_class": "state_changing", "parameters": {}},
    ]
    out = {t["name"]: t for t in _backfill_primitive_tools(tools)}
    assert out["kb_search"]["description"]
    assert out["kb_search"]["parameters"]["properties"]["query"]["type"] == "string"
    assert out["kb_search"]["parameters"]["required"] == ["query"]
    assert out["verify_identity"]["description"]
    # Domain (non-primitive) tools are left untouched — their args come from the
    # retrieved KB doc via the call_tool gate, not a typed schema.
    assert out["file_transaction_dispute"]["parameters"] == {}
    assert out["file_transaction_dispute"]["description"] == ""


def test_build_institution_seed_one_row_per_institution():
    spec = load_region_spec(_REGION_SPEC)
    seed = build_institution_seed(spec)
    assert len(seed) == len(spec.institutions)
    row = seed.iloc[0]
    assert row["language"] == spec.language  # cross-region content language threaded
    assert row["products_json"] and row["regulators_json"]


def test_explode_to_cluster_rows_batches_and_partitions():
    import pandas as pd

    spec = load_region_spec(_REGION_SPEC)
    # Stubbed Phase-1 output: one brief row per institution (no LLM).
    brief_df = pd.DataFrame([
        {"institution_id": i.id, "institution_brief": {"positioning": "p"}}
        for i in spec.institutions
    ])
    rows = explode_to_cluster_rows(
        brief_df, spec, docs_per_product=20, max_docs_per_cluster=8)

    for _, r in rows.iterrows():
        assert r["cluster_kind"] in ("product", "shared")
        # every cluster row carries the generated brief + its own context
        assert r["institution_brief"] and r["focus_json"] and r["doc_plan_json"]
        assert r["sibling_doc_index"] and int(r["target_doc_count"]) >= 1
        # per-doc, genre-aware length budget + stable uuid live in each plan entry
        import json as _json
        assert all(s.get("length") and s.get("uuid") for s in _json.loads(r["doc_plan_json"]))
    assert {"product", "shared"} <= set(rows["cluster_kind"])

    # 20 docs/product with cap 8 -> ceil(20/8)=3 product-cluster rows per product.
    import math
    nb = next(i for i in spec.institutions if i.id == "northwind_bank")
    nb_product = rows[(rows.institution_id == "northwind_bank")
                      & (rows.cluster_kind == "product")]
    assert len(nb_product) == len(nb.products) * math.ceil(20 / 8)
    # at least one shared cluster row per institution
    nb_shared = rows[(rows.institution_id == "northwind_bank")
                     & (rows.cluster_kind == "shared")]
    assert len(nb_shared) >= 1


def test_explode_to_cluster_rows_assigns_unique_cluster_ids():
    import pandas as pd

    spec = load_region_spec(_REGION_SPEC)
    brief_df = pd.DataFrame([
        {"institution_id": i.id, "institution_brief": {"positioning": "p"}}
        for i in spec.institutions
    ])
    rows = explode_to_cluster_rows(
        brief_df, spec, docs_per_product=20, max_docs_per_cluster=8)
    ids = list(rows["cluster_id"])
    assert all(ids), "every cluster row must carry a cluster_id"
    assert len(ids) == len(set(ids)), "cluster_id must be unique per cluster"
    # Deterministic: re-exploding the same brief yields identical ids.
    rows2 = explode_to_cluster_rows(
        brief_df, spec, docs_per_product=20, max_docs_per_cluster=8)
    assert list(rows2["cluster_id"]) == ids


_GOOD_DOCSET = {"documents": [{"doc_key": "faq_1", "title": "t", "body": "b"}]}


def _fake_docset_engine(monkeypatch, *, drop_by_attempt, bad_by_attempt=None):
    """Stub Phase-2 generation. ``drop_by_attempt[i]`` are the cluster_ids DD omits
    entirely on attempt i; ``bad_by_attempt[i]`` are the ones that come BACK but
    with an unusable payload (the truncated-response shape)."""
    from usersim.asset_gen.financial_services import pipeline as P

    calls: list = []
    bad_by_attempt = bad_by_attempt or {}

    class _Res:
        def __init__(self, df): self._df = df
        def load_dataset(self): return self._df

    class _Eng:
        def __init__(self, seed_df, drop, bad):
            self._seed = seed_df
            self._drop = drop
            self._bad = bad

        def create(self, builder, num_records=None):
            df = self._seed[~self._seed["cluster_id"].isin(self._drop)].copy()
            df["doc_set"] = [
                # A payload truncated mid-body does not parse, so _as_docset
                # coerces it to {"documents": []} -- but the row is HERE.
                '{"documents": [{"doc_key": "faq_1", "title": "t", "body": "trunc'
                if cid in self._bad else _GOOD_DOCSET
                for cid in df["cluster_id"]
            ]
            return _Res(df.reset_index(drop=True))

    def fake_build(models, cluster_seed):
        attempt = len(calls)
        calls.append(list(cluster_seed["cluster_id"]))
        return (
            _Eng(cluster_seed,
                 drop_by_attempt.get(attempt, set()),
                 bad_by_attempt.get(attempt, set())),
            object(),
        )

    monkeypatch.setattr(P, "build_docset_config", fake_build)
    monkeypatch.setattr(P.time, "sleep", lambda *_: None)
    return calls


def _seed(*cluster_ids):
    import pandas as pd
    return pd.DataFrame([
        {"cluster_id": cid, "doc_plan_json": "[]"} for cid in cluster_ids
    ])


def test_generate_docsets_with_backfill_regenerates_dropped_clusters(monkeypatch):
    """A cluster the hub drops on the first attempt is regenerated on its own on the
    next attempt (matched by cluster_id), so the final frame is complete."""
    from usersim.asset_gen.financial_services import pipeline as P

    calls = _fake_docset_engine(monkeypatch, drop_by_attempt={0: {"c2"}})
    result = P._generate_docsets_with_backfill(
        models=None, cluster_seed=_seed("c1", "c2", "c3"))

    assert set(result["cluster_id"]) == {"c1", "c2", "c3"}
    assert calls[0] == ["c1", "c2", "c3"]  # first attempt: everything
    assert calls[1] == ["c2"]              # retry: ONLY the dropped cluster


def test_generate_docsets_with_backfill_retries_an_unusable_payload(monkeypatch):
    """A response that ARRIVES but cannot be used gets the same retries as one the
    hub dropped.

    Success has to be judged on content. A truncated (or empty, or refused)
    payload still carries its cluster_id, so a presence-based predicate marked
    the cluster done and retried nothing -- the only thing that noticed was
    ``collect_institutions``, which drops the cluster with a warning and no path
    back to regeneration. Truncation is not hypothetical: a cluster is ten long
    documents in one response, and ``max_docs_per_cluster`` is a tunable knob.
    """
    from usersim.asset_gen.financial_services import pipeline as P

    calls = _fake_docset_engine(monkeypatch, drop_by_attempt={}, bad_by_attempt={0: {"c2"}})
    result = P._generate_docsets_with_backfill(
        models=None, cluster_seed=_seed("c1", "c2", "c3"))

    assert calls[1] == ["c2"], "the unusable cluster must be re-requested"
    assert set(result["cluster_id"]) == {"c1", "c2", "c3"}
    # And the row that survives is the GOOD one, not the truncated string.
    kept = {r["cluster_id"]: r["doc_set"] for _, r in result.iterrows()}
    assert kept["c2"] == _GOOD_DOCSET


def test_generate_docsets_with_backfill_drops_a_persistently_unusable_cluster(monkeypatch):
    """If it never comes back usable, it must not reach the corpus as an empty row.

    Attempts are bounded, and the cluster then surfaces in the 'backfill
    exhausted' warning rather than being silently dropped later at collect time.
    """
    from usersim.asset_gen.financial_services import pipeline as P

    bad = {i: {"c2"} for i in range(P.DOCSET_MAX_ATTEMPTS)}
    calls = _fake_docset_engine(monkeypatch, drop_by_attempt={}, bad_by_attempt=bad)
    result = P._generate_docsets_with_backfill(
        models=None, cluster_seed=_seed("c1", "c2", "c3"))

    assert set(result["cluster_id"]) == {"c1", "c3"}
    assert len(calls) == P.DOCSET_MAX_ATTEMPTS


def test_usable_docset_rows_keeps_low_faithfulness_clusters():
    """Faithfulness is a quality judgement made downstream, not a signal that the
    call failed. ``collect_institutions`` keeps those clusters and flags them, so
    the retry predicate must not filter them out."""
    import pandas as pd
    from usersim.asset_gen.financial_services import pipeline as P

    df = pd.DataFrame([
        {"cluster_id": "c1", "doc_set": _GOOD_DOCSET, "numeric_faithfulness_score": 1.0},
        {"cluster_id": "c2", "doc_set": _GOOD_DOCSET, "numeric_faithfulness_score": 4.0},
    ])
    assert set(P._usable_docset_rows(df)["cluster_id"]) == {"c1", "c2"}


def test_usable_docset_rows_falls_back_when_the_column_is_absent():
    """A frame with no doc_set column at all must not be emptied out."""
    import pandas as pd
    from usersim.asset_gen.financial_services import pipeline as P

    df = pd.DataFrame([{"cluster_id": "c1"}, {"cluster_id": "c2"}])
    assert set(P._usable_docset_rows(df)["cluster_id"]) == {"c1", "c2"}


def test_doc_length_hint_is_script_aware():
    # Latin scripts -> words; CJK -> characters; Indic -> a smaller word band
    # (heavier tokenization at the same token budget).
    assert "words" in doc_length_hint("en_US", "English (US)")
    assert "words" in doc_length_hint("pt_BR", "Brazilian Portuguese")
    assert "characters" in doc_length_hint("ja_JP", "Japanese")
    assert "characters" in doc_length_hint("ko_KR", "Korean")

    def _low(hint: str) -> int:
        return int(hint.split()[1].split("-")[0])

    # Devanagari word band sits below the Latin band (fewer words per token budget).
    assert _low(doc_length_hint("hi_Deva_IN", "Hindi")) < _low(doc_length_hint("en_US", "English"))


def test_doc_length_hint_gives_romanized_locales_the_latin_budget():
    """Romanized twins are Latin-script and must NOT get the Indic budget.

    ``hi_Latn_IN``'s tag carries its source language ("hi_latn_in hindi latin"),
    so a naive substring check matches the Indic branch on "hindi" and hands a
    romanized doc the much tighter Indic word band — under-filling every
    romanized document at the same token budget.
    """
    def _low(hint: str) -> int:
        return int(hint.split()[1].split("-")[0])

    latin = _low(doc_length_hint("en_US", "English"))
    assert _low(doc_length_hint("hi_Latn_IN", "Hindi Latin")) == latin
    assert _low(doc_length_hint("ta_Latn_IN", "Tamil")) == latin
    # ...while the native-script twins keep the (smaller) Indic band.
    assert _low(doc_length_hint("hi_Deva_IN", "Hindi Devanagari")) < latin
    assert _low(doc_length_hint("ta_Taml_IN", "Tamil")) < latin
    assert "words" in doc_length_hint("hi_Latn_IN", "Hindi Latin")


def test_doc_length_hint_is_genre_aware():
    # Narrative genres get a larger band than structured/terse genres, same locale.
    lo = lambda g: int(doc_length_hint("en_US", "English", g).split()[1].split("-")[0])
    assert lo("product_sheet") > lo("fee_schedule")      # rich > tight
    assert lo("policy_procedure") > lo("faq")
    assert lo("regulatory_note") > lo("eligibility_matrix")
    assert lo("disclosure") > lo("promo_notice")


def test_build_product_doc_plan_expands_to_target():
    spec = load_region_spec(_REGION_SPEC)
    product = spec.institutions[0].products[0]
    plan = build_product_doc_plan(product, target_docs=15)
    assert len(plan) == 15
    assert plan[0]["document_type"] == "product_sheet"
    assert all(s["product_category"] == product.id for s in plan)


def test_product_doc_plan_diversity_axes_make_repeats_distinct():
    spec = load_region_spec(_REGION_SPEC)
    product = spec.institutions[0].products[0]
    plan = build_product_doc_plan(product, target_docs=12)

    # every slot carries the diversity axes
    assert all(s["angle"] and s["audience"] for s in plan)

    # repeated genres must NOT share an angle (that's what drives distinctness)
    from collections import defaultdict
    angles_by_type = defaultdict(list)
    for s in plan:
        angles_by_type[s["document_type"]].append(s["angle"])
    for dtype, angles in angles_by_type.items():
        if len(angles) > 1:
            assert len(set(angles)) == len(angles), f"{dtype} has duplicate angles"

    # audiences vary across the set (not all the same reader)
    assert len({s["audience"] for s in plan}) > 1


def test_product_doc_plan_is_genre_diverse_at_scale():
    """At a large target the tail spreads across the expandable genres (faq /
    how_to_guide / troubleshooting / comparison) instead of becoming all-FAQ, and
    the new document types are represented."""
    from collections import Counter

    spec = load_region_spec(_REGION_SPEC)
    product = spec.institutions[0].products[0]
    plan = build_product_doc_plan(product, target_docs=60)
    counts = Counter(s["document_type"] for s in plan)
    # No single genre dominates (FAQ used to absorb the whole tail).
    top_share = max(counts.values()) / len(plan)
    assert top_share <= 0.30, f"a genre dominates ({top_share:.0%}): {dict(counts)}"
    # The tail is shared by several expandable genres, not FAQ alone.
    for g in ("faq", "how_to_guide", "troubleshooting", "comparison"):
        assert counts[g] >= 3, f"{g} underused: {dict(counts)}"
    # The added document types all appear.
    assert {"how_to_guide", "troubleshooting", "comparison", "rate_sheet",
            "security_advisory", "terms_conditions"} <= set(counts)


def test_brief_columns_are_structured():
    col = build_brief_columns()[0]
    assert col.column_type == "llm-structured"
    assert col.model_alias == DOC_GEN_MODEL_ALIAS
    schema = col.output_format  # DD normalizes to a JSON schema dict
    assert isinstance(schema, dict) and "ProductPositioning" in schema.get("$defs", {})


def test_docset_columns_include_structured_generation_and_judge_qc():
    cols = build_docset_columns()
    by_type = {c.column_type: c for c in cols}
    assert by_type["llm-structured"].model_alias == DOC_GEN_MODEL_ALIAS
    schema = by_type["llm-structured"].output_format
    assert isinstance(schema, dict) and "GeneratedDoc" in schema.get("$defs", {})
    assert "llm-judge" in by_type
    assert "expression" in by_type  # per-rubric score flatteners
    score_names = {s.name for s in docset_qc_scores()}
    assert {"NumericFaithfulness", "Cohesion", "Distinctness"} == score_names


def test_as_docset_normalizes_numpy_documents():
    # DD hands structured columns back through pandas -> nested lists can arrive as
    # numpy arrays; _as_docset must yield a plain list so truth-tests don't blow up.
    import numpy as np
    from usersim.asset_gen.financial_services.pipeline import _as_docset

    raw = {"documents": np.array(
        [{"doc_key": "a", "title": "A", "document_type": "faq", "body": "x"}],
        dtype=object)}
    out = _as_docset(raw)
    assert isinstance(out["documents"], list) and len(out["documents"]) == 1
    assert not _as_docset({"documents": np.array([], dtype=object)})["documents"]
    # explode must run without an "ambiguous truth value" error
    docs = explode_docset(out, id_prefix="X", institution_id="i",
                          spec_by_key={}, default_domain="retail_banking")
    assert docs and docs[0]["title"] == "A"


def test_embedding_columns():
    emb = build_embedding_columns()[0]
    assert emb.model_alias == EMBEDDING_MODEL_ALIAS and emb.target_column == "body"


def test_explode_docset_maps_to_scoped_loader_docs():
    docset = DocSet(documents=[
        {"doc_key": "acct_sheet", "title": "Account Overview",
         "document_type": "product_sheet", "body": "No monthly fee.",
         "mentions_tools": [], "references": ["dispute_policy"]},
        {"doc_key": "dispute_policy", "title": "Dispute Protocol",
         "document_type": "policy_procedure", "body": "Call file_dispute.",
         "mentions_tools": ["file_dispute"], "references": []},
    ])
    spec_by_key = {
        "acct_sheet": {"domain": "retail_banking", "product_category": "checking"},
        # The PLAN declares which tool a policy doc documents (see
        # build_shared_doc_plan); the model's own claim is not trusted.
        "dispute_policy": {"domain": "retail_banking", "product_category": "procedure",
                           "mentions_tools": ["file_dispute"]},
    }
    docs = explode_docset(
        docset, id_prefix="NB", institution_id="northwind_bank",
        spec_by_key=spec_by_key, default_domain="retail_banking",
    )
    assert {d["id"] for d in docs} == {"NB-ACCT_SHEET", "NB-DISPUTE_POLICY"}
    assert all(d["institution_id"] == "northwind_bank" for d in docs)
    dispute = next(d for d in docs if d["id"] == "NB-DISPUTE_POLICY")
    assert dispute["mentions_tools"] == ["file_dispute"]
    assert dispute["domain"] == "retail_banking"


class TestPlanOwnedTitles:
    """The doc title is deterministic plan metadata, not model output.

    A title is also the cross-reference handle every sibling is told to cite
    verbatim, so it has to be knowable before generation. Leaving it to the model
    silently broke that in any non-English locale: the first hi_Deva_IN corpus had
    93%-Devanagari bodies but 44% English customer-facing titles, because the plan
    handed over an English title AND demanded it be reproduced exactly.
    """

    _SPEC_DIR = (
        _asset_gen_dir() / "financial_services" / "region_spec"
    )

    def _plan_titles(self, locale: str) -> set:
        from usersim.asset_gen.financial_services.pipeline import build_shared_doc_plan

        spec = load_region_spec(self._SPEC_DIR / f"{locale}.yaml")
        out = set()
        for inst in spec.institutions:
            for product in inst.products:
                out |= {s["topic"] for s in build_product_doc_plan(product, 40, spec)}
            out |= {s["topic"] for s in build_shared_doc_plan(inst, spec)}
        return out

    def test_title_comes_from_the_plan_not_the_model(self):
        docset = DocSet(documents=[{
            "doc_key": "savings_faq", "title": "Regular Savings Account — overview",
            "document_type": "faq", "body": "x", "mentions_tools": [],
            "references": [],
        }])
        spec_by_key = {"savings_faq": {
            "domain": "retail_banking", "product_category": "savings",
            "topic": "नियमित बचत खाता — सिंहावलोकन",
        }}
        docs = explode_docset(
            docset, id_prefix="CB", institution_id="chandrika_bank",
            spec_by_key=spec_by_key, default_domain="retail_banking",
        )
        assert docs[0]["title"] == "नियमित बचत खाता — सिंहावलोकन"

    def test_off_plan_doc_keeps_the_models_title(self):
        """An invented doc_key has no plan slot, so there is no topic to prefer;
        dropping its title would lose the document entirely."""
        docset = DocSet(documents=[{
            "doc_key": "invented", "title": "Something The Model Made Up",
            "document_type": "faq", "body": "x", "mentions_tools": [],
            "references": [],
        }])
        docs = explode_docset(
            docset, id_prefix="CB", institution_id="chandrika_bank",
            spec_by_key={"other": {"topic": "T"}}, default_domain="retail_banking",
        )
        assert docs[0]["title"] == "Something The Model Made Up"

    @pytest.mark.parametrize("locale", ["en_US", "en_IN"])
    def test_shipped_english_titles_are_unchanged(self, locale: str):
        """LOCK: localizing the plan vocabulary must not touch the shipped banks.

        Both English locales omit ``doc_titles`` and author no
        ``display_name_local``, so every committed title must still be reproducible
        from the plan. This is what makes the change a provable no-op for en_US /
        en_IN rather than a silent corpus rewrite on their next regeneration.
        """
        import yaml

        planned = self._plan_titles(locale)
        root = (
            packaged_assets_dir() / "financial_services" / locale
        )
        titles = [
            d["title"]
            for corpus in root.rglob("corpus.yaml")
            for d in yaml.safe_load(corpus.read_text())["documents"]
        ]
        assert titles, f"no committed corpus for {locale}"
        assert [t for t in titles if t not in planned] == []

    def test_language_overlay_localizes_the_planned_titles(self):
        """The whole point: a language overlay's vocabulary reaches the title."""
        from usersim.engine.core.locale import expected_script_ranges

        ranges = expected_script_ranges("hi_Deva_IN")

        def devanagari_fraction(text: str) -> float:
            letters = [c for c in text if c.isalpha()]
            if not letters:
                return 1.0
            return sum(
                1 for c in letters
                if any(lo <= ord(c) <= hi for lo, hi in ranges)
            ) / len(letters)

        titles = self._plan_titles("hi_Deva_IN")
        # The in-language product name reaches the title (display_name_local), and
        # the angle is Hindi rather than the English default.
        assert "नियमित बचत खाता — सिंहावलोकन" in titles
        assert not any(t.startswith("Regular Savings Account") for t in titles)
        # Only the tool-reference titles may be Latin-dominated: a tool name is an
        # identifier that stays verbatim in every language, and those titles are
        # short, so the identifier legitimately outweighs the Hindi wrapper words.
        latin = {t for t in titles if devanagari_fraction(t) < 0.25}
        assert all("टूल संदर्भ" in t or "का उपयोग कब करें" in t for t in latin)

    def test_hindi_labels_cover_every_variable_and_domain(self):
        """An unmapped name falls back to the machine field name, which lands a raw
        ``below_mab_charge`` / ``retail_banking`` in reader-facing prose. Adding a
        product to the region model must not silently reintroduce that."""
        spec = load_region_spec(self._SPEC_DIR / "hi_Deva_IN.yaml")
        variables = {
            v for inst in spec.institutions for p in inst.products for v in p.variables
        }
        domains = {str(d) for inst in spec.institutions for d in inst.domains}
        assert variables - set(spec.doc_titles.variable_labels) == set()
        assert domains - set(spec.doc_titles.domain_labels) == set()

    def test_omitted_vocabulary_falls_back_to_the_english_defaults(self):
        from usersim.asset_gen.financial_services.pipeline import (
            _FAQ_INTENTS,
            build_product_doc_plan as plan,
        )

        spec = load_region_spec(_REGION_SPEC)
        product = spec.institutions[0].products[0]
        assert plan(product, 15, spec) == plan(product, 15, None)
        assert any(_FAQ_INTENTS[0] in s["angle"] for s in plan(product, 40, None))

    @pytest.mark.parametrize("field,value", [
        ("faq_variable_template", "no slot here"),
        ("regulatory_note_topic", "नियामक नोट्स"),
        ("tool_policy_topic", "कब उपयोग करें"),
    ])
    def test_template_missing_its_placeholder_is_rejected(self, field, value):
        """A template that loses its slot collapses every doc of that genre onto one
        title, which then de-duplicates down to a single document."""
        from pydantic import ValidationError

        from usersim.asset_gen.financial_services.spec import DocTitleVocabulary

        with pytest.raises(ValidationError):
            DocTitleVocabulary(**{field: value})


def test_mentions_tools_comes_from_the_plan_not_the_model():
    """The discoverable-tool gate must not be openable by prose.

    ``mentions_tools`` drives the sim-time gate (a tool is callable only once a doc
    naming it has been retrieved) and is a field on ``GeneratedDoc``, so the model
    could declare it anywhere. It did: 7.8% of en_IN customer-facing docs claimed
    tools they merely mentioned, so retrieving a generic how-to unlocked tools whose
    documentation was never read. The plan is now authoritative.
    """
    docset = DocSet(documents=[
        # A customer-facing doc claiming two tools it has no business unlocking.
        {"doc_key": "acct_howto", "title": "How to activate",
         "document_type": "how_to_guide",
         "body": "The agent will use activate_card to switch it on.",
         "mentions_tools": ["activate_card", "get_accounts"], "references": []},
        # A tool doc whose plan slot names its tool, but which forgot to say so.
        {"doc_key": "activate_card_tooldoc", "title": "activate_card tool reference",
         "document_type": "discoverable_tool_doc", "body": "Signature ...",
         "mentions_tools": [], "references": []},
    ])
    spec_by_key = {
        "acct_howto": {"domain": "retail_banking", "product_category": "checking"},
        "activate_card_tooldoc": {"domain": "retail_banking",
                                  "product_category": "procedure",
                                  "mentions_tools": ["activate_card"]},
    }
    docs = explode_docset(docset, id_prefix="NB", institution_id="northwind_bank",
                          spec_by_key=spec_by_key, default_domain="retail_banking")
    by_key = {d["id"]: d for d in docs}
    # The customer-facing doc unlocks NOTHING, whatever it claimed...
    assert by_key["NB-ACCT_HOWTO"]["mentions_tools"] == []
    # ...and the tool doc unlocks its tool even though the model omitted it.
    assert by_key["NB-ACTIVATE_CARD_TOOLDOC"]["mentions_tools"] == ["activate_card"]


def test_off_plan_doc_key_unlocks_nothing():
    """A drifted doc_key resolves to no slot, so it must fail CLOSED."""
    docset = DocSet(documents=[
        {"doc_key": "hallucinated_key", "title": "T",
         "document_type": "discoverable_tool_doc", "body": "b",
         "mentions_tools": ["close_account"], "references": []},
    ])
    docs = explode_docset(docset, id_prefix="NB", institution_id="northwind_bank",
                          spec_by_key={}, default_domain="retail_banking")
    assert docs[0]["mentions_tools"] == []


def test_references_are_a_prompt_sink_and_never_reach_the_corpus():
    """``GeneratedDoc.references`` is deliberately not corpus data.

    Its job is to give the model somewhere to park a sibling's slug so the slug does
    not end up in the prose -- the prompt asks for the cross-reference by TITLE in the
    body and the key here. Persisting it would mint a link graph nothing reads, out of
    keys the model is known to drift on (see ``test_off_plan_doc_key_unlocks_nothing``),
    so a reference to a hallucinated key would dangle. If a consumer ever needs the
    graph, resolve it through ``doc_uuid`` against the institution's real doc_keys
    rather than trusting these strings.
    """
    docset = DocSet(documents=[
        {"doc_key": "acct_sheet", "title": "T", "document_type": "product_sheet",
         "body": "b", "mentions_tools": [],
         "references": ["acct_faq", "a_key_that_does_not_exist"]},
    ])
    docs = explode_docset(
        docset, id_prefix="NB", institution_id="northwind_bank",
        spec_by_key={"acct_sheet": {"domain": "retail_banking",
                                    "product_category": "checking"}},
        default_domain="retail_banking",
    )
    assert "references" not in docs[0]


def test_the_prompt_still_offers_the_slug_sink():
    """The corpus stays free of raw doc_keys only while the prompt gives the model an
    alternative home for them, so the sink and the by-title rule travel together."""
    from usersim.asset_gen.financial_services.pipeline import GeneratedDoc
    from usersim.asset_gen.financial_services.prompts import DOCSET_PROMPT

    assert "references" in GeneratedDoc.model_fields
    assert "`references`" in DOCSET_PROMPT


def test_doc_uuid_is_deterministic_and_scoped():
    from usersim.asset_gen.financial_services.pipeline import doc_uuid
    import uuid as _uuid
    a = doc_uuid("en_US", "northwind_bank", "acct_sheet")
    assert a == doc_uuid("en_US", "northwind_bank", "acct_sheet")  # stable across calls
    assert _uuid.UUID(a)  # valid uuid
    # distinct across doc_key / institution / locale
    assert a != doc_uuid("en_US", "northwind_bank", "fee_schedule")
    assert a != doc_uuid("en_US", "solstice_invest", "acct_sheet")
    assert a != doc_uuid("ja_JP", "northwind_bank", "acct_sheet")


def test_explode_docset_uses_plan_uuid_for_id():
    from usersim.asset_gen.financial_services.pipeline import doc_uuid
    uid = doc_uuid("en_US", "northwind_bank", "acct_sheet")
    docset = DocSet(documents=[
        {"doc_key": "acct_sheet", "title": "T", "document_type": "product_sheet",
         "body": "b", "mentions_tools": [], "references": []},
    ])
    spec_by_key = {"acct_sheet": {"domain": "retail_banking",
                                  "product_category": "checking", "uuid": uid}}
    docs = explode_docset(docset, id_prefix="NB", institution_id="northwind_bank",
                          spec_by_key=spec_by_key, default_domain="retail_banking")
    assert docs[0]["id"] == uid  # id comes from the PLAN uuid, not the model's doc_key


def test_serialize_bank_round_trips_through_loader(tmp_path: Path):
    from usersim.engine.core.finance_bank import load_finance_bank

    region_meta = {
        "locale": "en_TT", "bank_id": "en_TT_test", "bank_version": "v0.0.1",
        "task_contract_version": "v0.1.0",
        "domain_regulators": {"retail_banking": {"regulator_text": "test"}},
    }
    inst = InstitutionBank(
        institution_meta={
            "institution_id": "testbank", "display_name": "Test Bank",
            "type": "bank", "brand_voice": "traditional",
            "domains": ["retail_banking"], "placeholder": True,
        },
        documents=[{
            "id": "D1", "title": "t", "body": "b", "domain": "retail_banking",
            "product_category": "checking", "document_type": "faq",
            "source_authority": "official", "placeholder": True, "mentions_tools": [],
        }],
        tools=[{
            "name": "kb_search", "description": "s", "discoverable": False,
            "side_effect_class": "read_only", "parameters": {},
        }],
        templates=[{
            "id": "T1", "tier": "dynamic", "task_type": "advisory_qa",
            "domain": "retail_banking", "placeholder": True,
            "persona_tags": ["region:any"], "opening_user_message": "hi",
            "account_state": {}, "param_space": {}, "gold_document_ids": [],
            "gold_tool_sequence": [], "expected_state_deltas": {},
        }],
    )
    serialize_bank(tmp_path / "en_TT", region_meta, [inst])

    bank = load_finance_bank(tmp_path / "en_TT")
    got = bank.institution("testbank")
    assert got is not None and got.doc_by_id("D1") is not None
    # No embeddings supplied -> no sidecar written.
    assert not (tmp_path / "en_TT" / "testbank" / "embeddings.parquet").exists()
    assert got.template_by_id("T1").tier == "dynamic"


def test_serialize_bank_writes_embedding_sidecar(tmp_path: Path):
    import pandas as pd

    region_meta = {"locale": "en_TT", "domain_regulators": {}}
    inst = InstitutionBank(
        institution_meta={"institution_id": "testbank", "type": "bank"},
        documents=[{"id": "D1", "title": "t", "body": "b", "domain": "retail_banking",
                    "product_category": "general", "document_type": "faq",
                    "source_authority": "official", "placeholder": True,
                    "mentions_tools": []}],
        embeddings={"D1": [0.1, 0.2, 0.3]},
    )
    serialize_bank(tmp_path / "en_TT", region_meta, [inst], write_tasks=False)

    # institutions are nested under their type dir (<type>/<id>/)
    sidecar = tmp_path / "en_TT" / "bank" / "testbank" / "embeddings.parquet"
    assert sidecar.exists()
    df = pd.read_parquet(sidecar)
    assert list(df["id"]) == ["D1"] and list(df.iloc[0]["embedding"]) == [0.1, 0.2, 0.3]


def test_cli_gen_assets_dry_run():
    from usersim.cli import main
    assert main(["gen-assets", "--domain", "financial_services", "--locale", "en_US",
                 "--dry-run", "--docs-per-product", "20", "--max-docs-per-cluster", "8"]) == 0


def test_generate_corpus_validates_endpoints_before_network():
    """generate_corpus fails fast (offline) if the gen model aliases are absent."""
    from usersim.asset_gen.financial_services.pipeline import generate_corpus
    from usersim.cli._models import ConfigError, ModelSpec, ModelsConfig

    cfg = ModelsConfig(
        providers=(),
        models=(ModelSpec(alias="judge_model", model="m", provider="nvidia"),),
    )
    spec = load_region_spec(_REGION_SPEC)
    with pytest.raises(ConfigError):
        generate_corpus(spec, cfg, "/tmp/unused")


def test_default_models_config_declares_gen_aliases():
    """The shipped models config declares the gen/embedding aliases + provider."""
    from usersim.asset_gen.financial_services.pipeline import validate_generation_models
    from usersim.cli._models import default_models_path, load_models_config
    cfg = load_models_config(default_models_path())
    validate_generation_models(cfg)  # must not raise
    assert any(p.name == "nvidia_inference_hub" for p in cfg.providers)
    # doc-gen model id must match the catalog entry (hub vendor prefix, lowercase).
    assert cfg.get("doc_gen_model").model == "nvidia/google/gemma-4-31b-it"


def test_off_plan_doc_key_is_reported_not_silent(caplog):
    """A localized doc_key loses product scope, and must say so.

    The doc-gen prompt tells the model to echo `doc_key` VERBATIM. When it does
    not — a real risk on a non-English run, where the model may translate the
    slug — the doc joins no plan slot, falls back to product_category="general",
    and then surfaces much later as a numeric-faithfulness failure ("typed values
    absent from prose") that points nowhere near the actual cause.
    """
    import logging

    from usersim.asset_gen.financial_services.pipeline import explode_docset

    plan = {"savings_product_sheet": {
        "uuid": "u1", "domain": "retail_banking",
        "product_category": "savings_regular"}}

    def _doc(key):
        return {"documents": [{"doc_key": key, "title": "T", "body": "B",
                               "document_type": "product_sheet",
                               "mentions_tools": []}]}

    # Happy path: the plan slot supplies scope and the deterministic uuid.
    ok = explode_docset(_doc("savings_product_sheet"), id_prefix="CB",
                        institution_id="cb", spec_by_key=plan,
                        default_domain="retail_banking")
    assert ok[0]["product_category"] == "savings_regular"
    assert ok[0]["id"] == "u1"

    with caplog.at_level(logging.WARNING):
        bad = explode_docset(_doc("नियमित_बचत_खाता"), id_prefix="CB",
                             institution_id="cb", spec_by_key=plan,
                             default_domain="retail_banking")
    assert bad[0]["product_category"] == "general"      # scope genuinely lost
    assert "not in the plan" in caplog.text
    assert "savings_product_sheet" in caplog.text       # names the expected key
    assert "नियमित_बचत_खाता" in caplog.text                # ...and what it got
