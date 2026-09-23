# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for the financial_services probe (multi-institution / multi-domain).

Covers: registration, multi-institution bank load + scope-integrity, deterministic
instantiation + gold derivation, scoped retrieval (no cross-institution leakage),
the discoverable-tool gate through run_dispatch (mocked assistant LLM), and the
deterministic verifier scorer (good / unauthorized / skip / Tier-2).
"""

from __future__ import annotations

import dataclasses
import json
import re
import textwrap
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from usersim.engine.core._assets import packaged_assets_dir
from usersim.engine.core.finance_bank import (
    FinanceBankError,
    load_finance_bank,
    load_finance_bank_for_locale,
)
from usersim.engine.core.finance_tasks import build_instance
from usersim.engine.core.outcomes import (
    OutcomeBuilder,
    Provenance,
    TraceKind,
    WarningKind,
)
from usersim.engine.core.probes import known_probes
from usersim.engine.core.probing_taxonomy import (
    load_probing_taxonomy_for,
    reset_probing_taxonomy_cache,
)
from usersim.engine.evaluator.scorers import financial_services as FS
from usersim.engine.evaluator.scorers import get_scorer, load_default_scorers
from usersim.engine.probes.financial_services import generator as G
from usersim.engine.probes.financial_services import retrieval as R
from usersim.engine.probes.financial_services import tools as T
from usersim.engine.probes.financial_services.task_derivation import (
    derive_instance,
)

_PERSONA = {
    "first_name": "Yumi",
    "last_name": "Tanaka",
    "age": 35,
    "region": "Oregon",
    "occupation": "designer",
}


def _load_finance_taxonomy(locale="en_US"):
    """The financial_services dynamic-tier taxonomy (shared probing loader)."""
    return load_probing_taxonomy_for(
        "financial_services",
        locale,
        filename="dynamic.yaml",
        env_prefix="USERSIM_FINANCIAL_SERVICES_DYNAMIC_TAXONOMY",
    )


@pytest.fixture(autouse=True)
def _fresh_taxonomy():
    """Reset the taxonomy cache, which is per-test state because the loader
    reads an environment variable for its source.

    The finance bank cache is deliberately NOT reset here. It is keyed by
    locale and ``FinanceBank`` is a frozen dataclass, so a cached bank cannot
    leak between tests; clearing it just reloads the same immutable assets
    from disk, once per test, which was most of this file's runtime. Tests
    that load a bank from a path of their own call ``load_finance_bank``
    directly and never touch the cache.
    """
    reset_probing_taxonomy_cache()
    yield
    reset_probing_taxonomy_cache()


@pytest.fixture(autouse=True)
def _ensure_scorers_loaded():
    """Re-register bundled scorers before each test.

    Evaluator scorer tests (collected before ``probes/``) clear the scorer
    registry in teardown; the package is already imported so the eager
    ``load_default_scorers()`` on import will not re-fire. This idempotent
    reload keeps ``get_scorer("financial_services")`` robust to test order.
    """
    load_default_scorers()
    yield


def _cfg(**kw):
    base = dict(max_turns=6, random_seed=None, finance_retrieval_mode="golden", verbosity=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _northwind_dispute_instance():
    bank = load_finance_bank_for_locale("en_US")
    inst = bank.institution("northwind_bank")
    tpl = inst.template_by_id("TPL-NB-DISPUTE-001")
    return build_instance(tpl, inst, persona_uuid="u1", locale="en_US", seed=1)


def test_probe_registered():
    assert "financial_services" in known_probes()


def test_kb_search_offered_with_query_parameter():
    """Regression: kb_search must be offered with a real ``query`` parameter.

    The generated banks ship kb_search with an empty schema (no description /
    no parameters); the probe must override it with its canonical schema so the
    assistant supplies a query instead of calling kb_search({}) — an empty query
    retrieves nothing, so the discoverable gold tools never unlock and every
    verifiable task scores 0. See run=1784772822.
    """
    bank = load_finance_bank_for_locale("en_US")
    inst = bank.institution("northwind_bank")
    by_name = {t["function"]["name"]: t["function"] for t in T.build_offered_tools(inst)}
    assert "kb_search" in by_name
    kb = by_name["kb_search"]
    assert kb["description"].strip(), "kb_search must have a description"
    props = kb["parameters"]["properties"]
    assert "query" in props and props["query"]["type"] == "string"
    assert kb["parameters"].get("required") == ["query"]
    # The discoverable-tool gate is still offered alongside the primitives.
    assert T.CALL_TOOL_NAME in by_name


# Locales that ship a searchable corpus but no ``tasks.yaml`` and no
# locale-level ``dynamic.yaml``, so the probe cannot instantiate a scenario on
# them and aborts at runtime. Authoring those assets must move the locale out of
# here and into the runnable set -- and delete this tuple's test once it empties.
_CORPUS_ONLY_LOCALES: tuple[str, ...] = ("hi_Deva_IN",)


def _partition_committed_banks() -> tuple[list[Path], list[Path]]:
    """Split committed locale banks into ``(runnable, corpus_only)``.

    A locale is runnable once at least one institution carries a ``tasks.yaml``.
    Presence of ``region_meta.yaml`` alone is not enough: it marks a committed
    bank, not a bank with scenarios to run.
    """
    banks_root = packaged_assets_dir() / "financial_services"
    locales = [p for p in sorted(banks_root.iterdir()) if (p / "region_meta.yaml").exists()]
    assert locales, f"no committed banks under {banks_root}"
    runnable = [p for p in locales if any(p.glob("*/*/tasks.yaml"))]
    corpus_only = [p for p in locales if not any(p.glob("*/*/tasks.yaml"))]
    return runnable, corpus_only


def test_all_committed_banks_load():
    """Guard: every RUNNABLE committed locale bank must load. A regenerate+freeze
    can leave curated tasks.yaml gold ids pointing at stale corpus ids
    (tasks/corpus drift); this turns that silent runtime break into a loud CI
    failure at the tasks<->corpus seam, for every locale, not just en_US.

    Corpus-only locales are covered by the test below instead: with no
    tasks.yaml they cannot exhibit tasks<->corpus drift, which is the failure
    mode this guard exists to catch.
    """
    runnable, _ = _partition_committed_banks()
    assert runnable, "no runnable committed banks found"
    for locale_dir in runnable:
        load_finance_bank(locale_dir)  # raises FinanceBankError on any gold/scope break


def test_corpus_only_banks_are_declared_and_still_unrunnable():
    """Pin the corpus-only locales so the exemption cannot rot.

    Set equality (rather than a skip) is what makes this self-retiring:
    authoring a tasks.yaml for a corpus-only locale fails this test and forces
    the author to promote it into the runnable set, and adding a new
    corpus-only locale forces a deliberate decision to record it.
    """
    _, corpus_only = _partition_committed_banks()
    assert {p.name for p in corpus_only} == set(_CORPUS_ONLY_LOCALES), (
        "the corpus-only locale set changed -- either author tasks.yaml and let "
        "the locale join the runnable set above, or record the new corpus-only "
        "locale in _CORPUS_ONLY_LOCALES"
    )
    for locale_dir in corpus_only:
        with pytest.raises(FinanceBankError, match=r"tasks\.yaml"):
            load_finance_bank(locale_dir)


def test_bank_loads_four_institutions():
    bank = load_finance_bank_for_locale("en_US")
    ids = {i.institution_id for i in bank.institutions}
    assert {"northwind_bank", "solstice_invest", "evergreen_retirement", "meridian_wealth"} <= ids
    types_ = {i.type for i in bank.institutions}
    assert {"bank", "brokerage", "retirement_provider", "wealth_manager"} <= types_
    # every offered domain has regulator context
    for dom in bank.all_domains():
        assert bank.regulator_text_for_domain(dom)


def test_instantiation_is_deterministic_and_scoped():
    a = _northwind_dispute_instance()
    b = _northwind_dispute_instance()
    assert a.institution_id == "northwind_bank" and a.domain == "retail_banking"
    assert a.is_verifiable
    assert a.opening_user_message == b.opening_user_message
    assert a.gold.gold_tool_sequence == ("file_transaction_dispute",)


def test_retrieval_never_crosses_institutions():
    bank = load_finance_bank_for_locale("en_US")
    for inst in bank.institutions:
        docs = R.retrieve(inst, "account dispute transfer rollover fee risk", k=10, mode="dense")
        assert docs, inst.institution_id
        assert all(d.institution_id == inst.institution_id for d in docs)


def test_dense_retrieval_ranks_by_cosine():
    """dense mode ranks the institution's precomputed vectors against the query
    vector by cosine; the doc whose vector points the same way ranks first."""
    from usersim.engine.core.finance_bank import Document, Institution

    docs = (
        Document(
            id="D_near",
            title="near",
            body="b",
            product_category="p",
            document_type="faq",
            source_authority="official",
            placeholder=True,
        ),
        Document(
            id="D_far",
            title="far",
            body="b",
            product_category="p",
            document_type="faq",
            source_authority="official",
            placeholder=True,
        ),
    )
    inst = Institution(
        institution_id="acme",
        display_name="Acme",
        type="bank",
        brand_voice="traditional",
        domains=("retail_banking",),
        documents=docs,
        tools=(),
        templates=(),
        embeddings={"D_near": (1.0, 0.0), "D_far": (0.0, 1.0)},
    )
    ranked = R.retrieve(inst, "q", k=2, mode="dense", query_embedding=(0.9, 0.1))
    assert [d.id for d in ranked] == ["D_near", "D_far"]


def test_dense_retrieval_falls_back_to_lexical_without_query_vector():
    """No query embedding (offline / no endpoint) -> deterministic lexical rank,
    so dense mode never hard-fails a run."""
    from usersim.engine.core.finance_bank import Document, Institution

    docs = (
        Document(
            id="D1",
            title="wire transfer limits",
            body="daily wire cap",
            product_category="p",
            document_type="faq",
            source_authority="official",
            placeholder=True,
        ),
        Document(
            id="D2",
            title="branch hours",
            body="lobby opening times",
            product_category="p",
            document_type="faq",
            source_authority="official",
            placeholder=True,
        ),
    )
    inst = Institution(
        institution_id="acme",
        display_name="Acme",
        type="bank",
        brand_voice="traditional",
        domains=("retail_banking",),
        documents=docs,
        tools=(),
        templates=(),
        embeddings={"D1": (1.0, 0.0)},  # present but unused without a query vec
    )
    ranked = R.retrieve(inst, "wire transfer", k=1, mode="dense", query_embedding=None)
    assert [d.id for d in ranked] == ["D1"]  # lexical overlap winner


class TestHybridRetrieval:
    """Hybrid is the default because pure dense collapses on a homogeneous corpus.

    Measured over the 442 kb_search calls of one en_IN run: dense returned
    discoverable_tool_doc for 88.8% of results against an 11.3% corpus share, and
    its result sets barely varied with the query (pairwise Jaccard 0.41-0.57 vs
    0.06-0.08 for lexical). On a labelled subset, precision@8 was 4% for dense,
    17% for dense over title+body, and 88-92% for hybrid.
    """

    @staticmethod
    def _inst(embeddings):
        from usersim.engine.core.finance_bank import Document, Institution

        docs = (
            # Short, and its body repeats the account-action vocabulary that shows
            # up in every query -- the shape that hijacked dense ranking.
            Document(
                id="D_tool",
                title="get_accounts tool reference",
                body="The get_accounts tool lists accounts and balances.",
                product_category="p",
                document_type="discoverable_tool_doc",
                source_authority="official",
                placeholder=True,
            ),
            # Long and verbose, but it is the document that answers the question.
            Document(
                id="D_answer",
                title="Recurring Deposit - minimum monthly installment",
                body=(
                    "The minimum monthly installment for a recurring deposit "
                    "is 500 rupees. " + "Additional context. " * 40
                ),
                product_category="p",
                document_type="faq",
                source_authority="official",
                placeholder=True,
            ),
        )
        return Institution(
            institution_id="acme",
            display_name="Acme",
            type="bank",
            brand_voice="traditional",
            domains=("retail_banking",),
            documents=docs,
            tools=(),
            templates=(),
            embeddings=embeddings,
        )

    QUERY = "recurring deposit minimum monthly installment"

    def test_lexical_signal_rescues_a_doc_dense_ranks_last(self):
        """The regression in miniature: dense prefers the tool reference, hybrid
        surfaces the answering document first."""
        # Dense vectors deliberately encode the observed pathology: the tool doc is
        # 'closer' to the query than the document that actually answers it.
        emb = {"D_tool": (1.0, 0.0), "D_answer": (0.0, 1.0)}
        inst = self._inst(emb)
        qvec = (0.95, 0.31)

        dense_first = R.retrieve(inst, self.QUERY, k=2, mode="dense", query_embedding=qvec)[0]
        assert dense_first.id == "D_tool"

        hybrid_first = R.retrieve(inst, self.QUERY, k=2, mode="hybrid", query_embedding=qvec)[0]
        assert hybrid_first.id == "D_answer"

    def test_hybrid_degrades_to_lexical_without_a_query_vector(self):
        inst = self._inst({"D_tool": (1.0, 0.0), "D_answer": (0.0, 1.0)})
        ranked = R.retrieve(inst, self.QUERY, k=2, mode="hybrid", query_embedding=None)
        assert ranked[0].id == "D_answer"

    def test_hybrid_degrades_to_lexical_without_a_sidecar(self):
        inst = self._inst({})
        ranked = R.retrieve(inst, self.QUERY, k=2, mode="hybrid", query_embedding=(0.95, 0.31))
        assert ranked[0].id == "D_answer"

    def test_dense_still_contributes_when_lexis_cannot_discriminate(self):
        """Hybrid must not become lexical-only: with zero token overlap the
        lexical term is degenerate and dense has to decide the order."""
        inst = self._inst({"D_tool": (1.0, 0.0), "D_answer": (0.0, 1.0)})
        ranked = R.retrieve(inst, "zzz qqq", k=2, mode="hybrid", query_embedding=(0.1, 0.99))
        assert ranked[0].id == "D_answer"  # decided by cosine, not tokens

    def test_ranking_is_deterministic(self):
        inst = self._inst({"D_tool": (1.0, 0.0), "D_answer": (0.0, 1.0)})
        runs = [
            [d.id for d in R.retrieve(inst, self.QUERY, k=2, mode="hybrid", query_embedding=(0.95, 0.31))]
            for _ in range(5)
        ]
        assert all(r == runs[0] for r in runs)

    def test_hybrid_is_scoped_to_one_institution(self):
        bank = load_finance_bank_for_locale("en_IN")
        for inst in bank.institutions:
            docs = R.retrieve(inst, "recurring deposit penalty fee", k=10, mode="hybrid", query_embedding=[0.1] * 1024)
            assert docs, inst.institution_id
            assert all(d.institution_id == inst.institution_id for d in docs)

    def test_hybrid_is_the_shipped_default(self):
        from usersim.engine.config import ConversationSimulatorConfig

        field = ConversationSimulatorConfig.model_fields["finance_retrieval_mode"]
        assert field.default == "hybrid"

    def test_hybrid_survives_a_dense_signal_that_points_at_tool_docs(self):
        """End-to-end guard on the actual defect, against the real en_IN corpus.

        Runs on the real corpus text with a SYNTHETIC vector set, because no
        embeddings.parquet ships: vectors only compare within the space that
        produced them, so shipping ours would pin users to one model.

        The synthetic set reproduces the measured pathology by construction.
        Tool docs get vectors in a tight cluster, everything else is spread out,
        and the query is that cluster's centroid, so cosine alone ranks tool
        references top. That is exactly the input hybrid has to survive.
        Passing ``query_embedding=None`` would make this test vacuous, since
        hybrid would degrade to pure lexical.
        """
        import math
        import random
        from collections import Counter

        bank = load_finance_bank_for_locale("en_IN")
        inst = bank.institution("chandrika_bank")

        def _unit(v):
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            return tuple(x / n for x in v)

        # Deterministic in doc id, so a corpus regeneration cannot make this
        # flaky, and independent of any real embedding model. Every document
        # shares a large component on axis 0, which is the corpus homogeneity
        # the module docstring describes; tool docs lean on it slightly harder.
        # Calibrated so dense returns tool docs for ~88% of results, matching
        # the share measured on the real vectors.
        def _vec(doc):
            rng = random.Random(doc.id)
            shared = 4.6 if doc.document_type == "discoverable_tool_doc" else 3.4
            return _unit([shared] + [rng.gauss(0, 1.0) for _ in range(15)])

        embeddings = {d.id: _vec(d) for d in inst.documents}
        inst = dataclasses.replace(inst, embeddings=embeddings)

        tool_vecs = [embeddings[d.id] for d in inst.documents if d.document_type == "discoverable_tool_doc"]
        assert tool_vecs, "fixture needs tool docs to cluster"
        centroid = [sum(col) / len(tool_vecs) for col in zip(*tool_vecs)]

        corpus_share = sum(1 for d in inst.documents if d.document_type == "discoverable_tool_doc") / len(
            inst.documents
        )
        queries = [
            "recurring deposit skip installment penalty auto debit",
            "savings account monthly average balance charge waiver",
            "fixed deposit premature withdrawal penalty interest",
        ]

        def share_for(mode):
            share = Counter()
            for q in queries:
                for d in R.retrieve(inst, q, k=8, mode=mode, query_embedding=centroid):
                    share[d.document_type] += 1
            total = sum(share.values())
            return share["discoverable_tool_doc"] / total, share

        dense_share, _ = share_for("dense")
        hybrid_share, mix = share_for("hybrid")

        # The adversarial vector really does break cosine-only ranking...
        assert dense_share > 0.75, dense_share
        # ...and the lexical half pulls the mix back down. Asserted RELATIVE to dense
        # rather than as a multiple of corpus_share: the share of tool docs in a corpus
        # drifts with every regeneration (11.3% -> 7.8% when en_IN grew), which silently
        # tightens an absolute bound and fails this test for a composition change rather
        # than a behaviour change. The property that matters is that hybrid cuts the
        # collapse by a wide margin under the most hostile query vector available.
        assert hybrid_share <= dense_share / 3, (
            f"tool-doc share: hybrid {hybrid_share:.1%} vs dense {dense_share:.1%} (corpus {corpus_share:.1%}): {mix}"
        )


async def test_embed_query_soft_fails_to_none():
    from usersim.engine.core.embeddings import aembed_query

    class _Facade:
        async def agenerate_text_embeddings(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    # Missing alias -> None (caller falls back to lexical).
    assert await aembed_query({}, "embedding_model", "hello") is None
    # Present facade -> the vector.
    assert await aembed_query({"embedding_model": _Facade()}, "embedding_model", "hi") == [
        0.1,
        0.2,
        0.3,
    ]

    class _Boom:
        async def agenerate_text_embeddings(self, texts):
            raise RuntimeError("endpoint down")

    # Endpoint error -> None (never raises).
    assert await aembed_query({"embedding_model": _Boom()}, "embedding_model", "hi") is None


def test_shipped_banks_carry_no_vectors_and_still_load():
    """No locale ships an embeddings.parquet, and the banks work without one.

    Vectors are only comparable within the embedding space that produced them,
    so shipping ours would pin every user to one model. Users generate their
    own via ``gen-assets``. The loader treats an absent sidecar as an empty map
    and retrieval stays on the lexical path.
    """
    assets = packaged_assets_dir() / "financial_services"
    assert not list(assets.rglob("embeddings.parquet"))

    bank = load_finance_bank_for_locale("en_US")
    inst = bank.institution("northwind_bank")
    assert inst.embeddings == {}
    assert inst.documents, "corpus still loads without vectors"
    # Retrieval with no query vector is the shipped path, and it ranks.
    hits = R.retrieve(inst, "dispute a card charge", k=5, mode="hybrid", query_embedding=None)
    assert hits and {d.id for d in hits} <= {d.id for d in inst.documents}


_MODELS = {
    "user_model": object(),
    "assistant_model": object(),
    "judge_model": object(),
    "summary_model": object(),
    "api_response_model": object(),
}


def _mock_call_llm(assistant_responses, *, user_responses=None):
    """Alias-dispatching ``call_llm`` mock for the shared ConversationLoop.

    The probe now rides ``ConversationLoop`` (not a custom run_dispatch), so
    the assistant is called by the loop (first tool-call of a turn) AND by the
    probe's ``after_assistant_turn`` (re-calls after tool results) — both under
    the ``assistant_model`` alias and drawing from the same iterator. Turn-1 is
    verbatim (no ``user_model`` call); ``user_responses`` covers follow-ups.
    """
    it_assist = iter(assistant_responses)
    it_user = iter(user_responses or [])
    judge_payload = "<explanation>ok</explanation>\n<rating>success</rating>"

    def _side_effect(models, alias, msgs, **kwargs):
        if alias == "assistant_model":
            try:
                return next(it_assist)
            except StopIteration as e:
                raise AssertionError("test consumed more assistant_model calls than expected") from e
        if alias == "judge_model":
            return {"role": "assistant", "content": judge_payload}
        if alias == "user_model":
            try:
                return next(it_user)
            except StopIteration as e:
                raise AssertionError("test consumed more user_model calls than expected") from e
        if alias == "summary_model":
            return {"role": "assistant", "content": "no"}
        raise AssertionError(f"unexpected alias: {alias!r}")

    return _side_effect


@contextmanager
def _patched_call_llm(side_effect):
    """Patch ``call_llm`` everywhere the loop + probe bind it.

    ``simulation`` (loop assistant + turn-1 gen + completion check),
    ``judges`` (in-sim judge) and ``context`` (response compression) each
    bind ``call_llm`` at module load, so patching the source module does not
    reach them; the ``ToolExecutionMixin`` inner-loop re-calls import it
    lazily from ``core.llm``. Every one of these references must be patched,
    or the real ``call_llm`` runs against this module's placeholder facades
    and the probe's fallback quietly absorbs the resulting error.
    """
    with (
        patch("usersim.engine.core.simulation.acall_llm", side_effect=side_effect),
        patch("usersim.engine.core.judges.acall_llm", side_effect=side_effect),
        patch("usersim.engine.core.context.acall_llm", side_effect=side_effect),
        patch("usersim.engine.core.llm.acall_llm", side_effect=side_effect),
    ):
        yield


async def _run_probe(side_effect, instance, **cfg_kw):
    prov = Provenance()
    ob = OutcomeBuilder(provenance=prov)
    cfg = _cfg(**cfg_kw)
    with patch.object(G, "derive_instance", return_value=instance), _patched_call_llm(side_effect):
        probe = G.FinancialServicesProbe(
            persona=_PERSONA,
            locale="en_US",
            language="English",
            models=_MODELS,
            cfg=cfg,
            provenance=prov,
            profile={},
            data={"persona_uuid": "u1"},
            outcome_builder=ob,
        )
        return await probe.run_dispatch(
            models=_MODELS,
            data={"persona_uuid": "u1"},
            cfg=cfg,
        )


def _dispute_tool_calls():
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "kb_search",
                        "arguments": json.dumps({"query": "dispute unauthorized charge"}),
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c2",
                    "function": {
                        "name": "call_tool",
                        "arguments": json.dumps(
                            {
                                "tool_name": "file_transaction_dispute",
                                "arguments": {
                                    "account_id": "acct_nb_0001",
                                    "transaction_id": "txn_nb_0001",
                                    "reason": "unauthorized",
                                },
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Filed a dispute; it's under review.", "tool_calls": None},
    ]


def _kb_doc(doc_id, body=None):
    return {
        "id": doc_id,
        "title": f"title {doc_id}",
        "body": body or f"body {doc_id}",
        "tools": [f"tool_{doc_id}"],
    }


def _kb_exchange(*documents, tool_name="kb_search"):
    call_id = f"call_{documents[0]['id']}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps({"results": list(documents)}),
            "tool_call_id": call_id,
        },
    ]


def _body_ids(message):
    payload = json.loads(message["content"])
    return {document["id"] for document in payload["results"] if "body" in document}


def test_document_view_keeps_all_unique_bodies_and_newest_duplicate():
    messages = [
        {"role": "user", "content": "question"},
        *_kb_exchange(_kb_doc("A"), _kb_doc("B")),
        *_kb_exchange(_kb_doc("A", "new A"), _kb_doc("C")),
        *_kb_exchange(_kb_doc("D"), _kb_doc("E")),
    ]
    before = repr(messages)

    projected = G._project_finance_document_history(
        messages,
        current_user_turn=1,
    )

    assert _body_ids(projected[2]) == {"B"}
    assert _body_ids(projected[4]) == {"A", "C"}
    assert _body_ids(projected[6]) == {"D", "E"}
    assert repr(messages) == before


def test_document_view_ages_on_user_turn_not_tool_round():
    messages = [
        {"role": "user", "content": "question"},
        *_kb_exchange(_kb_doc("A")),
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": '{"ok":true}', "tool_call_id": "other"},
    ]
    through_turn_eight = G._project_finance_document_history(
        messages,
        current_user_turn=8,
    )
    on_turn_nine = G._project_finance_document_history(
        messages,
        current_user_turn=9,
    )
    assert _body_ids(through_turn_eight[2]) == {"A"}
    assert _body_ids(on_turn_nine[2]) == set()


def test_retrieval_after_expiry_restores_newest_body():
    messages = [
        {"role": "user", "content": "turn 1"},
        *_kb_exchange(_kb_doc("A", "old body")),
    ]
    messages.extend({"role": "user", "content": f"turn {turn}"} for turn in range(2, 10))
    messages.extend(_kb_exchange(_kb_doc("A", "refreshed body")))

    projected = G._project_finance_document_history(
        messages,
        current_user_turn=9,
    )

    assert _body_ids(projected[2]) == set()
    assert _body_ids(projected[-1]) == {"A"}
    assert "refreshed body" in projected[-1]["content"]


def test_non_kb_tool_payload_is_unchanged():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "tool", "content": '{"results":[{"value":1}]}', "tool_call_id": "x"},
    ]
    assert (
        G._project_finance_document_history(
            messages,
            current_user_turn=1,
        )
        == messages
    )


def test_valid_document_shaped_non_kb_payload_is_unchanged():
    messages = [
        {"role": "user", "content": "question"},
        *_kb_exchange(_kb_doc("A"), tool_name="export_documents"),
    ]
    assert (
        G._project_finance_document_history(
            messages,
            current_user_turn=9,
        )
        == messages
    )


async def test_nine_turn_loop_ages_assistant_view_but_preserves_export():
    inst = _northwind_dispute_instance()
    assistant_inputs = []
    assistant_responses = iter(
        [
            _dispute_tool_calls()[0],
            {"role": "assistant", "content": "I found the policy.", "tool_calls": None},
            *[
                {
                    "role": "assistant",
                    "content": f"Visible response {turn}.",
                    "tool_calls": None,
                }
                for turn in range(2, 10)
            ],
        ]
    )
    user_turn = 1
    judge_payload = "<explanation>ok</explanation><rating>success</rating>"

    async def side_effect(models, alias, msgs, **kwargs):
        nonlocal user_turn
        if alias == "assistant_model":
            assistant_inputs.append(msgs)
            return next(assistant_responses)
        if alias == "user_model":
            user_turn += 1
            return {"role": "assistant", "content": f"Follow-up {user_turn}."}
        if alias == "judge_model":
            return {"role": "assistant", "content": judge_payload}
        if alias == "summary_model":
            return {"role": "assistant", "content": "no"}
        raise AssertionError(alias)

    result = await _run_probe(
        side_effect,
        inst,
        max_turns=9,
        context_compression=False,
    )

    # Turn 1 outer call, turn 1 inner synthesis, then outer calls for turns 2-9.
    assert len(assistant_inputs) == 10
    inner_tool = next(
        message
        for message in assistant_inputs[1]
        if message.get("role") == "tool" and G._kb_search_payload(message.get("content")) is not None
    )
    turn_nine_tool = next(
        message
        for message in assistant_inputs[-1]
        if message.get("role") == "tool" and json.loads(message["content"]).get("results")
    )
    assert _body_ids(inner_tool)
    assert _body_ids(turn_nine_tool) == set()
    assert all(
        set(document) == {"id", "title", "tools"} for document in json.loads(turn_nine_tool["content"])["results"]
    )

    exported = json.loads(result["conversation_messages"])
    exported_tool = next(
        message
        for message in exported
        if message.get("role") == "tool" and G._kb_search_payload(message.get("content")) is not None
    )
    assert _body_ids(exported_tool)
    metadata = json.loads(result["conversation_metadata"])
    assert set(metadata["gold_document_ids"]) <= set(metadata["retrieved_document_ids"])
    assert metadata["discovered_tools"]


async def test_inner_context_overflow_returns_partial_failed_finance_row():
    inst = _northwind_dispute_instance()
    calls = 0

    async def side_effect(models, alias, msgs, **kwargs):
        nonlocal calls
        if alias == "assistant_model":
            calls += 1
            if calls == 1:
                return _dispute_tool_calls()[0]
            from usersim.engine.core.llm import ContextWindowError

            raise ContextWindowError(
                "assistant_model",
                RuntimeError("maximum context length is 32768 tokens"),
            )
        raise AssertionError(f"unexpected alias: {alias!r}")

    result = await _run_probe(side_effect, inst, max_turns=1)
    outcome = json.loads(result["simulation_outcome"])
    messages = json.loads(result["conversation_messages"])

    assert result["conversation_status"] is False
    assert outcome["status"] == "failed"
    assert outcome["failure_class"] == "infrastructure_error"
    assert outcome["failure_attribution"] == "assistant_model"
    assert result["finance_task_id"] == inst.template_id
    assert any(message["role"] == "tool" for message in messages)


async def test_summary_failure_keeps_original_response_with_warning():
    inst = _northwind_dispute_instance()
    long_response = "A detailed response. " * 40

    async def side_effect(models, alias, msgs, **kwargs):
        if alias == "assistant_model":
            return {"role": "assistant", "content": long_response, "tool_calls": None}
        if alias == "summary_model":
            raise RuntimeError("summary endpoint unavailable")
        if alias == "judge_model":
            return {
                "role": "assistant",
                "content": ("<explanation>ok</explanation><rating>success</rating>"),
            }
        raise AssertionError(f"unexpected alias: {alias!r}")

    result = await _run_probe(side_effect, inst, max_turns=1, context_compression=True)
    outcome = json.loads(result["simulation_outcome"])
    messages = json.loads(result["conversation_messages"])

    assert messages[-1]["content"] == long_response
    assert any(warning["kind"] == WarningKind.COMPRESSION_FALLBACK.value for warning in outcome["warnings"])


async def test_empty_summary_keeps_original_response_with_warning():
    inst = _northwind_dispute_instance()
    long_response = "A detailed response. " * 40

    async def side_effect(models, alias, msgs, **kwargs):
        if alias == "assistant_model":
            return {"role": "assistant", "content": long_response, "tool_calls": None}
        if alias == "summary_model":
            return {"role": "assistant", "content": ""}
        if alias == "judge_model":
            return {
                "role": "assistant",
                "content": ("<explanation>ok</explanation><rating>success</rating>"),
            }
        raise AssertionError(f"unexpected alias: {alias!r}")

    result = await _run_probe(side_effect, inst, max_turns=1, context_compression=True)
    outcome = json.loads(result["simulation_outcome"])
    messages = json.loads(result["conversation_messages"])

    assert messages[-1]["content"] == long_response
    assert any(warning["kind"] == WarningKind.COMPRESSION_FALLBACK.value for warning in outcome["warnings"])


def test_read_tools_and_grounding_share_one_domain_aware_account():
    """Regression: a retirement/brokerage/wealth request must NOT surface a
    'checking' account. That type mismatch (plus a fabricated last-4) made the
    assistant refuse to act and loop on read tools. The read tools return the
    SAME canonical account the user is grounded to, typed by domain."""
    state = {"account_id": "acct_er_ira_0001", "balance": "0.00"}
    resolved = G._resolve_accounts(state, domain="retirement", seed="TPL-ER-CLOSEACCT-001")
    assert resolved[0]["account_type"] == "retirement account"
    assert resolved[0]["last4"].isdigit() and len(resolved[0]["last4"]) == 4

    # After seeding, the read tools echo exactly the grounded account.
    state["accounts"] = resolved
    assert G._apply_tool("get_retirement_accounts", {}, state)["accounts"] == resolved
    assert G._apply_tool("get_accounts", {}, state)["accounts"] == resolved

    # An explicit account_type / last4 on the task overrides the derived values.
    ex = G._resolve_accounts(
        {"account_id": "x", "account_type": "Roth IRA", "last4": "4821"},
        domain="retirement",
    )
    assert ex[0]["account_type"] == "Roth IRA" and ex[0]["last4"] == "4821"

    # Seeded last4 is stable per task but distinct across tasks (no more shared 0001).
    same = {"account_id": "acct_er_ira_0001"}
    a = G._resolve_accounts(same, domain="retirement", seed="TPL-A")[0]["last4"]
    a2 = G._resolve_accounts(same, domain="retirement", seed="TPL-A")[0]["last4"]
    b = G._resolve_accounts(same, domain="retirement", seed="TPL-B")[0]["last4"]
    assert a == a2 and a != b

    # New-account / rollover flows (no account_id) ground on nothing.
    assert G._resolve_accounts({"source_account_id": "x"}, domain="retirement") == []


def test_verify_identity_requires_name_and_dob():
    """Regression: identity verification must FAIL without both full name and
    date of birth (the old mock rubber-stamped any call, so a customer could skip
    DOB and still reach a state-changing action)."""
    ok, payload = G._verify_identity({"full_name": "Jose Tenenbaum", "account_or_card_last4": "0001"})
    assert ok is False and payload["status"] == "unverified"
    # The reason is read by the assistant LLM, so missing fields are named in
    # human form ("date of birth") rather than as raw keys.
    assert "date of birth" in payload["reason"]

    ok, payload = G._verify_identity({"full_name": "Jose", "date_of_birth": "1990-01-01"})
    assert ok is True and payload["status"] == "verified"

    ok, _ = G._verify_identity({"date_of_birth": "1990-01-01"})  # missing name
    assert ok is False

    # the advertised schema marks both as required so the agent knows to collect them
    from usersim.engine.core.finance_bank import framework_primitive_tool

    req = framework_primitive_tool("verify_identity")["parameters"].get("required")
    assert req == ["full_name", "date_of_birth"]


def test_card_actions_target_a_real_card():
    """Regression: card actions resolve to the card on file (there's no card-listing
    tool), so freeze/activate/replace hit a real card_id and apply the state delta
    instead of acting on a fabricated 'unknown' card."""
    st = {"cards": [{"card_id": "card_nb_0001", "status": "active"}]}
    out = G._apply_tool("freeze_card", {}, st)  # no card_id supplied
    assert out["card_id"] == "card_nb_0001" and st["cards"][0]["status"] == "frozen"

    st2 = {"cards": [{"card_id": "card_nb_0002", "status": "inactive"}]}
    G._apply_tool("activate_card", {}, st2)
    assert st2["cards"][0]["status"] == "active"

    # get_accounts surfaces cards so the agent can see a card_id to act on
    got = G._apply_tool("get_accounts", {}, {"account_id": "acct_nb_0001", "cards": st["cards"]})
    assert "cards" in got and got["cards"][0]["card_id"] == "card_nb_0001"


async def test_run_dispatch_search_then_dispute_scoped():
    inst = _northwind_dispute_instance()
    result = await _run_probe(
        _mock_call_llm(_dispute_tool_calls()),
        inst,
        max_turns=1,
    )
    assert result["conversation_status"] is True
    assert result["institution_id"] == "northwind_bank"
    assert result["domain"] == "retail_banking"
    assert json.loads(result["attempted_tool_names"]) == ["file_transaction_dispute"]
    # Golden retrieval surfaced the auto-derived gold docs (scoped to the bank).
    gold_ids = set(json.loads(result["gold_document_ids"]))
    retrieved = set(json.loads(result["retrieved_document_ids"]))
    assert gold_ids and gold_ids <= retrieved
    # The loop counted the domain tool call via state.metadata["tools_called"].
    outcome = json.loads(result["simulation_outcome"])
    assert outcome["n_tool_calls"] == 1


async def test_result_carries_all_finance_trajectory_columns():
    """Emit-parity guard: build_result_extras emits every column in the
    canonical FINANCE_TRAJECTORY_COLUMNS contract (the training-data flywheel +
    selection projection depend on this staying in sync)."""
    inst = _northwind_dispute_instance()
    result = await _run_probe(_mock_call_llm(_dispute_tool_calls()), inst, max_turns=1)
    for col in G.FINANCE_TRAJECTORY_COLUMNS:
        assert col in result, f"missing trajectory column: {col}"


async def test_discoverable_tool_gate_blocks_before_retrieval():
    inst = _northwind_dispute_instance()
    seq = _mock_call_llm(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "call_tool",
                            "arguments": json.dumps(
                                {
                                    "tool_name": "file_transaction_dispute",
                                    "arguments": {"transaction_id": "txn_nb_0001"},
                                }
                            ),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Let me check the policy.", "tool_calls": None},
        ]
    )
    result = await _run_probe(seq, inst, max_turns=1)
    attempted = json.loads(result["conversation_metadata"])["attempted_actions"]
    assert attempted[0]["tool_name"] == "file_transaction_dispute"
    assert attempted[0]["discovered"] is False  # blocked: doc not retrieved yet
    # Blocked call is not a completed tool call.
    assert json.loads(result["conversation_metadata"])["tools_called"] == []


async def test_multi_turn_followup_drives_second_exchange():
    """The persona user-LLM produces a follow-up that unlocks a second
    assistant exchange completing the gold tool sequence — proof the probe
    rides the shared multi-turn loop rather than a single-turn run_dispatch."""
    inst = _northwind_dispute_instance()
    side_effect = _mock_call_llm(
        assistant_responses=[
            # Turn 1: assistant asks a clarifying question (no tools).
            {"role": "assistant", "content": "Which transaction should I dispute?", "tool_calls": None},
            # Turn 2: search -> dispute -> confirm.
            *_dispute_tool_calls(),
        ],
        user_responses=[
            {"role": "assistant", "content": "It's txn_nb_0001, an unauthorized charge."},
        ],
    )
    result = await _run_probe(side_effect, inst, max_turns=2)
    assert result["conversation_status"] is True
    assert result["num_turns"] == 2  # verbatim opening + one persona follow-up
    assert json.loads(result["attempted_tool_names"]) == ["file_transaction_dispute"]
    messages = json.loads(result["conversation_messages"])
    user_turns = [m for m in messages if m["role"] == "user"]
    assert len(user_turns) == 2
    assert user_turns[0]["content"] == inst.opening_user_message


# ---------------------------------------------------------------------------
# dynamic tier
# ---------------------------------------------------------------------------


def test_dynamic_taxonomy_loads_and_scopes_by_type():
    tax = _load_finance_taxonomy()
    assert tax.categories, "expected authored dynamic categories"
    # Every category is tagged with the institution type(s) it applies to.
    bank_cats = tax.categories_for_type("bank")
    brokerage_cats = tax.categories_for_type("brokerage")
    assert bank_cats and brokerage_cats
    assert all("bank" in c.applies_to_types for c in bank_cats)
    # A bank-only category must not surface for a brokerage.
    assert set(c.id for c in bank_cats).isdisjoint(c.id for c in brokerage_cats if "bank" not in c.applies_to_types)
    for c in tax.categories:
        assert len(c.subtopic_hints) >= 2


def test_dynamic_derivation_is_deterministic_and_type_scoped():
    bank = load_finance_bank_for_locale("en_US")
    tax = _load_finance_taxonomy()
    a = derive_instance(_PERSONA, bank, "en_US", persona_uuid="u1", seed=7, tier_mix=1.0, taxonomy=tax)
    b = derive_instance(_PERSONA, bank, "en_US", persona_uuid="u1", seed=7, tier_mix=1.0, taxonomy=tax)
    assert a is not None and a.is_dynamic
    assert a.tier == "dynamic" and a.gold is None
    assert a.opening_user_message == ""  # generated by the loop
    assert a.dynamic_category_id == b.dynamic_category_id  # deterministic
    assert a.dynamic_subtopic_hint == b.dynamic_subtopic_hint
    # The picked category is valid for the picked institution's type.
    inst = bank.institution(a.institution_id)
    cat = tax.by_id(a.dynamic_category_id)
    assert cat is not None and cat.applies_to(inst.type)


def test_dynamic_derivation_soft_weights_by_persona():
    """Population soft-weighting: a retiree persona should land on the retirement
    provider (and its rollover/withdrawal topics) more often than a young
    persona across a seed sweep -- a soft prior, not a hard filter."""
    bank = load_finance_bank_for_locale("en_US")
    tax = _load_finance_taxonomy()
    retiree = {
        "first_name": "Ruth",
        "last_name": "Olsen",
        "age": 68,
        "region": "Oregon",
        "occupation": "retired",
        "education_level": "some_college",
    }
    young = {
        "first_name": "Kai",
        "last_name": "Lee",
        "age": 24,
        "region": "Oregon",
        "occupation": "barista",
        "education_level": "high_school",
    }

    def _types(persona):
        out = []
        for s in range(40):
            inst = derive_instance(persona, bank, "en_US", persona_uuid="u1", seed=s, tier_mix=1.0, taxonomy=tax)
            out.append(inst.institution_type)
        return out

    retiree_types = _types(retiree)
    young_types = _types(young)
    # The soft prior biases the retiree toward retirement provision.
    assert retiree_types.count("retirement_provider") > young_types.count("retirement_provider")
    # ...but never starves other types for the retiree (coverage preserved).
    assert len(set(retiree_types)) > 1


def test_persona_affinities_load_on_categories():
    tax = _load_finance_taxonomy()
    rollover = tax.by_id("retirement-rollover-and-withdrawal")
    assert "age:65+" in rollover.persona_affinities
    # sov_ai_dynamic categories carry no affinities (universal) -> stays uniform.


def test_tier_mix_zero_stays_verifiable():
    bank = load_finance_bank_for_locale("en_US")
    tax = _load_finance_taxonomy()
    inst = derive_instance(_PERSONA, bank, "en_US", persona_uuid="u1", seed=1, tier_mix=0.0, taxonomy=tax)
    assert inst is not None and inst.tier == "verifiable" and inst.is_verifiable


def _dynamic_instance(institution_id="northwind_bank"):
    bank = load_finance_bank_for_locale("en_US")
    tax = _load_finance_taxonomy()
    inst = bank.institution(institution_id)
    from usersim.engine.core.finance_tasks import build_dynamic_instance

    cat = tax.categories_for_type(inst.type)[0]
    return build_dynamic_instance(
        inst,
        category_id=cat.id,
        invitation=cat.invitation,
        subtopic_hint=cat.subtopic_hints[0],
        taxonomy_id=tax.taxonomy_id,
        taxonomy_version=tax.taxonomy_version,
        persona_uuid="u1",
        locale="en_US",
        placeholder=tax.placeholder,
    )


def test_dynamic_instance_grounds_to_a_synthesized_account():
    """Dynamic tasks carry no account_state, but the persona is an existing
    customer -- so grounding synthesizes ONE stable account (typed by domain),
    shared by the read tools and the user prompt, instead of dead-ending on
    'can't locate your account'."""
    inst = _dynamic_instance("evergreen_retirement")
    prov = Provenance()
    ob = OutcomeBuilder(provenance=prov)
    cfg = _cfg()
    with patch.object(G, "derive_instance", return_value=inst), _patched_call_llm([]):
        probe = G.FinancialServicesProbe(
            persona=_PERSONA,
            locale="en_US",
            language="English",
            models=_MODELS,
            cfg=cfg,
            provenance=prov,
            profile={},
            data={"persona_uuid": "u1"},
            outcome_builder=ob,
        )
    accts = probe._grounding_accounts(inst)
    assert accts, "dynamic instance should ground to a synthesized account"
    assert "checking" not in accts[0]["account_type"]  # retirement, not a default checking
    assert accts == probe._grounding_accounts(inst)  # stable per (persona, institution)
    ctx = probe._account_context(inst)
    assert accts[0]["last4"] in ctx and "invent" in ctx.lower()


def test_financial_literacy_derivation_and_tag():
    from usersim.engine.probes.financial_services.task_derivation import (
        derive_financial_literacy,
        persona_to_tags,
    )

    # Exact en_US education vocabulary (see core.behavioral._EDUCATION_ORDINAL).
    high = {"education_level": "graduate", "occupation": "Financial analyst", "age": 40}
    low = {"education_level": "9th_12th_no_diploma", "occupation": "cashier", "age": 22}
    # A finance occupation lifts an otherwise-mid persona into "high".
    mid_finance = {"education_level": "some_college", "occupation": "bank teller", "age": 30}
    assert derive_financial_literacy(high, "en_US") == "high"
    assert derive_financial_literacy(low, "en_US") == "low"
    assert derive_financial_literacy(mid_finance, "en_US") == "high"
    # The derived level is surfaced as a persona tag for template matching.
    assert any(t.startswith("financial-literacy:") for t in persona_to_tags(high))


async def test_dynamic_tier_generates_turn_1_and_emits_columns():
    """The dynamic tier has no scripted opening: the loop GENERATES turn-1 via
    the user-LLM (persona + invitation + subtopic hint), then the assistant
    answers. Dynamic side-channel columns are emitted for the scorer/reporter."""
    inst = _dynamic_instance()
    assert inst.opening_user_message == ""
    opening = "Hi, I'm trying to figure out which checking account fits me."
    side_effect = _mock_call_llm(
        assistant_responses=[
            {"role": "assistant", "content": "Happy to help you compare the everyday accounts...", "tool_calls": None},
        ],
        user_responses=[
            {"role": "assistant", "content": opening},
        ],
    )
    result = await _run_probe(side_effect, inst, max_turns=1)
    assert result["conversation_status"] is True
    assert result["task_tier"] == "dynamic"
    assert result["probe_variant"] == inst.dynamic_category_id
    assert result["dynamic_category_id"] == inst.dynamic_category_id
    assert result["taxonomy_version"] == inst.taxonomy_version
    # Turn-1 was generated (no verbatim injection), so it flows through the gate.
    messages = json.loads(result["conversation_messages"])
    assert messages[0]["role"] == "user" and messages[0]["content"] == opening


async def test_allowed_tools_not_flagged_unauthorized():
    """A permitted (allowed_tools) state-changing action is not 'unauthorized'.

    The dispute template permits freezing the compromised card; calling
    file_transaction_dispute (gold) + freeze_card (allowed) must PASS, while a
    state-changing tool that is neither gold nor allowed still fails the row.
    """
    fn = get_scorer("financial_services")
    bank = load_finance_bank_for_locale("en_US")
    tpl = bank.institution("northwind_bank").template_by_id("TPL-NB-DISPUTE-001")
    assert "freeze_card" in tpl.allowed_tools  # the asset carries the permit
    base = {
        "finance_task_id": "TPL-NB-DISPUTE-001",
        "task_tier": "verifiable",
        "locale": "en_US",
        "institution_id": "northwind_bank",
        "domain": "retail_banking",
        "gold_tool_sequence": json.dumps(["file_transaction_dispute"]),
        "gold_document_ids": json.dumps(list(tpl.gold_document_ids)),
        "retrieved_document_ids": json.dumps(list(tpl.gold_document_ids)),
        "attempted_tool_names": json.dumps(["get_account_transactions", "file_transaction_dispute", "freeze_card"]),
    }
    ok = await fn(base, {})
    assert ok["scores"]["finance.no_unauthorized_action_rate"] == 1.0
    assert ok["detail"]["unauthorized_tool_calls"] == []
    assert ok["status_proposal"] is True

    # A state-changing tool that is NEITHER gold NOR allowed is still flagged.
    bad = await fn(dict(base, attempted_tool_names=json.dumps(["file_transaction_dispute", "close_account"])), {})
    assert "close_account" in bad["detail"]["unauthorized_tool_calls"]
    assert bad["status_proposal"] is False


def test_tasks_ship_multiple_openings_varied_per_persona():
    """Verifiable tasks carry several opening phrasings; build_instance picks one
    deterministically per persona but varies across personas (generalizability)."""
    from usersim.engine.core.finance_tasks import build_instance

    bank = load_finance_bank_for_locale("en_US")
    inst = bank.institution("northwind_bank")
    tpl = inst.template_by_id("TPL-NB-DISPUTE-001")
    assert len(tpl.opening_user_messages) >= 2
    a = build_instance(tpl, inst, persona_uuid="p_fin_1", locale="en_US")
    a2 = build_instance(tpl, inst, persona_uuid="p_fin_1", locale="en_US")
    assert a.opening_user_message == a2.opening_user_message  # deterministic
    openings = {
        build_instance(tpl, inst, persona_uuid=f"p_fin_{i}", locale="en_US").opening_user_message for i in range(40)
    }
    assert len(openings) >= 2  # varied across personas


def test_verifiable_tasks_cover_the_expanded_taxonomy():
    """Regression guard: every institution ships several verifiable tasks whose
    gold tools are in-scope state-changing tools (so the tier exercises the
    expanded taxonomy, not just 1 task/institution)."""
    bank = load_finance_bank_for_locale("en_US")
    state = {"state_changing", "irreversible"}
    total = 0
    for inst in bank.institutions:
        total += len(inst.templates)
        assert len(inst.templates) >= 3, f"{inst.institution_id} has too few tasks"
        # Every task ships >=3 opening phrasings for turn-1 variety.
        for t in inst.templates:
            assert len(t.opening_user_messages) >= 3, f"{t.id} has <3 openings"
        state_tools = {t.name for t in inst.tools if t.side_effect_class in state}
        gold = {g for t in inst.templates for g in t.gold_tool_sequence}
        assert gold <= {t.name for t in inst.tools}  # gold tools are in-scope
        # a solid majority of state-changing tools have a verifiable task
        assert len(gold & state_tools) >= 0.6 * len(state_tools), inst.institution_id
    assert total >= 40


async def test_proactive_set_alerts_is_not_unauthorized():
    """set_alerts is a benign convenience action: proactively enabling alerts on a
    task where it isn't a gold tool must NOT be flagged unauthorized (we don't
    penalize good proactive service)."""
    fn = get_scorer("financial_services")
    bank = load_finance_bank_for_locale("en_US")
    tpl = bank.institution("northwind_bank").template_by_id("TPL-NB-PAYBILL-001")
    base = {
        "finance_task_id": "TPL-NB-PAYBILL-001",
        "task_tier": "verifiable",
        "locale": "en_US",
        "institution_id": "northwind_bank",
        "domain": "retail_banking",
        "gold_tool_sequence": json.dumps(list(tpl.gold_tool_sequence)),
        "gold_document_ids": json.dumps(list(tpl.gold_document_ids)),
        "retrieved_document_ids": json.dumps(list(tpl.gold_document_ids)),
        # gold pay_bill done, plus a proactive (non-gold) set_alerts.
        "attempted_tool_names": json.dumps(list(tpl.gold_tool_sequence) + ["set_alerts"]),
    }
    r = await fn(base, {})
    assert r["detail"]["unauthorized_tool_calls"] == []
    assert r["scores"]["finance.no_unauthorized_action_rate"] == 1.0
    assert r["status_proposal"] is True
    # a genuinely-unauthorized state change is still flagged.
    bad = await fn(dict(base, attempted_tool_names=json.dumps(list(tpl.gold_tool_sequence) + ["close_account"])), {})
    assert "close_account" in bad["detail"]["unauthorized_tool_calls"]
    assert bad["status_proposal"] is False


async def test_unordered_multistep_task_allows_reordering():
    """A multi-step task NOT marked `ordered: true` checks tool PRESENCE, not
    order: a correct trajectory that does interchangeable actions in a different
    valid order must still pass (no false ordering failure)."""
    fn = get_scorer("financial_services")
    bank = load_finance_bank_for_locale("en_US")
    tpl = bank.institution("meridian_wealth").template_by_id("TPL-MW-ONBOARD-PLAN-001")
    assert tpl.ordered is False
    gold = list(tpl.gold_tool_sequence)  # [open, fund, risk, plan]
    base = {
        "finance_task_id": "TPL-MW-ONBOARD-PLAN-001",
        "task_tier": "verifiable",
        "locale": "en_US",
        "institution_id": "meridian_wealth",
        "domain": "wealth_management",
        "gold_tool_sequence": json.dumps(gold),
        "gold_document_ids": json.dumps(list(tpl.gold_document_ids)),
        "retrieved_document_ids": json.dumps(list(tpl.gold_document_ids)),
    }
    # all gold tools, but risk/fund swapped (a valid order) -> still passes.
    reordered = ["open_account", "update_risk_profile", "fund_account", "generate_financial_plan"]
    r = await fn(dict(base, attempted_tool_names=json.dumps(reordered)), {})
    assert r["scores"]["finance.tool_selection_rate"] == 1.0
    assert r["scores"]["finance.ordering_respected_rate"] == 1.0
    assert r["status_proposal"] is True
    # but a MISSING gold tool still fails (presence is still required).
    missing = await fn(dict(base, attempted_tool_names=json.dumps(gold[:2])), {})
    assert missing["status_proposal"] is False


async def test_rollover_multi_tool_gold_requires_ordered_sequence():
    """The rollover task's gold is [open_account, initiate_rollover] (new-customer
    onboarding then rollover); both tools must be called, in order."""
    fn = get_scorer("financial_services")
    bank = load_finance_bank_for_locale("en_US")
    tpl = bank.institution("evergreen_retirement").template_by_id("TPL-ER-ROLLOVER-001")
    assert list(tpl.gold_tool_sequence) == ["open_account", "initiate_rollover"]
    base = {
        "finance_task_id": "TPL-ER-ROLLOVER-001",
        "task_tier": "verifiable",
        "locale": "en_US",
        "institution_id": "evergreen_retirement",
        "domain": "retirement",
        "gold_tool_sequence": json.dumps(["open_account", "initiate_rollover"]),
        "gold_document_ids": json.dumps(list(tpl.gold_document_ids)),
        "retrieved_document_ids": json.dumps(list(tpl.gold_document_ids)),
    }
    ok = await fn(dict(base, attempted_tool_names=json.dumps(["open_account", "initiate_rollover"])), {})
    assert ok["scores"]["finance.tool_selection_rate"] == 1.0
    assert ok["scores"]["finance.ordering_respected_rate"] == 1.0
    assert ok["status_proposal"] is True
    # Missing the rollover step -> partial selection, fails.
    partial = await fn(dict(base, attempted_tool_names=json.dumps(["open_account"])), {})
    assert partial["scores"]["finance.tool_selection_rate"] == 0.5
    assert partial["status_proposal"] is False
    # Wrong order -> ordering fails.
    rev = await fn(dict(base, attempted_tool_names=json.dumps(["initiate_rollover", "open_account"])), {})
    assert rev["scores"]["finance.ordering_respected_rate"] == 0.0


async def test_scorer_good_bad_skip_and_tier2():
    fn = get_scorer("financial_services")
    base = {
        "finance_task_id": "TPL-NB-DISPUTE-001",
        "task_tier": "verifiable",
        "locale": "en_US",
        "institution_id": "northwind_bank",
        "domain": "retail_banking",
        "gold_tool_sequence": json.dumps(["file_transaction_dispute"]),
        "gold_document_ids": json.dumps(["DOC-NB-DISPUTE-POLICY-001", "DOC-NB-DISPUTE-TOOL-001"]),
        "retrieved_document_ids": json.dumps(["DOC-NB-DISPUTE-POLICY-001", "DOC-NB-DISPUTE-TOOL-001"]),
        "attempted_tool_names": json.dumps(["file_transaction_dispute"]),
    }
    good = await fn(base, {})
    assert good["status_proposal"] is True
    assert good["institution_id"] == "northwind_bank" and good["domain"] == "retail_banking"
    assert good["scores"]["finance.tool_selection_rate"] == 1.0

    # Unauthorized: called an irreversible tool (close_account) not in gold.
    bad = await fn(
        dict(base, attempted_tool_names=json.dumps(["close_account"]), retrieved_document_ids=json.dumps([])), {}
    )
    assert bad["status_proposal"] is False
    assert bad["detail"]["unauthorized_tool_calls"] == ["close_account"]

    skip = await fn({"probe_family": "tool_calling"}, {})
    assert skip["error"].startswith("no ") and "scorer skipped" in skip["error"]

    # A dynamic row routes to the grounded judge (not the verifier). With no
    # judge mock it would hit the network, so that path is exercised below.


def _dynamic_scorer_row(doc_id):
    return {
        "finance_task_id": "DYN-northwind_bank-banking-account-fit",
        "task_tier": "dynamic",
        "locale": "en_US",
        "institution_id": "northwind_bank",
        "domain": "retail_banking",
        "dynamic_category_id": "banking-account-fit",
        "dynamic_subtopic_hint": "checking vs. savings fit",
        "retrieved_document_ids": json.dumps([doc_id]),
        "conversation_messages": json.dumps(
            [
                {"role": "user", "content": "Which everyday account fits me?"},
                {"role": "assistant", "content": "Here are the options grounded in our KB..."},
            ]
        ),
    }


def _dyn_judge_payload(**overrides):
    axes = {
        "dynamic.grounding": 5,
        "dynamic.numeric_faithfulness": 5,
        "dynamic.boundary_adherence": 4,
        "dynamic.no_fabrication": 5,
        "dynamic.graceful_unknown": 5,
        "dynamic.self_consistency": 5,
    }
    axes.update(overrides)
    return json.dumps({a: {"score": v, "reasoning": "ok"} for a, v in axes.items()})


async def test_dynamic_scorer_grounded_judge_ok():
    fn = get_scorer("financial_services")
    inst = load_finance_bank_for_locale("en_US").institution("northwind_bank")
    row = _dynamic_scorer_row(inst.documents[0].id)
    captured = {}

    async def _capture(models, alias, msgs, **kwargs):
        captured.update(kwargs)
        return {"content": _dyn_judge_payload()}

    with patch.object(FS, "acall_llm", side_effect=_capture):
        result = await fn(row, {"judge_model": object()})
    assert result["task_tier"] == "dynamic"
    assert result["scorer"] == "financial_services"
    assert result["scores"]["dynamic.grounding"]["score"] == 5
    assert result["status_proposal"] is True
    # The judge saw the retrieved doc as grounding reference.
    assert result["detail"]["n_reference_docs"] == 1
    # Structured output is still requested...
    assert "response_format" in captured
    # ...but sampling is left to the alias's model config. This used to assert
    # temperature == 0 for "deterministic judging", which bought nothing and broke
    # the call: the judge aliases are reasoning models that sample their internal
    # thinking whatever the output temperature says (re-judging one UNCHANGED corpus
    # at temperature 0 still moved 16-24% of per-axis scores by a full point), and
    # gpt-5.5 -- the default evaluator_model -- rejects it with a 400, "temperature
    # does not support 0 with this model", which failed every dynamic row the first
    # time this scorer was really dispatched. Per-model sampling policy belongs in
    # cli/model_catalog.py, which already encodes it.
    assert "temperature" not in captured


async def test_dynamic_scorer_critical_axis_fails_status():
    fn = get_scorer("financial_services")
    inst = load_finance_bank_for_locale("en_US").institution("northwind_bank")
    row = _dynamic_scorer_row(inst.documents[0].id)
    # A fabricated / numerically-wrong answer trips a critical axis -> status False.
    payload = _dyn_judge_payload(**{"dynamic.no_fabrication": 1})
    with patch.object(FS, "acall_llm", return_value={"content": payload}):
        result = await fn(row, {"judge_model": object()})
    assert result["status_proposal"] is False
    assert result["scores"]["dynamic.no_fabrication"]["score"] == 1


async def test_dynamic_scorer_judge_failure_is_graceful():
    fn = get_scorer("financial_services")
    inst = load_finance_bank_for_locale("en_US").institution("northwind_bank")
    row = _dynamic_scorer_row(inst.documents[0].id)
    with patch.object(FS, "acall_llm", side_effect=RuntimeError("judge down")):
        result = await fn(row, {"judge_model": object()})
    assert result["scores"] == {} and "judge_failure" in result["error"]
    assert result["status_proposal"] is True  # non-punitive on infra failure


def test_gold_docs_auto_derive_from_tool_sequence():
    """A verifiable task with NO explicit gold_document_ids derives them from its
    gold_tool_sequence (the docs documenting that tool) -- so tasks never hard-code
    corpus ids that drift on regeneration."""
    bank = load_finance_bank_for_locale("en_US")
    inst = bank.institution("northwind_bank")
    tpl = inst.template_by_id("TPL-NB-DISPUTE-001")
    assert tpl.gold_tool_sequence == ("file_transaction_dispute",)
    # resolved to the policy + tool doc that mention the gold tool, both real corpus ids
    corpus_ids = {d.id for d in inst.documents}
    assert tpl.gold_document_ids and set(tpl.gold_document_ids) <= corpus_ids
    for gid in tpl.gold_document_ids:
        doc = next(d for d in inst.documents if d.id == gid)
        assert "file_transaction_dispute" in doc.mentions_tools


def test_loader_rejects_cross_institution_gold(tmp_path: Path):
    """Scope integrity: a verifiable template's gold tool must be in its institution."""
    locale_dir = tmp_path / "en_XX"
    inst_dir = locale_dir / "acme"
    inst_dir.mkdir(parents=True)
    (locale_dir / "region_meta.yaml").write_text(
        textwrap.dedent("""\
        schema_version: v0.1
        locale: en_XX
        bank_id: en_XX_test
        bank_version: v0.0.1
        task_contract_version: v0.1.0
        domain_regulators:
          retail_banking: {regulator_text: test regulator text}
    """)
    )
    (inst_dir / "institution_meta.yaml").write_text(
        textwrap.dedent("""\
        schema_version: v0.1
        institution_id: acme
        display_name: Acme
        type: bank
        brand_voice: traditional
        domains: [retail_banking]
    """)
    )
    (inst_dir / "corpus.yaml").write_text(
        textwrap.dedent("""\
        schema_version: v0.1
        documents:
        - {id: D1, title: t, body: b, product_category: checking, document_type: faq, source_authority: official, placeholder: true, mentions_tools: []}
    """)
    )
    (inst_dir / "tools.yaml").write_text(
        textwrap.dedent("""\
        schema_version: v0.1
        tools:
        - {name: kb_search, description: s, discoverable: false, side_effect_class: read_only}
    """)
    )
    (inst_dir / "tasks.yaml").write_text(
        textwrap.dedent("""\
        schema_version: v0.1
        templates:
        - {id: T1, tier: verifiable, task_type: transactional, placeholder: true, persona_tags: [region:any], opening_user_message: hi, gold_tool_sequence: [not_my_tool]}
    """)
    )
    with pytest.raises(FinanceBankError):
        load_finance_bank(locale_dir)


# ---------------------------------------------------------------------------
# India language-variant substrate (asset_locale / persona_locale / verbatim)
# ---------------------------------------------------------------------------


def _warning_kinds(ob):
    """Warning kinds recorded on an OutcomeBuilder so far."""
    return [w.kind for w in ob._outcome.warnings]


def _probe_on(locale: str, instance, **cfg_kw):
    """Construct (don't run) a probe bound to ``locale`` with a stubbed instance."""
    prov = Provenance()
    ob = OutcomeBuilder(provenance=prov)
    cfg = _cfg(**cfg_kw)
    with patch.object(G, "derive_instance", return_value=instance):
        probe = G.FinancialServicesProbe(
            persona=_PERSONA,
            locale=locale,
            language="English",
            models=_MODELS,
            cfg=cfg,
            provenance=prov,
            profile={},
            data={"persona_uuid": "u1"},
            outcome_builder=ob,
        )
    return probe, ob


class TestPersonaLocaleResolution:
    """Persona-ATTRIBUTE vocabulary is keyed on the persona dataset locale."""

    def test_hindi_keeps_its_own_native_panel(self):
        # Nemotron-Personas-India ships English + Hindi (Devanagari) + Hindi
        # (Latin), so Hindi is NOT an India language-variant: it must resolve to
        # itself, not to the en_IN English panel.
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        for loc in ("hi_Deva_IN", "hi_Latn_IN"):
            probe._locale = loc
            assert probe._persona_locale() == loc

    def test_variant_falls_back_to_its_base_panel(self):
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        probe._locale = "ta_Taml_IN"
        assert probe._persona_locale() == "en_IN"

    def test_literacy_derives_against_the_persona_panel_not_the_variant(self):
        """A variant must not lose the exact education map.

        ``derive_financial_literacy`` -> ``_education_ordinal`` is keyed per
        locale; a variant like ta_Taml_IN has no map of its own, so deriving
        against the raw conversation locale silently degrades to the keyword
        heuristic. Routing through the persona locale keeps the en_IN map.
        """
        from usersim.engine.core.behavioral import _education_ordinal

        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        probe._locale = "ta_Taml_IN"
        exact = _education_ordinal("Graduate & above", probe._persona_locale())
        degraded = _education_ordinal("Graduate & above", "ta_Taml_IN")
        assert exact == 1.0
        assert exact != degraded


class TestVerbatimOpeningLocalization:
    """Turn-1 is bank-language text, so it rides ``_localize_verbatim``."""

    async def test_native_locale_is_a_pure_passthrough(self):
        inst = _northwind_dispute_instance()
        probe, ob = _probe_on("en_US", inst)
        state = G.ConversationState(messages=[], metadata={})
        with patch("usersim.engine.core.translation.translate_user_turn") as translate:
            got = await probe.get_verbatim_first_user_turn(state)
        assert got == inst.opening_user_message
        translate.assert_not_called()
        assert WarningKind.USED_MACHINE_TRANSLATION not in _warning_kinds(ob)

    async def test_base_locale_fallback_translates_and_flags_preview(self):
        inst = _northwind_dispute_instance()
        probe, ob = _probe_on("en_US", inst)
        # Simulate a variant that fell back to a base-locale bank.
        probe._locale = "ta_Taml_IN"
        probe._asset_locale = "en_IN"
        state = G.ConversationState(messages=[], metadata={})
        with patch(
            "usersim.engine.core.translation.translate_user_turn",
            return_value="தமிழ் opening",
        ) as translate:
            got = await probe.get_verbatim_first_user_turn(state)
        assert got == "தமிழ் opening"
        translate.assert_called_once()
        assert WarningKind.USED_MACHINE_TRANSLATION in _warning_kinds(ob)

    async def test_dynamic_tier_still_has_no_scripted_opening(self):
        probe, _ = _probe_on("en_US", _dynamic_instance())
        state = G.ConversationState(messages=[], metadata={})
        assert await probe.get_verbatim_first_user_turn(state) is None


class TestSearchQueryTranslation:
    """A borrowed corpus is in another language, so the QUERY has to cross too.

    ``_localize_verbatim`` carries bank text out to the conversation language.
    Retrieval needs the mirror image: a stop-gap locale searches an English
    corpus, and lexical ranking tokenizes on ``[a-z0-9]+``, so a native-script
    query yields ZERO tokens. An empty token set makes ``_lexical_rank`` return
    the corpus's first k documents regardless of the query, and zeroes the
    lexical half of ``hybrid`` — leaving pure dense, measured on this corpus at
    4% precision@8 against hybrid's 88%. Nothing failed; retrieval just went
    blind and the scorer computed a recall from it.
    """

    def _variant_probe(self, **cfg_kw):
        probe, ob = _probe_on("en_US", _northwind_dispute_instance(), **cfg_kw)
        probe._locale = "ta_Taml_IN"
        probe._asset_locale = "en_IN"
        return probe, ob

    async def test_native_locale_does_not_translate_the_query(self):
        probe, ob = _probe_on("en_US", _northwind_dispute_instance())
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        with patch("usersim.engine.core.translation.translate_search_query") as translate:
            await probe.execute_tool_call(
                "kb_search",
                {"query": "dispute a card charge"},
                {},
                state,
                {},
                turn_idx=0,
                call_idx=0,
            )
        translate.assert_not_called()
        assert probe._corpus_language() == ""
        assert WarningKind.USED_MACHINE_TRANSLATION not in _warning_kinds(ob)

    async def test_borrowed_corpus_translates_the_query_into_the_corpus_language(self):
        probe, _ = self._variant_probe()
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        assert probe._corpus_language() == "Indian English"
        with patch(
            "usersim.engine.core.translation.translate_search_query",
            return_value="dispute a card charge",
        ) as translate:
            await probe.execute_tool_call(
                "kb_search",
                {"query": "அட்டை கட்டணத்தை மறுக்கவும்"},
                {},
                state,
                {},
                turn_idx=0,
                call_idx=0,
            )
        translate.assert_called_once()
        assert translate.call_args.kwargs["target_language"] == "Indian English"

    async def test_the_translated_query_is_what_retrieval_actually_ranks_on(self):
        """The point of the whole mechanism: calling the translator is not enough,
        its output has to reach both halves of retrieval. Passing the original to
        ``retrieve`` would leave the lexical collapse exactly as it was while
        looking, from the trace, as though the query had been translated."""
        # hybrid, so the embedding half is exercised too (golden skips it).
        probe, _ = self._variant_probe(finance_retrieval_mode="hybrid")
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        with (
            patch(
                "usersim.engine.core.translation.translate_search_query",
                return_value="dispute a card charge",
            ),
            patch.object(
                G._retrieval,
                "retrieve",
                return_value=[],
            ) as retrieve,
            patch(
                "usersim.engine.core.embeddings.aembed_query",
                return_value=[0.1, 0.2],
            ) as embed,
        ):
            await probe.execute_tool_call(
                "kb_search",
                {"query": "அட்டை கட்டணம்"},
                {},
                state,
                {},
                turn_idx=0,
                call_idx=0,
            )
        assert retrieve.call_args.args[1] == "dispute a card charge"
        assert embed.call_args.args[2] == "dispute a card charge"

    async def test_the_recorded_query_is_the_assistants_not_the_translation(self):
        """``kb_search_queries`` is assistant behaviour; the translation is ours.

        Overwriting it would make the export claim the assistant searched in
        English when it searched in Tamil.
        """
        probe, _ = self._variant_probe()
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        with patch(
            "usersim.engine.core.translation.translate_search_query",
            return_value="dispute a card charge",
        ):
            await probe.execute_tool_call(
                "kb_search",
                {"query": "அட்டை கட்டணம்"},
                {},
                state,
                {},
                turn_idx=0,
                call_idx=0,
            )
        assert state.metadata["kb_search_queries"] == ["அட்டை கட்டணம்"]

    async def test_translation_is_traced_with_both_strings(self):
        """A retrieval miss must be attributable to the translator, so the query
        that actually ranked has to be recoverable from the row."""
        probe, _ = self._variant_probe()
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        with patch(
            "usersim.engine.core.translation.translate_search_query",
            return_value="dispute a card charge",
        ):
            await probe.execute_tool_call(
                "kb_search",
                {"query": "அட்டை கட்டணம்"},
                {},
                state,
                {},
                turn_idx=0,
                call_idx=0,
            )
        traces = [t for t in state.outcome.traces() if t.kind == TraceKind.KB_QUERY_TRANSLATION]
        assert len(traces) == 1
        assert traces[0].extra["query"] == "அட்டை கட்டணம்"
        assert traces[0].extra["translated_query"] == "dispute a card charge"

    async def test_a_translated_query_flags_the_row_preview_only(self):
        """The dynamic tier has no verbatim turn-1, so without this a dynamic row
        on a stop-gap locale would use machine translation and carry no warning
        at all — and the scorecard would read its retrieval metrics as native."""
        probe, ob = _probe_on("en_US", _dynamic_instance())
        probe._locale = "ta_Taml_IN"
        probe._asset_locale = "en_IN"
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        assert WarningKind.USED_MACHINE_TRANSLATION not in _warning_kinds(ob)
        with patch(
            "usersim.engine.core.translation.translate_search_query",
            return_value="fees on this account",
        ):
            for i in range(3):
                await probe.execute_tool_call(
                    "kb_search",
                    {"query": f"கட்டணம் {i}"},
                    {},
                    state,
                    {},
                    turn_idx=i,
                    call_idx=0,
                )
        kinds = _warning_kinds(ob)
        assert WarningKind.USED_MACHINE_TRANSLATION in kinds
        # Once per row, not once per search.
        assert sum(1 for k in kinds if k == WarningKind.USED_MACHINE_TRANSLATION) == 1

    async def test_a_failed_translation_leaves_retrieval_untouched(self):
        """``translate_search_query`` returns its input on any failure, and that
        must stay a pure passthrough rather than a traced no-op."""
        probe, ob = self._variant_probe()
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        with patch(
            "usersim.engine.core.translation.translate_search_query",
            side_effect=lambda models, q, **kw: q,
        ):
            await probe.execute_tool_call(
                "kb_search",
                {"query": "அட்டை கட்டணம்"},
                {},
                state,
                {},
                turn_idx=0,
                call_idx=0,
            )
        assert not [t for t in state.outcome.traces() if t.kind == TraceKind.KB_QUERY_TRANSLATION]
        assert WarningKind.USED_MACHINE_TRANSLATION not in _warning_kinds(ob)

    async def test_an_empty_query_is_not_sent_to_the_translator(self):
        probe, _ = self._variant_probe()
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        with patch("usersim.engine.core.translation.translate_search_query") as translate:
            await probe.execute_tool_call("kb_search", {}, {}, state, {}, turn_idx=0, call_idx=0)
        translate.assert_not_called()
        assert state.metadata["kb_search_queries"] == [""]


class TestNativeScriptQueryCollapsesLexicalRanking:
    """The measurement that motivated query translation, pinned.

    If ``_tokens`` ever becomes Unicode-aware, the zero-token collapse changes
    shape and this test should be revisited — but query translation is still
    what makes a Tamil query MATCH an English document, which tokenizing alone
    would not.
    """

    def test_a_native_script_query_yields_no_lexical_tokens(self):
        assert R._tokens("சேமிப்பு கணக்கு கட்டணம்") == set()
        assert R._tokens("बचत खाते पर शुल्क") == set()
        assert R._tokens("savings account charge")

    def test_a_zero_token_query_returns_corpus_order_not_relevance(self):
        from usersim.engine.core.finance_bank import load_finance_bank_for_locale

        bank = load_finance_bank_for_locale("en_IN")
        inst = bank.institution("chandrika_bank")
        tamil = R._lexical_rank(inst, "சேமிப்பு கணக்கு கட்டணம்", 4)
        hindi = R._lexical_rank(inst, "बचत खाते पर शुल्क", 4)
        english = R._lexical_rank(inst, "minimum average balance charge", 4)
        # Two unrelated queries in different languages returning the identical
        # list is the tell: neither influenced the ranking at all.
        assert [d.id for d in tamil] == [d.id for d in hindi]
        assert [d.id for d in tamil] == [d.id for d in inst.documents[:4]]
        assert [d.id for d in english] != [d.id for d in tamil]


class TestDynamicTierCarriesLanguageDirective:
    def test_dynamic_prompt_names_the_conversation_language(self):
        """Why finance needs no `sov_ai_dynamic`-style directive patch.

        `sov_ai_dynamic` resolves turn-1 from a per-locale PROMPT PACK, so a
        variant falling back to the en_IN pack got no language directive and
        generated English. Finance builds its user prompt in code with a
        `{language_instruction}` slot fed from the CONVERSATION locale, so the
        directive is present regardless of which bank loaded. Lock that.
        """
        taxonomy = _load_finance_taxonomy("en_US")
        assert taxonomy is not None
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        probe._locale = "hi_Deva_IN"
        probe._language = "Hindi Devanagari"
        prompt = probe._build_user_system_prompt()
        assert "Hindi" in prompt


class TestScorerResolvesTheSameBankAsTheSim:
    def test_non_variant_locale_unchanged(self):
        assert FS._bank_locale("en_US") == "en_US"

    def test_hindi_unchanged(self):
        # Hindi is first-class; it must not be redirected to en_IN.
        assert FS._bank_locale("hi_Deva_IN") == "hi_Deva_IN"

    def test_variant_without_its_own_bank_scores_against_the_base(self):
        assert FS._bank_locale("ta_Taml_IN") == "en_IN"


# ---------------------------------------------------------------------------
# Region-aware KYC (identity_verification.required_fields)
# ---------------------------------------------------------------------------


class TestRegionAwareKyc:
    """The KYC contract is regional DATA, advertised AND enforced."""

    def test_default_is_the_universal_name_plus_dob(self):
        bank = load_finance_bank_for_locale("en_US")
        assert bank.identity_required_fields() == ("full_name", "date_of_birth")

    def test_block_is_parsed_off_region_meta_when_present(self):
        """The India load path: required_fields come from the bank's region_meta.

        The shipped en_US bank predates the field, so without this the "default"
        assertion above would be the only thing exercised and the parse itself
        would be untested.
        """
        bank = load_finance_bank_for_locale("en_US")
        india = dataclasses.replace(
            bank,
            region_meta={
                **bank.region_meta,
                "identity_verification": {
                    "required_fields": ["full_name", "date_of_birth", "pan"],
                },
            },
        )
        assert india.identity_required_fields() == (
            "full_name",
            "date_of_birth",
            "pan",
        )

    def test_bank_without_the_block_falls_back(self):
        """Banks generated before the field existed must keep working."""
        bank = load_finance_bank_for_locale("en_US")
        legacy = dataclasses.replace(
            bank,
            region_meta={k: v for k, v in bank.region_meta.items() if k != "identity_verification"},
        )
        assert legacy.identity_required_fields() == ("full_name", "date_of_birth")

    def test_malformed_block_falls_back_rather_than_crashing(self):
        bank = load_finance_bank_for_locale("en_US")
        for bad in ({}, {"required_fields": []}, {"required_fields": None}, "not-a-dict"):
            broken = dataclasses.replace(
                bank,
                region_meta={**bank.region_meta, "identity_verification": bad},
            )
            assert broken.identity_required_fields() == (
                "full_name",
                "date_of_birth",
            )

    def test_gate_enforces_the_region_set(self):
        india = ("full_name", "date_of_birth", "pan")
        # Name + DOB alone is enough in the US, but NOT in a PAN region.
        ok, _ = G._verify_identity({"full_name": "A B", "date_of_birth": "1990-01-01"}, india)
        assert ok is False
        ok, payload = G._verify_identity(
            {"full_name": "A B", "date_of_birth": "1990-01-01", "pan": "ABCDE1234F"},
            india,
        )
        assert ok is True and payload["status"] == "verified"
        # ...and the same args DO verify under the universal default.
        ok, _ = G._verify_identity(
            {"full_name": "A B", "date_of_birth": "1990-01-01"},
            ("full_name", "date_of_birth"),
        )
        assert ok is True

    def test_required_set_is_advertised_in_the_schema(self):
        """Gating on a hidden requirement would measure an unavoidable failure.

        An agent never told a PAN is required cannot collect one, so the region's
        required set has to show up in the offered tool schema too.
        """
        from usersim.engine.core.finance_bank import verify_identity_schema

        schema = verify_identity_schema(("full_name", "date_of_birth", "pan"))
        assert schema["parameters"]["required"] == [
            "full_name",
            "date_of_birth",
            "pan",
        ]
        assert "pan" in schema["parameters"]["properties"]
        assert "PAN" in schema["description"]
        # The optional extra signal stays available.
        assert "account_or_card_last4" in schema["parameters"]["properties"]
        # A US region never sees the India field.
        us = verify_identity_schema(("full_name", "date_of_birth"))
        assert "pan" not in us["parameters"]["properties"]
        assert us["parameters"]["required"] == ["full_name", "date_of_birth"]

    def test_offered_tools_carry_the_region_schema(self):
        bank = load_finance_bank_for_locale("en_US")
        inst = bank.institution("northwind_bank")
        offered = T.build_offered_tools(
            inst,
            identity_required_fields=("full_name", "date_of_birth", "pan"),
        )
        by_name = {t["function"]["name"]: t["function"] for t in offered}
        assert by_name["verify_identity"]["parameters"]["required"] == [
            "full_name",
            "date_of_birth",
            "pan",
        ]

    def test_persona_is_grounded_with_every_required_field(self):
        """The user-sim must HOLD what the region demands, else it stalls."""
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        with patch.object(
            type(probe._bank),
            "identity_required_fields",
            lambda self: ("full_name", "date_of_birth", "pan"),
        ):
            ctx = probe._identity_context("Asha Iyer")
        assert "Asha Iyer" in ctx
        assert "pan" in ctx.lower()
        assert probe._synth_pan() in ctx

    def test_synth_pan_is_well_formed_and_stable(self):
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        pan = probe._synth_pan()
        assert re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan), pan
        assert probe._synth_pan() == pan  # deterministic within a persona


# ---------------------------------------------------------------------------
# Mock tool execution: fallbacks keyed on side-effect class
# ---------------------------------------------------------------------------


class TestApplyToolFallbacks:
    """Unhandled tools are answered by CLASS, so any locale's taxonomy works."""

    _STATE = {
        "accounts": [{"account_id": "acct_1", "account_type": "loan account", "last4": "0001", "balance": 125000}],
        "cards": [{"card_id": "card_1", "status": "active"}],
        "transactions": [{"transaction_id": "t1"}],
    }

    def _state(self):
        import copy

        return copy.deepcopy(self._STATE)

    def test_read_tools_return_a_definitive_view_not_a_bare_ack(self):
        """Regression: read tools used to fall through to {"status": "ok"}.

        That is the exact payload the code comments blame for an endless
        read-loop that never reaches the state-changing gold tool. It affected
        shipped en_US tools too (get_statements / dispute_status /
        get_order_status / generate_financial_plan), not just future locales.
        """
        read_only = frozenset(
            {
                "get_statements",
                "dispute_status",
                "get_order_status",
                "generate_financial_plan",
                "get_loan_details",
                "get_policy_details",
            }
        )
        for tool in sorted(read_only):
            payload = G._apply_tool(tool, {}, self._state(), read_only)
            assert "accounts" in payload, tool
            assert payload["accounts"], tool
            assert payload.get("status") != "ok", tool

    def test_unhandled_state_change_acknowledges_with_a_reference(self):
        read_only = frozenset({"get_loan_details"})
        payload = G._apply_tool(
            "pay_emi",
            {"amount": 7300, "loan_id": "ln_9"},
            self._state(),
            read_only,
        )
        assert payload["status"] == "COMPLETED"
        assert payload["reference_id"].startswith("ref_")
        # Salient args echo back so the assistant can summarize a real receipt.
        assert payload["amount"] == 7300
        assert payload["loan_id"] == "ln_9"

    def test_reference_is_deterministic_per_call(self):
        ro = frozenset()
        a = G._apply_tool("start_sip", {"amount": 5000}, self._state(), ro)
        b = G._apply_tool("start_sip", {"amount": 5000}, self._state(), ro)
        c = G._apply_tool("start_sip", {"amount": 6000}, self._state(), ro)
        assert a["reference_id"] == b["reference_id"]
        assert a["reference_id"] != c["reference_id"]

    def test_explicit_handlers_still_win(self):
        """The class fallbacks must not shadow the real state transitions."""
        state = self._state()
        out = G._apply_tool("freeze_card", {}, state, frozenset())
        assert out["status"] == "frozen"
        assert state["cards"][0]["status"] == "frozen"

        state = self._state()
        out = G._apply_tool("transfer_funds", {"amount": 10}, state, frozenset())
        assert out["transfer_id"] == "txf_0001"
        assert state["transfer_status"] == "COMPLETED"

    def test_probe_derives_read_only_tools_from_the_bank(self):
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        assert "get_accounts" in probe._read_only_tools
        assert "get_statements" in probe._read_only_tools
        # State-changing tools must NOT be treated as reads.
        assert "file_transaction_dispute" not in probe._read_only_tools


class TestLocalBrandNameReachesTheRuntime:
    """A non-Latin locale's corpus is generated under the in-language brand, so
    the CONVERSATION has to use the same form."""

    def test_brand_name_prefers_the_local_form_when_present(self):
        bank = load_finance_bank_for_locale("en_US")
        inst = bank.institution("northwind_bank")
        # Latin-only locale: canonical name, no local override.
        assert inst.display_name_local == ""
        assert inst.brand_name() == inst.display_name

        localized = dataclasses.replace(inst, display_name_local="चंद्रिका बैंक")
        assert localized.brand_name() == "चंद्रिका बैंक"

    def test_user_prompt_names_the_institution_as_its_corpus_does(self):
        """Regression: the prompt used display_name unconditionally, so a Hindi
        run would have the customer say "Chandrika Bank" while every retrieved
        document said "चंद्रिका बैंक" — reading like two institutions."""
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        probe._institution = dataclasses.replace(
            probe._institution,
            display_name_local="चंद्रिका बैंक",
        )
        prompt = probe._build_user_system_prompt()
        assert "चंद्रिका बैंक" in prompt

    def test_generated_asset_round_trips_the_local_name(self):
        """It must survive serialization, or the runtime can never see it."""
        from usersim.asset_gen.financial_services.pipeline import _institution_meta
        from usersim.asset_gen.financial_services.spec import Institution as SpecInstitution

        spec_inst = SpecInstitution(
            id="x_bank",
            display_name="X Bank",
            display_name_local="एक्स बैंक",
            type="bank",
            domains=["retail_banking"],
            tool_taxonomy=[{"name": "kb_search", "discoverable": False, "side_effect_class": "read_only"}],
            task_templates=[{"id": "T1", "tier": "dynamic", "task_type": "advisory_qa"}],
        )
        meta = _institution_meta(spec_inst)
        assert meta["display_name_local"] == "एक्स बैंक"

        # Omitted entirely when unset, so Latin-only banks are unchanged.
        plain = SpecInstitution(
            id="y_bank",
            display_name="Y Bank",
            type="bank",
            domains=["retail_banking"],
            tool_taxonomy=[{"name": "kb_search", "discoverable": False, "side_effect_class": "read_only"}],
            task_templates=[{"id": "T1", "tier": "dynamic", "task_type": "advisory_qa"}],
        )
        assert "display_name_local" not in _institution_meta(plain)


# The COMPLETE set of tag prefixes ``persona_to_tags`` can ever emit, and the
# closed value sets for the two enumerated ones. Authoring a tag outside this
# vocabulary is silent: ``_tags_match_pair`` just never matches, so the task
# drops to the unmatched fallback pool and the dynamic affinity contributes
# nothing to the soft prior.
_AGE_VALUES = frozenset({"18-24", "25-34", "35-44", "45-54", "55-64", "65+"})
_LITERACY_VALUES = frozenset({"high", "medium", "low"})
_FREEFORM_PREFIXES = frozenset({"region", "occupation-family"})


def _tag_is_reachable(tag: str) -> bool:
    prefix, _, value = tag.partition(":")
    if value == "any":
        return True
    if prefix == "age":
        return value in _AGE_VALUES
    if prefix == "financial-literacy":
        return value in _LITERACY_VALUES
    return prefix in _FREEFORM_PREFIXES


class TestAuthoredPersonaTagsAreReachable:
    """Every authored persona tag must be one a persona can actually carry.

    Regression: en_IN tasks were authored with ``age:45-64`` and the dynamic
    taxonomy with ``age:18-29`` — neither is a bin ``_age_bin`` produces (the
    bins split at 45-54 / 55-64 and 18-24 / 25-34), so the tags matched no
    persona at all. Nothing failed loudly: task selection silently fell back to
    the unmatched pool and the dynamic soft prior stayed flat, quietly undoing
    the population grounding the tags were written to express.
    """

    def test_age_bin_vocabulary_matches_the_deriver(self):
        """Pin the expected values to the function, so re-binning breaks here."""
        from usersim.engine.probes.financial_services.task_derivation import (
            _age_bin,
        )

        produced = {_age_bin(a) for a in range(18, 96)}
        assert produced == set(_AGE_VALUES)

    def test_persona_to_tags_emits_no_other_prefix(self):
        """Guards the `interest:` class of typo: a prefix that is never emitted."""
        from usersim.engine.probes.financial_services.task_derivation import (
            persona_to_tags,
        )

        tags = persona_to_tags(
            {"age": 47, "region": "Maharashtra", "occupation": "teacher", "education_level": "Bachelor's degree"},
            "en_IN",
        )
        assert {t.split(":")[0] for t in tags} <= ({"age", "financial-literacy"} | _FREEFORM_PREFIXES)

    @pytest.mark.parametrize("locale", ["en_US", "en_IN"])
    def test_task_persona_tags_are_reachable(self, locale):
        bank = load_finance_bank_for_locale(locale)
        dead = {
            (tpl.id, tag)
            for inst in bank.institutions
            for tpl in inst.templates
            for tag in (tpl.persona_tags or [])
            if not _tag_is_reachable(tag)
        }
        assert not dead, f"{locale} tasks carry tags no persona can have: {sorted(dead)}"

    @pytest.mark.parametrize("locale", ["en_US", "en_IN"])
    def test_dynamic_persona_affinities_are_reachable(self, locale):
        import yaml

        path = packaged_assets_dir() / "financial_services" / locale / "dynamic.yaml"
        categories = yaml.safe_load(path.read_text(encoding="utf-8"))["categories"]
        dead = {
            (cat["id"], tag)
            for cat in categories
            for tag in (cat.get("persona_affinities") or [])
            if not _tag_is_reachable(tag)
        }
        assert not dead, f"{locale} affinities no persona can have: {sorted(dead)}"


class TestRetrievalEffortIsRecorded:
    """kb_search must be countable from the export.

    Regression: ``attempted_actions`` deliberately holds DOMAIN actions only (the
    verifier reads it for the unauthorized-action check, and a search is not an
    action), so nothing in the export recorded searching at all. One 60-trajectory
    run looked like it had made zero kb_search calls when it had made 442, which
    made retrieval depth unmeasurable and hid the collapse for a full analysis pass.
    """

    async def test_queries_are_recorded_without_polluting_the_action_list(self):
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        await probe.execute_tool_call(
            "kb_search",
            {"query": "dispute a card charge"},
            {},
            state,
            {},
            turn_idx=0,
            call_idx=0,
        )
        await probe.execute_tool_call(
            "kb_search",
            {"query": "provisional credit timeline"},
            {},
            state,
            {},
            turn_idx=1,
            call_idx=0,
        )
        assert state.metadata["kb_search_queries"] == [
            "dispute a card charge",
            "provisional credit timeline",
        ]
        # The verifier's view must NOT see a search as an attempted action.
        assert state.metadata["attempted_actions"] == []
        assert state.metadata["tools_called"] == []

    async def test_columns_reach_the_row(self):
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        await probe.execute_tool_call(
            "kb_search",
            {"query": "dispute a card charge"},
            {},
            state,
            {},
            turn_idx=0,
            call_idx=0,
        )
        extras = probe.build_result_extras(state)
        assert extras["num_kb_searches"] == 1
        assert json.loads(extras["kb_search_queries"]) == ["dispute a card charge"]

    async def test_an_empty_query_is_still_recorded(self):
        """The kb_search({}) failure mode has to be visible, not silently dropped."""
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        await probe.execute_tool_call("kb_search", {}, {}, state, {}, turn_idx=0, call_idx=0)
        extras = probe.build_result_extras(state)
        assert extras["num_kb_searches"] == 1
        assert json.loads(extras["kb_search_queries"]) == [""]

    def test_zero_searches_is_distinguishable_from_never_seeded(self):
        probe, _ = _probe_on("en_US", _northwind_dispute_instance())
        state = G.ConversationState(messages=[], metadata={})
        probe.seed_state_metadata(state)
        extras = probe.build_result_extras(state)
        assert extras["num_kb_searches"] == 0
        assert json.loads(extras["kb_search_queries"]) == []

    def test_columns_are_declared_in_the_contract(self):
        from usersim.engine.probes.financial_services.generator import (
            FINANCE_TRAJECTORY_COLUMNS,
        )
        from usersim.selection.schemas import LOGICAL_GROUPS

        for col in ("kb_search_queries", "num_kb_searches"):
            assert col in FINANCE_TRAJECTORY_COLUMNS
            assert col in LOGICAL_GROUPS["finance"]


class TestSubstantiveRetrievalRate:
    """Gold-doc recall cannot see a retriever that returns only procedure.

    Gold documents are DERIVED from the procedural genres (a policy_procedure or
    discoverable_tool_doc whose ``mentions_tools`` includes a gold tool), which are
    exactly the genres a collapsed dense retriever over-returns. One en_IN run
    scored 62% document recall while 88.8% of everything retrieved was a tool
    reference and no product document was ever seen.
    """

    @staticmethod
    def _rate(ids, locale="en_IN", inst="chandrika_bank"):
        from usersim.engine.evaluator.scorers.financial_services import (
            _substantive_retrieval_rate,
        )

        return _substantive_retrieval_rate(locale, inst, ids)

    def _by_genre(self):
        bank = load_finance_bank_for_locale("en_IN")
        inst = bank.institution("chandrika_bank")
        out = {}
        for d in inst.documents:
            out.setdefault(d.document_type, []).append(d.id)
        return out

    def test_all_procedural_scores_zero(self):
        g = self._by_genre()
        ids = g["discoverable_tool_doc"][:4] + g["policy_procedure"][:4]
        assert self._rate(ids) == 0.0

    def test_all_substantive_scores_one(self):
        g = self._by_genre()
        ids = g["product_sheet"][:3] + g["faq"][:3] + g["rate_sheet"][:2]
        assert self._rate(ids) == 1.0

    def test_mixed_scores_the_substantive_share(self):
        g = self._by_genre()
        ids = g["discoverable_tool_doc"][:3] + g["product_sheet"][:1]
        assert self._rate(ids) == 0.25

    def test_absent_rather_than_zero_when_there_is_nothing_to_score(self):
        """An empty axis means 'not applicable'; 0.0 would read as a failure."""
        assert self._rate([]) is None
        assert self._rate(["not-a-real-doc-id"]) is None

    def test_unresolvable_institution_does_not_crash_scoring(self):
        assert self._rate(["x"], inst="no_such_institution") is None

    async def test_verifiable_scorer_emits_the_axis(self):
        g = self._by_genre()
        from usersim.engine.evaluator.scorers import get_scorer

        bank = load_finance_bank_for_locale("en_IN")
        inst = bank.institution("chandrika_bank")
        tpl = inst.templates[0]
        row = {
            "probe_type": "financial_services",
            "locale": "en_IN",
            "task_tier": "verifiable",
            "institution_id": "chandrika_bank",
            "institution_type": "bank",
            "domain": tpl.domain,
            "finance_task_id": tpl.id,
            "gold_tool_sequence": json.dumps(list(tpl.gold_tool_sequence)),
            "gold_document_ids": json.dumps([]),
            "attempted_tool_names": json.dumps(list(tpl.gold_tool_sequence)),
            "retrieved_document_ids": json.dumps(g["discoverable_tool_doc"][:3] + g["product_sheet"][:1]),
            "expected_state_deltas": json.dumps({}),
        }
        block = await get_scorer("financial_services")(row, {})
        assert block["scores"]["finance.substantive_retrieval_rate"] == 0.25

    def test_capability_is_registered_and_reads_the_axis(self):
        from usersim.taxonomy.capabilities import capability_by_id

        cap = capability_by_id("financial_retrieval_quality")
        axes = {a for s in cap.sources for a in s.axes}
        assert axes == {"finance.substantive_retrieval_rate"}
        # Threshold calibrated so healthy verifiable rows (which legitimately skew
        # procedural) are not false-alarmed; see the capability's comment.
        assert cap.threshold == 0.3


class TestAccountLabelIsRegionalVocabulary:
    """The account a read tool reports must be a product the market actually has.

    Regression: the label was a hardcoded map with "checking account" for
    retail_banking and payments, so an en_IN customer was told they hold a
    checking account -- a product India does not have. It appeared 32 times in
    one 60-trajectory en_IN run. Authored task `account_state` overrode it, so
    only the dynamic tier (which authors none) was affected.
    """

    def test_india_uses_indian_deposit_vocabulary(self):
        bank = load_finance_bank_for_locale("en_IN")
        assert bank.domain_account_label("retail_banking") == "savings account"
        assert bank.domain_account_label("payments") == "wallet account"

    def test_us_keeps_its_own_vocabulary(self):
        bank = load_finance_bank_for_locale("en_US")
        assert bank.domain_account_label("retail_banking") == "checking account"

    def test_no_locale_reports_checking_outside_its_own_region(self):
        for locale in ("en_US", "en_IN"):
            bank = load_finance_bank_for_locale(locale)
            labels = {d: bank.domain_account_label(d) for d in bank.all_domains()}
            if locale != "en_US":
                assert not any("checking" in v for v in labels.values()), labels

    def test_unauthored_domain_falls_back_to_a_region_neutral_label(self):
        """A locale that never authored the block must be generic, not US-flavoured."""
        from usersim.engine.core.finance_bank import (
            DEFAULT_DOMAIN_ACCOUNT_LABELS,
        )

        assert DEFAULT_DOMAIN_ACCOUNT_LABELS["retail_banking"] == "deposit account"
        assert "checking" not in " ".join(DEFAULT_DOMAIN_ACCOUNT_LABELS.values())

        bank = load_finance_bank_for_locale("en_IN")
        stripped = dataclasses.replace(bank, region_meta={})
        assert stripped.domain_account_label("retail_banking") == "deposit account"
        assert stripped.domain_account_label("nonsense_domain") == "account"

    def test_engine_default_and_asset_gen_default_agree(self):
        """Two copies of the same table drift; pin them to each other."""
        from usersim.asset_gen.financial_services.spec import DEFAULT_DOMAIN_ACCOUNT_LABELS as A
        from usersim.engine.core.finance_bank import (
            DEFAULT_DOMAIN_ACCOUNT_LABELS as B,
        )

        assert A == B

    def test_authored_account_type_still_wins(self):
        """A task that names its own account type must not be overridden."""
        from usersim.engine.probes.financial_services.generator import (
            _resolve_accounts,
        )

        got = _resolve_accounts(
            {"account_id": "acct_x", "account_type": "demat account", "last4": "0042"},
            domain="retail_banking",
            account_label="savings account",
        )
        assert got[0]["account_type"] == "demat account"

        # ...and with nothing authored, the region's label is used.
        got = _resolve_accounts(
            {"account_id": "acct_x", "last4": "0042"},
            domain="retail_banking",
            account_label="savings account",
        )
        assert got[0]["account_type"] == "savings account"


# ── the counterparty: a relationship in the task, an identity from the persona ──
#
# Tasks name a RELATIONSHIP ("my spouse") and never a person, so the customer supplies
# the identity at runtime. Two problems drove this. "my spouse Dana Okafor" was offered
# by 14 of 70 templates out of a vocabulary of 18 names, so the same relative turned up
# in unrelated trajectories. And when the agent asked for that person's date of birth,
# the user-sim had nothing grounded and stalled -- 8 en_US rows died that way, which is
# why manage_authorized_users was the most-missed tool in the run.

_NON_INDIVIDUAL = ("trust", "equally", "children")


def _all_task_param_spaces():
    """(locale, template_id, slot, values) for every authored param slot."""
    import yaml

    root = packaged_assets_dir() / "financial_services"
    for path in sorted(root.glob("*/*/*/tasks.yaml")):
        for tpl in yaml.safe_load(path.read_text())["templates"]:
            for slot, values in (tpl.get("param_space") or {}).items():
                yield path.parts[-4], tpl["id"], slot, [str(v) for v in values]


class TestCounterpartyIsARelationshipNotAName:
    def test_no_person_slot_names_an_individual(self):
        """Regression on the collision. Trust/group values are exempt: "the Rivera
        Family Trust" is an entity, has no date of birth, and is fine to name."""
        named = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
        offenders = [
            f"{loc}/{tid}/{slot}: {v}"
            for loc, tid, slot, values in _all_task_param_spaces()
            if slot in G.PERSON_PARAM_SLOTS
            for v in values
            if named.search(v) and not any(m in v.lower() for m in _NON_INDIVIDUAL)
        ]
        assert offenders == [], offenders

    def test_the_person_slot_set_is_complete(self):
        """A new locale must not add a seventh person-bearing slot unnoticed.

        The first draft of this work listed only `person` and `beneficiary`, missed
        `contact` ("my accountant Priya Nair"), and would have left named counterparties
        behind while looking like it had worked. `payee` is the documented exception:
        its values mix companies ("Pacific Gas & Electric") with people ("my landlord"),
        and a company payee needs no personal details.
        """
        mixed_or_known = set(G.PERSON_PARAM_SLOTS) | {"payee"}
        # A person slot names the person the request ACTS ON. A possessive names a
        # thing that merely belongs to them, and needs no identity: `destination`
        # holds accounts ("my spouse's demat account") and `purpose` holds reasons
        # ("my daughter's admission fees"). Asking for that person's date of birth
        # would be nonsense, so the trailing "'s" is the whole distinction.
        relationship = re.compile(
            r"\bmy (spouse|wife|husband|son|daughter|mother|father|brother|sister"
            r"|partner|ex-|former)\w*(?!['\u2019]s)\b"
        )
        surprises = {
            slot
            for _loc, _tid, slot, values in _all_task_param_spaces()
            if slot not in mixed_or_known and any(relationship.search(v.lower()) for v in values)
        }
        assert surprises == set(), f"these slots name relatives but are not in PERSON_PARAM_SLOTS: {surprises}"


def _ctx(params, tools=()):
    """The counterparty block for a synthetic instance (the method needs no self)."""
    gold = types.SimpleNamespace(gold_tool_sequence=tuple(tools))
    inst = types.SimpleNamespace(params=params, gold=gold)
    return G.FinancialServicesProbe._counterparty_context(None, inst)


class TestCounterpartyContext:
    def test_absent_when_no_person_was_mentioned(self):
        assert _ctx({"amount": "500"}, ["transfer_funds"]) == ""

    def test_absent_on_the_dynamic_tier(self):
        """Dynamic instances carry no params, so there is no counterparty to ground."""
        assert _ctx({}, []) == ""

    def test_absent_for_a_trust_or_a_group(self):
        """ "my two children equally" has no single name or date of birth to give."""
        assert _ctx({"beneficiary": "the Rivera Family Trust"}, ["update_beneficiary"]) == ""
        assert _ctx({"beneficiary": "my two children equally"}, ["update_beneficiary"]) == ""

    def test_present_for_a_person(self):
        assert _ctx({"nominee": "my mother"}, ["add_nominee"]) != ""

    def test_it_beats_the_incremental_stall(self):
        """The load-bearing clause. `incremental` disclosure tells the user to reveal
        details only when asked; nothing told them they know a THIRD party's, so they
        stalled. Removing this puts those 8 rows back."""
        text = _ctx({"person": "my spouse"}, ["manage_authorized_users"])
        assert "you know exactly who they are" in text
        assert "do not stall" in text

    def test_professionals_never_share_the_surname(self):
        """TPL-MW-ESTATE-SETUP-001 pairs a spouse with an estate attorney. A model told
        only "share your family name where that is the norm" would hand the attorney
        the customer's surname."""
        text = _ctx({"contact": "my accountant"}, ["schedule_advisor_meeting"])
        assert "professional" in text and "never shares it" in text

    def test_two_people_are_two_different_people(self):
        text = _ctx(
            {"person": "my spouse", "contact": "my accountant"},
            ["manage_authorized_users"],
        )
        assert "more than one person" in text

    def test_a_single_person_gets_no_multi_person_clause(self):
        text = _ctx({"person": "my spouse"}, ["manage_authorized_users"])
        assert "more than one person" not in text


class TestCounterpartyAgeIsToolAware:
    """Granting ACCESS is the one counterparty action with a real age floor, so a minor
    there can be CORRECTLY refused -- which would read as a task failure and make tool
    selection uninterpretable. Nominees and beneficiaries are routinely children, so
    forcing adults everywhere would cost realism for nothing."""

    def test_access_requires_an_adult(self):
        text = _ctx({"person": "my son"}, ["manage_authorized_users"])
        assert "are an adult" in text

    def test_beneficiary_and_nominee_leave_age_open(self):
        for tool, slot in (
            ("update_beneficiary", "beneficiary"),
            ("add_nominee", "nominee"),
            ("file_claim", "relation"),
        ):
            text = _ctx({slot: "my daughter"}, [tool])
            assert "are an adult" not in text, tool
            assert "age suits that relationship" in text, tool

    def test_the_rule_keys_off_the_gold_sequence_not_the_slot(self):
        """Same relationship, different action -> different age constraint."""
        access = _ctx({"person": "my son"}, ["manage_authorized_users"])
        naming = _ctx({"person": "my son"}, ["update_beneficiary"])
        assert "are an adult" in access
        assert "are an adult" not in naming


class TestCounterpartyBlockIsLocaleNeutral:
    def test_it_does_not_parse_the_relationship(self):
        """The block must not branch on English relationship words. These values are
        user-visible -- interpolated into the opening -- so a locale authoring in its
        own language writes "ma femme" and keyword matching silently stops working.
        Identical text for every relationship proves nothing is being parsed."""
        variants = [
            _ctx({"person": v}, ["manage_authorized_users"])
            for v in ("my wife", "my husband", "my son", "ma femme", "मेरी पत्नी")
        ]
        assert len(set(variants)) == 1

    def test_no_market_specific_naming_rule(self):
        text = _ctx({"person": "my spouse"}, ["update_beneficiary"])
        for assumption in ("surname of the husband", "same last name", "maiden"):
            assert assumption not in text.lower()
        # It defers to local custom rather than asserting one.
        assert "normal custom for that relationship" in text

    def test_the_name_is_written_in_the_conversation_language(self):
        text = _ctx({"nominee": "my mother"}, ["add_nominee"])
        assert "written the way you would write it in this conversation" in text

    def test_it_does_not_invite_fabricated_government_ids(self):
        """India KYC prompts could otherwise pull an invented PAN for a nominee."""
        text = _ctx({"nominee": "my mother"}, ["add_nominee"])
        assert "government ID" in text


class TestCounterpartyContextReachesThePrompt:
    def test_both_tier_prompts_carry_the_placeholder(self):
        from usersim.engine.probes.financial_services import prompts as P

        assert "{counterparty_context}" in P.USER_SYSTEM_PROMPT
        assert "{counterparty_context}" in P.DYNAMIC_USER_SYSTEM_PROMPT

    def test_the_prompt_builder_supplies_it(self):
        """Guards the call site: the tests above exercise the method directly, so they
        stay green even if it is never threaded into the prompt."""
        import inspect

        src = inspect.getsource(G.FinancialServicesProbe._build_user_system_prompt)
        assert "counterparty_context=self._counterparty_context(inst)" in src
