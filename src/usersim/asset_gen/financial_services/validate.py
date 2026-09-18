# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline vetting harness for a generated financial_services bank.

The generate + vet GATE (see the ``wellposed`` bucket): after a small live
generation, this checks the produced CORPUS before anything downstream builds on
it. Pure/offline -- reads the committed YAML + the ``embeddings.parquet`` sidecar
+ the ``_generation_report.json`` provenance file; no LLM/network.

Generation is corpus-only (``write_tasks=False``), while the runtime loader
requires ``tasks.yaml`` -- so the SEMANTIC checks read the bank directly, and the
structural ``load_finance_bank`` round-trip is reported as skipped (not failed)
for a corpus-only bank. Reference-agent solvability is deferred to task-gen.

Checks: structural_load, numeric_faithfulness (the key one -- spec typed values
appear in the prose), language_script, near_duplicate, coverage,
placeholder_discipline, scope, qc_scores.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from usersim.asset_gen.financial_services.spec import RegionSpec

#: cosine >= this within an institution flags a near-duplicate pair.
NEAR_DUP_THRESHOLD = 0.97
#: min fraction of letter codepoints in the locale's expected script.
MIN_SCRIPT_FRACTION = 0.6
#: QC faithfulness floor (0-4); clusters below are kept but flagged.
MIN_FAITHFULNESS_SCORE = 3

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class CheckResult:
    check: str
    scope: str  # institution id, or "bank"
    status: str  # PASS | WARN | FAIL
    detail: str = ""


@dataclass
class ValidationReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, check: str, scope: str, status: str, detail: str = "") -> None:
        self.results.append(CheckResult(check, scope, status, detail))

    @property
    def failed(self) -> bool:
        return any(r.status == FAIL for r in self.results)

    def counts(self) -> dict[str, int]:
        out = {PASS: 0, WARN: 0, FAIL: 0}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    def to_text(self) -> str:
        lines: list[str] = []
        by_scope: dict[str, list[CheckResult]] = {}
        for r in self.results:
            by_scope.setdefault(r.scope, []).append(r)
        for scope in sorted(by_scope):
            lines.append(f"  [{scope}]")
            for r in by_scope[scope]:
                lines.append(f"    {r.status:<4} {r.check}: {r.detail}")
        c = self.counts()
        lines.append(
            f"  summary: {c[PASS]} pass / {c[WARN]} warn / {c[FAIL]} fail -> {'FAIL' if self.failed else 'OK'}"
        )
        return "\n".join(lines)


# ── bank reading (corpus-only tolerant; no loader dependency) ───────────


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class _InstData:
    meta: dict[str, Any]
    documents: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    embeddings: Any  # pandas.DataFrame | None
    has_tasks: bool


def _read_bank(bank_dir: Path) -> tuple[dict[str, Any], dict[str, _InstData]]:
    region_meta = _load_yaml(bank_dir / "region_meta.yaml")
    institutions: dict[str, _InstData] = {}
    # Institutions live under a type dir (<type>/<id>/); discover layout-agnostically
    # via institution_meta.yaml (also tolerates a legacy flat layout).
    for meta_path in sorted(bank_dir.rglob("institution_meta.yaml")):
        idir = meta_path.parent
        meta = _load_yaml(meta_path)
        corpus = _load_yaml(idir / "corpus.yaml") if (idir / "corpus.yaml").exists() else {}
        tools = _load_yaml(idir / "tools.yaml") if (idir / "tools.yaml").exists() else {}
        emb = None
        emb_path = idir / "embeddings.parquet"
        if emb_path.exists():
            import pandas as pd

            emb = pd.read_parquet(emb_path)
        institutions[idir.name] = _InstData(
            meta=meta,
            documents=list(corpus.get("documents", []) or []),
            tools=list(tools.get("tools", []) or []),
            embeddings=emb,
            has_tasks=(idir / "tasks.yaml").exists(),
        )
    return region_meta, institutions


# ── individual checks ────────────────────────────────────────────────────


def _check_structural(bank_dir: Path, insts: dict[str, _InstData], rep: ValidationReport) -> None:
    corpus_only = any(not d.has_tasks for d in insts.values())
    try:
        from usersim.engine.core.finance_bank import (
            FinanceBankError,
            load_finance_bank,
        )
    except Exception as e:  # pragma: no cover - plugin import issue
        rep.add("structural_load", "bank", WARN, f"loader unavailable: {e}")
        return
    try:
        load_finance_bank(bank_dir)
        rep.add("structural_load", "bank", PASS, "loads via load_finance_bank")
    except FinanceBankError as e:
        if corpus_only:
            rep.add("structural_load", "bank", WARN, "corpus-only bank (no tasks.yaml) -> full structural load skipped")
        else:
            rep.add("structural_load", "bank", FAIL, str(e))


def _indian_grouped(iv: int) -> str:
    """``250000`` -> ``'2,50,000'`` (Indian digit grouping).

    The Indian system groups the last three digits, then in pairs, so a doc
    written for an Indian audience naturally renders a typed value in a form the
    Western ``{:,}`` grouping never produces.
    """
    sign = "-" if iv < 0 else ""
    digits = str(abs(iv))
    if len(digits) <= 3:
        return f"{sign}{digits}"
    head, tail = digits[:-3], digits[-3:]
    pairs: list[str] = []
    while len(head) > 2:
        pairs.insert(0, head[-2:])
        head = head[:-2]
    if head:
        pairs.insert(0, head)
    return f"{sign}{','.join(pairs)},{tail}"


def _scale_word_forms(iv: int) -> set:
    """Indian scale-word renderings of a value (``250000`` -> ``'2.5 lakh'``).

    Indian financial prose quotes large amounts in lakh (1e5) and crore (1e7)
    rather than in full digits, so the exact typed value can be faithfully
    present without its digit string appearing at all.
    """
    out: set = set()
    for unit, scale in (("lakh", 100_000), ("crore", 10_000_000)):
        if abs(iv) < scale:
            continue
        q = iv / scale
        # Only the clean renderings a writer would actually use.
        if abs(q * 100 - round(q * 100)) > 1e-9:
            continue
        text = f"{q:.2f}".rstrip("0").rstrip(".")
        out.update({f"{text} {unit}", f"{text} {unit}s"})
    return out


def _value_candidates(val: Any) -> list[str]:
    cands = {str(val)}
    if isinstance(val, bool):
        return list(cands)
    if isinstance(val, int) or (isinstance(val, float) and float(val).is_integer()):
        iv = int(val)
        # Plain digits, Western grouping, and Indian grouping / scale words, so a
        # locale-natural rendering counts as faithful (a US corpus will never
        # contain "lakh", so this cannot mask a wrong number).
        cands.update({str(iv), f"{iv:,}", _indian_grouped(iv)})
        cands.update(_scale_word_forms(iv))
    if isinstance(val, float):
        cands.update({f"{val}", f"{val:.2f}", f"{val:,.2f}"})
    return [c for c in cands if c]


def _value_present(val: Any, text: str) -> bool:
    return any(c in text for c in _value_candidates(val))


def _check_numeric_faithfulness(spec: RegionSpec | None, insts: dict[str, _InstData], rep: ValidationReport) -> None:
    if spec is None:
        rep.add("numeric_faithfulness", "bank", WARN, "skipped: no region_spec provided (cannot ground values)")
        return
    for inst in spec.institutions:
        idata = insts.get(inst.id)
        if idata is None:
            rep.add("numeric_faithfulness", inst.id, FAIL, "institution missing from bank")
            continue
        for product in inst.products:
            prod_docs = [d for d in idata.documents if d.get("product_category") == product.id]
            text = " ".join(f"{d.get('title', '')} {d.get('body', '')}" for d in prod_docs)
            missing = [
                f"{name}={var.value}"
                for name, var in product.variables.items()
                if isinstance(var.value, (int, float)) and not _value_present(var.value, text)
            ]
            if not prod_docs:
                rep.add("numeric_faithfulness", inst.id, FAIL, f"{product.id}: no documents produced for product")
            elif missing:
                rep.add(
                    "numeric_faithfulness", inst.id, FAIL, f"{product.id}: typed values absent from prose: {missing}"
                )
            else:
                rep.add("numeric_faithfulness", inst.id, PASS, f"{product.id}: all typed values present")


def _check_variable_name_leak(spec: RegionSpec | None, insts: dict[str, _InstData], rep: ValidationReport) -> None:
    """Flag customer-facing prose that prints a spec FIELD KEY.

    The doc-gen prompt hands the model each product's variables as a JSON object
    and demands the values verbatim; a model that also echoes the KEY produces
    "an interest_rate_annual of 0%", which reads like an unfilled template. The
    numeric check cannot catch it (the number IS right), and the per-doc judge
    only registers it as slightly lower realism, so check it deterministically.
    """
    if spec is None:
        rep.add("variable_name_leak", "bank", WARN, "skipped: no region_spec provided (cannot know the field names)")
        return
    for inst in spec.institutions:
        idata = insts.get(inst.id)
        if idata is None:
            continue
        keys = {k for p in inst.products for k in p.variables}
        if not keys:
            continue
        pattern = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in keys), key=len, reverse=True)) + r")\b")
        offenders: list[str] = []
        seen_keys: set = set()
        for d in idata.documents:
            found = pattern.findall(f"{d.get('title', '')} {d.get('body', '')}")
            if found:
                offenders.append(str(d.get("id")))
                seen_keys.update(found)
        if offenders:
            rep.add(
                "variable_name_leak",
                inst.id,
                WARN,
                f"{len(offenders)}/{len(idata.documents)} doc(s) print a raw "
                f"field name in prose (e.g. {sorted(seen_keys)[:4]}); "
                f"regenerate or hand-fix: {offenders[:5]}",
            )
        else:
            rep.add("variable_name_leak", inst.id, PASS, "no raw field names in prose")


#: Genres written FOR THE AGENT, where naming a tool is the point. Everything else in
#: the corpus is customer-facing prose and must not name one.
_AGENT_FACING_GENRES: frozenset = frozenset(
    {
        "policy_procedure",  # numbered agent steps referencing the exact tool
        "discoverable_tool_doc",  # the tool's signature, args and effect
        "regulatory_note",  # "the applicable rule and the AGENT's obligation"
    }
)


def _check_tool_name_leak(insts: dict[str, _InstData], rep: ValidationReport) -> None:
    """Flag customer-facing prose that names an internal tool.

    A customer-facing document should say what the customer can ask for; which tool
    serves it is an internal detail belonging to the agent-facing set. Naming one reads
    like a leaked runbook, and the per-doc judge only registers it as lower realism.

    Long-standing rather than new: measured at ~10% of how_to_guide + troubleshooting
    docs in BOTH locales before any genre conventions existed. Deterministic to check
    because every tool name is snake_case, so an exact-token match cannot fire on
    ordinary prose ("close account" does not match ``close_account``).
    """
    for iid, idata in sorted(insts.items()):
        names = {str(t.get("name")) for t in idata.tools if t.get("name")}
        names = {n for n in names if "_" in n}
        if not names:
            continue
        pattern = re.compile(r"\b(" + "|".join(sorted((re.escape(n) for n in names), key=len, reverse=True)) + r")\b")
        customer = [d for d in idata.documents if d.get("document_type") not in _AGENT_FACING_GENRES]
        offenders: list[str] = []
        seen: set = set()
        for d in customer:
            found = pattern.findall(f"{d.get('title', '')} {d.get('body', '')}")
            if found:
                offenders.append(str(d.get("id")))
                seen.update(found)
        if not customer:
            # Emit a row anyway: a silent check reads as a check that did not run.
            rep.add("tool_name_leak", iid, PASS, "no customer-facing documents to check")
            continue
        if offenders:
            rep.add(
                "tool_name_leak",
                iid,
                WARN,
                f"{len(offenders)}/{len(customer)} customer-facing doc(s) name an "
                f"internal tool in prose (e.g. {sorted(seen)[:4]}); those belong in "
                f"the agent-facing policy/tool docs: {offenders[:5]}",
            )
        else:
            rep.add("tool_name_leak", iid, PASS, "no internal tool names in customer-facing prose")


#: A generated doc_key looks like ``<product>_<genre>[_<n>]`` or ``<tool>_policy`` /
#: ``<tool>_tooldoc``. Quoting one in prose is meaningless to a reader (it is an
#: internal slug) and in practice comes with deferring the substance the document owes.
_DOC_KEY_RE = re.compile(
    r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_(?:faq|product_sheet|fee_schedule|how_to_guide|"
    r"troubleshooting|comparison|disclosure|eligibility_matrix|rate_sheet|promo_notice|"
    r"security_advisory|terms_conditions|policy|tooldoc)(?:_\d+)?\b"
    r"|\bfaq_\d+\b"
)


def _check_doc_key_leak(insts: dict[str, _InstData], rep: ValidationReport) -> None:
    """Flag prose that cites a sibling by its raw ``doc_key`` instead of its title.

    The prompt asks for cross-references BY TITLE, with keys confined to the
    ``references`` field. When a body says "see 1-Year Fixed Deposit - faq_5" the
    reader gets an internal slug and the document has deferred its own substance:
    such docs average 2.38 on the judge's self-containment axis against 2.78 for
    clean ones. Rare (~0.7% of documents) but unambiguous, so worth a deterministic
    check rather than leaving it to the judge.
    """
    for iid, idata in sorted(insts.items()):
        # A tool doc's key is ``<tool>_policy`` / ``<tool>_tooldoc``, so a tool whose
        # OWN name ends in _policy (renew_policy, update_policy) matches the key shape.
        # Those are tool-name leaks, reported by _check_tool_name_leak; excluding them
        # here keeps one string from being blamed under two different checks.
        tool_names = {str(t.get("name")) for t in idata.tools if t.get("name")}
        offenders: list[str] = []
        seen: set = set()
        for d in idata.documents:
            found = [m for m in _DOC_KEY_RE.findall(str(d.get("body", ""))) if m not in tool_names]
            if found:
                offenders.append(str(d.get("id")))
                seen.update(found)
        if offenders:
            rep.add(
                "doc_key_leak",
                iid,
                WARN,
                f"{len(offenders)}/{len(idata.documents)} doc(s) cite a sibling by "
                f"raw doc_key instead of its title (e.g. {sorted(seen)[:4]}); "
                f"regenerate or hand-fix: {offenders[:5]}",
            )
        else:
            rep.add("doc_key_leak", iid, PASS, "cross-references use titles, not doc_keys")


def _in_ranges(cp: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def _script_fraction(text: str, ranges: tuple[tuple[int, int], ...]) -> float | None:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return None
    in_expected = sum(1 for c in letters if _in_ranges(ord(c), ranges))
    return in_expected / len(letters)


def _check_language_script(region_meta: dict[str, Any], insts: dict[str, _InstData], rep: ValidationReport) -> None:
    locale = str(region_meta.get("locale", ""))
    try:
        from usersim.engine.core.locale import expected_script_ranges

        ranges = expected_script_ranges(locale)
    except Exception as e:  # pragma: no cover
        rep.add("language_script", "bank", WARN, f"locale util unavailable: {e}")
        return
    if not ranges:
        rep.add("language_script", "bank", WARN, f"locale {locale!r} not registered")
        return
    for iid, idata in insts.items():
        bad = []
        for d in idata.documents:
            frac = _script_fraction(f"{d.get('title', '')} {d.get('body', '')}", ranges)
            if frac is not None and frac < MIN_SCRIPT_FRACTION:
                bad.append(f"{d.get('id')}={frac:.2f}")
        if bad:
            rep.add(
                "language_script", iid, FAIL, f"docs below {MIN_SCRIPT_FRACTION} expected-script fraction: {bad[:5]}"
            )
        else:
            rep.add("language_script", iid, PASS, "prose in expected script")


def _check_near_duplicate(insts: dict[str, _InstData], rep: ValidationReport) -> None:
    import numpy as np

    for iid, idata in insts.items():
        emb = idata.embeddings
        if emb is None:
            rep.add("near_duplicate", iid, WARN, "no embeddings.parquet sidecar")
            continue
        if len(emb) < 2:
            rep.add("near_duplicate", iid, PASS, "fewer than 2 embedded docs")
            continue
        ids = list(emb["id"])
        vecs = np.array([np.asarray(v, dtype=float) for v in emb["embedding"]])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sim = (vecs / norms) @ (vecs / norms).T
        dups = [
            (ids[i], ids[j], round(float(sim[i, j]), 3))
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
            if sim[i, j] >= NEAR_DUP_THRESHOLD
        ]
        if dups:
            rep.add(
                "near_duplicate", iid, WARN, f"{len(dups)} near-duplicate pair(s) >= {NEAR_DUP_THRESHOLD}: {dups[:5]}"
            )
        else:
            rep.add("near_duplicate", iid, PASS, "no near-duplicates")


def _check_coverage(insts: dict[str, _InstData], rep: ValidationReport) -> None:
    if not insts:
        rep.add("coverage", "bank", FAIL, "no institutions in bank")
        return
    for iid, idata in insts.items():
        n = len(idata.documents)
        rep.add("coverage", iid, FAIL if n == 0 else PASS, "empty corpus" if n == 0 else f"{n} documents")


def _check_placeholder(insts: dict[str, _InstData], rep: ValidationReport) -> None:
    for iid, idata in insts.items():
        non_ph = [d.get("id") for d in idata.documents if not d.get("placeholder", False)]
        if non_ph:
            rep.add("placeholder_discipline", iid, WARN, f"{len(non_ph)} doc(s) not flagged placeholder: {non_ph[:5]}")
        else:
            rep.add("placeholder_discipline", iid, PASS, "all docs placeholder:true")


def _check_scope(insts: dict[str, _InstData], rep: ValidationReport) -> None:
    for iid, idata in insts.items():
        domains = set(idata.meta.get("domains", []) or [])
        bad = [
            f"{d.get('id')}:{d.get('domain')}"
            for d in idata.documents
            if d.get("domain") and d.get("domain") not in domains
        ]
        if bad:
            rep.add("scope", iid, FAIL, f"docs with out-of-institution domain: {bad[:5]}")
        else:
            rep.add("scope", iid, PASS, "all docs within institution domains")


def _check_tool_schemas(insts: dict[str, _InstData], rep: ValidationReport) -> None:
    """Framework primitives (kb_search / verify_identity) must ship a real schema.

    An empty schema (no description / no parameters) leaves the assistant calling
    kb_search({}) with no query — it retrieves nothing and never unlocks the
    discoverable gold tools, so every verifiable task fails. The generator now
    backfills these from ``core.finance_bank.FRAMEWORK_PRIMITIVE_TOOLS``; this
    check guards against a bank that predates the backfill or bypasses it.
    """
    try:
        from usersim.engine.core.finance_bank import FRAMEWORK_PRIMITIVE_TOOLS
    except Exception:  # pragma: no cover - plugin import issue
        rep.add("tool_schemas", "bank", WARN, "could not import primitive contract")
        return
    for iid, idata in insts.items():
        bad = []
        for t in idata.tools:
            name = t.get("name")
            if name not in FRAMEWORK_PRIMITIVE_TOOLS:
                continue
            params = t.get("parameters") or {}
            if not t.get("description") or not (params.get("properties") or params):
                bad.append(name)
        if bad:
            rep.add(
                "tool_schemas",
                iid,
                FAIL,
                f"framework primitive(s) with empty schema: {bad} (regenerate: the pipeline backfills these)",
            )
        else:
            rep.add("tool_schemas", iid, PASS, "framework primitives have real schemas")

        # Every DISCOVERABLE domain tool should be documented by a
        # policy_procedure / discoverable_tool_doc (gold-doc derivation needs
        # one, and the tool can't be authored into a verifiable task without it).
        # A generation drop leaves a tool undocumented -> WARN so it's caught
        # before task-authoring instead of a load-time error.
        tool_doc_types = {"policy_procedure", "discoverable_tool_doc"}
        documented = set()
        for doc in idata.documents:
            if doc.get("document_type") in tool_doc_types:
                documented.update(doc.get("mentions_tools") or [])
        undocumented = [
            t.get("name")
            for t in idata.tools
            if t.get("discoverable")
            and t.get("name") not in FRAMEWORK_PRIMITIVE_TOOLS
            and t.get("name") not in documented
        ]
        if undocumented:
            rep.add(
                "tool_doc_coverage",
                iid,
                WARN,
                f"discoverable tool(s) with no policy/tool doc: {undocumented} "
                "(regenerate to restore; cannot be a verifiable-task gold tool)",
            )
        else:
            rep.add("tool_doc_coverage", iid, PASS, "every discoverable tool has a policy/tool doc")


#: Identifiers a tool doc must not require, because nothing in the conversation can
#: supply them. ``account_id`` and ``card_id`` are deliberately absent: ``get_accounts``
#: resolves those from a last-4, so they have a source.
_UNSOURCED_ID_RE = re.compile(
    r"\b(customer_id|client_id|user_id|member_id|person_id|payee_id"
    r"|beneficiary_id|nominee_id|recipient_id)\b"
)

#: The genres that document a tool's signature for the agent.
_TOOL_DOC_GENRES: frozenset = frozenset({"discoverable_tool_doc", "policy_procedure"})


def _check_unsourced_tool_arguments(insts, rep: ValidationReport) -> None:
    """Flag tool docs whose signature demands an id nobody can provide.

    The doc IS the schema here -- ``tools.yaml`` carries empty parameters on purpose so
    the agent has to retrieve the doc to learn the signature (the discoverable-tool
    mechanic). That makes an invented argument authoritative: the agent believes it,
    asks the customer for a ``user_id`` they cannot possibly know, and a completable
    task dies on a documentation artifact rather than on the model's reasoning.

    Seen for real: solstice_invest's ``manage_authorized_users`` doc specified
    ``user_id: string`` for the spouse being added, so the agent collected the spouse's
    name, DOB and phone, then refused to act for want of an id -- while northwind's and
    meridian's versions of the same tool asked for ``user_details`` and worked.
    """
    for iid, idata in insts.items():
        offenders = []
        for doc in idata.documents:
            if doc.get("document_type") not in _TOOL_DOC_GENRES:
                continue
            found = sorted(set(_UNSOURCED_ID_RE.findall(doc.get("body") or "")))
            if found:
                offenders.append((doc.get("id"), found))
        if offenders:
            ids = [d for d, _ in offenders][:5]
            names = sorted({n for _, f in offenders for n in f})
            rep.add(
                "unsourced_tool_argument",
                iid,
                WARN,
                f"{len(offenders)} tool doc(s) require an id with no source in the "
                f"conversation (e.g. {names}); the agent will ask the customer for it "
                f"and stall: {ids}",
            )
        else:
            rep.add("unsourced_tool_argument", iid, PASS, "tool arguments are all obtainable in conversation")


def _check_qc_scores(bank_dir: Path, rep: ValidationReport) -> None:
    path = bank_dir / "_generation_report.json"
    if not path.exists():
        rep.add("qc_scores", "bank", WARN, "no _generation_report.json (provenance/QC unknown)")
        return
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        rep.add("qc_scores", "bank", FAIL, f"unreadable _generation_report.json: {e}")
        return
    low = []
    for iid, idata in (report.get("institutions", {}) or {}).items():
        for c in idata.get("clusters", []) or []:
            f = c.get("numeric_faithfulness")
            if f is not None and f < MIN_FAITHFULNESS_SCORE:
                low.append(f"{iid}/{c.get('cluster_kind')}={f}")
    if low:
        rep.add(
            "qc_scores",
            "bank",
            WARN,
            f"clusters below faithfulness floor {MIN_FAITHFULNESS_SCORE} (kept flagged): {low[:8]}",
        )
    else:
        rep.add("qc_scores", "bank", PASS, "all clusters meet the faithfulness floor")


# ── entry point ─────────────────────────────────────────────────────────


def validate_bank(bank_dir: str | Path, region_spec: RegionSpec | None = None) -> ValidationReport:
    """Vet a generated financial_services bank; returns a :class:`ValidationReport`.

    ``region_spec`` is the source of truth for the numeric-faithfulness check; if
    omitted that check is skipped (WARN).
    """
    bank_dir = Path(bank_dir)
    rep = ValidationReport()
    if not bank_dir.is_dir():
        rep.add("structural_load", "bank", FAIL, f"bank dir not found: {bank_dir}")
        return rep
    region_meta, insts = _read_bank(bank_dir)

    _check_structural(bank_dir, insts, rep)
    _check_coverage(insts, rep)
    _check_numeric_faithfulness(region_spec, insts, rep)
    _check_variable_name_leak(region_spec, insts, rep)
    _check_tool_name_leak(insts, rep)
    _check_doc_key_leak(insts, rep)
    _check_language_script(region_meta, insts, rep)
    _check_near_duplicate(insts, rep)
    _check_placeholder(insts, rep)
    _check_scope(insts, rep)
    _check_tool_schemas(insts, rep)
    _check_unsourced_tool_arguments(insts, rep)
    _check_qc_scores(bank_dir, rep)
    return rep
