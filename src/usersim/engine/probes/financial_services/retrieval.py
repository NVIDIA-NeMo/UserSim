# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieval backends for the financial_services probe.

Whole-document retrieval only (no chunking in v1). Backends behind one
interface, all SCOPED to a single institution (retrieval never crosses
institutions):

- ``hybrid`` — DEFAULT. Dense cosine blended with lexical overlap, each
  min-max normalised over the institution's documents. Pure dense collapses on
  this corpus (see below); the lexical term restores topical discrimination
  while the dense term still supplies paraphrase matching.
- ``dense``  — cosine alone over the precomputed per-doc embeddings shipped in
  the institution's ``embeddings.parquet`` sidecar, ranked against the query
  vector the caller embedded with the same model. Retained as an ABLATION (it is
  known-weak here) rather than as a recommended mode.
- ``golden`` — inject the task's gold documents directly (isolates reasoning
  from retrieval; needs no embedding endpoint).

All modes degrade to the deterministic lexical-overlap fallback when no query
vector is supplied (no embedding endpoint) or the institution ships no
embeddings, so offline runs still work.

Why hybrid is the default
-------------------------
Measured on the en_IN corpus over the 442 ``kb_search`` calls of one 60-trajectory
run: pure dense returned ``discoverable_tool_doc`` for 88.8% of results, against
an 11.3% corpus share, and its result sets barely depended on the query (pairwise
Jaccard 0.41-0.57 across unrelated queries, vs 0.06-0.08 for lexical). On a
labelled subset, precision@8 for on-topic substantive documents was 4% for dense,
17% for dense over ``title + body``, 88% for hybrid, and 100% for lexical alone.

The cause is corpus homogeneity, not a misconfiguration: ~170 documents about six
products at one institution in one register span a cosine range of roughly
0.58-0.77, so ranking turns on differences of 0.02-0.05. Tool references win that
margin systematically because they are short (104 median words vs 219 for a
product sheet, so less diluted) and their bodies are dense with the
account-action vocabulary every query contains. Query/passage ``input_type`` was
ruled out — the hub ignores the parameter (cosine 0.9999 between modes).

Both return whole ``Document`` objects and never mutate the bank.

Lexical tokens are Unicode words
--------------------------------
A token is a run of letters, combining marks and decimal digits, after case
folding and NFC normalisation. Accented French and German words, ``ß`` (folded
to ``ss``) and Indic scripts therefore tokenize whole, and composed and
decomposed accents compare equal. Other number characters, such as ``½``,
separate words, so ``59½`` still matches a search for ``59``. Scripts written
without spaces between words, such as Japanese, form one token per unbroken run,
so lexical matching is coarse there.

Tokenizing is not matching: a Tamil query still shares no word with an English
document, so every document's lexical score ties and the ranking ignores the
query. The probe therefore translates the query into the corpus's language
before calling in (``_search_query`` in ``generator.py``), which is a no-op
whenever the corpus is already in the conversation language.
"""

from __future__ import annotations

import functools
import math
import unicodedata
from typing import Sequence

from usersim.engine.core.finance_bank import Document, Institution


def _is_word_char(ch: str) -> bool:
    """Letters, combining marks and decimal digits, in any script."""
    category = unicodedata.category(ch)
    return category[0] in "LM" or category == "Nd"


@functools.lru_cache(maxsize=16384)
def _tokens(text: str) -> frozenset[str]:
    """The case-folded words in ``text``.

    Cached because every query re-scores the institution's whole corpus, and a
    document's words never change.
    """
    words: set[str] = set()
    current: list[str] = []
    for ch in unicodedata.normalize("NFC", text.casefold()):
        if _is_word_char(ch):
            current.append(ch)
        elif current:
            words.add("".join(current))
            current.clear()
    if current:
        words.add("".join(current))
    return frozenset(words)


def _lexical_rank(institution: Institution, query: str, k: int) -> list[Document]:
    q = _tokens(query)
    if not q:
        return list(institution.documents[:k])
    scored = []
    for d in institution.documents:
        overlap = len(q & _tokens(f"{d.title} {d.body}"))
        scored.append((overlap, d))
    scored.sort(key=lambda s: (-s[0], s[1].id))
    return [d for _, d in scored[:k]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _dense_rank(
    institution: Institution,
    query: str,
    query_embedding: Sequence[float] | None,
    k: int,
) -> list[Document]:
    """Cosine top-k over the institution's precomputed doc vectors.

    Falls back to lexical ranking when the query was not embedded (no endpoint)
    or the institution ships no embeddings sidecar, so offline runs still work.
    """
    embeddings = getattr(institution, "embeddings", None) or {}
    if not query_embedding or not embeddings:
        return _lexical_rank(institution, query, k)
    scored = []
    for d in institution.documents:
        vec = embeddings.get(d.id)
        if not vec:
            continue
        scored.append((_cosine(query_embedding, vec), d))
    if not scored:
        return _lexical_rank(institution, query, k)
    scored.sort(key=lambda s: (-s[0], s[1].id))
    return [d for _, d in scored[:k]]


#: Weight on the lexical term in ``hybrid``. An even split: the two signals are on
#: incomparable scales before normalisation, and neither is trustworthy alone here
#: (dense collapses onto short tool references, lexical cannot match a paraphrase).
_LEXICAL_WEIGHT = 0.5


def _minmax(values: list[float]) -> list[float]:
    """Scale to [0, 1]. Cosine and token-overlap counts are not comparable
    otherwise; a degenerate (all-equal) signal contributes nothing rather than
    dividing by zero."""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-9:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]


def _hybrid_rank(
    institution: Institution,
    query: str,
    query_embedding: Sequence[float] | None,
    k: int,
) -> list[Document]:
    """Normalised dense cosine + normalised lexical overlap.

    Degrades to pure lexical when there is no query vector or no sidecar — which
    is the stronger of the two signals on this corpus anyway, so an offline run
    loses paraphrase matching rather than relevance.
    """
    embeddings = getattr(institution, "embeddings", None) or {}
    docs = list(institution.documents)
    if not query_embedding or not embeddings or not docs:
        return _lexical_rank(institution, query, k)

    q = _tokens(query)
    dense_raw, lex_raw = [], []
    for d in docs:
        vec = embeddings.get(d.id)
        dense_raw.append(_cosine(query_embedding, vec) if vec else 0.0)
        lex_raw.append(float(len(q & _tokens(f"{d.title} {d.body}"))) if q else 0.0)

    dense, lex = _minmax(dense_raw), _minmax(lex_raw)
    w = _LEXICAL_WEIGHT
    scored = [(w * lex[i] + (1.0 - w) * dense[i], docs[i]) for i in range(len(docs))]
    scored.sort(key=lambda s: (-s[0], s[1].id))
    return [d for _, d in scored[:k]]


def retrieve(
    institution: Institution,
    query: str,
    *,
    k: int = 8,
    mode: str = "golden",
    gold_document_ids: Sequence[str] = (),
    distractor_k: int = 3,
    query_embedding: Sequence[float] | None = None,
) -> list[Document]:
    """Return up to ``k`` documents for ``query``, SCOPED to ``institution``.

    Retrieval never crosses institutions. ``hybrid`` (the default mode at the
    probe) blends normalised dense cosine with normalised lexical overlap;
    ``dense`` uses cosine alone and is kept as an ablation; ``golden`` returns the
    gold docs first (padded with lexical-nearest distractors from the same
    institution up to ``k``). Every mode degrades to lexical when no query vector
    / no sidecar is available.
    """
    if mode == "golden":
        gold = [d for gid in gold_document_ids if (d := institution.doc_by_id(gid))]
        if distractor_k > 0:
            gold_ids = {d.id for d in gold}
            for d in _lexical_rank(institution, query, k):
                if d.id not in gold_ids:
                    gold.append(d)
                    gold_ids.add(d.id)
                if len(gold) >= max(k, len(gold_document_ids) + distractor_k):
                    break
        return gold[:k] if k else gold
    if mode == "hybrid":
        return _hybrid_rank(institution, query, query_embedding, k)
    if mode == "dense":
        return _dense_rank(institution, query, query_embedding, k)
    # unknown mode -> safe deterministic default
    return _lexical_rank(institution, query, k)
