# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query-embedding helper for dense retrieval (probe-time).

Corpus document vectors are precomputed offline in the asset_gen pipeline and
shipped in each institution's ``embeddings.parquet`` sidecar (loaded by
``finance_bank``). At probe time the dense retriever embeds the *query* with the
same embedding model so query and document vectors are comparable.

Kept separate from ``core/llm.py`` (which is chat-only): embedding models route
through a distinct OpenAI-compatible ``/v1/embeddings`` endpoint. The call goes
through the same DataDesigner ``ModelFacade`` the rest of the runtime uses
(``generate_text_embeddings``), so no new client wiring is required — only an
embedding-model alias in the ``models`` dict.

Best-effort by design: any failure (missing alias, endpoint down, model does not
support embeddings) returns ``None`` so ``retrieval.py`` degrades gracefully to
the offline lexical path rather than failing the simulation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("usersim.engine")

DEFAULT_EMBEDDING_ALIAS = "embedding_model"


def embed_query(
    models: Dict[str, Any],
    alias: str,
    text: str,
) -> Optional[List[float]]:
    """Embed a single query string; return its vector or ``None`` on any error.

    ``None`` is a soft signal to the caller to fall back to lexical retrieval;
    this function never raises.
    """
    if not models or not text:
        return None
    facade = models.get(alias)
    if facade is None:
        logger.debug(
            "  |-- embeddings: no facade for alias %r; dense query falls back to lexical",
            alias,
        )
        return None
    embed_fn = getattr(facade, "generate_text_embeddings", None)
    if not callable(embed_fn):
        logger.debug(
            "  |-- embeddings: facade %r has no generate_text_embeddings; falling back to lexical",
            alias,
        )
        return None
    try:
        vectors = embed_fn([text])
    except Exception as e:  # noqa: BLE001 — soft-fail to lexical
        logger.warning(
            "  |-- embeddings: query embedding via %r failed (%s: %s); falling back to lexical",
            alias,
            type(e).__name__,
            e,
        )
        return None
    if not vectors:
        return None
    return list(vectors[0])
