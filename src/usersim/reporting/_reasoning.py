# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Post-hoc reasoning-token estimator.

DataDesigner's normalised ``Usage`` dataclass exposes only ``input_tokens``
/ ``output_tokens`` / ``total_tokens`` (see ``data_designer/engine/models/
clients/types.py``). Provider-specific reasoning telemetry (OpenAI's
``completion_tokens_details.reasoning_tokens``, Gemini's
``thoughts_token_count``, Anthropic's extended-thinking breakdown) is
stripped at the DD abstraction boundary -- there is no provider-captured
reasoning count to read at sim time.

To surface a reasoning-token estimate anyway, we derive it post-hoc at
report-build time from data we DO have on the trajectory parquet:

    reasoning_tokens (estimate) = output_tokens - tiktoken(visible_text)

This module is intentionally read-only on the trajectory parquet -- no
LLM is re-called, no eval data is touched. Estimates are flagged with
``source = "estimated"`` and a 5% per-alias noise-floor clamp is applied
post-aggregation: alias-level estimates below 5% of output are zeroed
out and marked ``unavailable`` because they're indistinguishable from
tokenizer-mismatch drift between ``cl100k_base`` and the provider's
actual tokenizer.

Aliases:

- ``assistant_model``  ->  estimable from `assistant`-role messages
  (including ``tool_calls``, which the provider counts toward
  ``completion_tokens`` but doesn't surface as ``content``).
- ``user_model``       ->  estimable from `user`-role messages. Approximate
  for probes that inject turn-1 verbatim from an asset bank rather than
  generating it via user_model -- the over-count is bounded.
- ``api_response_model`` ->  estimable from `tool`-role messages.
  tool_calling probes only.
- ``judge_model`` / ``summary_model`` ->  ``unavailable`` (their output
  isn't preserved verbatim on the trajectory).

Encoding selection: ``cl100k_base`` for everything (v1). Per-model
encoding lookup is deferred -- the model string lives on the run manifest
(``_manifest.json``), not on ``simulation_outcome``. Tokenizer drift
between ``cl100k_base`` and the provider's actual tokenizer is the
dominant error term and is what the noise floor is sized to swallow.

Future direction: if DataDesigner adds provider-captured reasoning to
``Usage`` (`tracked here for follow-up`), Phase A capture can be
reintroduced as a faster / more accurate path.
"""

from __future__ import annotations

import functools
import json
from typing import Any

# Aliases that produce visible conversation turns (estimable post-hoc).
_VISIBLE_OUTPUT_ALIASES: tuple[str, ...] = (
    "assistant_model",
    "user_model",
    "api_response_model",
)

# Aliases whose output isn't preserved on the trajectory -- always
# ``unavailable`` because Phase B has no visible content to tokenize.
_UNAVAILABLE_ALIASES: tuple[str, ...] = ("judge_model", "summary_model")


# Per-alias noise floor: aggregate reasoning estimates below this fraction
# of the alias's output tokens are clamped to zero and flagged
# ``unavailable``. Sized to swallow tokenizer-mismatch drift between
# ``cl100k_base`` and the provider's actual tokenizer (we've seen ~1%
# drift on Gemma / NVIDIA inference hub). Real reasoning models clear
# this floor by 4-10x in practice (see roadmap entry for the empirical
# evidence).
_REASONING_NOISE_FLOOR_RATIO: float = 0.05


@functools.lru_cache(maxsize=8)
def _encoder(name: str):
    """Cached tiktoken encoder, or ``None`` when one can't be built.

    ``get_encoding`` fetches the BPE vocabulary over HTTPS on first use --
    tiktoken ships the tokenizer, not the vocabulary -- so an air-gapped or
    proxied host cannot construct an encoder at all. Returning ``None``
    routes those rows to ``unavailable``, the tier that already exists for
    aliases with no visible content, instead of failing the whole report
    build. ``lru_cache`` also bounds the cost to one failed lookup per
    process rather than one per row.
    """
    import tiktoken

    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return None


def _decode_messages(messages: Any) -> list | None:
    """Coerce a stored ``conversation_messages`` cell into a list of dicts.

    The trajectory parquet stores it as either a JSON string or a native
    list (depending on the writer). Returns ``None`` on malformed input
    so the caller can mark the row's estimate as ``unavailable``.
    """
    if messages is None:
        return None
    if isinstance(messages, list):
        return messages
    if isinstance(messages, str):
        try:
            parsed = json.loads(messages)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, list) else None
    # numpy/pandas array shapes
    try:
        return list(messages)
    except (TypeError, ValueError):
        return None


def _count_visible_tokens_for_alias(
    messages: Any,
    alias: str,
    encoding_name: str,
) -> int | None:
    """Tokenize the visible content the alias produced in this conversation.

    Returns ``None`` when the messages are malformed (caller marks
    ``unavailable``).

    For ``assistant_model``, we tokenize ``content`` PLUS the JSON-
    stringified ``tool_calls`` block. The provider counts both toward
    ``completion_tokens``, so omitting ``tool_calls`` would over-attribute
    them to "reasoning" for tool_calling probes -- a real correctness gap.
    """
    msgs = _decode_messages(messages)
    if msgs is None:
        return None
    enc = _encoder(encoding_name)
    if enc is None:
        return None

    if alias == "assistant_model":
        chunks = []
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") != "assistant":
                continue
            chunks.append(m.get("content") or "")
            tool_calls = m.get("tool_calls")
            if tool_calls:
                chunks.append(json.dumps(tool_calls, separators=(",", ":")))
        text = "\n".join(chunks)
    elif alias == "user_model":
        text = "\n".join(m.get("content") or "" for m in msgs if isinstance(m, dict) and m.get("role") == "user")
    elif alias == "api_response_model":
        text = "\n".join(m.get("content") or "" for m in msgs if isinstance(m, dict) and m.get("role") == "tool")
    else:
        return None

    try:
        return len(enc.encode(text))
    except Exception:
        # Tiktoken can fail on certain input shapes -- mark as unavailable
        # rather than crashing the whole report build.
        return None


def estimate_reasoning_for_row(
    conversation_messages: Any,
    alias: str,
    output_tokens: int,
    *,
    encoding_name: str = "cl100k_base",
) -> tuple[int, str]:
    """Per-row, per-alias reasoning-token estimate.

    Returns ``(reasoning_tokens, source)`` where ``source`` is one of:

    - ``"estimated"``: ``reasoning = output_tokens - tiktoken(visible)``,
      clamped to non-negative. Per-row negative diffs (tiktoken over-
      counted vs the provider's tokenizer) silently clamp to 0 and
      contribute as ``estimated``; the post-aggregation noise floor in
      ``reporting/resource.py`` decides whether the alias's total is
      real or just drift.
    - ``"unavailable"``: alias has no visible content on the trajectory
      (``judge_model`` / ``summary_model``), ``conversation_messages``
      is malformed, or no tokenizer could be built (see :func:`_encoder`).
    """
    if alias not in _VISIBLE_OUTPUT_ALIASES:
        return 0, "unavailable"
    visible = _count_visible_tokens_for_alias(conversation_messages, alias, encoding_name)
    if visible is None:
        return 0, "unavailable"
    diff = int(output_tokens) - int(visible)
    if diff < 0:
        # Tokenizer drift -- clamp. The noise floor at aggregation will
        # decide whether this alias produced real reasoning or just
        # drift across the run.
        return 0, "estimated"
    return diff, "estimated"
