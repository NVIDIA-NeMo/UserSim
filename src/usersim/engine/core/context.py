# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context compression for multi-turn conversations.

Summarizes older assistant responses to keep model context manageable
as conversations grow. The full transcript is always preserved separately
for judges and final output.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from usersim.engine.core.llm import acall_llm, append_debug_record, replaying_assistant_reasoning

logger = logging.getLogger("usersim.engine")

MODEL_SUMMARY = "summary_model"

_SUMMARY_PROMPT = (
    "Summarize the following assistant response in 1-2 plain-text sentences.\n"
    "Focus on: what the assistant communicated, key information provided, "
    "and any questions the assistant asked the user.\n"
    "Write in the same language and script as the assistant response.\n"
    "Do NOT mention tool names, function names, API calls, or internal mechanisms.\n"
    "Do NOT use markdown, bullet points, or any formatting.\n\n"
    "Assistant response:\n{response}"
)


async def summarize_response(
    models: dict[str, Any],
    assistant_content: str,
) -> str:
    """Condense an assistant response into a 1-2 sentence user-facing summary."""
    if not assistant_content or not assistant_content.strip():
        return ""

    prompt = _SUMMARY_PROMPT.format(response=assistant_content)
    msgs = [{"role": "user", "content": prompt}]

    t0 = time.monotonic()
    resp = await acall_llm(models, MODEL_SUMMARY, msgs)
    summary = resp.get("content", "") if isinstance(resp, dict) else ""
    elapsed = time.monotonic() - t0
    orig_words = len(assistant_content.split())
    summ_words = len(summary.split()) if summary else 0
    logger.debug(
        f"  |-- summary_model: {len(assistant_content)} chars / {orig_words} words -> "
        f"{len(summary)} chars / {summ_words} words ({elapsed:.1f}s)"
    )

    append_debug_record(
        {
            "alias": "context_compression",
            "elapsed_s": round(elapsed, 2),
            "original_len": len(assistant_content),
            "summary_len": len(summary),
            "original": assistant_content,
            "summary": summary.strip(),
        }
    )

    return summary.strip()


def compress_history(
    messages: list[dict[str, Any]],
    position_summary_map: dict[int, str],
    window: int = 1,
) -> list[dict[str, Any]]:
    """Return a copy of *messages* with older assistant entries replaced by summaries.

    Only entries whose index appears in *position_summary_map* are candidates.
    The last *window* such entries are kept verbatim; older ones are swapped.
    Tool-call-only assistant messages (no text content) are never touched.
    """
    if window < 1:
        raise ValueError("compression window must be at least 1")
    if not position_summary_map:
        return list(messages)

    positions = sorted(position_summary_map.keys())
    keep_positions = set(positions[-window:]) if len(positions) >= window else set(positions)

    result: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i in position_summary_map and i not in keep_positions:
            summary = position_summary_map[i]
            result.append(
                {
                    **msg,
                    "content": f"[Summary of previous response: {summary}]",
                }
            )
        else:
            result.append(msg)
    return result


def prepare_assistant_history(
    state: Any,
    probe: Any,
    cfg: Any,
) -> list[dict[str, Any]]:
    """Build the assistant's per-call view without mutating the transcript.

    Compression runs before the probe transform because summary keys refer to
    canonical message positions. The finance probe uses the transform to bound
    retrieved document bodies; every other probe inherits the identity view.

    An episode that replays the assistant's reasoning sends its history
    uncompressed, so the assistant sees what it produced: a summary beside
    the full original reasoning is a history it never produced. Read from the
    same setting the request code reads, so the two never disagree.
    """
    messages: list[dict[str, Any]]
    compress = getattr(cfg, "context_compression", False) and not replaying_assistant_reasoning()
    if compress and getattr(state, "conv_summaries", None):
        messages = compress_history(
            state.messages,
            state.conv_summaries,
            window=getattr(cfg, "compression_window", 1),
        )
    else:
        messages = list(state.messages)

    transform = getattr(probe, "transform_assistant_history", None)
    if transform is None:
        return messages
    return transform(
        messages,
        current_user_turn=sum(1 for message in state.messages if message.get("role") == "user"),
    )
