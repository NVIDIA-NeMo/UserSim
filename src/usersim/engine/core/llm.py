# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM call helpers and majority voting logic for conversation simulation."""

from __future__ import annotations

import json
import logging
import os
import random as _random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

_MAX_RETRIES = 3
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0

logger = logging.getLogger("usersim.engine")

# Full prompts, completions, reasoning content, and tool calls. Opt-in:
# a simulator run must not drop a transcript of everything it sent to a
# provider into whatever directory the user happened to be standing in.
# Enable with USERSIM_DEBUG_LOG=<path>, or programmatically via
# set_debug_log_path(). The default location sits under the gitignored
# output/ tree so an enabled run still cannot be committed by accident.
_DEFAULT_DEBUG_LOG_PATH = Path("output/debug/debug_llm_responses.jsonl")
_DEBUG_LOG_PATH = Path(os.environ.get("USERSIM_DEBUG_LOG") or _DEFAULT_DEBUG_LOG_PATH)
_DEBUG_LOG_ENABLED = bool(os.environ.get("USERSIM_DEBUG_LOG"))
_CALL_COUNTER = 0
_CALL_LOCK = threading.Lock()
_PENDING_RECORDS: List[Dict[str, Any]] = []

_CONV_LOCAL = threading.local()


class ContextWindowError(RuntimeError):
    """A deterministic context-capacity failure for one model alias."""

    def __init__(self, alias: str, original: Exception) -> None:
        self.alias = alias
        self.original = original
        super().__init__(f"{alias} context window exceeded: {type(original).__name__}: {original}")


#: Multiplier applied to a model's configured ``max_tokens`` for locales whose
#: script tokenizes more densely than English. 2.0 keeps the same amount of
#: *text* within budget rather than the same number of tokens.
#:
#: This scales the configured value instead of substituting a constant. A
#: constant silently became a CAP once the catalog default rose above it, so
#: the locales meant to get more room got half of what English got, and the
#: resulting truncation reads as a quality failure rather than a budget one.
NON_ASCII_TOKEN_SCALE = 2.0


def scaled_max_tokens(facade: Any, scale: float) -> Dict[str, int]:
    """Return ``{"max_tokens": n}`` scaled from what ``facade`` is configured with.

    Returns an empty dict when the model sets no budget, which is how
    reasoning models are configured: substituting one here would reintroduce
    the parameter they reject.
    """
    configured = None
    try:
        params = facade._model_config.inference_parameters
        configured = getattr(params, "max_tokens", None)
    except AttributeError:
        return {}
    if not configured:
        return {}
    return {"max_tokens": int(configured * scale)}


def _rejects_max_tokens(error: Exception) -> bool:
    """Detect the provider error meaning "send max_completion_tokens instead".

    Reasoning endpoints refuse ``max_tokens`` outright rather than ignoring
    it, so this is a permanent failure that retrying unchanged cannot fix.
    Matched on substrings because the wording differs across providers.
    """
    text = str(error).lower()
    return "max_completion_tokens" in text and ("unsupported" in text or "not supported" in text or "instead" in text)


def _is_context_window_error(error: Exception) -> bool:
    """Recognize context failures through provider wrapper/cause chains."""
    pending: List[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        name = type(current).__name__.lower()
        kind = str(getattr(current, "kind", "")).lower()
        text = str(current).lower()
        if (
            "contextwindowexceeded" in name
            or "context_window_exceeded" in kind
            or "contextwindowexceeded" in text
            or "maximum context length" in text
            or "context length exceeded" in text
        ):
            return True
        for linked in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return False


# ── Per-model call statistics (thread-safe) ──────────────────────────────
_MODEL_STATS: Dict[str, Dict[str, float]] = {}


# ── Per-trajectory outcome hook (thread-local) ──────────────────────────
# generator.py sets this to the current ConversationState.outcome builder
# at the start of generate() and clears it at the end. Every successful
# call_llm invocation feeds its per-call tokens/calls/latency into the
# builder as well as the cross-trajectory MODEL_STATS aggregate. Exposed
# via set_current_outcome_builder() so the simulator's per-row resource
# profile is populated without threading state through every caller.


def set_current_outcome_builder(builder) -> None:  # type: ignore[no-untyped-def]
    """Set (or clear) the thread-local OutcomeBuilder receiving per-call stats.

    Pass ``None`` to clear. Safe to call with any object exposing
    ``record_call(alias, input_tokens, output_tokens, elapsed_s)`` —
    typed loosely to avoid a circular import from core.outcomes.
    """
    if builder is None:
        if hasattr(_CONV_LOCAL, "outcome_builder"):
            delattr(_CONV_LOCAL, "outcome_builder")
    else:
        _CONV_LOCAL.outcome_builder = builder


def _feed_outcome_builder(alias: str, input_tokens: int, output_tokens: int, elapsed_s: float) -> None:
    """Forward per-call resource stats to the thread-local outcome builder, if set.

    Reasoning tokens are NOT captured here. DataDesigner's normalised
    ``Usage`` dataclass exposes only ``input_tokens`` / ``output_tokens``
    / ``total_tokens`` (see ``data_designer/engine/models/clients/types.py``)
    -- it strips ``completion_tokens_details`` and any provider-specific
    reasoning fields at the abstraction layer. The report-build path
    estimates reasoning post-hoc via tiktoken on visible content and
    applies a 5% noise-floor clamp. See ``reporting/_reasoning.py``.
    """
    builder = getattr(_CONV_LOCAL, "outcome_builder", None)
    if builder is None:
        return
    try:
        builder.record_call(
            alias=alias,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_s=elapsed_s,
        )
    except Exception as e:  # defensive — we never break a simulation on telemetry
        logger.debug(f"  |-- outcome builder record_call failed: {e}")


def _has_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
            or 0x3040 <= cp <= 0x309F  # Hiragana
            or 0x30A0 <= cp <= 0x30FF  # Katakana
            or 0xAC00 <= cp <= 0xD7AF  # Hangul Syllables
        ):
            return True
    return False


def _word_count(text: str) -> int:
    if not text:
        return 0
    if _has_cjk(text):
        return len(text)
    return len(text.split())


def _record_stat(
    alias: str,
    elapsed: float,
    content_len: int,
    word_cnt: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    with _CALL_LOCK:
        s = _MODEL_STATS.setdefault(
            alias,
            {
                "calls": 0,
                "total_s": 0.0,
                "total_chars": 0,
                "total_words": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            },
        )
        s["calls"] += 1
        s["total_s"] += elapsed
        s["total_chars"] += content_len
        s["total_words"] += word_cnt
        s["total_input_tokens"] += input_tokens
        s["total_output_tokens"] += output_tokens


def get_call_stats() -> Dict[str, Dict[str, Any]]:
    """Return a snapshot of per-model call statistics."""
    with _CALL_LOCK:
        return {alias: dict(s) for alias, s in _MODEL_STATS.items()}


# ── Per-record timing (thread-safe) ──────────────────────────────────────
_RECORD_TIMES: List[float] = []


def record_finished(elapsed: float) -> None:
    """Register a completed record's total wall-time."""
    with _CALL_LOCK:
        _RECORD_TIMES.append(elapsed)


def get_record_stats() -> Dict[str, float]:
    """Return record-level timing statistics."""
    with _CALL_LOCK:
        times = list(_RECORD_TIMES)
    if not times:
        return {}
    return {
        "records": len(times),
        "total_s": sum(times),
        "avg_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
    }


def set_debug_log_path(path: str | Path | None) -> None:
    """Set the debug log file path, or None to disable debug logging.

    Call this before a simulation run to separate preview and full-run logs.
    """
    global _DEBUG_LOG_PATH, _DEBUG_LOG_ENABLED
    if path is None:
        _DEBUG_LOG_ENABLED = False
    else:
        _DEBUG_LOG_PATH = Path(path)
        _DEBUG_LOG_ENABLED = True


def set_conversation_id(conv_id: str) -> None:
    """Set the conversation identifier for the current thread's debug logs."""
    _CONV_LOCAL.conv_id = conv_id


def append_debug_record(record: Dict[str, Any]) -> None:
    """Append an arbitrary record to the pending debug log."""
    with _CALL_LOCK:
        global _CALL_COUNTER
        _CALL_COUNTER += 1
        record.setdefault("seq", _CALL_COUNTER)
        record.setdefault("conv_id", getattr(_CONV_LOCAL, "conv_id", "unknown"))
        _PENDING_RECORDS.append(record)


def flush_debug_log() -> None:
    """Write all pending debug records to disk, sorted by conv_id then seq."""
    with _CALL_LOCK:
        records = list(_PENDING_RECORDS)
        _PENDING_RECORDS.clear()
    if not records or not _DEBUG_LOG_ENABLED:
        return
    records.sort(key=lambda r: (r["conv_id"], r["seq"]))
    with open(_DEBUG_LOG_PATH, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


from data_designer.engine.models.utils import ChatMessage


def _dicts_to_chat_messages(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Convert a list of plain dicts to ChatMessage objects for ModelFacade.

    A prior turn's ``reasoning_content`` is never forwarded: the
    assistant's ``reasoning_content`` is hard-coded to ``None`` below,
    matching ``ChatMessage``'s own default for the field. A trace is an
    artefact of the turn that produced it, not conversation state.

    Stored trajectories DO keep the trace on the assistant message (see
    ``assistant_message`` in ``core/probes.py``), so it stays available
    for analysis -- it is dropped only on the way back into a model.
    """
    out: List[ChatMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            out.append(ChatMessage.as_system(content))
        elif role == "assistant":
            out.append(
                ChatMessage.as_assistant(
                    content=content,
                    reasoning_content=None,
                    tool_calls=msg.get("tool_calls") or None,
                )
            )
        elif role == "tool":
            out.append(ChatMessage.as_tool(content, msg.get("tool_call_id", "")))
        else:
            out.append(ChatMessage.as_user(content))
    return out


def _assistant_message_to_dict(msg: Any) -> Dict[str, Any]:
    """Convert a DD AssistantMessage to a plain dict.

    DD v0.5.3 ChatCompletionResponse has a .message (AssistantMessage)
    with .content, .reasoning_content, and .tool_calls attributes.
    DD's ToolCall dataclass uses flat fields (id, name, arguments_json),
    so we convert to OpenAI format ({function: {name, arguments}}) which
    the verifier and generator code expects.
    """
    result: Dict[str, Any] = {
        "role": "assistant",
        "content": getattr(msg, "content", "") or "",
    }
    if getattr(msg, "reasoning_content", None):
        result["reasoning_content"] = msg.reasoning_content
    raw_tool_calls = getattr(msg, "tool_calls", None)
    if raw_tool_calls:
        converted = []
        for tc in raw_tool_calls:
            if hasattr(tc, "arguments_json"):
                converted.append(
                    {
                        "id": getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(tc, "name", ""),
                            "arguments": getattr(tc, "arguments_json", "{}"),
                        },
                    }
                )
            elif hasattr(tc, "model_dump"):
                converted.append(tc.model_dump())
            elif isinstance(tc, dict):
                converted.append(tc)
            else:
                converted.append(dict(tc))
        result["tool_calls"] = converted
    return result


def call_llm(
    models: Dict[str, Any],
    alias: str,
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Call an LLM via DD ModelFacade.completion().

    Returns a dict with role, content, and optionally reasoning_content
    and tool_calls. Logs timing at DEBUG level for verbosity=2.
    """
    facade = models.get(alias)
    if facade is None:
        raise ValueError(f"Model alias '{alias}' not found in models dict")

    has_tools = "tools" in kwargs and kwargs["tools"]
    tool_tag = f" + {len(kwargs['tools'])} tools" if has_tools else ""
    chat_messages = _dicts_to_chat_messages(messages)

    t0 = time.monotonic()
    logger.debug(f"  |-- LLM call: {alias} ({len(messages)} msgs{tool_tag}) ...")

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = facade.completion(chat_messages, **kwargs)
            break
        except Exception as e:
            if _is_context_window_error(e):
                raise ContextWindowError(alias, e) from e
            # Reasoning endpoints replaced ``max_tokens`` with
            # ``max_completion_tokens`` because hidden thinking tokens are
            # billed as output. Callers pass a budget without knowing which
            # spelling the model takes, so translate once and retry rather
            # than burning the backoff budget on an error that cannot
            # resolve itself.
            if _rejects_max_tokens(e) and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                logger.debug(
                    "  |-- %s rejects max_tokens; retrying with max_completion_tokens",
                    alias,
                )
                continue
            if attempt >= _MAX_RETRIES:
                raise
            delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
            jitter = _random.uniform(0, delay * 0.3)
            logger.warning(
                f"  |-- LLM retry {attempt + 1}/{_MAX_RETRIES} for {alias}: "
                f"{type(e).__name__}: {e} (backoff {delay + jitter:.1f}s)"
            )
            time.sleep(delay + jitter)

    elapsed = time.monotonic() - t0

    result = _assistant_message_to_dict(response.message)
    n_tool_calls = len(result.get("tool_calls", []))
    tc_tag = f", {n_tool_calls} tool_calls" if n_tool_calls else ""
    content = result.get("content", "")
    content_len = len(content)
    word_cnt = _word_count(content)
    usage = getattr(response, "usage", None)
    input_tokens = 0
    output_tokens = 0
    if usage:
        # DD's provider-independent Usage exposes only input_tokens /
        # output_tokens / total_tokens (see data_designer/engine/models/
        # clients/types.py). Provider-specific reasoning fields
        # (OpenAI's completion_tokens_details, Gemini's
        # thoughts_token_count, Anthropic's cache stats) are stripped
        # at the DD abstraction boundary. We don't try to read them
        # here; reasoning is estimated post-hoc at report build via
        # tiktoken on the trajectory's visible content. Some OpenAI-
        # shaped mocks expose prompt_tokens / completion_tokens, so
        # support both names.
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None) or 0
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None) or 0
    token_tag = f", {input_tokens}+{output_tokens} tok" if (input_tokens or output_tokens) else ""

    logger.debug(
        f"  |-- LLM done: {alias} in {elapsed:.1f}s ({content_len} chars / {word_cnt} words{tc_tag}{token_tag})"
    )
    _record_stat(alias, elapsed, content_len, word_cnt, input_tokens, output_tokens)
    _feed_outcome_builder(alias, input_tokens, output_tokens, elapsed)

    with _CALL_LOCK:
        global _CALL_COUNTER
        _CALL_COUNTER += 1
        seq = _CALL_COUNTER

    reasoning = getattr(response.message, "reasoning_content", None)
    conv_id = getattr(_CONV_LOCAL, "conv_id", "unknown")
    debug_record = {
        "seq": seq,
        "conv_id": conv_id,
        "alias": alias,
        "elapsed_s": round(elapsed, 2),
        "content_len": content_len,
        "reasoning_len": len(reasoning) if reasoning else 0,
        "n_tool_calls": n_tool_calls,
        "input_messages": [{"role": m.get("role"), "content": m.get("content", "")} for m in messages],
        "content": result.get("content", ""),
        "reasoning_content": reasoning,
        "tool_calls": result.get("tool_calls"),
    }
    try:
        with _CALL_LOCK:
            _PENDING_RECORDS.append(debug_record)
    except Exception:
        pass

    return result
