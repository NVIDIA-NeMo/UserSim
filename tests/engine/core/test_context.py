# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for pure per-call context preparation."""

from types import SimpleNamespace

import pytest

from usersim.engine.core.context import (
    _SUMMARY_PROMPT,
    compress_history,
    prepare_assistant_history,
)


class _IdentityProbe:
    def transform_assistant_history(self, messages, *, current_user_turn):
        assert current_user_turn == 1
        return list(messages)


def test_prepare_assistant_history_preserves_canonical_messages():
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "content": '{"ok":true}', "tool_call_id": "c1"},
    ]
    state = SimpleNamespace(messages=messages, conv_summaries={})
    cfg = SimpleNamespace(context_compression=True, compression_window=1)
    before = repr(messages)

    prepared = prepare_assistant_history(state, _IdentityProbe(), cfg)

    assert prepared == messages
    assert repr(messages) == before
    assert prepared is not messages


def test_compress_history_rejects_nonpositive_window():
    with pytest.raises(ValueError, match="at least 1"):
        compress_history(
            [{"role": "assistant", "content": "original"}],
            {0: "summary"},
            window=0,
        )


def test_summary_prompt_requires_source_language_and_script():
    assert "same language and script" in _SUMMARY_PROMPT
