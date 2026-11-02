# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``model_facades``: a models file's aliases as facades a probe runs on outside a pipeline.

The invariants: one facade per alias, built from the same provider, model and
inference settings ``usersim simulate`` would use, without calling a model or
writing where the caller runs, and warning about an unset key as a run does;
and a request made through one carries those settings plus the call's own
``tools`` and ``tool_choice``.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from usersim.cli._errors import ConfigError
from usersim.engine.core.llm import acall_llm
from usersim.engine.external import model_facades, replays_assistant_reasoning

# TEST-NET-3: never routable; the facades must not connect while being built.
ENDPOINT = "http://203.0.113.1/v1"
FORCED = {"type": "function", "function": {"name": "commit_move"}}
TOOL = {
    "type": "function",
    "function": {"name": "commit_move", "parameters": {"type": "object", "properties": {"move": {"type": "string"}}}},
}


def _models_file(tmp_path, assistant_extra: str = "", user_extra: str = "") -> Any:
    path = tmp_path / "models.toml"
    path.write_text(
        "providers = [\n"
        f'  {{ name = "local", endpoint = "{ENDPOINT}", provider_type = "openai", api_key = "FACADE_TEST_KEY" }},\n'
        "]\n"
        "models = [\n"
        '  { alias = "assistant_model", model = "test/assistant", provider = "local", '
        f"max_tokens = 1234, temperature = 0.3{assistant_extra} }},\n"
        '  { alias = "user_model", model = "test/user", provider = "local", '
        f"extra_body = {{ chat_template_kwargs = {{ enable_thinking = true }} }}{user_extra} }},\n"
        "]\n"
    )
    return path


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("FACADE_TEST_KEY", "not-a-real-key")


def test_every_alias_gets_a_facade_on_its_configured_model_and_provider(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    facades = model_facades(_models_file(tmp_path))

    assert list(cwd.iterdir()) == []  # building them writes nothing where the caller runs
    assert set(facades) == {"assistant_model", "user_model"}
    assert facades["assistant_model"].model_name == "test/assistant"
    assert facades["user_model"].model_name == "test/user"
    assert all(facade.model_provider.endpoint == ENDPOINT for facade in facades.values())


def test_a_missing_managed_assets_path_and_a_clean_working_directory(tmp_path, monkeypatch):
    """Data Designer's managed-assets path does not exist (its home seeding is skipped, so nothing creates it and
    nothing is written to the real home): the facades still build, and nothing is written where the caller runs."""
    import data_designer.interface.data_designer as dd_interface

    monkeypatch.setattr(dd_interface, "MANAGED_ASSETS_PATH", tmp_path / "nowhere" / "managed-assets")
    monkeypatch.setattr(dd_interface, "resolve_seed_default_model_settings", lambda: None)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    facades = model_facades(_models_file(tmp_path))

    assert set(facades) == {"assistant_model", "user_model"}
    assert list(cwd.iterdir()) == []
    assert not (tmp_path / "nowhere").exists()


def test_an_unset_key_is_warned_about_as_a_run_warns(tmp_path, monkeypatch, caplog):
    """The facades still build (Data Designer resolves the key when it builds them, and an unset variable resolves
    to its own name, which a request would then send), so the missing variable is named up front."""
    monkeypatch.delenv("FACADE_TEST_KEY")
    with caplog.at_level("WARNING", logger="usersim.engine"):
        model_facades(_models_file(tmp_path))
    assert "FACADE_TEST_KEY" in caplog.text


def test_a_subset_of_aliases_and_an_unknown_alias(tmp_path):
    path = _models_file(tmp_path)

    assert set(model_facades(path, aliases=["user_model"])) == {"user_model"}
    with pytest.raises(ConfigError, match="judge_model"):
        model_facades(path, aliases=["user_model", "judge_model"])


async def test_a_request_carries_the_alias_settings_and_the_calls_tools(tmp_path, monkeypatch):
    """The settings come from the models file; tools and a forced tool_choice from the call."""
    from data_designer.engine.models.clients.adapters.http_model_client import HttpModelClient

    sent: list[dict[str, Any]] = []

    async def capture(self, route, payload, extra_headers, model_name, timeout=None):
        sent.append({"route": route, **payload})
        return {
            "id": "c",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "t1", "type": "function", "function": {"name": "commit_move", "arguments": "{}"}}
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(HttpModelClient, "_apost", capture)
    facades = model_facades(_models_file(tmp_path))
    messages = [{"role": "system", "content": "Commit a move."}, {"role": "user", "content": "How was your week?"}]

    reply = await acall_llm(facades, "user_model", messages, tools=[TOOL], tool_choice=FORCED)
    await acall_llm(facades, "assistant_model", messages)

    patient, assistant = sent
    assert patient["route"].endswith("/chat/completions")
    assert (patient["model"], patient["tool_choice"]) == ("test/user", FORCED)
    assert [t["function"]["name"] for t in patient["tools"]] == ["commit_move"]
    assert patient["chat_template_kwargs"] == {"enable_thinking": True}
    assert (assistant["model"], assistant["max_tokens"], assistant["temperature"]) == ("test/assistant", 1234, 0.3)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["commit_move"]


def test_replay_follows_the_models_file(tmp_path):
    (tmp_path / "on").mkdir()
    (tmp_path / "other").mkdir()

    assert replays_assistant_reasoning(_models_file(tmp_path)) is False
    assert replays_assistant_reasoning(_models_file(tmp_path / "on", assistant_extra=", replay_reasoning = true"))
    with pytest.raises(ConfigError, match="assistant_model"):
        replays_assistant_reasoning(_models_file(tmp_path / "other", user_extra=", replay_reasoning = true"))


async def test_a_request_goes_out_on_an_async_client(tmp_path, monkeypatch):
    """Only the HTTP send is replaced, so Data Designer's own client is used, and a sync-mode client refuses an
    async call."""
    import httpx

    sent: list[str] = []

    async def post(self, url, **kwargs):
        sent.append(str(url))
        body = {
            "id": "c",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    facades = model_facades(_models_file(tmp_path), aliases=["user_model"])

    reply = await acall_llm(facades, "user_model", [{"role": "user", "content": "hi"}])

    assert reply["content"] == "ok"
    assert sent and sent[0].endswith("/chat/completions")


def test_the_callers_logging_is_left_as_it_is(tmp_path, monkeypatch):
    """As in a fresh process, where Data Designer would set up logging itself: the caller's handlers survive."""
    import data_designer.interface.data_designer as dd_interface

    monkeypatch.setattr(dd_interface, "is_logging_configured", lambda: False)
    root = logging.getLogger()
    handler, level = logging.NullHandler(), root.level
    root.addHandler(handler)
    try:
        model_facades(_models_file(tmp_path))
        assert handler in root.handlers and root.level == level
    finally:
        root.removeHandler(handler)
