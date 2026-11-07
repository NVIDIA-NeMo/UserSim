# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the credentials the tool_calling probe's simulated user holds."""

import re

import pytest

from usersim.engine.probes.tool_calling.credentials import (
    credentials_set_by,
    declared_credentials,
    draw_credentials,
    format_held_credentials,
    rejected_credential,
)


def _entry(name: str, credentials: object = None, properties: tuple[str, ...] = ("query", "api_key")) -> dict:
    entry: dict = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string", "description": p} for p in properties},
                "required": list(properties),
            },
        },
    }
    if credentials is not None:
        entry["credentials"] = credentials
    return entry


class TestDeclaredCredentials:
    def test_a_tool_without_a_declaration_declares_none(self) -> None:
        assert declared_credentials(_entry("search")) == {}

    def test_a_declaration_maps_parameters_to_credentials(self) -> None:
        assert declared_credentials(_entry("search", {"api_key": "SerpApi key"})) == {"api_key": "SerpApi key"}

    def test_the_wrapped_envelope_is_read_the_same_way(self) -> None:
        flat = _entry("search", {"api_key": "SerpApi key"})
        wrapped = {"tool": {"type": "function", "function": flat["function"]}, "credentials": flat["credentials"]}
        assert declared_credentials(wrapped) == {"api_key": "SerpApi key"}

    @pytest.mark.parametrize("credentials", [["api_key"], {"api_key": ""}, {"api_key": 3}])
    def test_a_malformed_declaration_is_refused(self, credentials: object) -> None:
        with pytest.raises(ValueError, match="must map parameter names to credential names"):
            declared_credentials(_entry("search", credentials))

    def test_a_credential_for_an_undefined_parameter_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"does not define: \['token'\]"):
            declared_credentials(_entry("search", {"token": "SerpApi key"}))


class TestCredentialsSetBy:
    def test_a_tool_without_a_declaration_sets_none(self) -> None:
        assert credentials_set_by(_entry("search", {"api_key": "SerpApi key"})) == {}

    def test_a_declaration_maps_parameters_to_the_credentials_they_replace(self) -> None:
        entry = _entry("change_password", properties=("old_password", "new_password"))
        entry["sets_credentials"] = {"new_password": "account password"}
        assert credentials_set_by(entry) == {"new_password": "account password"}

    @pytest.mark.parametrize(
        ("sets", "message"),
        [
            (["new_password"], "sets_credentials must map parameter names to credential names"),
            (
                {"token": "account password"},
                r"declares sets_credentials for parameters it does not define: \['token'\]",
            ),
        ],
    )
    def test_a_malformed_declaration_is_refused(self, sets: object, message: str) -> None:
        entry = _entry("change_password", properties=("new_password",))
        entry["sets_credentials"] = sets
        with pytest.raises(ValueError, match=message):
            credentials_set_by(entry)


class TestDrawCredentials:
    tools = (
        _entry("search", {"api_key": "SerpApi key"}),
        _entry("trends", {"api_key": "SerpApi key"}),
        _entry("weather", {"api_key": "WeatherAPI.com API key"}),
        _entry("calculator"),
    )

    def test_one_value_per_credential_the_tools_declare(self) -> None:
        held = draw_credentials(self.tools, "traj-1")
        assert sorted(held) == ["SerpApi key", "WeatherAPI.com API key"]
        assert held["SerpApi key"] != held["WeatherAPI.com API key"]

    def test_values_are_random_looking_keys(self) -> None:
        for value in draw_credentials(self.tools, "traj-1").values():
            assert re.fullmatch(r"[a-z0-9]{24}", value), value

    def test_one_conversation_always_holds_the_same_values(self) -> None:
        assert draw_credentials(self.tools, "traj-1") == draw_credentials(self.tools, "traj-1")

    def test_separate_conversations_hold_separate_values(self) -> None:
        values = {draw_credentials(self.tools, f"traj-{i}")["SerpApi key"] for i in range(20)}
        assert len(values) == 20, "two conversations were handed the same key"

    def test_tools_without_declarations_hold_nothing(self) -> None:
        assert draw_credentials([_entry("calculator")], "traj-1") == {}

    def test_held_credentials_are_listed_one_per_line(self) -> None:
        held = {"SerpApi key": "a" * 24, "WeatherAPI.com API key": "b" * 24}
        assert format_held_credentials(held) == f"- SerpApi key: {'a' * 24}\n- WeatherAPI.com API key: {'b' * 24}"


class TestRejectedCredential:
    declared = {"api_key": "SerpApi key"}
    held = {"SerpApi key": "k" * 24}

    def test_the_held_value_is_accepted(self) -> None:
        assert rejected_credential(self.declared, {"query": "q", "api_key": "k" * 24}, self.held) is None

    @pytest.mark.parametrize("arguments", [{"query": "q"}, {"api_key": ""}, {"api_key": "YOUR_API_KEY"}])
    def test_a_missing_empty_or_made_up_value_is_rejected(self, arguments: dict) -> None:
        assert rejected_credential(self.declared, arguments, self.held) == ("api_key", "SerpApi key")

    def test_a_tool_without_credentials_rejects_nothing(self) -> None:
        assert rejected_credential({}, {"query": "q"}, self.held) is None

    def test_a_credential_the_user_does_not_hold_is_rejected(self) -> None:
        assert rejected_credential(self.declared, {"api_key": None}, {}) == ("api_key", "SerpApi key")
