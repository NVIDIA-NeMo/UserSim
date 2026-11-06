# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tool catalog the tool_calling probe ships.

Each row's function specs reach the assistant model unchanged, as OpenAI-style
tools, so every spec in ``toolsets_seed.parquet`` must be valid JSON Schema that
a provider accepts, with every tool and parameter described and typed. A tool
that takes the caller's key for a service declares it under ``credentials``,
which the simulated user then holds.
"""

import json
import re
from collections import Counter
from collections.abc import Iterator

import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from usersim.engine.core.missing import is_missing
from usersim.engine.probes.tool_calling.credentials import credentials_set_by, declared_credentials
from usersim.engine.probes.tool_calling.probe import ToolCallingProbe

JSON_SCHEMA_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object", "null"})
_FUNCTION_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_PARAMETER_NAME = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")
#: The method and URL a REST tool's description ends with.
_ENDPOINT = re.compile(r" HTTP [A-Z]+ (\S+)$")
#: Parameter names that take nothing but the caller's credential for a service.
_KEY_NAME = re.compile(r"(?i)api_?key|secret|access_token|public_key|appid")
_PASSWORD_NAME = re.compile(r"(?i)pass(word|wd)")
#: Password parameters that take no password the user already holds, by
#: (toolset, tool, parameter): each sets a password on something new, stores
#: one, or is no secret.
_PASSWORDS_NOT_HELD = {
    ("ToolEyes/File/Pdf", "add_password", "password"),  # set on the PDF
    ("UltraTools/Agenda", "file_compression", "password"),  # set on the archive
    ("UltraTools/Document", "file_compression", "password"),  # set on the archive
    ("UltraTools/Finance", "activate_credit_card", "withdrawal_password"),  # set on the card
    ("UltraTools/Health", "password_management_tool", "password"),  # stored in a password manager
    ("UltraTools/Health", "password_management_tool", "password_hint"),  # a hint
    ("UltraTools/Health", "update_password", "new_password"),  # no tool in the toolset logs in with it
    ("UltraTools/Meeting", "video_conference_tool", "needPassword"),  # a yes/no flag
    ("UltraTools/Meeting", "video_conference_tool", "password"),  # set on the new meeting
    ("UltraTools/Travel", "password_management_tool", "password"),  # stored in a password manager
    ("UltraTools/Travel", "password_management_tool", "password_hint"),  # a hint
}


@pytest.fixture(scope="module")
def toolsets() -> list[tuple[str, int, list[dict]]]:
    """Each toolset as (label, num_tools, tool specs)."""
    df = pd.read_parquet(ToolCallingProbe.assets_dir() / "toolsets_seed.parquet")
    out = []
    for row in df.itertuples():
        name = "" if is_missing(row.toolset_name) else f"/{row.toolset_name}"
        out.append((f"{row.toolset_source}/{row.toolset_category}{name}", row.num_tools, json.loads(row.tools)))
    assert out, "the shipped tool catalog holds no toolsets"
    return out


def _schemas(schema: dict, path: str) -> Iterator[tuple[str, dict]]:
    """Each property and item schema inside a parameters schema, with its path."""
    for name, sub in (schema.get("properties") or {}).items():
        yield f"{path}.{name}", sub
        yield from _schemas(sub, f"{path}.{name}")
    if isinstance(schema.get("items"), dict):
        yield f"{path}[]", schema["items"]
        yield from _schemas(schema["items"], f"{path}[]")


def _tools(toolsets) -> Iterator[tuple[str, dict]]:
    for label, _, specs in toolsets:
        for spec in specs:
            yield f"{label}:{spec['function']['name']}", spec["function"]


def _fits(value, json_type: str) -> bool:
    """Whether JSON Schema's ``type`` keyword accepts the value (a boolean is no integer)."""
    return Draft202012Validator({"type": json_type}).is_valid(value)


def test_num_tools_counts_each_toolsets_specs(toolsets):
    wrong = [(label, n, len(specs)) for label, n, specs in toolsets if n != len(specs)]
    assert not wrong, f"num_tools differs from the number of specs (toolset, num_tools, specs): {wrong}"


def test_every_tool_is_a_function_spec_with_a_name_unique_in_its_toolset(toolsets):
    bad_shape = [
        f"{label}: {sorted(spec)}"
        for label, _, specs in toolsets
        for spec in specs
        if spec.get("type") != "function" or set(spec["function"]) - {"name", "description", "parameters"}
    ]
    bad_names = [where for where, f in _tools(toolsets) if not _FUNCTION_NAME.match(f["name"])]
    not_objects = [where for where, f in _tools(toolsets) if f["parameters"].get("type") != "object"]
    repeated = [
        f"{label}:{name}"
        for label, _, specs in toolsets
        for name, count in Counter(s["function"]["name"] for s in specs).items()
        if count > 1
    ]
    assert not bad_shape, f"specs outside the function envelope: {bad_shape[:10]}"
    assert not bad_names, f"names a provider rejects: {bad_names[:10]}"
    assert not not_objects, f"parameters that are not an object schema: {not_objects[:10]}"
    assert not repeated, f"names repeated within a toolset, which a provider rejects: {repeated}"


def test_parameter_names_are_ones_a_provider_accepts(toolsets):
    wrong = [
        f"{where}{path}"
        for where, f in _tools(toolsets)
        for path, _ in _schemas(f["parameters"], "")
        if not path.endswith("[]") and not _PARAMETER_NAME.match(path.rsplit(".", 1)[-1])
    ]
    assert not wrong, f"parameter names a provider rejects: {wrong[:10]}"


def test_every_schema_declares_a_json_schema_type(toolsets):
    """A missing type leaves the assistant guessing; a type outside JSON Schema fails the request."""
    wrong = [
        f"{where}{path}: {schema.get('type')!r}"
        for where, f in _tools(toolsets)
        for path, schema in _schemas(f["parameters"], "")
        if schema.get("type") not in JSON_SCHEMA_TYPES
    ]
    assert not wrong, f"schemas without a JSON Schema type: {wrong[:10]}"


def test_enum_values_have_the_declared_type(toolsets):
    wrong = [
        f"{where}{path}: {schema['type']} {schema['enum']}"
        for where, f in _tools(toolsets)
        for path, schema in _schemas(f["parameters"], "")
        if "enum" in schema and not all(_fits(v, schema["type"]) for v in schema["enum"])
    ]
    assert not wrong, f"enum values of another type: {wrong[:10]}"


def test_every_required_name_is_a_declared_parameter(toolsets):
    wrong = [
        f"{where}{path}: {sorted(set(schema['required']) - set(schema.get('properties') or {}))}"
        for where, f in _tools(toolsets)
        for path, schema in [("", f["parameters"]), *_schemas(f["parameters"], "")]
        if isinstance(schema.get("required"), list)
        and not set(schema["required"]) <= set(schema.get("properties") or {})
    ]
    assert not wrong, f"required names that are not parameters: {wrong[:10]}"


def test_parameters_hold_no_output_schema(toolsets):
    wrong = [where for where, f in _tools(toolsets) if "results" in f["parameters"]]
    assert not wrong, f"parameters schemas carrying a results schema: {wrong}"


def test_each_placeholder_in_an_endpoint_is_a_required_parameter(toolsets):
    """A tool for ``GET .../movie/{movie_id}`` can only be called with ``movie_id``."""
    wrong, checked = [], 0
    for where, f in _tools(toolsets):
        endpoint = _ENDPOINT.search(f["description"])
        if not endpoint:
            continue
        checked += 1
        params = f["parameters"]
        for name in re.findall(r"\{(\w+)\}", endpoint.group(1)):
            if name not in (params.get("properties") or {}) or name not in params.get("required", []):
                wrong.append(f"{where}: {name}")
    assert checked, "no tool description ends with an endpoint, so no placeholder was checked"
    assert not wrong, f"endpoint placeholders that are not required parameters: {wrong[:10]}"


#: The JSON Schema formats the catalog uses; anything else is a description, not a format.
_FORMATS = frozenset({"date-time", "date", "time", "email", "uri", "uuid"})
#: Keywords a parameter schema may carry: JSON Schema 2020-12 validation and annotation keywords.
_KEYWORDS = frozenset(
    {
        "type",
        "description",
        "properties",
        "required",
        "items",
        "enum",
        "default",
        "format",
        "examples",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "pattern",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "contentMediaType",
        "dependentRequired",
        "additionalProperties",
        "title",
        "const",
    }
)


def test_every_parameters_schema_is_valid_json_schema(toolsets):
    """Draft 2020-12 is the dialect providers validate a tool's parameters against."""
    wrong = []
    for where, f in _tools(toolsets):
        try:
            Draft202012Validator.check_schema(f["parameters"])
        except SchemaError as e:
            wrong.append(f"{where}: {e.message}")
    assert not wrong, f"parameters schemas that are not valid JSON Schema: {wrong[:10]}"


def test_every_array_declares_its_items(toolsets):
    wrong = [
        f"{where}{path}"
        for where, f in _tools(toolsets)
        for path, schema in _schemas(f["parameters"], "")
        if schema.get("type") == "array" and not isinstance(schema.get("items"), dict)
    ]
    assert not wrong, f"arrays without an item schema: {wrong[:10]}"


def test_schemas_use_json_schema_keywords_and_formats(toolsets):
    keywords, formats = [], []
    for where, f in _tools(toolsets):
        for path, schema in [("", f["parameters"]), *_schemas(f["parameters"], "")]:
            keywords += [f"{where}{path}: {k}" for k in schema if k not in _KEYWORDS]
            if "format" in schema and schema["format"] not in _FORMATS:
                formats.append(f"{where}{path}: {schema['format']!r}")
    assert not keywords, f"keywords outside JSON Schema: {keywords[:10]}"
    assert not formats, f"formats outside JSON Schema: {formats[:10]}"


def test_defaults_and_examples_have_the_declared_type(toolsets):
    wrong = [
        f"{where}{path}: {schema['type']} {value!r}"
        for where, f in _tools(toolsets)
        for path, schema in _schemas(f["parameters"], "")
        if schema.get("type") in JSON_SCHEMA_TYPES
        for value in ([schema["default"]] if "default" in schema else []) + list(schema.get("examples", []))
        if not _fits(value, schema["type"])
    ]
    assert not wrong, f"defaults or examples of another type: {wrong[:10]}"


def test_enum_values_are_distinct(toolsets):
    wrong = [
        f"{where}{path}"
        for where, f in _tools(toolsets)
        for path, schema in _schemas(f["parameters"], "")
        if "enum" in schema and len({json.dumps(v) for v in schema["enum"]}) != len(schema["enum"])
    ]
    assert not wrong, f"enums that repeat a value: {wrong[:10]}"


def test_every_tool_and_parameter_is_described(toolsets):
    tools = [where for where, f in _tools(toolsets) if not str(f.get("description") or "").strip()]
    params = [
        f"{where}{path}"
        for where, f in _tools(toolsets)
        for path, schema in _schemas(f["parameters"], "")
        if not path.endswith("[]") and not str(schema.get("description") or "").strip()
    ]
    assert not tools, f"tools without a description: {tools[:10]}"
    assert not params, f"parameters without a description: {params[:10]}"


def test_each_entry_holds_its_function_spec_and_at_most_its_credentials(toolsets):
    allowed = {"type", "function", "credentials", "sets_credentials"}
    extra = [
        f"{label}:{spec['function']['name']}: {sorted(set(spec) - allowed)}"
        for label, _, specs in toolsets
        for spec in specs
        if set(spec) - allowed
    ]
    assert not extra, f"entries with keys beside the spec and its credentials: {extra[:10]}"


def test_each_declared_credential_is_a_required_string_its_description_names(toolsets):
    wrong = []
    n_declared = 0
    for label, _, specs in toolsets:
        for spec in specs:
            parameters = spec["function"]["parameters"]
            for parameter, credential in declared_credentials(spec).items():
                n_declared += 1
                schema = parameters["properties"][parameter]
                if (
                    parameter not in parameters.get("required", [])
                    or schema.get("type") != "string"
                    or credential not in schema.get("description", "")
                ):
                    wrong.append(f"{label}:{spec['function']['name']}.{parameter} ({credential})")
    assert n_declared, "the shipped catalog declares no credentials"
    assert not wrong, f"credentials that are optional, not strings, or described without their name: {wrong[:10]}"


def test_every_key_parameter_is_declared_as_a_credential(toolsets):
    undeclared = [
        f"{label}:{spec['function']['name']}.{parameter}"
        for label, _, specs in toolsets
        for spec in specs
        for parameter in spec["function"]["parameters"].get("properties") or {}
        if _KEY_NAME.search(parameter) and parameter not in (spec.get("credentials") or {})
    ]
    assert not undeclared, f"key parameters whose credential the user would not hold: {undeclared[:10]}"


def test_every_password_is_held_set_or_listed_as_neither(toolsets):
    unclassified = [
        f"{label}:{spec['function']['name']}.{parameter}"
        for label, _, specs in toolsets
        for spec in specs
        for parameter in spec["function"]["parameters"].get("properties") or {}
        if _PASSWORD_NAME.search(parameter)
        and parameter not in declared_credentials(spec)
        and parameter not in credentials_set_by(spec)
        and (label, spec["function"]["name"], parameter) not in _PASSWORDS_NOT_HELD
    ]
    assert not unclassified, f"password parameters neither held, set nor listed: {unclassified[:10]}"


def test_each_set_credential_replaces_one_its_toolset_holds(toolsets):
    """A tool that sets a credential takes a string, and some tool of its toolset logs in with it."""
    wrong = []
    n_set = 0
    for label, _, specs in toolsets:
        held = {credential for spec in specs for credential in declared_credentials(spec).values()}
        for spec in specs:
            for parameter, credential in credentials_set_by(spec).items():
                n_set += 1
                schema = spec["function"]["parameters"]["properties"][parameter]
                if schema.get("type") != "string" or credential not in held:
                    wrong.append(f"{label}:{spec['function']['name']}.{parameter} ({credential})")
    assert n_set, "the shipped catalog sets no credentials"
    assert not wrong, f"set credentials that are not strings or that no tool of the toolset holds: {wrong[:10]}"
