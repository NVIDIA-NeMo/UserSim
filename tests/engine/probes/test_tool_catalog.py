# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tool catalog the tool_calling probe ships.

Each row's tools reach the assistant model unchanged, as OpenAI-style function
specs, so every spec in ``toolsets_seed.parquet`` must be one a provider accepts.
"""

import json
import re
from collections import Counter
from collections.abc import Iterator

import pandas as pd
import pytest

from usersim.engine.core.missing import is_missing
from usersim.engine.probes.tool_calling.generator import ToolCallingProbe

JSON_SCHEMA_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object", "null"})
_PYTHON_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}
_FUNCTION_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_PARAMETER_NAME = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")
#: The method and URL a REST tool's description ends with.
_ENDPOINT = re.compile(r" HTTP [A-Z]+ (\S+)$")


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
    if isinstance(value, bool) and json_type in ("number", "integer"):
        return False
    return isinstance(value, _PYTHON_TYPES[json_type])


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
