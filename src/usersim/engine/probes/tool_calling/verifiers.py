# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-call verification against JSON schemas for the tool_calling probe."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


class ToolCallVerifier:
    """Validates LLM-generated tool calls against tool JSON schemas."""

    def verify(
        self,
        tool_calls: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Tuple[bool, List[Dict[str, Any]], List[bool], List[Any], List[Dict[str, Any]]]:
        """Verify a list of tool calls against available tool definitions.

        Returns:
            (all_correct, error_messages, per_call_success, per_call_errors, correct_tool_calls)
        """
        if not tool_calls:
            return True, [], [], [], []
        all_success: List[bool] = []
        all_error: List[Any] = []
        tool_error_messages: List[Dict[str, Any]] = []
        correct_tool_calls: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            ok, err = self.verify_single_tool_call(tool_call, tools)
            all_success.append(ok)
            all_error.append(err)
            if not ok:
                tool_error_messages.append(
                    {
                        "role": "tool",
                        "content": (
                            f"Error in calling tool `{tool_call['function']['name']}` "
                            f"with arguments `{tool_call['function']['arguments']}`: {err}"
                        ),
                        "tool_call_id": tool_call["id"],
                    }
                )
            else:
                correct_tool_calls.append(tool_call)
        return all(all_success), tool_error_messages, all_success, all_error, correct_tool_calls

    def verify_single_tool_call(
        self,
        tool_call: Dict[str, Any],
        tools: List[Dict[str, Any]],
    ) -> Tuple[bool, str | None]:
        """Verify a single tool call against the available tool definitions."""
        try:
            tool_name = tool_call.get("function", {}).get("name")
            tool_arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))

            if not tool_name:
                return False, f"Invalid tool call: `{json.dumps(tool_call)}`"

            target_tool = None
            for tool_data in tools:
                if isinstance(tool_data, dict) and "tool" in tool_data:
                    tool_spec = tool_data["tool"]
                else:
                    tool_spec = tool_data
                function_spec = tool_spec.get("function", {})
                if function_spec.get("name") == tool_name:
                    target_tool = function_spec
                    break

            if not target_tool:
                return False, f"Tool `{tool_name}` name not found in available tools."

            parameters = target_tool.get("parameters", {})
            properties = parameters.get("properties", {})
            required_params = parameters.get("required", [])

            for required_param in required_params:
                if required_param not in tool_arguments:
                    return False, f"Required parameter `{required_param}` not found in tool call arguments."

            if isinstance(tool_arguments, str):
                tool_arguments = json.loads(tool_arguments)

            for arg_name, arg_value in tool_arguments.items():
                if arg_name not in properties:
                    return (
                        False,
                        f"Argument `{arg_name}` present in tool call arguments but not defined in tool schema.",
                    )

                param_spec = properties[arg_name]

                is_valid, error_message = self._validate_parameter_type(arg_value, param_spec)
                if not is_valid:
                    return False, f"Invalid argument `{arg_name}`: {error_message}"

                if "enum" in param_spec:
                    if arg_value not in param_spec["enum"]:
                        return (
                            False,
                            f"Unexpected value `{arg_value}` for argument `{arg_name}`, expected one of: {param_spec['enum']}",
                        )

            return True, None

        except Exception as e:
            return False, f"Unexpected error: {e}"

    def _validate_parameter_type(
        self,
        value: Any,
        param_spec: Dict[str, Any],
    ) -> Tuple[bool, str | None]:
        expected_type = param_spec.get("type")

        if expected_type == "string":
            if isinstance(value, str):
                return True, None
            return False, f"Expected type 'string', got {type(value)} for `{value}`"
        elif expected_type == "number":
            if isinstance(value, (int, float)):
                return True, None
            return False, f"Expected type 'number', got {type(value)} for `{value}`"
        elif expected_type == "integer":
            if isinstance(value, int):
                return True, None
            return False, f"Expected type 'integer', got {type(value)} for `{value}`"
        elif expected_type == "boolean":
            if isinstance(value, bool):
                return True, None
            return False, f"Expected type 'boolean', got {type(value)} for `{value}`"
        elif expected_type == "array":
            if not isinstance(value, list):
                return False, f"Expected type 'array', got {type(value)} for `{value}`"
            items_spec = param_spec.get("items")
            if items_spec:
                for item in value:
                    ok, err = self._validate_parameter_type(item, items_spec)
                    if not ok:
                        return False, f"Invalid item `{err}` in array `{value}`"
            return True, None
        elif expected_type == "object":
            if not isinstance(value, dict):
                return False, f"Expected type 'object', got {type(value)} for `{value}`"
            properties = param_spec.get("properties", {})
            required = param_spec.get("required", [])
            for req_prop in required:
                if req_prop not in value:
                    return False, f"Required property `{req_prop}` missing in object `{value}`"
            if isinstance(value, str):
                value = json.loads(value)
            for prop_name, prop_value in value.items():
                if prop_name in properties:
                    ok, err = self._validate_parameter_type(prop_value, properties[prop_name])
                    if not ok:
                        return False, f"Invalid property `{prop_name}`: {err}"
            return True, None
        elif expected_type is None:
            return True, None
        else:
            return True, None
