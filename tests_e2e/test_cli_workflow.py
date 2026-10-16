# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real CLI, HTTP transport, persisted rows, and downstream artifact contracts."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow.parquet as pq
import pytest

RUNTIME = Path(os.environ["E2E_RUNTIME"])
ARTIFACTS = Path(os.environ["E2E_ARTIFACTS"])


class Provider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        self.requests = []
        self.mode = "ok"
        self.failures = 0
        self.transient_remaining = 0
        super().__init__(("127.0.0.1", 0), Handler)


class Handler(BaseHTTPRequestHandler):
    server: Provider

    def log_message(self, format: str, *args: object) -> None:
        pass

    def send(self, code, body):
        encoded = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        self.send(200, {"object": "list", "data": []})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        model = body["model"]
        if model == "assistant_model" and body.get("tools") and self.server.transient_remaining:
            self.server.transient_remaining -= 1
            self.server.failures += 1
            self.send(500, {"error": {"message": "deliberate transient local failure", "type": "server_error"}})
            return
        if model == "evaluator_model" and self.server.mode == "http-error":
            self.server.failures += 1
            self.send(400, {"error": {"message": "deliberate local provider failure", "type": "invalid_request_error"}})
            return
        message = {"role": "assistant", "content": "A clear answer about the requested task."}
        if body.get("response_format", {}).get("type") == "json_schema":
            schema = body["response_format"]["json_schema"]["schema"]
            message["content"] = json.dumps(
                {
                    key: {"score": 4, "reasoning": "The response addresses the synthetic task."}
                    for key in schema["properties"]
                }
            )
            if model == "evaluator_model" and self.server.mode == "malformed":
                self.server.failures += 1
                message["content"] = "this is not JSON"
        elif model == "judge_model":
            message["content"] = "<explanation>The request is appropriate.</explanation><rating>success</rating>"
        elif model == "summary_model":
            message["content"] = "no"
        elif model == "api_response_model":
            message["content"] = '{"status": "ok", "temperature": 22}'
        elif model == "user_model" and body.get("tools"):
            message.update(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_move",
                            "type": "function",
                            "function": {
                                "name": "commit_move",
                                "arguments": json.dumps({"move": "express_distress", "topic": "", "intensity": 0.3}),
                            },
                        }
                    ],
                }
            )
        elif model == "user_model":
            prompt = json.dumps(body["messages"], ensure_ascii=False)
            message["content"] = (
                "東京の天気を調べてください。今日の外出の予定に使いたいです。"
                if "Japanese" in prompt
                else "Please check today's weather in Austin for my afternoon walk."
            )
        elif model == "assistant_model" and body.get("tools") and body.get("tool_choice") != "none":
            name = body["tools"][0]["function"]["name"]
            message.update(
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_e2e",
                            "type": "function",
                            "function": {"name": name, "arguments": '{"city":"Austin"}'},
                        }
                    ],
                }
            )
        self.send(
            200,
            {
                "id": "chatcmpl-local-e2e",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )


@contextmanager
def provider(work):
    server = Provider()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/health", timeout=5) as response:
            assert response.status == 200
        yield server
    finally:
        (work / "requests.json").write_text(json.dumps(server.requests, indent=2, ensure_ascii=False), encoding="utf-8")
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


def cli(cwd, name, *args, expected=0):
    try:
        result = subprocess.run(
            [str(RUNTIME / "bin/usersim"), *map(str, args)],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        (cwd / f"{name}.log").write_bytes((error.stdout or b"") + (error.stderr or b""))
        raise
    (cwd / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == expected, f"{name} exited {result.returncode}:\n{result.stdout}\n{result.stderr}"
    return result


def rows(path):
    files = sorted(path.rglob("part-*.parquet"))
    assert files, f"No persisted Parquet under {path}"
    return [
        {
            **{
                part.split("=", 1)[0]: part.split("=", 1)[1]
                for part in file.relative_to(path).parts[:-1]
                if "=" in part
            },
            **row,
        }
        for file in files
        for row in pq.ParquetFile(file).read().to_pylist()
    ]


def models(path, server):
    aliases = ("user_model", "assistant_model", "judge_model", "summary_model", "api_response_model", "evaluator_model")
    text = (
        "models = [\n"
        + ",\n".join(
            f'{{alias="{alias}", model="{alias}", provider="local", max_tokens=256, timeout=5, max_parallel_requests=1, extra_body={{e2e_marker="preserved", chat_template_kwargs={{enable_thinking=true, e2e_nested="preserved"}}}}}}'
            for alias in aliases
        )
        + f'\n]\n[[providers]]\nname="local"\nendpoint="http://127.0.0.1:{server.server_port}/v1"\nprovider_type="openai"\napi_key="E2E_API_KEY"\n'
    )
    path.write_text(text, encoding="utf-8")


def inputs(path):
    # Same stored-row/config fields used by test_cli_simulate_inputs.py. These
    # synthetic personas require no downloaded population or sampler patches.
    common = {
        "name": "conversation_messages",
        "column_type": "conversation-simulator",
        "max_turns": 1,
        "max_query_attempts": 1,
        "tools_column": "tools",
        "toolset_name_column": "toolset_name",
        "assets_dir": os.environ["E2E_ASSETS"],
        "random_seed": 1,
    }
    tool = {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Read current weather in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
    fixture = []
    for index, (locale, probe, name) in enumerate(
        (("en_US", "tool_calling", "Sarah"), ("ja_JP", "general_open_ended", "花子"))
    ):
        fixture.append(
            {
                "trajectory_id": f"synthetic-input-{index}",
                "persona_uuid": f"synthetic-persona-{index}",
                "locale": locale,
                "persona": {
                    "first_name": name,
                    "last_name": "Test",
                    "age": 42,
                    "city": "Austin",
                    "occupation": "teacher",
                    "education_level": "bachelors",
                },
                "probe_type": probe,
                "probe_variant": "default",
                "theme": {
                    "type": "weather",
                    "description": "Plan an afternoon walk.",
                    "tool_expected": probe == "tool_calling",
                },
                "tools": [tool],
                "toolset_name": "weather",
                "toolset_category": "weather",
                "usersim_config": {**common, "locale": locale},
            }
        )
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in fixture), encoding="utf-8")
    return fixture


@pytest.fixture(scope="module")
def completed_workflow():
    work = ARTIFACTS / "workflow"
    work.mkdir(parents=True, exist_ok=True)
    assert not (work / "src").exists()
    with provider(work) as server:
        config = work / "models.toml"
        models(config, server)
        source = work / "inputs.jsonl"
        fixture = inputs(source)
        server.transient_remaining = 1
        trajectories, evaluations = work / "trajectories", work / "evaluations"
        cli(work, "simulate", "simulate", "--inputs", source, "--models", config, "--out", trajectories)
        assert server.failures == 1
        saved = rows(trajectories)
        assert len(saved) == 2
        assert {row["input_id"] for row in saved} == {row["trajectory_id"] for row in fixture}
        assert {row["locale"] for row in saved} == {"en_US", "ja_JP"}
        for row in saved:
            outcome = json.loads(row["simulation_outcome"])
            assert outcome["status"] == "ok", outcome
            messages = json.loads(row["conversation_messages"])
            assert any(message["role"] == "assistant" for message in messages)
            if row["locale"] == "ja_JP":
                assert "東京" in json.dumps(messages, ensure_ascii=False)
            else:
                assert outcome["n_tool_calls"] >= 1
                assert any(message["role"] == "tool" for message in messages)
        before = {row["trajectory_id"] for row in saved}
        first_runs = sorted(p.name for p in trajectories.glob("run=*"))
        assert len(first_runs) == 1
        assert before.isdisjoint({row["trajectory_id"] for row in fixture})
        calls = len(server.requests)
        cli(work, "resume", "simulate", "--inputs", source, "--models", config, "--out", trajectories)
        assert len(rows(trajectories)) == 2
        assert {row["trajectory_id"] for row in rows(trajectories)} == before
        assert sorted(p.name for p in trajectories.glob("run=*")) == first_runs
        assert len(server.requests) == calls, "resume made provider calls for already saved rows"
        cli(work, "eval", "eval", "--trajectories", trajectories, "--models", config, "--out", evaluations)
        assert sorted(p.name for p in evaluations.glob("run=*")) == first_runs
        judged = rows(evaluations)
        assert len(judged) == 2
        assert {row["trajectory_id"] for row in judged} == before
        for row in judged:
            cell = json.loads(row["assistant_eval"])
            assert not cell["skipped"]
            assert cell["axes"]
            assert all(result["evaluator_model"]["score"] == 4 for result in cell["axes"].values())
        reports, curated = work / "reports", work / "curated"
        cli(work, "report", "report", "--trajectories", trajectories, "--evaluations", evaluations, "--out", reports)
        html = list(reports.rglob("*.html"))
        assert html and html[0].stat().st_size > 1000
        cli(work, "select", "select", "--trajectories", trajectories, "--evaluations", evaluations, "--out", curated)
        selected = rows(curated)
        assert len(selected) == 2
        assert {row["trajectory_id"] for row in selected} == before
        assert list(curated.rglob("_selection_manifest.json"))
        assistant = [request for request in server.requests if request["model"] == "assistant_model"]
        assert any(request.get("tools") for request in assistant)
        assert all(request.get("e2e_marker") == "preserved" for request in assistant)
        assert any(request.get("response_format", {}).get("type") == "json_schema" for request in server.requests)

    return trajectories


def test_installed_cli_workflow(completed_workflow):
    assert len(rows(completed_workflow)) == 2


def test_external_network_blocked():
    telemetry = subprocess.run(
        [
            str(RUNTIME / "bin/python"),
            "-c",
            "from data_designer.engine.models.telemetry import TELEMETRY_ENABLED; from data_designer.engine.models.facade import TELEMETRY_ENABLED as facade_enabled; assert not TELEMETRY_ENABLED and not facade_enabled",
        ],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert telemetry.returncode == 0, telemetry.stderr

    result = subprocess.run(
        [str(RUNTIME / "bin/python"), "-c", "import socket; socket.create_connection(('example.com',443),timeout=1)"],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "E2E blocks external network" in result.stderr


@pytest.mark.parametrize("mode", ["malformed", "http-error"])
def test_provider_failure_is_not_a_score(mode, completed_workflow):
    trajectories = completed_workflow
    work = ARTIFACTS / mode
    work.mkdir(parents=True, exist_ok=True)
    with provider(work) as server:
        server.mode = mode
        config = work / "models.toml"
        models(config, server)
        output = work / "evaluations"
        result = cli(
            work,
            "eval",
            "eval",
            "--trajectories",
            trajectories,
            "--models",
            config,
            "--axes",
            "helpfulness",
            "--out",
            output,
            expected=1,
        )
        assert server.failures >= 1
        assert (
            "no judge returned a verdict" in result.stderr
            if mode == "malformed"
            else "deliberate local provider failure" in result.stderr
        )
        if mode == "malformed":
            cells = [json.loads(row["assistant_eval"]) for row in rows(output)]
            assert len(cells) == 2
            assert all(cell["axes"] == {"helpfulness": {}} for cell in cells)
        else:
            assert not list(output.rglob("part-*.parquet"))


def test_tool_choice_and_call_extra_body():
    work = ARTIFACTS / "tool-choice"
    work.mkdir(parents=True, exist_ok=True)
    with provider(work) as server:
        config = work / "models.toml"
        models(config, server)
        source = work / "inputs.jsonl"
        row = inputs(source)[0]
        row["probe_type"] = "health_therapy_disclosure"
        row["probe_variant"] = "guarded"
        row["usersim_config"]["max_turns"] = 3
        source.write_text(json.dumps(row) + "\n", encoding="utf-8")
        cli(work, "simulate", "simulate", "--inputs", source, "--models", config, "--out", work / "trajectories")
        moves = [
            request
            for request in server.requests
            if request.get("tools") and request["tools"][0]["function"]["name"] == "commit_move"
        ]
        assert moves, "guarded follow-up did not call the real move HTTP path"
        assert all(request["tool_choice"] == "auto" for request in moves)
        assert all(request["e2e_marker"] == "preserved" for request in moves)
        assert all(
            request.get("chat_template_kwargs") == {"enable_thinking": False, "e2e_nested": "preserved"}
            for request in moves
        )
        outcome = json.loads(rows(work / "trajectories")[0]["simulation_outcome"])
        assert outcome["status"] == "completed_with_warnings", outcome
        assert {warning["kind"] for warning in outcome["warnings"]} == {"used_placeholder_clinical_profile"}
        assert outcome["failure_class"] is None
        assert outcome["n_turns"] == 3
