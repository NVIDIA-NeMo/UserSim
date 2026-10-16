# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The aggregate cannot report success when required jobs did not succeed."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_ci_results.py"
spec = importlib.util.spec_from_file_location("check_ci_results", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def successful_results():
    return {job: {"result": "success"} for job in checker.REQUIRED_JOBS}


def test_all_required_jobs_successful():
    assert checker.validate_results(json.dumps(successful_results())) == []


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", "pending", None])
@pytest.mark.parametrize("job", sorted(checker.REQUIRED_JOBS))
def test_each_unsuccessful_job_rejects_aggregate(job, result):
    results = successful_results()
    results[job] = {"result": result}
    assert checker.validate_results(json.dumps(results))


def test_missing_and_unexpected_jobs_are_rejected():
    results = successful_results()
    del results["e2e"]
    results["replacement"] = {"result": "success"}
    failures = checker.validate_results(json.dumps(results))
    assert any("e2e: missing" in failure for failure in failures)
    assert any("replacement: unexpected" in failure for failure in failures)


@pytest.mark.parametrize("raw", ["", "null", "[]", "{broken", '{"static": "success"}'])
def test_invalid_results_are_rejected(raw):
    assert checker.validate_results(raw)


def test_script_exit_status_matches_result_contract():
    for results, expected in [(successful_results(), 0), ({}, 1)]:
        env = dict(os.environ, CI_JOB_RESULTS=json.dumps(results))
        result = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True)
        assert result.returncode == expected
