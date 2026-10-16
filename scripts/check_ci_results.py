# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject incomplete or unsuccessful mandatory GitHub Actions job results."""

from __future__ import annotations

import json
import os
import sys

REQUIRED_JOBS = frozenset({"static", "test", "packaging", "e2e", "container"})


def validate_results(raw: str) -> list[str]:
    """Return failures for the exact required job set; no skips are permitted."""
    try:
        results = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ["CI_JOB_RESULTS must contain the JSON object produced by toJSON(needs)"]
    if not isinstance(results, dict):
        return ["CI_JOB_RESULTS must be a job-result object"]
    failures = []
    for job in sorted(REQUIRED_JOBS - results.keys()):
        failures.append(f"{job}: missing required job")
    for job in sorted(results.keys() - REQUIRED_JOBS):
        failures.append(f"{job}: unexpected job; update the explicit required job registry")
    for job in sorted(REQUIRED_JOBS & results.keys()):
        entry = results[job]
        result = entry.get("result") if isinstance(entry, dict) else None
        if result != "success":
            failures.append(f"{job}: expected success, received {result!r}")
    return failures


def main() -> int:
    failures = validate_results(os.environ.get("CI_JOB_RESULTS", ""))
    if failures:
        print("Mandatory CI jobs did not all succeed:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    print("Every mandatory CI job succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
