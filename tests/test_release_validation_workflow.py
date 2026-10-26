# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release gates must validate the archives handed to the publisher."""

from pathlib import Path

import yaml


def workflow():
    path = Path(__file__).resolve().parents[1] / ".github/workflows/release.yml"
    return yaml.safe_load(path.read_text())


def test_native_gates_and_original_wheel_run_before_archive_upload():
    steps = workflow()["jobs"]["build"]["steps"]
    commands = [
        step["run"]
        for step in steps
        if step.get("run", "").startswith("scripts/mise run ") and step["run"] != "scripts/mise run setup:tools"
    ]
    assert commands == [
        "scripts/mise run setup:ci",
        "scripts/mise run validate:native",
        "scripts/mise run check:dist",
        "scripts/mise run check:release-tag",
        "scripts/mise run test:e2e --wheel-dir dist",
        "scripts/mise run release:digests",
    ]
    assert workflow()["jobs"]["build"].get("environment") is None
    e2e = next(step for step in steps if step.get("run") == "scripts/mise run test:e2e --wheel-dir dist")
    assert e2e["env"]["WHEEL_DIR"] == "dist"
    uploads = [i for i, step in enumerate(steps) if step.get("uses", "").startswith("actions/upload-artifact@")]
    assert len(uploads) == 1
    assert uploads[0] == len(steps) - 1
    assert steps[uploads[0]]["with"] == {"name": "dist", "path": "dist/", "if-no-files-found": "error"}


def test_publisher_downloads_original_archives_and_cannot_rebuild():
    job = workflow()["jobs"]["release"]
    assert job["needs"] == "build"
    steps = job["steps"]
    downloads = [step for step in steps if step.get("uses", "").startswith("actions/download-artifact@")]
    assert len(downloads) == 1
    assert downloads[0]["with"] == {"name": "dist", "path": "dist"}
    assert not any("checkout@" in step.get("uses", "") for step in steps)
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "uv build" not in commands
    assert "make dist" not in commands
    assert 'gh release create "$TAG" dist/*' in commands
