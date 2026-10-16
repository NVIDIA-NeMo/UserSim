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
    commands = [step["run"] for step in steps if "run" in step]
    assert commands[:5] == [
        "make install-ci",
        "make validate-native",
        "make check-dist",
        "make check-release-tag",
        "make test-e2e WHEEL_DIR=dist",
    ]
    assert len(commands) == 6
    assert commands[5] == "make release-digests"
    assert workflow()["jobs"]["build"].get("environment") is None
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
