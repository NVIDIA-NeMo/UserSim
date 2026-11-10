# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The health probes share one package, and their assets are found there before any probe is registered."""

from __future__ import annotations

import subprocess
import sys


def test_a_probe_in_a_shared_package_finds_its_assets_in_a_fresh_process() -> None:
    code = (
        "from usersim.engine.core import probes\n"
        "from usersim.engine.core._assets import probe_assets_dir\n"
        "assert not probes._PROBE_REGISTRY, 'a probe was registered before the lookup'\n"
        "print(probe_assets_dir('health_triage_disclosure').parent.parent.name)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "health_disclosure"
