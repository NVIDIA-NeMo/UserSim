# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``usersim panel`` and the persona panel file it writes."""

from __future__ import annotations

import pandas as pd

from usersim import cli
from usersim.cli._models import bundled_models_path


def test_panel_reopens_with_plain_pandas(tmp_path, monkeypatch):
    """A panel is a file people open themselves, nested persona column included."""
    # Data Designer writes its run artifacts under the working directory.
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "panel.parquet"
    args = ["panel", "--locale", "en_US", "--num-personas", "2", "--out", str(out)]

    assert cli.main([*args, "--models", str(bundled_models_path())]) == 0

    panel = pd.read_parquet(out)
    assert len(panel) == 2
    assert set(panel["locale"]) == {"en_US"}
    assert isinstance(panel["persona"].iloc[0], dict)
