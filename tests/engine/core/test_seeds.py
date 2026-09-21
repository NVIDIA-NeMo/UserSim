# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Tests for core/seeds.py — per-probe seed loader with locale fallback."""

import json
from pathlib import Path

import pytest

from usersim.engine.core.seeds import load_seeds


@pytest.fixture
def seed_dir(tmp_path):
    """Build the per-probe layout this loader expects:

    tmp_path/
      general_open_ended/topics/{base,pt_BR}/topics.json
      general_educational/subjects/base/subjects.json
    """
    oe_base = tmp_path / "general_open_ended" / "topics" / "base"
    oe_base.mkdir(parents=True)
    base_topics = [
        {"type": "health", "description": "Health topics"},
        {"type": "finance", "description": "Finance topics"},
    ]
    (oe_base / "topics.json").write_text(json.dumps(base_topics))

    oe_pt = tmp_path / "general_open_ended" / "topics" / "pt_BR"
    oe_pt.mkdir(parents=True)
    pt_topics = [{"type": "sus", "description": "SUS healthcare"}]
    (oe_pt / "topics.json").write_text(json.dumps(pt_topics))

    ed_base = tmp_path / "general_educational" / "subjects" / "base"
    ed_base.mkdir(parents=True)
    (ed_base / "subjects.json").write_text(json.dumps([{"type": "math", "description": "Mathematics"}]))

    return tmp_path


class TestLoadSeeds:
    def test_loads_base_only(self, seed_dir: Path) -> None:
        seeds = load_seeds("en_US", "general_open_ended", "topics", seed_dir)
        assert len(seeds) == 2
        assert seeds[0]["type"] == "health"

    def test_locale_replaces_base(self, seed_dir: Path) -> None:
        seeds = load_seeds("pt_BR", "general_open_ended", "topics", seed_dir)
        assert len(seeds) == 1
        assert seeds[0]["type"] == "sus"

    def test_missing_kind_raises(self, seed_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_seeds("en_US", "general_open_ended", "nonexistent", seed_dir)

    def test_missing_probe_raises(self, seed_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_seeds("en_US", "no_such_probe", "topics", seed_dir)

    def test_locale_without_translation_falls_back_to_base(self, seed_dir: Path) -> None:
        seeds = load_seeds("ja_JP", "general_open_ended", "topics", seed_dir)
        assert len(seeds) == 2

    def test_educational_kind_loads(self, seed_dir: Path) -> None:
        seeds = load_seeds("en_US", "general_educational", "subjects", seed_dir)
        assert seeds[0]["type"] == "math"
