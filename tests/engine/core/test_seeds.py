# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core/seeds.py: the per-probe seed loader and its en_US fallback."""

import json
from pathlib import Path

import pytest

from usersim.engine.core.seeds import load_seeds


@pytest.fixture
def seed_dir(tmp_path):
    """Build the per-probe layout this loader expects:

    tmp_path/
      general_open_ended/{en_US,pt_BR}/topics.json
      general_educational/en_US/subjects.json
    """
    oe_us = tmp_path / "general_open_ended" / "en_US"
    oe_us.mkdir(parents=True)
    us_topics = [
        {"type": "health", "description": "Health topics"},
        {"type": "finance", "description": "Finance topics"},
    ]
    (oe_us / "topics.json").write_text(json.dumps(us_topics))

    oe_pt = tmp_path / "general_open_ended" / "pt_BR"
    oe_pt.mkdir(parents=True)
    pt_topics = [{"type": "sus", "description": "SUS healthcare"}]
    (oe_pt / "topics.json").write_text(json.dumps(pt_topics))

    ed_us = tmp_path / "general_educational" / "en_US"
    ed_us.mkdir(parents=True)
    (ed_us / "subjects.json").write_text(json.dumps([{"type": "math", "description": "Mathematics"}]))

    return tmp_path


class TestLoadSeeds:
    def test_en_us_reads_its_own_file(self, seed_dir: Path) -> None:
        seeds = load_seeds("en_US", "general_open_ended", "topics", seed_dir)
        assert len(seeds) == 2
        assert seeds[0]["type"] == "health"

    def test_a_locale_file_replaces_the_en_us_one(self, seed_dir: Path) -> None:
        seeds = load_seeds("pt_BR", "general_open_ended", "topics", seed_dir)
        assert len(seeds) == 1
        assert seeds[0]["type"] == "sus"

    def test_missing_kind_raises(self, seed_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_seeds("en_US", "general_open_ended", "nonexistent", seed_dir)

    def test_missing_probe_raises(self, seed_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_seeds("en_US", "no_such_probe", "topics", seed_dir)

    def test_a_locale_without_a_file_reads_the_en_us_one(self, seed_dir: Path) -> None:
        seeds = load_seeds("ja_JP", "general_open_ended", "topics", seed_dir)
        assert [s["type"] for s in seeds] == ["health", "finance"]

    def test_educational_kind_loads(self, seed_dir: Path) -> None:
        seeds = load_seeds("en_US", "general_educational", "subjects", seed_dir)
        assert seeds[0]["type"] == "math"
