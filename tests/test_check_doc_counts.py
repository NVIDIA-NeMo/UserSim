# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/check_doc_counts.py`` finds each documented count and reports one that doesn't match."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_doc_counts.py"
_spec = importlib.util.spec_from_file_location("check_doc_counts", _SCRIPT)
check_doc_counts = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check_doc_counts  # its dataclass looks its module up there
_spec.loader.exec_module(check_doc_counts)

_COUNTS = {"probes": 14, "scorers": 12, "bank_backed": 11}


def _docs(root: Path, counts: dict[str, int]) -> Path:
    """A docs tree holding each counted sentence once, stating ``counts``."""
    for sentence in check_doc_counts.SENTENCES:
        path = root / sentence.path
        path.parent.mkdir(parents=True, exist_ok=True)
        text = sentence.text.replace("{n}", str(counts[sentence.count]))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"Some prose.\n{text} more prose.\n")
    return root


@pytest.mark.parametrize("sentence", check_doc_counts.SENTENCES, ids=lambda s: f"{s.path}:{s.count}")
def test_the_docs_state_each_count_in_one_sentence(sentence) -> None:
    """The release check can only catch a stale count while its sentence is still there to read."""
    text = " ".join((_ROOT / sentence.path).read_text(encoding="utf-8").split())
    assert len(sentence.pattern.findall(text)) == 1, f"{sentence.path} should contain '{sentence.text}' once"


def test_matching_counts_pass(tmp_path: Path) -> None:
    assert check_doc_counts.check_counts(_docs(tmp_path, _COUNTS), _COUNTS) == []


def test_a_wrong_count_names_the_file_and_the_sentence(tmp_path: Path) -> None:
    root = _docs(tmp_path, _COUNTS)
    problems = check_doc_counts.check_counts(root, {**_COUNTS, "probes": 15})
    assert problems == [
        "README.md: says 14 where it should say '**15 probes ·'",
        "docs/probes.md: says 14 where it should say 'User Sim ships 15 probes,'",
    ]


def test_a_missing_sentence_is_reported(tmp_path: Path) -> None:
    root = _docs(tmp_path, _COUNTS)
    (root / "docs" / "outputs.md").write_text("No count here.\n", encoding="utf-8")
    problems = check_doc_counts.check_counts(root, _COUNTS)
    assert problems == [
        "docs/outputs.md: expected the sentence 'scorers, of which 12 ship today' once, found it 0 times"
    ]


def test_a_sentence_wrapped_across_lines_still_counts(tmp_path: Path) -> None:
    root = _docs(tmp_path, _COUNTS)
    (root / "docs" / "outputs.md").write_text("a set of scorers, of which 12\nship today (...)\n", encoding="utf-8")
    assert check_doc_counts.check_counts(root, _COUNTS) == []
