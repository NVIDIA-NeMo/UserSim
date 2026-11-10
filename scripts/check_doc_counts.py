# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the probe and scorer counts the docs state match what ships.

The README headline, ``docs/probes.md`` and ``docs/outputs.md`` say how many
probes and scorers User Sim ships. A contributor who adds a probe doesn't edit
those numbers; the release workflow runs this before anything is published, so
a release can't ship a stale count. Each count appears in one sentence, written
as a numeral, which :data:`SENTENCES` describes.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Sentence:
    """A sentence that states one count, with ``{n}`` standing for the number."""

    path: str
    count: str
    text: str

    @property
    def pattern(self) -> re.Pattern[str]:
        before, after = (re.escape(part) for part in self.text.split("{n}"))
        return re.compile(before + r"(\d+)" + after)


#: Every sentence that states a count. ``count`` names an entry of :func:`registry_counts`.
SENTENCES: tuple[Sentence, ...] = (
    Sentence("README.md", "probes", "**{n} probes ·"),
    Sentence("README.md", "scorers", "· {n} evaluator scorers ·"),
    Sentence("docs/probes.md", "probes", "User Sim ships {n} probes,"),
    Sentence("docs/probes.md", "bank_backed", "Of the probes, {n} draw on curated banks"),
    Sentence("docs/outputs.md", "scorers", "scorers, of which {n} ship today"),
)


def registry_counts() -> dict[str, int]:
    """How many probes, scorers and bank-backed probes are registered."""
    import usersim.engine.generator  # noqa: F401  (registers the built-in probes)
    from usersim.engine.core.probes import BankBackedProbe, known_probes, resolve_probe
    from usersim.engine.evaluator.scorers import list_scorers

    probes = known_probes()
    return {
        "probes": len(probes),
        "scorers": len(list_scorers()),
        "bank_backed": sum(issubclass(resolve_probe(label), BankBackedProbe) for label in probes),
    }


def check_counts(root: Path, counts: Mapping[str, int]) -> list[str]:
    """One problem for each sentence that is missing, repeated, or states the wrong number."""
    problems = []
    for sentence in SENTENCES:
        # Collapse whitespace, so a sentence still matches wherever its line wraps.
        text = " ".join((root / sentence.path).read_text(encoding="utf-8").split())
        found = sentence.pattern.findall(text)
        expected = sentence.text.replace("{n}", str(counts[sentence.count]))
        if len(found) != 1:
            problems.append(f"{sentence.path}: expected the sentence '{expected}' once, found it {len(found)} times")
        elif int(found[0]) != counts[sentence.count]:
            problems.append(f"{sentence.path}: says {found[0]} where it should say '{expected}'")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    counts = registry_counts()
    problems = check_counts(ROOT, counts)
    if problems:
        print("✗ the documented counts don't match what ships:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"✓ the docs state {counts['probes']} probes and {counts['scorers']} scorers, as registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
