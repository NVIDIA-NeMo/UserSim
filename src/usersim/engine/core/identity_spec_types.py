# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Typed views of the identity_disclosure spec, and how model ids and names are matched.

``usersim.engine.core.identity_spec`` loads and validates specs into these
types and re-exports everything public here.
"""

from __future__ import annotations

import fnmatch
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from functools import cache, cached_property
from typing import Any, Iterable, Mapping

from usersim.engine.core.localized import LocalizedText

#: Characters after which a model family name may start inside a model id:
#: path segments, tags and versions (``:``, ``@``), vendor prefixes
#: (``meta.llama3``) and glued platform prefixes (``databricks-claude``).
_NAME_START = r"(?:^|(?<=[/:@.\-]))"
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))


class TacticMode(str, Enum):
    """How a tactic's opening turn is produced."""

    #: The spec's text is sent as written.
    VERBATIM = "verbatim"
    #: The simulated user writes the turn from the spec's instruction.
    GENERATED = "generated"


class TacticKind(str, Enum):
    """Whether the opening turn names an AI product or company.

    Only neutral openings can show a model attributing itself to another
    developer unprompted.
    """

    NEUTRAL = "neutral"
    LEADING = "leading"


class ProblemCode(str, Enum):
    """The kind of problem a spec has, for the tools and tests that act on it."""

    UNREADABLE = "unreadable"
    UNKNOWN_KEY = "unknown_key"
    SCHEMA_VERSION = "schema_version"
    MISSING_FIELD = "missing_field"
    INVALID_VALUE = "invalid_value"
    DUPLICATE_ID = "duplicate_id"
    CYCLE = "cycle"
    NO_PARENT = "no_parent"
    EMPTY_SECTION = "empty_section"
    UNKNOWN_DEVELOPER = "unknown_developer"
    SHADOWED_EXAMPLE = "shadowed_example"
    UNKNOWN_COMPETITOR = "unknown_competitor"
    TOO_FEW_COMPETITORS = "too_few_competitors"
    UNKNOWN_PLACEHOLDER = "unknown_placeholder"
    SHARED_NAME = "shared_name"


@dataclass(frozen=True)
class SpecProblem:
    code: ProblemCode
    #: ``<file>::<field path>`` of the offending entry.
    where: str
    detail: str

    def __str__(self) -> str:
        return f"{self.where}: {self.detail}"


class IdentitySpecError(ValueError):
    """A spec cannot be loaded as written.

    ``problems`` lists every issue found, so a spec author fixes them in one
    pass rather than rediscovering the next one on each load.
    """

    def __init__(self, problems: list[SpecProblem]) -> None:
        extra = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
        super().__init__(f"{problems[0]}{extra}")
        self.problems = problems


# ---------------------------------------------------------------------------
# Typed views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Developer:
    id: str
    names: tuple[str, ...]
    products: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    #: Entries of the three lists above that are also ordinary words, given
    #: names or acronyms in a supported language.
    ambiguous: tuple[str, ...] = ()
    leaders: tuple[str, ...] = ()
    headquarters: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return self.names[0]

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return (*self.names, *self.products, *self.models)


@dataclass(frozen=True)
class IdentityRule:
    developers: tuple[str, ...]
    #: Model family patterns; empty when the rule declares an exact ``model``.
    match: tuple[str, ...] = ()
    #: An exact model id, as the models config names it.
    model: str = ""
    model_names: tuple[str, ...] = ()
    lineage: tuple[str, ...] = ()
    source: str = ""
    examples: tuple[str, ...] = ()
    #: Depth of the layer that declared the rule, 0 for the top. A rule's
    #: examples are checked against its own layer and the ones below, so an
    #: override that deliberately shadows inherited rules still loads.
    layer: int = field(default=0, compare=False)

    @property
    def label(self) -> str:
        return self.model or ", ".join(self.match)

    def matches(self, model_id: str) -> str | None:
        """The declared id or the pattern that matches ``model_id``, or None."""
        if self.model:
            return self.model if model_id.strip().casefold() == self.model.casefold() else None
        return match_model_id(self.match, model_id)


@dataclass(frozen=True)
class ExpectedIdentity:
    """The answers one model may give about who made it."""

    model_id: str
    developers: tuple[str, ...]
    model_names: tuple[str, ...]
    lineage: tuple[str, ...]
    #: The declared id or the pattern that matched.
    rule: str
    #: True when a rule names this exact model, False when it was inferred
    #: from a model family name in the id.
    declared: bool
    #: ``bank_id@bank_version`` of the layer that holds the rule.
    layer: str
    spec_version: str


@dataclass(frozen=True)
class Tactic:
    id: str
    mode: TacticMode
    neutral: bool = False
    text: LocalizedText | None = None
    instruction: LocalizedText | None = None
    pair: str = ""
    delay: bool = False
    placeholder: bool = False

    @property
    def kind(self) -> TacticKind:
        return TacticKind.NEUTRAL if self.neutral else TacticKind.LEADING


@dataclass(frozen=True)
class Reframing:
    id: str
    instruction: LocalizedText


@dataclass(frozen=True)
class Strategy:
    id: str
    reframings: tuple[Reframing, ...] = ()
    placeholder: bool = False

    def reframing_at(self, index: int) -> Reframing | None:
        """The reframing for the ``index``-th follow-up, wrapping around; None if there are none."""
        if not self.reframings:
            return None
        return self.reframings[index % len(self.reframings)]


@dataclass(frozen=True)
class IdentitySpec:
    schema_version: str
    bank_id: str
    bank_version: str
    placeholder: bool
    developers: Mapping[str, Developer]
    expected_identities: tuple[IdentityRule, ...]
    competitors: Mapping[str, tuple[str, ...]]
    tactics: tuple[Tactic, ...]
    strategies: tuple[Strategy, ...]
    #: ``bank_id@bank_version`` of every layer, the extending file first.
    layers: tuple[str, ...]
    #: sha256 of the merged content, so two runs that graded against the same
    #: spec can be told apart from two that merely share a version label.
    digest: str
    source_paths: tuple[str, ...] = field(default_factory=tuple)

    @property
    def version(self) -> str:
        return " > ".join(self.layers)

    def resolve(self, model_id: str) -> ExpectedIdentity | None:
        """The expected identity of ``model_id``, or None when no rule applies.

        A rule declaring this exact ``model`` wins over every pattern; among
        patterns the first match wins, an extending layer's rules first.
        """
        found = _first_match(self.expected_identities, model_id)
        if found is None:
            return None
        rule, matched = found
        return ExpectedIdentity(
            model_id=model_id,
            developers=rule.developers,
            model_names=rule.model_names,
            lineage=rule.lineage,
            rule=matched,
            declared=bool(rule.model),
            layer=self.layers[rule.layer],
            spec_version=self.version,
        )

    def tactic_by_id(self, tactic_id: str) -> Tactic | None:
        return next((t for t in self.tactics if t.id == tactic_id), None)

    def strategy_by_id(self, strategy_id: str) -> Strategy | None:
        return next((s for s in self.strategies if s.id == strategy_id), None)

    @cached_property
    def _name_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for dev in self.developers.values():
            for name in dev.vocabulary:
                index.setdefault(fold_name(name), dev.id)
        return index

    def developer_of(self, name: str) -> str | None:
        """The developer id a company, product or model family name belongs to, or None."""
        return self._name_index.get(fold_name(name))

    def competitor_pool(self, locale: str, *, exclude: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Competitor products for ``locale`` (else ``default``), minus those of excluded developers."""
        pool = self.competitors.get(locale) or self.competitors.get("default") or ()
        excluded = set(exclude)
        return tuple(p for p in pool if self.developer_of(p) not in excluded)

    def vocabulary(self) -> dict[str, dict[str, list[str]]]:
        """The names, products and models per developer, as stored on each row for grading."""
        return {
            dev.id: {
                "names": list(dev.names),
                "products": list(dev.products),
                "models": list(dev.models),
                "ambiguous": list(dev.ambiguous),
            }
            for dev in self.developers.values()
        }

    def rules(self) -> list[dict[str, Any]]:
        """The expected-identity rules, as stored on each row for grading."""
        return [
            {
                **({"model": r.model} if r.model else {"match": list(r.match)}),
                "developers": list(r.developers),
                "lineage": list(r.lineage),
            }
            for r in self.expected_identities
        ]


def match_model_id(patterns: Iterable[str], model_id: str) -> str | None:
    """The first of ``patterns`` that matches ``model_id``, or None.

    Matching ignores case. A pattern without ``/`` may start at the beginning
    of the id or right after any ``/``, ``:``, ``@``, ``.`` or ``-``, and must
    match the rest of the id from there, so ``llama*`` finds the family in
    ``meta-llama/Meta-Llama-3-8B-Instruct``, ``meta.llama3-3-70b-instruct-v1:0``
    and ``llama3.1:8b`` alike. A pattern containing ``/`` must match the whole id.
    """
    target = model_id.strip().casefold()
    for pattern in patterns:
        if _pattern_regex(pattern).search(target):
            return pattern
    return None


@cache
def _pattern_regex(pattern: str) -> re.Pattern[str]:
    body = fnmatch.translate(pattern.casefold())
    return re.compile(("^" if "/" in pattern else _NAME_START) + body)


def _first_match(rules: Iterable[IdentityRule], model_id: str) -> tuple[IdentityRule, str] | None:
    """The rule that decides ``model_id`` and what matched: exact declarations first, then patterns in order."""
    ordered = sorted(rules, key=lambda rule: not rule.model)
    for rule in ordered:
        matched = rule.matches(model_id)
        if matched is not None:
            return rule, matched
    return None


def fold_name(name: str) -> str:
    """A developer, product or model name as vocabulary lookups compare it.

    Folds width (full-width ``ＤｅｅｐＳｅｅｋ``) and case, drops zero-width
    characters, and collapses runs of whitespace.
    """
    return " ".join(unicodedata.normalize("NFKC", name).translate(_INVISIBLE).casefold().split())


class Vocabulary:
    """The row's names, products and models, for looking up developers in text."""

    def __init__(self, vocabulary: dict[str, dict[str, list[str]]]) -> None:
        entries: dict[str, tuple[str, bool]] = {}
        for developer, lists in vocabulary.items():
            ambiguous = {fold_name(name) for name in lists.get("ambiguous") or []}
            for key in ("names", "products", "models"):
                for name in lists.get(key) or []:
                    folded = fold_name(name)
                    if folded:
                        entries.setdefault(folded, (developer, folded in ambiguous))
        self.developer_ids: tuple[str, ...] = tuple(sorted(vocabulary))
        self.display_names = {dev: (lists.get("names") or [dev])[0] for dev, lists in vocabulary.items()}
        self._exact = {folded: dev for folded, (dev, _) in entries.items()}
        self._ambiguous = {folded for folded, (_, ambiguous) in entries.items() if ambiguous}
        self._patterns = [
            (folded, name_pattern(folded), dev, ambiguous)
            for folded, (dev, ambiguous) in sorted(entries.items(), key=lambda kv: -len(kv[0]))
        ]

    def entry_in(self, text: str | None) -> tuple[str, str] | None:
        """The developer and entry a claimed name refers to: an exact entry, else the longest entry it contains."""
        if not text:
            return None
        folded = fold_name(text)
        if folded in self._exact:
            return self._exact[folded], folded
        for entry, pattern, dev, _ in self._patterns:
            if pattern.search(folded):
                return dev, entry
        return None

    def developer_of(self, text: str | None) -> str | None:
        found = self.entry_in(text)
        return found[0] if found else None

    def claimed_developer(self, text: str | None, guess: str | None) -> str | None:
        """The developer a name in a claim refers to, as ``developer_of`` finds it.

        An entry that is also an ordinary word counts only when the extractor,
        which read the turn, guessed the same developer: "GPT-4o" is OpenAI's,
        while "artificial intelligence" in Hindi is not a claim to be Krutrim.
        """
        if not text:
            return None
        folded = fold_name(text)
        if folded in self._exact:
            candidates = [(folded, self._exact[folded], folded in self._ambiguous)]
        else:
            candidates = [
                (entry, dev, ambiguous) for entry, pattern, dev, ambiguous in self._patterns if pattern.search(folded)
            ]
        return next((dev for _, dev, ambiguous in candidates if not ambiguous or dev == guess), None)

    def developers_in(self, text: str | None, *, include_ambiguous: bool) -> set[str]:
        """Every developer an entry of which appears in ``text``."""
        if not text:
            return set()
        folded = fold_name(text)
        return {
            dev
            for _, pattern, dev, ambiguous in self._patterns
            if (include_ambiguous or not ambiguous) and pattern.search(folded)
        }

    def mentions(self, text: str, developers: Iterable[str]) -> list[tuple[int, int]]:
        """Every place ``text`` names one of ``developers``, as spans of the text as written.

        Any letter case and run of whitespace matches; a name ``developers_in``
        finds only after width or other normalisation has no span.
        """
        wanted = set(developers)
        return sorted(
            match.span()
            for entry, _, dev, ambiguous in self._patterns
            if dev in wanted and not ambiguous
            for match in _written_pattern(entry).finditer(text)
        )


@cache
def name_pattern(folded: str) -> re.Pattern[str]:
    """Match ``folded`` as a whole name: Latin letters and digits may not run on either side."""
    start = r"(?<![0-9a-z])" if folded[0].isascii() and folded[0].isalnum() else ""
    end = r"(?![0-9a-z])" if folded[-1].isascii() and folded[-1].isalnum() else ""
    return re.compile(start + re.escape(folded) + end)


@cache
def _written_pattern(folded: str) -> re.Pattern[str]:
    """``name_pattern`` for text as written: any letter case, any whitespace between words."""
    start = r"(?<![0-9a-z])" if folded[0].isascii() and folded[0].isalnum() else ""
    end = r"(?![0-9a-z])" if folded[-1].isascii() and folded[-1].isalnum() else ""
    return re.compile(start + r"\s+".join(map(re.escape, folded.split())) + end, re.IGNORECASE)
