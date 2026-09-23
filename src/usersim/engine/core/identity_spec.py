# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""The identity_disclosure spec: who each model should say it is, and how to ask.

One YAML file per probe, ``assets/<probe>/spec.yaml``, holds:

- ``developers``: who can be named, and how, in any script;
- ``expected_identities``: rules from a model id to the answers that model may
  give about itself, either declared for an exact id or inferred from a model
  family name found in the id;
- ``competitors``: the other products the probe names, per locale;
- ``tactics`` and ``strategies``: how the probe asks, and how it pushes back.

A spec may start with ``extends: <probe>`` and list only what it adds or
changes. That is how a user declares a model the shipped spec cannot infer
without copying it, and how a probe built on this one inherits it. The fields,
the matching and the layering rules are described in
``assets/identity_disclosure/SCHEMA.md``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from functools import cache, cached_property
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from usersim.engine.core._assets import asset_search_path, probe_assets_dir
from usersim.engine.core.localized import LocalizedText, LocalizedTextError

logger = logging.getLogger("usersim.engine")

PROBE_LABEL = "identity_disclosure"
SPEC_FILENAME = "spec.yaml"
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1"})

#: Placeholders tactic and reframing text may use; the probe fills them per row.
ALLOWED_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "true_developer",
        "true_model",
        "competitor",
        "competitor_developer",
        "competitor_list",
        "other_language",
        "previous_assistant_response",
    }
)
_COMPETITOR_PLACEHOLDERS = frozenset({"competitor", "competitor_developer", "competitor_list"})
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "bank_id",
        "bank_version",
        "placeholder",
        "extends",
        "developers",
        "expected_identities",
        "competitors",
        "tactics",
        "strategies",
        "select",
        "exclude",
    }
)
_FILTERABLE_SECTIONS = ("tactics", "strategies")
_CONTENT_SECTIONS = ("developers", "expected_identities", "competitors", "tactics", "strategies")

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


# ---------------------------------------------------------------------------
# Locating and loading
# ---------------------------------------------------------------------------


def spec_env_var(label: str = PROBE_LABEL) -> str:
    """The environment variable that replaces ``label``'s spec for a run."""
    return f"USERSIM_{label.upper()}_SPEC"


def default_identity_spec_path(label: str = PROBE_LABEL) -> Path:
    """Where ``label``'s spec lives on the asset search path."""
    return probe_assets_dir(label) / SPEC_FILENAME


def identity_spec_path(label: str = PROBE_LABEL) -> Path:
    """The top layer a run loads: the env override if set, else the asset path."""
    override = os.environ.get(spec_env_var(label))
    if override:
        return Path(override)
    return default_identity_spec_path(label)


def load_identity_spec(path: str | Path) -> IdentitySpec:
    """Load the spec at ``path``, following ``extends`` down the asset search path."""
    merged, layers, sources = _load_layers(Path(path), visited=())
    return _build_spec(merged, layers, sources)


_CACHE: dict[tuple[str, str, tuple[str, ...]], IdentitySpec] = {}
_CACHE_LOCK = threading.Lock()


def load_identity_spec_default(label: str = PROBE_LABEL) -> IdentitySpec:
    """The spec ``label`` runs with, loaded once per process and then cached.

    The cache key includes the asset search path, because the layers a spec
    inherits depend on it.
    """
    top = identity_spec_path(label)
    key = (label, str(top), tuple(str(root) for root in asset_search_path()))
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        spec = load_identity_spec(top)
        _CACHE[key] = spec
        logger.info(
            "identity_spec: loaded %s (%d rules, %d tactics, %d strategies) from %s",
            spec.version,
            len(spec.expected_identities),
            len(spec.tactics),
            len(spec.strategies),
            top,
        )
        return spec


def reset_identity_spec_cache() -> None:
    """Drop every cached spec. For tests, and after editing a spec in a session."""
    with _CACHE_LOCK:
        _CACHE.clear()


def validate_spec(label: str = PROBE_LABEL) -> list[SpecProblem]:
    """Every problem with the spec ``label`` would run with; empty when it loads cleanly."""
    try:
        load_identity_spec(identity_spec_path(label))
    except IdentitySpecError as e:
        return list(e.problems)
    except OSError as e:
        return [SpecProblem(ProblemCode.UNREADABLE, label, str(e))]
    return []


# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------


def _load_layers(path: Path, *, visited: tuple[Path, ...]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return the merged document, its layers and their source paths, top first."""
    path = path.resolve()
    if path in visited:
        chain = " > ".join(str(p) for p in (*visited, path))
        raise IdentitySpecError([SpecProblem(ProblemCode.CYCLE, str(path), f"extends forms a cycle: {chain}")])
    layer = _read_layer(path)
    for rule in layer.get("expected_identities") or []:
        if isinstance(rule, dict):
            rule["_layer"] = len(visited)
    name = f"{layer['bank_id']}@{layer['bank_version']}"
    parent_label = layer.get("extends")
    if parent_label is None:
        merged = {k: v for k, v in layer.items() if k not in ("extends", "select", "exclude")}
        return _apply_filters(merged, layer), [name], [str(path)]
    parent_path = _parent_path(parent_label, below=path)
    parent, names, sources = _load_layers(parent_path, visited=(*visited, path))
    return _apply_filters(_merge(parent, layer), layer), [name, *names], [str(path), *sources]


def _parent_path(parent_label: str, *, below: Path) -> Path:
    """The ``parent_label`` spec that ``below`` inherits from.

    The first copy on the asset search path, unless ``below`` is itself one
    of those copies, in which case the next one down.
    """
    candidates: list[Path] = []
    for root in asset_search_path():
        candidate = Path(root) / parent_label / SPEC_FILENAME
        if candidate.is_file():
            candidates.append(candidate.resolve())
    if below in candidates:
        candidates = candidates[candidates.index(below) + 1 :]
    if not candidates:
        detail = f"extends {parent_label!r}, but no {parent_label}/{SPEC_FILENAME} was found below it on the asset search path"
        raise IdentitySpecError([SpecProblem(ProblemCode.NO_PARENT, str(below), detail)])
    return candidates[0]


def _read_layer(path: Path) -> dict[str, Any]:
    where = str(path)
    try:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except FileNotFoundError:
        raise IdentitySpecError([SpecProblem(ProblemCode.UNREADABLE, where, "no such spec file")]) from None
    except yaml.YAMLError as e:
        raise IdentitySpecError([SpecProblem(ProblemCode.UNREADABLE, where, f"invalid YAML: {e}")]) from e
    if not isinstance(doc, dict):
        raise IdentitySpecError([SpecProblem(ProblemCode.UNREADABLE, where, "a spec must be a mapping")])

    problems: list[SpecProblem] = []
    unknown = sorted(set(doc) - _TOP_LEVEL_KEYS)
    if unknown:
        problems.append(
            SpecProblem(
                ProblemCode.UNKNOWN_KEY, where, f"unknown key(s) {unknown}; expected some of {sorted(_TOP_LEVEL_KEYS)}"
            )
        )
    version = doc.get("schema_version")
    if str(version) not in SUPPORTED_SCHEMA_VERSIONS:
        problems.append(
            SpecProblem(
                ProblemCode.SCHEMA_VERSION,
                f"{where}::schema_version",
                f"{version!r} is not supported; use one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}",
            )
        )
    for key in ("bank_id", "bank_version"):
        value = doc.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)) or not str(value).strip():
            problems.append(SpecProblem(ProblemCode.MISSING_FIELD, f"{where}::{key}", "required, a non-empty string"))
    if "extends" in doc and (not isinstance(doc["extends"], str) or not doc["extends"].strip()):
        problems.append(SpecProblem(ProblemCode.INVALID_VALUE, f"{where}::extends", "must name a probe"))
    for key in ("developers", "competitors"):
        if key in doc and not isinstance(doc[key], dict):
            problems.append(SpecProblem(ProblemCode.INVALID_VALUE, f"{where}::{key}", "must be a mapping"))
    for key in ("expected_identities", "tactics", "strategies"):
        if key in doc and not isinstance(doc[key], list):
            problems.append(SpecProblem(ProblemCode.INVALID_VALUE, f"{where}::{key}", "must be a list"))
    for key in ("tactics", "strategies"):
        items = doc.get(key) if isinstance(doc.get(key), list) else []
        seen: set[str] = set()
        for i, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
                problems.append(SpecProblem(ProblemCode.MISSING_FIELD, f"{where}::{key}[{i}]", "needs a string id"))
                continue
            if item["id"] in seen:
                # Within one layer a repeat would silently merge into the first.
                problems.append(SpecProblem(ProblemCode.DUPLICATE_ID, f"{where}::{key}[{item['id']}]", "duplicate id"))
            seen.add(item["id"])
    for key in ("select", "exclude"):
        if key not in doc:
            continue
        value = doc[key]
        if not isinstance(value, dict) or set(value) - set(_FILTERABLE_SECTIONS):
            detail = f"must map some of {list(_FILTERABLE_SECTIONS)} to lists of id globs"
            problems.append(SpecProblem(ProblemCode.INVALID_VALUE, f"{where}::{key}", detail))
            continue
        for section, patterns in value.items():
            if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
                problems.append(
                    SpecProblem(ProblemCode.INVALID_VALUE, f"{where}::{key}.{section}", "must be a list of id globs")
                )
    if problems:
        raise IdentitySpecError(problems)

    doc = dict(doc)
    doc["schema_version"] = str(doc["schema_version"])
    doc["bank_id"] = str(doc["bank_id"])
    doc["bank_version"] = str(doc["bank_version"])
    return doc


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key in ("schema_version", "bank_id", "bank_version", "placeholder"):
        if key in child:
            merged[key] = child[key]
    for key in ("developers", "competitors"):
        if key in child:
            merged[key] = {**(parent.get(key) or {}), **child[key]}
    if "expected_identities" in child:
        merged["expected_identities"] = [*child["expected_identities"], *(parent.get("expected_identities") or [])]
    if "tactics" in child:
        merged["tactics"] = _merge_by_id(
            parent.get("tactics") or [], child["tactics"], localized=("text", "instruction")
        )
    if "strategies" in child:
        merged["strategies"] = _merge_by_id(
            parent.get("strategies") or [],
            child["strategies"],
            localized=(),
            nested={"reframings": ("instruction",)},
        )
    return merged


def _merge_by_id(
    parent_items: list[dict[str, Any]],
    child_items: list[dict[str, Any]],
    *,
    localized: tuple[str, ...],
    nested: Mapping[str, tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Merge two id-keyed lists: same id merges field by field, new ids append."""
    out = [dict(item) for item in parent_items]
    position = {item.get("id"): i for i, item in enumerate(out)}
    for child in child_items:
        cid = child.get("id")
        if cid not in position:
            position[cid] = len(out)
            out.append(dict(child))
            continue
        base = out[position[cid]]
        merged = dict(base)
        for key, value in child.items():
            if key in localized and key in base:
                merged[key] = {**_renderings(base[key]), **_renderings(value)}
            elif nested and key in nested and key in base and isinstance(value, list):
                merged[key] = _merge_by_id(base[key] or [], value, localized=nested[key])
            else:
                merged[key] = value
        out[position[cid]] = merged
    return out


def _renderings(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {LocalizedText.DEFAULT_LOCALE: value}


def _apply_filters(merged: dict[str, Any], layer: dict[str, Any]) -> dict[str, Any]:
    """Apply the layer's ``select`` and then its ``exclude`` to the merged document."""
    for kind, keep in (("select", True), ("exclude", False)):
        for section, patterns in (layer.get(kind) or {}).items():
            items = merged.get(section) or []
            merged[section] = [
                item for item in items if any(fnmatch.fnmatchcase(item["id"], p) for p in patterns) == keep
            ]
    return merged


# ---------------------------------------------------------------------------
# Typing and validation
# ---------------------------------------------------------------------------


def _build_spec(merged: dict[str, Any], layers: list[str], sources: list[str]) -> IdentitySpec:
    top = sources[0]
    problems: list[SpecProblem] = []

    developers: dict[str, Developer] = {}
    for dev_id, entry in (merged.get("developers") or {}).items():
        where = f"{top}::developers.{dev_id}"
        if not isinstance(entry, dict):
            problems.append(SpecProblem(ProblemCode.INVALID_VALUE, where, "must be a mapping"))
            continue
        developers[str(dev_id)] = Developer(
            id=str(dev_id),
            names=_strings(entry.get("names"), f"{where}.names", problems, required=True),
            products=_strings(entry.get("products"), f"{where}.products", problems),
            models=_strings(entry.get("models"), f"{where}.models", problems),
            ambiguous=_strings(entry.get("ambiguous"), f"{where}.ambiguous", problems),
            leaders=_strings(entry.get("leaders"), f"{where}.leaders", problems),
            headquarters=_strings(entry.get("headquarters"), f"{where}.headquarters", problems),
        )

    rules: list[IdentityRule] = []
    for i, entry in enumerate(merged.get("expected_identities") or []):
        where = f"{top}::expected_identities[{i}]"
        if not isinstance(entry, dict) or ("model" in entry) == ("match" in entry):
            detail = "every rule needs either a 'model' id or a 'match' pattern (or list of patterns), not both"
            problems.append(SpecProblem(ProblemCode.MISSING_FIELD, where, detail))
            continue
        model = entry.get("model")
        if "model" in entry and (not isinstance(model, str) or not model.strip()):
            problems.append(SpecProblem(ProblemCode.INVALID_VALUE, f"{where}.model", "must be a model id"))
            continue
        raw_match = entry.get("match")
        match = _strings(
            [raw_match] if isinstance(raw_match, str) else raw_match,
            f"{where}.match",
            problems,
            required="match" in entry,
        )
        developer = entry.get("developer")
        rules.append(
            IdentityRule(
                match=match,
                model=model.strip() if isinstance(model, str) else "",
                developers=_strings(
                    (developer,) if isinstance(developer, str) else developer,
                    f"{where}.developer",
                    problems,
                    required=True,
                ),
                model_names=_strings(entry.get("model_names"), f"{where}.model_names", problems),
                lineage=_strings(entry.get("lineage"), f"{where}.lineage", problems),
                source=str(entry.get("source") or ""),
                examples=_strings(entry.get("examples"), f"{where}.examples", problems),
                layer=int(entry.get("_layer", 0)),
            )
        )

    competitors = {
        str(locale): _strings(products, f"{top}::competitors.{locale}", problems)
        for locale, products in (merged.get("competitors") or {}).items()
    }
    tactics = [_tactic(entry, f"{top}::tactics[{entry['id']}]", problems) for entry in merged.get("tactics") or []]
    strategies = [
        _strategy(entry, f"{top}::strategies[{entry['id']}]", problems) for entry in merged.get("strategies") or []
    ]
    if problems:
        raise IdentitySpecError(problems)

    spec = IdentitySpec(
        schema_version=merged["schema_version"],
        bank_id=merged["bank_id"],
        bank_version=merged["bank_version"],
        placeholder=bool(merged.get("placeholder", False)),
        developers=developers,
        expected_identities=tuple(rules),
        competitors=competitors,
        tactics=tuple(t for t in tactics if t is not None),
        strategies=tuple(s for s in strategies if s is not None),
        layers=tuple(layers),
        digest=_digest(merged),
        source_paths=tuple(sources),
    )
    problems = _cross_reference_problems(spec, top)
    if problems:
        raise IdentitySpecError(problems)
    return spec


def _digest(merged: dict[str, Any]) -> str:
    content = {key: merged.get(key) for key in _CONTENT_SECTIONS}
    content["expected_identities"] = [
        {k: v for k, v in rule.items() if k != "_layer"} if isinstance(rule, dict) else rule
        for rule in content["expected_identities"] or []
    ]
    payload = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tactic(entry: dict[str, Any], where: str, problems: list[SpecProblem]) -> Tactic | None:
    try:
        mode = TacticMode(entry.get("mode"))
    except ValueError:
        detail = f"must be one of {[m.value for m in TacticMode]}"
        problems.append(SpecProblem(ProblemCode.INVALID_VALUE, f"{where}.mode", detail))
        return None
    text = _localized(entry.get("text"), f"{where}.text", problems)
    instruction = _localized(entry.get("instruction"), f"{where}.instruction", problems)
    if mode is TacticMode.VERBATIM and text is None:
        problems.append(SpecProblem(ProblemCode.MISSING_FIELD, f"{where}.text", "a verbatim tactic needs text"))
    if mode is TacticMode.GENERATED and instruction is None:
        detail = "a generated tactic needs an instruction"
        problems.append(SpecProblem(ProblemCode.MISSING_FIELD, f"{where}.instruction", detail))
    _check_placeholders(text, f"{where}.text", problems)
    _check_placeholders(instruction, f"{where}.instruction", problems)
    return Tactic(
        id=entry["id"],
        mode=mode,
        neutral=bool(entry.get("neutral", False)),
        text=text,
        instruction=instruction,
        pair=str(entry.get("pair") or ""),
        delay=bool(entry.get("delay", False)),
        placeholder=bool(entry.get("placeholder", False)),
    )


def _strategy(entry: dict[str, Any], where: str, problems: list[SpecProblem]) -> Strategy | None:
    raw = entry.get("reframings") or []
    if not isinstance(raw, list):
        problems.append(SpecProblem(ProblemCode.INVALID_VALUE, f"{where}.reframings", "must be a list"))
        return None
    reframings: list[Reframing] = []
    seen: set[str] = set()
    for item in raw:
        rid = item.get("id") if isinstance(item, dict) else None
        if not isinstance(rid, str) or not rid.strip():
            problems.append(SpecProblem(ProblemCode.MISSING_FIELD, f"{where}.reframings", "needs a string id"))
            continue
        item_where = f"{where}.reframings[{rid}]"
        if rid in seen:
            problems.append(SpecProblem(ProblemCode.DUPLICATE_ID, item_where, "duplicate id"))
        seen.add(rid)
        instruction = _localized(item.get("instruction"), f"{item_where}.instruction", problems)
        if instruction is None:
            problems.append(SpecProblem(ProblemCode.MISSING_FIELD, f"{item_where}.instruction", "required"))
            continue
        _check_placeholders(instruction, f"{item_where}.instruction", problems)
        reframings.append(Reframing(id=rid, instruction=instruction))
    return Strategy(id=entry["id"], reframings=tuple(reframings), placeholder=bool(entry.get("placeholder", False)))


def _cross_reference_problems(spec: IdentitySpec, top: str) -> list[SpecProblem]:
    problems: list[SpecProblem] = []
    known = set(spec.developers)

    for kind, items in (("tactics", spec.tactics), ("strategies", spec.strategies)):
        ids = [item.id for item in items]
        if not ids:
            detail = "none left; a spec needs at least one (check select and exclude)"
            problems.append(SpecProblem(ProblemCode.EMPTY_SECTION, f"{top}::{kind}", detail))
        for dup in sorted({i for i in ids if ids.count(i) > 1}):
            problems.append(SpecProblem(ProblemCode.DUPLICATE_ID, f"{top}::{kind}[{dup}]", "duplicate id"))

    owners: dict[str, str] = {}
    for dev in spec.developers.values():
        where = f"{top}::developers.{dev.id}"
        folded = {fold_name(name) for name in dev.vocabulary}
        for name in sorted(folded):
            owner = owners.setdefault(name, dev.id)
            if owner != dev.id:
                detail = f"{name!r} is also listed under {owner!r}; a name must point to one developer"
                problems.append(SpecProblem(ProblemCode.SHARED_NAME, where, detail))
        for entry in dev.ambiguous:
            if fold_name(entry) not in folded:
                detail = f"{entry!r} is not one of this developer's names, products or models"
                problems.append(SpecProblem(ProblemCode.INVALID_VALUE, f"{where}.ambiguous", detail))

    for rule in spec.expected_identities:
        where = f"{top}::expected_identities[{rule.label}]"
        for dev in (*rule.developers, *rule.lineage):
            if dev not in known:
                problems.append(SpecProblem(ProblemCode.UNKNOWN_DEVELOPER, where, f"unknown developer {dev!r}"))
        if rule.model and rule.examples:
            detail = "examples illustrate 'match' patterns; a 'model' rule names its one id"
            problems.append(SpecProblem(ProblemCode.INVALID_VALUE, f"{where}.examples", detail))
        own_and_below = [r for r in spec.expected_identities if r.layer >= rule.layer]
        for example in rule.examples:
            found = _first_match(own_and_below, example)
            if found is None or found[0] is not rule:
                resolved = found[0].label if found else "no rule"
                detail = f"example {example!r} resolves to {resolved}; a rule listed earlier shadows it"
                problems.append(SpecProblem(ProblemCode.SHADOWED_EXAMPLE, where, detail))

    for locale, products in spec.competitors.items():
        for product in products:
            if spec.developer_of(product) is None:
                detail = f"{product!r} is not a name or product of any developer"
                problems.append(SpecProblem(ProblemCode.UNKNOWN_COMPETITOR, f"{top}::competitors.{locale}", detail))

    if _uses_competitors(spec):
        if not spec.competitors:
            detail = "tactics or reframings name a competitor, but none are listed"
            problems.append(SpecProblem(ProblemCode.TOO_FEW_COMPETITORS, f"{top}::competitors", detail))
        for rule in spec.expected_identities:
            excluded = (*rule.developers, *rule.lineage)
            for locale in spec.competitors:
                if len(spec.competitor_pool(locale, exclude=excluded)) < 2:
                    detail = (
                        f"fewer than two competitors remain for models matching {rule.label!r} "
                        f"once {', '.join(excluded)} are excluded"
                    )
                    problems.append(
                        SpecProblem(ProblemCode.TOO_FEW_COMPETITORS, f"{top}::competitors.{locale}", detail)
                    )
    return problems


def _uses_competitors(spec: IdentitySpec) -> bool:
    texts: list[LocalizedText] = []
    for tactic in spec.tactics:
        texts.extend(t for t in (tactic.text, tactic.instruction) if t is not None)
    for strategy in spec.strategies:
        texts.extend(r.instruction for r in strategy.reframings)
    return any(
        _COMPETITOR_PLACEHOLDERS & set(_PLACEHOLDER.findall(rendering))
        for t in texts
        for rendering in t.renderings.values()
    )


def _strings(value: Any, where: str, problems: list[SpecProblem], *, required: bool = False) -> tuple[str, ...]:
    if value is None or (required and value in ((), [])):
        if required:
            problems.append(SpecProblem(ProblemCode.MISSING_FIELD, where, "required, a non-empty list of strings"))
        return ()
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) and v.strip() for v in value):
        problems.append(SpecProblem(ProblemCode.INVALID_VALUE, where, "must be a list of non-empty strings"))
        return ()
    return tuple(v.strip() for v in value)


def _localized(value: Any, where: str, problems: list[SpecProblem]) -> LocalizedText | None:
    if value is None:
        return None
    try:
        return LocalizedText.from_yaml_value(value, field_path=where)
    except LocalizedTextError as e:
        problems.append(SpecProblem(ProblemCode.INVALID_VALUE, where, str(e)))
        return None


def _check_placeholders(value: LocalizedText | None, where: str, problems: list[SpecProblem]) -> None:
    if value is None:
        return
    for locale, rendering in value.renderings.items():
        unknown = sorted(set(_PLACEHOLDER.findall(rendering)) - ALLOWED_PLACEHOLDERS)
        if unknown:
            detail = f"unknown placeholder(s) {unknown}; allowed: {sorted(ALLOWED_PLACEHOLDERS)}"
            problems.append(SpecProblem(ProblemCode.UNKNOWN_PLACEHOLDER, f"{where}::{locale}", detail))
