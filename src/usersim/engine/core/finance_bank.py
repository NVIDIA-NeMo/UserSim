# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""financial_services bank loader + validator + lookup helpers.

A financial-services bank is a per-LOCALE bundle: a locale-level
``region_meta.yaml`` (regulators, currency, privacy, per-domain regulator text)
plus one subdirectory per INSTITUTION -- grouped under an institution-TYPE dir so
the layout stays stable across brand renames and multiple brands can share a
vertical -- each holding ``{institution_meta,corpus,tools,tasks}.yaml``:

    assets/financial_services/<locale>/
      region_meta.yaml
      <institution_type>/<institution_id>/
        institution_meta.yaml   corpus.yaml   tools.yaml   tasks.yaml

The schema lives in ``assets/financial_services/SCHEMA.md``.

Design notes:

- The runtime (probe + scorer) is self-contained: everything it needs lives in
  the committed bank, so the plugin never imports ``asset_gen``.
- A locale is a SET of institutions; each spans one or more domains. Retrieval,
  tools, and gold are always scoped to a single institution.
- Scope integrity: a verifiable template's gold docs/tools MUST belong to its
  own institution (a load-time error otherwise).
- Loader is strict; schema drift surfaces at load time, not simulate time.
- Cache is per-locale + thread-safe so probe and scorer see the same
  ``bank_version`` / ``task_contract_version``.
- Override the default path with ``USERSIM_FINANCIAL_SERVICES_BANK_<LOCALE>``
  pointing at a directory holding ``region_meta.yaml`` + institution subdirs.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Sequence,
)

logger = logging.getLogger("usersim.engine")

SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"v0.1"})
# Task tiers. ``verifiable`` templates are scripted + gold-derived; the
# ``dynamic`` tier is taxonomy-driven (generated turn-1, judge-scored) and does
# NOT ship scripted templates, so committed tasks.yaml carry verifiable
# templates only. ``dynamic`` is kept in the allow-set for forward-compat.
ALLOWED_TIERS: frozenset[str] = frozenset({"verifiable", "dynamic"})
ALLOWED_SIDE_EFFECTS: frozenset[str] = frozenset({"read_only", "state_changing", "irreversible"})
_PERSONA_TAG_RE = re.compile(
    r"^(region|age|occupation-family|education|interest|financial-literacy):"
    r"[a-zA-Z0-9_\-+]+$"
)


class FinanceBankError(ValueError):
    """A user-facing financial-services bank validation problem."""


# ---------------------------------------------------------------------------
# Typed views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    body: str
    product_category: str
    document_type: str
    source_authority: str
    placeholder: bool
    mentions_tools: tuple[str, ...] = ()
    institution_id: str = ""
    domain: str = ""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    discoverable: bool
    side_effect_class: str
    parameters: dict[str, Any] = field(default_factory=dict)
    institution_id: str = ""

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


# ---------------------------------------------------------------------------
# Framework primitive tools
# ---------------------------------------------------------------------------
#: Universal identity requirement, used when a bank's ``region_meta`` carries no
#: ``identity_verification`` block (every bank generated before the field
#: existed). Mirrors ``asset_gen.financial_services.spec``'s default.
DEFAULT_IDENTITY_REQUIRED_FIELDS: tuple[str, ...] = ("full_name", "date_of_birth")

#: Region-NEUTRAL account labels, used when a bank's ``region_meta`` carries no
#: ``account_labels`` block. Generic on purpose: a retail deposit product is named
#: differently in every market, so the fallback must be true everywhere rather
#: than quietly exporting US vocabulary. Mirrors
#: ``asset_gen.financial_services.spec.DEFAULT_DOMAIN_ACCOUNT_LABELS``.
DEFAULT_DOMAIN_ACCOUNT_LABELS: dict[str, str] = {
    "retail_banking": "deposit account",
    "payments": "payment account",
    "cards": "credit card account",
    "lending_mortgage": "loan account",
    "brokerage_investing": "brokerage account",
    "retirement": "retirement account",
    "wealth_management": "managed investment account",
    "insurance": "policy",
    "crypto": "crypto account",
}

# Every institution offers these two primitives, and the PROBE services them in
# its ``execute_tool_call`` (they are not per-institution DB tools). Their schema
# is therefore fixed HERE — the single source of truth shared by the generator
# (``asset_gen`` writes them into each ``tools.yaml``) and the runtime probe
# (``build_offered_tools`` backfills them). Region specs list these by name
# only; without this backfill the generated banks shipped them with empty
# description/parameters, so the assistant called ``kb_search({})`` — retrieving
# nothing and never unlocking the discoverable gold tools.
FRAMEWORK_PRIMITIVE_TOOLS: dict[str, dict[str, Any]] = {
    "kb_search": {
        "description": (
            "Search this institution's knowledge base and return the most "
            "relevant documents (product details, policies, and tool "
            "documentation). Use it to look up how to handle the customer's "
            "request and to discover the specialized tools you need — a "
            "discoverable tool only becomes callable after you have retrieved "
            "the document that documents it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what to find, e.g. "
                        "'dispute an unauthorized debit card transaction' or "
                        "'roll over a 401(k) to an IRA'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    "verify_identity": {
        "description": (
            "Verify the customer's identity before taking any account action. "
            "Collect every REQUIRED field listed below from the customer first "
            "(an account/card last-4 is optional extra signal). Verification "
            "fails if any required field is missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "date_of_birth": {"type": "string"},
                "account_or_card_last4": {"type": "string"},
            },
            "required": list(DEFAULT_IDENTITY_REQUIRED_FIELDS),
        },
    },
}

#: Per-field parameter schemas for the region-specific identity fields. Kept
#: beside the primitive so a region that requires one gets a self-documenting
#: parameter (format included) rather than a bare string the agent has to guess.
_IDENTITY_FIELD_SCHEMAS: dict[str, dict[str, Any]] = {
    "full_name": {
        "type": "string",
        "description": "The customer's full name as held on the account.",
    },
    "date_of_birth": {
        "type": "string",
        "description": "The customer's date of birth (YYYY-MM-DD).",
    },
    "pan": {
        "type": "string",
        "description": (
            "The customer's PAN (Permanent Account Number) — India's "
            "income-tax identifier used for financial KYC. Ten characters, "
            "formatted AAAAA9999A."
        ),
    },
    "aadhaar_last4": {
        "type": "string",
        "description": ("Last 4 digits of the customer's Aadhaar number. Never ask for the full Aadhaar number."),
    },
    "national_id_last4": {
        "type": "string",
        "description": "Last 4 digits of the customer's national ID number.",
    },
}


def verify_identity_schema(
    required_fields: Sequence[str],
) -> dict[str, Any]:
    """``verify_identity`` specialized to one region's KYC contract.

    The required set is regional data (see ``FinanceBank.identity_required_fields``),
    and it is advertised in the SCHEMA rather than only enforced in the mock:
    an agent that is never told a PAN is required cannot collect one, so gating
    on a hidden requirement would measure a failure the assistant had no way to
    avoid. Telling it and then enforcing is the fair test.
    """
    required = [f for f in required_fields if f in _IDENTITY_FIELD_SCHEMAS]
    if not required:
        required = list(DEFAULT_IDENTITY_REQUIRED_FIELDS)
    props: dict[str, Any] = {f: dict(_IDENTITY_FIELD_SCHEMAS[f]) for f in required}
    # Optional extra signal, always offered.
    props.setdefault("account_or_card_last4", {"type": "string"})
    pretty = " AND ".join(f.replace("_", " ") for f in required)
    return {
        "description": (
            "Verify the customer's identity before taking any account action. "
            f"{pretty.upper()} are REQUIRED to verify; collect them from the "
            "customer first (an account/card last-4 is optional extra signal). "
            "Verification fails if any required field is missing."
        ),
        "parameters": {
            "type": "object",
            "properties": props,
            "required": list(required),
        },
    }


def framework_primitive_tool(name: str) -> dict[str, Any] | None:
    """Canonical ``{description, parameters}`` for a framework primitive.

    Returns ``None`` for non-primitive (per-institution DB) tools. Callers get a
    fresh copy so mutating the result never corrupts the shared source.
    """
    spec = FRAMEWORK_PRIMITIVE_TOOLS.get(name)
    if spec is None:
        return None
    import copy as _copy

    return {
        "description": spec["description"],
        "parameters": _copy.deepcopy(spec["parameters"]),
    }


@dataclass(frozen=True)
class TaskTemplate:
    id: str
    tier: str  # verifiable (scripted; dynamic tier is taxonomy-driven, not templated)
    task_type: str
    placeholder: bool
    persona_tags: tuple[str, ...]
    # One or more turn-1 phrasings of the same task; build_instance picks one
    # deterministically per trajectory so the assistant sees varied openings
    # (generalizability) while the gold stays fixed. A task authored with a
    # single ``opening_user_message`` (str) is normalized to a 1-tuple.
    opening_user_messages: tuple[str, ...]
    account_state: dict[str, Any]
    param_space: dict[str, tuple[Any, ...]]
    gold_document_ids: tuple[str, ...]
    gold_tool_sequence: tuple[str, ...]
    expected_state_deltas: dict[str, Any]
    # State-changing tools that are PERMITTED (but not required) for this task —
    # legitimate actions the scenario may invite (e.g. freezing a compromised
    # card during a fraud dispute). The verifier requires ``gold_tool_sequence``
    # but does not penalize a state change in ``gold ∪ allowed_tools``; anything
    # state-changing outside that set is "unauthorized".
    allowed_tools: tuple[str, ...] = ()
    # Whether the verifier enforces the ORDER of ``gold_tool_sequence`` (a strict
    # subsequence check), vs. only requiring the tools be present. Default False:
    # most multi-step tasks bundle interchangeable actions, and strict ordering
    # false-fails a correct trajectory that picks a different valid order (and we
    # do not verify final DB state, so tool-name order is a weak proxy). Opt IN
    # with ``ordered: true`` for genuine dependency chains (e.g. cancel -> withdraw
    # -> close, open -> rollover) where a wrong order is actually wrong.
    ordered: bool = False
    institution_id: str = ""
    domain: str = ""


@dataclass(frozen=True)
class Institution:
    institution_id: str
    display_name: str
    type: str
    brand_voice: str
    domains: tuple[str, ...]
    documents: tuple[Document, ...]
    tools: tuple[ToolSpec, ...]
    templates: tuple[TaskTemplate, ...]
    #: In-language brand name for non-Latin locales; empty when the Latin
    #: ``display_name`` is what this locale's own documents use. Defaulted so
    #: existing callers and pre-existing banks are unaffected.
    display_name_local: str = ""
    placeholder: bool = False
    #: doc_id -> dense embedding vector, loaded from the ``embeddings.parquet``
    #: sidecar (empty when absent). Used by ``retrieval.py``'s ``dense`` mode;
    #: ``golden`` retrieval needs none, so this stays optional/offline.
    embeddings: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def doc_by_id(self, doc_id: str) -> Document | None:
        for d in self.documents:
            if d.id == doc_id:
                return d
        return None

    def brand_name(self) -> str:
        """The brand as THIS locale's own documents write it.

        A non-Latin locale generates its corpus under ``display_name_local``, so
        the conversation has to use the same form — otherwise the customer says
        "Chandrika Bank" while every retrieved document says "चंद्रिका बैंक",
        which reads like two different institutions. Falls back to the canonical
        Latin name, which is what Latin-script locales use.
        """
        return self.display_name_local or self.display_name

    def tool_by_name(self, name: str) -> ToolSpec | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def template_by_id(self, template_id: str) -> TaskTemplate | None:
        for t in self.templates:
            if t.id == template_id:
                return t
        return None

    def permanent_tools(self) -> tuple[ToolSpec, ...]:
        return tuple(t for t in self.tools if not t.discoverable)

    def matching_persona_tags(self, tags: Iterable[str]) -> tuple[TaskTemplate, ...]:
        wanted = set(tags)
        return tuple(t for t in self.templates if _tags_match(t.persona_tags, wanted))


@dataclass(frozen=True)
class FinanceBank:
    schema_version: str
    locale: str
    bank_id: str
    bank_version: str
    task_contract_version: str
    region_meta: dict[str, Any]
    institutions: tuple[Institution, ...]
    source_path: str | None = None

    # ── Lookups ─────────────────────────────────────────────────────

    def institution(self, institution_id: str) -> Institution | None:
        for inst in self.institutions:
            if inst.institution_id == institution_id:
                return inst
        return None

    def regulator_text_for_domain(self, domain: str) -> str:
        dr = self.region_meta.get("domain_regulators", {}) or {}
        ctx = dr.get(domain, {}) if isinstance(dr, dict) else {}
        return str(ctx.get("regulator_text", "")) if isinstance(ctx, dict) else ""

    def identity_required_fields(self) -> tuple[str, ...]:
        """Identity fields this region's agents must collect before acting.

        Read from ``region_meta.identity_verification.required_fields`` so the
        KYC contract is regional DATA, not code: India expects a PAN on top of
        name + DOB, the US stops at name + DOB. Falls back to the universal
        name + DOB pair for banks generated before the field existed, so older
        assets keep working unchanged.
        """
        iv = self.region_meta.get("identity_verification") or {}
        fields = iv.get("required_fields") if isinstance(iv, dict) else None
        if isinstance(fields, (list, tuple)):
            cleaned = tuple(str(f).strip() for f in fields if str(f).strip())
            if cleaned:
                return cleaned
        return DEFAULT_IDENTITY_REQUIRED_FIELDS

    def domain_account_label(self, domain: str) -> str:
        """This region's own name for the account a read tool reports.

        Read from ``region_meta.account_labels`` so product vocabulary is regional
        DATA, not code: "checking account" is a US product and reads as the wrong
        institution to an Indian customer, who holds a savings or current account.
        Falls back to a region-NEUTRAL label that is true in every market, so a
        locale that never authored the block is generic rather than wrong.
        """
        labels = self.region_meta.get("account_labels") or {}
        if isinstance(labels, dict):
            label = str(labels.get(domain, "") or "").strip()
            if label:
                return label
        return DEFAULT_DOMAIN_ACCOUNT_LABELS.get(domain, "account")

    def all_domains(self) -> tuple[str, ...]:
        out: list[str] = []
        for inst in self.institutions:
            for d in inst.domains:
                if d not in out:
                    out.append(d)
        return tuple(out)

    def provenance_summary(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "bank_version": self.bank_version,
            "task_contract_version": self.task_contract_version,
            "locale": self.locale,
            "n_institutions": len(self.institutions),
            "n_documents": sum(len(i.documents) for i in self.institutions),
            "n_templates": sum(len(i.templates) for i in self.institutions),
        }


def _tags_match(tpl_tags: tuple[str, ...], persona_tags: set) -> bool:
    """A template's persona_tags satisfied by ``persona_tags`` (``*:any`` wildcard)."""
    if not tpl_tags:
        return True
    for tag in tpl_tags:
        if ":" not in tag:
            if tag not in persona_tags:
                return False
            continue
        _, value = tag.split(":", 1)
        if value == "any":
            continue
        if tag not in persona_tags:
            return False
    return True


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_finance_bank(locale_dir: str | Path) -> FinanceBank:
    """Parse + validate a locale bank (region_meta + institution subdirs)."""
    root = Path(locale_dir)
    if not root.is_dir():
        raise FinanceBankError(f"{locale_dir}: bank directory not found")

    region_meta = _load_yaml(root / "region_meta.yaml")
    schema_version = _require_str(region_meta, "schema_version", str(root / "region_meta.yaml"))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise FinanceBankError(
            f"{root}::schema_version: {schema_version!r} not supported; "
            f"loader understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    locale = _require_str(region_meta, "locale", str(root))
    bank_id = _require_str(region_meta, "bank_id", str(root))
    bank_version = _require_str(region_meta, "bank_version", str(root))
    task_contract_version = _require_str(region_meta, "task_contract_version", str(root))

    # Institutions live one dir per brand, grouped under an institution-type dir
    # (``<locale>/<type>/<institution_id>/``). Discover them layout-agnostically by
    # finding every ``institution_meta.yaml`` (also tolerates a legacy flat layout).
    institutions: list[Institution] = []
    for meta_path in sorted(root.rglob("institution_meta.yaml")):
        inst_dir = meta_path.parent
        institutions.append(_build_institution(inst_dir, src=str(inst_dir)))

    if not institutions:
        raise FinanceBankError(f"{root}: no institution subdirectories found")

    # Cross-check: every domain any institution offers has regulator context.
    dr = region_meta.get("domain_regulators", {}) or {}
    for inst in institutions:
        for dom in inst.domains:
            if dom not in dr:
                raise FinanceBankError(
                    f"{root}/region_meta.yaml::domain_regulators: missing entry "
                    f"for domain {dom!r} offered by institution {inst.institution_id!r}"
                )

    bank = FinanceBank(
        schema_version=schema_version,
        locale=locale,
        bank_id=bank_id,
        bank_version=bank_version,
        task_contract_version=task_contract_version,
        region_meta=region_meta,
        institutions=tuple(institutions),
        source_path=str(root),
    )

    n_ph = sum(1 for i in institutions for d in i.documents if d.placeholder)
    if n_ph:
        logger.info(
            "finance_bank: loaded %s v%s (%d institutions, %d docs, "
            "%d placeholder) — downstream scorecards preview-only",
            bank_id,
            bank_version,
            len(institutions),
            sum(len(i.documents) for i in institutions),
            n_ph,
        )
    return bank


def _build_institution(inst_dir: Path, *, src: str) -> Institution:
    meta = _load_yaml(inst_dir / "institution_meta.yaml")
    iid = _require_str(meta, "institution_id", src)
    domains = tuple(str(d) for d in (meta.get("domains") or []))
    if not domains:
        raise FinanceBankError(f"{src}: institution {iid!r} declares no domains")

    documents = _build_documents(
        _load_yaml(inst_dir / "corpus.yaml"),
        src=str(inst_dir / "corpus.yaml"),
        institution_id=iid,
    )
    tools = _build_tools(
        _load_yaml(inst_dir / "tools.yaml"),
        src=str(inst_dir / "tools.yaml"),
        institution_id=iid,
    )
    tool_names = {t.name for t in tools}
    templates = _build_templates(
        _load_yaml(inst_dir / "tasks.yaml"),
        src=str(inst_dir / "tasks.yaml"),
        institution_id=iid,
        documents=documents,
        tool_names=tool_names,
    )
    embeddings = _load_embeddings(inst_dir / "embeddings.parquet")
    return Institution(
        institution_id=iid,
        display_name=str(meta.get("display_name", iid)),
        display_name_local=str(meta.get("display_name_local", "") or ""),
        type=_require_str(meta, "type", src),
        brand_voice=str(meta.get("brand_voice", "traditional")),
        domains=domains,
        documents=documents,
        tools=tools,
        templates=templates,
        placeholder=bool(meta.get("placeholder", False)),
        embeddings=embeddings,
    )


def _load_embeddings(path: Path) -> dict[str, tuple[float, ...]]:
    """Load the optional ``id -> vector`` dense-retrieval sidecar.

    Best-effort + non-fatal: a missing sidecar (or pandas/pyarrow being
    unavailable) yields an empty map, and ``retrieval.py`` degrades to the
    offline lexical/golden path. Never raises, so ``golden``-mode banks and
    CI stay hermetic.
    """
    if not path.exists():
        return {}
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover — pandas is a repo dep
        logger.warning("finance_bank: pandas unavailable, skipping %s: %s", path, e)
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.warning("finance_bank: failed to read %s: %s", path, e)
        return {}
    out: dict[str, tuple[float, ...]] = {}
    for _, row in df.iterrows():
        vec = row.get("embedding")
        if vec is None:
            continue
        try:
            out[str(row["id"])] = tuple(float(x) for x in vec)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Per-locale cache
# ---------------------------------------------------------------------------

_BANK_CACHE: dict[str, FinanceBank] = {}
_BANK_CACHE_LOCK = threading.Lock()


def default_finance_bank_dir(locale: str) -> Path:
    from usersim.engine.core._assets import probe_assets_dir

    return probe_assets_dir("financial_services") / locale


def finance_bank_dir_for(locale: str) -> Path:
    env_key = f"USERSIM_FINANCIAL_SERVICES_BANK_{locale.upper()}"
    override = os.environ.get(env_key)
    if override:
        return Path(override)
    return default_finance_bank_dir(locale)


def load_finance_bank_for_locale(locale: str) -> FinanceBank:
    """Return the cached bank for ``locale``, loading on first use (thread-safe)."""
    with _BANK_CACHE_LOCK:
        cached = _BANK_CACHE.get(locale)
        if cached is not None:
            return cached
        path = finance_bank_dir_for(locale)
        bank = load_finance_bank(path)
        _BANK_CACHE[locale] = bank
        logger.info(
            "finance_bank: loaded %s v%s (%d institutions) for locale=%s from %s",
            bank.bank_id,
            bank.bank_version,
            len(bank.institutions),
            locale,
            path,
        )
        return bank


def reset_finance_bank_cache() -> None:
    """Drop all cached banks. For tests, CLI reloads, bank-version bumps."""
    with _BANK_CACHE_LOCK:
        _BANK_CACHE.clear()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise FinanceBankError(f"{path}: file not found")
    try:
        with path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        raise FinanceBankError(f"{path}: YAML parse failure: {e}") from e
    if not isinstance(doc, dict):
        raise FinanceBankError(f"{path}: top-level must be a mapping")
    return doc


def _build_documents(doc: dict[str, Any], *, src: str, institution_id: str) -> tuple[Document, ...]:
    raw = doc.get("documents")
    if not isinstance(raw, list) or not raw:
        raise FinanceBankError(f"{src}::documents: must be a non-empty list")
    out: list[Document] = []
    seen: set = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise FinanceBankError(f"{src}::documents[{idx}]: must be a mapping")
        did = _require_str(entry, "id", src, ctx=f"documents[{idx}]")
        if did in seen:
            raise FinanceBankError(f"{src}::{did}: duplicate document id")
        seen.add(did)
        mentions = entry.get("mentions_tools", []) or []
        if not isinstance(mentions, list):
            raise FinanceBankError(f"{src}::{did}: mentions_tools must be a list")
        out.append(
            Document(
                id=did,
                title=_require_str(entry, "title", src, ctx=did),
                body=_require_str(entry, "body", src, ctx=did),
                product_category=_require_str(entry, "product_category", src, ctx=did),
                document_type=_require_str(entry, "document_type", src, ctx=did),
                source_authority=str(entry.get("source_authority", "official")),
                placeholder=_require_bool(entry, "placeholder", src, ctx=did),
                mentions_tools=tuple(str(m) for m in mentions),
                institution_id=institution_id,
                domain=str(entry.get("domain", "")),
            )
        )
    return tuple(out)


def _build_tools(doc: dict[str, Any], *, src: str, institution_id: str) -> tuple[ToolSpec, ...]:
    raw = doc.get("tools")
    if not isinstance(raw, list) or not raw:
        raise FinanceBankError(f"{src}::tools: must be a non-empty list")
    out: list[ToolSpec] = []
    seen: set = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise FinanceBankError(f"{src}::tools[{idx}]: must be a mapping")
        name = _require_str(entry, "name", src, ctx=f"tools[{idx}]")
        if name in seen:
            raise FinanceBankError(f"{src}::{name}: duplicate tool name")
        seen.add(name)
        side_effect = str(entry.get("side_effect_class", "read_only"))
        if side_effect not in ALLOWED_SIDE_EFFECTS:
            raise FinanceBankError(
                f"{src}::{name}: side_effect_class {side_effect!r} must be one of {sorted(ALLOWED_SIDE_EFFECTS)}"
            )
        params = entry.get("parameters", {}) or {}
        if not isinstance(params, dict):
            raise FinanceBankError(f"{src}::{name}: parameters must be a mapping")
        out.append(
            ToolSpec(
                name=name,
                description=str(entry.get("description", "")),
                discoverable=_require_bool(entry, "discoverable", src, ctx=name),
                side_effect_class=side_effect,
                parameters=params,
                institution_id=institution_id,
            )
        )
    return tuple(out)


#: Genres that document a tool (for auto-deriving a task's gold documents).
_TOOL_DOC_TYPES = ("policy_procedure", "discoverable_tool_doc")


def _derive_gold_docs(gold_tools: tuple[str, ...], documents: tuple["Document", ...]) -> tuple[str, ...]:
    """Gold docs for a tool task = the docs that DOCUMENT its gold tool(s).

    Resolved from the corpus (``mentions_tools`` ∩ gold tools, restricted to the
    tool-documenting genres) so tasks never hard-code doc ids that drift when the
    corpus is regenerated. Ordered policy first, then tool doc, then id.
    """
    tools = set(gold_tools)
    order = {t: i for i, t in enumerate(_TOOL_DOC_TYPES)}
    hits = [d for d in documents if set(d.mentions_tools) & tools and d.document_type in order]
    hits.sort(key=lambda d: (order[d.document_type], d.id))
    return tuple(d.id for d in hits)


def _build_templates(
    doc: dict[str, Any],
    *,
    src: str,
    institution_id: str,
    documents: tuple["Document", ...],
    tool_names: set,
) -> tuple[TaskTemplate, ...]:
    raw = doc.get("templates")
    if not isinstance(raw, list) or not raw:
        raise FinanceBankError(f"{src}::templates: must be a non-empty list")
    doc_ids = {d.id for d in documents}
    out: list[TaskTemplate] = []
    seen: set = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise FinanceBankError(f"{src}::templates[{idx}]: must be a mapping")
        tid = _require_str(entry, "id", src, ctx=f"templates[{idx}]")
        if tid in seen:
            raise FinanceBankError(f"{src}::{tid}: duplicate template id")
        seen.add(tid)
        tier = _require_str(entry, "tier", src, ctx=tid)
        if tier not in ALLOWED_TIERS:
            raise FinanceBankError(f"{src}::{tid}: tier {tier!r} must be one of {sorted(ALLOWED_TIERS)}")
        persona_tags = entry.get("persona_tags", []) or []
        if not isinstance(persona_tags, list):
            raise FinanceBankError(f"{src}::{tid}: persona_tags must be a list")
        for t in persona_tags:
            if not isinstance(t, str) or not _PERSONA_TAG_RE.match(t):
                raise FinanceBankError(
                    f"{src}::{tid}: persona_tag {t!r} does not match the `<prefix>:<value>` schema (see SCHEMA.md)"
                )
        gold_docs = tuple(str(d) for d in (entry.get("gold_document_ids", []) or []))
        gold_tools = tuple(str(t) for t in (entry.get("gold_tool_sequence", []) or []))
        allowed_tools = tuple(str(t) for t in (entry.get("allowed_tools", []) or []))
        # Scope integrity: Tier-1 gold must live within THIS institution.
        if tier == "verifiable":
            for t in gold_tools:
                if t not in tool_names:
                    raise FinanceBankError(
                        f"{src}::{tid}: gold_tool {t!r} not in institution "
                        f"{institution_id!r} tool taxonomy (cross-institution gold "
                        "is invalid)"
                    )
            for t in allowed_tools:
                if t not in tool_names:
                    raise FinanceBankError(
                        f"{src}::{tid}: allowed_tool {t!r} not in institution {institution_id!r} tool taxonomy"
                    )
            if gold_docs:
                # Explicit gold ids are an override -- must live in THIS corpus.
                for d in gold_docs:
                    if d not in doc_ids:
                        raise FinanceBankError(
                            f"{src}::{tid}: gold_document_id {d!r} not in institution "
                            f"{institution_id!r} corpus (cross-institution gold is invalid)"
                        )
            elif gold_tools:
                # Auto-derive gold docs from the gold tool(s): the docs that document
                # them. Keeps tasks free of brittle hand-typed corpus ids.
                gold_docs = _derive_gold_docs(gold_tools, documents)
                if not gold_docs:
                    raise FinanceBankError(
                        f"{src}::{tid}: no corpus document documents gold tool(s) "
                        f"{list(gold_tools)!r} (expected a {' or '.join(_TOOL_DOC_TYPES)} "
                        f"whose mentions_tools includes one) -- cannot derive gold docs"
                    )
        param_space_raw = entry.get("param_space", {}) or {}
        if not isinstance(param_space_raw, dict):
            raise FinanceBankError(f"{src}::{tid}: param_space must be a mapping")
        param_space = {k: tuple(v) if isinstance(v, list) else (v,) for k, v in param_space_raw.items()}
        out.append(
            TaskTemplate(
                id=tid,
                tier=tier,
                task_type=_require_str(entry, "task_type", src, ctx=tid),
                placeholder=_require_bool(entry, "placeholder", src, ctx=tid),
                persona_tags=tuple(persona_tags),
                opening_user_messages=_parse_openings(entry, src, tid),
                account_state=entry.get("account_state", {}) or {},
                param_space=param_space,
                gold_document_ids=gold_docs,
                gold_tool_sequence=gold_tools,
                expected_state_deltas=entry.get("expected_state_deltas", {}) or {},
                allowed_tools=allowed_tools,
                ordered=bool(entry.get("ordered", False)),
                institution_id=institution_id,
                domain=str(entry.get("domain", "")),
            )
        )
    return tuple(out)


def _parse_openings(d: dict[str, Any], src: str, tid: str) -> tuple[str, ...]:
    """Normalize a template's turn-1 phrasing(s) to a non-empty tuple.

    Accepts ``opening_user_messages`` (a list of strings, preferred) or the
    legacy singular ``opening_user_message`` (a string). At least one non-empty
    phrasing is required for a verifiable template.
    """
    msgs = d.get("opening_user_messages")
    if msgs is None:
        single = d.get("opening_user_message")
        msgs = [single] if single is not None else []
    if not isinstance(msgs, list):
        raise FinanceBankError(f"{src}::{tid}: opening_user_messages must be a list of strings")
    out = tuple(str(m).strip() for m in msgs if str(m).strip())
    if not out:
        raise FinanceBankError(f"{src}::{tid}: needs at least one opening_user_message(s)")
    return out


def _require_str(d: dict[str, Any], key: str, src: str, ctx: str | None = None) -> str:
    loc = f"{src}::{ctx}" if ctx else src
    if key not in d:
        raise FinanceBankError(f"{loc}: missing required field {key!r}")
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        raise FinanceBankError(f"{loc}: field {key!r} must be a non-empty string, got {type(v).__name__}")
    return v


def _require_bool(d: dict[str, Any], key: str, src: str, ctx: str | None = None) -> bool:
    loc = f"{src}::{ctx}" if ctx else src
    v = d.get(key, False)
    if not isinstance(v, bool):
        raise FinanceBankError(f"{loc}: field {key!r} must be a bool, got {type(v).__name__}")
    return v
