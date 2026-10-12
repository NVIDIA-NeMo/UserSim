# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load Data Designer model configurations from a TOML file.

The notebook hard-codes a list of ``dd.ModelConfig`` objects. The CLI
needs the same shape but expressed declaratively so users can edit a
text file (and pin it under version control) without touching Python.

A model TOML looks like::

    # Optional custom providers, for any OpenAI-compatible endpoint that is
    # not one of the built-in providers (``nvidia`` / ``openai`` /
    # ``openrouter``). ``api_key`` is the NAME of an environment variable,
    # never the key itself.
    [[providers]]
    name = "my-endpoint"
    endpoint = "https://my-inference-endpoint.example/v1"
    provider_type = "openai"
    api_key = "MY_ENDPOINT_API_KEY"

    [[models]]
    alias = "user_model"
    model = "openai/gpt-oss-20b"
    provider = "nvidia"
    max_tokens = 1024
    max_parallel_requests = 4
    temperature = 1.0
    top_p = 1.0

    [[models]]
    alias = "assistant_model"
    model = "openai/gpt-oss-120b"
    provider = "nvidia"
    timeout = 300
    extra_body = {reasoning_effort = "low"}

The simulator + evaluator both expect specific aliases:

- ``user_model`` (required for simulator) — drives the simulated user
- ``assistant_model`` (required for simulator) — the model under test
- ``api_response_model`` (tool-calling only) — synthesizes API responses
- ``judge_model`` (required for simulator) — in-sim user-query gate /
  capitulation classifier
- ``summary_model`` (optional) — context compression
- ``evaluator_model`` (required for evaluator) — rubric LLM for
  post-hoc trajectory evaluation; distinct from ``judge_model`` so the
  two roles can be sized / sourced independently

Missing required aliases raise ``ConfigError`` with a clear message.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from usersim.cli._errors import ConfigError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)


REQUIRED_SIMULATOR_ALIASES: tuple[str, ...] = (
    "user_model",
    "assistant_model",
    "judge_model",
)
RECOMMENDED_SIMULATOR_ALIASES: tuple[str, ...] = (
    "api_response_model",
    "summary_model",
)
REQUIRED_EVALUATOR_ALIASES: tuple[str, ...] = ("evaluator_model",)


@dataclass(frozen=True)
class ModelSpec:
    """Plain-old-data view of a model entry; converted to ``dd.ModelConfig`` on demand.

    ``temperature`` and ``top_p`` are ``Optional[float]``. ``None`` means
    "don't send this parameter to the model" -- some endpoints (e.g.
    Bedrock's newer Claude versions) deprecate ``top_p`` entirely and
    400 if it's present. Set to ``None`` in the catalog
    (``cli.model_catalog.INFERENCE_DEFAULTS``) for any model that
    rejects the param.
    """

    alias: str
    model: str
    provider: str
    max_tokens: int | None = None
    max_parallel_requests: int = 4
    temperature: float | None = 0.7
    top_p: float | None = 0.95
    timeout: float | None = None
    extra_body: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderSpec:
    """Plain-old-data view of a custom provider entry."""

    name: str
    endpoint: str
    provider_type: str
    api_key: str


@dataclass(frozen=True)
class ModelsConfig:
    providers: tuple[ProviderSpec, ...]
    models: tuple[ModelSpec, ...]
    #: Where this came from, so callers can report it. ``None`` for configs
    #: built in memory, such as in tests.
    source_path: Path | None = None

    def aliases(self) -> tuple[str, ...]:
        return tuple(m.alias for m in self.models)

    def get(self, alias: str) -> ModelSpec | None:
        for m in self.models:
            if m.alias == alias:
                return m
        return None


def load_models_config(path: str | Path) -> ModelsConfig:
    """Parse a model TOML and return a typed ``ModelsConfig``."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"models config not found: {p}")
    with p.open("rb") as f:
        try:
            doc = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"invalid TOML in {p}: {e}") from e

    providers = tuple(
        ProviderSpec(
            name=str(d["name"]),
            endpoint=str(d["endpoint"]),
            provider_type=str(d.get("provider_type", "openai")),
            api_key=str(d.get("api_key", "NVIDIA_API_KEY")),
        )
        for d in doc.get("providers", [])
    )

    raw_models = doc.get("models", [])
    if not raw_models:
        raise ConfigError(
            f"{p} has no [[models]] entries — at least user_model + assistant_model + judge_model are required"
        )
    models = tuple(_to_model_spec(d, source=p) for d in raw_models)

    config = ModelsConfig(providers=providers, models=models, source_path=p)
    aliases = set(config.aliases())
    if len(aliases) != len(models):
        raise ConfigError(f"{p} declares duplicate model aliases: {sorted(aliases)}")
    return config


def _to_model_spec(d: dict[str, Any], *, source: Path) -> ModelSpec:
    """Build a ``ModelSpec`` from a TOML row, filling omitted fields from
    the catalog (``cli.model_catalog.INFERENCE_DEFAULTS``).

    Resolution order, per field:

    1. Explicit value in the TOML row wins (TOML is self-describing).
    2. Catalog entry for ``model`` provides the default.
    3. Hard-coded usersim project default
       (``temperature=0.7`` / ``top_p=0.95`` / ``max_parallel_requests=4``).
       These are NOT DD's built-in defaults -- DD's
       ``ChatCompletionInferenceParams`` defaults to ``None`` for
       ``temperature`` / ``top_p`` -- they're project-level fallbacks
       preserved from before the catalog refactor.

    **Merge semantics: full replacement, NOT deep merge.** If a TOML row
    sets ``extra_body = {"thinking_budget": 100}`` and the catalog entry
    has ``extra_body = {"reasoning_effort": "low"}``, the resulting
    ``ModelSpec.extra_body`` is ``{"thinking_budget": 100}`` -- the
    catalog entry is fully replaced, not merged key-by-key. This makes
    the resolution rule trivially predictable (TOML row is
    self-describing for any field it sets) at the cost of forcing the
    user to re-state any catalog keys they want to keep.
    """
    # Lazy import to keep cli/--help working on environments without DD
    # installed (model_catalog has no DD dependency itself, but it's
    # convention in this module to keep heavy imports lazy).
    from usersim.cli.model_catalog import is_embedding_model, resolve_inference_defaults

    try:
        alias = str(d["alias"])
        model = str(d["model"])
        provider = str(d["provider"])
    except KeyError as e:
        raise ConfigError(
            f"{source} model entry missing required field: {e.args[0]} (need alias/model/provider)"
        ) from None

    catalog = resolve_inference_defaults(model)
    # An embedding model is catalogued in EMBEDDING_DEFAULTS and carries no
    # chat params, so its absence from INFERENCE_DEFAULTS is expected rather
    # than a gap to report.
    if not catalog and not is_embedding_model(model):
        logger.warning(
            "%s: alias=%r uses model=%r which isn't in "
            "cli/model_catalog.py:INFERENCE_DEFAULTS -- "
            "usersim project defaults will apply. Add an entry "
            "to the catalog if you want consistent params across runs.",
            source.name,
            alias,
            model,
        )

    # TOML has no null literal, so a row cannot write ``temperature = None``
    # to mean "send no temperature". ``drop_params`` is that expression: it
    # names the parameters to leave out of the request entirely. Needed for
    # reasoning models on a custom provider, which reject a non-default
    # temperature and reject top_p outright, and which the catalog cannot
    # cover because their ids are provider-specific.
    dropped = {str(x) for x in d.get("drop_params", ())}
    _DROPPABLE = {"temperature", "top_p", "max_tokens"}
    if unknown_drop := dropped - _DROPPABLE:
        raise ConfigError(
            f"{source.name}: alias={alias!r} has drop_params "
            f"{sorted(unknown_drop)}, which are not droppable. "
            f"Choose from {sorted(_DROPPABLE)}."
        )

    def pick(field: str, default: Any = None) -> Any:
        # TOML wins; otherwise catalog; otherwise hard-coded fallback.
        # NOTE: a catalog value of ``None`` is treated as "explicitly
        # drop this param" -- different from "field not in catalog".
        if field in dropped:
            return None
        if field in d:
            return d[field]
        if field in catalog:
            return catalog[field]
        return default

    def _maybe_float(v: Any) -> float | None:
        return None if v is None else float(v)

    return ModelSpec(
        alias=alias,
        model=model,
        provider=provider,
        max_tokens=pick("max_tokens"),
        max_parallel_requests=int(pick("max_parallel_requests", 4)),
        temperature=_maybe_float(pick("temperature", 0.7)),
        top_p=_maybe_float(pick("top_p", 0.95)),
        timeout=pick("timeout"),
        extra_body=pick("extra_body"),
    )


def require_aliases(config: ModelsConfig, *, required: tuple[str, ...], context: str) -> None:
    """Raise ``ConfigError`` if any required alias is absent."""
    have = set(config.aliases())
    missing = [a for a in required if a not in have]
    if missing:
        raise ConfigError(
            f"models config is missing required aliases for {context}: {missing}; declared aliases: {sorted(have)}"
        )


_OVERRIDE_KNOWN_KEYS = frozenset(
    {
        "model",
        "temperature",
        "top_p",
        "max_tokens",
        "extra_body",
    }
)


def parse_cli_model_overrides(pairs: list[str] | None) -> dict[str, str]:
    """Parse ``--model ALIAS=MODEL`` CLI values into an ``apply_model_overrides``
    mapping. ``pairs`` is the ``action="append"`` list (or ``None``). Whitespace
    is trimmed; a missing ``=`` or an empty side raises ``ConfigError``.
    """
    out: dict[str, str] = {}
    for item in pairs or ():
        alias, sep, model = item.partition("=")
        alias, model = alias.strip(), model.strip()
        if not sep or not alias or not model:
            raise ConfigError(f"--model expects ALIAS=MODEL, got {item!r}")
        out[alias] = model
    return out


def parse_cli_parallel_overrides(pairs: list[str] | None) -> dict[str, int]:
    """Parse ``--max-parallel ALIAS=N`` CLI values into an alias->int mapping.
    Raises ``ConfigError`` on a missing ``=``, empty side, or non-positive int.
    """
    out: dict[str, int] = {}
    for item in pairs or ():
        alias, sep, n = item.partition("=")
        alias, n = alias.strip(), n.strip()
        if not sep or not alias or not n:
            raise ConfigError(f"--max-parallel expects ALIAS=N, got {item!r}")
        try:
            val = int(n)
        except ValueError:
            raise ConfigError(f"--max-parallel N must be an integer, got {n!r}") from None
        if val < 1:
            raise ConfigError(f"--max-parallel N must be >= 1, got {val}")
        out[alias] = val
    return out


def apply_parallel_overrides(config: ModelsConfig, overrides: dict[str, int]) -> ModelsConfig:
    """Return a new ``ModelsConfig`` with ``max_parallel_requests`` swapped per alias.

    Unlike ``apply_model_overrides`` (which re-resolves the whole spec), this only
    touches the per-alias concurrency knob. An override targeting an unknown alias
    raises ``ConfigError``.
    """
    import dataclasses

    if not overrides:
        return config
    have = set(config.aliases())
    missing = [a for a in overrides if a not in have]
    if missing:
        raise ConfigError(f"--max-parallel targets unknown alias(es): {missing}; declared aliases: {sorted(have)}")
    new_models = tuple(
        dataclasses.replace(m, max_parallel_requests=overrides[m.alias]) if m.alias in overrides else m
        for m in config.models
    )
    return dataclasses.replace(config, models=new_models)


def apply_model_overrides(
    config: ModelsConfig,
    overrides: dict[str, str | dict[str, Any]],
) -> ModelsConfig:
    """Return a new ``ModelsConfig`` with ``model`` strings (and optionally
    other per-model knobs) swapped per alias.

    ``overrides`` maps alias to either:

    - **str** (simple form): ``{"judge_model": "openai/openai/gpt-5.5"}``
      Just swap the model. Catalog defaults for the new model fill in
      ``temperature`` / ``top_p`` / ``max_tokens`` / ``extra_body``.

    - **dict** (extended form): per-knob override on top of the catalog.
      The dict MUST include ``"model"`` and may optionally include any
      of ``"temperature"``, ``"top_p"``, ``"max_tokens"``, ``"extra_body"``::

          MODEL_OVERRIDES = {
              "assistant_model": {
                  "model": "nvidia/google/gemma-4-31b-it",
                  "extra_body": {},   # disable catalog's enable_thinking
              },
              "judge_model": {
                  "model": "openai/openai/gpt-5.5",
                  "temperature": 0.0, # deterministic eval
                  "max_tokens": 4096, # tighter than catalog default
              },
          }

      Per-knob precedence: **dict-form value > catalog default > old spec**.
      A key present in the dict (even with value ``None`` or ``{}``)
      ALWAYS wins, distinguishing "explicit clear" from "absent / fall
      through to catalog". Unknown keys raise ``ConfigError``.

    For each matching alias, the new model's catalog defaults
    (``cli.model_catalog.INFERENCE_DEFAULTS``) replace any per-model
    fields not explicitly overridden in the dict form; ``provider`` is
    inferred from the catalog (see below); the per-alias fields
    (``max_parallel_requests`` / ``timeout``) carry over unchanged.

    Provider inference rule:

    - If the new model is in ``cli.model_catalog.VLLM_DEFAULTS`` (i.e.
      self-hosted via vLLM), pick the first non-custom provider name
      that already appears in ``config.models`` (typically
      ``"nvidia"``).
    - If the new model is NOT in ``VLLM_DEFAULTS`` (i.e. served over an
      endpoint rather than a local vLLM), pick the first custom provider
      declared in ``config.providers``.
    - If neither rule matches, raise ``ConfigError``: there is no endpoint
      known to serve the model, and silently keeping the alias's previous
      provider would send the request somewhere that does not.

    Catalog re-resolution handles edge cases like swapping
    ``openai/gpt-oss-120b`` (catalog supplies
    ``{"reasoning_effort": "high"}``) for ``openai/openai/gpt-5.5``
    (catalog has no ``extra_body`` -- gpt-5.5 rejects
    ``reasoning_effort``). Without re-resolving, the swap would carry
    over the gpt-oss knob and the inference call would 400. Same idea
    for provider: swapping a hub-served model in for a self-hosted one
    would otherwise leave the alias pointing at the wrong endpoint.

    **Unknown-model safety.** When the new model isn't in the catalog
    (and the dict form doesn't pin ``extra_body`` explicitly),
    ``extra_body`` is **cleared** (not inherited from the old spec).
    ``extra_body`` keys are model-family-specific (gpt-oss
    ``reasoning_effort``, gemma ``chat_template_kwargs``, anthropic
    ``thinking``); inheriting one family's knob across to another model
    silently risks 400s or unexpected behavior at the inference layer.
    A WARNING-level log surfaces the cleared keys so the user can decide
    whether to re-add them explicitly. ``temperature`` / ``top_p`` /
    ``max_tokens`` continue to fall back to the old spec because they're
    generic sampling parameters that round-trip across providers.

    Empty / None ``overrides`` returns ``config`` unchanged.

    Raises ``ConfigError`` if an override targets an unknown alias, the
    dict form omits ``"model"``, the dict form contains unknown keys,
    or the override value is neither a str nor a dict.
    """
    import dataclasses

    if not overrides:
        return config
    from usersim.cli.model_catalog import (
        known_vllm_models,
        resolve_inference_defaults,
        resolve_vllm_defaults,
    )

    have_aliases = set(config.aliases())
    unknown = [a for a in overrides if a not in have_aliases]
    if unknown:
        raise ConfigError(
            f"model overrides target unknown alias(es) {unknown}; declared aliases: {sorted(have_aliases)}"
        )

    # Normalize each override to the dict form upfront so the per-spec
    # loop below can use a uniform shape. Fail fast on misconfiguration
    # so a typo doesn't silently slip through and hit the inference layer.
    normalized: dict[str, dict[str, Any]] = {}
    for alias, ov in overrides.items():
        if isinstance(ov, str):
            normalized[alias] = {"model": ov}
        elif isinstance(ov, dict):
            if "model" not in ov:
                raise ConfigError(
                    f"model override for alias {alias!r}: dict form must "
                    f"include a 'model' key; got keys={sorted(ov.keys())}"
                )
            unknown_keys = set(ov.keys()) - _OVERRIDE_KNOWN_KEYS
            if unknown_keys:
                raise ConfigError(
                    f"model override for alias {alias!r}: unknown key(s) "
                    f"{sorted(unknown_keys)}; supported: "
                    f"{sorted(_OVERRIDE_KNOWN_KEYS)}"
                )
            normalized[alias] = dict(ov)  # defensive copy
        else:
            raise ConfigError(
                f"model override for alias {alias!r}: must be a str "
                f"(model name) or a dict (per-knob override), got "
                f"{type(ov).__name__}"
            )

    custom_provider_names = {p.name for p in config.providers}

    # Pre-compute the provider used by self-hosted (built-in) aliases.
    # First non-custom provider seen in config.models -- typically "nvidia".
    # If no alias is currently on a built-in (e.g. the user's notebook
    # iteration has every alias hub-routed), fall back to the canonical
    # built-in name "nvidia" so a swap to a self-hosted model
    # (e.g. by a downstream config rewriter) routes correctly even though
    # no incumbent alias anchors the tier.
    self_hosted_provider: str = (
        next(
            (s.provider for s in config.models if s.provider not in custom_provider_names),
            None,
        )
        or "nvidia"
    )
    # Pre-compute the provider used by externally-routed (custom) aliases.
    # The first custom provider declared in the models config.
    external_provider: str | None = config.providers[0].name if config.providers else None

    new_specs: list[ModelSpec] = []
    for spec in config.models:
        ov = normalized.get(spec.alias)
        if ov is None:
            new_specs.append(spec)
            continue
        new_model = ov["model"]
        catalog = resolve_inference_defaults(new_model)
        if not catalog:
            # Catalog gap: warn so the user knows the new model has no
            # canonical defaults registered. The dict form's per-knob
            # override (if any) still applies on top.
            old_extra_body = spec.extra_body or {}
            extra_body_in_override = "extra_body" in ov
            if old_extra_body and not extra_body_in_override:
                # We're about to clear it (model-family-specific keys
                # don't bleed across families). Name the keys so the
                # user knows what to re-add explicitly if it WAS
                # intentional.
                logger.warning(
                    "model override: alias=%r switched to model=%r which "
                    "isn't in cli/model_catalog.py:INFERENCE_DEFAULTS -- "
                    "clearing the old model's extra_body (%s) so its "
                    "model-family-specific knobs don't bleed across; "
                    "temperature / top_p / max_tokens are kept from the "
                    "old spec. Add the new model to INFERENCE_DEFAULTS "
                    "(or set extra_body explicitly on the override) "
                    "to silence this warning.",
                    spec.alias,
                    new_model,
                    sorted(old_extra_body.keys()),
                )
            elif not extra_body_in_override:
                logger.warning(
                    "model override: alias=%r switched to model=%r which "
                    "isn't in cli/model_catalog.py:INFERENCE_DEFAULTS -- "
                    "temperature / top_p / max_tokens are kept from the "
                    "old spec. Add the new model to INFERENCE_DEFAULTS "
                    "to pick up its native sampling defaults.",
                    spec.alias,
                    new_model,
                )
            # If extra_body IS in the dict-form override, the user is
            # taking explicit responsibility -- no warning needed.

        # Provider inference: VLLM_DEFAULTS membership decides which tier
        # serves the model. A model outside that set needs an explicitly
        # declared provider; routing it to whatever the alias used before
        # would send the request to an endpoint that does not serve it.
        is_self_hosted = bool(resolve_vllm_defaults(new_model))
        if is_self_hosted:
            new_provider = self_hosted_provider
        elif external_provider is not None:
            new_provider = external_provider
        else:
            raise ConfigError(
                f"cannot route model {new_model!r} for alias {spec.alias!r}: it is "
                f"not served by the {self_hosted_provider!r} provider and the models "
                f"config declares no custom provider to send it to. Add a "
                f"[[providers]] block naming an OpenAI-compatible endpoint that "
                f"serves it, or pick a model from: "
                f"{', '.join(sorted(known_vllm_models()))}."
            )
        if new_provider != spec.provider:
            logger.info(
                "model override: alias=%r model=%r -> %r, provider auto-routed %r -> %r (%s tier)",
                spec.alias,
                spec.model,
                new_model,
                spec.provider,
                new_provider,
                "self-hosted" if is_self_hosted else "external",
            )

        # Per-knob precedence: dict-form value > catalog default > old
        # spec (or None for extra_body when the catalog is empty).
        # ``"key" in ov`` distinguishes "explicit override" (including
        # explicit ``None`` / ``{}``) from "absent" so users can clear
        # the catalog default when they need to (e.g. disable thinking
        # by setting ``extra_body={}`` on the gemma row).
        new_temperature = ov["temperature"] if "temperature" in ov else catalog.get("temperature", spec.temperature)
        new_top_p = ov["top_p"] if "top_p" in ov else catalog.get("top_p", spec.top_p)
        new_max_tokens = ov["max_tokens"] if "max_tokens" in ov else catalog.get("max_tokens", spec.max_tokens)
        if "extra_body" in ov:
            new_extra_body = ov["extra_body"]
        elif catalog:
            # Sampling params round-trip across families; extra_body
            # does NOT (gpt-oss's ``reasoning_effort`` would 400 on
            # gemma; gemma's ``chat_template_kwargs`` is meaningless
            # to gpt-5.5). Catalog wins when present; clear when absent.
            new_extra_body = catalog.get("extra_body")
        else:
            new_extra_body = None

        new_specs.append(
            dataclasses.replace(
                spec,
                model=new_model,
                provider=new_provider,
                temperature=new_temperature,
                top_p=new_top_p,
                max_tokens=new_max_tokens,
                extra_body=new_extra_body,
            )
        )
    return dataclasses.replace(config, models=tuple(new_specs))


def to_data_designer_kwargs(config: ModelsConfig) -> dict[str, Any]:
    """Convert ``ModelsConfig`` into kwargs for ``DataDesigner(...)``.

    Returns ``{"model_providers": [...]}`` if the config declares any
    custom providers, otherwise an empty dict (so DD uses its defaults).
    Lazily imports ``data_designer`` so the CLI ``--help`` works without
    DD installed.

    When custom providers are present, we MUST also include the built-in
    DD providers (``nvidia`` / ``openai`` / ``openrouter``) in the list.
    DD's ``DataDesigner.__init__`` treats a non-None ``model_providers``
    argument as a complete replacement of the defaults — passing just
    our custom proxy provider would drop the ``nvidia`` built-in that
    ``assistant_model`` references, raising ``UnknownProviderError`` at
    inference time. Custom providers take precedence on name conflicts
    so a user can override a built-in by re-declaring it.
    """
    if not config.providers:
        return {}
    from data_designer.config.default_model_settings import (
        get_builtin_model_providers,
        get_default_providers,
    )
    from data_designer.config.models import ModelProvider  # noqa: F401  (lazy)

    custom = [
        ModelProvider(
            name=p.name,
            endpoint=p.endpoint,
            provider_type=p.provider_type,
            api_key=p.api_key,
        )
        for p in config.providers
    ]
    # ``get_default_providers`` reads the user's seeded providers file
    # (which DD writes on first run with the built-ins). If that file
    # is missing for any reason, fall back to the constant built-ins
    # so the merge still produces a usable registry.
    try:
        defaults = get_default_providers() or get_builtin_model_providers()
    except FileNotFoundError:
        defaults = get_builtin_model_providers()

    custom_names = {p.name for p in custom}
    merged: list[Any] = list(custom) + [p for p in defaults if p.name not in custom_names]
    return {"model_providers": merged}


def to_model_configs(config: ModelsConfig) -> list[Any]:
    """Convert ``ModelsConfig`` into a list of ``dd.ModelConfig`` objects.

    Lazy-imports ``data_designer`` for the same reason as above.
    """
    import data_designer.config as dd

    from usersim.cli.model_catalog import is_embedding_model, resolve_embedding_defaults

    out: list[Any] = []
    for m in config.models:
        # Embedding models live on the /embeddings route -- they must use
        # EmbeddingInferenceParams (chat sampling knobs like temperature /
        # top_p / max_tokens are invalid there, and a chat health check 404s).
        if is_embedding_model(m.model):
            emb_defaults = resolve_embedding_defaults(m.model)
            emb_kwargs: dict[str, Any] = dict(
                max_parallel_requests=m.max_parallel_requests,
            )
            if m.timeout is not None:
                emb_kwargs["timeout"] = float(m.timeout)
            # Provider-specific request fields (e.g. qwen input_type / truncate). The
            # TOML/override extra_body wins; else fall back to the catalog default so
            # required fields are present even when an override cleared extra_body.
            extra = m.extra_body if m.extra_body is not None else emb_defaults.get("extra_body")
            if extra:
                emb_kwargs["extra_body"] = dict(extra)
            if emb_defaults.get("encoding_format"):
                emb_kwargs["encoding_format"] = emb_defaults["encoding_format"]
            if emb_defaults.get("dimensions") is not None:
                emb_kwargs["dimensions"] = emb_defaults["dimensions"]
            params = dd.EmbeddingInferenceParams(**emb_kwargs)
        else:
            params_kwargs: dict[str, Any] = dict(
                max_parallel_requests=m.max_parallel_requests,
            )
            # None means "don't send this param" -- some endpoints (e.g.
            # Bedrock's newer Claude versions) reject top_p with a 400
            # "deprecated for this model" error. Treat None as a signal
            # to omit the param from the inference call entirely.
            if m.temperature is not None:
                params_kwargs["temperature"] = m.temperature
            if m.top_p is not None:
                params_kwargs["top_p"] = m.top_p
            if m.max_tokens is not None:
                params_kwargs["max_tokens"] = int(m.max_tokens)
            if m.timeout is not None:
                params_kwargs["timeout"] = float(m.timeout)
            if m.extra_body is not None:
                params_kwargs["extra_body"] = dict(m.extra_body)
            params = dd.ChatCompletionInferenceParams(**params_kwargs)
        out.append(
            dd.ModelConfig(
                alias=m.alias,
                model=m.model,
                provider=m.provider,
                inference_parameters=params,
            )
        )
    return out


#: Filename that, when present, overrides the bundled default config. Named
#: ``*.local.toml`` so the gitignore rule covers it.
LOCAL_MODELS_FILENAME = "models.local.toml"


def bundled_models_path() -> Path:
    """Return the path to the models TOML shipped inside the package."""
    return Path(__file__).resolve().parent / "models_default.toml"


def local_models_path(start: Path | None = None) -> Path | None:
    """Return a ``models.local.toml`` found at or above ``start``, if any.

    Lets a developer point every entry point at their own endpoint without
    editing a tracked file. Returns ``None`` when there is none, which is the
    normal case for a user who has not created one.
    """
    here = (start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / LOCAL_MODELS_FILENAME
        if candidate.is_file():
            return candidate
    return None


def default_models_path() -> Path:
    """Return the models TOML to use when the caller named none.

    Resolution order, first match wins:

    1. ``$USERSIM_MODELS_CONFIG``, so one variable can redirect every command.
    2. A ``models.local.toml`` in the working directory or any parent.
    3. The bundled ``models_default.toml``.

    An explicit ``--models`` path, or a path passed directly in a notebook,
    bypasses all of this. Use :func:`describe_models_source` to report which
    one won, since silently running against a different endpoint than you
    expect is worth one line of output.
    """
    from_env = os.environ.get("USERSIM_MODELS_CONFIG")
    if from_env:
        path = Path(from_env).expanduser()
        if not path.is_file():
            raise ConfigError(
                f"USERSIM_MODELS_CONFIG points at {str(path)!r}, which is not a file. Unset it or correct the path."
            )
        return path
    return local_models_path() or bundled_models_path()


def describe_models_source(path: Path) -> str:
    """Return a one-line explanation of why ``path`` is the config in use."""
    resolved = Path(path).resolve()
    if os.environ.get("USERSIM_MODELS_CONFIG"):
        env_path = Path(os.environ["USERSIM_MODELS_CONFIG"]).expanduser()
        if env_path.is_file() and env_path.resolve() == resolved:
            return f"{resolved} (from $USERSIM_MODELS_CONFIG)"
    if resolved.name == LOCAL_MODELS_FILENAME:
        return f"{resolved} (local override; the bundled default is unused)"
    if resolved == bundled_models_path().resolve():
        return f"{resolved} (bundled default)"
    return str(resolved)


# Built-in DD providers and the env vars they read from. Mirrors
# ``data_designer.config.utils.constants.PREDEFINED_PROVIDERS`` so we
# can enumerate the right env var for a given provider name without
# importing DD (the warning helper has to work on doc-build VMs that
# don't have data_designer installed).
_BUILTIN_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Built-in DD provider endpoints. Same data as in
# ``data_designer.config.utils.constants.PREDEFINED_PROVIDERS`` but
# duplicated here so smoke_test_models can run without importing DD.
_BUILTIN_PROVIDER_ENDPOINTS: dict[str, str] = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def required_api_key_env_vars(config: ModelsConfig) -> dict[str, str]:
    """Map each API-key env var ``config`` needs to the provider that needs it.

    Covers custom providers' ``api_key`` field plus the env vars for any
    built-in providers (``nvidia`` / ``openai`` / ``openrouter``) referenced by
    a ``ModelSpec``. A provider we cannot map to an env var is omitted rather
    than guessed at.
    """
    custom_provider_envs = {p.name: p.api_key for p in config.providers}
    needed: dict[str, str] = {}  # env_var -> provider_name (first wins)
    for spec in config.models:
        env_var = custom_provider_envs.get(spec.provider) or _BUILTIN_PROVIDER_API_KEY_ENV.get(spec.provider)
        if env_var is None:
            continue
        needed.setdefault(env_var, spec.provider)
    return needed


def warn_if_missing_api_key(
    config: ModelsConfig | None = None,
) -> str | None:
    """If any required API-key env var is unset, return a warning; else None.

    With no ``config``, falls back to checking ``NVIDIA_API_KEY`` only —
    the historical behavior before per-provider env vars were a thing.

    With a ``config``, enumerates every env var the loaded model catalogue
    needs (custom providers' ``api_key`` field plus the env vars for any
    built-in providers — ``nvidia`` / ``openai`` / ``openrouter`` —
    referenced by ``ModelSpec.provider``) and warns on any that are unset.
    """
    if config is None:
        if not os.environ.get("NVIDIA_API_KEY"):
            return (
                "NVIDIA_API_KEY is not set; LLM calls will fail. "
                "Export your key before running simulate/eval, or use --dry-run."
            )
        return None

    needed = required_api_key_env_vars(config)
    missing = sorted(f"{env} (provider={provider!r})" for env, provider in needed.items() if not os.environ.get(env))
    if not missing:
        return None
    return (
        f"Missing API key env var(s): {', '.join(missing)}. Export them before running simulate/eval, or use --dry-run."
    )


def smoke_test_models(
    config: ModelsConfig,
    *,
    timeout_sec: float = 15.0,
    aliases: Iterable[str] | None = None,
) -> dict[str, tuple[bool, str]]:
    """Send a tiny probe to each (model, provider) in ``config``.

    Bypasses litellm / DD entirely and hits each provider directly with the
    resolved env-var credential -- chat models at ``/chat/completions`` and
    embedding models at ``/embeddings`` (routing by model type, so an embedding
    model isn't probed as a chat model, which the hub reports as a 404 unknown
    model_group). Catches the failure modes that DD's own per-run health
    check misses:

    - DD only health-checks one model per provider. Per-model gating
      (e.g. an entitlement was withdrawn from a specific model on
      build.nvidia.com) won't surface until the simulator hits that
      model mid-run.
    - The notebook process and the user's shell may have different
      ``NVIDIA_API_KEY`` values (notebook started before the key was
      refreshed). Direct curl masks this; an inline test running in
      the notebook process catches it.

    ``aliases`` lets callers scope the check to a subset of model
    aliases (e.g. ``aliases=REQUIRED_EVALUATOR_ALIASES`` from the eval
    notebook, which only invokes ``evaluator_model`` -- testing the
    simulator's user / assistant / judge / summary aliases there would
    burn HTTP round-trips on models the evaluator never calls).
    ``None`` (default) tests every alias in ``config``.

    Returns a dict mapping ``alias -> (ok: bool, message: str)``. A
    failed entry's message includes the HTTP status code + the first
    ~200 chars of the response body so the user can see whether it's
    auth (401), entitlement (403), model-not-found (404), or rate
    limit (429).

    Each call costs ~1 token of inference; total is at most one call
    per unique (model, provider) pair. Uses ``httpx`` (already a
    transitive dep) so doesn't pull in litellm overhead.
    """
    import httpx

    from usersim.cli.model_catalog import is_embedding_model, resolve_embedding_defaults

    custom_by_name = {p.name: p for p in config.providers}
    results: dict[str, tuple[bool, str]] = {}
    seen: set[tuple[str, str]] = set()

    alias_filter = set(aliases) if aliases is not None else None
    selected = config.models if alias_filter is None else [s for s in config.models if s.alias in alias_filter]

    for spec in selected:
        # Resolve endpoint + env-var name for the spec's provider.
        if spec.provider in custom_by_name:
            cp = custom_by_name[spec.provider]
            endpoint, env_var = cp.endpoint, cp.api_key
        elif spec.provider in _BUILTIN_PROVIDER_ENDPOINTS:
            endpoint = _BUILTIN_PROVIDER_ENDPOINTS[spec.provider]
            env_var = _BUILTIN_PROVIDER_API_KEY_ENV.get(spec.provider, "")
        else:
            results[spec.alias] = (False, f"unknown provider {spec.provider!r}")
            continue

        # Suffix the routing on every message so the notebook output makes
        # it unambiguous WHERE the test is hitting and WHICH model name it
        # sent. This is the most common point of confusion when smoke
        # tests fail ("which endpoint did you actually try?").
        route_suffix = f"  [{endpoint}, model={spec.model}]"

        # Dedup by (model, provider) -- same pair across aliases tests once.
        key_pair = (spec.model, spec.provider)
        if key_pair in seen:
            # Inherit the prior result for the same (model, provider).
            prior_alias = next(
                a for a, (_, m) in results.items() if (config.get(a).model, config.get(a).provider) == key_pair
            )
            results[spec.alias] = results[prior_alias]
            continue
        seen.add(key_pair)

        api_key = os.environ.get(env_var, "") if env_var else ""
        if not api_key:
            results[spec.alias] = (
                False,
                f"env var {env_var!r} not set" + route_suffix,
            )
            continue

        # Embedding models live on the /embeddings route and reject a
        # chat-completion body: probing them at /chat/completions makes the
        # hub's router report the model as an unknown chat model_group (404),
        # which looks like "model missing" but is really "wrong route". Route
        # by model type so the smoke test matches how DD actually calls each.
        if is_embedding_model(spec.model):
            url = f"{endpoint.rstrip('/')}/embeddings"
            payload = {"model": spec.model, "input": "ping"}
            # Provider-specific required fields (e.g. qwen input_type/truncate)
            # -- mirror to_model_configs so the probe body matches the real call.
            emb_defaults = resolve_embedding_defaults(spec.model)
            extra = spec.extra_body if spec.extra_body is not None else emb_defaults.get("extra_body")
            if extra:
                payload.update(extra)
            if emb_defaults.get("encoding_format"):
                payload["encoding_format"] = emb_defaults["encoding_format"]
        else:
            url = f"{endpoint.rstrip('/')}/chat/completions"
            # max_tokens=256: reasoning models (gpt-5.5, Claude, etc.) emit
            # thinking tokens that count against the budget, so a tight
            # cap (e.g. 1) causes an HTTP 400 "could not finish, max_tokens
            # reached" even when auth / routing / model existence are all
            # fine. 256 covers most reasoning preambles on a short prompt
            # while keeping per-call cost negligible (<$0.01 / smoke run).
            payload = {
                "model": spec.model,
                "messages": [{"role": "user", "content": "Say hi briefly."}],
                "max_tokens": 256,
            }
            # Send the sampling parameters this alias resolved to, so the
            # probe exercises what a real run sends. Without this the probe
            # passes on a model that rejects the configured temperature or
            # top_p, and the failure only appears later at the provider's
            # health check. ``None`` means the config omits the parameter,
            # so it is omitted here too.
            if spec.temperature is not None:
                payload["temperature"] = spec.temperature
            if spec.top_p is not None:
                payload["top_p"] = spec.top_p
            if spec.extra_body:
                payload.update(spec.extra_body)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=timeout_sec)
            if r.status_code == 400 and _rejects_max_tokens(r.text) and "max_tokens" in payload:
                # Reasoning endpoints count hidden thinking tokens toward the
                # output budget, so they take ``max_completion_tokens`` and
                # reject ``max_tokens`` outright. Retry under the name this
                # endpoint accepts rather than reporting a routing failure.
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                r = httpx.post(url, json=payload, headers=headers, timeout=timeout_sec)
        except Exception as e:
            results[spec.alias] = (
                False,
                f"{type(e).__name__}: {e}" + route_suffix,
            )
            continue

        if r.status_code == 200:
            results[spec.alias] = (True, "OK" + route_suffix)
        elif r.status_code == 400 and _is_max_tokens_400(r.text):
            # Provisioning works; the smoke test just under-budgeted a
            # reasoning model. Don't fail the run for this.
            results[spec.alias] = (
                True,
                "OK (output capped by smoke max_tokens; provisioning fine)" + route_suffix,
            )
        else:
            detail = r.text[:200].replace("\n", " ").strip()
            results[spec.alias] = (
                False,
                f"HTTP {r.status_code}: {detail}" + route_suffix,
            )

    return results


def _rejects_max_tokens(body: str) -> bool:
    """Detect the 400 that fires when an endpoint refuses ``max_tokens``.

    Reasoning endpoints replaced it with ``max_completion_tokens``, because
    hidden thinking tokens are billed as output and the older field was
    ambiguous about whether it capped them. Distinct from
    :func:`_is_max_tokens_400`, which is the budget actually running out.
    """
    lower = body.lower()
    return "max_completion_tokens" in lower and (
        "unsupported" in lower or "not supported" in lower or "instead" in lower
    )


def _is_max_tokens_400(body: str) -> bool:
    """Detect the specific 400 that fires when a reasoning model exhausts
    the smoke test's max_tokens budget. The error text varies slightly
    across providers (NVIDIA hub, OpenAI, Anthropic) so we match on the
    common substrings rather than a single canonical phrase."""
    lower = body.lower()
    return any(
        marker in lower
        for marker in (
            "max_tokens or model output limit was reached",
            "max_tokens reached",
            "could not finish the message because max_tokens",
        )
    )


def print_resolved_models(config: ModelsConfig) -> None:
    """One-line-per-alias summary of the loaded model configuration.

    Prints alias / model / provider / temperature / top_p / max_tokens /
    max_parallel_requests / optional ``extra_body`` + ``timeout``. Used by
    both notebooks and the CLI smoke output to confirm what was loaded
    before any inference call goes out.
    """
    used_providers = sorted({s.provider for s in config.models})
    if config.source_path is not None:
        print(f"Models config: {describe_models_source(config.source_path)}")
    print(f"Loaded {len(config.models)} models (providers: {used_providers or '(builtins only)'})")
    print("Resolved inference defaults (TOML row + cli/model_catalog.py):")
    for spec in config.models:
        extras = ""
        if spec.extra_body:
            extras += f"  extra_body={spec.extra_body}"
        if spec.timeout:
            extras += f"  timeout={spec.timeout}"
        print(
            f"  {spec.alias:20s} {spec.model:30s} provider={spec.provider:25s} "
            f"temp={spec.temperature}  top_p={spec.top_p}  "
            f"max_tokens={spec.max_tokens}  max_parallel={spec.max_parallel_requests}"
            f"{extras}"
        )


def prompt_for_provider_keys(config: ModelsConfig) -> None:
    """Prompt for any provider env-var that is needed by a CURRENTLY-USED
    alias and not already set in ``os.environ``.

    Uses ``getpass`` so the value is not echoed; idempotent (skips
    providers whose env var is already populated). No-op for built-in
    providers (their env vars are documented and the user is expected to
    set them out-of-band; the smoke test catches missing-key cases with
    a clean HTTP 401 / "auth failed" message).
    """
    import getpass as _getpass

    used = {spec.provider for spec in config.models}
    for provider in config.providers:
        if provider.name not in used:
            continue
        if os.environ.get(provider.api_key):
            continue
        value = _getpass.getpass(f"Enter {provider.api_key} ({provider.endpoint}): ")
        if value:
            os.environ[provider.api_key] = value


def smoke_test_models_or_raise(
    config: ModelsConfig,
    *,
    timeout_sec: float = 15.0,
    aliases: Iterable[str] | None = None,
) -> dict[str, tuple[bool, str]]:
    """Run ``smoke_test_models`` and raise ``RuntimeError`` if any check fails.

    Prints one line per alias with a check / cross badge so the user sees
    exactly which (model, provider) pair tripped before the exception.
    Returns the same result dict ``smoke_test_models`` would return so
    callers (tests, the notebook) can inspect / log if they catch the raise.

    ``aliases`` scopes the check to a subset (see ``smoke_test_models``).
    """
    selected = list(aliases) if aliases is not None else None
    scope_msg = (
        f"({len(selected)} alias{'es' if len(selected) != 1 else ''}: {selected})"
        if selected is not None
        else f"({len(config.models)} aliases)"
    )
    print(f"\nSmoke-testing each (model, provider) pair {scope_msg} -- ~256 output tokens each, sub-cent cost...")
    results = smoke_test_models(config, timeout_sec=timeout_sec, aliases=selected)
    failed: list[str] = []
    for alias, (ok, msg) in results.items():
        badge = "\u2713" if ok else "\u2717"
        print(f"  {badge} {alias:20s} {msg}")
        if not ok:
            failed.append(alias)
    if failed:
        raise RuntimeError(
            f"{len(failed)} model(s) failed the smoke test: "
            f"{failed}. See messages above. Fix the underlying issue "
            "(env var, model gating, endpoint URL) before running the "
            "evaluator / simulator."
        )
    return results
