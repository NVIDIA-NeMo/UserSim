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
from typing import Any

from usersim.engine.errors import ConfigError

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
    (``engine.model_catalog.INFERENCE_DEFAULTS``) for any model that
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
    #: Send this model the reasoning of its own earlier messages, for a model
    #: trained to receive it. Honoured on ``assistant_model`` only; see
    #: :func:`assistant_replays_reasoning`.
    replay_reasoning: bool = False


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
    the catalog (``engine.model_catalog.INFERENCE_DEFAULTS``).

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
    from usersim.engine.model_catalog import is_embedding_model, resolve_inference_defaults

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
            "engine/model_catalog.py:INFERENCE_DEFAULTS -- "
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
        replay_reasoning=replay_flag(pick("replay_reasoning", False), where=f"{source.name}: alias={alias!r}"),
    )


def replay_flag(value: Any, *, where: str) -> bool:
    """``replay_reasoning`` as given, which must be a real boolean.

    A string such as ``"false"`` is refused rather than read as true.
    """
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: replay_reasoning must be true or false, got {value!r}")
    return value


def assistant_replays_reasoning(config: ModelsConfig, *, store_reasoning: bool) -> bool:
    """Whether a run replays the assistant's own reasoning; refuses a setup that cannot.

    ``replay_reasoning`` is honoured on ``assistant_model`` only: the
    assistant's requests are the only ones that hold its own earlier messages,
    and a stored message does not record which model produced it. Raises
    ``ConfigError`` when the switch is set on another alias, when reasoning is
    not stored (there would be nothing to replay), or when the assistant's
    provider is not OpenAI-compatible, since only that Data Designer adapter
    sends ``reasoning_content`` back (the Anthropic one drops it).
    """
    others = sorted(spec.alias for spec in config.models if spec.replay_reasoning and spec.alias != "assistant_model")
    if others:
        raise ConfigError(
            f"replay_reasoning is set on {others}; it applies to assistant_model only, the one model whose "
            "requests hold its own earlier messages"
        )
    assistant = next((spec for spec in config.models if spec.alias == "assistant_model"), None)
    if assistant is None or not assistant.replay_reasoning:
        return False
    if not store_reasoning:
        raise ConfigError(
            "replay_reasoning on assistant_model needs the reasoning kept, but store_reasoning is off: "
            "drop --no-store-reasoning, or for --inputs, materialize the rows without it; or turn replay off"
        )
    provider_type = next(
        (
            provider.provider_type
            for provider in resolved_model_providers(config)
            if provider.name == assistant.provider
        ),
        None,
    )
    # Data Designer normalizes the type's case before choosing an adapter.
    if provider_type is not None and provider_type.lower() != "openai":
        raise ConfigError(
            f"replay_reasoning on assistant_model needs an OpenAI-compatible provider; {assistant.provider!r} is "
            f"{provider_type!r}, and Data Designer does not send reasoning back through it"
        )
    return True


def require_aliases(config: ModelsConfig, *, required: tuple[str, ...], context: str) -> None:
    """Raise ``ConfigError`` if any required alias is absent."""
    have = set(config.aliases())
    missing = [a for a in required if a not in have]
    if missing:
        raise ConfigError(
            f"models config is missing required aliases for {context}: {missing}; declared aliases: {sorted(have)}"
        )


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


def resolved_model_providers(config: ModelsConfig) -> list[Any]:
    """The providers Data Designer serves this config's models through.

    The config's own providers merged with the defaults, or, when it
    declares none, the defaults Data Designer loads for itself, from the
    user's providers file. A provider's ``extra_body`` and ``extra_headers``
    are merged into every request it serves.
    """
    providers = to_data_designer_kwargs(config).get("model_providers")
    if providers is not None:
        return providers
    from data_designer.config.default_model_settings import get_builtin_model_providers, get_default_providers

    try:
        return get_default_providers()
    except FileNotFoundError:
        # Data Designer writes its providers file, seeded with the built-ins,
        # the first time it runs.
        return get_builtin_model_providers()


def to_model_configs(config: ModelsConfig) -> list[Any]:
    """Convert ``ModelsConfig`` into a list of ``dd.ModelConfig`` objects.

    Lazy-imports ``data_designer`` for the same reason as above.
    """
    import data_designer.config as dd

    from usersim.engine.model_catalog import is_embedding_model, resolve_embedding_defaults

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
    """Return the path to the models TOML shipped inside the package.

    The bundled models files ship beside the CLI that documents them, in
    ``usersim/cli/``; this reads the file there without importing the CLI.
    """
    return Path(__file__).resolve().parents[1] / "cli" / "models_default.toml"


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
BUILTIN_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Built-in DD provider endpoints. Same data as in
# ``data_designer.config.utils.constants.PREDEFINED_PROVIDERS`` but
# kept here so the CLI's model smoke test runs without importing DD.
BUILTIN_PROVIDER_ENDPOINTS: dict[str, str] = {
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
        env_var = custom_provider_envs.get(spec.provider) or BUILTIN_PROVIDER_API_KEY_ENV.get(spec.provider)
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
