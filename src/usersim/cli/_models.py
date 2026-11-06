# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the CLI does with a models file beyond loading it.

Overrides from ``--model`` and ``--parallel`` flags, the model smoke test,
the printed summary of the resolved models, and the prompt for missing
provider keys. Loading and validating the file, the alias requirements and
the conversion to Data Designer model configs are in
:mod:`usersim.engine.models_file`, which code below the CLI imports.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Iterable

from usersim.engine.errors import ConfigError
from usersim.engine.models_file import (
    BUILTIN_PROVIDER_API_KEY_ENV,
    BUILTIN_PROVIDER_ENDPOINTS,
    ModelsConfig,
    ModelSpec,
    describe_models_source,
    replay_flag,
)

if sys.version_info >= (3, 11):
    pass
else:
    pass

logger = logging.getLogger(__name__)


_OVERRIDE_KNOWN_KEYS = frozenset(
    {
        "model",
        "temperature",
        "top_p",
        "max_tokens",
        "extra_body",
        "replay_reasoning",
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

    - **str** (simple form): ``{"judge_model": "openai/gpt-oss-120b"}``
      Just swap the model. Catalog defaults for the new model fill in
      ``temperature`` / ``top_p`` / ``max_tokens`` / ``extra_body``.

    - **dict** (extended form): per-knob override on top of the catalog.
      The dict MUST include ``"model"`` and may optionally include any
      of ``"temperature"``, ``"top_p"``, ``"max_tokens"``, ``"extra_body"``,
      ``"replay_reasoning"``. Replay is kept when the model stays the same and
      turned off when it changes, unless the override sets it::

          overrides = {
              "assistant_model": {
                  "model": "google/gemma-4-31b-it",
                  "extra_body": {},   # disable catalog's enable_thinking
              },
              "judge_model": {
                  "model": "openai/gpt-oss-120b",
                  "temperature": 0.0, # deterministic eval
                  "max_tokens": 2048, # tighter than catalog default
              },
          }

      Per-knob precedence: **dict-form value > catalog default > old spec**.
      A key present in the dict (even with value ``None`` or ``{}``)
      ALWAYS wins, distinguishing "explicit clear" from "absent / fall
      through to catalog". Unknown keys raise ``ConfigError``.

    For each matching alias, the new model's catalog defaults
    (``engine.model_catalog.INFERENCE_DEFAULTS``) replace any per-model
    fields not explicitly overridden in the dict form; ``provider`` is
    inferred from the catalog (see below); the per-alias fields
    (``max_parallel_requests`` / ``timeout``) carry over unchanged.

    Provider inference rule:

    - If the new model is in ``engine.model_catalog.VLLM_DEFAULTS`` (i.e.
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
    ``{"reasoning_effort": "high"}``) for ``gpt-5.6-terra`` (catalog
    omits ``temperature`` and ``top_p``, which the endpoint rejects
    alongside ``reasoning_effort``). Without re-resolving, the swap would
    carry over gpt-oss's sampling values and the call would 400. Same idea
    for provider: swapping a hosted model in for a self-hosted one
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
    from usersim.engine.model_catalog import (
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
                    "isn't in engine/model_catalog.py:INFERENCE_DEFAULTS -- "
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
                    "isn't in engine/model_catalog.py:INFERENCE_DEFAULTS -- "
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

        # Replay belongs to the model it was set for, not to the alias: a new
        # model starts with it off unless the override turns it on.
        if "replay_reasoning" in ov:
            new_replay = replay_flag(ov["replay_reasoning"], where=f"model override for alias {spec.alias!r}")
        else:
            new_replay = spec.replay_reasoning and new_model == spec.model
            if spec.replay_reasoning and not new_replay:
                logger.warning(
                    "model override: alias=%r switched to model=%r, so replay_reasoning is off; set it on the "
                    "override to keep it",
                    spec.alias,
                    new_model,
                )
        new_specs.append(
            dataclasses.replace(
                spec,
                model=new_model,
                provider=new_provider,
                temperature=new_temperature,
                top_p=new_top_p,
                max_tokens=new_max_tokens,
                extra_body=new_extra_body,
                replay_reasoning=new_replay,
            )
        )
    return dataclasses.replace(config, models=tuple(new_specs))


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

    from usersim.engine.model_catalog import is_embedding_model, resolve_embedding_defaults

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
        elif spec.provider in BUILTIN_PROVIDER_ENDPOINTS:
            endpoint = BUILTIN_PROVIDER_ENDPOINTS[spec.provider]
            env_var = BUILTIN_PROVIDER_API_KEY_ENV.get(spec.provider, "")
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
    print("Resolved inference defaults (TOML row + engine/model_catalog.py):")
    for spec in config.models:
        extras = ""
        if spec.extra_body:
            extras += f"  extra_body={spec.extra_body}"
        if spec.timeout:
            extras += f"  timeout={spec.timeout}"
        if spec.replay_reasoning:
            extras += "  replay_reasoning=True"
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
