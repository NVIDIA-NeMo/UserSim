# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-model defaults shared by the simulator, evaluator, and CLI.

When you add support for a new model (gemma-4, claude-opus-4.7, nemotron-super,
...), edit this file -- nothing else in the repo should hard-code model-specific
knobs.

Two registries, both keyed by the DD ``model`` string (the same value that
appears in ``ModelSpec.model`` / ``[[models]] model = "..."``):

- ``INFERENCE_DEFAULTS`` -- DD ``ChatCompletionInferenceParams`` defaults that
  are genuinely per-model (model-architecture properties: which sampling
  parameters the model expects, which ``extra_body`` knobs it accepts).
  Consumed by ``cli/_models.py:_to_model_spec`` to fill in any field the TOML
  omits. Explicit TOML values always win.

- ``VLLM_DEFAULTS`` -- vLLM server-config defaults for self-hosted
  models. Also decides self-hosted vs externally-hosted provider routing
  in ``cli/_models.py``. Externally-hosted models
  (e.g. via the NVIDIA inference hub) belong in ``INFERENCE_DEFAULTS`` only --
  are called via HTTPS rather than by spinning up vLLM.

What's intentionally NOT in this catalog:

- ``timeout`` -- usually only set on the critical-path alias (assistant_model)
  to absorb reasoning latency. Lives in the TOML row.
- ``max_parallel_requests`` -- per-alias rate budget. Lives in the TOML row.
- ``provider`` -- per-deployment routing decision. Lives in the TOML row.

``max_tokens`` is per-model in the catalog at a uniform 8192 rather than
per-alias, to absorb non-ASCII tokenization variance: a tight per-job limit
costs truncation on Japanese, Hindi and Korean locales. Override per-alias by
setting ``max_tokens`` explicitly on the TOML row if a particular alias
needs a different ceiling.

Cluster-specific variants of ``extra_args`` are NOT modeled here yet. If we
move to a second cluster with different memory ceilings, swap this for a
``dict[str, dict[str, dict]]`` keyed by ``(model, cluster)`` and add a
``cluster`` knob to the notebook.

**Reproducibility tripwire:** silently bumping a default here changes
behavior for every TOML-driven run that didn't explicitly pin the field.
That's the point of having a catalog (one place to update), but be aware.
Production runs should either (a) capture the resolved ``ModelsConfig``
into run metadata so reports show what was actually used, or (b) pin the
catalog SHA in the run record. Both are out of scope for this initial
landing -- worth revisiting once adoption matures.
"""

from __future__ import annotations

from typing import Any

INFERENCE_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-20b": {
        "temperature": 1.0,
        "top_p": 1.0,
        # 8192 max_tokens uniformly across the simulator -- non-ASCII locales
        # (Japanese, Hindi, Korean, etc.) tokenize ~3x more densely than
        # English in the OpenAI BPE family, so a 1024 / 2048 / 256 per-alias
        # cap (the previous per-job tuning) can truncate responses mid-token
        # for the same number of characters. A uniform high ceiling trades
        # tighter generation budgets for safety against truncation.
        "max_tokens": 8192,
        # reasoning_effort=high uniformly across aliases that use 20b.
        # Trade slower / more expensive calls for higher answer quality;
        # apply consistently to user_model + api_response_model + summary_model.
        "extra_body": {"reasoning_effort": "high"},
    },
    "openai/gpt-oss-120b": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 8192,
        # reasoning_effort=high for both 120b aliases (assistant + judge).
        # The judge benefits most -- harder rubric calls get more thinking
        # tokens; assistant under test runs at the same effort users expect.
        "extra_body": {"reasoning_effort": "high"},
    },
    # Same gpt-oss-120b weights, but routed through the multi-vendor
    # inference hub (nvidia-inference-hub provider) instead of build.nvidia.com.
    # Useful when build.nvidia.com gates the model and you want to keep the
    # model under test the same. The catalog's auto-routing picks
    # ``nvidia-inference-hub`` automatically because this string is NOT in
    # VLLM_DEFAULTS (the hub serves it; we don't self-host).
    "nvidia/openai/gpt-oss-120b": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 8192,
        "extra_body": {"reasoning_effort": "high"},
    },
    # The double "openai/" prefix is intentional: the first segment is the
    # vendor namespace a multi-vendor hub expects, the second is the model
    # family. Other vendors on such a hub follow the same
    # <vendor>/<provider>/<model> shape, e.g.
    # "aws/anthropic/bedrock-claude-opus-4-7".
    # gpt-5.4: same hub-served reasoning family as gpt-5.5 (used as the asset-gen
    # judge + in the simulate notebook). Reasoning model -> no extra_body
    # (rejects gpt-oss reasoning_effort); generous timeout for thinking tokens.
    "openai/openai/gpt-5.4": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 8192,
        "timeout": 300,
    },
    "openai/openai/gpt-5.5": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 8192,
        # gpt-5.5 is a reasoning model -- emits internal thinking tokens
        # before visible output. 5 minutes is generous for a single
        # reply; if the hub is slower than that, treat it as an infra
        # problem and let the record fail rather than waiting longer
        # (slow hangs poison batch throughput more than fast failures).
        "timeout": 300,
        # gpt-5.5 rejects extra_body.reasoning_effort (gpt-oss-only knob).
        # Leave extra_body unset so the inference call doesn't carry it.
    },
    "aws/anthropic/bedrock-claude-opus-4-7": {
        # Claude Opus 4.7 rejects any non-default value for temperature,
        # top_p, or top_k -- the API returns 400. Per Anthropic's
        # migration guidance ("the safest migration path is to omit
        # these parameters entirely from requests"), suppress all three
        # via ``None`` (see ModelSpec docstring + to_model_configs --
        # ``None`` means "don't send the param" rather than "send the
        # default value"). Use prompting to guide behavior instead.
        "temperature": None,
        "top_p": None,
        "max_tokens": 8192,
        # Anthropic doesn't expose reasoning_effort; leave extra_body unset.
    },
    # Nemotron-3-Super-120B (LatentMoE, ~12B active params).
    # Per the model card: temperature=1.0, top_p=0.95 across ALL tasks
    # and serving backends. Reasoning is toggled via chat-template
    # ``enable_thinking`` (default ON) rather than an extra_body knob,
    # so leave extra_body unset.
    #
    # Two entries (same weights, two gateways):
    #   - "nvidia/nvidia/nemotron-3-super-v3" -- hub form, used in
    #     MODEL_OVERRIDES for hub-routed iteration.
    #   - "nvidia/nemotron-3-super" -- vLLM served name (matches
    #     ``--served-model-name`` from the model card's vLLM example),
    #     used by VLLM_DEFAULTS for self-hosting.
    #
    # A downstream config rewriter may swap hub form for served name.
    "nvidia/nvidia/nemotron-3-super-v3": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
    },
    "nvidia/nemotron-3-super": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
    },
    # Gemma-4-31B-Instruct via the NVIDIA inference hub.
    # Reasoning is gated by the chat template's ``enable_thinking`` flag,
    # which the hub forwards to vLLM via ``chat_template_kwargs`` only.
    # Other documented triggers (Google API top-level ``enable_thinking``;
    # the model-card ``<|think|>`` system token; OpenAI-style
    # ``reasoning_effort``) all fail silently against this gateway --
    # baseline calls return zero ``reasoning_content``. With this kwarg,
    # gemma emits ``message.reasoning_content`` separately from
    # ``message.content`` and ``completion_tokens`` jumps from ~500 to
    # ~880 on a multi-step word problem (1235 chars of reasoning vs
    # ~zero on the baseline). See ``tools/probe_gemma_thinking.py`` for
    # the per-trigger probe trace; the empirical evidence is the only
    # reason we picked the chat-template-kwargs form -- the model card
    # documents three triggers and only this one fires here.
    #
    # Sampling defaults follow gemma-4-instruct's documented values
    # (temperature=1.0, top_p=0.95). max_tokens=8192 leaves room for
    # reasoning + visible answer; thinking-mode completions can be 2x
    # the non-thinking length on rubric-heavy turns.
    #
    # NOTE: provider-captured reasoning still doesn't reach our LLM
    # wrapper -- DataDesigner's ``Usage`` dataclass strips
    # ``completion_tokens_details`` at the abstraction boundary (probe
    # confirmed ``usage.completion_tokens_details`` is ``None`` even on
    # variant 3). Reasoning tokens surface via the report-time tiktoken
    # estimator on the trajectory's visible content; with thinking
    # enabled, output_tokens grows but visible content stays roughly
    # the same length, so the noise-floor clamp in
    # ``reporting/_reasoning.py`` should now correctly lift gemma's
    # assistant reasoning above the 5% threshold.
    "nvidia/google/gemma-4-31b-it": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
}

VLLM_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-20b": {
        "tensor_parallelism": 1,
        "node_count": 1,
        "extra_args": [
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", "32768",
            "--max-num-seqs", "256",
        ],
    },
    "openai/gpt-oss-120b": {
        "tensor_parallelism": 1,
        "node_count": 1,
        "extra_args": [
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", "32768",
            "--max-num-seqs", "256",
        ],
    },
    # gpt-5.5 (and any other hub-served model) NOT in this map -- it's
    # externally hosted on the NVIDIA inference hub, never self-hosted.
    # Excluding hub-served models here is intentional: membership in this
    # registry is what marks a model as self-hostable.
    "nvidia/nemotron-3-super": {
        # vLLM ``--model`` (HF path) differs from the served name DD
        # references, so ``model_name`` is carried separately while
        # keeping the catalog key ``nvidia/nemotron-3-super`` as the
        # ``served_model_name`` (what API clients send).
        #
        # NVFP4 quantization fits on a single B200 (or DGX Spark);
        # cheaper / simpler ops than the FP8 variant (which required
        # 4 x H100/H200 + expert-parallel). For older H100/H200 fleets
        # without B200s, switch to the FP8 path instead:
        # ``nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8``,
        # tensor_parallelism=4, plus --enable-expert-parallel and
        # --mamba-ssm-cache-dtype float32 (see the FP8 model card).
        # Requires vLLM >= 0.20.0.
        "model_name": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "tensor_parallelism": 1,  # NVFP4 single-GPU on B200/DGX Spark
        "node_count": 1,
        "extra_args": [
            # Tuned for our cluster sweeps -- shorter context than the
            # model card default (16k vs 256k) keeps batch throughput
            # higher when conversations are short, and the multithreaded
            # loader cuts cold-start time. Override per-sweep if you need
            # the longer context window.
            "--async-scheduling",
            "--dtype", "auto",
            "--kv-cache-dtype", "fp8",
            "--trust-remote-code",
            "--attention-backend", "TRITON_ATTN",
            "--gpu-memory-utilization", "0.9",
            "--enable-chunked-prefill",
            "--max-num-seqs", "4096",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            # This model needs a reasoning parser that vLLM does not bundle.
            # Fetch it and point the flag at wherever you put it:
            #   wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py
            "--reasoning-parser-plugin",
            "/path/to/super_v3_reasoning_parser.py",
            "--reasoning-parser", "super_v3",
            "--max-model-len", "16384",
            "--model-loader-extra-config",
            '{"enable_multithread_load":true,"num_threads":16}',
            "--enable-prefix-caching",
        ],
    },
}


def resolve_inference_defaults(model: str) -> dict[str, Any]:
    """Return a copy of the inference defaults for ``model``.

    Returns an empty dict if ``model`` isn't in the catalog. Callers
    (the TOML loader) fall back to the existing hard-coded defaults
    in that case.
    """
    return dict(INFERENCE_DEFAULTS.get(model, {}))


def resolve_vllm_defaults(model: str) -> dict[str, Any]:
    """Return a copy of the vLLM defaults for ``model``.

    Returns an empty dict if the model is not self-hosted (e.g.
    externally served by the NVIDIA inference hub).
    """
    return dict(VLLM_DEFAULTS.get(model, {}))


# Embedding models are served on the ``/embeddings`` route, NOT
# ``/chat/completions`` -- Data Designer must register them with
# ``EmbeddingInferenceParams`` (see ``to_model_configs``) or its health check
# (a chat ``generate``) 404s with "model could not be found". List embedding
# model ids here; the ``embed`` substring is also treated as a fallback signal.
EMBEDDING_MODELS: frozenset[str] = frozenset({
    "nvidia/nvidia/nemotron-3-embed-1b",
    "nvidia/qwen/qwen3-embedding-0.6b",
})

# Per-embedding-model request defaults applied by ``to_model_configs`` (in addition
# to EmbeddingInferenceParams' own defaults: encoding_format=float). ``extra_body``
# carries provider-specific top-level fields the /embeddings route requires.
#
# NOTE on asymmetric models (qwen3-embedding): the CORPUS is indexed as
# ``input_type=passage``; the sim-time retriever embeds the user query ``query``-side.
# ``truncate=NONE`` errors (rather than silently truncating) if a doc exceeds the
# model's context, so over-long docs surface instead of being quietly clipped.
EMBEDDING_DEFAULTS: dict[str, dict[str, Any]] = {
    "nvidia/qwen/qwen3-embedding-0.6b": {
        "extra_body": {"input_type": "passage", "truncate": "NONE"},
    },
}


def is_embedding_model(model: str) -> bool:
    """True if ``model`` is an embedding model (embeddings route, not chat)."""
    m = (model or "").lower()
    return model in EMBEDDING_MODELS or "embed" in m


def resolve_embedding_defaults(model: str) -> dict[str, Any]:
    """Return a copy of the embedding request defaults for ``model`` (or empty)."""
    return dict(EMBEDDING_DEFAULTS.get(model, {}))


def known_inference_models() -> set[str]:
    """Return the set of models with inference defaults defined."""
    return set(INFERENCE_DEFAULTS)


def known_embedding_models() -> set[str]:
    """Return the set of embedding models with request defaults defined."""
    return set(EMBEDDING_DEFAULTS)


def known_vllm_models() -> set[str]:
    """Return the set of self-hosted models with vLLM defaults defined."""
    return set(VLLM_DEFAULTS)
