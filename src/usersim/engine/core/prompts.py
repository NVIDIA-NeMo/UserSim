# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Prompts the conversation loop owns, shared by every probe.

Per-probe prompts live in ``probes/<probe>/prompts.py`` and are overridable
from ``assets/<probe>/prompts.yaml``. The ones here belong to the loop
rather than to any one probe — they are asked on every trajectory whatever
the probe is doing — so they have no natural home under a probe folder.

A probe can still override one for itself: ``BaseProbe`` looks the prompt
up under the probe's own label first (so
``assets/<probe>/prompts.yaml`` wins) and falls back to the default
here. That keeps the loop's default in one place while leaving a probe
free to sharpen it for its own shape.

Other loop-owned prompts are still constants in their own modules
(``_EARLY_STOP_PROMPT`` and the summariser in ``core/context.py``,
``JUDGE_FOLLOWUP_PROMPT`` in ``core/judges.py``, the role-anchor and
frustration text in ``core/behavioral.py``). Moving those here is the same
change again and worth doing; this module exists so there is somewhere for
them to go.
"""

from __future__ import annotations

#: Per-turn assistant quality judge. The loop runs this on every assistant
#: turn and records the verdict in ``assistant_judge_ratings`` and an
#: ``ASSISTANT_QUALITY`` trace.
#:
#: Whether it also GATES depends on ``cfg.max_assistant_attempts``:
#:
#: - ``1`` (the default) — reporting only. The verdict is recorded and the
#:   candidate stands either way, which is what keeps a run measuring the
#:   assistant under test rather than a best-of-N of it. Scoring the
#:   assistant is the out-of-sim evaluator's job.
#: - ``> 1`` — gating. ``_generate_and_judge_assistant_turn`` calls this
#:   inside its attempt loop; a negative verdict rolls the candidate off
#:   the transcript and resamples. That IS best-of-N by construction, and
#:   is intended for SDG-style runs where the assistant is a data
#:   generator rather than a model under test.
#:
#: Consequence worth knowing before editing this text, or overriding it in
#: an asset file: above a budget of 1, this prompt decides which assistant
#: candidates survive. An ``assets/<probe>/prompts.yaml`` change can shift
#: what a run keeps without touching any Python.
#:
#: Deliberately generic. It asks only whether the reply addresses the user
#: and is coherent, because it runs for every probe; anything
#: probe-specific belongs in that probe's own ``prompts.yaml`` under
#: ``assistant_judge_prompt``.
ASSISTANT_JUDGE_PROMPT = """\
You are an expert evaluator. Judge the assistant's response in this conversation.

Previous conversation:
<PREV_CONVERSATION>
{conversation_history}
</PREV_CONVERSATION>

Assistant response to judge:
<ASSISTANT_RESPONSE>
{assistant_response}
</ASSISTANT_RESPONSE>

<RUBRIC>
Criteria for success:
- The response addresses what the user asked.
- The response is a coherent, complete reply relevant to the conversation context.

Criteria for failure:
- The response does not address the user's question or request.
- The response is unintelligible, empty, or a non-sequitur that ignores the conversation context.
</RUBRIC>

Strictly respond in the following format:
<explanation>
[Your detailed explanation and justification for the rating.]
</explanation>
<rating>
[Must be either 'success' or 'failure' strictly]
</rating>"""
