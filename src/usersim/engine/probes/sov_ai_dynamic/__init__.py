# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Population-probing probe package.

Sibling of :mod:`usersim.engine.probes.sov_ai_facts`,
not a replacement. The two probes cover complementary halves of the
sovereign-AI readiness question:

- ``sov_ai_facts`` — *curated depth*: a reviewer-
  vetted fact bank with ground truth, false premises, and graceful-
  unknown reference answers. Tests precisely whether the assistant
  recovers the right facts on a fixed slate.
- ``sov_ai_dynamic`` — *dynamic breadth*: a per-locale
  probing taxonomy of broad topic categories (geography, history,
  culture, poetry-and-music, public-services). The user-agent generates
  the opening turn in the persona's voice, anchored to one category
  invitation and a rotated sub-topic hint. There is **no ground truth**;
  out-of-sim heuristic scorers (suspected fabrication, self-consistency,
  graceful unknown) flag failure modes from the response alone.

The two probes share the same persona-tag vocabulary and route
through the same in-sim infrastructure (``ConversationState``,
``OutcomeBuilder``, ``acall_llm``), and their outputs converge in the
combined sovereign-AI scorecard.

Public entry point: :func:`generator.simulate_sov_ai_dynamic`.
"""
