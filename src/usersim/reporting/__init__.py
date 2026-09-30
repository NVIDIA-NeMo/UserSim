# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Software-and-Model-Evaluation

"""Reporting — turn trajectory + eval parquets into the capability dashboard.

Two layers ship from this package:

- :class:`CapabilityReport` (built by :func:`build_capability_report`) —
  the run-level contract: capability x locale cells with evidence state,
  coverage by locale × probe, a prioritized triage queue, and an
  embedded Sim-Health rollup. The same JSON payload feeds both the
  static no-build renderer (:func:`render_capability_report_html`) and
  the React/Vega dashboard written by
  :func:`write_capability_dashboard_artifacts`.

- :class:`SimHealthBundle` (built by :func:`build_sim_health`) — *did
  the simulator do its job?* Status counts, failure-mode taxonomy with
  per-actor attribution, D1–D4 + MATTR behavioral diagnostics,
  persona-grounding rate, fourth-wall trigger rate, gate-retry
  distributions, and a per-bundle resource profile. Embedded into the
  capability dashboard as the "Sim Health" panel; also re-exported here
  for ad-hoc analysis use.

Wide → long for inter-judge agreement and intersectional aggregation
is done in the builder via :mod:`pandas` rather than persisted in long
form at rest.
"""

from usersim.reporting.capability_report import (
    CapabilityCell,
    CapabilityReport,
    CoverageCell,
    TriageItem,
    build_capability_report,
    render_capability_report_html,
)
from usersim.reporting.comparison import (
    ComparisonCell,
    ComparisonEntry,
    ComparisonReport,
    ModelRollup,
    build_comparison_report,
    discover_comparison_runs,
    select_runs_for_comparison,
)
from usersim.reporting.comparison_dashboard import (
    assign_palette,
    write_comparison_dashboard_artifacts,
)
from usersim.reporting.dashboard import write_capability_dashboard_artifacts
from usersim.reporting.diagnostics import (
    compute_diagnostics_frame,
    compute_front_loading_ratio,
    compute_frustration_marker_fraction,
    compute_mattr,
    compute_polite_turn_fraction,
    compute_response_length_by_locale,
    compute_verbosity_cv,
)
from usersim.reporting.eval_manifest import (
    MANIFEST_FILENAME as EVAL_SAMPLE_MANIFEST_FILENAME,
)
from usersim.reporting.eval_manifest import (
    EvalSampleManifest,
    EvalSamplePass,
    read_eval_sample_manifest,
    record_eval_sample_pass,
)
from usersim.reporting.identity import identity_contradictions
from usersim.reporting.resource import (
    ResourceProfile,
    aggregate_resource_profile,
)
from usersim.reporting.runs import print_available_runs, print_comparison_runs
from usersim.reporting.sim_health import SimHealthBundle, build_sim_health

__all__ = [
    "CapabilityCell",
    "CapabilityReport",
    "ComparisonCell",
    "ComparisonEntry",
    "ComparisonReport",
    "CoverageCell",
    "EVAL_SAMPLE_MANIFEST_FILENAME",
    "EvalSampleManifest",
    "EvalSamplePass",
    "ModelRollup",
    "ResourceProfile",
    "SimHealthBundle",
    "TriageItem",
    "aggregate_resource_profile",
    "assign_palette",
    "build_capability_report",
    "build_comparison_report",
    "build_sim_health",
    "compute_diagnostics_frame",
    "compute_front_loading_ratio",
    "compute_frustration_marker_fraction",
    "compute_mattr",
    "compute_polite_turn_fraction",
    "compute_response_length_by_locale",
    "compute_verbosity_cv",
    "discover_comparison_runs",
    "identity_contradictions",
    "print_available_runs",
    "print_comparison_runs",
    "read_eval_sample_manifest",
    "record_eval_sample_pass",
    "render_capability_report_html",
    "select_runs_for_comparison",
    "write_capability_dashboard_artifacts",
    "write_comparison_dashboard_artifacts",
]
