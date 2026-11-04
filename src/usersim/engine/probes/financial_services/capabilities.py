# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The capability rows ``financial_services``'s rows provide evidence for."""

from __future__ import annotations

from usersim.extension import CapabilityDefinition, EvidenceSource

# financial_services institution types that get their OWN verifiable-task-success
# capability row (a per-institution-type breakdown of financial_task_success).
# Add a type here when a new institution type ships; a type absent from a given
# locale's bank just renders as a no-data cell for that locale. Kept explicit
# (not derived from the spec's full vocabulary) so the dashboard only carries
# rows for types we actually author banks for.
FINANCE_CAPABILITY_INSTITUTION_TYPES: tuple[str, ...] = (
    "bank",
    "brokerage",
    "retirement_provider",
    "wealth_manager",
    # Authored by the India (hi_Deva_IN) bank, whose institution set is chosen by
    # Indian retail prevalence rather than mirrored from en_US: a UPI-first
    # neobank, a life/health insurer, and an NBFC (deposit-less consumer/gold-loan
    # lender). en_US ships no bank of these types, so those cells render as
    # no-data for en_US — and vice versa for retirement_provider/wealth_manager.
    "neobank",
    "insurer",
    "nbfc",
)


def _finance_verifiable_by_institution_type() -> tuple[CapabilityDefinition, ...]:
    """One verifiable-task-success capability per institution type.

    Reuses the aggregate ``financial_task_success`` axes/policy, scoped to a
    single ``institution_type`` via ``row_filter``. This gives the dashboard a
    per-type breakdown that scales as institution types are added and stays
    locale-aware (types not present in a locale render as no-data).
    """
    defs: list[CapabilityDefinition] = []
    for itype in FINANCE_CAPABILITY_INSTITUTION_TYPES:
        pretty = itype.replace("_", " ")
        defs.append(
            CapabilityDefinition(
                id=f"financial_task_success_{itype}",
                label=f"Financial task success — {pretty}",
                description=(
                    f"Verifiable financial-services task success for {pretty} "
                    "institutions (deterministic verifier), stratified from the "
                    "aggregate financial_task_success capability."
                ),
                sources=(
                    EvidenceSource(
                        probe="financial_services",
                        scorer="financial_services",
                        # Success = verifiable state/action outcome only.
                        # Retrieval recall is a SEPARATE diagnostic
                        # (financial_retrieval_recall), deliberately not folded
                        # in here so access and utilization stay disentangled.
                        axes=(
                            "finance.tool_selection_rate",
                            "finance.ordering_respected_rate",
                            "finance.no_unauthorized_action_rate",
                        ),
                    ),
                ),
                threshold=0.9,
                scale="rate",
                aggregation_policy="critical_axis",
                critical_axes=(
                    "finance.tool_selection_rate",
                    "finance.no_unauthorized_action_rate",
                ),
                row_filter={"institution_type": itype},
                next_action=(
                    f"Inspect failing {pretty} trajectories; check retrieval "
                    "recall and tool selection for this institution type."
                ),
            )
        )
    return tuple(defs)


CAPABILITIES: tuple[CapabilityDefinition, ...] = (
    CapabilityDefinition(
        id="financial_task_success",
        label="Financial task success (verifiable)",
        description=(
            "On Tier-1 financial-services tasks, does the model retrieve the right "
            "policy/tool docs and execute the correct tool sequence without "
            "unauthorized state changes? (deterministic verifier)"
        ),
        sources=(
            EvidenceSource(
                probe="financial_services",
                scorer="financial_services",
                # Success = verifiable state/action outcome only; retrieval
                # recall is reported separately (financial_retrieval_recall).
                axes=(
                    "finance.tool_selection_rate",
                    "finance.ordering_respected_rate",
                    "finance.no_unauthorized_action_rate",
                ),
            ),
        ),
        threshold=0.9,
        scale="rate",
        aggregation_policy="critical_axis",
        critical_axes=(
            "finance.tool_selection_rate",
            "finance.no_unauthorized_action_rate",
        ),
        next_action=(
            "Inspect failing dispute/transaction trajectories; if "
            "financial_retrieval_recall is also low, fix retrieval first, "
            "else focus on tool selection / policy reasoning."
        ),
    ),
    CapabilityDefinition(
        id="financial_retrieval_recall",
        label="Financial KB retrieval recall",
        description=(
            "On verifiable financial-services tasks, did the agent actually "
            "retrieve the gold knowledge-base documents its tool-use should "
            "depend on? DIAGNOSTIC: this is the retrieval-ACCESS half of the "
            "access-vs-utilization split (cf. tau-Knowledge). Read it ALONGSIDE "
            "financial_task_success — low recall points at retrieval; high "
            "recall with low task success points at policy/reasoning."
        ),
        sources=(
            EvidenceSource(
                probe="financial_services",
                scorer="financial_services",
                axes=("finance.document_recall_rate",),
            ),
        ),
        threshold=0.8,
        scale="rate",
        aggregation_policy="critical_axis",
        critical_axes=("finance.document_recall_rate",),
        next_action=(
            "If recall is low, improve retrieval (embeddings / query "
            "formulation / chunking) before attributing failures to reasoning."
        ),
    ),
    CapabilityDefinition(
        id="financial_retrieval_quality",
        label="Financial KB retrieval quality",
        description=(
            "Of everything the agent retrieved, how much of it carried the FACTS "
            "the customer asked about (product sheets, rates, fees, FAQs) rather "
            "than tool/procedure text? DIAGNOSTIC, and the necessary companion to "
            "financial_retrieval_recall: gold documents are derived from the "
            "procedural genres, so recall alone can read healthy while the agent "
            "retrieved nothing but tool references and answered from memory. "
            "Unlike recall this needs no gold, so it covers BOTH tiers — "
            "including the dynamic tier, where it is the only retrieval signal."
        ),
        sources=(
            EvidenceSource(
                probe="financial_services",
                scorer="financial_services",
                axes=("finance.substantive_retrieval_rate",),
            ),
        ),
        # Procedural documents are legitimately retrieved (the discoverable-tool
        # gate requires them), and a verifiable task whose job is to find a tool
        # doc will properly skew that way -- so the bar is "the facts were reached
        # too", not "no procedure". Calibrated on the collapsed en_IN run scored
        # both as-run and replayed through the shipped retriever: at 0.3, 98% of
        # healthy rows pass (97% of verifiable ones) while only 13% of collapsed
        # rows do. Tightening to 0.5 would fail 30% of healthy verifiable rows.
        threshold=0.3,
        scale="rate",
        aggregation_policy="critical_axis",
        critical_axes=("finance.substantive_retrieval_rate",),
        next_action=(
            "If this is low while recall looks fine, the retriever is returning "
            "only tool/policy docs — check the retrieval mode (pure dense "
            "collapses on a narrow corpus) before trusting any grounding score."
        ),
    ),
    CapabilityDefinition(
        id="dynamic_grounding",
        label="Grounded open-ended advice",
        description=(
            "On open-ended (dynamic-tier) questions, does the model stay grounded "
            "in the institution's knowledge base, faithful to its authoritative "
            "numbers, within its advice boundary, and free of fabrication? "
            "CROSS-DOMAIN: financial_services is one evidence source today; future "
            "domain probes (healthcare/insurance) add more on the SAME dynamic.* "
            "axes."
        ),
        sources=(
            EvidenceSource(
                probe="financial_services",
                scorer="financial_services",
                axes=(
                    "dynamic.grounding",
                    "dynamic.numeric_faithfulness",
                    "dynamic.boundary_adherence",
                    "dynamic.no_fabrication",
                    "dynamic.graceful_unknown",
                    "dynamic.self_consistency",
                ),
            ),
        ),
        threshold=4.0,
        scale="score_1_5",
        aggregation_policy="critical_axis",
        # A numerically-wrong or fabricated grounded answer is the worst failure.
        critical_axes=(
            "dynamic.numeric_faithfulness",
            "dynamic.no_fabrication",
        ),
        next_action=(
            "Inspect low-grounding / fabricated advisory trajectories; check retrieval quality and the boundary rubric."
        ),
    ),
) + _finance_verifiable_by_institution_type()

#: What each rate axis its scorer computes measures, in a reviewer's terms.
AXIS_DESCRIPTIONS: dict[str, str] = {
    "finance.tool_selection_rate": (
        "Share of the task's required tools the assistant actually called. 0% means "
        "it called none of them -- usually it read the knowledge base and then "
        "stopped, or asked for details instead of acting."
    ),
    "finance.ordering_respected_rate": (
        "Whether the required tools were called in the order the task depends on. "
        "Only enforced for tasks whose steps genuinely depend on each other."
    ),
    "finance.document_recall_rate": (
        "Share of the task's gold knowledge-base documents the assistant retrieved. "
        "This measures retrieval ACCESS, separately from whether the assistant then "
        "used what it read."
    ),
    "finance.no_unauthorized_action_rate": (
        "Whether the assistant avoided state-changing calls the task never asked "
        "for. Below 1.0 means it changed something on its own initiative."
    ),
    "finance.substantive_retrieval_rate": (
        "Share of retrieved documents carrying FACTS (products, rates, fees, FAQs) "
        "rather than tool/procedure text. A composition measure, not an accuracy "
        "one: a task whose job is to find a tool doc properly skews procedural, so "
        "low is expected there. It exists to catch a retriever that returns only "
        "procedure and never reaches the facts."
    ),
}
