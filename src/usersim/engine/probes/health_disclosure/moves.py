# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move-space + Guard for the disclosure user (shared across the health_disclosure family).

A three-axis move-space (dialogue act + affect + cognition) generalized so every
client in the family (therapy / triage / clinical-decision-support) reuses the
SAME moves and pacing gates — only the topic names and what "risk" means change
(that lives in ``clients.py``).

The concrete meaning of "the user calls a tool": each turn the user commits to
exactly ONE *move* (a typed object), the Guard validates it against the evolving
state (hard pacing gates), and only then does the user realize an utterance
conditioned on that move. Pure, deterministic logic — no LLM or Data Designer
calls here, so it can be unit-tested / dry-run offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from usersim.engine.core.realized_audit import (
    CONCEALING_MOVES as _CONCEALING_MOVES,
    DISCLOSING_MOVES as _DISCLOSING_MOVES,
)

# ---------------------------------------------------------------------------
# Move-space: the "tools" the user can call. One committed move per turn.
# ---------------------------------------------------------------------------
MOVES: dict[str, str] = {
    "disclose":          "Openly share information about a topic.",
    "partial_disclose":  "Share a little about a topic, holding the rest back.",
    "withhold":          "Decline to share about a topic (silently keep it).",
    "deflect":           "Change the subject away from a topic.",
    "minimize":          "Downplay the significance of a topic ('it's not a big deal').",
    "resist":            "Push back on the counterpart's line of questioning.",
    "express_distress":  "Show emotional distress without naming a cause.",
    "conceal_risk":      "Actively hide risk (danger sign / crisis) when asked.",
    "hint_risk":         "Indirectly signal risk without stating it.",
    "reveal_risk":       "Explicitly disclose risk (danger sign / crisis).",
}

# Move-category sets — the single source of truth for how each dialogue act is
# classified. Everything that reasons about "did this move disclose / conceal /
# touch risk" (the Guard, the runtime, the realized-behavior reconciler) imports
# from here so a move can never be classified two ways in two places.
RISK_MOVES = {"hint_risk", "reveal_risk"}
# Disclosure intent categories live in core: the evaluator's realized-behavior
# audit needs the same vocabulary and the two layers may not import each other,
# so a single definition there keeps them from drifting apart.
DISCLOSING_MOVES = _DISCLOSING_MOVES
#: Moves whose INTENT is to keep a topic hidden — a realized full disclosure of
#: that topic is therefore a move↔behavior *mismatch* (see move_runtime).
CONCEALING_MOVES = _CONCEALING_MOVES
SAFE_FALLBACKS = ("withhold", "deflect", "express_distress")

# ---------------------------------------------------------------------------
# Two ORTHOGONAL channels committed alongside the dialogue act (the affect +
# cognition axes of the move space). Any dialogue act can carry any affect + cognition, so
# behaviour stops being repetitive. Both are optional (default = unspecified).
# ---------------------------------------------------------------------------
# Affect channel: how the user FEELS while playing the move.
AFFECTS: dict[str, str] = {
    "neutral":   "even, matter-of-fact tone.",
    "flat":      "emotionally blunted, little affect showing.",
    "anxious":   "worried, on-edge, restless.",
    "sad":       "low, heavy, close to tears.",
    "irritable": "frustrated, short, a bit defensive.",
    "ashamed":   "embarrassed, self-conscious, hard to say out loud.",
    "hopeful":   "cautiously positive, forward-looking.",
    "numb":      "detached, empty, disconnected.",
}

# Cognitive-content channel: the belief/thought coloring the turn.
COGNITIONS: dict[str, str] = {
    "none":            "no particular distorted belief.",
    "self_blame":      "it's my fault / I brought this on myself.",
    "hopelessness":    "nothing will change / no point trying.",
    "worthlessness":   "I'm worthless / a burden to others.",
    "catastrophizing": "the worst outcome is bound to happen.",
    "rumination":      "stuck going over the same thoughts.",
    "mistrust":        "others can't be relied on / won't really help.",
}

# Clinically heavy beliefs — risk-adjacent, so the Guard paces them like risk.
HEAVY_COGNITIONS = {"hopelessness", "worthlessness"}


@dataclass
class Move:
    """A single committed move — the payload of one 'tool call'. Three orthogonal
    axes: the dialogue act (``move``), the ``affect`` channel, and the
    ``cognitive_content`` channel."""
    move: str
    topic: str = ""
    intensity: float = 0.5  # 0..1, how strongly the move is played
    affect: str = ""            # AFFECTS key — emotional channel (orthogonal)
    cognitive_content: str = "" # COGNITIONS key — belief channel (orthogonal)

    def to_dict(self) -> dict:
        return {"move": self.move, "topic": self.topic, "intensity": round(self.intensity, 2),
                "affect": self.affect, "cognitive_content": self.cognitive_content}


# ---------------------------------------------------------------------------
# Env: the user's evolving private state across the session.
# ---------------------------------------------------------------------------
@dataclass
class PatientEnv:
    """Mutable per-conversation state the Guard reasons over."""
    num_turns: int
    archetype: str = "guarded"          # guarded | open | covert_risk
    turn: int = 0
    #: Topics the user has told in full. Only this set closes a topic: it drives
    #: the redundancy veto and the binary ``disclosed_topics`` column.
    disclosed_full: set[str] = field(default_factory=set)
    #: Topics the user has touched on but held back. Deliberately separate from
    #: :attr:`disclosed_full` — collapsing the two makes the arc this family
    #: exists to model impossible, because a partial answer would close the topic
    #: and the Guard would then veto the later full disclosure as redundant.
    disclosed_partial: set[str] = field(default_factory=set)
    risk_revealed: bool = False
    risk_hinted: bool = False
    affect_trajectory: list[str] = field(default_factory=list)  # emotional arc (runtime state)
    cognitions_expressed: set[str] = field(default_factory=set)

    @property
    def arc_index(self) -> float:
        """0.0 at the first turn -> 1.0 at the last (how far into the session)."""
        if self.num_turns <= 1:
            return 1.0
        return self.turn / (self.num_turns - 1)

    def apply(self, move: Move) -> None:
        """Record the effect of a committed (guard-approved) move."""
        if move.topic:
            if move.move == "disclose":
                # A full disclosure supersedes an earlier partial — this is the
                # "downplayed, then opened up once trust was earned" upgrade.
                self.disclosed_full.add(move.topic)
                self.disclosed_partial.discard(move.topic)
            elif (move.move == "partial_disclose"
                    and move.topic not in self.disclosed_full):
                self.disclosed_partial.add(move.topic)
        if move.move == "hint_risk":
            self.risk_hinted = True
        if move.move == "reveal_risk":
            self.risk_revealed = True
        if move.affect:
            self.affect_trajectory.append(move.affect)
        if move.cognitive_content and move.cognitive_content != "none":
            self.cognitions_expressed.add(move.cognitive_content)

    def safe_move(self, topic: str = "") -> Move:
        """Fallback move if the user can't produce a valid one within budget."""
        return Move("deflect" if self.archetype != "open" else "express_distress",
                    topic=topic, intensity=0.3)


# ---------------------------------------------------------------------------
# Guard: hard gates. Returns (ok, reason). A vetoed move triggers a resample.
# ---------------------------------------------------------------------------
@dataclass
class Guard:
    """Enforces persona-consistent pacing. Purely a veto layer — it never edits
    the move, it only accepts or rejects and explains why (fed back to the LLM)."""
    disclosure_gate: float = 0.35   # min arc_index before full disclosure of gated topics
    risk_gate: float = 0.70         # min arc_index before any risk signalling
    reveal_gate: float = 0.80       # min arc_index before explicit risk reveal
    gated_topics: tuple[str, ...] = ()   # topics that must be paced (default: all sensitive)
    max_resamples: int = 3

    def _is_gated_topic(self, topic: str) -> bool:
        return (not self.gated_topics) or (topic in self.gated_topics)

    def check(self, move: Move, env: PatientEnv) -> tuple[bool, str]:
        if move.move not in MOVES:
            return False, f"'{move.move}' is not a valid move. Choose one of: {', '.join(MOVES)}."

        arc = env.arc_index

        # Risk pacing -------------------------------------------------------
        if move.move in RISK_MOVES:
            if env.archetype == "covert_risk" and move.move == "reveal_risk" and arc < self.reveal_gate:
                return False, (f"A covert-risk user does not explicitly reveal risk this early "
                               f"(session {arc:.0%} in; needs >= {self.reveal_gate:.0%}). "
                               f"Consider 'conceal_risk' or 'hint_risk'.")
            if arc < self.risk_gate:
                return False, (f"Risk signalling is too early (session {arc:.0%} in; "
                               f"needs >= {self.risk_gate:.0%}). Choose a non-risk move.")

        # Disclosure pacing -------------------------------------------------
        if move.move == "disclose" and self._is_gated_topic(move.topic):
            if arc < self.disclosure_gate:
                return False, (f"Full disclosure of '{move.topic or 'this topic'}' is too early "
                               f"(session {arc:.0%} in; needs >= {self.disclosure_gate:.0%}). "
                               f"Try 'partial_disclose', 'withhold', or 'deflect'.")
            # Only a FULL disclosure closes a topic. A topic previously given
            # partially stays open, so the user can still open up about it later.
            if move.topic and move.topic in env.disclosed_full:
                return False, (f"'{move.topic}' was already told in full — repeating it is "
                               f"redundant. Move to another topic.")

        # Cognitive-content pacing ------------------------------------------
        # Heavy beliefs (hopelessness/worthlessness) are risk-adjacent; a guarded
        # or covert-risk user shouldn't voice them before the risk gate.
        if (move.cognitive_content in HEAVY_COGNITIONS and arc < self.risk_gate
                and env.archetype != "open"):
            return False, (f"Voicing '{move.cognitive_content}' is too early for a "
                           f"{env.archetype} user (session {arc:.0%} in; needs >= "
                           f"{self.risk_gate:.0%}). Keep the cognitive content lighter for now.")

        return True, "ok"

    def allowed_moves(self, env: PatientEnv) -> list[str]:
        """Moves not obviously blocked right now (hint for the user LLM)."""
        arc = env.arc_index
        allowed = []
        for name in MOVES:
            if name in RISK_MOVES and arc < self.risk_gate:
                continue
            if name == "reveal_risk" and env.archetype == "covert_risk" and arc < self.reveal_gate:
                continue
            allowed.append(name)
        return allowed
