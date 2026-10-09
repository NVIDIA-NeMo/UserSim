# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``GuardedMoveMixin`` — a reusable, auditable move-space capability for probes.

Compose it onto any ``BaseProbe`` (``class X(GuardedMoveMixin, BaseProbe)``) to
give the simulated user an *auditable, Guard-paced* move-space: each follow-up
turn the user commits ONE move via a native ``commit_move`` tool call, a
deterministic Guard validates its pacing (veto -> native tool-result -> re-commit),
and the committed move + Guard verdict are recorded as audit columns.

Design goals (idiomatic + extensible, matching the repo's other capability
mixins — ``BankReframingMixin``, ``AgenticMixin``):

* **Domain-agnostic.** The mixin hardcodes no topics, no notion of "risk", no
  prompts. Everything domain-specific is supplied by the host probe through
  :meth:`guarded_move_config` (an opaque config mapping passed straight to the
  per-turn engine in :mod:`.move_runtime`). A family in another domain can reuse
  this mixin by supplying its own config mapping of the same shape.
* **Per-trajectory enablement, not process-global state.** The capability is
  toggled by the resolved ``probe_variant`` (the same per-row discriminator the
  rest of the framework uses for stratification) — see
  :meth:`guarded_moves_enabled`. Default OFF, so a host probe's plain behaviour
  is unchanged until the ``guarded`` variant is selected. An optional env var
  (``GUARDED_MOVE_ENV``) is honoured only as a local-dev override.

The per-turn engine (tool schema, propose/guard loop, env/Guard construction)
lives in :mod:`.move_runtime`; this mixin is only the ``BaseProbe`` seam that
wires that engine into the conversation loop's hooks.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Mapping

from usersim.engine.probes.health_disclosure import move_runtime as mr

logger = logging.getLogger("usersim.engine")

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")

#: Guards the (rarely-written) process-wide default move hook — see
#: ``GuardedMoveMixin.register_move_hook``.
_MOVE_HOOK_LOCK = threading.Lock()

#: Every column ``GuardedMoveMixin.build_result_extras`` adds on the guarded
#: variant. The engine keeps only the columns named in
#: ``ConversationSimulatorConfig.side_effect_columns`` and drops the rest, with
#: only a logged warning, so each of these must be listed there too; the tests
#: check both that the guarded extras are exactly these and that the config
#: declares them.
GUARDED_RESULT_COLUMNS: tuple[str, ...] = (
    "moves_enabled",
    "moves_played",
    "moves_detail",
    "guard_veto_count",
    "disclosed_topics",
    "committed_disclosure_levels",
    "concealment_topics",
    "disclosure_coverage",
    "risk_present",
    "risk_noun",
    "risk_revealed",
    "risk_revealed_turn",
    "patient_archetype",
    "turn_budget",
    "risk_opportunity",
)


def env_override(name: str) -> bool | None:
    """Tri-state read of a boolean env var: ``True``/``False`` for a recognised
    on/off token, else ``None`` (unset or unrecognised).

    The single truthy/falsey parser for the health_disclosure feature — both the
    move-space toggle here and the stand-in flag in ``base.py`` route through it,
    so there is exactly one definition of what counts as on/off.
    """
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return None


class GuardedMoveMixin:
    """Adds a Guard-paced, auditable ``commit_move`` seam to a ``BaseProbe``.

    Host probes must implement :meth:`guarded_move_config` and call
    :meth:`init_guarded_moves` at the end of their ``__init__``.
    """

    #: ``probe_variant`` label that turns the guarded move-space ON.
    GUARDED_VARIANT: str = "guarded"
    #: Optional env var name that overrides the variant (local dev only). Set on
    #: the host class to keep a convenience switch; leave ``None`` to rely solely
    #: on the per-trajectory variant.
    GUARDED_MOVE_ENV: str | None = None

    #: Process-wide default env-harness move hook (a serializable-payload sink;
    #: see ``move_runtime._move_payload``). Lives on the *capability*, not in a
    #: domain module, so any family reusing the mixin shares one seam. Prefer
    #: per-trajectory injection via ``guarded_move_config()["move_hook"]``; this
    #: default is the ergonomic "observe every trajectory" switch. Either channel
    #: is separable under concurrency because every payload is tagged with the
    #: trajectory identity (:meth:`_move_identity`).
    _default_move_hook: "mr.MoveHook" | None = None

    # ── env-harness hook registration (domain-agnostic capability seam) ──
    @classmethod
    def register_move_hook(
        cls,
        fn: "mr.MoveHook" | None,
    ) -> "mr.MoveHook" | None:
        """Register (or clear, with ``None``) the process-wide default move hook.

        Returns the previous default so callers can restore it. A hook that
        raises is swallowed downstream — a harness error never breaks the sim.
        This is a convenience for attaching one observer to *all* trajectories;
        for per-trajectory control, supply ``move_hook`` in the config mapping
        returned by :meth:`guarded_move_config`, which takes precedence.
        """
        with _MOVE_HOOK_LOCK:
            prev = GuardedMoveMixin._default_move_hook
            GuardedMoveMixin._default_move_hook = fn
            return prev

    # ── internals ───────────────────────────────────────────────────
    def _scaffold_locale(self) -> str:
        """Locale to resolve the engine's prompt packs against.

        ``BaseProbe`` sets ``_asset_locale`` to the locale whose packs/banks
        actually resolved (the India base ``en_IN`` for a language-variant that
        has no native pack yet), so keying the scaffolds off it keeps a variant
        from aborting on a missing prompt. Falls back to the conversation
        locale on a bare host that never ran ``BaseProbe.__init__``.
        """
        return getattr(self, "_asset_locale", None) or getattr(self, "_locale", None) or "en_US"

    # ── configuration hooks the host probe supplies ─────────────────
    def guarded_move_config(self) -> Mapping[str, Any]:
        """Return the domain config mapping (topics / risk meaning / proposal
        framing / ``profiles_env`` ...). Passed opaquely to ``move_runtime``.

        Override in the host probe (the health family returns ``self.CLIENT``).
        """
        raise NotImplementedError(f"{type(self).__name__} must implement guarded_move_config()")

    def guarded_moves_enabled(self) -> bool:
        """Whether the guarded move-space is active for THIS trajectory.

        Primary switch: the resolved ``probe_variant`` equals
        :attr:`GUARDED_VARIANT`. If :attr:`GUARDED_MOVE_ENV` is set on the class,
        that env var is an optional local-dev override that forces on/off.
        """
        env = getattr(self, "GUARDED_MOVE_ENV", None)
        if env:
            override = env_override(env)
            if override is not None:
                return override
        variant = str((getattr(self, "_data", None) or {}).get("probe_variant", "")).strip().lower()
        return variant == self.GUARDED_VARIANT

    # ── lifecycle ───────────────────────────────────────────────────
    def init_guarded_moves(self, *, disclosure_style: str, max_turns: int) -> None:
        """Build the env / Guard / topic list when enabled.

        Call from the host ``__init__`` AFTER ``super().__init__`` (so
        ``self._persona`` / ``self._data`` exist). A no-op when disabled — the
        host probe's plain behaviour is then unchanged.
        """
        self._moves_on: bool = self.guarded_moves_enabled()
        self._gm_config: Mapping[str, Any] = {}
        self._env = None
        self._guard = None
        self._topics: list[str] = []
        self._has_risk: bool = False
        self._risk_revealed_turn: int | None = None
        # Env-harness seam, resolved per-instance (never read from a global at
        # emit time): config-supplied hook wins, else the capability default.
        self._move_hook: "mr.MoveHook" | None = None
        self._move_identity: dict[str, Any] = {}
        if not self._moves_on:
            return
        cfg = self.guarded_move_config()
        self._gm_config = cfg
        self._move_hook = self._resolve_move_hook(cfg)
        self._move_identity = self._build_move_identity(cfg)
        # The hidden clinical profile is derived from the versioned clinical-
        # profile bank by BankBackedProbe (self._task); ``getattr`` keeps the
        # mixin usable on a bare host that has no bank-backed task.
        task = getattr(self, "_task", None)
        profile = dict(getattr(task, "profile", None) or {})
        self._env, self._guard, self._topics, self._has_risk = mr.build_env(
            num_turns=max_turns,
            profile=profile,
            disclosure_style=disclosure_style,
            client=cfg,
        )

    def _resolve_move_hook(self, cfg: Mapping[str, Any]) -> "mr.MoveHook" | None:
        """Resolve the env-harness hook for THIS trajectory, carried on the
        instance (never read from a global at emit time): a per-trajectory
        ``move_hook`` in the config mapping wins, else the capability's
        process-wide default registered via :meth:`register_move_hook`.
        """
        return cfg.get("move_hook") or GuardedMoveMixin._default_move_hook

    def _build_move_identity(self, cfg: Mapping[str, Any]) -> dict[str, Any]:
        """Trajectory identity stamped on every emitted move payload.

        Uses the framework's canonical, content-derived ``trajectory_id`` (set on
        ``self._data`` by the generator) plus persona/family/variant, so an
        out-of-band harness can attribute concurrently-generated rows — the whole
        point of not sharing one un-tagged process-global callback.

        ``trajectory_id`` is the reliable join key. ``persona_uuid`` is
        **best-effort**: synthetic and custom personas need not carry a uuid, in
        which case it is ``None``. Don't build a harness-side join on it.
        """
        persona = getattr(self, "_persona", None) or {}
        data = getattr(self, "_data", None) or {}
        uuid = str(persona.get("uuid") or persona.get("persona_uuid") or "").strip()
        return {
            "trajectory_id": data.get("trajectory_id"),
            "persona_uuid": uuid or None,
            "probe_family": data.get("probe_family") or cfg.get("family"),
            "probe_variant": data.get("probe_variant"),
        }

    # ── per-turn seam (BaseProbe hooks) ─────────────────────────────
    async def format_followup_user_instructions(self, turn_idx: int, state: Any) -> list[str]:
        """Commit a Guard-validated move for this turn and instruct the user.

        Returns ``[]`` when the move-space is disabled, so the host probe's base
        disclosure behaviour is untouched.
        """
        if not getattr(self, "_moves_on", False) or self._env is None or self._guard is None:
            return []
        self._env.turn = turn_idx
        move, veto_log, reasoning = await mr.propose_and_guard(
            self._models,
            self._env,
            self._guard,
            self._topics,
            self._has_risk,
            state,
            self._gm_config,
            self._scaffold_locale(),
            emit=getattr(self, "_move_hook", None),
            identity=getattr(self, "_move_identity", None),
        )
        self._env.apply(move)
        if move.move == "reveal_risk" and self._risk_revealed_turn is None:
            self._risk_revealed_turn = turn_idx
        rec = move.to_dict()
        rec["turn"] = turn_idx
        rec["reasoning"] = reasoning
        rec["vetoes"] = veto_log
        state.metadata.setdefault("disclosure_moves", []).append(rec)
        return [mr.move_instruction(move)]

    def allow_early_stop_at_turn(self, turn_idx: int, state: Any) -> bool:
        """Keep sessions running to full length when the move-space is on, so the
        disclosure arc reaches the Guard's later pacing gates (risk)."""
        return not getattr(self, "_moves_on", False)

    def build_result_extras(self, state: Any) -> dict:
        """Add the move/Guard audit columns on top of the host's extras."""
        extras = super().build_result_extras(state)  # type: ignore[misc]
        if not getattr(self, "_moves_on", False):
            return extras
        moves = state.metadata.get("disclosure_moves", []) or []
        # Binary column: "disclosed" means told in full. Partials are carried
        # separately in ``committed_disclosure_levels``, which grades them — so
        # counting them here as full would contradict the graded scorer.
        disclosed = sorted(self._env.disclosed_full) if self._env is not None else []
        topics = list(self._topics)
        coverage = (len(set(disclosed) & set(topics)) / len(topics)) if topics else 0.0
        # EVERY structured column below is JSON-encoded, because that is the
        # store's convention and not a stylistic choice: the framework persists
        # `conversation_messages`, `simulation_outcome`, `simulation_traces` and
        # the tool_calling probe's `tools` the same way, so a trajectory parquet
        # is otherwise entirely scalar columns.
        #
        # A raw list or dict here becomes a NESTED pyarrow dtype, and that breaks
        # the store in two separate ways:
        #
        #   * writing — an empty dict is a struct with no child fields, which
        #     Parquet cannot represent at all, and the failure aborts the whole
        #     batch rather than the offending row;
        #   * reading — pandas records the dtype in file metadata as e.g.
        #     ``list<element: string>[pyarrow]`` and cannot reconstruct a nested
        #     ArrowDtype from that string, so ``pd.read_parquet`` raises
        #     ``TypeError: data type ... not understood``. The evaluator reads the
        #     trajectory frame that way, so EVERY guarded trajectory became
        #     unreadable — the run generates fine and then cannot be scored.
        #
        # The scorer coerces all of these back via ``_as_list`` / ``_as_levels``.
        dumps = lambda v: json.dumps(v, ensure_ascii=False, default=str)  # noqa: E731
        extras.update(
            {
                "moves_enabled": True,
                "moves_played": dumps([m.get("move") for m in moves]),
                "moves_detail": dumps(moves),
                "guard_veto_count": sum(len(m.get("vetoes") or []) for m in moves),
                # Committed-INTENT columns (what the move-space decided).
                "disclosed_topics": dumps(disclosed),
                # Graded committed intent (full vs partial), so the scorer's fallback
                # (when realized audit is unavailable) can grade partial_disclose too.
                "committed_disclosure_levels": dumps(_committed_levels(moves)),
                "concealment_topics": dumps(topics),
                "disclosure_coverage": round(coverage, 3),
                "risk_present": bool(self._has_risk),
                # What "risk" concretely means for this client ("self-harm", "an
                # urgent red-flag symptom", ...). Persisted because the evaluator's
                # realized-behavior audit interpolates it into the auditor prompt and
                # cannot reach the client config from there; without it the auditor is
                # asked about the useless generic "risk".
                "risk_noun": self._gm_config.get("risk_noun") or "risk",
                "risk_revealed": bool(self._env.risk_revealed) if self._env else False,
                "risk_revealed_turn": self._risk_revealed_turn,
                "patient_archetype": self._env.archetype if self._env else None,
                # Turn budget + whether the paced risk-reveal window was structurally
                # reachable within it. The scorer uses ``turn_budget`` to stratify
                # coverage (a Guard-pacing-confounded axis) and ``risk_opportunity`` to
                # avoid charging a non-revealed risk as a safety miss when the budget
                # was too short for the arc to complete (see move_runtime).
                "turn_budget": int(self._env.num_turns) if self._env is not None else None,
                "risk_opportunity": (
                    mr.risk_reveal_reachable(self._env, self._guard)
                    if (self._env is not None and self._guard is not None)
                    else None
                ),
            }
        )
        # No realized-behavior columns here: what the utterances actually
        # disclosed is audited by health_disclosure_concealment at eval time, off
        # the columns above. The audit needs an LLM, and a call inside column
        # assembly would put a network round-trip (and its cost, latency and
        # failure surface) on the simulation path.
        return extras


def _committed_levels(moves: list[dict]) -> dict:
    """Graded committed intent per topic from the move list (full > partial)."""
    order = {"none": 0, "partial": 1, "full": 2}
    levels: dict[str, str] = {}
    for m in moves:
        topic = str(m.get("topic") or "")
        name = str(m.get("move") or "")
        if not topic or name not in ("disclose", "partial_disclose"):
            continue
        grade = "full" if name == "disclose" else "partial"
        if order[grade] > order.get(levels.get(topic, "none"), 0):
            levels[topic] = grade
    return levels
