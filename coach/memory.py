"""
memory.py — persistent state (the "Memory" box).

Short-term state: current block phase/week, rolling readiness.
Long-term state: training maxes, every logged session, PR history.

Stored as a single JSON file so it survives between runs — the thing a chat
LLM fundamentally lacks. Swap this for SQLite/a real DB without touching callers.
"""

from __future__ import annotations

import json
import os

DEFAULT_PATH = os.path.join("data", "state.json")
LIFTS = ["squat", "bench", "deadlift"]


def default_state() -> dict:
    return {
        "lifter": {
            "name": None,
            "units": "kg",
            "bodyweight_kg": None,
            "experience": None,         # "novice" | "intermediate" | "advanced"
            "goals_notes": "",          # free-text summary captured by the LLM
            "training_notes": "",       # ongoing context the LLM records
        },
        "lifts": {lift: {"training_max": None, "best_est_1rm": None} for lift in LIFTS},
        # The current block. type=None means the agent hasn't proposed one yet
        # (onboarding mode). The LLM commits a block via propose_block().
        # weekly_plan is the multi-day session structure for the block — a
        # list of training days, each with ordered exercises (primary lift,
        # secondary work, accessories). Empty {} until commit_block populates it.
        "block": {
            "type": None,
            "label": None,
            "week": 1,
            "duration_weeks": None,
            "rationale": None,
            "focus_lifts": [],
            "started_at": None,
            "consecutive_overreach": 0,
            "in_deload": False,
            "weekly_plan": {"days": []},
        },
        # Record of completed blocks — drives adaptive next-block selection.
        "block_history": [],
        # Macrocycle plan: the ordered list of UPCOMING blocks the lifter is
        # planning to run, in calendar order. Built by the LLM via the
        # propose_season tool, usually during onboarding when a competition
        # date is known. Each entry is a TENTATIVE block plan (type +
        # duration + planned_start + rationale) — the detailed weekly_plan
        # is built fresh when the block becomes active, so future blocks
        # can adapt to how training actually goes.
        # competition_date / competition_lifts are nullable.
        "season_plan": {
            "competition_date": None,
            "competition_lifts": None,
            "blocks": [],
        },
        "readiness": 1.0,
        "sessions": [],
        "prs": [],
        # The exact prescription last issued by plan_next(), per lift.
        "pending_prescriptions": {},
        # Persistent chat history so the browser UI survives a refresh.
        # Each entry: {"role": "user"|"assistant", "content": str}.
        "chat_history": [],
        # Progression policy state — populated only when the bandit is
        # active. {"name": "rules"} keeps RulePolicy explicit; switching
        # to bandit fills "cells" with the learned Q-table. See policy.py.
        "policy_state": {"name": "rules"},
        # Delayed-reward bookkeeping. When the policy decides progress /
        # hold / deload after a logged session, we stash the context and
        # action here; the NEXT logged session computes the reward and
        # feeds it back via policy.update(). One slot per lift.
        "pending_decisions": {},
    }


def load_state(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return default_state()
    with open(path, "r") as f:
        state = json.load(f)
    # Forward-compat: ensure all expected keys exist on older saves.
    state.setdefault("pending_prescriptions", {})
    state.setdefault("block_history", [])
    state.setdefault("chat_history", [])
    state.setdefault("policy_state", {"name": "rules"})
    state.setdefault("pending_decisions", {})
    sp = state.setdefault("season_plan", {})
    sp.setdefault("competition_date", None)
    sp.setdefault("competition_lifts", None)
    sp.setdefault("blocks", [])
    lifter = state.setdefault("lifter", {})
    lifter.setdefault("bodyweight_kg", None)
    lifter.setdefault("experience", None)
    lifter.setdefault("goals_notes", "")
    lifter.setdefault("training_notes", "")
    block = state.setdefault("block", {})
    block.setdefault("type", None)
    block.setdefault("label", None)
    block.setdefault("duration_weeks", None)
    block.setdefault("rationale", None)
    block.setdefault("focus_lifts", [])
    block.setdefault("started_at", None)
    block.setdefault("in_deload", False)
    block.setdefault("consecutive_overreach", 0)
    block.setdefault("weekly_plan", {"days": []})
    return state


def save_state(state: dict, path: str = DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def record_session(state: dict, session: dict) -> None:
    state["sessions"].append(session)


def record_pr(state: dict, lift: str, weight: float, reps: int, est_1rm: float) -> None:
    state["prs"].append({"lift": lift, "weight": weight, "reps": reps,
                         "est_1rm": round(est_1rm, 1)})


def recent_sessions(state: dict, lift: str | None = None, n: int = 5) -> list[dict]:
    items = state["sessions"]
    if lift:
        items = [s for s in items if s.get("lift") == lift]
    return items[-n:]
