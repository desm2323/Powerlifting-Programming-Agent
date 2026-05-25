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
        "lifter": {"name": None, "units": "kg"},
        "lifts": {lift: {"training_max": None, "best_est_1rm": None} for lift in LIFTS},
        "goal": {"lift": None, "target": None, "weeks": None},
        "block": {"phase": "accumulation", "week": 1, "weeks_per_phase": 4,
                  "consecutive_overreach": 0},
        "readiness": 1.0,
        "sessions": [],
        "prs": [],
    }


def load_state(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return default_state()
    with open(path, "r") as f:
        return json.load(f)


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
