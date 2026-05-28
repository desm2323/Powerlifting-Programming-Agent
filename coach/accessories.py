"""
accessories.py — the accessory-exercise knowledge base.

A lockout failure on bench is a different problem from a lockout failure on
deadlift, and the fix is a different exercise at a different relative load.
This module is the structured catalog: per (lift, sticking_point), a ranked
list of accessories with intensity_pct (relative to the MAIN lift's training
max), sets/reps, and a one-line rationale for the report.

Loads come from `loading.load_for_intensity(tm, intensity_pct)` — the LLM
never invents the kg. That's the same offline-first invariant as the rest of
the engine.
"""

from __future__ import annotations

from . import loading

# Each entry: (name, intensity_pct vs main-lift TM, sets, reps, rationale).
# Ordered best -> alternative; suggest_for() returns the first by default.
ACCESSORIES: dict[str, dict[str, list[dict]]] = {
    "bench": {
        "lockout": [
            {"name": "Close-grip bench", "intensity_pct": 0.75, "sets": 3, "reps": 6,
             "rationale": "Triceps-dominant pattern that mirrors lockout."},
            {"name": "2-board press", "intensity_pct": 0.85, "sets": 3, "reps": 3,
             "rationale": "Overloads lockout range with heavier-than-bench loads."},
            {"name": "Floor press", "intensity_pct": 0.70, "sets": 3, "reps": 5,
             "rationale": "Removes leg drive — isolates the top half."},
        ],
        "off_the_chest": [
            {"name": "Paused bench (2-3s)", "intensity_pct": 0.80, "sets": 4, "reps": 3,
             "rationale": "Builds dead-stop strength out of the hole."},
            {"name": "Spoto press", "intensity_pct": 0.75, "sets": 3, "reps": 5,
             "rationale": "Hovers 1\" off the chest — bottom-range tension."},
            {"name": "Pin press (chest height)", "intensity_pct": 0.75, "sets": 3, "reps": 4,
             "rationale": "Starts from a dead stop at the weakest point."},
        ],
    },
    "squat": {
        "out_of_the_hole": [
            {"name": "Paused squat (3s)", "intensity_pct": 0.80, "sets": 4, "reps": 3,
             "rationale": "Eliminates stretch reflex — pure bottom-position strength."},
            {"name": "Tempo squat (3s eccentric)", "intensity_pct": 0.70, "sets": 4, "reps": 5,
             "rationale": "Slow descent builds control and stability in the hole."},
            {"name": "Pin squat (below parallel)", "intensity_pct": 0.75, "sets": 3, "reps": 3,
             "rationale": "Dead-stop start from the exact sticking position."},
        ],
        "lockout": [
            {"name": "High-pin squat", "intensity_pct": 0.90, "sets": 3, "reps": 3,
             "rationale": "Overloads the top half above the sticking point."},
            {"name": "1.25 squat", "intensity_pct": 0.75, "sets": 3, "reps": 4,
             "rationale": "Reinforces drive through the mid-range transition."},
            {"name": "Anderson squat", "intensity_pct": 0.80, "sets": 3, "reps": 3,
             "rationale": "Concentric-only out of pins — kills momentum-based lockouts."},
        ],
    },
    "deadlift": {
        "off_the_floor": [
            {"name": "Deficit deadlift (1-2\")", "intensity_pct": 0.75, "sets": 3, "reps": 4,
             "rationale": "Extended ROM forces stronger off-floor mechanics."},
            {"name": "Paused deadlift (1s below knee)", "intensity_pct": 0.75, "sets": 4, "reps": 3,
             "rationale": "Pauses force tension and strict positioning off the floor."},
            {"name": "Snatch-grip deadlift", "intensity_pct": 0.70, "sets": 3, "reps": 5,
             "rationale": "Wider grip = bigger ROM and stronger upper back off the floor."},
        ],
        "lockout": [
            {"name": "Block pull (2\" blocks)", "intensity_pct": 0.90, "sets": 3, "reps": 3,
             "rationale": "Overloads from above the sticking point."},
            {"name": "Rack pull (knee height)", "intensity_pct": 1.00, "sets": 3, "reps": 3,
             "rationale": "Heavy upper-back / lockout overload."},
            {"name": "Romanian deadlift", "intensity_pct": 0.65, "sets": 3, "reps": 6,
             "rationale": "Posterior-chain builder — lockout is hip-extension-limited."},
        ],
    },
}


def suggest_for(lift: str, sticking_point: str,
                training_max: float | None) -> dict | None:
    """Return the top-ranked accessory for (lift, sticking_point), with the
    load already computed against the main lift's training max."""
    by_lift = ACCESSORIES.get(lift)
    if not by_lift:
        return None
    options = by_lift.get(sticking_point)
    if not options:
        return None
    pick = dict(options[0])  # top-ranked
    if training_max:
        pick["load_kg"] = loading.load_for_intensity(training_max, pick["intensity_pct"])
    else:
        pick["load_kg"] = None
    return pick


def list_for(lift: str, sticking_point: str,
             training_max: float | None) -> list[dict]:
    """Return all candidates for (lift, sticking_point) with loads computed."""
    by_lift = ACCESSORIES.get(lift, {})
    options = by_lift.get(sticking_point, [])
    out = []
    for opt in options:
        d = dict(opt)
        d["load_kg"] = (loading.load_for_intensity(training_max, d["intensity_pct"])
                        if training_max else None)
        out.append(d)
    return out
