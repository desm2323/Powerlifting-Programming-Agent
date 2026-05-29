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
        "weak_upper_back": [
            {"name": "Barbell row", "intensity_pct": 0.0, "sets": 4, "reps": 8,
             "rationale": "Builds the upper-back shelf that stabilises the press."},
            {"name": "Weighted pull-up", "intensity_pct": 0.0, "sets": 3, "reps": 6,
             "rationale": "Loadable vertical pull for lat/upper-back size + tightness."},
            {"name": "Face pull", "intensity_pct": 0.0, "sets": 3, "reps": 15,
             "rationale": "Rear-delt / external-rotator health and scap stability."},
        ],
        "weak_leg_drive": [
            {"name": "Leg press", "intensity_pct": 0.0, "sets": 3, "reps": 10,
             "rationale": "Builds the leg strength that drives the bench off the chest."},
            {"name": "Hip thrust", "intensity_pct": 0.0, "sets": 3, "reps": 10,
             "rationale": "Reinforces the hip/leg drive used in the arch."},
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
            {"name": "Box squat", "intensity_pct": 0.75, "sets": 3, "reps": 3,
             "rationale": "Reinforces position + control at the bottom."},
        ],
        "lockout": [
            {"name": "High-pin squat", "intensity_pct": 0.90, "sets": 3, "reps": 3,
             "rationale": "Overloads the top half above the sticking point."},
            {"name": "1.25 squat", "intensity_pct": 0.75, "sets": 3, "reps": 4,
             "rationale": "Reinforces drive through the mid-range transition."},
            {"name": "Anderson squat", "intensity_pct": 0.80, "sets": 3, "reps": 3,
             "rationale": "Concentric-only out of pins — kills momentum-based lockouts."},
        ],
        # Muscle / quality weaknesses (isolation + machine — RPE-only, no %TM).
        "weak_quads": [
            {"name": "Leg press", "intensity_pct": 0.0, "sets": 3, "reps": 10,
             "rationale": "Loadable quad builder with low spinal cost."},
            {"name": "Bulgarian split squat", "intensity_pct": 0.0, "sets": 3, "reps": 8,
             "rationale": "Unilateral quad work — fixes side-to-side weakness."},
            {"name": "Leg extension", "intensity_pct": 0.0, "sets": 3, "reps": 12,
             "rationale": "Isolates the quads (incl. rectus femoris) the squat under-trains."},
        ],
        "weak_core": [
            {"name": "Weighted carry", "intensity_pct": 0.0, "sets": 3, "reps": 1,
             "rationale": "Loadable trunk/bracing stability under load."},
            {"name": "Hanging leg raise", "intensity_pct": 0.0, "sets": 3, "reps": 12,
             "rationale": "Loadable anterior-core work that progresses (unlike planks)."},
            {"name": "Weighted decline sit-up", "intensity_pct": 0.0, "sets": 3, "reps": 10,
             "rationale": "Progressive trunk flexion strength."},
        ],
        "forward_lean": [
            {"name": "Good morning", "intensity_pct": 0.0, "sets": 3, "reps": 6,
             "rationale": "Strengthens the posterior chain that controls torso angle."},
            {"name": "Reverse hyper", "intensity_pct": 0.0, "sets": 3, "reps": 12,
             "rationale": "Low-cost erector / glute work to hold a more upright torso."},
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
        "grip": [
            {"name": "Farmer's carry", "intensity_pct": 0.0, "sets": 3, "reps": 1,
             "rationale": "Heavy loaded carry — direct grip + trunk endurance."},
            {"name": "Plate pinch", "intensity_pct": 0.0, "sets": 3, "reps": 1,
             "rationale": "Targets pinch / thumb grip the bar doesn't fully tax."},
            {"name": "Heavy barbell hold", "intensity_pct": 0.90, "sets": 3, "reps": 1,
             "rationale": "Overload hold at/above pulling weight for grip carryover."},
        ],
        "lower_back": [
            {"name": "Back extension", "intensity_pct": 0.0, "sets": 3, "reps": 12,
             "rationale": "Loadable erector work to support the pull."},
            {"name": "Reverse hyper", "intensity_pct": 0.0, "sets": 3, "reps": 12,
             "rationale": "Low-cost erector / glute work — Bromley's erector fix."},
            {"name": "Good morning", "intensity_pct": 0.0, "sets": 3, "reps": 6,
             "rationale": "Builds the spinal-erector + hip-hinge strength the pull needs."},
        ],
    },
}


# Map free-text weakness phrasings onto catalog sticking-point keys, so the
# weakness tool resolves what a lifter actually says ("deadlift lockout",
# "weak off the floor", "no quads", "grip gives out") to a catalog entry.
# Order matters — earlier (more specific) phrases win.
WEAKNESS_SYNONYMS: dict[str, dict[str, str]] = {
    "squat": {
        "out of the hole": "out_of_the_hole", "bottom": "out_of_the_hole",
        "hole": "out_of_the_hole", "depth": "out_of_the_hole",
        "stability": "out_of_the_hole", "unstable": "out_of_the_hole",
        "lockout": "lockout", "lock out": "lockout", "top": "lockout",
        "quad": "weak_quads", "quads": "weak_quads",
        "core": "weak_core", "trunk": "weak_core", "abs": "weak_core",
        "forward lean": "forward_lean", "lean": "forward_lean",
        "tipping": "forward_lean", "good morning": "forward_lean",
    },
    "bench": {
        "lockout": "lockout", "lock out": "lockout", "triceps": "lockout",
        "tricep": "lockout", "top": "lockout",
        "off the chest": "off_the_chest", "chest": "off_the_chest",
        "bottom": "off_the_chest", "bar path": "off_the_chest",
        "upper back": "weak_upper_back", "back": "weak_upper_back",
        "lats": "weak_upper_back",
        "leg drive": "weak_leg_drive", "legs": "weak_leg_drive",
    },
    "deadlift": {
        "off the floor": "off_the_floor", "floor": "off_the_floor",
        "start": "off_the_floor", "bottom": "off_the_floor",
        "lockout": "lockout", "lock out": "lockout", "top": "lockout",
        "grip": "grip", "hands": "grip", "hold": "grip",
        "lower back": "lower_back", "low back": "lower_back",
        "back": "lower_back", "erector": "lower_back",
    },
}


def normalize_weakness(lift: str, weakness: str) -> str | None:
    """Resolve a free-text weakness ('deadlift lockout', 'weak off the floor',
    'grip gives out') to a catalog sticking-point key. Returns None if no
    phrase matches. Exact catalog keys pass through unchanged."""
    by_lift = ACCESSORIES.get(lift, {})
    text = (weakness or "").lower().strip()
    if text in by_lift:
        return text
    syns = WEAKNESS_SYNONYMS.get(lift, {})
    # Longest phrases first so 'off the floor' wins over 'floor'.
    for phrase in sorted(syns, key=len, reverse=True):
        if phrase in text:
            return syns[phrase]
    return None


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
