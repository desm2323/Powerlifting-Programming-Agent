"""
blocks.py — block-type knowledge base.

A real coach doesn't blindly cycle a lifter through accumulation -> strength
-> peak -> taper. They diagnose what the lifter needs RIGHT NOW and pick the
right block type. Four core block types here, each with:
  - prescription template (sets/reps/intensity/RPE cap)
  - deload-week template
  - typical duration in weeks
  - human-readable "focus" line for explanations

The LLM picks WHICH block via the propose_block tool. This module is the
catalogue it picks from — the knowledge base.
"""

from __future__ import annotations

from . import loading
from .expert_knowledge import is_rpe_only_accessory

BLOCK_TYPES = {
    "volume": {
        "label": "Volume Block",
        "default_weeks": 4,
        "sets": 4, "reps": 6, "intensity_pct": 0.72, "rpe_cap": 8.0,
        "weekly_set_target": 14,
        "focus": ("build work capacity — drive hypertrophy and refine "
                  "technique under fatigue"),
        "indication": ("use when work capacity is low, after a long peak, "
                       "or when stalled lifts need an accumulation phase"),
        "deload": {"sets": 2, "reps": 5, "intensity_pct": 0.65, "rpe_cap": 7.0,
                   "weekly_set_target": 7,
                   "focus": "deload: drop volume, recover"},
    },
    "technique": {
        "label": "Technique Block",
        "default_weeks": 3,
        "sets": 4, "reps": 5, "intensity_pct": 0.65, "rpe_cap": 7.0,
        "weekly_set_target": 12,
        "focus": ("ingrain movement patterns — paused/tempo work at lower "
                  "loads, strict execution"),
        "indication": ("use when execution issues are limiting strength "
                       "expression — bar path, depth, leg drive, sticking "
                       "point caused by technique not strength"),
        "deload": {"sets": 2, "reps": 4, "intensity_pct": 0.60, "rpe_cap": 6.5,
                   "weekly_set_target": 6,
                   "focus": "deload: light technique refinement"},
    },
    "strength": {
        "label": "Strength Block",
        "default_weeks": 4,
        "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9.0,
        "weekly_set_target": 11,
        "focus": ("express force — heavier loads, lower volume, build "
                  "absolute strength"),
        "indication": ("use AFTER a volume or technique block — express the "
                       "capacity you just built at higher intensities"),
        "deload": {"sets": 2, "reps": 3, "intensity_pct": 0.70, "rpe_cap": 7.0,
                   "weekly_set_target": 5,
                   "focus": "deload: light, snappy work"},
    },
    "peaking": {
        "label": "Peaking Block",
        "default_weeks": 3,
        # rpe_cap 9.0 (down from 9.5): Ben is explicit — NEVER train main
        # lifts to failure (zrOWMftnvQw @ 4:35, n9Wyxr4Yks0 @ 7:55). The
        # week-3 wave adds +0.5, so the final week tops at RPE 9.5 (1 RIR),
        # not 10 (failure). RPE 9 IS the peak attempt selector, not a max.
        "sets": 3, "reps": 1, "intensity_pct": 0.92, "rpe_cap": 9.0,
        "weekly_set_target": 7,
        "focus": "sharpen near-maximal singles, then taper into a test day",
        "indication": ("use in the final 2-4 weeks before a meet or max "
                       "attempt — neural prep, not strength building"),
        "deload": {"sets": 2, "reps": 1, "intensity_pct": 0.75, "rpe_cap": 7.0,
                   "weekly_set_target": 4,
                   "focus": "deload/taper: total recovery before testing"},
    },
}

VALID_BLOCK_TYPES = tuple(BLOCK_TYPES.keys())


# Per-week wave: how each productive week of a block differs from the others.
# Each entry has an intensity multiplier (applied to the exercise's base
# intensity_pct), a label, and optional set/rep deltas. Real coaching never
# has all weeks at identical loads — there's always a wave or a ramp.
#
# Each wave carries 6 entries so 4-, 5- and 6-week blocks all render with a
# distinct prescription per week (week_modifier clamps to the last entry for
# longer blocks). Week 6 ramps intensity slightly past week 5 while holding
# RPE flat, keeping 6-week blocks under the safe RPE ceiling.
BLOCK_WAVES = {
    "volume": [
        {"label": "Establish",   "intensity_mult": 0.92, "set_delta": 0, "rep_delta": 0, "rpe_delta": -1.5},
        {"label": "Build",       "intensity_mult": 0.96, "set_delta": 0, "rep_delta": 0, "rpe_delta": -1.0},
        {"label": "Develop",     "intensity_mult": 1.00, "set_delta": 0, "rep_delta": 0, "rpe_delta": -0.5},
        {"label": "Push",        "intensity_mult": 1.02, "set_delta": 1, "rep_delta": 0, "rpe_delta":  0.0},
        {"label": "Peak volume", "intensity_mult": 1.04, "set_delta": 1, "rep_delta": 0, "rpe_delta": +0.5},
        {"label": "Overreach",   "intensity_mult": 1.06, "set_delta": 2, "rep_delta": 0, "rpe_delta": +0.5},
    ],
    "technique": [
        {"label": "Slow & strict", "intensity_mult": 0.88, "set_delta": 0, "rep_delta": 0, "rpe_delta": -1.5},
        {"label": "Consolidate",   "intensity_mult": 0.92, "set_delta": 0, "rep_delta": 0, "rpe_delta": -1.0},
        {"label": "Re-load",       "intensity_mult": 0.96, "set_delta": 0, "rep_delta": 0, "rpe_delta": -0.5},
        {"label": "Express",       "intensity_mult": 1.00, "set_delta": 0, "rep_delta": 0, "rpe_delta":  0.0},
        {"label": "Confirm",       "intensity_mult": 1.02, "set_delta": 0, "rep_delta": 0, "rpe_delta": +0.5},
        {"label": "Refine",        "intensity_mult": 1.04, "set_delta": 0, "rep_delta": 0, "rpe_delta": +0.5},
    ],
    "strength": [
        {"label": "Re-introduce", "intensity_mult": 0.92, "set_delta": 0,  "rep_delta": +1, "rpe_delta": -1.5},
        {"label": "Build",        "intensity_mult": 0.96, "set_delta": 0,  "rep_delta": 0,  "rpe_delta": -1.0},
        {"label": "Strain",       "intensity_mult": 0.98, "set_delta": 0,  "rep_delta": 0,  "rpe_delta": -0.5},
        {"label": "Heavy",        "intensity_mult": 1.00, "set_delta": 0,  "rep_delta": 0,  "rpe_delta":  0.0},
        {"label": "Top sets",     "intensity_mult": 1.04, "set_delta": -1, "rep_delta": -1, "rpe_delta": +0.5},
        {"label": "Peak strain",  "intensity_mult": 1.06, "set_delta": -1, "rep_delta": -1, "rpe_delta": +0.5},
    ],
    "peaking": [
        {"label": "Open heavy",     "intensity_mult": 0.92, "set_delta": 0,  "rep_delta": 0, "rpe_delta": -1.0},
        {"label": "Build-up",       "intensity_mult": 0.96, "set_delta": 0,  "rep_delta": 0, "rpe_delta": -0.5},
        {"label": "Second attempt", "intensity_mult": 1.00, "set_delta": 0,  "rep_delta": 0, "rpe_delta":  0.0},
        {"label": "Third attempt",  "intensity_mult": 1.03, "set_delta": 0,  "rep_delta": 0, "rpe_delta": +0.5},
        {"label": "Taper",          "intensity_mult": 1.05, "set_delta": -1, "rep_delta": 0, "rpe_delta": +0.5},
        {"label": "Openers / meet", "intensity_mult": 1.05, "set_delta": -2, "rep_delta": 0, "rpe_delta": +0.5},
    ],
}


def week_modifier(block_type: str, week_index: int) -> dict:
    """Return the wave entry for week N (1-indexed). Clamps to the last week
    if the lifter's block is longer than the wave."""
    if block_type not in BLOCK_WAVES:
        return {"label": "", "intensity_mult": 1.0, "set_delta": 0,
                "rep_delta": 0, "rpe_delta": 0.0}
    wave = BLOCK_WAVES[block_type]
    idx = max(0, min(week_index - 1, len(wave) - 1))
    return dict(wave[idx])


def compute_exercise_load(exercise: dict, training_max: float | None,
                          block_type: str, week: int,
                          in_deload: bool = False) -> dict:
    """Produce concrete numbers (sets/reps/top_weight) for an exercise on a
    given week. Pure function — no state writes.

    Role-aware behaviour (matches Ben's framework — see expert_knowledge.py):
      role in {primary, secondary}: applies the block's per-week wave
        (intensity_mult / set_delta / rep_delta). In deload week, swaps
        in the block's deload prescription wholesale.
      role == 'accessory': preserves the exercise's prescribed sets /
        reps / intensity_pct / rpe_cap. Accessories don't inherit a
        strength-block's rep_delta — a lateral raise stays at 3x12 even
        when the primary squat shifts to 3x2. In deload week, accessory
        sets drop by 1 (min 1) to manage fatigue, but reps and RPE stay.

    `exercise` shape: {name, lift, role, sets, reps, intensity_pct, rpe_cap, rationale}.
    `training_max` may be None for accessories that aren't %TM-driven.
    """
    role = exercise.get("role", "primary")
    is_main = role in ("primary", "secondary")

    # An accessory whose load can't be derived from a barbell training max
    # (DB / cable / machine / weighted pull-up / etc.) has no meaningful
    # %TM. Force intensity_pct to 0 so the downstream load math + plate
    # breakdown are skipped and only the RPE prescription is shown. See
    # expert_knowledge.is_rpe_only_accessory.
    if not is_main and is_rpe_only_accessory(exercise.get("name", "")):
        exercise = dict(exercise)
        exercise["intensity_pct"] = 0.0

    # A primary/secondary main-lift exercise must always carry a real %TM.
    # If one arrives with an impossibly low intensity, fall back to the
    # block's default working-week intensity so the lifter still gets a
    # calibrated weight rather than an RPE-only prescription.
    if is_main and (exercise.get("intensity_pct") or 0.0) < 0.4:
        exercise = dict(exercise)
        exercise["intensity_pct"] = BLOCK_TYPES[block_type]["intensity_pct"]

    if in_deload:
        if is_main:
            dl = deload_prescription(block_type)
            sets = dl["sets"]
            reps = dl["reps"]
            intensity_pct = dl["intensity_pct"]
            rpe_cap = dl["rpe_cap"]
        else:
            # Accessories: cut volume modestly, preserve rep range + RPE.
            sets = max(1, exercise["sets"] - 1)
            reps = exercise["reps"]
            intensity_pct = exercise["intensity_pct"]
            rpe_cap = exercise.get("rpe_cap", 8.5)
        label = "Deload"
    elif is_main:
        mod = week_modifier(block_type, week)
        sets = max(1, exercise["sets"] + mod["set_delta"])
        reps = max(1, exercise["reps"] + mod["rep_delta"])
        intensity_pct = max(0.4, min(1.0,
                                     exercise["intensity_pct"] * mod["intensity_mult"]))
        base_rpe = exercise.get("rpe_cap", 8.5)
        # Ben's framework: RPE waves alongside intensity across the block
        # (e.g. RPE 7 -> 7.5 -> 8 -> 8.5 over a 4-week strength block).
        # Clamped to [5.0, 10.0] so we never prescribe failure or below 5RIR.
        rpe_cap = max(5.0, min(10.0, base_rpe + mod.get("rpe_delta", 0.0)))
        label = mod["label"]
    else:
        # Accessory in a productive week: preserve as prescribed.
        sets = exercise["sets"]
        reps = exercise["reps"]
        intensity_pct = exercise["intensity_pct"]
        rpe_cap = exercise.get("rpe_cap", 8.5)
        label = week_modifier(block_type, week).get("label", "")

    # intensity_pct <= 0 signals "non-%TM exercise" — DB lateral raise,
    # cable pushdown, machine work, bodyweight. No kg or plate breakdown
    # is computed; the UI displays the RPE prescription only.
    if intensity_pct > 0.01 and training_max:
        weight = loading.load_for_intensity(training_max, intensity_pct)
    else:
        weight = None
    return {
        "sets": sets, "reps": reps,
        "intensity_pct": round(intensity_pct, 3),
        "rpe_cap": rpe_cap,
        "top_weight": weight,
        "week_label": label,
    }


def block_prescription(block_type: str) -> dict:
    """Return the working-week prescription for a block type."""
    if block_type not in BLOCK_TYPES:
        raise ValueError(f"Unknown block type: {block_type}")
    bt = BLOCK_TYPES[block_type]
    return {"sets": bt["sets"], "reps": bt["reps"],
            "intensity_pct": bt["intensity_pct"], "rpe_cap": bt["rpe_cap"],
            "weekly_set_target": bt["weekly_set_target"],
            "focus": bt["focus"]}


def deload_prescription(block_type: str) -> dict:
    """Return the deload-week prescription for a block type."""
    if block_type not in BLOCK_TYPES:
        raise ValueError(f"Unknown block type: {block_type}")
    return dict(BLOCK_TYPES[block_type]["deload"])


def block_catalogue() -> list[dict]:
    """All block types as a list — used by the LLM to choose between them."""
    out = []
    for key, bt in BLOCK_TYPES.items():
        out.append({"type": key, "label": bt["label"],
                    "default_weeks": bt["default_weeks"],
                    "focus": bt["focus"],
                    "indication": bt["indication"]})
    return out
