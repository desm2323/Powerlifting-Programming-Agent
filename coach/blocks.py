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


# --- RPE-anchored progression model -----------------------------------------
#
# "RPE dictates the weight on the bar, not the other way round." (Ben Johnson,
# Klt4SLA64iE @ 4:50). So the engine prescribes a rep + RPE target per week and
# the bar weight FOLLOWS from the lifter's current 1RM estimate (the anchor)
# via loading.load_for_rpe. This makes both within- and across-block
# progression real: as the anchor rises from logged performance, every load
# rises with it.
#
# TOP_SET_TRAJECTORY — the heavy top set climbs to a peak on the FINAL
# productive week. The trajectory is anchored to the last week and steps BACK
# one week at a time, so the peak always lands on the final week regardless of
# block length (a 4-week strength block reproduces Ben's RPE wave [6,7,8,9];
# a 5-week block gives [5,6,7,8,9]). Reps step DOWN toward a heavy single for
# strength/peaking (Ben's triples -> doubles -> singles), stay higher for
# volume/technique.
TOP_SET_TRAJECTORY = {
    "volume":    {"reps": 3, "rpe": 8.0, "rep_step": 1, "rpe_step": 0.5, "min_rpe": 6.0, "max_reps": 6},
    "technique": {"reps": 4, "rpe": 7.0, "rep_step": 0, "rpe_step": 0.5, "min_rpe": 5.5, "max_reps": 5},
    "strength":  {"reps": 1, "rpe": 9.0, "rep_step": 1, "rpe_step": 1.0, "min_rpe": 6.0, "max_reps": 5},
    "peaking":   {"reps": 1, "rpe": 9.0, "rep_step": 0, "rpe_step": 0.5, "min_rpe": 7.5, "max_reps": 2},
}

# BACKOFF_RATIO — the back-off load is a fixed fraction of THAT WEEK'S TOP-SET
# weight. So back-offs are ALWAYS lighter than the top set and climb WITH it
# but more slowly (the top set is the strength driver; back-offs are
# maintenance in a strength block / volume work in a volume block — they must
# never out-pace the top set). Ben's "back-off load static across the block"
# means static PERCENTAGES on a rising e1RM; we don't model e1RM, so a fixed
# fraction of the (climbing) top set reproduces the same intent simply.
# Volume back-offs sit relatively heaviest (more volume work); peaking lightest
# (volume is cut going into a meet).
BACKOFF_RATIO = {
    "volume": 0.85, "technique": 0.82, "strength": 0.78, "peaking": 0.75,
}

WEEK_LABELS = {
    "volume":    ["Establish", "Build", "Develop", "Push", "Peak volume", "Overreach"],
    "technique": ["Slow & strict", "Consolidate", "Re-load", "Express", "Confirm", "Refine"],
    "strength":  ["Re-introduce", "Build", "Strain", "Heavy", "Top sets", "Peak strain"],
    "peaking":   ["Open heavy", "Build-up", "Second attempt", "Third attempt", "Taper", "Openers / meet"],
}


def _week_label(block_type: str, week: int) -> str:
    labels = WEEK_LABELS.get(block_type, [])
    if not labels:
        return ""
    return labels[max(0, min(week - 1, len(labels) - 1))]


def week_modifier(block_type: str, week_index: int) -> dict:
    """Display label for week N (1-indexed). Loads are computed by
    compute_exercise_load via the RPE trajectory, not a multiplier — this
    only supplies the human-readable week name for the UI."""
    return {"label": _week_label(block_type, week_index)}


def _productive_weeks(block_type: str, duration_weeks: int | None) -> int:
    return int(duration_weeks) if duration_weeks else BLOCK_TYPES[block_type]["default_weeks"]


def _readiness_shave(readiness: float) -> float:
    """Translate fatigue into an RPE reduction. Readiness 1.0 -> no shave;
    lower readiness pulls the prescribed top-set RPE down a notch so the bar
    weight follows (decision: readiness modulates RPE, not kg directly)."""
    return round(max(0.0, 1.0 - readiness) / 0.25) * 0.5


def top_set_for_week(block_type: str, week: int,
                     duration_weeks: int | None = None,
                     readiness: float = 1.0) -> tuple[int, float]:
    """(reps, rpe) for the heavy top set on week N. Peaks on the final
    productive week; earlier weeks ramp up to it; readiness can shave it down."""
    spec = TOP_SET_TRAJECTORY[block_type]
    n = _productive_weeks(block_type, duration_weeks)
    weeks_from_end = max(0, n - max(1, min(week, n)))
    reps = min(spec["max_reps"], spec["reps"] + spec["rep_step"] * weeks_from_end)
    base_rpe = max(spec["min_rpe"], spec["rpe"] - spec["rpe_step"] * weeks_from_end)
    rpe = max(5.0, base_rpe - _readiness_shave(readiness))
    return int(reps), round(rpe, 1)


def backoff_load_for_week(block_type: str, week: int,
                          anchor_1rm: float | None,
                          duration_weeks: int | None = None,
                          readiness: float = 1.0) -> float | None:
    """Back-off WEIGHT for week N: a fixed fraction of that week's top-set
    weight, so it is always lighter than the top set and climbs more slowly.
    Returns None when there's no anchor."""
    if not anchor_1rm:
        return None
    reps, rpe = top_set_for_week(block_type, week, duration_weeks, readiness)
    top_weight = loading.load_for_rpe(anchor_1rm, reps, rpe)
    return loading.round_to_increment(top_weight * BACKOFF_RATIO[block_type])


def _implied_rpe(anchor_1rm: float | None, weight: float | None,
                 reps: int) -> float:
    """Approximate the RPE that `weight` x `reps` represents against the
    anchor (the inverse of load_for_rpe). Back-offs come out low — fast bar
    speed with reps in reserve, exactly Ben's back-off intent. Clamped to a
    sane display range."""
    if not anchor_1rm or not weight or weight <= 0:
        return 6.0
    reps_to_failure = 30 * (anchor_1rm / weight - 1)
    rir = reps_to_failure - reps
    return max(5.0, min(8.5, round((10 - rir) * 2) / 2))


def compute_exercise_load(exercise: dict, training_max: float | None,
                          block_type: str, week: int,
                          in_deload: bool = False,
                          duration_weeks: int | None = None,
                          readiness: float = 1.0) -> dict:
    """Produce concrete numbers (sets/reps/top_weight) for an exercise on a
    given week. Pure function — no state writes. `training_max` is the
    lifter's working-1RM anchor; the displayed kg comes from it via the RPE
    trajectory (loading.load_for_rpe), so loads track current strength.

    Role-aware behaviour (grounded in the tri-source KB):
      primary/secondary TOP SET (1 set): climbs to the block's terminal RPE
        on the final productive week — a genuinely heavy finish.
      primary/secondary BACK-OFF (>1 set): climbs in volume/technique blocks
        (Sebastian/Bromley), held static in strength/peaking (Ben).
      accessory: RPE-only when non-barbell (is_rpe_only_accessory), else a
        %-of-anchor load. Sets drop by 1 in deload; reps + RPE preserved.

    `exercise` shape: {name, lift, role, sets, reps, intensity_pct, rpe_cap, ...}.
    """
    role = exercise.get("role", "primary")
    is_main = role in ("primary", "secondary")
    name = exercise.get("name", "")
    label = "Deload" if in_deload else _week_label(block_type, week)

    # --- Accessories: RPE-only (non-barbell) or %-of-anchor (barbell) --------
    if not is_main:
        rpe_only = (is_rpe_only_accessory(name)
                    or (exercise.get("intensity_pct") or 0.0) <= 0.01)
        sets = max(1, exercise.get("sets", 3) - 1) if in_deload else exercise.get("sets", 3)
        reps = exercise.get("reps", 10)
        rpe_cap = exercise.get("rpe_cap", 8.5)
        if rpe_only or not training_max:
            weight, intensity_pct = None, 0.0
        else:
            intensity_pct = max(0.0, min(1.0, exercise.get("intensity_pct", 0.0)))
            weight = (loading.load_for_intensity(training_max, intensity_pct)
                      if intensity_pct > 0.01 else None)
        return {"sets": sets, "reps": reps, "intensity_pct": round(intensity_pct, 3),
                "rpe_cap": rpe_cap, "top_weight": weight, "week_label": label}

    # --- Main lifts: deload override ----------------------------------------
    # Deload is deliberately light recovery work, so it stays %-based (a fixed
    # fraction of the anchor) rather than an RPE target — an RPE-derived load
    # at the deload rep/RPE would land too heavy to actually deload.
    if in_deload:
        dl = deload_prescription(block_type)
        weight = (loading.load_for_intensity(training_max, dl["intensity_pct"])
                  if training_max else None)
        return {"sets": dl["sets"], "reps": dl["reps"],
                "intensity_pct": dl["intensity_pct"], "rpe_cap": dl["rpe_cap"],
                "top_weight": weight, "week_label": "Deload"}

    # --- Main lifts: productive week, RPE-anchored --------------------------
    # A single-set primary/secondary entry is the heavy TOP SET; multi-set
    # entries are the back-off / volume work.
    if exercise.get("sets", 1) <= 1:
        # TOP SET: the heavy driver — RPE-anchored, climbs to the block's peak.
        reps, rpe = top_set_for_week(block_type, week, duration_weeks, readiness)
        sets = 1
        weight = loading.load_for_rpe(training_max, reps, rpe) if training_max else None
    else:
        # BACK-OFF: a fraction of that week's top set — always lighter than the
        # top set, and climbs with it but more slowly. RPE shown is implied
        # from the load (comes out low: fast bar speed, reps in reserve).
        reps = max(1, exercise.get("reps", 5))
        sets = exercise.get("sets", 4)
        weight = backoff_load_for_week(block_type, week, training_max,
                                       duration_weeks, readiness)
        rpe = _implied_rpe(training_max, weight, reps)
    intensity_pct = round(weight / training_max, 3) if (weight and training_max) else 0.0
    return {"sets": sets, "reps": reps, "intensity_pct": intensity_pct,
            "rpe_cap": round(rpe, 1), "top_weight": weight, "week_label": label}


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
