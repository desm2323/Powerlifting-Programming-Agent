"""
periodization.py — the block model (part of the "Knowledge base" box).

A general LLM has no model of where you are in a training cycle, so it can't
shift volume and intensity coherently over weeks. Here the phase is explicit
state, and each phase has defined set/rep/intensity targets. Moving through
accumulation -> strength -> peak -> taper is a small state machine.

Targets below are sensible defaults for an intermediate lifter. They are the
KNOBS you (or a coach) tune — not the numbers the agent invents at runtime.
"""

from __future__ import annotations

# Ordered sequence of a simple block (mesocycle).
PHASE_SEQUENCE = ["accumulation", "strength", "peak", "taper"]

# Per-phase targets for the main lift's top working set(s):
#   sets/reps          : prescribed work
#   intensity_pct      : fraction of TRAINING MAX for the top set
#   rpe_cap            : effort ceiling — exceeding it signals fatigue
#   weekly_set_target  : hard sets/week for that lift (drives volume tracking)
PHASES = {
    "accumulation": {"sets": 4, "reps": 6, "intensity_pct": 0.72, "rpe_cap": 8.0,
                     "weekly_set_target": 14, "focus": "build work capacity and volume"},
    "strength":     {"sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9.0,
                     "weekly_set_target": 11, "focus": "express strength at higher loads"},
    "peak":         {"sets": 3, "reps": 1, "intensity_pct": 0.92, "rpe_cap": 9.5,
                     "weekly_set_target": 7,  "focus": "sharpen near-maximal singles"},
    "taper":        {"sets": 2, "reps": 1, "intensity_pct": 0.85, "rpe_cap": 8.0,
                     "weekly_set_target": 4,  "focus": "shed fatigue, rehearse openers"},
}


def phase_prescription(phase: str) -> dict:
    """Return the prescription template for a phase."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    return dict(PHASES[phase])


def next_phase(phase: str) -> str | None:
    """The phase that follows `phase`, or None if the block is complete."""
    i = PHASE_SEQUENCE.index(phase)
    return PHASE_SEQUENCE[i + 1] if i + 1 < len(PHASE_SEQUENCE) else None
