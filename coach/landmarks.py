"""
landmarks.py — weekly volume landmarks (part of the "Knowledge base" box).

MEV = minimum effective volume, MAV = maximum adaptive volume,
MRV = maximum recoverable volume (hard sets per lift per week).

These are the rails that stop the agent from prescribing unrecoverable volume —
a failure mode general LLMs fall into because they have no concept of a ceiling.
Values are illustrative defaults; expose them per-lifter later.
"""

from __future__ import annotations

LANDMARKS = {
    "squat":    {"mev": 8,  "mav": 14, "mrv": 18},
    "bench":    {"mev": 8,  "mav": 16, "mrv": 22},
    "deadlift": {"mev": 6,  "mav": 10, "mrv": 13},
}


def volume_status(lift: str, weekly_sets: int) -> str:
    """Classify a weekly set count against this lift's landmarks."""
    lm = LANDMARKS.get(lift)
    if lm is None:
        return "unknown"
    if weekly_sets < lm["mev"]:
        return "below_mev"        # too little to drive adaptation
    if weekly_sets > lm["mrv"]:
        return "above_mrv"        # likely unrecoverable
    if weekly_sets > lm["mav"]:
        return "high"             # productive but watch fatigue
    return "optimal"
