"""
guardrails.py — the safety / policy-adherence layer (the "Guardrails" box).

Even with a good policy, an autonomous agent needs hard limits. These functions
run BEFORE any decision is committed. Two jobs:
  1. Keep the math sane (progression caps, forced deloads, volume ceilings).
  2. Keep the agent in its lane (pain / injury / nutrition -> defer to a human).

This maps directly to the "policy adherence" requirement from the deployment
lecture: the agent must obey rules regardless of what the reasoning layer wants.
"""

from __future__ import annotations

from .landmarks import volume_status

# Max training-max increase allowed per completed cycle, by lift (kg).
TM_INCREASE_CAP = {"squat": 5.0, "bench": 2.5, "deadlift": 5.0}

# Free-text signals that take the agent out of scope.
PAIN_TERMS = ("pain", "sharp", "tweak", "tweaked", "pop", "popped", "injury",
              "injured", "strain", "pinch", "numb", "hurt")
NUTRITION_TERMS = ("diet", "calorie", "calories", "cut", "bulk", "lose weight",
                   "weight loss", "macros", "fasting")


def cap_tm_increase(lift: str, proposed_increase: float) -> float:
    """Clamp a proposed training-max bump to a safe per-cycle ceiling."""
    cap = TM_INCREASE_CAP.get(lift, 2.5)
    return min(proposed_increase, cap)


def must_deload(block: dict) -> bool:
    """Force a deload at the end of a phase's week count, or after overreach."""
    end_of_phase = block["week"] >= block["weeks_per_phase"]
    return end_of_phase or block.get("consecutive_overreach", 0) >= 2


def check_volume(lift: str, weekly_sets: int) -> tuple[bool, str]:
    """(ok, message). Blocks prescriptions that exceed MRV."""
    status = volume_status(lift, weekly_sets)
    if status == "above_mrv":
        return False, f"{weekly_sets} sets exceeds MRV for {lift}; capping volume."
    if status == "below_mev":
        return True, f"{weekly_sets} sets is below MEV for {lift}; consider more work."
    return True, ""


def scope_check(notes: str) -> dict:
    """
    Scan free-text notes for out-of-scope content.
    Returns {"in_scope": bool, "advisory": str|None}.
    """
    text = (notes or "").lower()
    if any(t in text for t in PAIN_TERMS):
        return {"in_scope": False,
                "advisory": ("Pain/injury mentioned. I won't push load on this — "
                             "please get this assessed by a physio or coach before "
                             "progressing.")}
    if any(t in text for t in NUTRITION_TERMS):
        return {"in_scope": False,
                "advisory": ("Nutrition is outside this tool's scope. For diet or "
                             "weight-management guidance, consult a registered "
                             "dietitian or your coach.")}
    return {"in_scope": True, "advisory": None}
