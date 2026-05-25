"""
readiness.py — fatigue/readiness state + the progression POLICY.

This is the heart of what general LLMs cannot do: fatigue is a hidden variable
that accumulates over sessions, and the decision to progress / hold / deload is
a function of that variable plus the latest session. We treat it as state.

Framing for later (ties to the RL half of the course):
    state  = (strength estimate, readiness)
    action = progress | hold | deload
    reward = strength gained without breaching the fatigue ceiling
The decide_progression() function below is a hand-written POLICY. Swapping it
for a learned contextual-bandit / RL policy is the headline v2 upgrade and does
NOT require touching anything else — see the roadmap in the README.
"""

from __future__ import annotations

# Readiness is a score in [0, 1]. 1.0 = fresh, primed to progress.
READINESS_FLOOR = 0.0
READINESS_CEIL = 1.0


def update_readiness(prev_score: float, session_eval: dict, rpe_cap: float) -> float:
    """
    Update the readiness score from one session's evaluation.

    session_eval keys used: effective_rpe (float), missed_reps (int).
    Going over the phase's RPE cap, or missing reps, drains readiness.
    A clean session under the cap lets it recover slightly.
    """
    score = prev_score
    rpe = session_eval.get("effective_rpe", rpe_cap)
    missed = session_eval.get("missed_reps", 0)

    overshoot = max(0.0, rpe - rpe_cap)
    score -= 0.12 * overshoot            # effort above plan = fatigue
    score -= 0.15 * missed               # missed reps = strong fatigue/limit signal
    if overshoot == 0 and missed == 0:
        score += 0.05                    # recovered a little

    return max(READINESS_FLOOR, min(READINESS_CEIL, round(score, 3)))


def decide_progression(readiness: float, session_eval: dict,
                       consecutive_overreach: int) -> str:
    """
    POLICY (v1, rule-based). Returns "progress" | "hold" | "deload".

    Order matters: safety/limit signals are checked before progress.
    """
    if session_eval.get("flags"):                 # pain/injury flagged -> never push
        return "hold"
    if readiness < 0.45 or consecutive_overreach >= 2:
        return "deload"
    if session_eval.get("missed_reps", 0) > 0:
        return "hold"
    if readiness >= 0.7 and session_eval.get("effective_rpe", 10) <= \
            session_eval.get("rpe_cap", 9):
        return "progress"
    return "hold"


def is_overreach(session_eval: dict, rpe_cap: float) -> bool:
    """Did this session run hotter than the phase intended?"""
    return session_eval.get("effective_rpe", 0) > rpe_cap or \
        session_eval.get("missed_reps", 0) > 0
