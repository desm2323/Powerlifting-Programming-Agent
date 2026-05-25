"""
loading.py — deterministic strength math (the "Tools" box in the architecture).

This is the part a general-purpose LLM gets wrong: it guesses weights from
text patterns instead of computing them from the lifter's actual numbers.
Here, every weight comes from a formula grounded in the lifter's training max.
The LLM never picks a number.
"""

from __future__ import annotations


def round_to_increment(weight: float, increment: float = 2.5) -> float:
    """Round a load to the nearest loadable increment (default 2.5 kg)."""
    return round(weight / increment) * increment


def epley_1rm(weight: float, reps: int) -> float:
    """Estimate a one-rep max from a set taken near failure (Epley)."""
    return weight * (1 + reps / 30)


def brzycki_1rm(weight: float, reps: int) -> float:
    """Estimate a one-rep max (Brzycki). Diverges from Epley at high reps."""
    if reps >= 37:
        raise ValueError("Brzycki is undefined at 37+ reps")
    return weight * 36 / (37 - reps)


def est_1rm_from_rpe(weight: float, reps: int, rpe: float) -> float:
    """
    Estimate 1RM from a set that was NOT taken to failure, using RPE.

    RPE 10 = 0 reps in reserve, RPE 9 = 1 RIR, etc. We convert the set into an
    equivalent "reps to failure" and feed that to Epley. This is the bridge
    between how lifters actually report effort and a usable 1RM estimate.
    """
    reps_in_reserve = max(0.0, 10 - rpe)
    reps_to_failure = reps + reps_in_reserve
    return epley_1rm(weight, reps_to_failure)


def training_max(one_rm: float, fraction: float = 0.9) -> float:
    """Working training max — a sustainable % of true 1RM (5/3/1 uses ~90%)."""
    return round_to_increment(one_rm * fraction)


def load_for_intensity(tm: float, intensity_pct: float, increment: float = 2.5) -> float:
    """Convert a %-of-training-max target into a real, loadable barbell weight."""
    return round_to_increment(tm * intensity_pct, increment)


def plate_breakdown(target: float, bar: float = 20.0,
                    available=(25, 20, 15, 10, 5, 2.5, 1.25)) -> list[float]:
    """
    Plates needed PER SIDE to reach `target` (kg). Returns [] if target <= bar.
    Greedy from heaviest plate down — fine for standard kg plate sets.
    """
    per_side = (target - bar) / 2
    if per_side <= 0:
        return []
    plates: list[float] = []
    remaining = per_side
    for p in available:
        while remaining >= p - 1e-9:
            plates.append(p)
            remaining -= p
    return plates
