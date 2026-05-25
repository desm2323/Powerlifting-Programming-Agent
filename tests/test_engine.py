"""Tests for the deterministic engine. These are 'verifiable tasks' — exact
oracle answers — which is the cheap, reliable kind of eval to build first."""

from coach import loading, readiness


def test_round_to_increment():
    assert loading.round_to_increment(101.0) == 100.0
    assert loading.round_to_increment(101.3, 1.25) == 101.25


def test_epley_1rm():
    # 100kg x 5 -> 100 * (1 + 5/30) ~= 116.67
    assert round(loading.epley_1rm(100, 5), 2) == 116.67


def test_est_1rm_from_rpe():
    # 100x3 @ RPE 9 (1 RIR) behaves like a 4-rep-to-failure set
    rpe = loading.est_1rm_from_rpe(100, 3, 9)
    failure = loading.epley_1rm(100, 4)
    assert abs(rpe - failure) < 1e-6


def test_load_for_intensity():
    assert loading.load_for_intensity(150, 0.85) == 127.5


def test_plate_breakdown():
    # 100kg on a 20kg bar -> 40kg/side -> 25 + 15
    assert loading.plate_breakdown(100) == [25, 15]
    assert loading.plate_breakdown(20) == []


def test_readiness_drains_on_overshoot():
    ev = {"effective_rpe": 10, "missed_reps": 1}
    after = readiness.update_readiness(1.0, ev, rpe_cap=9.0)
    assert after < 1.0


def test_policy_deloads_when_fatigued():
    ev = {"effective_rpe": 10, "missed_reps": 0, "rpe_cap": 9}
    assert readiness.decide_progression(0.3, ev, consecutive_overreach=0) == "deload"


def test_policy_progresses_when_fresh():
    ev = {"effective_rpe": 8, "missed_reps": 0, "rpe_cap": 9}
    assert readiness.decide_progression(0.9, ev, consecutive_overreach=0) == "progress"


def test_pain_flag_blocks_progress():
    ev = {"effective_rpe": 7, "missed_reps": 0, "rpe_cap": 9, "flags": ["pain"]}
    assert readiness.decide_progression(0.9, ev, consecutive_overreach=0) == "hold"
