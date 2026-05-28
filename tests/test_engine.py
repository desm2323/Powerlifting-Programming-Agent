"""Tests for the deterministic engine. These are 'verifiable tasks' — exact
oracle answers — which is the cheap, reliable kind of eval to build first."""

import os
import tempfile

from datetime import date

from coach import (accessories, blocks, expert_knowledge, guardrails, loading,
                   memory, readiness, reasoning)
from coach.chat import parse_intent
from coach.loop import Agent, _compute_start_date, _earliest_start_from_history


def _new_agent(tmp_path: str, *, with_block: bool = False) -> Agent:
    """Helper — init a fresh agent with maxes, optionally commit a default
    volume block so engine paths that require an active block work."""
    agent = Agent(tmp_path)
    agent.init_program("Test", {"squat": 150, "bench": 100, "deadlift": 190},
                       bodyweight_kg=85, experience="intermediate")
    if with_block:
        agent.commit_block("volume", 4, "test block", ["squat"])
    return agent


def _acc(name: str, lift: str, intensity_pct: float = 0.0,
         rpe_cap: float = 9) -> dict:
    """Minimal accessory entry for test fixtures. Default RPE-only."""
    return {"name": name, "lift": lift, "role": "accessory",
            "sets": 3, "reps": 10,
            "intensity_pct": intensity_pct, "rpe_cap": rpe_cap}


def _generic_accessories(lift: str) -> list[dict]:
    """Two generic accessories appropriate for the given lift's day.
    Used by tests that need a plan to pass the validator's ≥2-accessory
    rule but don't care about the specific accessory choices."""
    picks = {
        "squat": ["Leg press", "Leg curl"],
        "bench": ["DB row", "Tricep pushdown"],
        "deadlift": ["Barbell row", "Back extension"],
    }
    return [_acc(name, lift) for name in picks.get(lift, picks["bench"])]


# --- loading.py: deterministic math ----------------------------------------

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


# --- readiness.py: fatigue model + policy ----------------------------------

def test_readiness_drains_on_overshoot():
    ev = {"effective_rpe": 10, "missed_reps": 1}
    after = readiness.update_readiness(1.0, ev, rpe_cap=9.0)
    assert after < 1.0


def test_readiness_recovers_on_clean_session():
    ev = {"effective_rpe": 7, "missed_reps": 0}
    after = readiness.update_readiness(0.7, ev, rpe_cap=8.0)
    assert after > 0.7


def test_readiness_clamped_to_ceiling():
    ev = {"effective_rpe": 6, "missed_reps": 0}
    after = readiness.update_readiness(0.98, ev, rpe_cap=8.0)
    assert after <= 1.0


def test_policy_deloads_when_fatigued():
    ev = {"effective_rpe": 10, "missed_reps": 0, "rpe_cap": 9}
    assert readiness.decide_progression(0.3, ev, consecutive_overreach=0) == "deload"


def test_policy_progresses_when_fresh():
    ev = {"effective_rpe": 8, "missed_reps": 0, "rpe_cap": 9}
    assert readiness.decide_progression(0.9, ev, consecutive_overreach=0) == "progress"


def test_pain_flag_blocks_progress():
    ev = {"effective_rpe": 7, "missed_reps": 0, "rpe_cap": 9, "flags": ["pain"]}
    assert readiness.decide_progression(0.9, ev, consecutive_overreach=0) == "hold"


def test_is_overreach_detects_rpe_overshoot():
    assert readiness.is_overreach({"effective_rpe": 9.5, "missed_reps": 0},
                                  rpe_cap=8.0) is True


def test_is_overreach_detects_missed_reps():
    assert readiness.is_overreach({"effective_rpe": 7.5, "missed_reps": 1},
                                  rpe_cap=8.0) is True


def test_is_overreach_false_for_clean_session():
    assert readiness.is_overreach({"effective_rpe": 7.5, "missed_reps": 0},
                                  rpe_cap=8.0) is False


# --- guardrails.py: safety layer -------------------------------------------

def test_guardrails_cap_tm_increase_per_lift():
    assert guardrails.cap_tm_increase("squat", 10.0) == 5.0
    assert guardrails.cap_tm_increase("bench", 10.0) == 2.5
    assert guardrails.cap_tm_increase("deadlift", 10.0) == 5.0
    assert guardrails.cap_tm_increase("squat", 2.5) == 2.5


def test_guardrails_check_volume_blocks_above_mrv():
    ok, msg = guardrails.check_volume("deadlift", 18)
    assert ok is False
    assert "MRV" in msg


def test_guardrails_check_volume_warns_below_mev():
    ok, msg = guardrails.check_volume("squat", 4)
    assert ok is True
    assert "MEV" in msg


def test_guardrails_check_volume_optimal_is_silent():
    ok, msg = guardrails.check_volume("bench", 12)
    assert ok is True and msg == ""


def test_guardrails_scope_check_pain_is_out_of_scope():
    out = guardrails.scope_check("squatted heavy but my knee tweaked badly")
    assert out["in_scope"] is False
    assert "physio" in out["advisory"].lower()


def test_guardrails_scope_check_nutrition_is_out_of_scope():
    out = guardrails.scope_check("should I cut weight before the meet?")
    assert out["in_scope"] is False
    assert "dietitian" in out["advisory"].lower()


def test_guardrails_reject_primary_and_secondary_same_lift_same_day():
    """The cramming bug: a day with primary squat + secondary squat on the
    same session must be rejected."""
    plan = {"days": [
        {"label": "Cramming day", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            {"name": "Paused squat", "lift": "squat", "role": "secondary",
             "sets": 4, "reps": 5, "intensity_pct": 0.70, "rpe_cap": 8},
        ]},
        {"label": "Other day", "exercises": [
            {"name": "Bench", "lift": "bench", "role": "primary",
             "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
        ]},
    ]}
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "primary" in err.lower() and "secondary" in err.lower()
    assert "squat" in err.lower()


def test_guardrails_reject_too_many_deadlift_sessions():
    """Deadlift cap is 2/wk — 3 sessions must be rejected."""
    def _dl_day(label):
        return {"label": label, "exercises": [
            {"name": "Deadlift", "lift": "deadlift", "role": "primary",
             "sets": 3, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            *_generic_accessories("deadlift"),
        ]}
    plan = {"days": [_dl_day("Mon"), _dl_day("Wed"), _dl_day("Fri")]}
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "deadlift" in err.lower() and "2" in err


def test_guardrails_accept_clean_split():
    """A reasonable 4-day split must pass validation."""
    plan = {"days": [
        {"label": "Mon — Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            *_generic_accessories("squat")]},
        {"label": "Tue — Bench", "exercises": [
            {"name": "Bench", "lift": "bench", "role": "primary",
             "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            *_generic_accessories("bench")]},
        {"label": "Thu — Deadlift", "exercises": [
            {"name": "Deadlift", "lift": "deadlift", "role": "primary",
             "sets": 3, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            *_generic_accessories("deadlift")]},
        # Secondary days now require 2 role='secondary' entries (top set
        # + back-offs) — same two-entry pattern as primary days.
        {"label": "Sat — Bench Volume", "exercises": [
            {"name": "Tempo bench (top set)", "lift": "bench",
             "role": "secondary", "sets": 1, "reps": 5,
             "intensity_pct": 0.75, "rpe_cap": 7},
            {"name": "Tempo bench (back-offs)", "lift": "bench",
             "role": "secondary", "sets": 3, "reps": 6,
             "intensity_pct": 0.65, "rpe_cap": 6},
            *_generic_accessories("bench")]},
    ]}
    assert guardrails.validate_weekly_plan(plan) is None


def test_guardrails_reject_avoided_exercise_in_plan():
    """Planks / push-ups / burpees / ab wheel / BW lunges etc. are on
    the AVOID list and must be rejected at plan commit."""
    plan = {"days": [
        {"label": "Mon — Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            {"name": "Weighted plank", "lift": "squat", "role": "accessory",
             "sets": 3, "reps": 30, "intensity_pct": 0.30, "rpe_cap": 8}]},
        {"label": "Tue — Bench", "exercises": [
            {"name": "Bench", "lift": "bench", "role": "primary",
             "sets": 4, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8}]},
    ]}
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "weighted plank" in err.lower()
    assert "avoid list" in err.lower()


def test_guardrails_avoid_check_normalised():
    """Case + whitespace shouldn't bypass the avoid filter."""
    for bad in ("PLANK", "  Push-up  ", "Ab Wheel", "burpee"):
        plan = {"days": [
            {"label": "Day 1", "exercises": [
                {"name": "Squat", "lift": "squat", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
                {"name": bad, "lift": "squat", "role": "accessory",
                 "sets": 3, "reps": 10, "intensity_pct": 0.30, "rpe_cap": 8}]},
            {"label": "Day 2", "exercises": [
                {"name": "Bench", "lift": "bench", "role": "primary",
                 "sets": 4, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8}]},
        ]}
        err = guardrails.validate_weekly_plan(plan)
        assert err is not None, f"{bad!r} should have been rejected"


def test_commit_block_accepts_bare_list_weekly_plan():
    """The LLM sometimes sends weekly_plan as a flat list of days instead of
    {'days': [...]}. The agent must accept either shape without crashing."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        bare_list = [
            {"label": "Day 1 — Squat", "exercises": [
                {"name": "Squat", "lift": "squat", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
                *_generic_accessories("squat"),
            ]},
            {"label": "Day 2 — Bench", "exercises": [
                {"name": "Bench", "lift": "bench", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
                *_generic_accessories("bench"),
            ]},
        ]
        out = agent.commit_block("strength", 4, "test", [], weekly_plan=bare_list)
        assert "committed" in out
        assert len(agent.state["block"]["weekly_plan"]["days"]) == 2


def test_commit_block_rejects_bad_plan():
    """commit_block must surface the validator's error instead of writing
    state. The LLM's propose_block tool then sees the error and retries."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        bad_plan = {"days": [
            {"label": "Bad day", "exercises": [
                {"name": "Squat", "lift": "squat", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
                {"name": "Paused squat", "lift": "squat", "role": "secondary",
                 "sets": 3, "reps": 5, "intensity_pct": 0.70, "rpe_cap": 8},
            ]},
            {"label": "Bench day", "exercises": [
                {"name": "Bench", "lift": "bench", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9}]},
        ]}
        out = agent.commit_block("strength", 4, "test", [], weekly_plan=bad_plan)
        assert "error" in out
        # State must be unchanged.
        assert agent.state["block"]["type"] is None


def test_guardrails_reject_stacked_primary_quality_variations():
    """Two bench-pattern primary-quality variations on one day (tempo
    bench + close-grip bench) double the main-lift fatigue and must be
    rejected."""
    plan = {"days": [
        {"label": "Mon — Heavy Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 1, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            {"name": "Leg press", "lift": "squat", "role": "accessory",
             "sets": 3, "reps": 8, "intensity_pct": 0.0, "rpe_cap": 8},
            {"name": "RDL", "lift": "deadlift", "role": "accessory",
             "sets": 3, "reps": 8, "intensity_pct": 0.60, "rpe_cap": 8}]},
        {"label": "Fri — Secondary Bench", "exercises": [
            {"name": "Tempo bench (3-1-0)", "lift": "bench",
             "role": "secondary", "sets": 4, "reps": 5,
             "intensity_pct": 0.65, "rpe_cap": 7},
            {"name": "Close-grip bench", "lift": "bench",
             "role": "accessory", "sets": 3, "reps": 6,
             "intensity_pct": 0.70, "rpe_cap": 7}]},
    ]}
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "primary-quality" in err.lower()
    assert "bench" in err.lower()
    # Names of both offending exercises should appear in the error.
    assert "tempo" in err.lower()
    assert "close-grip" in err.lower() or "close grip" in err.lower()


def test_guardrails_reject_paused_squat_plus_pin_squat_same_day():
    """Both are squat-pattern primary-quality. Stacking = engine reject."""
    plan = {"days": [
        {"label": "Mon — Squat", "exercises": [
            {"name": "Paused squat", "lift": "squat", "role": "secondary",
             "sets": 3, "reps": 5, "intensity_pct": 0.70, "rpe_cap": 7},
            {"name": "Pin squat", "lift": "squat", "role": "accessory",
             "sets": 3, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8}]},
        {"label": "Thu — Bench", "exercises": [
            {"name": "Bench", "lift": "bench", "role": "primary",
             "sets": 4, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8},
            {"name": "DB row", "lift": "bench", "role": "accessory",
             "sets": 3, "reps": 10, "intensity_pct": 0.0, "rpe_cap": 9},
            {"name": "Tricep pushdown", "lift": "bench",
             "role": "accessory", "sets": 3, "reps": 12,
             "intensity_pct": 0.0, "rpe_cap": 9}]},
    ]}
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "squat" in err.lower()


def test_guardrails_reject_zero_accessories_on_primary_day():
    """A primary day with only the top set + back-offs and no accessories
    falls below the minimum accessory count and must be rejected."""
    plan = {"days": [
        {"label": "Mon — Heavy Squat", "exercises": [
            {"name": "Squat (top set)", "lift": "squat", "role": "primary",
             "sets": 1, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 8},
            {"name": "Squat (back-offs)", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 5, "intensity_pct": 0.68, "rpe_cap": 7}]},
        {"label": "Tue — Heavy Bench", "exercises": [
            {"name": "Bench (top set)", "lift": "bench", "role": "primary",
             "sets": 1, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 8},
            {"name": "DB row", "lift": "bench", "role": "accessory",
             "sets": 3, "reps": 10, "intensity_pct": 0.0, "rpe_cap": 9},
            {"name": "Tricep pushdown", "lift": "bench",
             "role": "accessory", "sets": 3, "reps": 12,
             "intensity_pct": 0.0, "rpe_cap": 9}]},
    ]}
    # Block-type defaults to None → non-peaking threshold (≥2 accessories).
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "accessor" in err.lower()
    # Specifically mentions day 1 (the squat day with zero accessories).
    assert "day 1" in err.lower() or "heavy squat" in err.lower()


def test_guardrails_reject_one_accessory_on_secondary_day():
    """A secondary day with only one accessory falls below the ≥2
    minimum for non-peaking blocks. The secondary movement here has a
    valid 2-entry top set + back-offs so the accessory-count rule is
    what's being exercised, not the secondary-entry-count rule."""
    plan = {"days": [
        {"label": "Mon — Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8},
            {"name": "RDL", "lift": "deadlift", "role": "accessory",
             "sets": 3, "reps": 8, "intensity_pct": 0.60, "rpe_cap": 8},
            {"name": "Leg curl", "lift": "squat", "role": "accessory",
             "sets": 3, "reps": 10, "intensity_pct": 0.0, "rpe_cap": 9}]},
        {"label": "Thu — Secondary Squat", "exercises": [
            {"name": "Paused squat (top set)", "lift": "squat",
             "role": "secondary", "sets": 1, "reps": 5,
             "intensity_pct": 0.70, "rpe_cap": 7},
            {"name": "Paused squat (back-offs)", "lift": "squat",
             "role": "secondary", "sets": 3, "reps": 5,
             "intensity_pct": 0.62, "rpe_cap": 6},
            {"name": "Leg extension", "lift": "squat", "role": "accessory",
             "sets": 3, "reps": 12, "intensity_pct": 0.0, "rpe_cap": 9}]},
    ]}
    err = guardrails.validate_weekly_plan(plan, block_type="volume")
    assert err is not None
    assert "accessor" in err.lower()


def test_guardrails_peaking_allows_minimum_one_accessory():
    """Bromley's peaking rule: drop high-volume isolation. Engine relaxes
    the minimum to 1 accessory for peaking blocks."""
    plan = {"days": [
        {"label": "Mon — Squat singles", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 3, "reps": 1, "intensity_pct": 0.90, "rpe_cap": 9},
            {"name": "Leg curl", "lift": "squat", "role": "accessory",
             "sets": 2, "reps": 12, "intensity_pct": 0.0, "rpe_cap": 8}]},
        {"label": "Thu — Bench singles", "exercises": [
            {"name": "Bench", "lift": "bench", "role": "primary",
             "sets": 3, "reps": 1, "intensity_pct": 0.90, "rpe_cap": 9},
            {"name": "Tricep pushdown", "lift": "bench",
             "role": "accessory", "sets": 2, "reps": 12,
             "intensity_pct": 0.0, "rpe_cap": 8}]},
    ]}
    # With block_type='peaking', 1 accessory per day is acceptable.
    assert guardrails.validate_weekly_plan(
        plan, block_type="peaking") is None
    # But the SAME plan as a volume block would be rejected.
    err = guardrails.validate_weekly_plan(plan, block_type="volume")
    assert err is not None and "accessor" in err.lower()


def test_guardrails_accept_clean_3_accessory_primary_day():
    """Sanity: a well-formed day with main + 3 accessories passes."""
    plan = {"days": [
        {"label": "Mon — Heavy Squat", "exercises": [
            {"name": "Squat (top set)", "lift": "squat", "role": "primary",
             "sets": 1, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 8},
            {"name": "Squat (back-offs)", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 5, "intensity_pct": 0.68, "rpe_cap": 7},
            {"name": "Leg press", "lift": "squat", "role": "accessory",
             "sets": 3, "reps": 8, "intensity_pct": 0.0, "rpe_cap": 8},
            {"name": "Leg curl", "lift": "squat", "role": "accessory",
             "sets": 3, "reps": 12, "intensity_pct": 0.0, "rpe_cap": 9},
            {"name": "Hanging leg raise", "lift": "squat",
             "role": "accessory", "sets": 3, "reps": 12,
             "intensity_pct": 0.0, "rpe_cap": 9}]},
        {"label": "Tue — Bench", "exercises": [
            {"name": "Bench", "lift": "bench", "role": "primary",
             "sets": 4, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8},
            {"name": "DB row", "lift": "bench", "role": "accessory",
             "sets": 3, "reps": 10, "intensity_pct": 0.0, "rpe_cap": 9},
            {"name": "Tricep pushdown", "lift": "bench",
             "role": "accessory", "sets": 3, "reps": 12,
             "intensity_pct": 0.0, "rpe_cap": 9}]},
    ]}
    assert guardrails.validate_weekly_plan(plan) is None


def test_pq_variation_dedup_allows_top_set_plus_backoff_of_same_variation():
    """Two entries of the SAME primary-quality variation (Ben's top set
    + back-offs naming pattern) are FINE — they match the same token and
    are allowed. Different variations stacked = still rejected."""
    plan = {"days": [
        {"label": "Mon — Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8},
            *_generic_accessories("squat")]},
        {"label": "Thu — Secondary Bench", "exercises": [
            # Top set + back-offs of the SAME variation — allowed.
            {"name": "Tempo bench (top set)", "lift": "bench",
             "role": "secondary", "sets": 1, "reps": 5,
             "intensity_pct": 0.75, "rpe_cap": 7},
            {"name": "Tempo bench (back-offs)", "lift": "bench",
             "role": "secondary", "sets": 3, "reps": 6,
             "intensity_pct": 0.65, "rpe_cap": 6},
            *_generic_accessories("bench")]},
    ]}
    assert guardrails.validate_weekly_plan(plan) is None


def test_pq_variation_dedup_still_rejects_different_variations():
    """Sanity: the dedup fix shouldn't accidentally allow stacking
    DIFFERENT primary-quality variations of the same pattern."""
    plan = {"days": [
        {"label": "Mon — Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8},
            *_generic_accessories("squat")]},
        {"label": "Thu — Secondary Bench", "exercises": [
            # DIFFERENT variations — still rejected.
            {"name": "Tempo bench", "lift": "bench", "role": "secondary",
             "sets": 4, "reps": 5, "intensity_pct": 0.65, "rpe_cap": 7},
            {"name": "Close-grip bench", "lift": "bench",
             "role": "accessory", "sets": 3, "reps": 6,
             "intensity_pct": 0.70, "rpe_cap": 7}]},
    ]}
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "different" in err.lower() or "tempo" in err.lower()


def test_validator_rejects_db_bench_as_primary():
    """A DB / cable / machine movement tagged role='primary' would have
    the engine compute a %TM load that's meaningless for a dumbbell.
    Such movements must be rejected from the primary/secondary slots so
    they stay accessories."""
    plan = {"days": [
        {"label": "Mon — Bench", "exercises": [
            {"name": "DB bench press", "lift": "bench", "role": "primary",
             "sets": 4, "reps": 6, "intensity_pct": 0.57, "rpe_cap": 7},
            *_generic_accessories("bench")]},
        {"label": "Thu — Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            *_generic_accessories("squat")]},
    ]}
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "db bench" in err.lower()
    assert "primary" in err.lower() or "accessory" in err.lower()


def test_validator_rejects_cable_pushdown_as_secondary():
    """Same rule — cable / machine / isolation can't be role=secondary."""
    plan = {"days": [
        {"label": "Mon — Bench", "exercises": [
            {"name": "Bench", "lift": "bench", "role": "primary",
             "sets": 4, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8},
            *_generic_accessories("bench")]},
        {"label": "Thu — Bench Secondary", "exercises": [
            {"name": "Cable tricep pushdown", "lift": "bench",
             "role": "secondary", "sets": 4, "reps": 12,
             "intensity_pct": 0.40, "rpe_cap": 9},
            *_generic_accessories("bench")]},
    ]}
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "pushdown" in err.lower()


def test_validator_requires_secondary_backoff_for_non_technique():
    """Single-entry secondary fails for volume/strength/peaking blocks."""
    plan = {"days": [
        {"label": "Mon — Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8},
            *_generic_accessories("squat")]},
        {"label": "Thu — Bench Secondary", "exercises": [
            {"name": "Tempo bench", "lift": "bench", "role": "secondary",
             "sets": 4, "reps": 5, "intensity_pct": 0.65, "rpe_cap": 7},
            *_generic_accessories("bench")]},
    ]}
    for bt in ("volume", "strength", "peaking"):
        err = guardrails.validate_weekly_plan(plan, block_type=bt)
        assert err is not None, f"single-entry secondary should fail for {bt}"
        assert "back-off" in err.lower() or "secondary" in err.lower()


def test_validator_allows_single_secondary_in_technique_block():
    """Technique blocks are exempt — single moderate-volume secondary OK."""
    plan = {"days": [
        {"label": "Mon — Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 3, "reps": 5, "intensity_pct": 0.65, "rpe_cap": 7},
            *_generic_accessories("squat")]},
        {"label": "Thu — Bench Secondary (technique)", "exercises": [
            {"name": "Tempo bench", "lift": "bench", "role": "secondary",
             "sets": 3, "reps": 5, "intensity_pct": 0.60, "rpe_cap": 6},
            *_generic_accessories("bench")]},
    ]}
    assert guardrails.validate_weekly_plan(
        plan, block_type="technique") is None


def test_calendar_sort_reorders_rest_days_into_calendar_slot():
    """Rest days listed out of order get sorted into their calendar slot:
    [Mon, Tue, Thu, Fri, Sat, Wed-Rest, Sun-Rest] becomes
    [Mon, Tue, Wed, Thu, Fri, Sat, Sun]."""
    from coach.loop import _calendar_sort_days
    days = [
        {"label": "Monday — Squat"},
        {"label": "Tuesday — Bench"},
        {"label": "Thursday — Deadlift"},
        {"label": "Friday — Secondary Bench"},
        {"label": "Saturday — Secondary Squat"},
        {"label": "Wednesday — Rest"},
        {"label": "Sunday — Rest"},
    ]
    out = _calendar_sort_days(days)
    labels = [d["label"] for d in out]
    assert labels == [
        "Monday — Squat",
        "Tuesday — Bench",
        "Wednesday — Rest",
        "Thursday — Deadlift",
        "Friday — Secondary Bench",
        "Saturday — Secondary Squat",
        "Sunday — Rest",
    ]


def test_calendar_sort_unlabelled_days_go_last_keeping_order():
    """Days without an identifiable weekday in their label keep their
    relative order and go AFTER weekday-labelled days."""
    from coach.loop import _calendar_sort_days
    days = [
        {"label": "Day A — Squat"},
        {"label": "Monday — Bench"},
        {"label": "Day B — Deadlift"},
    ]
    out = _calendar_sort_days(days)
    labels = [d["label"] for d in out]
    assert labels == [
        "Monday — Bench",
        "Day A — Squat",
        "Day B — Deadlift",
    ]


def test_commit_block_sorts_days_by_calendar_order():
    """End-to-end: committing a plan with rest days appended at the end
    stores them in calendar order."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        plan = {"days": [
            {"label": "Monday — Squat", "exercises": [
                {"name": "Squat", "lift": "squat", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 8},
                *_generic_accessories("squat")]},
            {"label": "Tuesday — Bench", "exercises": [
                {"name": "Bench", "lift": "bench", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 8},
                *_generic_accessories("bench")]},
            {"label": "Thursday — Deadlift", "exercises": [
                {"name": "Deadlift", "lift": "deadlift", "role": "primary",
                 "sets": 3, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 8},
                *_generic_accessories("deadlift")]},
            # Rest days appended at the end — engine should reorder.
            {"label": "Wednesday — Rest", "exercises": []},
            {"label": "Sunday — Rest", "exercises": []},
        ]}
        agent.commit_block("strength", 4, "test", [], weekly_plan=plan)
        stored_labels = [d["label"]
                         for d in agent.state["block"]["weekly_plan"]["days"]]
        # Wednesday should now be between Tuesday and Thursday.
        wed_idx = stored_labels.index("Wednesday — Rest")
        tue_idx = stored_labels.index("Tuesday — Bench")
        thu_idx = stored_labels.index("Thursday — Deadlift")
        assert tue_idx < wed_idx < thu_idx


def test_pattern_of_primary_quality_variation_classifies_correctly():
    """Direct test of the classifier."""
    # Bench-pattern primary-quality variations
    for name in ("Tempo bench", "Close-grip bench", "Paused bench",
                 "Spoto press", "1.25 bench", "Long-pause bench"):
        assert expert_knowledge.pattern_of_primary_quality_variation(name) \
               == "bench_pattern", f"{name!r} should be bench_pattern"
    # Squat-pattern
    for name in ("Paused squat", "Pin squat", "Tempo squat",
                 "Anderson squat", "1.25 squat"):
        assert expert_knowledge.pattern_of_primary_quality_variation(name) \
               == "squat_pattern", f"{name!r} should be squat_pattern"
    # Deadlift-pattern
    for name in ("Deficit deadlift", "Paused deadlift",
                 "Snatch-grip deadlift"):
        assert expert_knowledge.pattern_of_primary_quality_variation(name) \
               == "deadlift_pattern", f"{name!r} should be deadlift_pattern"
    # Regular comp lifts are NOT primary-quality variations
    for name in ("Squat", "Bench", "Deadlift", "Bench press"):
        assert expert_knowledge.pattern_of_primary_quality_variation(name) \
               is None, f"{name!r} should NOT be a primary-quality variation"
    # True accessories are not primary-quality variations
    for name in ("DB bench", "Tricep pushdown", "Leg press",
                 "Romanian deadlift", "Barbell row"):
        assert expert_knowledge.pattern_of_primary_quality_variation(name) \
               is None, f"{name!r} should NOT be flagged"


def test_guardrails_accept_rest_day_with_empty_exercises():
    """Rest days (label like 'Wednesday — Rest', exercises=[]) are valid —
    the LLM uses them so the UI can render the full 7-day week with
    explicit rest markers instead of forcing the lifter to figure it out."""
    plan = {"days": [
        {"label": "Monday — Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            *_generic_accessories("squat")]},
        {"label": "Tuesday — Bench", "exercises": [
            {"name": "Bench", "lift": "bench", "role": "primary",
             "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            *_generic_accessories("bench")]},
        {"label": "Wednesday — Rest", "exercises": []},
        {"label": "Thursday — Deadlift", "exercises": [
            {"name": "Deadlift", "lift": "deadlift", "role": "primary",
             "sets": 3, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            *_generic_accessories("deadlift")]},
        {"label": "Sunday — Active Recovery", "exercises": []},
    ]}
    assert guardrails.validate_weekly_plan(plan) is None


def test_guardrails_reject_empty_day_without_rest_label():
    """An empty day with a normal training label is a bug — the LLM
    forgot exercises. Must reject with a hint about labelling it as rest."""
    plan = {"days": [
        {"label": "Monday — Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            *_generic_accessories("squat")]},
        {"label": "Tuesday — Bench Volume", "exercises": []},
    ]}
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "no exercises" in err.lower()
    assert "rest" in err.lower()  # hints at the fix


def test_guardrails_reject_all_rest_days():
    """A weekly_plan that's ALL rest days is invalid — need training days."""
    plan = {"days": [
        {"label": "Monday — Rest", "exercises": []},
        {"label": "Tuesday — Rest", "exercises": []},
    ]}
    err = guardrails.validate_weekly_plan(plan)
    assert err is not None
    assert "training" in err.lower()


def test_guardrails_rest_days_dont_count_toward_session_caps():
    """A plan with 2 DL training days + 1 rest day labelled 'Deload Day'
    must NOT trip the 'too many DL sessions' rule — rest days are skipped."""
    plan = {"days": [
        {"label": "Mon — DL", "exercises": [
            {"name": "Deadlift", "lift": "deadlift", "role": "primary",
             "sets": 3, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
            *_generic_accessories("deadlift")]},
        {"label": "Wed — DL secondary", "exercises": [
            # Two-entry secondary (top set + back-offs of same variation).
            {"name": "Paused deadlift (top set)", "lift": "deadlift",
             "role": "secondary", "sets": 1, "reps": 5,
             "intensity_pct": 0.70, "rpe_cap": 7},
            {"name": "Paused deadlift (back-offs)", "lift": "deadlift",
             "role": "secondary", "sets": 3, "reps": 5,
             "intensity_pct": 0.60, "rpe_cap": 6},
            *_generic_accessories("deadlift")]},
        {"label": "Fri — Deload Day", "exercises": []},
    ]}
    # 2 DL sessions + rest day = OK. (Rest day doesn't add a 3rd DL session.)
    assert guardrails.validate_weekly_plan(plan) is None


def test_is_rest_day_helper():
    """Direct test of the rest-day detector."""
    assert guardrails._is_rest_day({"label": "Wednesday — Rest", "exercises": []})
    assert guardrails._is_rest_day({"label": "Active Recovery", "exercises": []})
    assert guardrails._is_rest_day({"label": "Sunday — off day", "exercises": []})
    # Non-rest labels with empty exercises are NOT rest days (LLM bug).
    assert not guardrails._is_rest_day({"label": "Wednesday — Squat", "exercises": []})
    # Rest label with actual exercises = NOT a rest day (treat as light training).
    assert not guardrails._is_rest_day({
        "label": "Active Recovery", "exercises": [{"name": "Walk"}]
    })


def test_guardrails_scope_check_normal_session_is_in_scope():
    out = guardrails.scope_check("felt strong, opener was easy")
    assert out["in_scope"] is True
    assert out["advisory"] is None


# --- blocks.py: knowledge base ---------------------------------------------

def test_blocks_all_four_types_defined():
    """volume/technique/strength/peaking must all exist with deload templates."""
    for t in ("volume", "technique", "strength", "peaking"):
        assert t in blocks.BLOCK_TYPES
        rx = blocks.block_prescription(t)
        dl = blocks.deload_prescription(t)
        assert rx["rpe_cap"] >= dl["rpe_cap"]  # deload should be lighter effort
        assert dl["weekly_set_target"] < rx["weekly_set_target"]


def test_blocks_week_wave_differs_week_to_week():
    """Real coaching never has identical weeks — the wave must produce
    different intensity multipliers across the productive weeks of a block."""
    for block_type in ("volume", "strength", "peaking"):
        mults = [blocks.week_modifier(block_type, w)["intensity_mult"]
                 for w in range(1, 5)]
        assert len(set(mults)) > 1, f"{block_type} wave is flat"


def test_blocks_compute_exercise_load_applies_wave():
    """The exercise load helper must apply the block's wave to the base
    intensity. Week 4 of strength is heavier than week 1."""
    ex = {"name": "Squat", "lift": "squat", "role": "primary",
          "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9.0}
    wk1 = blocks.compute_exercise_load(ex, 150, "strength", 1)
    wk4 = blocks.compute_exercise_load(ex, 150, "strength", 4)
    assert wk4["top_weight"] > wk1["top_weight"]
    # Deload override produces the deload prescription, not the exercise's.
    dl = blocks.compute_exercise_load(ex, 150, "strength", 3, in_deload=True)
    assert dl["week_label"] == "Deload"
    assert dl["rpe_cap"] <= 7.5


def test_blocks_rpe_waves_across_block():
    """An exercise prescribed at RPE 8 should NOT stay at RPE 8 across
    all weeks. Ben's framework climbs RPE alongside intensity — early
    weeks lighter (lower RPE), late weeks heavier (higher RPE)."""
    ex = {"name": "Squat", "lift": "squat", "role": "primary",
          "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 8.0}
    rpes = [blocks.compute_exercise_load(ex, 150, "strength", w)["rpe_cap"]
            for w in range(1, 5)]
    # Must be strictly non-decreasing and end higher than it started.
    assert rpes[-1] > rpes[0]
    for a, b in zip(rpes, rpes[1:]):
        assert b >= a
    # Final week's RPE should be at or near the prescribed cap.
    assert rpes[-1] >= 8.0


def test_primary_with_zero_intensity_falls_back_to_block_default():
    """A primary lift arriving with intensity_pct=0 (RPE-only) would
    render 'use a weight that hits target RPE' instead of a calibrated
    weight. The engine substitutes the block's default intensity_pct so
    primary/secondary lifts always carry a real prescription."""
    primary = {"name": "Squat (top set)", "lift": "squat", "role": "primary",
               "sets": 1, "reps": 3, "intensity_pct": 0.0, "rpe_cap": 8.0}
    out = blocks.compute_exercise_load(primary, 150, "volume", 2)
    # MUST get a real weight — not None.
    assert out["top_weight"] is not None
    # Should be in the ballpark of the volume block default (0.72 × wave).
    assert 80 <= out["top_weight"] <= 130


def test_secondary_with_zero_intensity_also_falls_back():
    """Same safety net applies to role='secondary' (paused squat, tempo
    bench, deficit DL etc.) — they're main-lift work and need real %TM."""
    secondary = {"name": "Paused squat", "lift": "squat", "role": "secondary",
                 "sets": 3, "reps": 5, "intensity_pct": 0.0, "rpe_cap": 7.0}
    out = blocks.compute_exercise_load(secondary, 150, "strength", 2)
    assert out["top_weight"] is not None


def test_primary_with_normal_intensity_unchanged():
    """Sanity: the safety net only fires when intensity_pct is impossibly
    low. Normal values (0.55-0.94) pass through untouched."""
    primary = {"name": "Squat", "lift": "squat", "role": "primary",
               "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9.0}
    out = blocks.compute_exercise_load(primary, 150, "strength", 2)
    # Should NOT be the block default — should reflect 0.85 × wave.
    # Strength wave week 2 intensity_mult = 0.98 → 0.833 → 125 kg-ish.
    assert out["top_weight"] is not None
    assert 120 <= out["top_weight"] <= 130


def test_accessory_with_zero_intensity_stays_rpe_only():
    """The new primary-safety-net must NOT bleed into accessories — an
    accessory with intensity_pct=0 is the CORRECT encoding for RPE-only."""
    accessory = {"name": "DB lateral raise", "lift": "bench",
                 "role": "accessory",
                 "sets": 3, "reps": 12, "intensity_pct": 0.0, "rpe_cap": 9.0}
    out = blocks.compute_exercise_load(accessory, 100, "volume", 2)
    # Stays None — accessory RPE-only behavior preserved.
    assert out["top_weight"] is None


def test_peaking_block_rpe_caps_under_failure():
    """Ben: NEVER train main lifts to failure (zrOWMftnvQw @ 4:35,
    n9Wyxr4Yks0 @ 7:55). Peaking's final-week wave (+0.5) must keep the
    top set at OR BELOW RPE 9.5 — i.e. base rpe_cap must be <= 9.0."""
    base = blocks.BLOCK_TYPES["peaking"]
    assert base["rpe_cap"] <= 9.0
    # Final-week wave for peaking is +0.5 → peak RPE never exceeds 9.5.
    peak_ex = {"name": "Squat", "lift": "squat", "role": "primary",
               "sets": 1, "reps": 1, "intensity_pct": 0.92,
               "rpe_cap": base["rpe_cap"]}
    last_week_rpe = blocks.compute_exercise_load(
        peak_ex, 150, "peaking", 3)["rpe_cap"]
    assert last_week_rpe <= 9.5


# --- earliest start: archived block's expected end is respected -----------

def test_earliest_start_from_history_returns_none_without_history():
    state = {"block_history": []}
    assert _earliest_start_from_history(state) is None


def test_earliest_start_from_history_computes_expected_end():
    """productive weeks + 1 deload week from started_at."""
    state = {"block_history": [
        {"started_at": "2026-06-01", "duration_weeks": 4},
    ]}
    end = _earliest_start_from_history(state)
    # June 1 (Mon) + 5 weeks = July 6 (Mon)
    assert end == date(2026, 7, 6)


def test_compute_start_date_anchors_to_earliest():
    """Today is May 27 (Wed) but the previous block hadn't ended yet
    (expected end July 6). Next Monday from July 6 = July 6 itself."""
    today = date(2026, 5, 27)
    earliest = date(2026, 7, 6)
    plan = {"days": [{"label": "Monday — Heavy Squat", "exercises": []}]}
    start = _compute_start_date(plan, today, earliest=earliest)
    assert start == date(2026, 7, 6)
    assert start.strftime("%A") == "Monday"


def test_compute_start_date_earliest_ignored_when_in_past():
    """If earliest is in the past, use today as the anchor."""
    today = date(2026, 5, 27)
    earliest = date(2026, 1, 1)  # past
    plan = {"days": [{"label": "Monday — Heavy Squat", "exercises": []}]}
    start = _compute_start_date(plan, today, earliest=earliest)
    # Falls back to next-Monday from today = June 1.
    assert start == date(2026, 6, 1)


def test_commit_block_after_archive_uses_previous_block_expected_end():
    """After archiving a volume block (started June 1, 4wk), the next
    block's start date is computed from the previous block's expected
    end (July 6), not from today's next Monday."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        # Force the volume block to have started June 1, 2026 (hypothetical)
        agent.state["block"]["started_at"] = "2026-06-01"
        agent.state["block"]["duration_weeks"] = 4
        agent.archive_current_block("done well")
        # Now commit a strength block. Today is whatever the test runs on;
        # what we care about is: started_at >= July 6 (volume's exp. end).
        plan = {"days": [
            {"label": "Monday — Heavy Squat", "exercises": [
                {"name": "Squat", "lift": "squat", "role": "primary",
                 "sets": 1, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 8},
                *_generic_accessories("squat")]},
            {"label": "Thursday — Heavy Bench", "exercises": [
                {"name": "Bench", "lift": "bench", "role": "primary",
                 "sets": 1, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 8},
                *_generic_accessories("bench")]},
        ]}
        agent.commit_block("strength", 4, "express force", [], weekly_plan=plan)
        started = date.fromisoformat(agent.state["block"]["started_at"])
        # Must be at least the day after July 6 (volume's exp. end) AND
        # land on a Monday.
        assert started >= date(2026, 7, 6)
        assert started.strftime("%A") == "Monday"


def test_blocks_rpe_clamped_to_safe_range():
    """rpe_cap must never exceed 10 (= failure) or drop below 5."""
    high_ex = {"name": "Squat", "lift": "squat", "role": "primary",
               "sets": 4, "reps": 3, "intensity_pct": 0.90, "rpe_cap": 9.8}
    # Peak strength week adds rpe_delta=+0.5; 9.8+0.5=10.3 must clamp to 10.
    out = blocks.compute_exercise_load(high_ex, 150, "strength", 4)
    assert out["rpe_cap"] <= 10.0

    low_ex = {"name": "Squat", "lift": "squat", "role": "primary",
              "sets": 4, "reps": 6, "intensity_pct": 0.65, "rpe_cap": 5.5}
    # Volume wk1 has rpe_delta=-1.0; 5.5-1.0=4.5 must clamp to 5.0.
    out = blocks.compute_exercise_load(low_ex, 150, "volume", 1)
    assert out["rpe_cap"] >= 5.0


def test_blocks_rpe_wave_skips_accessories():
    """RPE wave applies ONLY to primary/secondary work — accessories
    keep their prescribed RPE (Ben's hypertrophy work stays close to
    failure across the block)."""
    acc = {"name": "DB lateral raise", "lift": "bench", "role": "accessory",
           "sets": 3, "reps": 12, "intensity_pct": 0.0, "rpe_cap": 9.0}
    for w in range(1, 5):
        out = blocks.compute_exercise_load(acc, 100, "strength", w)
        assert out["rpe_cap"] == 9.0


def test_blocks_accessory_preserves_prescribed_sets_reps():
    """The wave (set_delta, rep_delta, intensity_mult) must apply ONLY to
    primary/secondary main-lift work. An accessory's prescribed sets/reps
    must survive any productive week unchanged — Ben's KB is explicit that
    hypertrophy accessories stay 2-3x4-10 RPE 8-10 across the block.

    Uses a barbell accessory (RDL) here so the wave-scoping is the only
    behavior under test — for DB / cable / machine accessories the RPE-only
    normalizer additionally collapses intensity_pct to 0; that path has its
    own tests below."""
    accessory = {"name": "Romanian deadlift", "lift": "deadlift",
                 "role": "accessory",
                 "sets": 3, "reps": 8, "intensity_pct": 0.60, "rpe_cap": 8.0}
    # Strength block week 4 ("Top sets") has set_delta=-1, rep_delta=-1,
    # intensity_mult=1.04 — none of these should touch the RDL.
    out = blocks.compute_exercise_load(accessory, 200, "strength", 4)
    assert out["sets"] == 3
    assert out["reps"] == 8
    assert out["intensity_pct"] == 0.60


def test_blocks_primary_still_waves_when_accessory_skipped():
    """Sanity: scoping the wave to main-lift work doesn't break it for
    primary/secondary."""
    primary = {"name": "Squat", "lift": "squat", "role": "primary",
               "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9.0}
    secondary = {"name": "Paused squat", "lift": "squat", "role": "secondary",
                 "sets": 3, "reps": 5, "intensity_pct": 0.70, "rpe_cap": 8.0}
    p1 = blocks.compute_exercise_load(primary, 150, "strength", 1)
    p4 = blocks.compute_exercise_load(primary, 150, "strength", 4)
    assert p4["top_weight"] > p1["top_weight"]

    s1 = blocks.compute_exercise_load(secondary, 150, "strength", 1)
    s4 = blocks.compute_exercise_load(secondary, 150, "strength", 4)
    assert s4["top_weight"] > s1["top_weight"]


def test_blocks_deload_only_overwrites_main_lift():
    """Deload week must NOT replace a lateral raise with the block's
    main-lift deload prescription (2x5 @ 65%). Accessory keeps reps and
    RPE; only sets are trimmed for fatigue management."""
    accessory = {"name": "DB lateral raise", "lift": "bench",
                 "role": "accessory",
                 "sets": 3, "reps": 12, "intensity_pct": 0.30, "rpe_cap": 9.0}
    out = blocks.compute_exercise_load(accessory, 100, "volume", 4,
                                       in_deload=True)
    # Sets trimmed by 1 for fatigue management.
    assert out["sets"] == 2
    # Reps + RPE preserved — NOT overwritten by the main-lift deload (2x5).
    assert out["reps"] == 12
    assert out["rpe_cap"] == 9.0
    assert out["week_label"] == "Deload"

    # Primary still uses the block's deload prescription.
    primary = {"name": "Squat", "lift": "squat", "role": "primary",
               "sets": 4, "reps": 6, "intensity_pct": 0.72, "rpe_cap": 8.0}
    dl_main = blocks.compute_exercise_load(primary, 150, "volume", 4,
                                           in_deload=True)
    assert dl_main["sets"] == 2  # from volume deload prescription
    assert dl_main["reps"] == 5
    assert dl_main["rpe_cap"] == 7.0


def test_blocks_accessory_intensity_zero_skips_load():
    """LLM sets intensity_pct=0 for non-bar accessories (DB lateral
    raise, cable pushdown, machine row). Engine must return
    top_weight=None so the UI hides the meaningless kg + plate display."""
    accessory = {"name": "DB lateral raise", "lift": "bench",
                 "role": "accessory",
                 "sets": 3, "reps": 12, "intensity_pct": 0.0, "rpe_cap": 9.0}
    out = blocks.compute_exercise_load(accessory, 100, "volume", 2)
    assert out["top_weight"] is None
    # Reps + RPE still surfaced — the prescription is RPE-driven.
    assert out["reps"] == 12
    assert out["rpe_cap"] == 9.0


def test_blocks_accessory_intensity_nonzero_still_loads():
    """A loaded accessory (RDL with %TM) must still get a top_weight."""
    rdl = {"name": "RDL", "lift": "deadlift", "role": "accessory",
           "sets": 3, "reps": 8, "intensity_pct": 0.60, "rpe_cap": 8.0}
    out = blocks.compute_exercise_load(rdl, 200, "volume", 2)
    assert out["top_weight"] is not None
    assert out["top_weight"] == 120.0


# --- RPE-only accessory auto-normalization ---------------------------------
#
# Accessories whose load can't be derived from a barbell TM (weighted
# pull-up, DB / cable / machine work) have no meaningful %TM — a "60% of
# squat TM" added to a pull-up would be a near-world-record load.
# compute_exercise_load zeroes out intensity_pct for accessory names that
# match the RPE-only token list, so the UI shows only the RPE
# prescription. These tests pin that behaviour.

def test_rpe_only_accessory_weighted_pullup_normalized():
    """A weighted pull-up given a %TM collapses to RPE-only — no kg, no
    plate breakdown; the lifter picks added weight that hits the RPE."""
    pullup = {"name": "Weighted pull-up", "lift": "squat", "role": "accessory",
              "sets": 3, "reps": 6, "intensity_pct": 0.60, "rpe_cap": 8.0}
    out = blocks.compute_exercise_load(pullup, 150, "strength", 2)
    assert out["top_weight"] is None
    assert out["intensity_pct"] == 0.0
    assert out["reps"] == 6  # rep range + RPE survive
    assert out["rpe_cap"] == 8.0


def test_rpe_only_accessory_db_bench_normalized():
    """DB bench must NOT render as 'X kg + plates per side'. DB strength is
    decoupled from barbell bench TM."""
    db = {"name": "DB bench press", "lift": "bench", "role": "accessory",
          "sets": 3, "reps": 10, "intensity_pct": 0.55, "rpe_cap": 8.0}
    out = blocks.compute_exercise_load(db, 100, "volume", 2)
    assert out["top_weight"] is None
    assert out["intensity_pct"] == 0.0


def test_rpe_only_accessory_lateral_raise_normalized():
    """Lateral raise is fundamentally RPE-driven — no %TM relationship."""
    raise_ex = {"name": "DB lateral raise", "lift": "bench", "role": "accessory",
                "sets": 3, "reps": 12, "intensity_pct": 0.40, "rpe_cap": 9.0}
    out = blocks.compute_exercise_load(raise_ex, 100, "volume", 1)
    assert out["top_weight"] is None


def test_rpe_only_accessory_machine_and_cable_normalized():
    """Cable and machine work is implement-bound — no meaningful %TM."""
    for name in ("Cable tricep pushdown", "Leg extension", "Lat pulldown",
                 "Pec deck fly", "Chest press machine", "Hack squat",
                 "Leg press", "Cable face pull"):
        ex = {"name": name, "lift": "bench", "role": "accessory",
              "sets": 3, "reps": 10, "intensity_pct": 0.50, "rpe_cap": 8.0}
        out = blocks.compute_exercise_load(ex, 100, "volume", 1)
        assert out["top_weight"] is None, f"{name!r} should have been normalized"


def test_rpe_only_accessory_isolation_normalized():
    """Bicep curls, tricep extensions, face pulls, flys, etc."""
    for name in ("Preacher curl", "Hammer curl", "Skullcrusher",
                 "JM press", "Cable face pull", "Rear-delt fly",
                 "DB front raise"):
        ex = {"name": name, "lift": "bench", "role": "accessory",
              "sets": 3, "reps": 8, "intensity_pct": 0.50, "rpe_cap": 8.0}
        out = blocks.compute_exercise_load(ex, 100, "volume", 1)
        assert out["top_weight"] is None, f"{name!r} should have been normalized"


def test_rpe_only_does_not_touch_barbell_accessories():
    """Genuine barbell accessories (RDL, barbell row, good morning) must
    keep their %TM-driven load — they share a clear pattern with a main lift."""
    for name, tm, expected_close_to in (
        ("Romanian deadlift", 200, 120.0),
        ("Barbell row", 100, 60.0),
        ("Good morning", 200, 120.0),
    ):
        ex = {"name": name, "lift": "deadlift", "role": "accessory",
              "sets": 3, "reps": 8, "intensity_pct": 0.60, "rpe_cap": 8.0}
        out = blocks.compute_exercise_load(ex, tm, "volume", 2)
        assert out["top_weight"] is not None, f"{name!r} should keep its %TM load"


def test_rpe_only_does_not_touch_secondary_movements():
    """A close-grip bench or paused squat in role='secondary' is the SECONDARY
    main lift, not an accessory — it must keep its %TM. The normalizer only
    fires on role='accessory'."""
    cg = {"name": "Close-grip bench", "lift": "bench", "role": "secondary",
          "sets": 4, "reps": 5, "intensity_pct": 0.75, "rpe_cap": 8.0}
    out = blocks.compute_exercise_load(cg, 100, "volume", 2)
    assert out["top_weight"] is not None


def test_rpe_only_does_not_touch_primary():
    """Primary main lifts always get %TM — never normalized."""
    sq = {"name": "Squat", "lift": "squat", "role": "primary",
          "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9.0}
    out = blocks.compute_exercise_load(sq, 150, "strength", 2)
    assert out["top_weight"] is not None


def test_is_rpe_only_accessory_function():
    """Direct test of the detection function — substring match, case-insensitive."""
    assert expert_knowledge.is_rpe_only_accessory("Weighted pull-up")
    assert expert_knowledge.is_rpe_only_accessory("WEIGHTED PULL-UP")
    assert expert_knowledge.is_rpe_only_accessory("  DB bench  ")
    assert expert_knowledge.is_rpe_only_accessory("Single-arm DB row")
    assert expert_knowledge.is_rpe_only_accessory("Leg extension")
    assert expert_knowledge.is_rpe_only_accessory("Cable woodchop")
    assert expert_knowledge.is_rpe_only_accessory("Hyperextension")
    assert expert_knowledge.is_rpe_only_accessory("Glute-ham raise")
    assert expert_knowledge.is_rpe_only_accessory("Bulgarian split squat (DB)")
    assert expert_knowledge.is_rpe_only_accessory("Farmer carry")
    # Negatives — barbell movements must NOT match.
    assert not expert_knowledge.is_rpe_only_accessory("Squat")
    assert not expert_knowledge.is_rpe_only_accessory("Bench")
    assert not expert_knowledge.is_rpe_only_accessory("Deadlift")
    assert not expert_knowledge.is_rpe_only_accessory("Romanian deadlift")
    assert not expert_knowledge.is_rpe_only_accessory("Close-grip bench")
    assert not expert_knowledge.is_rpe_only_accessory("Paused squat")
    assert not expert_knowledge.is_rpe_only_accessory("Deficit deadlift")
    assert not expert_knowledge.is_rpe_only_accessory("Barbell row")
    assert not expert_knowledge.is_rpe_only_accessory("Block pull")


def test_compute_start_date_today_is_first_day():
    """If the first day's label names today's weekday, start today."""
    # 2026-05-25 is a Monday.
    monday = date(2026, 5, 25)
    plan = {"days": [{"label": "Monday — Heavy Squat", "exercises": []}]}
    assert _compute_start_date(plan, monday) == monday


def test_compute_start_date_skips_to_next_monday():
    """If today is Wednesday and the plan's first day is Monday, the block
    starts the following Monday — not today."""
    # 2026-05-27 is a Wednesday.
    wednesday = date(2026, 5, 27)
    plan = {"days": [{"label": "Monday — Heavy Squat", "exercises": []}]}
    start = _compute_start_date(plan, wednesday)
    assert start == date(2026, 6, 1)  # the following Monday
    assert start.strftime("%A") == "Monday"


def test_compute_start_date_first_day_is_tuesday():
    """The next-occurrence rule works for any weekday, not just Monday."""
    monday = date(2026, 5, 25)
    plan = {"days": [{"label": "Tuesday — Heavy Bench", "exercises": []}]}
    assert _compute_start_date(plan, monday) == date(2026, 5, 26)


def test_compute_start_date_fallback_to_next_monday_on_unlabeled():
    """If the first day's label doesn't name a weekday, default to next
    Monday so the lifter always gets a clean week-start."""
    wednesday = date(2026, 5, 27)
    plan = {"days": [{"label": "Day 1 — Heavy Squat", "exercises": []}]}
    assert _compute_start_date(plan, wednesday) == date(2026, 6, 1)


def test_commit_block_uses_deterministic_start_date(tmp_path):
    """End-to-end: committing a Monday-first plan from a Wednesday
    schedules start for next Monday."""
    p = str(tmp_path / "state.json")
    agent = _new_agent(p)
    plan = {"days": [
        {"label": "Monday — Heavy Squat", "exercises": [
            {"name": "Squat", "lift": "squat", "role": "primary",
             "sets": 1, "reps": 3, "intensity_pct": 0.82, "rpe_cap": 8},
            *_generic_accessories("squat")]},
        {"label": "Thursday — Heavy Bench", "exercises": [
            {"name": "Bench", "lift": "bench", "role": "primary",
             "sets": 1, "reps": 3, "intensity_pct": 0.82, "rpe_cap": 8},
            *_generic_accessories("bench")]},
    ]}
    agent.commit_block("strength", 5, "test", ["squat"], weekly_plan=plan)
    started_at = agent.state["block"]["started_at"]
    # Engine-computed; must parse as a date and be a Monday.
    sd = date.fromisoformat(started_at)
    assert sd.strftime("%A") == "Monday"
    assert sd >= date.today()


def test_blocks_accessory_deload_floors_at_one_set():
    """Trimming sets-by-1 must clamp at 1, not go to 0."""
    accessory = {"name": "Face pull", "lift": "bench", "role": "accessory",
                 "sets": 1, "reps": 15, "intensity_pct": 0.30, "rpe_cap": 9.0}
    out = blocks.compute_exercise_load(accessory, 100, "volume", 4,
                                       in_deload=True)
    assert out["sets"] == 1  # not 0


def test_blocks_catalogue_includes_indication():
    """LLM needs to know WHEN to pick each block — indication must be present."""
    cat = blocks.block_catalogue()
    assert len(cat) == 4
    for entry in cat:
        assert entry["indication"]
        assert entry["focus"]
        assert entry["default_weeks"] >= 2


# --- agent: init + block commit + lifecycle --------------------------------

def test_init_program_records_lifter_context_and_leaves_block_empty():
    """New: init does NOT auto-create a block — the agent diagnoses one
    through chat instead."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = Agent(path)
        agent.init_program("Sam", {"squat": 150, "bench": 100, "deadlift": 190},
                           bodyweight_kg=85, experience="intermediate")
        assert agent.state["lifter"]["name"] == "Sam"
        assert agent.state["lifter"]["bodyweight_kg"] == 85
        assert agent.state["lifter"]["experience"] == "intermediate"
        assert agent.state["block"]["type"] is None
        # TM equals the entered 1RM (no 5/3/1 safety buffer applied).
        assert agent.state["lifts"]["squat"]["training_max"] == 150.0


def test_commit_block_populates_prescriptions_immediately():
    """commit_block must generate week-1 prescriptions for all 3 lifts so the
    UI can render the program without an explicit plan_next call."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        assert agent.state["pending_prescriptions"] == {}
        agent.commit_block("strength", 4, "ready to express force", [])
        pending = agent.state["pending_prescriptions"]
        assert set(pending.keys()) == {"squat", "bench", "deadlift"}
        # Strength wave week 1: intensity_mult 0.92, rep_delta +1.
        # Base 4x3 @ 85% on squat TM 150 -> week 1: 4x4 @ ~78.2% = 117.5kg.
        sq = pending["squat"]
        assert sq["sets"] == 4 and sq["reps"] == 4
        assert sq["top_weight"] == 117.5


def test_commit_block_writes_rationale_and_focus():
    """The LLM's propose_block tool calls commit_block — it must persist
    rationale and focus_lifts for later review."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        out = agent.commit_block("volume", 4, "bench stalled at 100",
                                 ["bench"])
        assert "committed" in out
        b = agent.state["block"]
        assert b["type"] == "volume"
        assert b["duration_weeks"] == 4
        assert b["rationale"] == "bench stalled at 100"
        assert b["focus_lifts"] == ["bench"]
        assert b["week"] == 1


def test_commit_block_stores_weekly_plan_verbatim():
    """A full weekly_plan from the LLM is persisted unchanged in block state
    so the UI can render it day-by-day. Minimum 2 days per the validator."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        plan = {"days": [
            {"label": "Day 1 — Heavy Squat",
             "exercises": [
                 {"name": "Squat", "lift": "squat", "role": "primary",
                  "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9,
                  "rationale": None},
                 {"name": "Block pull", "lift": "deadlift", "role": "accessory",
                  "sets": 3, "reps": 3, "intensity_pct": 0.90, "rpe_cap": 8,
                  "rationale": "grip overload"},
                 _acc("Leg curl", "squat"),
             ]},
            {"label": "Day 2 — Bench",
             "exercises": [
                 {"name": "Bench", "lift": "bench", "role": "primary",
                  "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9,
                  "rationale": None},
                 *_generic_accessories("bench"),
             ]},
        ]}
        agent.commit_block("strength", 4, "test", ["squat"], weekly_plan=plan)
        stored = agent.state["block"]["weekly_plan"]
        assert len(stored["days"]) == 2
        assert stored["days"][0]["exercises"][1]["name"] == "Block pull"
        assert stored["days"][0]["exercises"][1]["rationale"] == "grip overload"


def test_archive_current_block_moves_to_history():
    """When a block completes, archive_current_block pushes it to history
    and clears the active block (forcing the next propose_block)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        agent.archive_current_block("completed cleanly")
        assert agent.state["block"]["type"] is None
        assert len(agent.state["block_history"]) == 1
        assert agent.state["block_history"][0]["outcome"] == "completed cleanly"


def test_plan_next_refuses_when_no_block_set():
    """Without an active block, plan_next can't prescribe — the agent must
    converse first to pick a block."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)  # no block committed
        # Should print a friendly message and NOT crash or invent a plan.
        agent.plan_next("squat")
        assert agent.state["pending_prescriptions"] == {}


def test_log_session_refuses_when_no_block_set():
    """Same: logging is meaningless without a prescription to evaluate against."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        sessions_before = len(agent.state["sessions"])
        agent.log_session("squat", 100, 5, 7.0, notes="random work")
        assert len(agent.state["sessions"]) == sessions_before


# --- per-session prescription tracking (from earlier refactor) --------------

def test_prescription_persisted_by_plan():
    """plan_next stores what it told the lifter to do."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        agent.plan_next("squat")
        rx = agent.state["pending_prescriptions"]["squat"]
        # Volume wave week 1: intensity_mult 0.92. Base 4x6 @ 72% on
        # TM 150 -> 4x6 @ 66.2% = 99.4 -> rounds to 100.0.
        assert rx["sets"] == 4 and rx["reps"] == 6
        assert rx["top_weight"] == 100.0
        assert rx["block_type"] == "volume"


def test_log_evaluates_against_issued_prescription_not_block_template():
    """A heavy 3-rep set logged against a 3-rep prescription must NOT
    report 3 missed reps just because the volume block template says 6."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        agent.state["pending_prescriptions"]["squat"] = {
            "sets": 4, "reps": 3, "top_weight": 127.5, "intensity_pct": 0.85,
            "rpe_cap": 9.0, "block_type": "strength", "week": 1,
        }
        agent.log_session("squat", 127.5, 3, reported_rpe=8.0, notes="solid")
        last = agent.state["sessions"][-1]
        assert last["missed_reps"] == 0


def test_prescription_cleared_after_log():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        agent.plan_next("squat")
        assert "squat" in agent.state["pending_prescriptions"]
        agent.log_session("squat", 97.5, 6, reported_rpe=7.5, notes="clean")
        assert "squat" not in agent.state["pending_prescriptions"]


# --- deload state machine --------------------------------------------------

def test_deload_decision_enters_deload_week_not_phase_advance():
    """Headline fix: a 'deload' decision enters a deload week IN THE SAME
    block, not skips to the next."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        agent.state["block"]["week"] = 4  # at the end
        agent.plan_next("squat")
        agent.log_session("squat", 97.5, 6, reported_rpe=7.5, notes="clean")
        assert agent.state["block"]["type"] == "volume"  # SAME block
        assert agent.state["block"]["in_deload"] is True


def test_deload_completion_archives_block():
    """After deload week completes, the block is archived to history and
    cleared — the LLM must propose the NEXT block via chat."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        agent.state["block"]["in_deload"] = True
        agent.plan_next("squat")
        agent.log_session("squat", 87.5, 5, reported_rpe=6.5, notes="light")
        # Block done -> archived -> current block is now empty.
        assert agent.state["block"]["type"] is None
        assert len(agent.state["block_history"]) == 1


def test_plan_during_deload_uses_deload_prescription():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        agent.state["block"]["in_deload"] = True
        agent.plan_next("squat")
        rx = agent.state["pending_prescriptions"]["squat"]
        # volume deload: 2x5 @ 65% TM
        assert rx["sets"] == 2 and rx["reps"] == 5
        assert rx["intensity_pct"] == 0.65
        assert rx["rpe_cap"] == 7.0


# --- ReAct tool dispatch ---------------------------------------------------

def test_react_tool_dispatch_calculate_load():
    out = reasoning.dispatch_tool({}, "calculate_load",
                                  {"training_max": 150, "intensity_pct": 0.85})
    assert out == {"weight_kg": 127.5}


def test_react_tool_dispatch_state_lookup_block_status():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        block = reasoning.dispatch_tool(agent.state, "get_block_status", {})
        assert block["active_block"]["type"] == "volume"
        assert block["lifter"]["experience"] == "intermediate"
        assert block["past_blocks_count"] == 0
        # Calendar context for block-start timing.
        today = block["today"]
        assert "date" in today and "weekday" in today and "next_monday" in today
        assert today["weekday"] in ("Monday", "Tuesday", "Wednesday", "Thursday",
                                    "Friday", "Saturday", "Sunday")


def test_react_tool_dispatch_unknown_tool_is_graceful():
    out = reasoning.dispatch_tool({}, "nuke_database", {})
    assert "error" in out


def test_react_tool_list_block_types():
    out = reasoning.dispatch_tool({}, "list_block_types", {})
    assert len(out["block_types"]) == 4
    types = {b["type"] for b in out["block_types"]}
    assert types == {"volume", "technique", "strength", "peaking"}


def test_react_propose_block_via_tool_commits_state():
    """The LLM's propose_block tool must update state via the agent, including
    the supplied weekly_plan with days + exercises."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        plan = {"days": [
            {"label": "Day 1 — Heavy Squat",
             "exercises": [
                 {"name": "Squat", "lift": "squat", "role": "primary",
                  "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9.0,
                  "rationale": None},
                 # Use leg press + leg curl as accessories. (Paused squat
                 # would be a primary-quality variation stacked with the
                 # squat — rejected by the validator's rule 5.)
                 _acc("Leg press", "squat"),
                 _acc("Leg curl", "squat"),
             ]},
            {"label": "Day 2 — Bench",
             "exercises": [
                 {"name": "Bench", "lift": "bench", "role": "primary",
                  "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9.0,
                  "rationale": None},
                 *_generic_accessories("bench"),
             ]},
            {"label": "Day 3 — Deadlift",
             "exercises": [
                 {"name": "Deadlift", "lift": "deadlift", "role": "primary",
                  "sets": 3, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9.0,
                  "rationale": None},
                 *_generic_accessories("deadlift"),
             ]},
        ]}
        out = reasoning.dispatch_tool(
            agent.state, "propose_block",
            {"block_type": "strength", "duration_weeks": 4,
             "rationale": "ready for higher intensities", "focus_lifts": [],
             "weekly_plan": plan},
            agent=agent,
        )
        assert "committed" in out
        assert agent.state["block"]["type"] == "strength"
        # Plan must round-trip into state.
        assert len(agent.state["block"]["weekly_plan"]["days"]) == 3
        # Primary lift's prescription auto-populated (Squat is primary on day 1)
        assert "squat" in agent.state["pending_prescriptions"]
        # The Paused squat (accessory @ 0.70) should NOT be the primary —
        # the heavier Squat @ 0.85 wins as the pending prescription.
        assert agent.state["pending_prescriptions"]["squat"]["exercise_name"] == "Squat"


def test_react_propose_block_without_agent_errors_cleanly():
    """Read-only call path: the mutating tool returns an error if no agent."""
    out = reasoning.dispatch_tool(
        {}, "propose_block",
        {"block_type": "volume", "duration_weeks": 4, "rationale": "x"},
    )
    assert "error" in out


def test_react_record_lifter_context_appends_notes():
    """goals_notes / training_notes should append over multiple turns."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        reasoning.dispatch_tool(agent.state, "record_lifter_context",
                                {"goals_notes": "add 20kg to squat"},
                                agent=agent)
        reasoning.dispatch_tool(agent.state, "record_lifter_context",
                                {"goals_notes": "meet in 6 months"},
                                agent=agent)
        notes = agent.state["lifter"]["goals_notes"]
        assert "add 20kg" in notes and "meet in 6 months" in notes


def test_react_review_current_block_archives():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        out = reasoning.dispatch_tool(
            agent.state, "review_current_block",
            {"outcome": "went well, bench moved up"}, agent=agent,
        )
        assert "archived" in out
        assert len(agent.state["block_history"]) == 1


def test_coach_react_offline_falls_back_gracefully(monkeypatch):
    """Without an API key, coach_react never crashes — just an offline note."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = reasoning.coach_react("what should I do?", {"sessions": [], "block": {}})
    assert "offline" in out["text"].lower()
    assert out["steps"] == []


# --- chat intent parsing ---------------------------------------------------

def test_chat_parses_natural_log_message():
    out = parse_intent("just squatted 130x3 at RPE 9, felt heavy")
    assert out["intent"] == "log"
    assert out["lift"] == "squat" and out["weight"] == 130.0
    assert out["reps"] == 3 and out["rpe"] == 9.0
    assert "felt heavy" in out["notes"]


def test_chat_parses_bench_with_for_separator():
    out = parse_intent("hit bench 100 for 5 RPE 8")
    assert out["intent"] == "log"
    assert out["lift"] == "bench" and out["weight"] == 100.0 and out["reps"] == 5


def test_chat_parses_plan_intent():
    assert parse_intent("what should I do today?")["intent"] == "plan"


def test_chat_parses_status_intent():
    assert parse_intent("where am i in the block?")["intent"] == "status"


def test_chat_routes_unknown_to_ask():
    out = parse_intent("is it normal for my hips to shoot up on heavy squats?")
    assert out["intent"] == "ask"


def test_chat_exit():
    assert parse_intent("exit")["intent"] == "exit"
    assert parse_intent("bye")["intent"] == "exit"


# --- accessories knowledge base --------------------------------------------

def test_accessories_suggest_computes_load_from_training_max():
    pick = accessories.suggest_for("bench", "lockout", training_max=100.0)
    assert pick["name"] == "Close-grip bench" and pick["load_kg"] == 75.0


def test_accessories_no_tm_returns_pct_only():
    pick = accessories.suggest_for("squat", "out_of_the_hole", training_max=None)
    assert pick["load_kg"] is None


def test_diagnose_per_lift_with_loads_in_suggestion():
    state = {
        "lifts": {"bench": {"training_max": 100.0},
                  "squat": {"training_max": 150.0},
                  "deadlift": {"training_max": 200.0}},
        "sessions": [],
    }
    history = [
        {"lift": "bench", "sticking_point": "lockout"},
        {"lift": "bench", "sticking_point": "lockout"},
        {"lift": "squat", "sticking_point": "lockout"},
    ]
    findings = reasoning.diagnose(history, state)
    assert len(findings) == 1
    f = findings[0]
    assert f["lift"] == "bench" and "75.0 kg" in f["suggestion"]


def test_recommend_accessory_tool():
    state = {"lifts": {"deadlift": {"training_max": 200.0}}, "sessions": []}
    out = reasoning.dispatch_tool(state, "recommend_accessory",
                                  {"lift": "deadlift",
                                   "sticking_point": "off_the_floor"})
    assert len(out["candidates"]) >= 2
    assert out["candidates"][0]["load_kg"] == 150.0


# --- expert_knowledge.py: curated Ben-style coaching KB --------------------

def test_expert_knowledge_no_avoided_items_leak_in():
    """Data integrity: nothing in the avoid list (plank, ab wheel, push-up,
    burpee, etc.) may appear inside EXPERT_ACCESSORIES — otherwise the KB
    contradicts itself."""
    for lift, targets in expert_knowledge.EXPERT_ACCESSORIES.items():
        for target, picks in targets.items():
            for pick in picks:
                assert not expert_knowledge.is_avoided(pick["name"]), (
                    f"avoided exercise '{pick['name']}' leaked into "
                    f"EXPERT_ACCESSORIES[{lift!r}][{target!r}]"
                )


def test_expert_knowledge_every_lift_has_general_pool():
    """Every lift must have a 'general' fallback target — that's what
    recommend_accessories() falls back to when no weakness is known."""
    for lift in ("squat", "bench", "deadlift"):
        assert "general" in expert_knowledge.EXPERT_ACCESSORIES[lift], (
            f"{lift} missing a 'general' accessory pool"
        )


def test_expert_knowledge_recommend_returns_ranked_picks():
    picks = expert_knowledge.recommend_accessories("bench", "tricep", count=3)
    assert len(picks) == 3
    # Tier 1 picks come first.
    assert picks[0]["tier"] <= picks[1]["tier"] <= picks[2]["tier"]
    # Ben's tier-1 tricep picks per QHGcZaDuwpQ + a5xE7WbtxRs.
    names = [p["name"] for p in picks]
    assert "Single-arm tricep push-down" in names
    assert "JM press (Smith machine)" in names


def test_expert_knowledge_every_pick_has_source_citation():
    """Every accessory entry must cite a Ben video — the KB's value over
    a generic LLM is that recommendations are grounded in real content."""
    for lift, targets in expert_knowledge.EXPERT_ACCESSORIES.items():
        for target, picks in targets.items():
            for pick in picks:
                assert pick.get("source"), (
                    f"missing source citation in "
                    f"EXPERT_ACCESSORIES[{lift!r}][{target!r}]: {pick['name']}"
                )


def test_expert_knowledge_block_type_overrides_reps():
    """Volume block tightens compound-accessory reps to (6, 10); strength
    block tightens isolation reps to (6, 10) for fatigue management."""
    vol = expert_knowledge.recommend_accessories(
        "bench", "tricep", block_type="volume", count=3)
    jm = next(p for p in vol if p["name"] == "JM press (Smith machine)")
    # JM press is compound_accessory -> volume block overrides to (6, 10)
    assert jm["reps"] == (6, 10)

    strength = expert_knowledge.recommend_accessories(
        "bench", "tricep", block_type="strength", count=3)
    pushdown = next(p for p in strength
                    if p["name"] == "Single-arm tricep push-down")
    # Single-arm pushdown is isolation -> strength block overrides to (6, 10)
    assert pushdown["reps"] == (6, 10)


def test_expert_knowledge_unknown_target_falls_back_to_general():
    picks = expert_knowledge.recommend_accessories("squat", "nonexistent_target")
    general_names = {p["name"] for p in
                     expert_knowledge.EXPERT_ACCESSORIES["squat"]["general"]}
    for p in picks:
        assert p["name"] in general_names


def test_expert_knowledge_avoid_list_normalised():
    assert expert_knowledge.is_avoided("plank")
    assert expert_knowledge.is_avoided("PLANK")
    assert expert_knowledge.is_avoided("  weighted plank  ")
    assert expert_knowledge.is_avoided("ab wheel")
    assert expert_knowledge.is_avoided("burpee")
    assert not expert_knowledge.is_avoided("JM press (Smith machine)")
    assert not expert_knowledge.is_avoided("Romanian deadlift")


def test_expert_knowledge_prescription_patterns_have_required_fields():
    for name in ("top_set_backoff_primary", "top_set_backoff_secondary",
                 "straight_sets"):
        p = expert_knowledge.prescription_pattern(name)
        assert p is not None, f"missing pattern: {name}"
        assert "description" in p
        assert "use_for" in p
        assert "source" in p

    # Top-set + back-off patterns need top_set + backoff nested.
    for name in ("top_set_backoff_primary", "top_set_backoff_secondary"):
        p = expert_knowledge.prescription_pattern(name)
        assert "top_set" in p and "backoff" in p
        # Back-off load is at 60-75% of 1RM — Ben's key correction over
        # the "90% of top set" misconception.
        assert p["backoff"]["intensity_pct_low"] >= 0.55
        assert p["backoff"]["intensity_pct_high"] <= 0.80
        # Back-offs sit at RPE 5-7 (5-10 reps in reserve).
        assert p["backoff"]["rpe_low"] <= 6.0
        assert p["backoff"]["load_static_across_block"] is True


def test_expert_knowledge_accessory_rep_range_helper():
    # Ben's collapsed ranges — close to the canonical 4-10 hypertrophy zone.
    assert expert_knowledge.accessory_rep_range("compound_accessory", "volume") == (6, 10)
    assert expert_knowledge.accessory_rep_range("isolation", "volume") == (8, 12)
    assert expert_knowledge.accessory_rep_range("compound_accessory", "strength") == (4, 8)
    # Unknown block type -> safe default in Ben's hypertrophy zone.
    assert expert_knowledge.accessory_rep_range("isolation", "unknown") == (8, 12)


def test_expert_knowledge_volume_landmarks_match_ben_numbers():
    """Ben's exact set counts from 1OXhV1uxPpw."""
    sq = expert_knowledge.volume_landmark("squat")
    assert sq["sets_per_week_low"] == 6 and sq["sets_per_week_high"] == 10
    bn = expert_knowledge.volume_landmark("bench")
    assert bn["sets_per_week_low"] == 9 and bn["sets_per_week_high"] == 16
    dl = expert_knowledge.volume_landmark("deadlift")
    assert dl["sets_per_week_low"] == 5 and dl["sets_per_week_high"] == 8
    # Conventional DL rep cap is 6 per n9Wyxr4Yks0.
    assert dl["rep_cap"] == 6


def test_expert_knowledge_block_periodization_defaults():
    """406rClFwNxY: 5-week blocks, RPE wave 5-9, top-set rep cycle of
    triples/doubles/singles, 3 blocks per intermediate PR attempt."""
    bp = expert_knowledge.block_periodization()
    assert bp["default_block_length_weeks"] == 5
    assert bp["rpe_wave_5wk"] == [5, 6, 7, 8, 9]
    assert bp["top_set_rep_cycle"] == ["triples", "doubles", "singles"]
    assert bp["blocks_per_pr_attempt"]["intermediate"] == 3


def test_expert_knowledge_hypertrophy_checklist_covers_ben_slots():
    """Ben's checklist (QHGcZaDuwpQ @ 17:40): vertical pulls (2), horizontal
    pulls (2), one chest/shoulder press, three delt heads, bicep, tricep."""
    slots = {item["slot"] for item in expert_knowledge.hypertrophy_checklist()}
    # Two vertical pulls (upper + lower lats).
    assert "vertical_pull_upper_lats" in slots
    assert "vertical_pull_lower_lats" in slots
    # Two horizontal pulls (upper back + lats).
    assert "horizontal_pull_upper_back" in slots
    assert "horizontal_pull_lats" in slots
    # One press supplement (chest OR shoulder, choose by weakness).
    assert "press_supplement" in slots
    # All three delt heads.
    assert {"lateral_delt", "front_delt", "rear_delt"}.issubset(slots)
    # Bicep + tricep slots.
    assert "bicep" in slots
    assert {"tricep_isolation", "tricep_compound"}.issubset(slots)


def test_expert_knowledge_strength_variation_lookup():
    """Pin squat for depth, pause DL 1\" for start position, long-pause
    bench for chest tightness — Ben's stated picks for weak-point fixes."""
    depth_picks = expert_knowledge.recommend_strength_variation("squat", "depth")
    assert any("Pin squat" in p["name"] for p in depth_picks)

    start_picks = expert_knowledge.recommend_strength_variation(
        "deadlift", "start_position")
    assert any("Pause deadlift" in p["name"] for p in start_picks)

    tight_picks = expert_knowledge.recommend_strength_variation(
        "bench", "chest_tightness")
    assert any("Long-pause" in p["name"] for p in tight_picks)


def test_chat_commit_check_flags_hallucinated_commit(tmp_path):
    """If the LLM says 'I've committed' but propose_block wasn't called,
    the chat layer must surface a clear warning so the lifter isn't
    misled by a fake confirmation + empty program panel."""
    from coach.chat import Chat
    p = str(tmp_path / "state.json")
    chat = Chat(p)
    # Hallucinated commit: text claims success, but steps is empty.
    fake_out = {"text": "I've committed a 4-week volume block. Block starts Monday.",
                "steps": []}
    rendered = chat._with_commit_check(fake_out)
    assert "didn't actually call propose_block" in rendered.lower()


def test_chat_commit_check_passes_when_propose_succeeds(tmp_path):
    """If propose_block was called successfully, the warning must NOT
    appear — that would be a false alarm."""
    from coach.chat import Chat
    p = str(tmp_path / "state.json")
    chat = Chat(p)
    # propose_block ran and returned a success dict (no 'error' key).
    fake_out = {"text": "I've committed a strength block. Block starts Monday.",
                "steps": [("propose_block", {"block_type": "strength"},
                           {"ok": True, "block": {"type": "strength"}})]}
    rendered = chat._with_commit_check(fake_out)
    assert "didn't actually call propose_block" not in rendered.lower()
    assert "rejected" not in rendered.lower()


def test_chat_commit_check_surfaces_validator_error(tmp_path):
    """If propose_block was called but the validator rejected it, the
    chat layer must surface the validator's error message verbatim."""
    from coach.chat import Chat
    p = str(tmp_path / "state.json")
    chat = Chat(p)
    fake_out = {"text": "I've committed a volume block. Block starts Monday.",
                "steps": [("propose_block", {"block_type": "volume"},
                           {"error": "Day 1 has BOTH primary AND secondary squat"})]}
    rendered = chat._with_commit_check(fake_out)
    assert "rejected" in rendered.lower()
    assert "primary" in rendered.lower() and "secondary" in rendered.lower()


def test_chat_season_check_flags_hallucinated_season_plan(tmp_path):
    """When the reply claims future blocks are planned but propose_season
    was never called, the chat layer surfaces a warning so the lifter
    isn't misled by an empty Season Plan tab."""
    from coach.chat import Chat
    p = str(tmp_path / "state.json")
    chat = Chat(p)
    # LLM committed a block but mentioned "future blocks" without
    # calling propose_season.
    fake_out = {
        "text": ("I've committed a 4-week volume block. Future blocks are "
                 "planned to transition into strength and peaking phases, "
                 "culminating in your competition on October 12."),
        "steps": [("propose_block", {"block_type": "volume"},
                   {"committed": {"type": "volume"}})],
    }
    rendered = chat._with_commit_check(fake_out)
    assert "didn't actually call propose_season" in rendered.lower()


def test_chat_season_check_quiet_when_propose_season_was_called(tmp_path):
    """If propose_season WAS called this turn, no warning."""
    from coach.chat import Chat
    p = str(tmp_path / "state.json")
    chat = Chat(p)
    fake_out = {
        "text": ("I've laid out the macrocycle: 5 blocks ending in "
                 "peaking. Committed the first volume block."),
        "steps": [
            ("propose_season", {"blocks_plan": [...]},
             {"committed_season_plan": {"blocks": [{"block_type": "volume"}]}}),
            ("propose_block", {"block_type": "volume"},
             {"committed": {"type": "volume"}}),
        ],
    }
    rendered = chat._with_commit_check(fake_out)
    assert "didn't actually call propose_season" not in rendered.lower()


def test_chat_season_check_quiet_when_plan_already_exists(tmp_path):
    """If a season_plan ALREADY exists from a previous turn (just not
    updated this turn), don't false-alarm."""
    from coach.chat import Chat
    p = str(tmp_path / "state.json")
    chat = Chat(p)
    # Pre-populate a season_plan.
    chat.agent.commit_season_plan(
        "2026-10-15", None,
        blocks=[{"block_type": "strength", "duration_weeks": 4,
                 "rationale": "x"}],
    )
    fake_out = {
        "text": ("Your next block will be strength — leading up to your "
                 "competition."),
        "steps": [],
    }
    rendered = chat._with_commit_check(fake_out)
    assert "didn't actually call propose_season" not in rendered.lower()


def test_chat_season_check_quiet_when_no_season_claim(tmp_path):
    """A normal reply that doesn't claim macrocycle planning passes through."""
    from coach.chat import Chat
    p = str(tmp_path / "state.json")
    chat = Chat(p)
    fake_out = {
        "text": "Your readiness is good. Stick with the plan.",
        "steps": [],
    }
    rendered = chat._with_commit_check(fake_out)
    assert rendered == fake_out["text"]


def test_chat_commit_check_quiet_when_no_commit_claim(tmp_path):
    """A normal reply that doesn't claim a commit should pass through
    unchanged — no false warnings on every turn."""
    from coach.chat import Chat
    p = str(tmp_path / "state.json")
    chat = Chat(p)
    fake_out = {"text": "Your readiness is good. Stick with the plan.",
                "steps": []}
    rendered = chat._with_commit_check(fake_out)
    assert rendered == fake_out["text"]


def test_dispatch_get_prescription_pattern():
    """ReAct tool surfaces Ben's primary/secondary top-set+back-off layouts."""
    out = reasoning.dispatch_tool({}, "get_prescription_pattern",
                                  {"role": "primary"})
    assert "pattern" in out
    assert out["pattern_name"] == "top_set_backoff_primary"
    assert out["pattern"]["backoff"]["load_static_across_block"] is True

    bad = reasoning.dispatch_tool({}, "get_prescription_pattern",
                                  {"role": "nonsense"})
    assert "error" in bad


def test_dispatch_lookup_accessories():
    """ReAct tool returns ranked Ben picks for a (lift, target) pair."""
    out = reasoning.dispatch_tool({}, "lookup_accessories",
                                  {"lift": "bench", "target": "tricep",
                                   "count": 2})
    assert len(out["picks"]) == 2
    # Every pick must carry its source citation.
    for p in out["picks"]:
        assert "source" in p and p["source"]


def test_dispatch_get_volume_landmark():
    out = reasoning.dispatch_tool({}, "get_volume_landmark", {"lift": "bench"})
    assert out["landmark"]["sets_per_week_low"] == 9
    assert out["landmark"]["sets_per_week_high"] == 16


def test_dispatch_get_hypertrophy_checklist():
    out = reasoning.dispatch_tool({}, "get_hypertrophy_checklist", {})
    slots = {item["slot"] for item in out["checklist"]}
    assert "lateral_delt" in slots
    assert "tricep_isolation" in slots


def test_chat_intent_plan_renders_current_block():
    """Bare 'plan' and unambiguous 'what should I do today' route to PLAN."""
    assert parse_intent("plan")["intent"] == "plan"
    assert parse_intent("today's plan")["intent"] == "plan"
    assert parse_intent("what should i do today?")["intent"] == "plan"
    assert parse_intent("next workout?")["intent"] == "plan"


def test_commit_block_allows_replacing_active_block_without_sessions():
    """A freshly committed block can be revised (e.g. add a Sunday rest)
    by committing again, as long as no sessions have been logged against
    the active block_type + week yet."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        # No sessions logged yet.
        assert agent.state["sessions"] == []
        # Now try to revise the plan (different exercises, same block_type).
        revised = {"days": [
            {"label": "Mon — Squat", "exercises": [
                {"name": "Squat", "lift": "squat", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.78, "rpe_cap": 8},
                *_generic_accessories("squat")]},
            {"label": "Tue — Bench", "exercises": [
                {"name": "Bench", "lift": "bench", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.78, "rpe_cap": 8},
                *_generic_accessories("bench")]},
            # NEW: Sunday rest added
            {"label": "Sunday — Rest", "exercises": []},
        ]}
        out = agent.commit_block("volume", 4, "revised: added Sun rest",
                                 [], weekly_plan=revised)
        assert "committed" in out
        # The revised plan is now active.
        labels = [d["label"]
                  for d in agent.state["block"]["weekly_plan"]["days"]]
        assert "Sunday — Rest" in labels


def test_commit_block_refuses_replace_when_sessions_logged():
    """The protective case: once the lifter has logged at least ONE
    session against the active block, replacement requires explicit
    archive (review_current_block) first."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        # Log a session against the active block.
        memory.record_session(agent.state, {
            "lift": "squat", "weight": 100, "reps": 6,
            "effective_rpe": 7.0, "missed_reps": 0,
            "sticking_point": None, "notes": "",
            "block_type": agent.state["block"]["type"],
            "block_week": agent.state["block"]["week"],
        })
        # Try to replace — should be REFUSED.
        new_plan = {"days": [
            {"label": "Mon — Squat", "exercises": [
                {"name": "Squat", "lift": "squat", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
                *_generic_accessories("squat")]},
            {"label": "Tue — Bench", "exercises": [
                {"name": "Bench", "lift": "bench", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
                *_generic_accessories("bench")]},
        ]}
        out = agent.commit_block("strength", 4, "switch direction", [],
                                 weekly_plan=new_plan)
        assert "error" in out
        assert "REFUSED" in out["error"]
        assert "LOGGED SESSIONS" in out["error"] or "logged" in out["error"].lower()


def test_commit_block_refuses_to_overwrite_active_block():
    """Once the lifter has logged sessions against the active block,
    committing a new block must be refused with an instructive error
    (the active block must be archived first via review_current_block).
    This prevents a 'preview the next block' call from silently
    overwriting an in-progress block."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)  # volume block active
        # Log a session — now the active block is "in use".
        memory.record_session(agent.state, {
            "lift": "squat", "weight": 100, "reps": 6,
            "effective_rpe": 7.0, "missed_reps": 0,
            "sticking_point": None, "notes": "",
            "block_type": agent.state["block"]["type"],
            "block_week": agent.state["block"]["week"],
        })
        # Pretend the LLM (mis)called propose_block again to "preview".
        new_plan = {"days": [
            {"label": "Mon — Squat", "exercises": [
                {"name": "Squat", "lift": "squat", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
                *_generic_accessories("squat")]},
            {"label": "Tue — Bench", "exercises": [
                {"name": "Bench", "lift": "bench", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
                *_generic_accessories("bench")]},
        ]}
        out = agent.commit_block("strength", 4, "preview strength block",
                                 [], weekly_plan=new_plan)
        assert "error" in out
        assert "REFUSED" in out["error"]
        assert "get_season_plan" in out["error"]
        # Volume block must still be active — NOT replaced.
        assert agent.state["block"]["type"] == "volume"


def test_commit_block_allows_replace_after_explicit_archive():
    """The legit path: archive first, then commit a fresh block."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        agent.archive_current_block("changing direction mid-cycle")
        # No active block now — commit succeeds.
        out = agent.commit_block("strength", 4, "express force", [],
                                 weekly_plan={"days": [
                                     {"label": "Mon — Squat", "exercises": [
                                         {"name": "Squat", "lift": "squat",
                                          "role": "primary", "sets": 4, "reps": 3,
                                          "intensity_pct": 0.85, "rpe_cap": 9},
                                         *_generic_accessories("squat")]},
                                     {"label": "Tue — Bench", "exercises": [
                                         {"name": "Bench", "lift": "bench",
                                          "role": "primary", "sets": 4, "reps": 3,
                                          "intensity_pct": 0.85, "rpe_cap": 9},
                                         *_generic_accessories("bench")]},
                                 ]})
        assert "committed" in out
        assert agent.state["block"]["type"] == "strength"


def test_commit_block_replace_active_flag_overrides_refusal():
    """Internal escape hatch: replace_active=True allows overwrite when
    the caller has explicitly confirmed (used by future UI flows, not
    exposed to the LLM)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        out = agent.commit_block("strength", 4, "force replace", [],
                                 weekly_plan={"days": [
                                     {"label": "Mon — Squat", "exercises": [
                                         {"name": "Squat", "lift": "squat",
                                          "role": "primary", "sets": 4, "reps": 3,
                                          "intensity_pct": 0.85, "rpe_cap": 9},
                                         *_generic_accessories("squat")]},
                                     {"label": "Tue — Bench", "exercises": [
                                         {"name": "Bench", "lift": "bench",
                                          "role": "primary", "sets": 4, "reps": 3,
                                          "intensity_pct": 0.85, "rpe_cap": 9},
                                         *_generic_accessories("bench")]},
                                 ]},
                                 replace_active=True)
        assert "committed" in out
        assert agent.state["block"]["type"] == "strength"


def test_chat_intent_show_me_next_block_goes_to_llm():
    """'Show me the next block' should reach the LLM so it can call
    get_season_plan, NOT route to plan_next (current-block renderer)."""
    for phrasing in (
        "could you show me what the upcoming volume block could look like",
        "show me what comes after this",
        "what would the next block look like",
        "i mean the one after that one",
        "preview the strength block for me",
        "what's my next block",
    ):
        out = parse_intent(phrasing)
        assert out["intent"] == "ask", f"{phrasing!r} routed to {out}"


def test_chat_intent_plan_my_next_block_goes_to_llm():
    """'plan my next block' contains the word 'plan' but means future-block
    planning, not 'show me today's session'. It must route to the LLM (ask
    intent), not the current-block renderer (PLAN intent)."""
    out = parse_intent("plan my next block")
    assert out["intent"] == "ask", f"expected ask, got {out}"
    out = parse_intent("plan the next blocks for me")
    assert out["intent"] == "ask"
    out = parse_intent("plan my season")
    assert out["intent"] == "ask"
    out = parse_intent("can you plan ahead for the next 3 blocks")
    assert out["intent"] == "ask"
    out = parse_intent("plan a strength block after this")
    assert out["intent"] == "ask"
    out = parse_intent("show me what comes after this block")
    assert out["intent"] == "ask"


# --- season plan (macrocycle) ----------------------------------------------

def test_season_plan_default_state_is_empty():
    """A fresh agent has an empty season plan — competition date null,
    no upcoming blocks."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        sp = agent.state["season_plan"]
        assert sp["competition_date"] is None
        assert sp["competition_lifts"] is None
        assert sp["blocks"] == []


def test_commit_season_plan_stores_full_macrocycle():
    """Lay out a 4-block macrocycle and verify it persists."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        out = agent.commit_season_plan(
            competition_date="2026-10-15",
            competition_lifts={"squat": 180, "bench": 130, "deadlift": 210},
            blocks=[
                {"block_type": "volume", "duration_weeks": 4,
                 "planned_start": "2026-06-01",
                 "rationale": "Rebuild base after shoulder rehab."},
                {"block_type": "strength", "duration_weeks": 4,
                 "planned_start": "2026-07-06",
                 "rationale": "Express force after volume base."},
                {"block_type": "strength", "duration_weeks": 4,
                 "planned_start": "2026-08-10",
                 "rationale": "Push absolute strength near meet specs."},
                {"block_type": "peaking", "duration_weeks": 3,
                 "planned_start": "2026-09-21",
                 "rationale": "Taper into meet on Oct 15."},
            ],
        )
        assert "committed_season_plan" in out
        sp = agent.state["season_plan"]
        assert sp["competition_date"] == "2026-10-15"
        assert sp["competition_lifts"]["deadlift"] == 210
        assert len(sp["blocks"]) == 4
        assert sp["blocks"][-1]["block_type"] == "peaking"


def test_commit_season_plan_rejects_unknown_block_type():
    """Bad block_type returns an error and DOES NOT mutate state."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        out = agent.commit_season_plan(
            competition_date=None, competition_lifts=None,
            blocks=[
                {"block_type": "volume", "duration_weeks": 4,
                 "rationale": "ok"},
                {"block_type": "bogus_block", "duration_weeks": 4,
                 "rationale": "nope"},
            ],
        )
        assert "error" in out
        assert "bogus_block" in out["error"]
        assert agent.state["season_plan"]["blocks"] == []  # untouched


def test_commit_season_plan_rejects_bad_duration():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        out = agent.commit_season_plan(
            None, None,
            blocks=[{"block_type": "volume", "duration_weeks": 0,
                     "rationale": "x"}],
        )
        assert "error" in out
        assert "duration_weeks" in out["error"]


def test_archive_does_not_touch_season_plan():
    """Archive only writes block_history + clears the active block. It must
    NOT pop from season_plan — pop happens at COMMIT time (commit_block).
    The old design (pop on archive by type-match) broke when two
    consecutive blocks shared a type (V → V → S) because archiving the
    first V also removed the planned second V."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)  # volume block active
        agent.commit_season_plan(
            "2026-10-15", None,
            blocks=[
                {"block_type": "volume", "duration_weeks": 4, "rationale": "v2"},
                {"block_type": "strength", "duration_weeks": 4, "rationale": "s1"},
                {"block_type": "peaking", "duration_weeks": 3, "rationale": "p"},
            ],
        )
        before = list(agent.state["season_plan"]["blocks"])
        agent.archive_current_block("done well")
        # season_plan untouched — pop happens on commit, not archive.
        assert agent.state["season_plan"]["blocks"] == before


def test_commit_pops_season_plan_when_type_matches():
    """The new behaviour: when propose_block commits a block whose type
    matches season_plan.blocks[0], pop that entry. This is the symmetric
    half of the V → V → S bug fix — pop on commit, not archive."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        agent.commit_season_plan(
            "2026-10-15", None,
            blocks=[
                {"block_type": "volume", "duration_weeks": 4, "rationale": "v1"},
                {"block_type": "strength", "duration_weeks": 4, "rationale": "s1"},
                {"block_type": "peaking", "duration_weeks": 3, "rationale": "p"},
            ],
        )
        out = agent.commit_block("volume", 4, "rebuild base", [],
                                 weekly_plan={"days": [
                                     {"label": "Mon — Squat", "exercises": [
                                         {"name": "Squat", "lift": "squat",
                                          "role": "primary", "sets": 4, "reps": 3,
                                          "intensity_pct": 0.75, "rpe_cap": 8},
                                         *_generic_accessories("squat")]},
                                     {"label": "Tue — Bench", "exercises": [
                                         {"name": "Bench", "lift": "bench",
                                          "role": "primary", "sets": 4, "reps": 3,
                                          "intensity_pct": 0.75, "rpe_cap": 8},
                                         *_generic_accessories("bench")]},
                                 ]})
        assert "committed" in out
        # Volume entry was popped — only strength + peaking remain.
        upcoming = agent.state["season_plan"]["blocks"]
        assert len(upcoming) == 2
        assert upcoming[0]["block_type"] == "strength"


def test_commit_refused_when_type_mismatches_season_plan():
    """When the season plan's next entry is Volume but a Strength block is
    committed, the engine refuses with a clear error pointing at the
    planned next block."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        agent.commit_season_plan(
            "2026-10-15", None,
            blocks=[
                {"block_type": "volume", "duration_weeks": 4, "rationale": "v2"},
                {"block_type": "strength", "duration_weeks": 4, "rationale": "s1"},
            ],
        )
        out = agent.commit_block("strength", 4, "skipping ahead", [],
                                 weekly_plan={"days": [
                                     {"label": "Mon — Squat", "exercises": [
                                         {"name": "Squat", "lift": "squat",
                                          "role": "primary", "sets": 4, "reps": 3,
                                          "intensity_pct": 0.85, "rpe_cap": 9}]},
                                     {"label": "Tue — Bench", "exercises": [
                                         {"name": "Bench", "lift": "bench",
                                          "role": "primary", "sets": 4, "reps": 3,
                                          "intensity_pct": 0.85, "rpe_cap": 9}]},
                                 ]})
        assert "error" in out
        assert "REFUSED" in out["error"]
        assert "volume" in out["error"]  # tells LLM what was planned
        # State unchanged — no block became active.
        assert agent.state["block"]["type"] is None
        # season_plan unchanged — no entries popped.
        assert len(agent.state["season_plan"]["blocks"]) == 2


def test_commit_override_season_plan_flag_allows_deviation():
    """Internal escape hatch: when the LLM has explicitly revised the
    plan via update_season_plan first (or has lifter approval), it can
    pass override_season_plan=True to bypass the mismatch refusal."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        agent.commit_season_plan(
            "2026-10-15", None,
            blocks=[
                {"block_type": "volume", "duration_weeks": 4, "rationale": "v"},
            ],
        )
        out = agent.commit_block("strength", 4, "override", [],
                                 weekly_plan={"days": [
                                     {"label": "Mon — Squat", "exercises": [
                                         {"name": "Squat", "lift": "squat",
                                          "role": "primary", "sets": 4, "reps": 3,
                                          "intensity_pct": 0.85, "rpe_cap": 9},
                                         *_generic_accessories("squat")]},
                                     {"label": "Tue — Bench", "exercises": [
                                         {"name": "Bench", "lift": "bench",
                                          "role": "primary", "sets": 4, "reps": 3,
                                          "intensity_pct": 0.85, "rpe_cap": 9},
                                         *_generic_accessories("bench")]},
                                 ]},
                                 override_season_plan=True)
        assert "committed" in out


def test_archive_stores_weekly_plan_in_history():
    """Past blocks must persist their weekly_plan so the Season Plan UI
    can show the program that was run when the user expands a completed
    block's card."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        plan = {"days": [
            {"label": "Mon — Squat", "exercises": [
                {"name": "Squat", "lift": "squat", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
                *_generic_accessories("squat")]},
            {"label": "Tue — Bench", "exercises": [
                {"name": "Bench", "lift": "bench", "role": "primary",
                 "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9},
                *_generic_accessories("bench")]},
        ]}
        agent.commit_block("volume", 4, "test", [], weekly_plan=plan)
        agent.archive_current_block("solid block, all lifts up")
        record = agent.state["block_history"][-1]
        assert "weekly_plan" in record
        assert len(record["weekly_plan"]["days"]) == 2
        assert record["outcome"] == "solid block, all lifts up"


def test_propose_block_result_includes_start_date_summary():
    """The LLM was hallucinating dates ('starts on June 1' when the real
    date was July 6). The return value now has a pre-formatted 'summary'
    string the LLM can quote — defense against date hallucination."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        out = agent.commit_block("volume", 4, "test", [],
                                 weekly_plan={"days": [
                                     {"label": "Monday — Heavy Squat",
                                      "exercises": [
                                          {"name": "Squat", "lift": "squat",
                                           "role": "primary", "sets": 4, "reps": 3,
                                           "intensity_pct": 0.75, "rpe_cap": 8},
                                          *_generic_accessories("squat")]},
                                     {"label": "Tuesday — Heavy Bench",
                                      "exercises": [
                                          {"name": "Bench", "lift": "bench",
                                           "role": "primary", "sets": 4, "reps": 3,
                                           "intensity_pct": 0.75, "rpe_cap": 8},
                                          *_generic_accessories("bench")]},
                                 ]})
        assert "started_at" in out
        assert "summary" in out
        # Summary must contain the actual date in BOTH formats.
        assert out["started_at"] in out["summary"]
        assert "COMMITTED" in out["summary"]


def test_commit_season_plan_preserves_competition_date_when_omitted():
    """The real bug: LLM called propose_season twice — second call omitted
    competition_date, which used to overwrite the existing one with None.
    Now it MERGES: omitted (None) values preserve what was already set."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        # First call sets the meet date.
        agent.commit_season_plan(
            "2026-10-12", {"squat": 180, "bench": 130, "deadlift": 210},
            blocks=[{"block_type": "volume", "duration_weeks": 4,
                     "rationale": "v"}],
        )
        # Second call omits date + lifts (LLM forgot, or just revising blocks).
        agent.commit_season_plan(
            None, None,
            blocks=[
                {"block_type": "volume", "duration_weeks": 4, "rationale": "v"},
                {"block_type": "strength", "duration_weeks": 4, "rationale": "s"},
            ],
        )
        sp = agent.state["season_plan"]
        # Date + lifts preserved.
        assert sp["competition_date"] == "2026-10-12"
        assert sp["competition_lifts"]["deadlift"] == 210
        # Blocks updated.
        assert len(sp["blocks"]) == 2


def test_commit_season_plan_updates_competition_date_when_supplied():
    """Sanity: passing a NEW date explicitly DOES update it. The merge
    only kicks in when None is passed."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        agent.commit_season_plan(
            "2026-10-12", None,
            blocks=[{"block_type": "volume", "duration_weeks": 4,
                     "rationale": "v"}],
        )
        # Lifter explicitly moved the meet.
        agent.commit_season_plan(
            "2026-11-01", None,
            blocks=[{"block_type": "volume", "duration_weeks": 4,
                     "rationale": "v"}],
        )
        assert agent.state["season_plan"]["competition_date"] == "2026-11-01"


def test_season_plan_outline_persists():
    """Each upcoming block can carry an 'outline' (list of day labels +
    exercise names, no weights) so the UI can render a preview."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        agent.commit_season_plan(
            "2026-10-15", None,
            blocks=[
                {"block_type": "volume", "duration_weeks": 4,
                 "rationale": "rebuild",
                 "outline": [
                     {"label": "Mon — Heavy Squat",
                      "exercises": ["Squat (top set)", "Squat (back-offs)",
                                    "RDL", "Leg press"]},
                     {"label": "Tue — Heavy Bench",
                      "exercises": ["Bench (top set)", "Bench (back-offs)",
                                    "DB row", "Tricep pushdown"]},
                 ]},
            ],
        )
        block = agent.state["season_plan"]["blocks"][0]
        assert len(block["outline"]) == 2
        assert "Squat (top set)" in block["outline"][0]["exercises"]


def test_season_plan_outline_defaults_to_empty_list():
    """Outline is optional — entries without one persist as empty list."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        agent.commit_season_plan(
            None, None,
            blocks=[{"block_type": "volume", "duration_weeks": 4,
                     "rationale": "x"}],
        )
        assert agent.state["season_plan"]["blocks"][0]["outline"] == []


def test_update_season_plan_partial():
    """update_season_plan changes only the supplied fields."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        agent.commit_season_plan(
            "2026-10-15", {"squat": 180},
            blocks=[{"block_type": "volume", "duration_weeks": 4,
                     "rationale": "x"}],
        )
        # Update only competition_date.
        agent.update_season_plan(competition_date="2026-11-01")
        sp = agent.state["season_plan"]
        assert sp["competition_date"] == "2026-11-01"
        assert sp["competition_lifts"] == {"squat": 180}  # untouched
        assert len(sp["blocks"]) == 1  # untouched


def test_dispatch_get_season_plan_tool():
    """ReAct tool surfaces past + current + future blocks."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path, with_block=True)
        agent.commit_season_plan(
            "2026-10-15", None,
            blocks=[
                {"block_type": "strength", "duration_weeks": 4, "rationale": "after"},
            ],
        )
        out = reasoning.dispatch_tool(agent.state, "get_season_plan", {})
        assert out["competition_date"] == "2026-10-15"
        assert out["active_block"] is not None
        assert out["active_block"]["type"] == "volume"
        assert len(out["upcoming_blocks"]) == 1


def test_dispatch_propose_season_tool_commits():
    """The LLM's propose_season tool writes state via the agent."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        agent = _new_agent(path)
        out = reasoning.dispatch_tool(
            agent.state, "propose_season",
            {"competition_date": "2026-10-15",
             "competition_lifts": {"squat": 180, "bench": 130, "deadlift": 210},
             "blocks_plan": [
                 {"block_type": "volume", "duration_weeks": 4,
                  "rationale": "rebuild", "planned_start": "2026-06-01"},
                 {"block_type": "strength", "duration_weeks": 4,
                  "rationale": "express", "planned_start": "2026-07-06"},
                 {"block_type": "peaking", "duration_weeks": 3,
                  "rationale": "taper", "planned_start": "2026-08-10"},
             ]},
            agent=agent,
        )
        assert "committed_season_plan" in out
        assert len(agent.state["season_plan"]["blocks"]) == 3


def test_dispatch_propose_season_without_agent_errors():
    """Stateful tool gracefully errors without an agent."""
    out = reasoning.dispatch_tool(
        {}, "propose_season",
        {"blocks_plan": [{"block_type": "volume", "duration_weeks": 4,
                          "rationale": "x"}]},
    )
    assert "error" in out


def test_season_tools_registered_in_schemas():
    """The 3 new season-plan tools must appear in TOOL_SCHEMAS."""
    schema_names = {s["name"] for s in reasoning.TOOL_SCHEMAS}
    assert {"get_season_plan", "propose_season",
            "update_season_plan"}.issubset(schema_names)


def test_block_stacking_intermediate_matches_ben_schedule():
    """Ben's exact schedule (406rClFwNxY @ 8:45): intermediate stacks 3
    blocks toward a PR — block 1 RPE 7.5-8, block 2 RPE 8-8.5, block 3
    RPE 9-10."""
    sched = expert_knowledge.block_stacking_schedule("intermediate")
    assert sched["blocks_to_pr"] == 3
    blocks_sched = sched["schedule"]
    assert blocks_sched[0]["position"] == 1
    assert blocks_sched[0]["top_set_rpe_low"] == 7.5
    assert blocks_sched[0]["top_set_rpe_high"] == 8.0
    assert blocks_sched[0]["role"] == "deposit"
    assert blocks_sched[1]["top_set_rpe_low"] == 8.0
    assert blocks_sched[1]["top_set_rpe_high"] == 8.5
    assert blocks_sched[2]["role"] == "PR"
    assert blocks_sched[2]["top_set_rpe_low"] == 9.0
    assert blocks_sched[2]["top_set_rpe_high"] == 10.0


def test_block_stacking_novice_prs_every_block():
    """Beginners can demonstrate progress block-to-block (406rClFwNxY @ 6:54)."""
    sched = expert_knowledge.block_stacking_schedule("novice")
    assert sched["blocks_to_pr"] == 1
    assert sched["schedule"][0]["role"] == "PR"


def test_block_stacking_advanced_uses_4_blocks():
    """Advanced lifters need more deposits (406rClFwNxY @ 11:43)."""
    sched = expert_knowledge.block_stacking_schedule("advanced")
    assert sched["blocks_to_pr"] == 4
    assert len(sched["schedule"]) == 4
    # Final block is the PR.
    assert sched["schedule"][-1]["role"] == "PR"
    assert sched["schedule"][-1]["top_set_rpe_high"] == 10.0


def test_block_stacking_unknown_falls_back_to_intermediate():
    """Robustness: unknown experience level falls back to the most-common path."""
    sched = expert_knowledge.block_stacking_schedule("space_alien")
    assert sched["blocks_to_pr"] == 3  # intermediate default


def test_block_stacking_principles_present():
    """Coaching principles should be cite-able from the dict."""
    principles = expert_knowledge.BLOCK_STACKING["principles"]
    assert any("deposit" in p.lower() for p in principles)
    assert any("max every block" in p.lower() or "max each block" in p.lower()
               or "max each" in p.lower() or "every block" in p.lower()
               for p in principles)


def test_time_to_peak_default_5_weeks():
    """Ben's personal time-to-peak (406rClFwNxY @ 5:36)."""
    ttp = expert_knowledge.time_to_peak()
    assert ttp["default_weeks"] == 5
    assert ttp["range_weeks"] == (4, 6)
    assert len(ttp["process"]) >= 3  # multi-step process documented


def test_rpe_usage_core_principle_matches_ben():
    """Klt4SLA64iE @ 4:50 — RPE dictates the weight, not vice versa."""
    rpe = expert_knowledge.rpe_usage_principles()
    assert "rpe dictates" in rpe["core_principle"].lower()
    assert "klt4sla64ie" in rpe["source"].lower()
    # Common-mistake explanation present.
    assert "calculator" in rpe["common_mistake"].lower() or \
           "pre-pick" in rpe["common_mistake"].lower()
    # How-to-use list has at least 4 actionable items.
    assert len(rpe["how_to_use"]) >= 4


def test_dispatch_get_block_stacking_schedule_tool():
    """ReAct tool surfaces the schedule with principles."""
    out = reasoning.dispatch_tool({}, "get_block_stacking_schedule",
                                  {"experience": "intermediate"})
    assert "stacking" in out
    assert out["stacking"]["blocks_to_pr"] == 3
    assert "principles" in out
    # Tool default — no experience supplied.
    out_default = reasoning.dispatch_tool({}, "get_block_stacking_schedule", {})
    assert "stacking" in out_default


def test_dispatch_get_rpe_usage_principles_tool():
    """ReAct tool returns the RPE-usage principle structure."""
    out = reasoning.dispatch_tool({}, "get_rpe_usage_principles", {})
    assert "core_principle" in out
    assert "common_mistake" in out
    assert "how_to_use" in out
    assert "stress_dose_framing" in out


def test_new_periodization_tools_in_schemas():
    """Both new tools must appear in TOOL_SCHEMAS."""
    names = {s["name"] for s in reasoning.TOOL_SCHEMAS}
    assert "get_block_stacking_schedule" in names
    assert "get_rpe_usage_principles" in names


# --- Alexander Bromley KB --------------------------------------------------

def test_bromley_accessory_progression_has_three_methods():
    """Rep-range method (default), DDP, block-level shifting."""
    p = expert_knowledge.accessory_progression_methods()
    assert "rep_range_method" in p["methods"]
    assert "dynamic_double_progression" in p["methods"]
    assert "block_level_shifting" in p["methods"]
    # Effort targets defined.
    et = p["effort_targets"]
    assert et["starting_rpe_week_1"] == 7.0
    assert et["peak_rpe_block_end"] == 10.0
    # Cites Bromley source.
    assert "czEsWD56hCU" in p["source"]


def test_bromley_progression_rep_range_is_preferred_default():
    """The preferred method (per the video) is rep-range, not DDP."""
    p = expert_knowledge.accessory_progression_methods()
    rr = p["methods"]["rep_range_method"]
    assert "preferred" in rr["name"].lower() or \
           "default" in rr["name"].lower()
    # DDP is positioned as first-accessory-only.
    ddp = p["methods"]["dynamic_double_progression"]
    assert ("first" in ddp["best_for"].lower()
            or "fresh" in ddp["best_for"].lower())


def test_bromley_progression_warns_against_rpe10_on_heavy_compound():
    """Spinal-loading accessories shouldn't hit RPE 10 even though
    'go ham' applies to lighter isolation."""
    p = expert_knowledge.accessory_progression_methods()
    exc = p["effort_targets"]["exception"]
    assert "deficit" in exc.lower() or "good morning" in exc.lower() \
           or "compound" in exc.lower()


def test_classify_movement_leverage_advantaged_examples():
    """Board press, pin press, rack pull, trap bar = advantaged
    (peak-phase CNS work)."""
    for ex in ("Board press", "Pin press", "Rack pull",
               "Trap bar deadlift", "Banded squat"):
        out = expert_knowledge.classify_movement_leverage(ex)
        assert out["category"] == "advantaged", (
            f"{ex!r} should be advantaged, got {out['category']}")
        # Advantaged is for strength + peaking.
        assert "peaking" in out["best_for_phase"] or \
               "strength" in out["best_for_phase"]


def test_classify_movement_leverage_disadvantaged_examples():
    """Deficit DL, long-pause bench, pause squat, front squat,
    stiff-leg DL = disadvantaged (volume / hypertrophy)."""
    for ex in ("Deficit deadlift", "Long-pause bench",
               "Pause squat", "Front squat", "Stiff-leg deadlift",
               "Spoto press"):
        out = expert_knowledge.classify_movement_leverage(ex)
        assert out["category"] == "disadvantaged", (
            f"{ex!r} should be disadvantaged, got {out['category']}")
        # Disadvantaged is for volume + technique.
        assert "volume" in out["best_for_phase"] or \
               "technique" in out["best_for_phase"]


def test_classify_movement_leverage_isolation():
    """Cable pushdown, lateral raise, leg curl = lighter isolation."""
    for ex in ("Cable tricep pushdown", "DB lateral raise",
               "Lying leg curl", "Leg extension", "Face pull"):
        out = expert_knowledge.classify_movement_leverage(ex)
        assert out["category"] == "lighter_isolation", (
            f"{ex!r} should be lighter_isolation, got {out['category']}")


def test_classify_movement_leverage_neutral_compound():
    """Good morning, dip, BSS, JM press = neutral compound accessory."""
    for ex in ("Good morning", "Weighted dip",
               "Bulgarian split squat", "JM press", "Barbell row"):
        out = expert_knowledge.classify_movement_leverage(ex)
        assert out["category"] == "neutral_compound_accessory", (
            f"{ex!r} should be neutral compound, got {out['category']}")


def test_classify_movement_leverage_unknown_returns_unknown():
    """Unrecognized exercise → unknown category, no crash."""
    out = expert_knowledge.classify_movement_leverage("Atlas stone toss")
    assert out["category"] == "unknown"


def test_classify_movement_leverage_empty_input():
    """Empty / None input is handled gracefully."""
    assert expert_knowledge.classify_movement_leverage("")["category"] == "unknown"
    assert expert_knowledge.classify_movement_leverage(None)["category"] == "unknown"


def test_leverage_for_block_volume_favors_disadvantaged():
    """Volume block: disadvantaged primary."""
    out = expert_knowledge.leverage_for_block("volume")
    assert out["primary_leverage"] == "disadvantaged"


def test_leverage_for_block_peaking_favors_advantaged_drops_volume():
    """Peaking: advantaged primary, drop high-volume isolation."""
    out = expert_knowledge.leverage_for_block("peaking")
    assert out["primary_leverage"] == "advantaged"
    assert out["tertiary_leverage"] == "drop"


def test_leverage_for_block_strength_favors_neutral_compound():
    out = expert_knowledge.leverage_for_block("strength")
    assert out["primary_leverage"] == "neutral_compound_accessory"


def test_leverage_for_block_unknown_returns_error():
    out = expert_knowledge.leverage_for_block("bogus_block")
    assert "error" in out


def test_weakness_targeting_returns_both_options():
    """Pattern targeting + muscle targeting + novice caveat."""
    out = expert_knowledge.weakness_targeting_options()
    assert "pattern_targeting" in out
    assert "muscle_targeting" in out
    assert "novice_caveat" in out
    # Pattern examples cite specific sticking points.
    pattern = out["pattern_targeting"]["examples"]
    assert any("lockout" in p["weakness"].lower() for p in pattern)
    assert any("hole" in p["weakness"].lower() for p in pattern)


def test_dormant_muscle_protocol_has_rep_range_15_20():
    """High-rep wake-up = 15-20 reps per Bromley."""
    p = expert_knowledge.dormant_muscle_protocol()
    assert p["prescription"]["reps_range"] == (15, 20)
    assert p["prescription"]["rpe_range"][0] >= 8.0


def test_dormant_muscle_protocol_squat_includes_rectus_femoris():
    """Squat day's dormant muscle = rectus femoris (biarticular quad)."""
    p = expert_knowledge.dormant_muscle_protocol("squat")
    assert "recommended_for_lift" in p
    muscles = [m["muscle"] for m in p["recommended_for_lift"]]
    assert "rectus femoris" in muscles


def test_dormant_muscle_protocol_deadlift_includes_erectors():
    """DL day dormant muscle = spinal erectors (Bromley's case study)."""
    p = expert_knowledge.dormant_muscle_protocol("deadlift")
    muscles = [m["muscle"] for m in p["recommended_for_lift"]]
    assert "spinal erectors" in muscles


def test_session_order_has_four_positions():
    """Main → secondary main → primary accessory → secondary accessory."""
    p = expert_knowledge.session_order_principle()
    grad = p["session_gradient"]
    assert len(grad) == 4
    # Precision decreases down the gradient.
    precisions = [g["performance_precision"].lower() for g in grad]
    assert "high" in precisions[0]
    assert "high" in precisions[1]
    assert "low" in precisions[3] or "stress" in precisions[3]


def test_session_order_lighter_isolation_rpe_8_10():
    """RPE 8-10 OK on lighter isolation (Bromley's 'go ham' rule)."""
    p = expert_knowledge.session_order_principle()
    last = p["session_gradient"][-1]
    assert "8" in last["rpe_target"] and "10" in last["rpe_target"]


# --- Bromley ReAct tools ---------------------------------------------------

def test_dispatch_get_accessory_progression_methods():
    out = reasoning.dispatch_tool(
        {}, "get_accessory_progression_methods", {})
    assert "methods" in out
    assert "rep_range_method" in out["methods"]


def test_dispatch_classify_movement_leverage_advantaged():
    out = reasoning.dispatch_tool(
        {}, "classify_movement_leverage", {"exercise_name": "Board press"})
    assert out["category"] == "advantaged"


def test_dispatch_classify_movement_leverage_disadvantaged():
    out = reasoning.dispatch_tool(
        {}, "classify_movement_leverage",
        {"exercise_name": "Deficit deadlift"})
    assert out["category"] == "disadvantaged"


def test_dispatch_get_leverage_for_block_peaking():
    out = reasoning.dispatch_tool(
        {}, "get_leverage_for_block", {"block_type": "peaking"})
    assert out["primary_leverage"] == "advantaged"


def test_dispatch_get_weakness_targeting_options():
    out = reasoning.dispatch_tool(
        {}, "get_weakness_targeting_options", {})
    assert "pattern_targeting" in out
    assert "muscle_targeting" in out


def test_dispatch_get_dormant_muscle_protocol_squat():
    out = reasoning.dispatch_tool(
        {}, "get_dormant_muscle_protocol", {"lift": "squat"})
    muscles = [m["muscle"] for m in out["recommended_for_lift"]]
    assert "rectus femoris" in muscles


def test_dispatch_get_session_order_principle():
    out = reasoning.dispatch_tool({}, "get_session_order_principle", {})
    assert "session_gradient" in out
    assert len(out["session_gradient"]) == 4


def test_all_bromley_tools_in_schemas():
    """All 6 Bromley tools must appear in TOOL_SCHEMAS."""
    names = {s["name"] for s in reasoning.TOOL_SCHEMAS}
    expected = {"get_accessory_progression_methods",
                "classify_movement_leverage",
                "get_leverage_for_block",
                "get_weakness_targeting_options",
                "get_dormant_muscle_protocol",
                "get_session_order_principle"}
    assert expected.issubset(names), (
        f"missing tools: {expected - names}")


# --- block wave: weeks must be strictly distinct through week 5 ------------

def test_block_wave_week4_and_week5_differ():
    """In a 5-week block, week 4 and week 5 must produce different
    prescriptions (a wave shorter than the block would clamp them to the
    same entry)."""
    for bt in blocks.VALID_BLOCK_TYPES:
        wk4 = blocks.week_modifier(bt, 4)
        wk5 = blocks.week_modifier(bt, 5)
        # At least intensity_mult OR rpe_delta OR set_delta must differ.
        diff = (wk4["intensity_mult"] != wk5["intensity_mult"]
                or wk4["rpe_delta"] != wk5["rpe_delta"]
                or wk4["set_delta"] != wk5["set_delta"]
                or wk4["rep_delta"] != wk5["rep_delta"])
        assert diff, f"{bt}: week 4 == week 5 (wave too short)"


def test_block_wave_all_5_weeks_distinct():
    """Weeks 1-5 of every block_type's wave must be pairwise distinct on
    at least one of (intensity_mult, rpe_delta, set_delta, rep_delta)."""
    for bt in blocks.VALID_BLOCK_TYPES:
        seen = set()
        for wk in range(1, 6):
            mod = blocks.week_modifier(bt, wk)
            key = (mod["intensity_mult"], mod["rpe_delta"],
                   mod["set_delta"], mod["rep_delta"])
            assert key not in seen, (
                f"{bt}: week {wk} duplicates an earlier week's modifiers")
            seen.add(key)


def test_block_wave_week6_supported():
    """6-week blocks must also render without clamping repeats (the wave
    has 6 entries). Week 6 differs from week 5 on at least one axis."""
    for bt in blocks.VALID_BLOCK_TYPES:
        wk5 = blocks.week_modifier(bt, 5)
        wk6 = blocks.week_modifier(bt, 6)
        diff = (wk5["intensity_mult"] != wk6["intensity_mult"]
                or wk5["rpe_delta"] != wk6["rpe_delta"]
                or wk5["set_delta"] != wk6["set_delta"]
                or wk5["rep_delta"] != wk6["rep_delta"])
        assert diff, f"{bt}: week 5 == week 6"


def test_block_wave_clamps_for_7_plus():
    """Beyond 6 weeks the wave clamps gracefully to the last entry —
    not a bug, just a hard ceiling. Week 7 == Week 6 is acceptable."""
    for bt in blocks.VALID_BLOCK_TYPES:
        wk6 = blocks.week_modifier(bt, 6)
        wk8 = blocks.week_modifier(bt, 8)
        assert wk6 == wk8


def test_block_wave_rpe_does_not_exceed_safe_ceiling_at_week6():
    """6-week block with a peak-RPE primary (rpe_cap 9.0 per the
    block-stack ceiling) must not push past 10.0 in the final week."""
    primary = {"name": "Squat", "lift": "squat", "role": "primary",
               "sets": 4, "reps": 3, "intensity_pct": 0.85, "rpe_cap": 9.0}
    for bt in blocks.VALID_BLOCK_TYPES:
        out = blocks.compute_exercise_load(primary, 150, bt, 6)
        assert out["rpe_cap"] <= 10.0, (
            f"{bt}: week-6 RPE {out['rpe_cap']} exceeds failure ceiling")


# --- macrocycle sizing ------------------------------------------------------

def test_compute_macrocycle_sizing_5_months_rebuilding():
    """5 months out (22 weeks), rebuilding from injury → 4 blocks
    (Volume → Volume → Strength → Peaking)."""
    out = expert_knowledge.compute_macrocycle_sizing(
        weeks_to_comp=22, experience="intermediate",
        lifter_status="rebuilding from shoulder injury")
    assert out["num_blocks"] == 4
    types = [b["block_type"] for b in out["sequence"]]
    assert types == ["volume", "volume", "strength", "peaking"]
    # Final block is the PR / peak block.
    assert out["sequence"][-1]["role"] == "PR"
    # Earlier blocks are deposits.
    for b in out["sequence"][:-1]:
        assert b["role"] == "deposit"


def test_compute_macrocycle_sizing_3_months_advanced():
    out = expert_knowledge.compute_macrocycle_sizing(
        weeks_to_comp=13, experience="advanced", lifter_status="advanced")
    assert out["num_blocks"] == 3
    types = [b["block_type"] for b in out["sequence"]]
    assert types == ["strength", "strength", "peaking"]


def test_compute_macrocycle_sizing_6_months_default():
    out = expert_knowledge.compute_macrocycle_sizing(
        weeks_to_comp=26, experience="intermediate")
    assert out["num_blocks"] == 5
    types = [b["block_type"] for b in out["sequence"]]
    assert types[-1] == "peaking"
    assert types[0] == "volume"


def test_compute_macrocycle_sizing_2_months():
    """Short prep — 2 blocks, no volume phase."""
    out = expert_knowledge.compute_macrocycle_sizing(
        weeks_to_comp=8, experience="intermediate")
    assert out["num_blocks"] == 2
    types = [b["block_type"] for b in out["sequence"]]
    assert types == ["strength", "peaking"]


def test_compute_macrocycle_sizing_handles_extreme_short():
    """4 weeks or less = 1 block (peaking only). Edge case."""
    out = expert_knowledge.compute_macrocycle_sizing(weeks_to_comp=4)
    assert out["num_blocks"] == 1
    assert out["sequence"][0]["block_type"] == "peaking"


def test_compute_macrocycle_sizing_returns_planned_weeks():
    """Each sequence entry carries its planned_weeks (5 wks each)."""
    out = expert_knowledge.compute_macrocycle_sizing(weeks_to_comp=22)
    for b in out["sequence"]:
        assert b["planned_weeks"] == 5


def test_compute_macrocycle_sizing_cites_source():
    """Has a source citation per the project's traceability rule."""
    out = expert_knowledge.compute_macrocycle_sizing(weeks_to_comp=22)
    assert "406rClFwNxY" in out["source"]


def test_dispatch_compute_macrocycle_sizing():
    """ReAct tool dispatches correctly."""
    out = reasoning.dispatch_tool(
        {}, "compute_macrocycle_sizing",
        {"weeks_to_comp": 22, "lifter_status": "rebuilding from injury"})
    assert out["num_blocks"] == 4


def test_compute_macrocycle_sizing_tool_in_schema():
    names = {s["name"] for s in reasoning.TOOL_SCHEMAS}
    assert "compute_macrocycle_sizing" in names


# --- Sebastian Oreb KB ------------------------------------------------------

def test_sebastian_exercise_categories_has_all_8_categories():
    """The 8-category structural balance model (tjC6ilWMEMM @ 22:34)."""
    cats = expert_knowledge.exercise_categories()
    expected = {"horizontal_push", "vertical_push",
                "horizontal_pull", "vertical_pull",
                "knee_dominant_squat", "hip_dominant_squat",
                "hip_hinge", "isolation_joint_health"}
    assert expected.issubset(set(cats.keys()))
    # Every category has examples + a source citation.
    for key, info in cats.items():
        assert info["examples"], f"{key} has no example exercises"
        assert "tjC6ilWMEMM" in info["source"]


def test_sebastian_knee_vs_hip_dominant_split_correctly():
    """High-bar / stiletto = knee-dominant. Low-bar / paused = hip-dominant.
    These ARE different categories per Sebastian's framework — Bulgarian
    split squat (forward lean) belongs to hip-dominant, not knee-dominant."""
    cats = expert_knowledge.exercise_categories()
    knee = " ".join(cats["knee_dominant_squat"]["examples"]).lower()
    hip = " ".join(cats["hip_dominant_squat"]["examples"]).lower()
    assert "high-bar" in knee or "stiletto" in knee
    assert "front squat" in knee
    assert "low-bar" in hip
    assert "paused squat" in hip


def test_check_redundancy_detects_same_category():
    """Two exercises from the same category are flagged as redundant."""
    out = expert_knowledge.check_redundancy(
        "Bench press", "DB bench press")
    assert out["redundant"] is True
    assert out["shared_category"] == "horizontal_push"


def test_check_redundancy_passes_different_categories():
    """Primary squat + secondary pull = different categories = NOT redundant."""
    out = expert_knowledge.check_redundancy(
        "Low-bar back squat", "Barbell row")
    assert out["redundant"] is False


def test_check_redundancy_flags_squat_secondary_bug():
    """The exact example from Sebastian's video: low-bar squat primary +
    Bulgarian split squat (forward lean) secondary = both hip-dominant =
    redundant. The lifter should pick a knee-dominant secondary instead."""
    out = expert_knowledge.check_redundancy(
        "Low-bar back squat", "Bulgarian split squat (forward lean)")
    assert out["redundant"] is True
    assert out["shared_category"] == "hip_dominant_squat"


def test_neglected_muscle_targets_squat_returns_rectus_femoris():
    """Squat-day accessory checklist must include rectus femoris isolation."""
    out = expert_knowledge.neglected_muscle_targets("squat")
    assert isinstance(out, list)
    keys = [e["key"] for e in out]
    assert "rectus_femoris" in keys
    rf = next(e for e in out if e["key"] == "rectus_femoris")
    assert "leg extension" in rf["isolation"].lower()


def test_neglected_muscle_targets_deadlift_returns_short_head_BF():
    """DL-day accessory checklist must include short head biceps femoris
    (only knee flexion targets it) + glute shortened position (DL only
    loads lengthened glute)."""
    out = expert_knowledge.neglected_muscle_targets("deadlift")
    keys = [e["key"] for e in out]
    assert "short_head_biceps_femoris" in keys
    assert "glute_shortened_position" in keys


def test_neglected_muscle_targets_bench_returns_lateral_delt():
    out = expert_knowledge.neglected_muscle_targets("bench")
    keys = [e["key"] for e in out]
    assert "lateral_delt_after_horizontal_press" in keys


def test_training_splits_2day_recommends_whole_body_not_upper_lower():
    """Sebastian: 2-day = whole-body BOTH days. Upper/lower for 2-day
    saturates muscles → junk volume (x_uhGTQGrAg @ 17:53)."""
    split = expert_knowledge.training_splits(2)
    assert split["preferred"] == "whole_body"
    assert "upper/lower" in split["avoid"].lower()
    # Both days are whole-body, not upper / lower.
    structure = split["structure"]
    assert len(structure) == 2
    for day in structure:
        assert "whole body" in day["focus"].lower()


def test_training_splits_3day_avoids_ppl():
    """3-day PPL has insufficient quad + push frequency."""
    split = expert_knowledge.training_splits(3)
    assert "insufficient" in split["avoid_ppl"].lower() or \
           "1x/wk" in split["avoid_ppl"].lower()


def test_training_splits_4day_is_sebastians_preferred():
    """4-day is the 'five gold stars' default for intermediate+."""
    split = expert_knowledge.training_splits(4)
    assert split["preferred"] == "sebastian_4day"
    # Structure: upper-horiz / lower-quad / upper-vert / lower-post.
    days = [d["focus"].lower() for d in split["structure"]]
    assert any("horizontal" in d for d in days)
    assert any("vertical" in d for d in days)
    assert any("quad" in d for d in days)
    assert any("posterior" in d for d in days)


def test_training_splits_5day_prefers_2sq_1dl_2upper():
    """Sebastian's 5-day: 2 squat + 1 deadlift + 2 upper. DL needs more
    recovery than squat (disagrees with Poliquin's symmetric split)."""
    split = expert_knowledge.training_splits(5)
    assert split["preferred"] == "sebastian_5day"
    focuses = [d["focus"].lower() for d in split["structure"]]
    sq_days = sum(1 for f in focuses if "squat" in f)
    dl_days = sum(1 for f in focuses if "deadlift" in f)
    assert sq_days == 2
    assert dl_days == 1


def test_training_splits_clamps_out_of_range():
    """1-day not catalogued, 7-day not recommended — clamp to [2, 6]."""
    assert expert_knowledge.training_splits(1)["preferred"] == \
           expert_knowledge.training_splits(2)["preferred"]
    assert expert_knowledge.training_splits(8)["preferred"] == \
           expert_knowledge.training_splits(6)["preferred"]


def test_five_step_model_has_all_5_phases():
    """Learn-to-move → work-capacity → hypertrophy → strength → peaking."""
    model = expert_knowledge.five_step_model()
    assert len(model) == 5
    names = [phase["name"].lower() for phase in model]
    assert "learn to move" in names[0]
    assert "work capacity" in names[1]
    assert "hypertrophy" in names[2]
    assert "strength" in names[3]
    assert "peaking" in names[4]


def test_five_step_model_hypertrophy_uses_muscle_group_denom():
    """Phase 3 hypertrophy: 10-20 sets per MUSCLE GROUP per week."""
    h = expert_knowledge.five_step_model(3)
    assert h["volume_target"] == (10, 20)
    assert "muscle group" in h["volume_unit"].lower()


def test_five_step_model_strength_uses_movement_pattern_denom():
    """Phase 4 strength: 6-12 sets per MOVEMENT PATTERN per week (Rhea &
    colleagues 2009). Different denominator from hypertrophy."""
    s = expert_knowledge.five_step_model(4)
    assert s["volume_target"] == (6, 12)
    assert "movement pattern" in s["volume_unit"].lower()
    assert s["intensity_pct_min"] >= 0.80


def test_five_step_model_peaking_skips_aesthetics():
    """Aesthetics goals SKIP the peaking phase — peak is body comp via
    nutrition, not a 1RM peak (xZ2QTewSMuk @ 43:09)."""
    p = expert_knowledge.five_step_model(5)
    assert "aesthetics" in p["skip_for"].lower()


def test_loadability_rules_stretched_position_capped():
    """Stretched-position exercises (stiletto squat, flared bench,
    deficit DL) max ~75% intensity, NEVER 1RM (highest injury risk)."""
    rules = expert_knowledge.loadability_rules()
    stretched = next(r for r in rules
                     if r["profile"] == "compound_stretched_position")
    assert stretched["max_intensity_pct"] <= 0.80
    # Examples include the known high-risk movements.
    examples_txt = " ".join(stretched["examples"]).lower()
    assert "stiletto" in examples_txt or "deficit" in examples_txt


def test_volume_targets_hypertrophy_vs_strength_differ():
    """The critical distinction: different denominators + different ranges."""
    h = expert_knowledge.volume_targets("hypertrophy")
    s = expert_knowledge.volume_targets("strength")
    assert h["unit"] != s["unit"]
    assert "muscle group" in h["unit"].lower()
    assert "movement pattern" in s["unit"].lower()
    assert h["sets_range"] == (10, 20)
    assert s["sets_range"] == (6, 12)


# --- Sebastian ReAct tools --------------------------------------------------

def test_dispatch_get_exercise_categories():
    out = reasoning.dispatch_tool({}, "get_exercise_categories", {})
    assert "categories" in out
    assert "knee_dominant_squat" in out["categories"]
    assert "tjC6ilWMEMM" in out["source"]


def test_dispatch_check_exercise_redundancy_flags_same_plane():
    out = reasoning.dispatch_tool(
        {}, "check_exercise_redundancy",
        {"ex_a": "Bench press", "ex_b": "Flat DB bench press"})
    assert out["redundant"] is True


def test_dispatch_check_exercise_redundancy_passes_different():
    out = reasoning.dispatch_tool(
        {}, "check_exercise_redundancy",
        {"ex_a": "Low-bar back squat", "ex_b": "Bench press"})
    assert out["redundant"] is False


def test_dispatch_get_neglected_muscle_targets_squat():
    out = reasoning.dispatch_tool(
        {}, "get_neglected_muscle_targets", {"main_lift": "squat"})
    keys = [e["key"] for e in out["neglected_muscle_targets"]]
    assert "rectus_femoris" in keys


def test_dispatch_recommend_training_split_4day():
    out = reasoning.dispatch_tool(
        {}, "recommend_training_split",
        {"days_per_week": 4, "experience": "intermediate"})
    assert out["split"]["preferred"] == "sebastian_4day"


def test_dispatch_get_five_step_model_phase_3():
    out = reasoning.dispatch_tool({}, "get_five_step_model", {"phase": 3})
    assert "hypertrophy" in out["model"]["name"].lower()


def test_dispatch_get_loadability_rules():
    out = reasoning.dispatch_tool({}, "get_loadability_rules", {})
    profiles = [r["profile"] for r in out["rules"]]
    assert "compound_mid_range" in profiles
    assert "compound_stretched_position" in profiles
    assert "isolation" in profiles


def test_dispatch_get_volume_targets_strength():
    out = reasoning.dispatch_tool({}, "get_volume_targets",
                                  {"mode": "strength"})
    assert out["targets"]["sets_range"] == (6, 12)


def test_sebastian_tools_all_in_schemas():
    """All 7 new Sebastian tools must appear in TOOL_SCHEMAS."""
    names = {s["name"] for s in reasoning.TOOL_SCHEMAS}
    expected = {"get_exercise_categories", "check_exercise_redundancy",
                "get_neglected_muscle_targets", "recommend_training_split",
                "get_five_step_model", "get_loadability_rules",
                "get_volume_targets"}
    assert expected.issubset(names)


def test_dispatch_get_strength_variation():
    out = reasoning.dispatch_tool({}, "get_strength_variation",
                                  {"lift": "squat", "weakness": "depth"})
    assert any("Pin squat" in p["name"] for p in out["picks"])


def test_new_expert_tools_registered_in_schemas():
    """All five Ben-KB tools must appear in TOOL_SCHEMAS so the LLM sees them."""
    schema_names = {s["name"] for s in reasoning.TOOL_SCHEMAS}
    assert {"get_prescription_pattern", "lookup_accessories",
            "get_volume_landmark", "get_hypertrophy_checklist",
            "get_strength_variation"}.issubset(schema_names)


def test_expert_knowledge_accessory_rep_ranges_categories_present():
    """Ben's four prescription categories from QHGcZaDuwpQ / 6UPsJiD7QcE."""
    cats = expert_knowledge.ACCESSORY_REP_RANGES
    assert "hypertrophy_isolation" in cats
    assert "hypertrophy_compound" in cats
    assert "heavy_compound_substitute" in cats  # leg press / Bulgarian split squat
    assert "strength_variation" in cats
    # Strength-variation reps are 3-6 at RPE 5-7 (back-off zone, 60-75%).
    sv = cats["strength_variation"]
    assert sv["reps_low"] == 3 and sv["reps_high"] == 6
    assert sv["rpe_high"] <= 7.0
    assert sv["intensity_pct_low"] >= 0.55
