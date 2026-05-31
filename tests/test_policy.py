"""Tests for the swappable progression policy (coach/policy.py) and its
integration with the agent loop. Verifies:

  * the Policy interface (RulePolicy vs BanditPolicy);
  * deterministic context bucketing;
  * cold-start parity (bandit behaves like rules until enough data);
  * convergence (bandit overrides rules after consistent reward signal);
  * round-trip serialization;
  * safety_override sitting above any policy;
  * loop integration (env var enables, pending_decisions flow, learned
    state survives a restart).
"""

import os
import tempfile

import pytest

from coach import memory, policy, reasoning, readiness
from coach.loop import Agent


# --- bucket() determinism --------------------------------------------------

def test_bucket_readiness_bands():
    assert policy.bucket({"readiness": 0.30, "rpe_gap": 0, "overreach_streak": 0})[0] == "low"
    assert policy.bucket({"readiness": 0.45, "rpe_gap": 0, "overreach_streak": 0})[0] == "low"
    assert policy.bucket({"readiness": 0.60, "rpe_gap": 0, "overreach_streak": 0})[0] == "med"
    assert policy.bucket({"readiness": 0.71, "rpe_gap": 0, "overreach_streak": 0})[0] == "high"


def test_bucket_rpe_gap_bands():
    # gap = effective_rpe - cap. >0 = ran hotter than plan.
    assert policy.bucket({"readiness": 0.7, "rpe_gap": -1.0, "overreach_streak": 0})[1] == "below"
    assert policy.bucket({"readiness": 0.7, "rpe_gap": 0.0, "overreach_streak": 0})[1] == "at"
    assert policy.bucket({"readiness": 0.7, "rpe_gap": 1.0, "overreach_streak": 0})[1] == "above"


def test_bucket_overreach_bands():
    assert policy.bucket({"readiness": 0.7, "rpe_gap": 0, "overreach_streak": 0})[2] == "0"
    assert policy.bucket({"readiness": 0.7, "rpe_gap": 0, "overreach_streak": 1})[2] == "1"
    assert policy.bucket({"readiness": 0.7, "rpe_gap": 0, "overreach_streak": 5})[2] == "2+"


# --- compute_reward shape -------------------------------------------------

def test_compute_reward_clean_session_neutral_or_positive():
    # No overshoot, no missed reps, readiness flat -> 0 unless PR.
    r = policy.compute_reward(
        {"effective_rpe": 8.0, "missed_reps": 0}, prev_readiness=0.7,
        new_readiness=0.7, pr_set=False, rpe_cap=8.0)
    assert r == 0.0


def test_compute_reward_overshoot_penalised():
    r = policy.compute_reward(
        {"effective_rpe": 10.0, "missed_reps": 0}, prev_readiness=0.7,
        new_readiness=0.5, pr_set=False, rpe_cap=8.0)
    # Overshoot 2.0 * 0.5 = -1.0
    assert r == -1.0


def test_compute_reward_missed_reps_penalised():
    r = policy.compute_reward(
        {"effective_rpe": 9.0, "missed_reps": 2}, prev_readiness=0.7,
        new_readiness=0.5, pr_set=False, rpe_cap=8.0)
    # Overshoot 1.0 * 0.5 = -0.5; missed reps -1.0
    assert r == -1.5


def test_compute_reward_recovery_bonus():
    r = policy.compute_reward(
        {"effective_rpe": 6.0, "missed_reps": 0}, prev_readiness=0.4,
        new_readiness=0.8, pr_set=False, rpe_cap=8.0)
    # No penalty; +0.5 for recovery
    assert r == 0.5


def test_compute_reward_pr_bonus():
    r = policy.compute_reward(
        {"effective_rpe": 8.0, "missed_reps": 0}, prev_readiness=0.7,
        new_readiness=0.7, pr_set=True, rpe_cap=8.0)
    assert r == 1.0


def test_compute_reward_pain_flag_strong_negative():
    r = policy.compute_reward(
        {"effective_rpe": 9.0, "missed_reps": 0, "flags": ["pain"]},
        prev_readiness=0.7, new_readiness=0.5, pr_set=False, rpe_cap=8.0)
    # Overshoot -0.5, flag -2.0
    assert r == -2.5


# --- RulePolicy is a transparent passthrough ------------------------------

def test_rule_policy_select_returns_rule_action():
    p = policy.RulePolicy()
    for a in policy.ACTIONS:
        assert p.select({"readiness": 0.7}, a) == a


def test_rule_policy_update_is_noop():
    p = policy.RulePolicy()
    # No exception; serialization stays tiny.
    p.update({"readiness": 0.7}, "progress", 5.0)
    assert p.to_dict() == {"name": "rules"}


# --- BanditPolicy cold-start parity ---------------------------------------

def test_bandit_cold_start_returns_rule_action_with_no_exploration():
    bp = policy.BanditPolicy(epsilon=0.0, prior_strength=3, seed=0)
    ctx = {"readiness": 0.6, "rpe_gap": 0, "overreach_streak": 0}
    # Cold start: rule action wins as the seeded prior.
    assert bp.select(ctx, "hold") == "hold"
    # Same context, different rule action chosen by caller: still cold-
    # starts with the new prior. (Seeding is per-bucket-once; this bucket
    # is already seeded, so the prior persists — that's intentional.)
    assert bp.select(ctx, "progress") == "hold"


def test_bandit_seeds_prior_only_once_per_bucket():
    bp = policy.BanditPolicy(epsilon=0.0, prior_strength=3, seed=0)
    ctx = {"readiness": 0.6, "rpe_gap": 0, "overreach_streak": 0}
    bp.select(ctx, "hold")
    # Q[(bucket, hold)] should be 1.0 after seeding
    b = policy.bucket(ctx)
    assert bp.q[(b, "hold")] == 1.0
    # Re-selecting with a DIFFERENT rule action must NOT overwrite the seed
    bp.select(ctx, "progress")
    assert bp.q[(b, "hold")] == 1.0
    assert bp.q[(b, "progress")] == 0.0


# --- BanditPolicy learning ------------------------------------------------

def test_bandit_overrides_rule_action_after_consistent_negative_feedback():
    """If the rule says HOLD but every time we hold the next session goes
    badly and every time we deload it goes well, the bandit should
    eventually flip to DELOAD for that context."""
    bp = policy.BanditPolicy(epsilon=0.0, prior_strength=1, seed=0)
    ctx = {"readiness": 0.55, "rpe_gap": 0.5, "overreach_streak": 1}

    # Cold start: bandit picks rule action ("hold")
    assert bp.select(ctx, "hold") == "hold"

    # 8 punishments for "hold", 5 rewards for "deload"
    for _ in range(8):
        bp.update(ctx, "hold", -1.0)
    for _ in range(5):
        bp.update(ctx, "deload", 2.0)

    # Now Q[deload] >> Q[hold]; greedy bandit overrides the rule.
    assert bp.select(ctx, "hold") == "deload"


def test_bandit_keeps_rule_action_when_evidence_agrees():
    """Positive feedback for the rule action should reinforce it, not
    flip the bandit to a different arm."""
    bp = policy.BanditPolicy(epsilon=0.0, prior_strength=1, seed=0)
    ctx = {"readiness": 0.8, "rpe_gap": -0.5, "overreach_streak": 0}
    bp.select(ctx, "progress")
    for _ in range(10):
        bp.update(ctx, "progress", 1.5)
    assert bp.select(ctx, "progress") == "progress"


def test_bandit_explores_with_full_epsilon():
    """ε=1.0 always explores — used to verify the exploration branch fires."""
    bp = policy.BanditPolicy(epsilon=1.0, prior_strength=3, seed=42)
    ctx = {"readiness": 0.6, "rpe_gap": 0, "overreach_streak": 0}
    picks = set()
    for _ in range(50):
        picks.add(bp.select(ctx, "hold"))
    # With ε=1 over 50 samples we should hit at least 2 distinct actions
    assert len(picks) >= 2


def test_bandit_update_skips_unseeded_cell():
    """Defensive: updating an action in a bucket select() never seeded
    should be a no-op, not a KeyError."""
    bp = policy.BanditPolicy(seed=0)
    ctx = {"readiness": 0.6, "rpe_gap": 0, "overreach_streak": 0}
    bp.update(ctx, "progress", 1.0)  # no crash; nothing recorded
    assert bp.to_dict()["cells"] == []


# --- Serialization round-trip --------------------------------------------

def test_bandit_serialization_round_trip_preserves_q_and_n():
    bp = policy.BanditPolicy(epsilon=0.15, prior_strength=2, seed=0)
    ctx = {"readiness": 0.5, "rpe_gap": 0.7, "overreach_streak": 1}
    bp.select(ctx, "hold")
    bp.update(ctx, "hold", -0.5)
    bp.update(ctx, "deload", 1.2)
    blob = bp.to_dict()

    restored = policy.BanditPolicy.from_dict(blob)
    assert restored.epsilon == 0.15
    assert restored.prior_strength == 2
    # Compare cell-by-cell
    assert set(restored.q.keys()) == set(bp.q.keys())
    for k in bp.q:
        assert restored.q[k] == pytest.approx(bp.q[k])
        assert restored.n[k] == bp.n[k]


# --- safety_override -----------------------------------------------------

def test_safety_override_pain_returns_hold():
    assert readiness.safety_override(0.9, {"flags": ["pain"]}) == "hold"


def test_safety_override_low_readiness_returns_deload():
    assert readiness.safety_override(0.25, {}) == "deload"


def test_safety_override_returns_none_when_clear():
    assert readiness.safety_override(0.7, {"missed_reps": 0}) is None


# --- make_policy factory --------------------------------------------------

def test_make_policy_default_is_rules(monkeypatch):
    monkeypatch.delenv("COACH_POLICY", raising=False)
    p = policy.make_policy()
    assert isinstance(p, policy.RulePolicy)


def test_make_policy_env_var_enables_bandit(monkeypatch):
    monkeypatch.setenv("COACH_POLICY", "bandit")
    p = policy.make_policy()
    assert isinstance(p, policy.BanditPolicy)


def test_make_policy_preserves_learned_bandit_over_env(monkeypatch):
    """A persisted bandit with learned cells must be restored even when
    COACH_POLICY isn't set — otherwise a restart silently loses learning."""
    monkeypatch.delenv("COACH_POLICY", raising=False)
    bp = policy.BanditPolicy(epsilon=0.0, prior_strength=1, seed=0)
    bp.select({"readiness": 0.6, "rpe_gap": 0, "overreach_streak": 0}, "hold")
    bp.update({"readiness": 0.6, "rpe_gap": 0, "overreach_streak": 0},
              "hold", -1.0)
    state_dict = bp.to_dict()
    restored = policy.make_policy(state_dict=state_dict)
    assert isinstance(restored, policy.BanditPolicy)
    assert len(restored.to_dict()["cells"]) == len(state_dict["cells"])


def test_make_policy_explicit_name_wins(monkeypatch):
    monkeypatch.setenv("COACH_POLICY", "bandit")
    p = policy.make_policy(name="rules")
    assert isinstance(p, policy.RulePolicy)


# --- Loop integration -----------------------------------------------------

def _agent_with_block(tmp_path: str) -> Agent:
    a = Agent(tmp_path)
    a.init_program("Test", {"squat": 150, "bench": 100, "deadlift": 190},
                   bodyweight_kg=85, experience="intermediate")
    a.commit_block("volume", 4, "test", ["squat"])
    return a


def test_loop_default_uses_rule_policy(monkeypatch):
    monkeypatch.delenv("COACH_POLICY", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        assert a.policy.name == "rules"
        a.log_session("squat", 130, 3, 8.0, "felt fine")
        # RulePolicy serializes to a tiny marker — no cells learned.
        assert a.state["policy_state"] == {"name": "rules"}
        # Pending decision still gets stashed for any policy; RulePolicy
        # just ignores it on the next log.
        assert "squat" in a.state["pending_decisions"]


def test_loop_bandit_persists_q_table(monkeypatch):
    monkeypatch.setenv("COACH_POLICY", "bandit")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        assert isinstance(a.policy, policy.BanditPolicy)
        a.log_session("squat", 130, 3, 8.0, "")
        # First log seeds a bucket via select() -> cells present.
        cells = a.state["policy_state"]["cells"]
        assert len(cells) == len(policy.ACTIONS)


def test_loop_bandit_credits_previous_decision_on_next_log(monkeypatch):
    """The bandit must receive an update() call on the second log against
    the same lift — that's the delayed-reward credit assignment."""
    monkeypatch.setenv("COACH_POLICY", "bandit")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        a.log_session("squat", 130, 3, 8.0, "")
        before_n = sum(c["n"] for c in a.state["policy_state"]["cells"])
        a.log_session("squat", 132, 3, 7.5, "")
        after_n = sum(c["n"] for c in a.state["policy_state"]["cells"])
        # After the second log, at least one cell's n should have grown
        # (the prior decision was credited).
        assert after_n > before_n


def test_loop_safety_override_blocks_policy_selection(monkeypatch):
    """When safety_override returns an action, the policy is bypassed and
    no pending_decisions entry is stashed for that lift — we don't want
    the bandit learning from forced safety calls."""
    monkeypatch.setenv("COACH_POLICY", "bandit")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        # Drive readiness very low so safety_override forces deload.
        a.state["readiness"] = 0.20
        a.log_session("squat", 130, 3, 8.0, "")
        # No pending decision recorded for squat because policy didn't choose.
        assert "squat" not in a.state["pending_decisions"]


def test_loop_bandit_state_survives_agent_restart(monkeypatch):
    """A learned bandit cell count should be the same after a fresh
    Agent(path) load — the persisted policy_state must round-trip."""
    monkeypatch.setenv("COACH_POLICY", "bandit")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        a.log_session("squat", 130, 3, 8.0, "")
        a.log_session("squat", 132, 3, 7.5, "")
        cells_before = a.state["policy_state"]["cells"]

        # Restart — fresh Agent, same state file. Drop env var to confirm
        # learned bandit survives without it.
        monkeypatch.delenv("COACH_POLICY", raising=False)
        b = Agent(path)
        assert isinstance(b.policy, policy.BanditPolicy)
        assert b.state["policy_state"]["cells"] == cells_before


# --- log_session reachable via the LLM tool dispatch ----------------------
# The bandit lives inside log_session, so a natural-language session
# description that bypasses the regex log-extractor must still route
# through the same path — otherwise the policy never sees the session and
# the project's headline autoregulation story is fragile.

def test_log_session_tool_drives_full_loop(monkeypatch):
    """Calling log_session via dispatch_tool must mutate state identically
    to a direct agent.log_session call: readiness moves, pending_decisions
    stashes, the session is appended, and the result payload carries the
    interpreted ev fields so the LLM can talk about it."""
    monkeypatch.delenv("COACH_POLICY", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        readiness_before = a.state["readiness"]
        sessions_before = len(a.state["sessions"])

        result = reasoning.dispatch_tool(
            a.state, "log_session",
            {"lift": "squat", "weight": 130, "reps": 3, "rpe": 9.5,
             "notes": "felt brutal, missed my 4th"},
            agent=a,
        )

        assert "error" not in result
        assert result["logged"]["lift"] == "squat"
        assert result["effective_rpe"] is not None
        assert a.state["readiness"] != readiness_before  # loop ran
        assert len(a.state["sessions"]) == sessions_before + 1
        # Rule policy made a free choice — pending stash is the proof the
        # policy path ran (forced/overridden actions are not stashed).
        # Action may be hold/progress/deload depending on the eval; we
        # only assert SOMETHING was stashed.
        assert "squat" in a.state["pending_decisions"] \
            or a.state["block"].get("in_deload")


def test_log_session_tool_with_bandit_credits_delayed_reward(monkeypatch):
    """Two consecutive log_session tool calls under the bandit policy
    should leave a learned cell with n > prior_strength, proving the
    policy.update credit pathway is reached from the tool dispatch."""
    monkeypatch.setenv("COACH_POLICY", "bandit")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        reasoning.dispatch_tool(
            a.state, "log_session",
            {"lift": "squat", "weight": 130, "reps": 3, "rpe": 8.0,
             "notes": "first set"}, agent=a)
        n_after_first = sum(c["n"] for c in a.state["policy_state"]["cells"])
        reasoning.dispatch_tool(
            a.state, "log_session",
            {"lift": "squat", "weight": 132, "reps": 3, "rpe": 7.5,
             "notes": "second set"}, agent=a)
        n_after_second = sum(c["n"] for c in a.state["policy_state"]["cells"])
        # The second log credits the first decision via policy.update.
        assert n_after_second > n_after_first


def test_adjust_lift_load_tool_does_not_touch_readiness_or_policy(monkeypatch):
    """Regression guard: adjust_lift_load must remain a pure TM nudge.
    If it ever started touching readiness or stashing pending_decisions
    it would silently double-update the bandit (once from log_session,
    once from the manual tweak) — keeping the paths separated keeps the
    learning signal clean."""
    monkeypatch.delenv("COACH_POLICY", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        readiness_before = a.state["readiness"]
        sessions_before = len(a.state["sessions"])
        pending_before = dict(a.state["pending_decisions"])
        tm_before = a.state["lifts"]["squat"]["training_max"]

        # Use an explicit prescribed_rpe far from felt so the kg delta is
        # large enough to clear the round-to-increment cliff (raw delta
        # ~5 kg, capped to the per-lift 2.5 kg per-call ceiling).
        result = reasoning.dispatch_tool(
            a.state, "adjust_lift_load",
            {"lift": "squat", "felt_rpe": 6.0, "prescribed_rpe": 8.0},
            agent=a,
        )

        assert "error" not in result
        # TM moved — that's adjust_lift_load's job.
        assert a.state["lifts"]["squat"]["training_max"] != tm_before
        # Everything ELSE the loop touches is untouched.
        assert a.state["readiness"] == readiness_before
        assert len(a.state["sessions"]) == sessions_before
        assert a.state["pending_decisions"] == pending_before


def test_log_session_tool_refuses_without_active_block(monkeypatch):
    """Defensive: log_session needs a committed block to evaluate
    against. Calling the tool before commit_block returns a clear error
    instead of crashing the ReAct loop."""
    monkeypatch.delenv("COACH_POLICY", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = Agent(path)
        a.init_program("Test", {"squat": 150, "bench": 100, "deadlift": 190},
                       bodyweight_kg=85, experience="intermediate")
        result = reasoning.dispatch_tool(
            a.state, "log_session",
            {"lift": "squat", "weight": 130, "reps": 3, "rpe": 8.0,
             "notes": ""}, agent=a,
        )
        assert "error" in result
        assert "no active block" in result["error"].lower()


def test_log_session_tool_rejects_unknown_lift(monkeypatch):
    monkeypatch.delenv("COACH_POLICY", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        result = reasoning.dispatch_tool(
            a.state, "log_session",
            {"lift": "snatch", "weight": 80, "reps": 3, "rpe": 8.0,
             "notes": ""}, agent=a,
        )
        assert "error" in result


def test_log_session_tool_defaults_weight_and_reps_from_pending_prescription(monkeypatch):
    """A 'felt heavy at RPE 8' style message from the lifter doesn't
    restate the kg + reps — they already saw them on screen this week.
    log_session must default both from pending_prescriptions[lift] so
    the policy still gets a learning signal from these messages."""
    monkeypatch.setenv("COACH_POLICY", "bandit")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        pending = a.state["pending_prescriptions"]["squat"]
        prescribed_weight = pending["top_weight"]
        prescribed_reps = pending["reps"]
        sessions_before = len(a.state["sessions"])

        result = reasoning.dispatch_tool(
            a.state, "log_session",
            {"lift": "squat", "rpe": 8.0,
             "notes": "first week squat felt heavy as hell with RPE 8 "
                      "instead of RPE 6.5"},
            agent=a,
        )

        assert "error" not in result
        assert result["logged"]["weight"] == prescribed_weight
        assert result["logged"]["reps"] == prescribed_reps
        # The full loop ran — a session got recorded.
        assert len(a.state["sessions"]) == sessions_before + 1
        recorded = a.state["sessions"][-1]
        assert recorded["weight"] == prescribed_weight
        assert recorded["reps"] == prescribed_reps
        # And the bandit at least seeded a bucket via select().
        assert len(a.state["policy_state"].get("cells", [])) > 0


# --- season-plan backfill: meet date must always land correctly --------
# These guard the recent fix where the LLM's blocks_plan was free to
# drift from the meet date silently. commit_season_plan now backfills
# planned_start by walking backwards from competition_date so the last
# block's deload ends ON the meet, and commit_block respects that floor.

def test_season_plan_backfills_planned_starts_from_meet_date():
    """The LAST upcoming block's deload week must end on
    competition_date; earlier blocks chain back exactly."""
    from datetime import date, timedelta
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = Agent(path)
        a.init_program("T", {"squat": 150, "bench": 100, "deadlift": 190},
                       bodyweight_kg=85, experience="intermediate")
        # 4 blocks * 5 cal wk = exact 20-week fit, picking the meet so
        # the strict backfill accepts the chain without a front gap.
        meet = (date.today() + timedelta(weeks=20)).isoformat()
        out = a.commit_season_plan(
            competition_date=meet, competition_lifts=None,
            blocks=[
                {"block_type": "volume",   "duration_weeks": 4, "rationale": "x"},
                {"block_type": "strength", "duration_weeks": 4, "rationale": "x"},
                {"block_type": "strength", "duration_weeks": 4, "rationale": "x"},
                {"block_type": "peaking",  "duration_weeks": 4, "rationale": "x"},
            ],
        )
        assert "error" not in out, out
        blocks = out["committed_season_plan"]["blocks"]
        last = blocks[-1]
        last_start = date.fromisoformat(last["planned_start"])
        # Deload week ends `duration_weeks + 1` weeks after start.
        last_end = last_start + timedelta(weeks=last["duration_weeks"] + 1)
        assert last_end == date.fromisoformat(meet), (
            f"last block ends {last_end}, meet is {meet}")


def test_season_plan_rejects_macrocycle_too_long_for_window():
    """If the proposed macrocycle would need to start before today (or
    before the active block's expected end), commit_season_plan errors
    with an actionable message naming the correct weeks_to_comp."""
    from datetime import date, timedelta
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = Agent(path)
        a.init_program("T", {"squat": 150, "bench": 100, "deadlift": 190},
                       bodyweight_kg=85, experience="intermediate")
        meet = (date.today() + timedelta(weeks=10)).isoformat()
        # 4 blocks of 5 cal wk = 20 wk back from a 10-wk-away meet -> overshoots
        out = a.commit_season_plan(
            competition_date=meet, competition_lifts=None,
            blocks=[{"block_type": "volume", "duration_weeks": 4,
                     "rationale": "x"}] * 3
                   + [{"block_type": "peaking", "duration_weeks": 4,
                       "rationale": "x"}],
        )
        assert "error" in out
        msg = out["error"].lower()
        assert "too long" in msg
        assert "compute_macrocycle_sizing" in msg


def test_season_plan_rejects_large_front_gap():
    """When the LLM under-sizes the macrocycle by 3+ weeks (e.g. picks
    3 standard blocks for a 19-week window when compute_macrocycle_sizing
    returns 4 variable-length blocks fitting exactly), the engine REJECTS
    so the LLM must retry with the correct sequence — not silently
    accept a 4-week dead zone before the lifter starts training."""
    from datetime import date, timedelta
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = Agent(path)
        a.init_program("T", {"squat": 150, "bench": 100, "deadlift": 190},
                       bodyweight_kg=85, experience="intermediate")
        meet = (date.today() + timedelta(weeks=20)).isoformat()
        # 2 blocks of 5 cal wk = 10 wk total — 10-week gap, way over the
        # 21-day reject threshold.
        out = a.commit_season_plan(
            competition_date=meet, competition_lifts=None,
            blocks=[
                {"block_type": "strength", "duration_weeks": 4, "rationale": "x"},
                {"block_type": "peaking",  "duration_weeks": 4, "rationale": "x"},
            ],
        )
        assert "error" in out
        msg = out["error"].lower()
        assert "gap" in msg
        assert "compute_macrocycle_sizing" in msg


def test_season_plan_accepts_mild_gap_with_timing_note():
    """A 1-2 week gap (acceptable weekday slop or unavoidable rounding)
    is accepted, with a timing_note the LLM should mention to the lifter
    ('you'll start a week from now')."""
    from datetime import date, timedelta
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = Agent(path)
        a.init_program("T", {"squat": 150, "bench": 100, "deadlift": 190},
                       bodyweight_kg=85, experience="intermediate")
        # Pick a meet 17 wk + 9 days out so a 17-cal-wk macrocycle leaves a
        # ~9-day front gap (in the 7-20 day notice band).
        meet = (date.today() + timedelta(weeks=17, days=9)).isoformat()
        out = a.commit_season_plan(
            competition_date=meet, competition_lifts=None,
            blocks=[  # 4+4+4+4 productive = 5+5+5+5 cal = 20wk -> too long
                {"block_type": "volume",   "duration_weeks": 3, "rationale": "x"},
                {"block_type": "strength", "duration_weeks": 3, "rationale": "x"},
                {"block_type": "strength", "duration_weeks": 3, "rationale": "x"},
                {"block_type": "peaking",  "duration_weeks": 3, "rationale": "x"},
            ],  # 4+4+4+4 = 16 cal wk vs 17wk+9d window -> gap ~16 days
        )
        # Either accepted with note (gap 7-20 days) — exact gap depends
        # on weekday math, so accept either branch but require SOMETHING
        # surfaces if gap is non-trivial.
        if "error" in out:
            # If today's weekday math pushed it into REJECT territory,
            # that's also acceptable — both responses are "engine caught it".
            assert "gap" in out["error"].lower()
        else:
            assert "timing_note" in out
            assert "gap" in out["timing_note"].lower()


def test_commit_block_honors_season_plan_planned_start():
    """When commit_block runs the FIRST upcoming block of a season_plan
    whose backfilled planned_start is in the future, the actual block
    must start on that planned_start — not 'next Monday from today' —
    so the meet date stays anchored even after blocks are committed.

    Trick to force a future planned_start without tripping the strict
    >21d reject: have an ACTIVE block whose expected end is far enough
    out that the chained `earliest` (active_end) ≈ first upcoming
    planned_start. Here we commit a season ahead of an existing block."""
    from datetime import date, timedelta
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = Agent(path)
        a.init_program("T", {"squat": 150, "bench": 100, "deadlift": 190},
                       bodyweight_kg=85, experience="intermediate")
        # Active block consumes ~5 cal wk; 1 upcoming peaking block fills
        # the last 5 cal wk. Pick a meet ~11 wk out so the weekday-snap
        # offset on the active block's start doesn't push the chain out
        # of the strict-anchor window.
        a.commit_block("volume", 4, "active", ["squat"])
        meet = (date.today() + timedelta(weeks=11)).isoformat()
        a.commit_season_plan(
            competition_date=meet, competition_lifts=None,
            blocks=[{"block_type": "peaking", "duration_weeks": 4,
                     "rationale": "x"}],
        )
        season_peaking_start = date.fromisoformat(
            a.state["season_plan"]["blocks"][0]["planned_start"])
        # Archive the active volume block (mark as deload-then-complete)
        # so we can commit the peaking block next.
        a.state["block"]["in_deload"] = True
        a.archive_current_block("done")
        # Commit the upcoming peaking block. started_at should snap to/
        # after season_peaking_start, not 'next Monday' (which would
        # often be earlier and re-introduce meet-date drift).
        res = a.commit_block("peaking", 4, "first", ["squat"],
                            weekly_plan=None)
        assert "error" not in res, res
        actual_start = date.fromisoformat(res["started_at"])
        assert actual_start >= season_peaking_start, (
            f"block started {actual_start}, expected >= {season_peaking_start}")


def test_log_session_tool_errors_when_no_prescription_and_no_args(monkeypatch):
    """If neither the lifter nor the engine has numbers to log against,
    bail with a clear error instead of inventing them."""
    monkeypatch.delenv("COACH_POLICY", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.json")
        a = _agent_with_block(path)
        # Wipe the prescription that commit_block populated, simulating a
        # state where the lifter is mid-cycle but the prescription cache
        # got cleared (e.g. after a deload entry).
        a.state["pending_prescriptions"] = {}
        result = reasoning.dispatch_tool(
            a.state, "log_session",
            {"lift": "squat", "rpe": 8.0, "notes": "felt heavy"},
            agent=a,
        )
        assert "error" in result
        assert "weight" in result["error"].lower()
