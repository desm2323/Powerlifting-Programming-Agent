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

from coach import memory, policy, readiness
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
