"""
policy.py — the progression policy as a swappable component.

The agent loop decides progress / hold / deload after every logged session.
v1 of that decision lives in `readiness.decide_progression` as hand-written
rules. This module wraps that same decision behind a Policy interface so the
loop can run either:

  * `RulePolicy`     — the existing hand-written rules (default).
  * `BanditPolicy`   — a contextual ε-greedy bandit that learns per-lifter
                      which action wins in each (readiness, RPE gap,
                      overreach streak) bucket. Cold-starts with the rule
                      policy as a prior, so behaviour is identical until
                      enough sessions have been logged to override it.

The interface is deliberately tiny:
    select(context, rule_action) -> str          # pick the action
    update(context, action, reward) -> None      # learn from the outcome

So swapping the policy never requires touching the engine, validator, LLM,
or UI — the headline "you can put a learned policy behind the same agent"
extension promised in the README and the report.

Maps to the course:
    state    = bucketed (readiness, RPE-gap, overreach-streak) context
    action   = progress | hold | deload
    reward   = computed on the NEXT logged session (delayed reward — the
               classic credit-assignment shape that motivates RL).
"""

from __future__ import annotations

import os
import random
from typing import Callable

ACTIONS: tuple[str, ...] = ("progress", "hold", "deload")

# Default exploration rate. Small — we're not playing a slot machine, we're
# nudging an already-sensible rule policy when its calls look wrong in
# context-specific ways. The cost of a bad exploration step here is one
# wasted session, so keep it conservative.
DEFAULT_EPSILON = 0.10

# Prior strength: how many "imaginary" wins the rule policy's action gets
# in a fresh bucket. 3 means the bandit needs roughly 3 contradicting
# samples in that bucket before it'll override the rule. Low enough to
# learn from a few weeks of real data; high enough not to flip on noise.
DEFAULT_PRIOR_STRENGTH = 3


# -- context bucketing ----------------------------------------------------
# Discretizing the context keeps the Q-table small (3*3*3 = 27 cells),
# JSON-serializable, and learnable from the handful of sessions a single
# lifter logs per block. A linear bandit (LinUCB) would be more sample-
# efficient but needs numpy; this fits the stdlib-only constraint.

def _readiness_band(readiness: float) -> str:
    if readiness <= 0.45:
        return "low"
    if readiness <= 0.70:
        return "med"
    return "high"


def _rpe_gap_band(gap: float) -> str:
    # gap = effective_rpe - rpe_cap. >0 = ran hotter than plan.
    if gap < -0.5:
        return "below"
    if gap <= 0.5:
        return "at"
    return "above"


def _overreach_band(streak: int) -> str:
    if streak <= 0:
        return "0"
    if streak == 1:
        return "1"
    return "2+"


def bucket(context: dict) -> tuple[str, str, str]:
    """Map a raw context dict to a discrete (readiness, rpe-gap, overreach)
    bucket. Pure function — same input always gives the same bucket."""
    return (
        _readiness_band(float(context.get("readiness", 1.0))),
        _rpe_gap_band(float(context.get("rpe_gap", 0.0))),
        _overreach_band(int(context.get("overreach_streak", 0))),
    )


def compute_reward(this_eval: dict, prev_readiness: float,
                   new_readiness: float, pr_set: bool,
                   rpe_cap: float) -> float:
    """Reward signal for the PREVIOUS decision, computed from the NEXT
    session's outcome. Higher = the previous action turned out well.

    Components:
      - overshoot penalty (smooth): every RPE point over the cap costs 0.5
      - missed-reps penalty: -1.0 if any rep was missed
      - flag penalty: -2.0 if pain/injury was flagged
      - recovery bonus: +0.5 if readiness rose materially
      - PR bonus: +1.0 if a new estimated 1RM was set
    """
    r = 0.0
    overshoot = max(0.0, float(this_eval.get("effective_rpe", 0)) - float(rpe_cap))
    r -= 0.5 * overshoot
    if this_eval.get("missed_reps", 0) > 0:
        r -= 1.0
    if this_eval.get("flags"):
        r -= 2.0
    if new_readiness > prev_readiness + 0.1:
        r += 0.5
    if pr_set:
        r += 1.0
    return round(r, 3)


# -- the Policy interface -------------------------------------------------

class Policy:
    """Base interface every policy implements. The loop talks to this, not
    to the concrete class — so swapping policies is a config flip."""

    name: str = "base"

    def select(self, context: dict, rule_action: str) -> str:
        raise NotImplementedError

    def update(self, context: dict, action: str, reward: float) -> None:
        # Stateless policies (RulePolicy) ignore updates; learned policies
        # use them. No-op default keeps the interface symmetric.
        return None

    def to_dict(self) -> dict:
        return {"name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        return cls()

    def describe(self, context: dict, rule_action: str) -> str:
        """One-line trace describing what this policy did and why. Loop
        prints this when present so the bandit's behaviour is visible in
        the demo trace."""
        return f"{self.name} -> {rule_action.upper()}"


class RulePolicy(Policy):
    """The hand-written v1 policy, wearing the Policy interface. select()
    just returns the rule action the loop already computed. Default."""

    name = "rules"

    def select(self, context: dict, rule_action: str) -> str:
        return rule_action

    def describe(self, context: dict, rule_action: str) -> str:
        return f"rules -> {rule_action.upper()}"


class BanditPolicy(Policy):
    """Contextual ε-greedy bandit.

    State: a Q-table mapping (bucket, action) -> running mean reward, plus
    a visit count per cell for the incremental average. Q is seeded so the
    rule policy's action starts at value 1.0 and others at 0.0 — meaning
    cold-start behaviour matches RulePolicy exactly, and only sustained
    contradicting evidence overrides it.

    With ε > 0, the bandit occasionally explores a non-greedy action. That's
    how it discovers cases the rules get wrong (e.g. a lifter who actually
    tolerates 'progress' at moderate readiness because their recovery curve
    is faster than the rule's 0.7 threshold assumes).
    """

    name = "bandit"

    def __init__(self, epsilon: float = DEFAULT_EPSILON,
                 prior_strength: int = DEFAULT_PRIOR_STRENGTH,
                 seed: int | None = None):
        self.epsilon = float(epsilon)
        self.prior_strength = int(prior_strength)
        self.q: dict[tuple, float] = {}
        self.n: dict[tuple, int] = {}
        # Seedable RNG so tests are deterministic. None -> non-deterministic.
        self._rng = random.Random(seed) if seed is not None else random.Random()
        # Diagnostic — last call's bucket + action + (q-values, count). Read
        # by describe() for the trace line; never persisted.
        self._last: dict | None = None

    # -- internal --------------------------------------------------------
    def _seed_if_new(self, b: tuple, rule_action: str) -> None:
        """First time we see a bucket, plant the rule policy as a prior:
        the rule's action gets value 1.0 with `prior_strength` count, the
        others get 0.0 with count 0. Subsequent updates pull the means
        toward observed reward."""
        if (b, ACTIONS[0]) in self.q:
            return
        for a in ACTIONS:
            self.q[(b, a)] = 1.0 if a == rule_action else 0.0
            self.n[(b, a)] = self.prior_strength if a == rule_action else 0

    def _argmax(self, b: tuple, rule_action: str) -> str:
        # Tie-break toward the rule action so the prior wins ties cleanly.
        best, best_q = rule_action, self.q.get((b, rule_action), 0.0)
        for a in ACTIONS:
            if a == rule_action:
                continue
            q = self.q.get((b, a), 0.0)
            if q > best_q:
                best, best_q = a, q
        return best

    # -- Policy interface -----------------------------------------------
    def select(self, context: dict, rule_action: str) -> str:
        b = bucket(context)
        self._seed_if_new(b, rule_action)
        explore = self._rng.random() < self.epsilon
        if explore:
            action = self._rng.choice(ACTIONS)
            mode = "explore"
        else:
            action = self._argmax(b, rule_action)
            mode = "greedy"
        self._last = {
            "bucket": b, "action": action, "mode": mode,
            "rule_action": rule_action,
            "q": {a: round(self.q.get((b, a), 0.0), 3) for a in ACTIONS},
            "n": {a: self.n.get((b, a), 0) for a in ACTIONS},
        }
        return action

    def update(self, context: dict, action: str, reward: float) -> None:
        b = bucket(context)
        # If select() never seeded this bucket (e.g. update is being called
        # in a context the bandit has never seen — shouldn't happen via the
        # loop, but defensive), skip rather than write a half-initialized cell.
        if (b, action) not in self.q:
            return
        self.n[(b, action)] += 1
        n = self.n[(b, action)]
        # Incremental running average: Q_n = Q_{n-1} + (r - Q_{n-1}) / n
        self.q[(b, action)] += (reward - self.q[(b, action)]) / n

    def describe(self, context: dict, rule_action: str) -> str:
        l = self._last
        if not l:
            return f"bandit (cold) -> {rule_action.upper()}"
        qs = ", ".join(f"{a}={l['q'][a]:+.2f}" for a in ACTIONS)
        tag = "EXPLORE" if l["mode"] == "explore" else "greedy"
        if l["action"] != l["rule_action"]:
            tag += f" / overrode rules ({l['rule_action']})"
        return f"bandit({tag}) -> {l['action'].upper()}  [{qs}]"

    # -- serialization --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "epsilon": self.epsilon,
            "prior_strength": self.prior_strength,
            "cells": [
                {"bucket": list(k[0]), "action": k[1],
                 "q": round(v, 6), "n": self.n.get(k, 0)}
                for k, v in self.q.items()
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BanditPolicy":
        bp = cls(
            epsilon=d.get("epsilon", DEFAULT_EPSILON),
            prior_strength=d.get("prior_strength", DEFAULT_PRIOR_STRENGTH),
        )
        for cell in d.get("cells", []):
            key = (tuple(cell["bucket"]), cell["action"])
            bp.q[key] = float(cell["q"])
            bp.n[key] = int(cell["n"])
        return bp


# -- factory --------------------------------------------------------------

def make_policy(name: str | None = None,
                state_dict: dict | None = None) -> Policy:
    """Build a policy from a name and (optionally) prior persisted state.

    Resolution order:
      1. explicit `name` argument (mainly for tests).
      2. saved BanditPolicy with cells already learned — preserve it
         across restarts regardless of env, so a lifter doesn't lose the
         personalised table by forgetting to set COACH_POLICY.
      3. COACH_POLICY env var (the usual toggle).
      4. saved state's name field (e.g. "rules" from a fresh state).
      5. default "rules".
    """
    if state_dict and state_dict.get("name") == "bandit" \
            and state_dict.get("cells"):
        return BanditPolicy.from_dict(state_dict)
    if name:
        resolved = name
    elif os.environ.get("COACH_POLICY"):
        resolved = os.environ["COACH_POLICY"]
    elif state_dict and state_dict.get("name"):
        resolved = state_dict["name"]
    else:
        resolved = "rules"
    if resolved == "bandit":
        if state_dict and state_dict.get("name") == "bandit":
            return BanditPolicy.from_dict(state_dict)
        return BanditPolicy()
    return RulePolicy()
