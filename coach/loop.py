"""
loop.py — the agent loop that ties every box together.

This is the perceive -> reason -> act cycle from the PAGE framework:
  PERCEIVE : read the goal / the logged session
  REASON   : interpret feedback -> update readiness -> run guardrails -> decide
  ACT      : generate the next prescription / adjust the training max
  PERSIST  : write state back to memory

Each step prints a labelled line so the loop is visible on screen — which is
exactly what makes a good 2-minute demo video.
"""

from __future__ import annotations

from . import loading, periodization, readiness, memory, guardrails, reasoning


def _trace(tag: str, msg: str) -> None:
    print(f"  [{tag:<8}] {msg}")


class Agent:
    def __init__(self, state_path: str = memory.DEFAULT_PATH):
        self.path = state_path
        self.state = memory.load_state(state_path)

    def _save(self):
        memory.save_state(self.state, self.path)

    # -- init: set maxes + goal, lay out the first prescription ---------------
    def init_program(self, name: str, maxes: dict, goal: dict):
        self.state["lifter"]["name"] = name
        for lift, one_rm in maxes.items():
            self.state["lifts"][lift]["best_est_1rm"] = one_rm
            self.state["lifts"][lift]["training_max"] = loading.training_max(one_rm)
        self.state["goal"] = goal
        self.state["block"] = {"phase": "accumulation", "week": 1,
                               "weeks_per_phase": 4, "consecutive_overreach": 0}
        self.state["readiness"] = 1.0
        self._save()
        print(f"Program initialised for {name}. Goal: +{goal['target']}"
              f" on {goal['lift']} in {goal['weeks']} weeks.\n")
        self.plan_next()

    # -- plan: produce the next session from phase + training max -------------
    def plan_next(self, lift: str | None = None):
        block = self.state["block"]
        phase = block["phase"]
        rx = periodization.phase_prescription(phase)
        lifts = [lift] if lift else memory.LIFTS
        print(f"Block: {phase} (week {block['week']}/{block['weeks_per_phase']}) "
              f"— {rx['focus']}")
        print(f"Readiness: {self.state['readiness']}\n")
        for lf in lifts:
            tm = self.state["lifts"][lf]["training_max"]
            if tm is None:
                continue
            top = loading.load_for_intensity(tm, rx["intensity_pct"])
            plates = loading.plate_breakdown(top)
            ok, vol_msg = guardrails.check_volume(lf, rx["weekly_set_target"])
            print(f"  {lf.upper():9} {rx['sets']}x{rx['reps']} @ {top} kg "
                  f"(~{int(rx['intensity_pct']*100)}% TM, cap RPE {rx['rpe_cap']})")
            if plates:
                print(f"            plates/side: {plates}")
            if vol_msg:
                print(f"            note: {vol_msg}")
        print()

    # -- log: the feedback loop — interpret, decide, adjust -------------------
    def log_session(self, lift: str, weight: float, performed_reps: int,
                    reported_rpe: float | None, notes: str = ""):
        block = self.state["block"]
        rx = periodization.phase_prescription(block["phase"])
        print(f"Logging {lift} {weight}kg x{performed_reps} "
              f"(RPE {reported_rpe if reported_rpe is not None else '?'})\n")

        # PERCEIVE
        _trace("PERCEIVE", f"note: {notes!r}" if notes else "no note provided")
        scope = guardrails.scope_check(notes)
        if not scope["in_scope"]:
            _trace("GUARD", "out of scope")
            print(f"\n  -> {scope['advisory']}\n")
            return

        # REASON
        ev = reasoning.interpret_feedback(notes, rx["reps"], performed_reps,
                                          reported_rpe, rx["rpe_cap"])
        _trace("REASON", f"interpreted: RPE {ev['effective_rpe']}, "
                         f"missed {ev['missed_reps']}, sticking {ev['sticking_point']}")

        prev = self.state["readiness"]
        self.state["readiness"] = readiness.update_readiness(prev, ev, rx["rpe_cap"])
        _trace("REASON", f"readiness {prev} -> {self.state['readiness']}")

        if readiness.is_overreach(ev, rx["rpe_cap"]):
            block["consecutive_overreach"] += 1
        else:
            block["consecutive_overreach"] = 0

        forced = guardrails.must_deload(block)
        decision = "deload" if forced else readiness.decide_progression(
            self.state["readiness"], ev, block["consecutive_overreach"])
        _trace("GUARD", f"forced_deload={forced}, "
                        f"overreach_streak={block['consecutive_overreach']}")
        _trace("DECIDE", decision.upper())

        # ACT
        est = loading.est_1rm_from_rpe(weight, performed_reps, ev["effective_rpe"])
        memory.record_session(self.state, {
            "lift": lift, "weight": weight, "reps": performed_reps,
            "effective_rpe": ev["effective_rpe"], "missed_reps": ev["missed_reps"],
            "sticking_point": ev["sticking_point"], "notes": notes,
            "est_1rm": round(est, 1),
        })

        best = self.state["lifts"][lift]["best_est_1rm"] or 0
        if est > best:
            memory.record_pr(self.state, lift, weight, performed_reps, est)
            self.state["lifts"][lift]["best_est_1rm"] = round(est, 1)
            _trace("ACT", f"new estimated 1RM for {lift}: {round(est,1)} kg")

        self._apply_decision(decision, lift, block)

        # PERSIST + EXPLAIN
        self._save()
        print(f"\n  -> {reasoning.explain(decision, {'readiness': self.state['readiness']})}\n")
        for f in reasoning.diagnose(memory.recent_sessions(self.state, lift, n=6)):
            print(f"  ! recurring weakness ({f['occurrences']}x {f['weakness']}): "
                  f"{f['suggestion']}\n")

    def _apply_decision(self, decision: str, lift: str, block: dict):
        if decision == "progress":
            tm = self.state["lifts"][lift]["training_max"]
            bump = guardrails.cap_tm_increase(lift, 5.0)
            self.state["lifts"][lift]["training_max"] = loading.round_to_increment(tm + bump)
            _trace("ACT", f"{lift} training max {tm} -> "
                          f"{self.state['lifts'][lift]['training_max']} kg")
        elif decision == "deload":
            nxt = periodization.next_phase(block["phase"])
            if nxt:
                block["phase"], block["week"] = nxt, 1
            else:
                block["phase"], block["week"] = "accumulation", 1  # new mesocycle
            block["consecutive_overreach"] = 0
            self.state["readiness"] = min(1.0, self.state["readiness"] + 0.4)
            _trace("ACT", f"advance to {block['phase']} (week 1), readiness restored")
        else:  # hold
            block["week"] += 1
            _trace("ACT", f"repeat load; advance to week {block['week']}")

    # -- status ---------------------------------------------------------------
    def status(self):
        s = self.state
        print(f"Lifter: {s['lifter']['name']}  ({s['lifter']['units']})")
        b = s["block"]
        print(f"Block:  {b['phase']} week {b['week']}/{b['weeks_per_phase']}  "
              f"| readiness {s['readiness']}")
        print("Training maxes:")
        for lf in memory.LIFTS:
            d = s["lifts"][lf]
            print(f"  {lf:9} TM {d['training_max']} kg  | est 1RM {d['best_est_1rm']} kg")
        if s["prs"]:
            print(f"PRs logged: {len(s['prs'])} (latest est 1RM "
                  f"{s['prs'][-1]['est_1rm']} kg on {s['prs'][-1]['lift']})")
