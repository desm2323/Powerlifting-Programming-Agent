"""
loop.py — the agent loop that ties every box together.

This is the perceive -> reason -> act cycle from the PAGE framework:
  PERCEIVE : read the lifter / the logged session
  REASON   : interpret feedback -> update readiness -> run guardrails -> decide
  ACT      : generate the next prescription / adjust training max / advance block
  PERSIST  : write state back to memory

Each step prints a labelled line so the loop is visible on screen — which is
exactly what makes a good 2-minute demo video.

Block model: the lifter is on ONE block at a time (volume / technique /
strength / peaking). The LLM picks the block via the propose_block tool in
reasoning.py — this loop is what RUNS the chosen block week by week.
"""

from __future__ import annotations

from datetime import date, timedelta

from . import blocks as blocks_mod
from . import guardrails, loading, memory, readiness, reasoning

# Back-compat: existing module code uses the bare `blocks` name extensively.
blocks = blocks_mod


_WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday",
                  "friday", "saturday", "sunday")


def _calendar_sort_days(days: list[dict]) -> list[dict]:
    """Sort days by the weekday named in each label (Mon → Sun) so rest
    days sit in their calendar slot rather than wherever they were listed.
    Days whose label names no weekday keep their relative order and sort
    after the weekday-labelled ones. Example: a plan listed as
    [Mon, Tue, Thu, Fri, Sat, Wed-Rest, Sun-Rest] becomes
    [Mon, Tue, Wed-Rest, Thu, Fri, Sat, Sun-Rest]."""
    def key(idx_day):
        idx, day = idx_day
        label = (day.get("label") or "").lower()
        for wd_index, wd_name in enumerate(_WEEKDAY_NAMES):
            if wd_name in label:
                return (0, wd_index, idx)  # weekday-labelled, sort by Mon..Sun
        return (1, 0, idx)  # unlabelled — keep original order at the end
    return [day for _, day in sorted(enumerate(days), key=key)]


def _fill_rest_days(days: list[dict]) -> list[dict]:
    """Build a full Monday→Sunday week: training days at their weekday, every
    other weekday filled with a rest marker. Any existing rest entries are
    dropped and regenerated, so rest days always appear in the right slot
    regardless of whether the proposed plan included them.

    Falls back to a plain calendar sort if any training day's label doesn't
    name a weekday (can't place it on the calendar)."""
    training = [d for d in days
                if (d.get("exercises") or []) and not guardrails._is_rest_day(d)]
    by_weekday: dict[int, dict] = {}
    for d in training:
        wd = guardrails._weekday_index(d.get("label", ""))
        if wd is None:
            return _calendar_sort_days(days)
        by_weekday[wd] = d
    week: list[dict] = []
    for i, name in enumerate(_WEEKDAY_NAMES):
        week.append(by_weekday[i] if i in by_weekday
                    else {"label": f"{name.capitalize()} — Rest", "exercises": []})
    return week


def _compute_start_date(weekly_plan: dict, today: date,
                        earliest: date | None = None) -> date:
    """Block starts on the next occurrence of the first day's weekday,
    on or after max(today, earliest). If today is Wed and the plan's first
    day is labelled 'Monday — Heavy Squat', start NEXT Monday (5 days later).

    `earliest` lets the caller defer the start when the PREVIOUS block
    hasn't actually finished in calendar time yet — e.g. the lifter
    archives a block during a hypothetical / fast-forward session and the
    next block should still pick up after the original block's scheduled
    end, not immediately. See commit_block.

    Falls back to next Monday if the first day's label doesn't name a
    weekday explicitly.
    """
    anchor = max(today, earliest) if earliest else today
    days = weekly_plan.get("days") if isinstance(weekly_plan, dict) else None
    first_label = ((days[0].get("label", "") if days else "") or "").lower()
    target = None
    for i, name in enumerate(_WEEKDAY_NAMES):
        if name in first_label:
            target = i
            break
    if target is None:
        target = 0  # default: next Monday
    days_ahead = (target - anchor.weekday()) % 7
    return anchor + timedelta(days=days_ahead)


def _earliest_start_from_history(state: dict) -> date | None:
    """If the lifter just archived a block, the next one shouldn't start
    BEFORE the previous block would have finished on the calendar.
    Returns the expected end (started_at + duration_weeks productive + 1
    deload week) of the most recent archived block, or None if no history."""
    history = state.get("block_history", []) or []
    if not history:
        return None
    last = history[-1]
    started_str = last.get("started_at")
    dur = last.get("duration_weeks")
    if not started_str or not isinstance(dur, int):
        return None
    try:
        started = date.fromisoformat(started_str)
    except (ValueError, TypeError):
        return None
    # productive weeks + 1 deload week → the day AFTER the block ends
    return started + timedelta(weeks=dur + 1)


def _trace(tag: str, msg: str) -> None:
    print(f"  [{tag:<8}] {msg}")


class Agent:
    def __init__(self, state_path: str = memory.DEFAULT_PATH):
        self.path = state_path
        self.state = memory.load_state(state_path)

    def _save(self):
        memory.save_state(self.state, self.path)

    # -- init: record the lifter; the LLM decides the first block via chat ----
    def init_program(self, name: str, maxes: dict,
                     bodyweight_kg: float | None = None,
                     experience: str | None = None):
        lifter = self.state["lifter"]
        lifter["name"] = name
        lifter["bodyweight_kg"] = bodyweight_kg
        lifter["experience"] = experience
        for lift, one_rm in maxes.items():
            self.state["lifts"][lift]["best_est_1rm"] = one_rm
            # Training max = the 1RM the lifter entered. No safety buffer —
            # the RPE caps + readiness model do the auto-regulation that
            # 5/3/1's 90% TM was originally designed to handle.
            self.state["lifts"][lift]["training_max"] = loading.round_to_increment(one_rm)
        # IMPORTANT: do NOT auto-pick a block. The agent should converse with
        # the lifter first and choose based on diagnosis, not assumption.
        self.state["block"] = memory.default_state()["block"]
        self.state["readiness"] = 1.0
        self._save()
        bw = f"{bodyweight_kg}kg, " if bodyweight_kg else ""
        exp = f"{experience} lifter" if experience else ""
        meta = f" ({bw}{exp})" if (bw or exp) else ""
        print(f"Welcome, {name}{meta}. Maxes recorded: "
              f"sq {maxes['squat']} / bn {maxes['bench']} / dl {maxes['deadlift']}.")
        print("Now tell me what you're training for and what's bugging you "
              "right now — I'll diagnose and pick the right block.")

    # -- commit / review block: written to state by the LLM via tools ---------
    def commit_block(self, block_type: str, duration_weeks: int,
                     rationale: str, focus_lifts: list[str] | None = None,
                     weekly_plan: dict | None = None,
                     replace_active: bool = False,
                     override_season_plan: bool = False) -> dict:
        """Commit a freshly chosen block. Called by the LLM's propose_block
        tool after diagnosis. `weekly_plan` is the full day-by-day structure
        (see Refactor design). If omitted, a sensible default is generated.

        Immediately computes pending_prescriptions for each main lift so
        logging works right away.

        Safety: if there's already an active block, REFUSES the commit and
        returns an error with instructions. The LLM must call
        review_current_block first (or pass replace_active=True after the
        lifter explicitly confirms a mid-block change). This prevents the
        LLM from silently destroying an in-progress block when the lifter
        just asks 'show me the next block' — see propose_block's docstring.
        """
        active = self.state["block"]
        # Has the lifter actually started training this block? Sessions
        # logged against the active block_type + block_week count.
        has_sessions = any(
            s.get("block_type") == active.get("type")
            and s.get("block_week") == active.get("week")
            for s in self.state.get("sessions", [])
        )
        # An active block the lifter hasn't trained yet (zero logged sessions)
        # is being EDITED, not replaced by the next block in a stack. So
        # "redesign my block" overwrites it IN PLACE: keep its start date and
        # don't consume/advance the season sequence.
        in_place = bool(active.get("type")) and not has_sessions
        if active.get("type") and has_sessions and not replace_active:
            return {"error": (
                f"REFUSED: there's already an active block "
                f"({active['type']}, week {active.get('week', '?')}/"
                f"{active.get('duration_weeks', '?')}) WITH LOGGED "
                f"SESSIONS. DO NOT call propose_block to PREVIEW a "
                f"future block — that commits to state. To DESCRIBE "
                f"upcoming blocks, call get_season_plan and read "
                f"upcoming_blocks. To INTENTIONALLY replace the active "
                f"block mid-cycle (only after the lifter explicitly "
                f"asks to switch), call review_current_block first to "
                f"archive it, THEN propose_block.")}
        # If active block has NO logged sessions, the lifter just got
        # this plan and hasn't trained yet — allow the LLM to revise
        # freely within the same conversation turn. This is the common
        # "actually can you add Sunday rest" / "swap one accessory"
        # flow. The replacement silently overwrites; the engine's
        # other validation rules still apply.
        if block_type not in blocks.VALID_BLOCK_TYPES:
            return {"error": f"unknown block type: {block_type}"}

        # When a season_plan with upcoming blocks exists, a new commit
        # must match the next planned block_type unless override_season_plan
        # is set. This keeps the committed sequence aligned with the
        # macrocycle instead of silently skipping a planned block.
        sp = self.state.get("season_plan", {}) or {}
        upcoming = sp.get("blocks", []) if isinstance(sp, dict) else []
        if upcoming and not override_season_plan and not in_place:
            planned = upcoming[0]
            if planned.get("block_type") != block_type:
                return {"error": (
                    f"REFUSED: season_plan calls for "
                    f"'{planned.get('block_type')}' next "
                    f"(rationale: {planned.get('rationale', '?')}), but "
                    f"you tried to commit '{block_type}'. To follow the "
                    f"plan: call propose_block with block_type="
                    f"'{planned.get('block_type')}'. To intentionally "
                    f"deviate: call update_season_plan first with the "
                    f"revised macrocycle, THEN call propose_block — and "
                    f"in your reply to the lifter, mention the "
                    f"deviation explicitly.")}
        meta = blocks.BLOCK_TYPES[block_type]
        # Accept either {"days": [...]} OR a bare list of days — the LLM
        # sometimes emits the array directly instead of wrapping it.
        if isinstance(weekly_plan, list):
            weekly_plan = {"days": weekly_plan}
        has_days = (isinstance(weekly_plan, dict)
                    and isinstance(weekly_plan.get("days"), list)
                    and weekly_plan["days"])
        plan = weekly_plan if has_days else (
            self._default_weekly_plan(block_type, focus_lifts or []))
        # Normalize day order to the calendar week so rest days sit in
        # their slot rather than wherever they were listed.
        if isinstance(plan, dict) and isinstance(plan.get("days"), list):
            plan = dict(plan)
            plan["days"] = _calendar_sort_days(plan["days"])
        # Guardrail: reject plans that violate basic fatigue/structure rules.
        # The LLM gets a clear error and retries with a fixed plan.
        # block_type lets the validator relax accessory-count rules for
        # peaking blocks (Bromley: "drop high-volume isolation going into
        # peak prep").
        plan_error = guardrails.validate_weekly_plan(plan, block_type=block_type)
        if plan_error:
            return {"error": plan_error}
        # Deterministic start date: the first session day's next occurrence
        # on or after max(today, previous_block.expected_end). The LLM no
        # longer guesses this. Using the previous block's expected end as
        # a floor means: if the lifter "fast-forwards" by archiving early
        # (or asks the LLM to commit the next block hypothetically), the
        # new block still lands AFTER where the previous one would have
        # ended on the calendar, instead of doubling back to today.
        # In-place redesign keeps the original start date (snapped to the
        # first day's weekday in case the plan's day order changed). Otherwise
        # the block starts on or after the previous block's scheduled end
        # (season stacking) or today.
        if in_place and active.get("started_at"):
            try:
                anchor = date.fromisoformat(active["started_at"])
            except (ValueError, TypeError):
                anchor = date.today()
            start = _compute_start_date(plan, anchor)
        else:
            start = _compute_start_date(
                plan, date.today(),
                earliest=_earliest_start_from_history(self.state))
        # Fill rest days deterministically so the committed plan always shows a
        # full Mon-Sun week. Done after the start date is computed so start
        # timing is unaffected.
        plan = dict(plan)
        plan["days"] = _fill_rest_days(plan["days"])
        self.state["block"] = {
            "type": block_type,
            "label": meta["label"],
            "week": 1,
            "duration_weeks": int(duration_weeks),
            "rationale": rationale,
            "focus_lifts": focus_lifts or [],
            "started_at": start.isoformat(),
            "consecutive_overreach": 0,
            "in_deload": False,
            "weekly_plan": plan,
        }
        self.state["pending_prescriptions"] = {}
        self._compute_prescriptions()

        # The season_plan holds the UPCOMING blocks. Once this block is
        # committed it's no longer "upcoming", so drop the matching head
        # entry. Popping here (rather than at archive time) keeps the
        # head correct even when consecutive blocks share a type.
        if upcoming and not override_season_plan and not in_place:
            if upcoming and upcoming[0].get("block_type") == block_type:
                self.state["season_plan"]["blocks"] = upcoming[1:]

        self._save()
        return {
            "committed": self.state["block"],
            "started_at": start.isoformat(),
            "summary": (f"Block COMMITTED: {block_type} ({duration_weeks} "
                        f"productive weeks + 1 deload). Engine-computed "
                        f"start: {start.strftime('%A, %B %d, %Y')} "
                        f"({start.isoformat()}). USE THIS EXACT DATE in "
                        f"your reply to the lifter — do not restate the "
                        f"previous block's date from earlier context."),
        }

    def _default_weekly_plan(self, block_type: str,
                             focus_lifts: list[str]) -> dict:
        """Fallback weekly plan if the LLM didn't supply one. A balanced
        4-day split (Sebastian): squat + bench trained 2x/week (heavy PRIMARY
        day + lighter SECONDARY day), deadlift 1x, rest days filling the rest
        of the week. The LLM should ALWAYS supply its own plan — this exists
        so the engine never commits an empty/invalid block, and it models the
        structure the validator now requires. intensity_pct here is only used
        for exercise selection + validation; real loads come from the RPE
        trajectory in compute_exercise_load."""
        meta = blocks.block_prescription(block_type)
        sets, reps = meta["sets"], meta["reps"]
        intensity, rpe_cap = meta["intensity_pct"], meta["rpe_cap"]
        accessory_picks = {
            "squat": ("Leg curl", "Hanging leg raise"),
            "bench": ("DB row", "Tricep pushdown"),
            "deadlift": ("Barbell row", "Back extension"),
        }

        def main(lift: str, role: str, lighter: bool = False) -> list[dict]:
            inten = max(0.55, intensity - (0.10 if lighter else 0.0))
            cap = max(5.0, rpe_cap - (1.0 if lighter else 0.0))
            return [
                {"name": f"{lift.capitalize()} (top set)", "lift": lift,
                 "role": role, "sets": 1, "reps": reps,
                 "intensity_pct": inten, "rpe_cap": cap, "rationale": None},
                {"name": f"{lift.capitalize()} (back-offs)", "lift": lift,
                 "role": role, "sets": max(2, sets - 1), "reps": reps + 2,
                 "intensity_pct": max(0.55, inten - 0.10),
                 "rpe_cap": max(5.0, cap - 1.0), "rationale": None},
            ]

        def acc(lift: str, n: int = 2) -> list[dict]:
            return [{"name": name, "lift": lift, "role": "accessory",
                     "sets": 3, "reps": 10, "intensity_pct": 0.0,
                     "rpe_cap": 8, "rationale": "default accessory"}
                    for name in accessory_picks[lift][:n]]

        def rest(label: str) -> dict:
            return {"label": label, "exercises": []}

        return {"days": [
            {"label": "Monday — Heavy Squat",
             "exercises": main("squat", "primary") + acc("squat")},
            {"label": "Tuesday — Heavy Bench",
             "exercises": main("bench", "primary") + acc("bench")},
            rest("Wednesday — Rest"),
            {"label": "Thursday — Deadlift",
             "exercises": main("deadlift", "primary") + acc("deadlift")},
            {"label": "Friday — Secondary Squat & Bench",
             "exercises": (main("squat", "secondary", lighter=True)
                           + main("bench", "secondary", lighter=True)
                           + acc("squat"))},
            rest("Saturday — Rest"),
            rest("Sunday — Rest"),
        ]}

    def _compute_prescriptions(self, lifts: list[str] | None = None) -> None:
        """Write the current week's pending prescription for each main lift.
        For each lift, find its highest-intensity (primary) exercise in the
        weekly plan, apply the block's week-wave, and store as pending.
        Logging evaluates against THIS — accessories are guidance, not tracked.
        """
        block = self.state["block"]
        block_type = block.get("type")
        if not block_type:
            return
        in_deload = block.get("in_deload", False)
        plan = block.get("weekly_plan", {}).get("days", [])
        wanted = set(lifts or memory.LIFTS)
        primary_for_lift: dict[str, dict] = {}
        for day in plan:
            for ex in day.get("exercises", []):
                lf = ex.get("lift")
                if lf not in wanted or lf not in memory.LIFTS:
                    continue
                # Pick the exercise that drives this lift the hardest —
                # highest intensity_pct wins; ties broken by sets*reps.
                cur = primary_for_lift.get(lf)
                if (cur is None
                        or ex["intensity_pct"] > cur["intensity_pct"]
                        or (ex["intensity_pct"] == cur["intensity_pct"]
                            and ex["sets"] * ex["reps"] > cur["sets"] * cur["reps"])):
                    primary_for_lift[lf] = ex
        for lf, ex in primary_for_lift.items():
            tm = self.state["lifts"][lf]["training_max"]
            if tm is None:
                continue
            computed = blocks.compute_exercise_load(
                ex, tm, block_type, block["week"], in_deload=in_deload,
                duration_weeks=block.get("duration_weeks"),
                readiness=self.state.get("readiness", 1.0))
            self.state["pending_prescriptions"][lf] = {
                "exercise_name": ex["name"],
                "sets": computed["sets"], "reps": computed["reps"],
                "top_weight": computed["top_weight"],
                "intensity_pct": computed["intensity_pct"],
                "rpe_cap": computed["rpe_cap"],
                "block_type": block_type, "week": block["week"],
                "week_label": computed["week_label"],
            }

    def archive_current_block(self, outcome: str) -> dict:
        """Move the current block into history with a coach-written outcome
        note. Called by review_current_block when a block completes.

        Stores the full weekly_plan in the history record so the UI can
        render a completed block's program on demand.

        Leaves season_plan.blocks untouched — the upcoming-block head is
        popped at commit time, so archiving must not pop again (otherwise
        two consecutive blocks of the same type would lose an entry)."""
        b = self.state["block"]
        if not b.get("type"):
            return {"error": "no active block to archive"}
        # Don't archive an untrained block. A block with no logged sessions
        # isn't "completed" — recording it as done puts a false completion in
        # history AND pushes the next block's start past this block's
        # scheduled end. "Redesign / change my block" means edit it in place:
        # the LLM should re-propose with replace_active=True, not archive.
        trained = any(
            s.get("block_type") == b.get("type")
            and s.get("block_week") == b.get("week")
            for s in self.state.get("sessions", [])
        )
        if not b.get("in_deload") and not trained:
            return {"error": (
                "REFUSED: this block has no logged sessions, so it is NOT "
                "completed. To CHANGE or REDESIGN it, call propose_block with "
                "replace_active=True — that replaces the untrained block in "
                "place and keeps its start date. Only archive (review_current_"
                "block) a block the lifter has actually trained.")}
        record = {
            "type": b["type"], "label": b["label"],
            "duration_weeks": b["duration_weeks"],
            "rationale": b["rationale"],
            "focus_lifts": b["focus_lifts"],
            "started_at": b["started_at"],
            "completed_at": date.today().isoformat(),
            "outcome": outcome,
            # Persist the program so a completed block can be reviewed later.
            "weekly_plan": b.get("weekly_plan", {"days": []}),
        }
        self.state["block_history"].append(record)
        # Reset block to unset — the LLM must propose the next one.
        self.state["block"] = memory.default_state()["block"]
        self.state["pending_prescriptions"] = {}
        self._save()
        return {"archived": record}

    # -- season plan (macrocycle) --------------------------------------------
    def commit_season_plan(self, competition_date: str | None,
                           competition_lifts: dict | None,
                           blocks: list[dict]) -> dict:
        """Replace the season_plan's blocks list; MERGE the meta fields.

        Specifically: `blocks` is always treated as the new full upcoming
        list (replaces what was there). But `competition_date` and
        `competition_lifts` only update when supplied — passing None keeps
        the existing value. This stops the LLM from accidentally moving
        the meet date when it calls propose_season again to adjust the
        block sequence (real bug: meet drifted Oct 12 → Nov 27 between
        two consecutive propose_season calls because the LLM omitted the
        date the second time and we used to overwrite with None).
        """
        cleaned: list[dict] = []
        for i, entry in enumerate(blocks or []):
            bt = entry.get("block_type")
            if bt not in blocks_mod.VALID_BLOCK_TYPES:
                return {"error": (f"season_plan blocks[{i}] has unknown "
                                  f"block_type {bt!r}. Valid: "
                                  f"{list(blocks_mod.VALID_BLOCK_TYPES)}.")}
            dur = entry.get("duration_weeks")
            if not isinstance(dur, int) or dur < 1:
                return {"error": (f"season_plan blocks[{i}] needs "
                                  f"duration_weeks (int >= 1).")}
            cleaned.append({
                "block_type": bt,
                "duration_weeks": int(dur),
                "planned_start": entry.get("planned_start"),  # ISO date or null
                "rationale": entry.get("rationale", "") or "",
                "focus_lifts": entry.get("focus_lifts") or [],
                # Optional preview of what the block will look like — a list
                # of day entries with just exercise NAMES (no weights), so
                # the lifter can see structure without committing loads.
                # Shape: [{"label": "Mon — Heavy Squat",
                #          "exercises": ["Squat (top set)", "Squat (back-offs)",
                #                        "Leg press", "RDL"]},
                #         {"label": "Tue — Heavy Bench", ...}, ...]
                "outline": entry.get("outline") or [],
            })
        existing = self.state.get("season_plan", {}) or {}
        merged_date = (competition_date
                       if competition_date is not None
                       else existing.get("competition_date"))
        merged_lifts = (competition_lifts
                        if competition_lifts is not None
                        else existing.get("competition_lifts"))
        self.state["season_plan"] = {
            "competition_date": merged_date,
            "competition_lifts": merged_lifts,
            "blocks": cleaned,
        }
        self._save()
        return {"committed_season_plan": self.state["season_plan"]}

    def update_season_plan(self, competition_date: str | None = None,
                           competition_lifts: dict | None = None,
                           blocks: list[dict] | None = None) -> dict:
        """Partial update — only the supplied fields are changed. Useful when
        a block went better/worse than expected and the LLM wants to revise
        ONLY the future entries without touching competition info."""
        sp = self.state.setdefault("season_plan", {
            "competition_date": None, "competition_lifts": None, "blocks": [],
        })
        if competition_date is not None:
            sp["competition_date"] = competition_date
        if competition_lifts is not None:
            sp["competition_lifts"] = competition_lifts
        if blocks is not None:
            # Re-validate via commit_season_plan's path.
            check = self.commit_season_plan(sp["competition_date"],
                                            sp["competition_lifts"], blocks)
            return check
        self._save()
        return {"season_plan": sp}

    # -- plan: print this week's schedule day-by-day --------------------------
    def plan_next(self, lift: str | None = None):
        """Print the current week of the active block in day-by-day form.
        The `lift` arg is ignored now that we render the full schedule; the
        per-lift filter doesn't make sense once a weekly plan exists."""
        block = self.state["block"]
        block_type = block.get("type")
        if not block_type:
            print("No active block yet. Chat with me first so I can diagnose "
                  "what you need (volume / technique / strength / peaking).\n")
            return
        self._compute_prescriptions()
        in_deload = block.get("in_deload", False)
        week_label = ("DELOAD" if in_deload
                      else f"week {block['week']}/{block['duration_weeks']}")
        wk_mod = blocks.week_modifier(block_type, block["week"])
        sub_label = f" — {wk_mod['label']}" if wk_mod.get("label") and not in_deload else ""
        print(f"Block: {block['label']} ({week_label}{sub_label})")
        if block.get("rationale"):
            print(f"Why:   {block['rationale']}")
        if block.get("focus_lifts"):
            print(f"Focus: {', '.join(block['focus_lifts'])}")
        print(f"Readiness: {self.state['readiness']}\n")
        plan = block.get("weekly_plan", {}).get("days", [])
        if not plan:
            print("  (no weekly plan structured yet)\n")
            return
        for i, day in enumerate(plan, 1):
            label = day.get("label", f"Day {i}")
            print(f"  {label}")
            for ex in day.get("exercises", []):
                tm = (self.state["lifts"][ex["lift"]]["training_max"]
                      if ex.get("lift") in memory.LIFTS else None)
                c = blocks.compute_exercise_load(
                    ex, tm, block_type, block["week"], in_deload,
                    duration_weeks=block.get("duration_weeks"),
                    readiness=self.state.get("readiness", 1.0))
                wt = f"{c['top_weight']} kg" if c["top_weight"] else "-"
                pct = f"{int(c['intensity_pct']*100)}% TM" if tm else "BW/other"
                role_tag = (f"[{ex['role']}]" if ex.get("role") and ex["role"] != "primary"
                            else "")
                print(f"     - {ex['name']:28} "
                      f"{blocks.format_sets_reps(ex['name'], c['sets'], c['reps'])} "
                      f"@ {wt:>9} ({pct}, RPE {c['rpe_cap']}) {role_tag}")
                if ex.get("rationale"):
                    print(f"         why: {ex['rationale']}")
            print()
        self._save()

    # -- log: the feedback loop — interpret, decide, adjust -------------------
    def log_session(self, lift: str, weight: float, performed_reps: int,
                    reported_rpe: float | None, notes: str = ""):
        block = self.state["block"]
        if not block.get("type"):
            print("No active block yet — I can't evaluate a session against a "
                  "prescription that doesn't exist. Let's chat first to set "
                  "up your block.\n")
            return
        pending = self.state.get("pending_prescriptions", {}).get(lift)
        block_type = block["type"]
        if pending:
            rx = dict(pending)
        elif block.get("in_deload"):
            rx = blocks.deload_prescription(block_type)
        else:
            rx = blocks.block_prescription(block_type)
        print(f"Logging {lift} {weight}kg x{performed_reps} "
              f"(RPE {reported_rpe if reported_rpe is not None else '?'})\n")
        if pending:
            ex_name = pending.get("exercise_name", lift.capitalize())
            _trace("PERCEIVE", f"vs prescription ({ex_name}) "
                               f"{rx['sets']}x{rx['reps']} @ {rx['top_weight']}kg "
                               f"(cap RPE {rx['rpe_cap']})")

        _trace("PERCEIVE", f"note: {notes!r}" if notes else "no note provided")
        scope = guardrails.scope_check(notes)
        if not scope["in_scope"]:
            _trace("GUARD", "out of scope")
            print(f"\n  -> {scope['advisory']}\n")
            return

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

        if block.get("in_deload"):
            decision = "complete_deload"
            _trace("GUARD", "in deload week — completion is the only valid action")
        else:
            forced = self._end_of_block_or_overreach(block)
            decision = "deload" if forced else readiness.decide_progression(
                self.state["readiness"], ev, block["consecutive_overreach"])
            _trace("GUARD", f"forced_deload={forced}, "
                            f"overreach_streak={block['consecutive_overreach']}")
        _trace("DECIDE", decision.upper())

        est = loading.est_1rm_from_rpe(weight, performed_reps, ev["effective_rpe"])
        memory.record_session(self.state, {
            "lift": lift, "weight": weight, "reps": performed_reps,
            "effective_rpe": ev["effective_rpe"], "missed_reps": ev["missed_reps"],
            "sticking_point": ev["sticking_point"], "notes": notes,
            "est_1rm": round(est, 1),
            "block_type": block_type, "block_week": block["week"],
        })

        best = self.state["lifts"][lift]["best_est_1rm"] or 0
        if est > best:
            memory.record_pr(self.state, lift, weight, performed_reps, est)
            self.state["lifts"][lift]["best_est_1rm"] = round(est, 1)
            _trace("ACT", f"new estimated 1RM for {lift}: {round(est,1)} kg")

        self._apply_decision(decision, lift, block)
        self.state["pending_prescriptions"].pop(lift, None)
        self._save()

        print(f"\n  -> {reasoning.explain(decision, {'readiness': self.state['readiness']})}\n")
        for f in reasoning.diagnose(memory.recent_sessions(self.state, lift, n=6),
                                    self.state):
            print(f"  ! recurring weakness on {f['lift']} "
                  f"({f['occurrences']}x {f['weakness']}): {f['suggestion']}\n")

    def _end_of_block_or_overreach(self, block: dict) -> bool:
        """End of the block's productive weeks, or a 2-session overreach streak."""
        duration = block.get("duration_weeks") or 0
        end_of_block = duration and block["week"] >= duration
        return bool(end_of_block) or block.get("consecutive_overreach", 0) >= 2

    def _apply_decision(self, decision: str, lift: str, block: dict):
        if decision == "progress":
            tm = self.state["lifts"][lift]["training_max"]
            # Move the working-1RM anchor toward the latest estimated 1RM
            # (a strong logged set), capped per session so one fluke can't
            # spike all future loads. If the estimate isn't above the anchor,
            # hold rather than chase noise. Loads then track demonstrated
            # strength both within and across blocks.
            best = self.state["lifts"][lift].get("best_est_1rm") or tm
            step = min(max(0.0, best - tm), guardrails.cap_tm_increase(lift, 5.0))
            new_tm = loading.round_to_increment(tm + step)
            self.state["lifts"][lift]["training_max"] = new_tm
            _trace("ACT", f"{lift} anchor {tm} -> {new_tm} kg "
                          f"(toward est 1RM {best})")
        elif decision == "deload":
            # Enter a deload WEEK within the current block.
            block["in_deload"] = True
            block["consecutive_overreach"] = 0
            self.state["readiness"] = min(1.0, self.state["readiness"] + 0.2)
            self.state["pending_prescriptions"] = {}
            _trace("ACT", f"enter deload week in {block['type']} block — "
                          f"reduce volume, recover")
        elif decision == "complete_deload":
            # Deload done -> block is COMPLETE. Archive it and clear the
            # current block. The LLM proposes the next block next chat.
            outcome = (f"completed {block.get('duration_weeks', '?')} weeks "
                       f"+ deload; final readiness {self.state['readiness']}")
            _trace("ACT", f"block complete — archiving {block['type']} for review")
            self.archive_current_block(outcome)
            self.state["readiness"] = min(1.0, self.state["readiness"] + 0.3)
            print("  -> Block complete. Chat with me to plan the next one — "
                  "I'll look at how this block went and propose what's next.")
        else:  # hold
            block["week"] += 1
            _trace("ACT", f"repeat load; advance to week {block['week']}")

    # -- feedback-driven in-place adjustment (no new block) -------------------
    def adjust_lift_load(self, lift: str, felt_rpe: float,
                         prescribed_rpe: float | None = None) -> dict:
        """Nudge a lift's working-1RM anchor based on how a session FELT vs the
        prescribed top-set RPE — WITHOUT rebuilding the block. If the top set
        was prescribed at RPE 8 but felt like RPE 7 (easier), the lifter is
        stronger than the anchor assumed, so the anchor (and every load that
        follows) climbs; if it felt like RPE 9 (harder), it drops.

        kg delta = (prescribed - felt) * 2.5, capped at the per-lift ceiling.
        This is the engine doing the math — the LLM only interprets the fuzzy
        feedback into (lift, felt_rpe)."""
        if lift not in self.state["lifts"]:
            return {"error": f"unknown lift '{lift}'"}
        block = self.state["block"]
        if not block.get("type"):
            return {"error": "no active block — nothing to adjust"}
        week = block.get("week", 1)
        reps, traj_rpe = blocks.top_set_for_week(
            block["type"], week, duration_weeks=block.get("duration_weeks"),
            readiness=self.state.get("readiness", 1.0))
        if prescribed_rpe is None:
            pend = self.state.get("pending_prescriptions", {}).get(lift)
            prescribed_rpe = (pend.get("rpe_cap") if pend else None) or traj_rpe
        tm = self.state["lifts"][lift]["training_max"]
        cap = guardrails.TM_INCREASE_CAP.get(lift, 2.5)
        raw = (float(prescribed_rpe) - float(felt_rpe)) * 2.5
        delta = loading.round_to_increment(max(-cap, min(cap, raw)))
        new_tm = max(20.0, loading.round_to_increment(tm + delta))
        old_top = loading.load_for_rpe(tm, reps, prescribed_rpe)
        self.state["lifts"][lift]["training_max"] = new_tm
        self._compute_prescriptions([lift])
        new_top = loading.load_for_rpe(new_tm, reps, prescribed_rpe)
        self._save()
        return {
            "lift": lift, "prescribed_rpe": prescribed_rpe,
            "felt_rpe": float(felt_rpe), "kg_delta": new_tm - tm,
            "old_training_max": tm, "new_training_max": new_tm,
            "old_top_weight": old_top, "new_top_weight": new_top,
            "block_unchanged": True,
        }

    def advance_block_week(self) -> dict:
        """Move the active block to its next week (or into the deload after the
        last productive week). Keeps the same block — used when the lifter is
        happy with a week and wants to move on."""
        block = self.state["block"]
        if not block.get("type"):
            return {"error": "no active block"}
        if block.get("in_deload"):
            return {"info": "already on the deload week — this block is wrapping up"}
        duration = block.get("duration_weeks") or 0
        if duration and block["week"] >= duration:
            block["in_deload"] = True
            self.state["pending_prescriptions"] = {}
            self._save()
            return {"advanced_to": "deload", "from_week": block["week"]}
        block["week"] += 1
        self._compute_prescriptions()
        self._save()
        return {"advanced_to_week": block["week"], "duration_weeks": duration}

    # -- status ---------------------------------------------------------------
    def status(self):
        s = self.state
        lifter = s["lifter"]
        meta = []
        if lifter.get("bodyweight_kg"):
            meta.append(f"{lifter['bodyweight_kg']}kg bw")
        if lifter.get("experience"):
            meta.append(lifter["experience"])
        meta_str = f"  ({', '.join(meta)})" if meta else ""
        print(f"Lifter: {lifter['name']}{meta_str}")
        b = s["block"]
        if b.get("type"):
            week_label = ("DELOAD" if b.get("in_deload")
                          else f"week {b['week']}/{b['duration_weeks']}")
            print(f"Block:  {b['label']} ({week_label})  "
                  f"| readiness {s['readiness']}")
            if b.get("rationale"):
                print(f"Why:    {b['rationale']}")
            if b.get("focus_lifts"):
                print(f"Focus:  {', '.join(b['focus_lifts'])}")
        else:
            print("Block:  none — chat with the coach to set one up.")
        print("Estimated training maxes:")
        for lf in memory.LIFTS:
            d = s["lifts"][lf]
            print(f"  {lf:9} est. TM {d['training_max']} kg  | est 1RM {d['best_est_1rm']} kg")
        if s["prs"]:
            print(f"PRs logged: {len(s['prs'])} (latest est 1RM "
                  f"{s['prs'][-1]['est_1rm']} kg on {s['prs'][-1]['lift']})")
        if s["block_history"]:
            print(f"Past blocks: {len(s['block_history'])}")
            for h in s["block_history"][-3:]:
                print(f"  - {h['type']} ({h['duration_weeks']}wk): {h['outcome']}")
        self._print_season_plan(brief=True)

    def _print_season_plan(self, brief: bool = False) -> None:
        """Print the macrocycle: past / current / future blocks. brief=True
        only prints a header line (used by status); brief=False prints the
        full timeline (used by the `season` CLI command)."""
        s = self.state
        sp = s.get("season_plan", {}) or {}
        upcoming = sp.get("blocks", []) or []
        history = s.get("block_history", []) or []
        active = s.get("block", {})
        has_active = bool(active.get("type"))
        comp_date = sp.get("competition_date")

        if not upcoming and not history and not has_active:
            if not brief:
                print("No season plan yet — chat with the coach about your "
                      "competition date so they can build one.")
            return

        header = ["Season plan"]
        if comp_date:
            header.append(f"competition {comp_date}")
        if sp.get("competition_lifts"):
            cl = sp["competition_lifts"]
            bits = ", ".join(f"{lf} {cl.get(lf)}kg"
                             for lf in ("squat", "bench", "deadlift")
                             if cl.get(lf))
            if bits:
                header.append(f"targets {bits}")
        print("\n" + " · ".join(header))

        if brief and not upcoming:
            return

        for h in history:
            print(f"  ✓ {h['type']:9} ({h['duration_weeks']}wk)  "
                  f"{h.get('started_at', '?')} → {h.get('completed_at', '?')}"
                  f"   {h.get('outcome', '')}")
        if has_active:
            wk = ("DELOAD" if active.get("in_deload")
                  else f"wk {active.get('week', '?')}/{active.get('duration_weeks', '?')}")
            print(f"  ▶ {active['type']:9} ({active.get('duration_weeks', '?')}wk)  "
                  f"started {active.get('started_at', '?')}  · {wk}")
        for u in upcoming:
            ps = u.get("planned_start") or "tbd"
            print(f"  ○ {u['block_type']:9} ({u['duration_weeks']}wk)  "
                  f"planned {ps}   {u.get('rationale', '')}")
