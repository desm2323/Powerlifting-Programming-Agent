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

from .expert_knowledge import (is_avoided, is_rpe_only_accessory,
                               _match_primary_quality_variation,
                               pattern_of_primary_quality_variation)
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


def check_volume(lift: str, weekly_sets: int) -> tuple[bool, str]:
    """(ok, message). Blocks prescriptions that exceed MRV."""
    status = volume_status(lift, weekly_sets)
    if status == "above_mrv":
        return False, f"{weekly_sets} sets exceeds MRV for {lift}; capping volume."
    if status == "below_mev":
        return True, f"{weekly_sets} sets is below MEV for {lift}; consider more work."
    return True, ""


# Max number of training sessions per week (any role on the main lift) before
# fatigue + recovery cost exceeds the benefit. Accessories don't count.
MAX_SESSIONS_PER_WEEK = {"squat": 3, "bench": 4, "deadlift": 2}

# Tokens that mark a weekly_plan day as a rest / recovery day. Rest days
# legitimately have no exercises — they're informational so the UI can render
# the full 7-day week with explicit rest markers, instead of the lifter
# having to figure out which days are off by absence.
REST_DAY_TOKENS = ("rest", "recovery", "off day", "active recovery", "deload day")


def _is_rest_day(day: dict) -> bool:
    """A day is a rest day if its label names rest/recovery AND it has no
    exercises. A day labelled 'Active recovery — light walk + foam roll'
    with actual exercises is treated as a normal (light) training day."""
    label = (day.get("label") or "").lower()
    exercises = day.get("exercises") or []
    return bool(any(tok in label for tok in REST_DAY_TOKENS)) and not exercises


def validate_weekly_plan(plan: dict,
                         block_type: str | None = None) -> str | None:
    """Reject malformed plans that violate basic powerlifting fatigue rules.
    Returns an error string if the LLM should retry; None if the plan is OK.

    Rules enforced:
      1. Same main lift can't have primary AND secondary on the SAME day —
         that's training the same pattern twice in one session, way too
         fatiguing.
      2. Deadlift: max 2 sessions/week. Squat: max 3. Bench: max 4.
         (Counts primary + secondary roles. Accessories are unlimited.)
      3. Each day must have at least one exercise.
      4. The plan must have at least 2 training days.
      5. No two PRIMARY-QUALITY VARIATIONS of the same pattern (bench /
         squat / deadlift) on the same day — e.g. tempo bench +
         close-grip bench together is double-up bench-pattern fatigue.
      6. Each non-rest training day has at least 2 accessories (≥1 for
         peaking blocks). Bromley czEsWD56hCU @ 3:46 — sessions are
         main → secondary main → primary accessory → lighter isolation.

    `block_type` is optional but enables the peaking-relaxed accessory
    minimum (peaking blocks legitimately have fewer accessories).
    """
    # Accept either {"days": [...]} or a bare list of days — the LLM
    # sometimes emits the array directly.
    if isinstance(plan, list):
        days = plan
    elif isinstance(plan, dict):
        days = plan.get("days")
    else:
        days = None
    if not isinstance(days, list) or len(days) < 2:
        # Likely cause: LLM truncated mid-JSON (output token limit) or
        # emitted weekly_plan={} / {"days": []} without filling it in.
        got = (f"got {type(plan).__name__}" if not isinstance(plan, (dict, list))
               else f"got days={days!r}")
        return (
            f"weekly_plan is malformed or empty ({got}). It MUST be an "
            f"object with a 'days' array of at least 2 training days. "
            f"Shape: {{\"days\": [{{\"label\": \"Monday — Heavy Squat\", "
            f"\"exercises\": [{{name, lift, role, sets, reps, "
            f"intensity_pct, rpe_cap}}, ...]}}, ...]}}. If you got cut "
            f"off mid-JSON: try again with shorter rationales (5-10 "
            f"words each) so the full plan fits. DO NOT fall back to "
            f"prose — describe it in chat AFTER you successfully commit, "
            f"not instead.")
    training_day_count = sum(1 for d in days
                             if isinstance(d, dict) and not _is_rest_day(d))
    if training_day_count < 2:
        return ("weekly_plan needs at least 2 TRAINING days (rest days "
                "don't count). Add more training days or remove rest days.")

    main_lift_session_counts = {"squat": 0, "bench": 0, "deadlift": 0}

    for i, day in enumerate(days):
        exercises = day.get("exercises") if isinstance(day, dict) else None
        if not isinstance(exercises, list):
            return (f"Day {i+1} '{day.get('label', '?')}' has malformed "
                    f"'exercises' field — must be a list.")
        if not exercises:
            # Rest days are allowed (label like 'Wednesday — Rest') and skip
            # all per-day checks below. Empty exercises with a non-rest label
            # is treated as an LLM mistake.
            if _is_rest_day(day):
                continue
            return (f"Day {i+1} '{day.get('label', '?')}' has no exercises. "
                    f"Either add at least one exercise, OR label it as a "
                    f"rest day (e.g. 'Wednesday — Rest' / 'Sunday — Active "
                    f"Recovery') with exercises=[] — the validator will then "
                    f"accept it as a rest marker.")

        # Rule 0: no exercise from the AVOID list (planks, ab wheel,
        # push-ups, BW squats/lunges, conditioning movements). These
        # contradict the Ben-grounded knowledge base's "every accessory
        # needs a clear purpose and must progress with load" principle.
        for ex in exercises:
            name = ex.get("name", "")
            if name and is_avoided(name):
                return (f"Day {i+1} '{day.get('label', '?')}' contains "
                        f"'{name}', which is on the AVOID list (doesn't "
                        f"progress with load / is conditioning / is "
                        f"trained better by the main lift). Replace it "
                        f"with a cited Ben-KB pick: call "
                        f"lookup_accessories(lift, target) for hypertrophy "
                        f"accessories or get_strength_variation(lift, "
                        f"weakness) for a weak-point fix.")

        # Rule 1: same lift can't be primary + secondary same day
        roles_for_lift: dict[str, set[str]] = {}
        same_day_main_lifts: set[str] = set()
        for ex in exercises:
            lift = ex.get("lift")
            role = ex.get("role")
            if lift in main_lift_session_counts and role in ("primary", "secondary"):
                roles_for_lift.setdefault(lift, set()).add(role)
                same_day_main_lifts.add(lift)
        for lift, roles in roles_for_lift.items():
            if "primary" in roles and "secondary" in roles:
                return (f"Day {i+1} '{day.get('label', '?')}' has BOTH primary "
                        f"AND secondary {lift} in the same session — too "
                        f"fatiguing. To fix this, EITHER move the secondary "
                        f"{lift} to a different day (this is usually what "
                        f"the lifter means by 'secondary day'), OR drop "
                        f"the secondary role and re-categorize that exercise "
                        f"as role='accessory' (a true variation like paused "
                        f"or tempo that doesn't count as a separate session). "
                        f"Don't keep both as primary+secondary on the same day.")

        # Count one session per main lift that appears as primary or secondary
        # on this day (multiple exercises for the same lift on the same day
        # still count as ONE session for frequency).
        for lift in same_day_main_lifts:
            main_lift_session_counts[lift] += 1

        # Rule 5a: RPE-only-named exercises (DB / cable / machine /
        # weighted pull-up / lateral raise / etc.) can't be tagged
        # role='primary' or role='secondary' — those slots drive %TM
        # load math, which is meaningless for non-barbell movements.
        # Such exercises belong in role='accessory'.
        for ex in exercises:
            role = ex.get("role")
            name = ex.get("name", "")
            if role in ("primary", "secondary") and is_rpe_only_accessory(name):
                return (f"Day {i+1} '{day.get('label', '?')}' has "
                        f"'{name}' as role='{role}', but DB / cable / "
                        f"machine / lighter-accessory movements can't "
                        f"be primary or secondary — those slots are for "
                        f"the barbell comp lift or its primary-quality "
                        f"variations (close-grip / tempo / paused / "
                        f"deficit / etc.). Re-tag '{name}' as "
                        f"role='accessory' OR replace it with a barbell "
                        f"variation if you wanted secondary work.")

        # Rule 5b: no two DISTINCT primary-quality variations of the
        # same pattern on the same day (tempo bench + close-grip bench
        # = both bench-pattern primary-quality → double-up fatigue,
        # rejected). Same-variation top set + back-off entries (e.g.
        # 'Tempo bench (top set)' + 'Tempo bench (back-offs)') match
        # the SAME token and are ALLOWED — they're Ben's standard
        # two-entry pattern.
        pq_by_pattern: dict[str, list[tuple[str, str]]] = {}
        for ex in exercises:
            match = _match_primary_quality_variation(ex.get("name", ""))
            if match is None:
                continue
            pattern, tok = match
            pq_by_pattern.setdefault(pattern, []).append((ex["name"], tok))
        for pattern, entries in pq_by_pattern.items():
            distinct_tokens = {tok for _, tok in entries}
            if len(distinct_tokens) >= 2:
                # Report one representative exercise per distinct token.
                seen = set()
                names = []
                for name, tok in entries:
                    if tok not in seen:
                        names.append(name)
                        seen.add(tok)
                return (f"Day {i+1} '{day.get('label', '?')}' has "
                        f"{len(distinct_tokens)} DIFFERENT primary-"
                        f"quality variations of the same pattern "
                        f"({pattern.replace('_', ' ')}): {names}. These "
                        f"are all heavy compound variations of the same "
                        f"lift — stacking them same-day doubles fatigue. "
                        f"Pick ONE as the session's secondary movement "
                        f"(top set + back-off both naming THAT same "
                        f"variation), move the others to separate days, "
                        f"OR replace with a true accessory from a "
                        f"DIFFERENT category (tricep isolation, pulling, "
                        f"vertical push) — NOT another bench variation.")

        # Rule 5c: secondary days follow top-set + back-off (≥2 entries
        # for the same lift with role='secondary'). Technique blocks
        # are exempted — a pure-technique single-entry secondary is OK
        # there per the prompt's exception.
        secondary_counts_per_lift: dict[str, int] = {}
        for ex in exercises:
            if ex.get("role") == "secondary":
                lift = ex.get("lift")
                if lift in main_lift_session_counts:
                    secondary_counts_per_lift[lift] = \
                        secondary_counts_per_lift.get(lift, 0) + 1
        if (block_type or "").lower() != "technique":
            for lift, count in secondary_counts_per_lift.items():
                if count < 2:
                    return (f"Day {i+1} '{day.get('label', '?')}' has "
                            f"only {count} role='secondary' entry for "
                            f"{lift}. Secondary days default to TOP "
                            f"SET + BACK-OFF (same two-entry pattern "
                            f"as primary, just lighter — Ben + "
                            f"Sebastian). Output two role='secondary' "
                            f"entries for {lift}: one labelled "
                            f"'(top set)' (~1×3-5 reps), one "
                            f"'(back-offs)' (~3-4×5-8 reps). Both use "
                            f"the SAME variation name (e.g. both "
                            f"'Tempo bench'). The engine's PQ-variation "
                            f"dedup allows this (same token = same "
                            f"variation, not redundant).")

        # Rule 6: minimum accessory count per training day. Peaking
        # blocks legitimately have fewer accessories (Bromley
        # XogX2nvekME @ 23:33 — 'drop this stuff going into peak prep').
        # Non-peaking blocks should have 2-4 accessories per the
        # prompt's "every training day should have 2-4 accessories"
        # rule. Engine enforces minimum so the LLM can't silently skip.
        accessory_count = sum(1 for ex in exercises
                              if ex.get("role") == "accessory")
        min_accessories = 1 if (block_type or "").lower() == "peaking" else 2
        if accessory_count < min_accessories:
            return (f"Day {i+1} '{day.get('label', '?')}' has "
                    f"{accessory_count} accessory exercises "
                    f"(role='accessory') — need at least "
                    f"{min_accessories}. Per Bromley's session "
                    f"structure: main → optional secondary main → "
                    f"primary accessory → lighter isolation. Add "
                    f"accessories targeting the day's neglected muscles "
                    f"(call get_neglected_muscle_targets for the lift) "
                    f"or hypertrophy picks (lookup_accessories). "
                    f"Typical primary day: main + 2-4 accessories. "
                    f"Typical secondary day: variation + 2-3 accessories.")

    # Rule 2: weekly session caps
    for lift, count in main_lift_session_counts.items():
        cap = MAX_SESSIONS_PER_WEEK[lift]
        if count > cap:
            return (f"{count} {lift} sessions/week exceeds the recovery cap "
                    f"of {cap}. Drop one or move it to an accessory role.")

    return None


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
