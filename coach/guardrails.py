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


_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")


def _weekday_index(label: str) -> int | None:
    """Weekday index (Mon=0 .. Sun=6) named in a day label, or None if the
    label doesn't name a weekday (e.g. 'Day 1 — Squat')."""
    low = (label or "").lower()
    for i, name in enumerate(_WEEKDAYS):
        if name in low:
            return i
    return None


def _suggest_swap_day(conflict_pair: tuple[int, int],
                      available_rest_wds: set[int]) -> tuple[int, int] | None:
    """Given a back-to-back (a, b) weekday pair for one lift, suggest
    which of the two to move and which available rest weekday to move it
    to so the resulting spacing is ≥48h. Returns (from_wd, to_wd) — the
    weekday to move FROM and the weekday to move TO — or None if no
    rest weekday in the plan offers a valid landing slot.

    Picks the swap with the SMALLEST circular distance from the original
    conflict day, so the suggestion stays close to what the LLM intended
    (minimal schedule disruption)."""
    a, b = conflict_pair
    best: tuple[int, int, int] | None = None  # (d_move, from_wd, to_wd)
    for keep, move in ((a, b), (b, a)):
        for candidate in available_rest_wds:
            if candidate == keep:
                continue
            # Circular distance to the kept day — need ≥2 for proper 48h+ gap.
            raw = abs(candidate - keep)
            d_keep = min(raw, 7 - raw)
            if d_keep < 2:
                continue
            # Circular distance from the original day we're moving FROM
            # (we want the LLM's plan to change as little as possible).
            raw2 = abs(candidate - move)
            d_move = min(raw2, 7 - raw2)
            if best is None or d_move < best[0]:
                best = (d_move, move, candidate)
    return (best[1], best[2]) if best else None


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
    # Weekday index each lift is trained on (for the consecutive-day check).
    lift_weekdays: dict[str, list[int | None]] = {
        "squat": [], "bench": [], "deadlift": []}
    # Whether each lift has a PRIMARY session of the COMP LIFT itself (a plain
    # squat/bench/deadlift, not a variation) — for the >=1-primary-per-lift rule.
    comp_primary_seen = {"squat": False, "bench": False, "deadlift": False}
    # Collect ALL violations so the LLM fixes them in ONE retry, not one per
    # round-trip (each retry re-sends the whole prompt — expensive).
    errors: list[str] = []

    # Reject duplicate weekday labels: two day-entries on the SAME weekday
    # (e.g. a 'Monday — Heavy Squat' training day AND a 'Monday — Rest') is
    # impossible in a 7-day week and scrambles the calendar + rest layout.
    seen_weekday: dict[int, str] = {}
    for d in days:
        if not isinstance(d, dict):
            continue
        wd = _weekday_index(d.get("label", ""))
        if wd is None:
            continue
        if wd in seen_weekday:
            errors.append(
                f"Two days are both on {_WEEKDAYS[wd].capitalize()} "
                f"('{seen_weekday[wd]}' and '{d.get('label', '?')}'). Each "
                f"weekday may appear only ONCE — give every training day AND "
                f"rest day a DISTINCT weekday (Mon-Sun), and don't emit more "
                f"than 7 day-entries total.")
        else:
            seen_weekday[wd] = d.get("label", "?")

    for i, day in enumerate(days):
        if not isinstance(day, dict):
            # The LLM occasionally produces a malformed days[] where some
            # entries are bare strings or keys/values got flattened (likely
            # a JSON-encoding glitch under output-token pressure). Without
            # this check, day.get(...) later crashes the validator with an
            # uncaught AttributeError; with it, the LLM gets a clear error
            # and re-emits the plan.
            errors.append(f"Day {i+1} is not a structured day-object "
                          f"({type(day).__name__}: {str(day)[:60]!r}). Each "
                          f"entry in `days` must be {{label: ..., exercises: "
                          f"[...]}}. If your output got cut off mid-JSON, "
                          f"shorten the rationales (5-10 words each) and "
                          f"re-emit the full plan.")
            continue
        exercises = day.get("exercises")
        if not isinstance(exercises, list):
            errors.append(f"Day {i+1} '{day.get('label', '?')}' has malformed "
                          f"'exercises' field — must be a list.")
            continue
        if not exercises:
            # Rest days are allowed (label like 'Wednesday — Rest') and skip
            # all per-day checks below. Empty exercises with a non-rest label
            # is treated as an LLM mistake.
            if _is_rest_day(day):
                continue
            errors.append(f"Day {i+1} '{day.get('label', '?')}' has no "
                          f"exercises. Either add at least one exercise, OR "
                          f"label it a rest day (e.g. 'Wednesday — Rest') with "
                          f"exercises=[] — the validator accepts that as a "
                          f"rest marker.")
            continue

        # Rule 0: no exercise from the AVOID list (planks, ab wheel,
        # push-ups, BW squats/lunges, conditioning movements). These
        # contradict the Ben-grounded knowledge base's "every accessory
        # needs a clear purpose and must progress with load" principle.
        for ex in exercises:
            name = ex.get("name", "")
            if name and is_avoided(name):
                errors.append(f"Day {i+1} '{day.get('label', '?')}' contains "
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
                errors.append(f"Day {i+1} '{day.get('label', '?')}' has BOTH primary "
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
        day_weekday = _weekday_index(day.get("label", ""))
        for lift in same_day_main_lifts:
            main_lift_session_counts[lift] += 1
            lift_weekdays[lift].append(day_weekday)
        # A primary entry whose name is the PLAIN comp lift (starts with
        # 'squat'/'bench'/'deadlift', e.g. 'Deadlift (top set)') is the lift's
        # comp-lift primary. A variation (paused squat, block pull, close-grip
        # bench) does NOT start with the lift name, so it doesn't count — this
        # is robust even for variations the PQ-token list doesn't know.
        for ex in exercises:
            lf = ex.get("lift")
            if (ex.get("role") == "primary" and lf in comp_primary_seen
                    and (ex.get("name", "") or "").strip().lower().startswith(lf)):
                comp_primary_seen[lf] = True

        # Rule 5a: RPE-only-named exercises (DB / cable / machine /
        # weighted pull-up / lateral raise / etc.) can't be tagged
        # role='primary' or role='secondary' — those slots drive %TM
        # load math, which is meaningless for non-barbell movements.
        # Such exercises belong in role='accessory'.
        for ex in exercises:
            role = ex.get("role")
            name = ex.get("name", "")
            if role in ("primary", "secondary") and is_rpe_only_accessory(name):
                errors.append(f"Day {i+1} '{day.get('label', '?')}' has "
                        f"'{name}' as role='{role}', but DB / cable / "
                        f"machine / lighter-accessory movements can't "
                        f"be primary or secondary — those slots are for "
                        f"the barbell comp lift or its primary-quality "
                        f"variations (close-grip / tempo / paused / "
                        f"deficit / etc.). Re-tag '{name}' as "
                        f"role='accessory' OR replace it with a barbell "
                        f"variation if you wanted secondary work.")

        # Rule 5d: a loadable BARBELL variation of a comp lift (paused /
        # pin / deficit / tempo / close-grip / block pull) is SECONDARY
        # main-lift work — Sebastian's top-set + variation model accumulates
        # pattern volume through it (tjC6ilWMEMM @ 18:16). Tagging it
        # role='accessory' buries real main-lift work among isolation; a
        # weak-point variation especially belongs on the lift's day as a
        # back-off, or on a dedicated secondary day.
        for ex in exercises:
            if ex.get("role") == "accessory":
                m = _match_primary_quality_variation(ex.get("name", ""))
                if m is not None:
                    patt = m[0].replace("_pattern", "")
                    errors.append(f"Day {i+1} '{day.get('label', '?')}' has "
                            f"'{ex.get('name')}' as role='accessory', but it's "
                            f"a barbell {patt} VARIATION — that's SECONDARY "
                            f"main-lift work, NOT an accessory. Put it as the "
                            f"back-off on the {patt}'s primary day (role="
                            f"'primary', a 2nd entry for that lift), or on a "
                            f"separate SECONDARY day (role='secondary', top set "
                            f"+ back-off). Accessories are isolation / DB / "
                            f"cable / machine / row / arms only.")

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
                errors.append(f"Day {i+1} '{day.get('label', '?')}' has "
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
                    errors.append(f"Day {i+1} '{day.get('label', '?')}' has "
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
            errors.append(f"Day {i+1} '{day.get('label', '?')}' has "
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
            errors.append(f"{count} {lift} sessions/week exceeds the recovery "
                    f"cap of {cap}. Drop one or move it to an accessory role.")

    # Rule 7: frequency floor. Sebastian (x_uhGTQGrAg): 2x/week per body
    # part is optimal — one HEAVY primary day + one LIGHTER/variation
    # secondary day. For a volume/strength block with 3+ training days,
    # squat and bench must each be trained at least twice (this is what
    # creates secondary days). Deadlift is exempt — it's the most fatiguing
    # lift, 1x/week is correct. Technique/peaking blocks are exempt (lower
    # frequency / max specificity). 2-day blocks are exempt (a deliberate
    # low-frequency whole-body choice).
    bt = (block_type or "").lower()
    if bt in ("volume", "strength") and training_day_count >= 3:
        for lift in ("squat", "bench"):
            if main_lift_session_counts[lift] < 2:
                errors.append(f"{lift} is trained only "
                        f"{main_lift_session_counts[lift]}x this week, but a "
                        f"{bt} block with {training_day_count} training days "
                        f"should hit squat and bench ~2x/week (Sebastian: 2x "
                        f"per body part is optimal). Add a SECONDARY {lift} "
                        f"day — a lighter top set + back-off, or a barbell "
                        f"variation (paused / tempo / close-grip). Deadlift "
                        f"may stay 1x (most fatiguing). This is what creates "
                        f"the primary + secondary day structure.")

    # Rule 8: a heavy compound needs recovery — the SAME main lift must not
    # land on CONSECUTIVE calendar days (Mon squat + Tue squat leaves no
    # recovery; space heavy work 48-72h, or stack the lift's second session
    # later in the week). Skipped when day labels don't name weekdays — the
    # engine can't tell the calendar order then.
    # Available rest weekdays = weekdays NOT used by any training day across
    # ALL lifts (explicit rest days + unmentioned weekdays). The validator
    # uses these to suggest a concrete swap target in the error message,
    # which dramatically cuts the LLM's retry count on this rule (observed:
    # 5+ propose_block attempts looping on consecutive-day errors with no
    # actionable hint about WHERE to move the offending day).
    training_wds: set[int] = set()
    for d in days:
        if not isinstance(d, dict):
            continue
        wd = _weekday_index(d.get("label", ""))
        if wd is None or _is_rest_day(d):
            continue
        training_wds.add(wd)
    available_rest_wds = set(range(7)) - training_wds

    for lift, wds in lift_weekdays.items():
        if len(wds) < 2 or any(w is None for w in wds):
            continue
        ordered = sorted(set(wds))
        pair = next(((a, b) for a, b in zip(ordered, ordered[1:])
                     if b - a == 1), None)
        # The week repeats, so Sunday (6) then Monday (0) is also back-to-back.
        if pair is None and 0 in ordered and 6 in ordered:
            pair = (6, 0)
        if pair:
            a, b = pair
            suggestion = _suggest_swap_day(pair, available_rest_wds)
            extra = ""
            if suggestion:
                from_wd, to_wd = suggestion
                extra = (f" Concrete fix: move {_WEEKDAYS[from_wd].capitalize()}'s "
                         f"{lift} to {_WEEKDAYS[to_wd].capitalize()} (free "
                         f"slot in your current plan, ≥48h from the other "
                         f"{lift} day).")
            errors.append(f"{lift} is scheduled on consecutive days "
                    f"({_WEEKDAYS[a].capitalize()} + "
                    f"{_WEEKDAYS[b].capitalize()}) — a heavy compound "
                    f"needs 48-72h between sessions. Move one {lift} day "
                    f"so its two sessions aren't back-to-back (mind the "
                    f"Sunday→Monday wrap — the week repeats).{extra}")

    # Rule 9: a strength/volume block must not collapse into a SINGLE-lift
    # program. Addressing a weakness ADDS focused work (an extra secondary /
    # back-off / accessory for the weak lift) to a balanced program — it must
    # not drop the rest of the lifts (Sebastian: prioritise one quality,
    # MAINTAIN the others). At least 2 of the 3 competition lifts must be
    # trained. Peaking/technique are exempt (may specialise).
    if bt in ("volume", "strength"):
        trained_lifts = [lift for lift in ("squat", "bench", "deadlift")
                         if main_lift_session_counts[lift] > 0]
        if len(trained_lifts) < 2:
            only = trained_lifts[0] if trained_lifts else "nothing"
            errors.append(f"This {bt} block only trains {only}. A weakness is "
                    f"addressed by ADDING focused work (an extra secondary / "
                    f"back-off / accessory for the weak lift) to a BALANCED "
                    f"program — not by dropping the other lifts. Keep training "
                    f"squat, bench and deadlift; give the weak point one extra "
                    f"focused exposure (e.g. a secondary/tertiary day or the "
                    f"back-off on its primary day).")

    # Rule 10: in a volume/strength/peaking block, each TRAINED lift needs at
    # least one PRIMARY session of the COMP LIFT itself — a weakness variation
    # (block pull, deficit, paused) SUPPLEMENTS the comp lift, it doesn't
    # replace it. So a lift trained only through a variation / secondary work,
    # with no primary conventional lift, is rejected. Technique blocks are
    # exempt — pattern practice can centre on a variation (tempo / paused).
    for lift in ("squat", "bench", "deadlift") if bt in (
            "volume", "strength", "peaking") else ():
        if main_lift_session_counts[lift] > 0 and not comp_primary_seen[lift]:
            errors.append(
                f"{lift} is trained only through variations / secondary work — "
                f"there's no PRIMARY session of the competition {lift} itself. "
                f"Add one primary {lift} day (the comp lift as role='primary', "
                f"top set + back-off), and keep the weakness variation (e.g. "
                f"block pull / deficit / paused) as that day's back-off or a "
                f"secondary day. The variation supplements the comp lift; it "
                f"doesn't replace it.")

    if not errors:
        return None
    if len(errors) == 1:
        return errors[0]
    # Multiple problems → report them ALL so the LLM fixes everything in its
    # NEXT propose_block (one call), instead of one-at-a-time retries that
    # burn the tool-call budget and re-send the whole prompt each round.
    return ("The plan has several issues — FIX ALL of them in your NEXT "
            "propose_block (a single call; do not fix them one at a time):\n"
            + "\n".join(f"  {n}. {e}" for n, e in enumerate(errors, 1)))


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
