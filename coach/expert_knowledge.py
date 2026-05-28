"""
expert_knowledge.py — coaching knowledge extracted from three independent
authoritative voices in powerlifting programming:

  1. BEN L. JOHNSON (@BenLJohnson1996) — top-set + back-off methodology,
     RPE-driven auto-regulation, volume landmarks, block stacking.
     Transcripts in docs/ben_transcripts/.

  2. SEBASTIAN OREB (Australian Strength Coach) — exercise selection,
     redundancy avoidance, 8-category structural balance, training-split
     catalog, 5-step periodization model.
     Transcripts in docs/sebastian_transcripts/.

  3. ALEXANDER BROMLEY (Empire Barbell / Base Strength) — accessory
     PROGRESSION methodology (rep-range method, DDP), advantaged vs
     disadvantaged movement leverage, pattern-vs-muscle weakness
     targeting, dormant-muscle high-rep wake-up.
     Transcripts in docs/bromley_transcripts/.

The three voices are COMPLEMENTARY — each covers a gap the others
de-emphasize:
  - Ben: top-set+backoff structure, RPE-as-prescription, block-stack RPE
  - Sebastian: exercise selection, redundancy detection, splits catalog,
    muscle-group-vs-movement-pattern volume denominator
  - Bromley: HOW to progress accessories week-to-week (Ben + Sebastian
    cover what to pick; Bromley covers how to drive it forward), leverage
    classification, weakness-targeting decision framework

The system prompt teaches the LLM to use all three, citing each source
by `video_id @ MM:SS` timestamp.

Source videos (Ben L. Johnson, @BenLJohnson1996). Transcripts
shipped in docs/ben_transcripts/. Every KB entry's `source` field cites
the video ID + a timestamp where the recommendation appears, so the
report can be backed up directly.

  zrOWMftnvQw — Top-set/back-off approach (his 247.5/160/300kg lifts)
  QHGcZaDuwpQ — Your accessory exercises are ruining your strength
  1OXhV1uxPpw — How much training volume is best for strength?
  Klt4SLA64iE — You are using RPE wrong in your strength training
  a5xE7WbtxRs — How I went from 60kg bench to 160kg NATURAL
  n9Wyxr4Yks0 — How I'd get to a 300kg deadlift in 2026
  406rClFwNxY — How to stack training blocks (peaking done right)
  6UPsJiD7QcE — Why your PPL program isn't working (bodybuilding split)

Additional corroborating sources:
  21_YfhLwL6E — How often should you squat/bench/deadlift for strength?
                (frequency: SQ 2-3x / BN 3-4x / DL 1-2x per week)
  g7NdVO9U9gQ — How to Program Secondary Days (accessories for strength)
                (RPE-driven hypertrophy work + main-lift-pattern strength
                 accessories on secondary days)

CORE BEN PRINCIPLES THIS KB ENCODES

1. RPE dictates the weight on the bar, not the other way round.
   Don't pre-pick a kg target and force the RPE to match it; pick the
   weight that hits the prescribed RPE on the day.
   (Klt4SLA64iE @ 4:50)

2. Primary lifts use top-set + back-off. The top set is the heaviest
   set of the day — low velocity skill practice + a daily benchmark for
   readiness. Back-offs are the stress that drives adaptation; they
   should be SUBMAXIMAL (5-10 reps in reserve) and sit at 60-75% of 1RM
   for 3-6 reps. Don't grind back-offs. Bar speed matters.
   (zrOWMftnvQw @ 12:30, 7:30)

3. Back-off LOAD stays static across the block while top sets climb.
   Only bump back-off load when the top sets plateau AND you're well
   recovered. Most lifters mess this up by adding back-off weight every
   week and burning out by week 4.
   (zrOWMftnvQw @ 14:35, 20:50)

4. Block-to-block, periodize the TOP-SET rep range:
     block 1 triples -> block 2 doubles -> block 3 singles -> repeat.
   Avoids desensitisation to singles and gives the top set variety
   across a training year.
   (n9Wyxr4Yks0 @ 9:00)

5. Time-to-peak = 4-5 weeks for most lifters; Ben's own is 5. Don't try
   to PR every block — stack 2-3 developmental blocks before a PR
   attempt. Most blocks should be submax developmental work.
   (406rClFwNxY @ 4:00, 7:00)

6. Accessories serve TWO distinct purposes — programmed DIFFERENTLY:
     STRENGTH accessory (main-lift variation with weak-point purpose):
       3-6 sets x 3-6 reps @ 60-75% 1RM, with one top set ramping
       RPE 5->9 across the block. Pin squat, pause bench, banded DL.
     HYPERTROPHY accessory (muscle-group bodybuilding work):
       2-3 sets x 4-10 reps (up to 12) @ RPE 8-10 (close to failure).
       Mechanical tension drives growth; you must progressively
       overload, not chase pump.
   (QHGcZaDuwpQ @ 8:30, 14:00, 20:00)

7. Hypertrophy accessory checklist (Ben's default new-client template):
   vertical pulls (2 grips: wide overhand for upper lats, neutral for
   lower lats), horizontal pulls (2 grips: wide/high-elbow for upper
   back, neutral/closer-elbow for lats), ONE chest OR shoulder press
   (whichever is the lifter's weak point), all three delt planes
   (lateral / front / rear), one bicep, 1-2 tricep isolations.
   (QHGcZaDuwpQ @ 17:40)

8. Volume landmarks (main-lift work only, accessories NOT counted):
   Squat 6-10 / Bench 9-16 / Deadlift 5-8 sets per week.
   (1OXhV1uxPpw @ 7:40)

9. Frequency: Squat 2-3x, Bench 3-4x, Deadlift 1-2x per week.
   (1OXhV1uxPpw, n9Wyxr4Yks0, a5xE7WbtxRs)

10. Never train main lifts to failure. Comes with disproportionate
    fatigue, impaired motor learning, worse technique in subsequent
    sessions. Save failure training for hypertrophy accessories only.
    (zrOWMftnvQw @ 4:35, n9Wyxr4Yks0 @ 7:55)

Loads still come from loading.load_for_intensity — this module
prescribes STRUCTURE / EXERCISE / REP-RPE RANGES, never a kg.
"""

from __future__ import annotations


def _src(video_id: str, timestamp: str) -> str:
    """Compact source-citation helper. Format: 'video_id @ MM:SS'."""
    return f"{video_id} @ {timestamp}"


# ===========================================================================
# PRESCRIPTION PATTERNS — primary / secondary main lift session layouts.
# ===========================================================================
#
# Top set + back-off is Ben's default for intermediate+ lifters. The numbers
# here come straight from zrOWMftnvQw (the example primary/secondary day
# walkthrough at 17:30 and the 5-week block example at 19:00) and a5xE7WbtxRs
# (bench-specific top-set+back-off example at 4:30).

PRESCRIPTION_PATTERNS: dict[str, dict] = {
    "top_set_backoff_primary": {
        "description": ("Primary main-lift day. One heavy top set (1-3 "
                        "reps at the prescribed RPE) followed by back-off "
                        "volume at 60-75% 1RM for 3-6 reps. Back-off load "
                        "stays static across the block while the top set "
                        "climbs week-to-week."),
        "top_set": {
            "sets": 1,
            "reps_low": 1, "reps_high": 3,
            "rpe_low": 7.0, "rpe_high": 9.0,
            "intensity_pct_low": 0.80, "intensity_pct_high": 0.93,
            "note": "Block 1 use triples, block 2 doubles, block 3 singles, repeat.",
        },
        "backoff": {
            "sets_low": 3, "sets_high": 5,
            "reps_low": 3, "reps_high": 6,
            "rpe_low": 5.0, "rpe_high": 7.0,
            "intensity_pct_low": 0.60, "intensity_pct_high": 0.75,
            "load_static_across_block": True,
            "note": ("Focus on FAST BAR SPEED. 5-10 reps in reserve. "
                     "Don't grind. Don't add weight unless top sets "
                     "plateau AND you feel fresh."),
        },
        "use_for": "Heaviest session of the week for each main lift.",
        "source": _src("zrOWMftnvQw", "9:55"),
    },
    "top_set_backoff_secondary": {
        "description": ("Secondary main-lift day. Same structure as "
                        "primary but lighter top set and lower-intensity "
                        "back-offs. Either the regular lift or a "
                        "weak-point variation."),
        "top_set": {
            "sets": 1,
            "reps_low": 1, "reps_high": 6,
            "rpe_low": 5.0, "rpe_high": 7.5,
            "intensity_pct_low": 0.65, "intensity_pct_high": 0.80,
            "note": "Higher reps + lower RPE than primary day so the two don't bleed.",
        },
        "backoff": {
            "sets_low": 3, "sets_high": 4,
            "reps_low": 4, "reps_high": 6,
            "rpe_low": 5.0, "rpe_high": 7.0,
            "intensity_pct_low": 0.55, "intensity_pct_high": 0.70,
            "load_static_across_block": True,
        },
        "use_for": "Secondary session of the week for a main lift (lighter / variation day).",
        "source": _src("zrOWMftnvQw", "10:45"),
    },
    "straight_sets": {
        "description": ("Same load every set. Simpler than top-set + "
                        "back-off but loses the daily auto-regulation. "
                        "Ben uses this rarely — fine for novices or for "
                        "very stable secondary work."),
        "sets_low": 3, "sets_high": 5,
        "reps_low": 5, "reps_high": 8,
        "intensity_pct_low": 0.65, "intensity_pct_high": 0.80,
        "rpe_low": 7.0, "rpe_high": 8.5,
        "use_for": "Novices, secondary lifts, technique-emphasis blocks.",
        "source": _src("zrOWMftnvQw", "1:00"),
    },
}


# ===========================================================================
# VOLUME LANDMARKS — Ben's per-week main-lift set counts (1OXhV1uxPpw).
# ===========================================================================
#
# These are MAIN-LIFT direct sets only — leg press, hack squat, leg
# extension, etc. are EXTRA and not counted here. Within these ranges,
# intensity split is ~75% of sets at 60-75% 1RM (the back-off zone) and
# ~25% at 80%+ (the top sets).

VOLUME_LANDMARKS: dict[str, dict] = {
    "squat": {
        "sets_per_week_low": 6, "sets_per_week_high": 10,
        "sessions_per_week_low": 2, "sessions_per_week_high": 3,
        "rep_cap": 8,
        "source": _src("1OXhV1uxPpw", "7:40"),
    },
    "bench": {
        "sets_per_week_low": 9, "sets_per_week_high": 16,
        "sessions_per_week_low": 3, "sessions_per_week_high": 4,
        "rep_cap": 8,
        "source": _src("1OXhV1uxPpw", "8:15"),
        "note": ("Bench responds well to high frequency. Ben benched 4x/wk "
                 "to hit 160kg; multiple of his clients hit 4-plate doing "
                 "the same. a5xE7WbtxRs @ 3:55"),
    },
    "deadlift": {
        "sets_per_week_low": 5, "sets_per_week_high": 8,
        "sessions_per_week_low": 1, "sessions_per_week_high": 2,
        "rep_cap": 6,
        "source": _src("1OXhV1uxPpw", "8:55"),
        "note": ("Conventional DL keep reps <=6 (lower back fatigue). "
                 "Sumo and RDL can go higher. Max ~5 sets per session. "
                 "n9Wyxr4Yks0 @ 5:50"),
    },
}


# ===========================================================================
# BLOCK PERIODIZATION — time-to-peak + block stacking (406rClFwNxY).
# ===========================================================================
#
# Ben's training-year model: write ONE week of training and run it until
# you see two down-weeks in a row. That defines your time-to-peak. Stack
# 2-3 developmental blocks (submaximal work) before a PR attempt.

BLOCK_PERIODIZATION: dict = {
    "default_block_length_weeks": 5,
    "block_length_range_weeks": (4, 6),
    "top_set_rep_cycle": ["triples", "doubles", "singles"],
    "rpe_wave_5wk": [5, 6, 7, 8, 9],
    "rpe_wave_4wk": [6, 7, 8, 9],
    "blocks_per_pr_attempt": {
        "novice": 1,
        "intermediate": 3,
        "advanced": 4,
    },
    "source": _src("406rClFwNxY", "4:00"),
    "note": ("Most blocks should be developmental, NOT max-attempt blocks. "
             "Pushing for a PR every block burns intermediate+ lifters out "
             "by week 4 of every cycle. 406rClFwNxY @ 7:30."),
}


# ===========================================================================
# BLOCK_STACKING — per-position RPE schedule across a multi-block stack
# (406rClFwNxY @ 8:45-9:55, 12:48).
# ===========================================================================
#
# Ben's headline principle: don't try to PR every block — STACK blocks toward
# a final PR / competition. The earlier blocks are "deposits"; you leave RPE
# in the tank. The final block in the stack is where you express the strength
# you've built.
#
# Schedule (intermediate, 3-block stack — Ben's own example, 406rClFwNxY @ 8:45):
#   Block 1 (deposit):  top set RPE 7.5-8     ("a single at RPE 7.5 to 8")
#   Block 2 (deposit):  top set RPE 8-8.5     ("slightly heavier than block 1")
#   Block 3 (PR):       top set RPE 9-10      ("send it")
#
# Novice (PR every block — beginners can demonstrate progress block-to-block):
#   Block 1:            top set RPE 8-9
#
# Advanced (Ben's own pattern when needing more deposits — 4 blocks):
#   Blocks 1-2 (deposits): top set RPE 7-8
#   Block 3 (deposit):     top set RPE 8-9
#   Block 4 (PR):          top set RPE 9-10
#
# The "bank deposit" analogy (406rClFwNxY @ 10:00): if you max out each
# block you reset your account to zero (max attempts come with fatigue and
# need recovery). Deposit, deposit, then withdraw on the PR block.
#
# Working backwards from competition (406rClFwNxY @ 11:43): the meet should
# land on the final week of the FINAL block in your stack. So if your block
# length is 5 weeks and you stack 3 blocks, the meet is on week 5 of block 3
# — 15 weeks out. Work backwards to set start dates.

BLOCK_STACKING: dict = {
    "novice": {
        "blocks_to_pr": 1,
        "schedule": [
            {"position": 1, "role": "PR",
             "top_set_rpe_low": 8.0, "top_set_rpe_high": 9.0,
             "note": "Beginners can demonstrate progress block-to-block."},
        ],
        "source": _src("406rClFwNxY", "6:54"),
    },
    "intermediate": {
        "blocks_to_pr": 3,
        "schedule": [
            {"position": 1, "role": "deposit",
             "top_set_rpe_low": 7.5, "top_set_rpe_high": 8.0,
             "note": "First deposit — leave RPE in the tank, build momentum."},
            {"position": 2, "role": "deposit",
             "top_set_rpe_low": 8.0, "top_set_rpe_high": 8.5,
             "note": "Slightly heavier deposit. Don't max — block 3 is the PR."},
            {"position": 3, "role": "PR",
             "top_set_rpe_low": 9.0, "top_set_rpe_high": 10.0,
             "note": "Send it. This is where you express the deposits."},
        ],
        "source": _src("406rClFwNxY", "8:45"),
    },
    "advanced": {
        "blocks_to_pr": 4,
        "schedule": [
            {"position": 1, "role": "deposit",
             "top_set_rpe_low": 7.0, "top_set_rpe_high": 8.0,
             "note": "First deposit."},
            {"position": 2, "role": "deposit",
             "top_set_rpe_low": 7.5, "top_set_rpe_high": 8.5,
             "note": "Second deposit."},
            {"position": 3, "role": "deposit",
             "top_set_rpe_low": 8.0, "top_set_rpe_high": 9.0,
             "note": "Heavier deposit, still submaximal."},
            {"position": 4, "role": "PR",
             "top_set_rpe_low": 9.0, "top_set_rpe_high": 10.0,
             "note": "PR block — express the accumulated deposits."},
        ],
        "source": _src("406rClFwNxY", "11:43"),
    },
    "principles": [
        "Stack blocks toward a single PR — don't max every block.",
        "Block 1-N-1 are deposits (sub-maximal). Block N is the PR.",
        "Maxing out resets fatigue 'to zero' (bank deposit analogy, "
        "406rClFwNxY @ 10:00).",
        "Working backwards from comp: meet falls on the FINAL WEEK of the "
        "FINAL BLOCK in the stack (406rClFwNxY @ 11:43).",
        "After each block: brief review (was block length right? did I "
        "peak when expected? what to change?) — 406rClFwNxY @ 13:08.",
    ],
}


# ===========================================================================
# TIME_TO_PEAK — bottom-up programming (406rClFwNxY @ 3:30-6:13).
# ===========================================================================
#
# Don't pre-write 12 weeks. Write ONE week and run it until performance peaks
# AND THEN two down weeks in a row. That gives you your time-to-peak (your
# personal block length). Ben's is 5 weeks; ranges 4-6 for most.

TIME_TO_PEAK: dict = {
    "process": [
        "Write ONE week of training (no pre-planned 12-week block).",
        "Run that template with minimal changes week-to-week.",
        "Add weight (or push RPE) until performance peaks.",
        "Keep going. Time-to-peak is the LAST good week BEFORE you see "
        "two down weeks in a row.",
        "Ben's personal time-to-peak is 5 weeks. Most intermediates 4-5.",
    ],
    "default_weeks": 5,
    "range_weeks": (4, 6),
    "source": _src("406rClFwNxY", "5:36"),
}


# ===========================================================================
# RPE_USAGE — Ben's core RPE principle (Klt4SLA64iE).
# ===========================================================================
#
# THE one principle to internalize: RPE dictates the weight on the bar. NOT
# the other way round. Most lifters get this backwards — they pick a weight
# from a calculator, then force the RPE to fit. That's not auto-regulation,
# that's just doing fixed loads with an RPE label slapped on.

RPE_USAGE: dict = {
    "core_principle": (
        "RPE dictates the weight on the bar, not the other way round. "
        "Klt4SLA64iE @ 4:50."
    ),
    "common_mistake": (
        "Pre-picking a kg number (from a calculator or PR%) and trying to "
        "force the RPE to match. That ignores daily preparedness and "
        "produces inappropriate stress doses — the lifter overshoots, "
        "accumulates fatigue, and stalls. Klt4SLA64iE @ 0:51-2:14."
    ),
    "how_to_use": [
        ("Don't pre-select the weight before the gym. Go in with the RPE "
         "+ rep target — that IS the prescription."),
        ("Warm up progressively (5s → 3s → 2 → 1). After each set, ask "
         "'how many more reps could I do?' The number tells you "
         "RIR / effective RPE."),
        ("Use the final warm-up as a daily readiness check. If 90% of "
         "your planned top set feels fast and easy → make the jump. If "
         "it feels grindy → drop the planned top set 5-10kg, OR skip "
         "the jump and stay at the warm-up. Klt4SLA64iE @ 10:25."),
        ("Block 1 of any progression is EXPERIMENTAL. Treat week 1 as "
         "calibration — find where you actually start, then plan small "
         "weekly increments (Ben adds ~10kg/wk on squat at submax "
         "RPEs). Klt4SLA64iE @ 9:08."),
        ("From block 2 onward: use the LAST block's actual top set "
         "weight at the same RPE as your starting point, +5-10kg. "
         "Klt4SLA64iE @ 11:32."),
        ("Lift the weight that SHOULD be on the bar (RPE-driven), NOT "
         "the weight you WANT on the bar (ego-driven). Klt4SLA64iE @ "
         "13:18."),
    ],
    "stress_dose_framing": (
        "You get stronger by accumulating an appropriate dose of stress, "
        "NOT by forcing more weight on the bar. The body responds to "
        "stress, not to the number printed on the plates. "
        "Klt4SLA64iE @ 2:33, 11:05."
    ),
    "warmup_feedback_rule": (
        "If warm-ups feel terrible, ADJUST DOWN. Don't grind out the "
        "planned top set to an RPE 10 just because the spreadsheet "
        "says so. That overshoot poisons the next 2-3 weeks of the "
        "block. Klt4SLA64iE @ 10:33-11:00."
    ),
    "source": _src("Klt4SLA64iE", "4:50"),
}


# ===========================================================================
# HYPERTROPHY ACCESSORY CHECKLIST — Ben's default new-client template.
# ===========================================================================
#
# From QHGcZaDuwpQ @ 17:40. Each slot below is one weekly accessory
# requirement. Lifters who can train more frequently (PPL split) can hit
# every slot every cycle; lifters on 3-4 main-lift days should distribute
# these across the week.
#
# All slots: 2-3 sets x 4-10 reps @ RPE 8-10 (close to failure for the
# mechanical-tension stimulus). Heavier compound accessories that double
# as main-lift volume substitutes (leg press as extra squat work,
# Bulgarian split squat) drop to RPE 6-7. (QHGcZaDuwpQ @ 19:00).

HYPERTROPHY_CHECKLIST: list[dict] = [
    {"slot": "vertical_pull_upper_lats",
     "default_exercise": "Wide-grip lat pulldown (overhand)",
     "alternates": ["Wide-grip pull-up", "Pulldown to chest, elbows flared"],
     "muscle_target": "Upper lats, traps, rear delts",
     "rationale": ("Flared-elbow vertical pull biases upper-lat fibers. "
                   "Pair with a closer-grip neutral pulldown for full lat "
                   "coverage."),
     "source": _src("QHGcZaDuwpQ", "17:55")},

    {"slot": "vertical_pull_lower_lats",
     "default_exercise": "Neutral-grip lat pulldown",
     "alternates": ["Neutral-grip pull-up", "Close-grip pulldown"],
     "muscle_target": "Lower lats",
     "rationale": ("Tucked-elbow vertical pull targets lower-lat fibers. "
                   "Pair with wider overhand pulldown."),
     "source": _src("QHGcZaDuwpQ", "18:05")},

    {"slot": "horizontal_pull_upper_back",
     "default_exercise": "Chest-supported wide-grip row",
     "alternates": ["T-bar row (chest supported)", "High-elbow cable row"],
     "muscle_target": "Traps, rhomboids, rear delts, upper back",
     "rationale": ("Wide-grip + high-elbow path targets traps and rear "
                   "delts. Chest support removes lower-back fatigue — "
                   "important if you've just deadlifted or squatted."),
     "source": _src("QHGcZaDuwpQ", "18:25")},

    {"slot": "horizontal_pull_lats",
     "default_exercise": "Neutral-grip cable row",
     "alternates": ["Single-arm DB row", "Seal row"],
     "muscle_target": "Lat fibers (mid back)",
     "rationale": ("Closer-elbow path targets lat fibers more than "
                   "traps. Complements the wide-grip variation."),
     "source": _src("QHGcZaDuwpQ", "18:35")},

    {"slot": "press_supplement",
     "default_exercise": "Chest press machine OR DB press",
     "alternates": ["Incline DB press", "Shoulder press (if delts are weak)"],
     "muscle_target": "Chest OR shoulders (pick by weak point)",
     "rationale": ("Pick ONE based on the lifter's weak point. Ben does "
                   "chest press machine because his pecs lag; he doesn't "
                   "do shoulder press because his shoulder press is "
                   "already strong. a5xE7WbtxRs @ 7:50."),
     "source": _src("QHGcZaDuwpQ", "18:50")},

    {"slot": "lateral_delt",
     "default_exercise": "DB lateral raise",
     "alternates": ["Cable lateral raise"],
     "muscle_target": "Medial delt",
     "rationale": ("Cue: push the DBs out to the walls at your sides "
                   "(not shrug them up). 6UPsJiD7QcE @ 11:30."),
     "source": _src("QHGcZaDuwpQ", "19:05")},

    {"slot": "front_delt",
     "default_exercise": "DB front raise",
     "alternates": ["Plate front raise"],
     "muscle_target": "Anterior delt",
     "rationale": ("Still useful even if you bench frequently. "
                   "\"Your front delts are not too big or strong unless "
                   "you're a 500-lb bencher.\" a5xE7WbtxRs @ 9:00."),
     "source": _src("QHGcZaDuwpQ", "19:10")},

    {"slot": "rear_delt",
     "default_exercise": "Rear-delt fly (machine)",
     "alternates": ["Cable rear-delt fly", "Face pull"],
     "muscle_target": "Posterior delt",
     "rationale": "Shoulder balance + health alongside heavy benching.",
     "source": _src("QHGcZaDuwpQ", "19:15")},

    {"slot": "bicep",
     "default_exercise": "Preacher curl",
     "alternates": ["Incline DB curl (shoulder extended for long head)"],
     "muscle_target": "Biceps",
     "rationale": ("Pick one with shoulder flexion (preacher) AND one "
                   "with shoulder extension (incline) to cover both "
                   "long-head and short-head bias."),
     "source": _src("QHGcZaDuwpQ", "19:30")},

    {"slot": "tricep_isolation",
     "default_exercise": "Single-arm tricep push-down",
     "alternates": ["Cable tricep push-down (two-arm)"],
     "muscle_target": "Triceps (medial/lateral heads)",
     "rationale": ("Lock the elbow in place — the ONLY movement should "
                   "be elbow extension. 6UPsJiD7QcE @ 12:00."),
     "source": _src("QHGcZaDuwpQ", "19:35")},

    {"slot": "tricep_compound",
     "default_exercise": "JM press (Smith machine)",
     "alternates": ["Overhead tricep extension (DB or cable)"],
     "muscle_target": "Triceps (long head)",
     "rationale": ("Smith JM press loads triceps the heaviest of any "
                   "isolation. Overhead extension is the long-head "
                   "stretched-position option. a5xE7WbtxRs @ 8:30."),
     "source": _src("QHGcZaDuwpQ", "19:40")},
]


# ===========================================================================
# STRENGTH_ACCESSORIES — main-lift variations for weak-point fixes.
# ===========================================================================
#
# These are role='secondary' movements (separate day) or used to REPLACE
# the regular lift on the secondary day. Ben emphasises they need a
# SPECIFIC PURPOSE — don't add a pin squat just because you saw it on
# Instagram. (QHGcZaDuwpQ @ 11:20).
#
# Prescription: top_set_backoff_secondary pattern (above) — one top set
# ramping RPE 5->9 over the block, 3-6 sets x 3-6 reps @ 60-75% 1RM.

STRENGTH_ACCESSORIES: dict[str, dict[str, list[dict]]] = {
    "squat": {
        "depth": [
            {"name": "Pin squat (pins set at competition depth)",
             "rationale": ("Forces self-organisation into correct depth. "
                           "You hit the pins, the bar settles, you stand "
                           "up — natural feedback for hitting depth without "
                           "verbal cuing."),
             "source": _src("QHGcZaDuwpQ", "5:50")},
            {"name": "Pause squat (3s in the hole)",
             "rationale": "Holds bottom position — builds bottom strength + depth confidence.",
             "source": _src("QHGcZaDuwpQ", "5:30")},
        ],
        "load_reduction": [
            {"name": "Tempo squat (3s eccentric)",
             "rationale": ("Forces lighter load on the bar without changing "
                           "anything else in the program. Useful when "
                           "fatigue is climbing."),
             "source": _src("QHGcZaDuwpQ", "7:10")},
        ],
    },
    "bench": {
        "bar_path": [
            {"name": "Pin bench press (pins at chest height)",
             "rationale": ("Forces you to press the bar back-and-up from a "
                           "dead start. Fixes a forward-drifting bar path."),
             "source": _src("QHGcZaDuwpQ", "3:25")},
            {"name": "Tempo bench (3s eccentric)",
             "rationale": ("Slow descent gives you time to think about arm "
                           "path, touch point, leg drive. Ben hated doing "
                           "these but the results were undeniable. a5xE7WbtxRs @ 5:55."),
             "source": _src("a5xE7WbtxRs", "5:55")},
        ],
        "chest_tightness": [
            {"name": "Long-pause bench (2-3s on chest)",
             "rationale": ("Hard to pause if your setup is loose — forces "
                           "you to maintain tightness, leg drive, correct "
                           "touch point. Ben's go-to fix for guys who dump "
                           "the bar at the bottom."),
             "source": _src("QHGcZaDuwpQ", "16:30")},
        ],
    },
    "deadlift": {
        "start_position": [
            {"name": "Pause deadlift (1s, 1\" off floor)",
             "rationale": ("Hard to pause in a bad start position — bar "
                           "drifting forward, shoulders behind bar, upper "
                           "back rounded. The pause makes you self-correct "
                           "into a stronger position."),
             "source": _src("QHGcZaDuwpQ", "15:35")},
            {"name": "Banded deadlift",
             "rationale": ("Band pulls horizontally, encourages upper-back "
                           "extension at mid-pull. Fixed Ben's own "
                           "upper-back rounding."),
             "source": _src("QHGcZaDuwpQ", "6:25")},
        ],
        "fatigue_management": [
            {"name": "Romanian deadlift (RDL)",
             "rationale": ("Lighter hip-hinge stimulus when twice-a-week "
                           "deadlifting is too much. Ben's go-to substitute "
                           "for secondary DL day during back pain. n9Wyxr4Yks0 @ 3:08."),
             "source": _src("n9Wyxr4Yks0", "3:08")},
            {"name": "Hip thrust",
             "rationale": "Glute-focused hinge with low spinal cost — replacement for secondary DL day.",
             "source": _src("n9Wyxr4Yks0", "3:32")},
        ],
    },
}


# ===========================================================================
# EXPERT_ACCESSORIES — per-(lift, target) hypertrophy accessory pool.
# ===========================================================================
#
# These are SAME-DAY HYPERTROPHY accessories — what goes on a bench day
# after the bench primary, on a squat day after the squat primary, etc.
# Prescription is the hypertrophy zone: 2-3 sets x 4-10 reps RPE 8-10
# (close to failure). Heavier compound subs that double as main-lift
# volume drop to RPE 6-7.
#
# Picks here are taken from Ben's stated checklist (QHGcZaDuwpQ) plus
# his own training program walkthrough (6UPsJiD7QcE).
#
# Each entry:
#   name, pattern ("compound_accessory" | "isolation"), sets, reps, rpe,
#   rationale, source, tier (1 = top pick, 2 = alternative, 3 = fallback).

EXPERT_ACCESSORIES: dict[str, dict[str, list[dict]]] = {
    "bench": {
        "tricep": [
            {"name": "Single-arm tricep push-down", "pattern": "isolation",
             "sets": (2, 3), "reps": (4, 10), "rpe": (8, 10),
             "rationale": "Ben's tier-1 tricep isolation. Lock the elbow, isolate extension.",
             "source": _src("QHGcZaDuwpQ", "19:35"), "tier": 1},
            {"name": "JM press (Smith machine)", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (4, 10), "rpe": (8, 10),
             "rationale": "Heaviest direct tricep loading available. Ben uses Smith JM press.",
             "source": _src("a5xE7WbtxRs", "8:30"), "tier": 1},
            {"name": "Overhead tricep extension (DB or cable)", "pattern": "isolation",
             "sets": (2, 3), "reps": (4, 12), "rpe": (8, 10),
             "rationale": "Long-head stretched-position option. Alternative to JM press.",
             "source": _src("a5xE7WbtxRs", "8:35"), "tier": 2},
        ],
        "delts": [
            {"name": "DB lateral raise", "pattern": "isolation",
             "sets": (2, 3), "reps": (6, 10), "rpe": (8, 10),
             "rationale": "Medial delt. Cue: push DBs out to the walls (not shrug up).",
             "source": _src("6UPsJiD7QcE", "11:30"), "tier": 1},
            {"name": "DB front raise", "pattern": "isolation",
             "sets": (2, 3), "reps": (6, 12), "rpe": (8, 10),
             "rationale": "Anterior delt. Useful even if you bench frequently.",
             "source": _src("a5xE7WbtxRs", "9:00"), "tier": 1},
            {"name": "Rear-delt fly (machine)", "pattern": "isolation",
             "sets": (2, 3), "reps": (8, 12), "rpe": (8, 10),
             "rationale": "Posterior delt. Shoulder balance + health alongside heavy benching.",
             "source": _src("a5xE7WbtxRs", "8:45"), "tier": 1},
        ],
        "lats": [
            {"name": "Chest-supported wide-grip row", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (5, 10), "rpe": (8, 10),
             "rationale": "Upper back / traps / rear delts. Chest support removes lower-back load.",
             "source": _src("QHGcZaDuwpQ", "18:25"), "tier": 1},
            {"name": "Neutral-grip lat pulldown", "pattern": "isolation",
             "sets": (2, 3), "reps": (6, 12), "rpe": (8, 10),
             "rationale": "Lower-lat fibers via tucked-elbow vertical pull.",
             "source": _src("QHGcZaDuwpQ", "18:05"), "tier": 1},
            {"name": "Wide-grip lat pulldown (overhand)", "pattern": "isolation",
             "sets": (2, 3), "reps": (6, 12), "rpe": (8, 10),
             "rationale": "Upper-lat fibers via flared-elbow vertical pull.",
             "source": _src("QHGcZaDuwpQ", "17:55"), "tier": 1},
        ],
        "chest_supplement": [
            {"name": "Chest press machine", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (5, 10), "rpe": (8, 10),
             "rationale": ("Ben's pick when pecs are the weak point. Stable, "
                           "easy to progressively overload over time."),
             "source": _src("a5xE7WbtxRs", "7:50"), "tier": 1},
            {"name": "Incline DB press", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (5, 10), "rpe": (8, 10),
             "rationale": "Upper pec emphasis. Shoulder-friendly DB version.",
             "source": _src("6UPsJiD7QcE", "8:30"), "tier": 2},
            {"name": "Pec deck fly", "pattern": "isolation",
             "sets": (2, 3), "reps": (6, 10), "rpe": (8, 10),
             "rationale": "Pure pec isolation. Cue: touch elbows together (not hands).",
             "source": _src("6UPsJiD7QcE", "10:30"), "tier": 2},
        ],
        "general": [
            {"name": "Chest-supported wide-grip row", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (5, 10), "rpe": (8, 10),
             "rationale": "Upper back / traps / rear delts — primary bench-day pull.",
             "source": _src("QHGcZaDuwpQ", "18:25"), "tier": 1},
            {"name": "Single-arm tricep push-down", "pattern": "isolation",
             "sets": (2, 3), "reps": (4, 10), "rpe": (8, 10),
             "rationale": "Ben's tier-1 tricep isolation for lockout support.",
             "source": _src("QHGcZaDuwpQ", "19:35"), "tier": 1},
            {"name": "DB lateral raise", "pattern": "isolation",
             "sets": (2, 3), "reps": (6, 10), "rpe": (8, 10),
             "rationale": "Medial delt size.",
             "source": _src("6UPsJiD7QcE", "11:30"), "tier": 1},
            {"name": "Rear-delt fly (machine)", "pattern": "isolation",
             "sets": (2, 3), "reps": (8, 12), "rpe": (8, 10),
             "rationale": "Shoulder balance + health.",
             "source": _src("a5xE7WbtxRs", "8:45"), "tier": 2},
        ],
    },
    "squat": {
        "quad_hypertrophy": [
            {"name": "Leg extension", "pattern": "isolation",
             "sets": (2, 3), "reps": (8, 12), "rpe": (8, 10),
             "rationale": ("Quad as the limiter — \"all stress on one muscle, "
                           "the most efficient exercise for hypertrophy.\""),
             "source": _src("QHGcZaDuwpQ", "9:20"), "tier": 1},
            {"name": "Leg press", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (6, 10), "rpe": (6, 8),
             "rationale": ("Heavier compound sub for extra squat volume. "
                           "Lower RPE (6-7) because of fatigue cost vs "
                           "isolation."),
             "source": _src("QHGcZaDuwpQ", "19:00"), "tier": 1},
            {"name": "Hack squat (machine)", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (6, 10), "rpe": (7, 8),
             "rationale": "Quad-biased pattern with low stabiliser cost.",
             "source": _src("1OXhV1uxPpw", "7:55"), "tier": 2},
            {"name": "Bulgarian split squat (DB)", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (6, 10), "rpe": (6, 8),
             "rationale": "Unilateral quad + glute work. Lower RPE because fatigue cost.",
             "source": _src("QHGcZaDuwpQ", "19:00"), "tier": 2},
        ],
        "posterior_chain": [
            {"name": "Romanian deadlift", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (5, 8), "rpe": (7, 8),
             "rationale": "Hip-hinge hamstrings. Caps at RPE 8 to manage low-back fatigue.",
             "source": _src("n9Wyxr4Yks0", "3:08"), "tier": 1},
            {"name": "Lying leg curl", "pattern": "isolation",
             "sets": (2, 3), "reps": (8, 12), "rpe": (8, 10),
             "rationale": "Hamstring isolation. Knee-flexion bias complements RDL.",
             "source": _src("QHGcZaDuwpQ", "18:35"), "tier": 1},
            {"name": "Seated leg curl", "pattern": "isolation",
             "sets": (2, 3), "reps": (8, 12), "rpe": (8, 10),
             "rationale": "Hamstring isolation under hip flexion (long-muscle-length stretch).",
             "source": _src("QHGcZaDuwpQ", "18:35"), "tier": 1},
            {"name": "45-degree hyperextension", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (8, 12), "rpe": (8, 10),
             "rationale": "Loaded hip extension. Glutes + erectors at low spinal cost.",
             "source": _src("QHGcZaDuwpQ", "19:00"), "tier": 2},
        ],
        "general": [
            {"name": "Leg press", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (6, 10), "rpe": (6, 8),
             "rationale": "Quad volume with low spinal cost. Default squat-day quad work.",
             "source": _src("QHGcZaDuwpQ", "19:00"), "tier": 1},
            {"name": "Romanian deadlift", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (5, 8), "rpe": (7, 8),
             "rationale": "Posterior-chain default — hamstrings carry to squat depth.",
             "source": _src("n9Wyxr4Yks0", "3:08"), "tier": 1},
            {"name": "Lying leg curl", "pattern": "isolation",
             "sets": (2, 3), "reps": (8, 12), "rpe": (8, 10),
             "rationale": "Hamstring isolation.",
             "source": _src("QHGcZaDuwpQ", "18:35"), "tier": 2},
        ],
    },
    "deadlift": {
        "back_thickness": [
            {"name": "Chest-supported wide-grip row", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (5, 10), "rpe": (8, 10),
             "rationale": "Upper-back thickness with no extra low-back load post-DL.",
             "source": _src("QHGcZaDuwpQ", "18:25"), "tier": 1},
            {"name": "Wide-grip lat pulldown (overhand)", "pattern": "isolation",
             "sets": (2, 3), "reps": (6, 12), "rpe": (8, 10),
             "rationale": "Upper-lat width for bar control during the pull.",
             "source": _src("QHGcZaDuwpQ", "17:55"), "tier": 1},
            {"name": "Neutral-grip cable row", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (6, 10), "rpe": (8, 10),
             "rationale": "Lat fibres via closer-elbow horizontal pull.",
             "source": _src("QHGcZaDuwpQ", "18:35"), "tier": 2},
        ],
        "posterior_chain": [
            {"name": "Romanian deadlift", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (5, 8), "rpe": (7, 8),
             "rationale": ("Lighter hinge stimulus when 2x/wk DL is too much. "
                           "Ben's substitute for secondary DL during back issues."),
             "source": _src("n9Wyxr4Yks0", "3:08"), "tier": 1},
            {"name": "45-degree hyperextension", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (8, 12), "rpe": (8, 10),
             "rationale": "Loaded hip extension at low spinal cost.",
             "source": _src("QHGcZaDuwpQ", "19:00"), "tier": 1},
        ],
        "general": [
            {"name": "Chest-supported wide-grip row", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (5, 10), "rpe": (8, 10),
             "rationale": "Back thickness — default DL-day pull.",
             "source": _src("QHGcZaDuwpQ", "18:25"), "tier": 1},
            {"name": "Romanian deadlift", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (5, 8), "rpe": (7, 8),
             "rationale": "Posterior chain — hamstring hinge work.",
             "source": _src("n9Wyxr4Yks0", "3:08"), "tier": 1},
            {"name": "45-degree hyperextension", "pattern": "compound_accessory",
             "sets": (2, 3), "reps": (8, 12), "rpe": (8, 10),
             "rationale": "Glutes + erectors, low spinal cost.",
             "source": _src("QHGcZaDuwpQ", "19:00"), "tier": 2},
        ],
    },
}


# ===========================================================================
# ACCESSORY_REP_RANGES — by accessory CATEGORY, not by block type.
# ===========================================================================
#
# Ben's framework doesn't strongly modulate accessory reps by block type
# (volume vs strength vs peaking). It modulates by ACCESSORY TYPE:
#
#   hypertrophy_isolation       2-3 sets x 4-10 reps  RPE 8-10
#   hypertrophy_compound        2-3 sets x 4-10 reps  RPE 8-10
#   heavy_compound_substitute   2-3 sets x 6-10 reps  RPE 6-7    (leg press / Bulgarian split squat)
#   strength_variation          3-6 sets x 3-6  reps  RPE 5-7    (back-off zone, 60-75% 1RM)
#
# The "by block type" override that the agent expects is kept as a
# back-compat shim that maps to these category defaults (volume/technique
# blocks use heavier compound sub ranges; strength/peaking blocks use
# isolation-leaning ranges to manage fatigue). It's a thin convenience —
# the canonical ranges live in ACCESSORY_REP_RANGES.

ACCESSORY_REP_RANGES: dict[str, dict] = {
    "hypertrophy_isolation": {
        "sets_low": 2, "sets_high": 3,
        "reps_low": 4, "reps_high": 10,
        "rpe_low": 8.0, "rpe_high": 10.0,
        "source": _src("QHGcZaDuwpQ", "20:00")},
    "hypertrophy_compound": {
        "sets_low": 2, "sets_high": 3,
        "reps_low": 4, "reps_high": 10,
        "rpe_low": 8.0, "rpe_high": 10.0,
        "source": _src("QHGcZaDuwpQ", "20:00")},
    "heavy_compound_substitute": {
        "sets_low": 2, "sets_high": 3,
        "reps_low": 6, "reps_high": 10,
        "rpe_low": 6.0, "rpe_high": 7.0,
        "note": "Leg press, Bulgarian split squat — extra squat volume but with low RPE.",
        "source": _src("QHGcZaDuwpQ", "19:00")},
    "strength_variation": {
        "sets_low": 3, "sets_high": 6,
        "reps_low": 3, "reps_high": 6,
        "rpe_low": 5.0, "rpe_high": 7.0,
        "intensity_pct_low": 0.60, "intensity_pct_high": 0.75,
        "source": _src("QHGcZaDuwpQ", "14:30")},
}

# Back-compat: the engine's per-week wave expects rep ranges by accessory
# pattern + block type. Map block types to Ben's category defaults.
ACCESSORY_REP_RANGES_BY_BLOCK: dict[str, dict[str, tuple[int, int]]] = {
    "volume":    {"compound_accessory": (6, 10),  "isolation": (8, 12),  "core": (8, 15)},
    "technique": {"compound_accessory": (5, 10),  "isolation": (8, 12),  "core": (8, 15)},
    "strength":  {"compound_accessory": (4, 8),   "isolation": (6, 10),  "core": (8, 12)},
    "peaking":   {"compound_accessory": (3, 6),   "isolation": (6, 10),  "core": (8, 12)},
}


# ===========================================================================
# AVOID_AS_ACCESSORY — exercises that contradict Ben's principles.
# ===========================================================================
#
# Ben's stated criteria for an accessory (QHGcZaDuwpQ @ 11:20): MUST have
# a specific purpose (strength weak-point fix OR hypertrophy for a muscle
# group). Avoid items here fail those criteria — they don't progress with
# load (planks, ab wheel as anti-extension bodyweight), are trained by
# the main lifts already (push-ups when you bench), or are conditioning
# rather than strength/hypertrophy (burpees, mountain climbers).
#
# Ben doesn't enumerate this list by name in the videos — it's derived
# from the stated philosophy ("muscles grow from mechanical tension; you
# need to add weight over time"). Items here are blocked by the
# validator at plan-commit time.

AVOID_AS_ACCESSORY: frozenset[str] = frozenset({
    "plank", "side plank", "weighted plank",
    "ab wheel", "ab wheel rollout", "weighted ab wheel",
    "russian twist", "weighted russian twist",
    "bodyweight squat", "air squat",
    "bodyweight lunge", "walking lunge (bodyweight)",
    "push-up", "pushup", "weighted push-up",
    "mountain climber", "burpee", "jumping jack",
    "jump rope", "high knees",
    "crunch", "sit-up (bodyweight)", "bicycle crunch",
})


# ===========================================================================
# RPE_ONLY_ACCESSORY_TOKENS — accessories whose load CANNOT be derived from a
# main-lift TM. These are prescribed by RPE: the lifter picks a weight that
# hits the target RPE in the prescribed rep range.
# ===========================================================================
#
# WHY: deriving an accessory load from a barbell TM is only valid when the
# movement scales with that TM. For bodyweight-plus-add-on movements
# (weighted pull-up) or implement-bound ones (DB, cable, machine), a "%TM"
# is physically meaningless — e.g. "Weighted pull-up @ 60% of squat TM"
# would be a near-world-record amount of added weight. These movements are
# prescribed by RPE instead.
#
# Ben's framework treats hypertrophy accessories as RPE-driven (QHGcZaDuwpQ
# @ 20:00, 6UPsJiD7QcE): pick a weight that puts you 0-2 reps from failure
# on the last set. NOT a fraction of squat/bench/deadlift TM.
#
# This list is matched as case-insensitive *substrings* against the exercise
# name (so "DB lateral raise", "Single-arm DB row", "Heavy DB bench press"
# all match "db "). The engine zeroes out intensity_pct when a name matches,
# regardless of the value supplied, so a stray %TM never reaches the lifter.
#
# NOT in this list (these legitimately use main-lift %TM):
#   - Barbell movements that share a clear pattern with the main lift:
#     RDL, good morning, barbell row, snatch-grip DL, deficit DL, paused DL,
#     close-grip bench, paused bench, spoto press, paused/pin/tempo squat,
#     Anderson squat, 1.25 squat, block pull, rack pull.
#   - These are normally role='secondary' anyway, not role='accessory'.

RPE_ONLY_ACCESSORY_TOKENS: tuple[str, ...] = (
    # Dumbbell anything — load varies wildly with the lifter's DB strength.
    "db ", "dumbbell", "dumb-bell",
    # Cable / machine work — implement-bound, no %TM relationship.
    "cable", "machine", "pec deck", "pulldown", "pull-down",
    "leg extension", "leg curl", "hack squat", "leg press",
    "chest press machine", "chest-press machine",
    "smith machine", "preacher curl",
    # Bodyweight + added-weight movements — load is incidental, not %TM.
    "pull-up", "pullup", "pull up", "chin-up", "chinup", "chin up",
    "dip", "weighted dip",
    "hyperextension", "back extension", "45-degree hyperextension",
    "45 degree hyperextension", "ghr", "glute-ham raise", "glute ham raise",
    # Carries — load is grip-limited, not %TM.
    "farmer carry", "farmer's carry", "farmers carry",
    "plate pinch", "captain of crush", "captain-of-crush", "dead hang",
    # Isolation work — fundamentally RPE-driven.
    "lateral raise", "front raise", "rear-delt fly", "rear delt fly",
    "face pull", "band pull-apart", "band pull apart",
    "tricep push-down", "tricep pushdown", "tricep extension",
    "skullcrusher", "skull crusher", "jm press",
    "curl",  # bicep curl, hammer curl, incline curl, etc.
    "fly", "flye",
    # Abs / core — bodyweight or implement-bound.
    "hanging leg raise", "decline sit-up", "decline situp",
    "cable woodchop", "cable wood chop", "ab",
    # Split-squat / Bulgarian work — held by DBs, not bar.
    "split squat", "bulgarian split", "step-up", "step up",
    "single-leg", "single leg",
)


# ===========================================================================
# PRIMARY_QUALITY_VARIATIONS — heavy compound variations that are PRIMARY-
# QUALITY work (same fatigue cost as the regular comp lift) and must NOT
# stack same-day with another primary-quality variation of the same pattern.
# ===========================================================================
#
# Heavy full-ROM compound variations carry the same fatigue cost as the
# regular comp lift, so two of them from the same pattern (e.g. tempo bench
# + close-grip bench) on one day double the main-lift stimulus. Each
# pattern lists the variation name fragments the validator matches;
# `pattern_of_primary_quality_variation` + a rule in guardrails.py reject
# the stacking.

PRIMARY_QUALITY_VARIATIONS: dict[str, tuple[str, ...]] = {
    "bench_pattern": (
        "close-grip bench", "close grip bench", "cgbp",
        "paused bench", "pause bench",
        "long-pause bench", "long pause bench",
        "spoto press", "spoto bench",
        "tempo bench",
        "1.25 bench",
    ),
    "squat_pattern": (
        "paused squat", "pause squat",
        "pin squat",
        "tempo squat",
        "anderson squat",
        "1.25 squat",
    ),
    "deadlift_pattern": (
        "deficit deadlift", "deficit dl",
        "paused deadlift", "paused dl",
        "snatch-grip deadlift", "snatch grip deadlift", "snatch-grip dl",
    ),
}


def pattern_of_primary_quality_variation(name: str) -> str | None:
    """If `name` is a primary-quality variation, return the pattern key
    (bench_pattern / squat_pattern / deadlift_pattern). Otherwise None.

    Substring match, case-insensitive."""
    match = _match_primary_quality_variation(name)
    return match[0] if match else None


def _match_primary_quality_variation(name: str) -> tuple[str, str] | None:
    """Internal: return (pattern, matched_token) so the validator can
    distinguish 'two entries of the SAME variation' (e.g. 'Tempo bench
    (top set)' + 'Tempo bench (back-offs)') from 'two entries of
    DIFFERENT variations' (e.g. tempo bench + close-grip bench).
    Same-variation pairs ARE allowed (Ben's top-set + back-off pattern);
    different-variation pairs are the bug we want to reject."""
    n = (name or "").lower().strip()
    if not n:
        return None
    for pattern, tokens in PRIMARY_QUALITY_VARIATIONS.items():
        for tok in tokens:
            if tok in n:
                return (pattern, tok)
    return None


def is_rpe_only_accessory(name: str) -> bool:
    """True if `name` matches an RPE-only accessory pattern.

    Substring match, case-insensitive. Used by the engine to defensively
    zero out intensity_pct on accessories whose load cannot be meaningfully
    expressed as a fraction of a main-lift TM (DB work, cable / machine
    work, weighted pull-ups, lateral raises, curls, leg extension, etc.).

    See RPE_ONLY_ACCESSORY_TOKENS for the full token list and rationale.
    """
    n = (name or "").lower().strip()
    return any(tok in n for tok in RPE_ONLY_ACCESSORY_TOKENS)


# ===========================================================================
# SEBASTIAN OREB — exercise selection, redundancy, training splits, 5-step
# periodization model. Citations: tjC6ilWMEMM (exercise selection & order),
# xZ2QTewSMuk (periodization performance vs aesthetics), x_uhGTQGrAg
# (training splits). Transcripts: docs/sebastian_transcripts/.
# ===========================================================================

# 8-CATEGORY STRUCTURAL BALANCE MODEL (tjC6ilWMEMM @ 22:34)
# Sebastian's framework for avoiding exercise redundancy: when picking
# accessories or secondary work, pull from a DIFFERENT category than the
# main movement. The 8 categories cover the major movement planes.

SEBASTIAN_EXERCISE_CATEGORIES: dict[str, dict] = {
    "horizontal_push": {
        "examples": ["Bench press", "Close-grip bench", "Paused bench",
                     "DB bench press", "Incline DB press (shallow)",
                     "Floor press", "Pin press (chest height)",
                     "Spoto press", "Tempo bench"],
        "muscles": ["pec major", "anterior delt", "tricep"],
        "note": "Press perpendicular to torso (arms forward).",
        "source": _src("tjC6ilWMEMM", "22:34"),
    },
    "vertical_push": {
        "examples": ["Overhead press", "Seated DB press", "Push press",
                     "High incline DB press", "Z press",
                     "Shoulder press machine"],
        "muscles": ["anterior + lateral delt", "tricep", "upper pec"],
        "note": "Press overhead (arms vertical).",
        "source": _src("tjC6ilWMEMM", "22:50"),
    },
    "horizontal_pull": {
        "examples": ["Barbell row", "Pendlay row", "T-bar row",
                     "DB row (single arm)", "Seal row",
                     "Chest-supported row", "Cable row", "Inverted row"],
        "muscles": ["lats (mid)", "rhomboids", "rear delt", "biceps"],
        "note": "Pull perpendicular to torso.",
        "source": _src("tjC6ilWMEMM", "23:05"),
    },
    "vertical_pull": {
        "examples": ["Pull-up", "Chin-up", "Lat pulldown",
                     "Neutral-grip pulldown", "Wide-grip pulldown",
                     "Weighted pull-up", "Straight-arm pulldown"],
        "muscles": ["lats (full length)", "lower traps", "biceps"],
        "note": "Pull overhead (arms vertical).",
        "source": _src("tjC6ilWMEMM", "23:15"),
    },
    "knee_dominant_squat": {
        "examples": ["High-bar back squat (elevated heels)",
                     "Front squat (elevated heels)",
                     "Stiletto squat", "Hack squat (machine)",
                     "Leg press (high foot for quads)",
                     "Split squat (knee over toe, vertical torso)"],
        "muscles": ["quads (vastus group + rectus femoris partially)",
                    "glutes (less)"],
        "note": ("Vertical torso, knee forward progression over toe, "
                 "maximal knee bend (hamstrings press against calf). "
                 "NOT suitable for 1RM — use moderate-to-high reps."),
        "source": _src("tjC6ilWMEMM", "23:23"),
    },
    "hip_dominant_squat": {
        "examples": ["Low-bar back squat", "Box squat (parallel)",
                     "Paused squat", "Bulgarian split squat (forward lean)",
                     "Goblet squat (sitting back)", "Pin squat"],
        "muscles": ["glutes (mid-range)", "hamstrings", "lower back",
                    "quads (less)"],
        "note": ("Forward torso lean, more hip flexion, less knee bend. "
                 "Powerlifting comp squat lives here. Most loadable squat "
                 "category — heavy/low reps appropriate."),
        "source": _src("tjC6ilWMEMM", "23:45"),
    },
    "hip_hinge": {
        "examples": ["Conventional deadlift", "Sumo deadlift",
                     "Romanian deadlift", "Snatch-grip deadlift",
                     "Deficit deadlift", "Paused deadlift",
                     "Block pull", "Rack pull", "Good morning",
                     "Hip thrust (fully shortened glute)",
                     "45-degree back extension"],
        "muscles": ["glutes (lengthened in DL; shortened in hip thrust)",
                    "hamstrings (long head BF)", "spinal erectors",
                    "lats (isometric)"],
        "note": ("Hip flexion/extension is the primary mover, minimal "
                 "knee bend. Hip thrust + DL pair for full glute range "
                 "(shortened + lengthened)."),
        "source": _src("tjC6ilWMEMM", "39:30"),
    },
    "isolation_joint_health": {
        "examples": ["Leg extension (rectus femoris)",
                     "Leg curl (short head BF)", "Nordic drops",
                     "Calf raise", "DB lateral raise", "Face pull",
                     "External rotation (band/cable)",
                     "Tricep pushdown", "Bicep curl",
                     "Copenhagen plank", "Hanging leg raise"],
        "muscles": ["target muscle isolated"],
        "note": ("Low-cost feel-good closers OR joint-health prehab. "
                 "Sit at end of session — minimal fatigue cost, address "
                 "biarticular muscle gaps (rectus femoris, short head BF) "
                 "or shoulder/hip stability."),
        "source": _src("tjC6ilWMEMM", "8:01"),
    },
}


# EXERCISE REDUNDANCY (tjC6ilWMEMM @ 6:02, 13:35)
# Two exercises are redundant when they train the same muscle in the same
# way through the same joint actions with the same resistance profile,
# without adding meaningful new stimulus. Sebastian's solution: the second
# exercise of a session should address what was NEGLECTED by the first.

EXERCISE_REDUNDANCY_RULES: dict = {
    "definition": (
        "Two exercises both effective in isolation but unnecessary "
        "together — same muscle, same joint actions, same resistance "
        "profile. (Sebastian Oreb, tjC6ilWMEMM @ 6:02)"
    ),
    "examples_redundant": [
        ("Flat barbell bench press", "Flat DB bench press",
         "Same muscle (pec major), same joint actions, same resistance "
         "profile — DB just swaps the implement."),
        ("Low-bar back squat", "Bulgarian split squat (forward lean)",
         "Both hip-dominant squats. Picking BSS as the secondary to a "
         "low-bar squat doesn't address what the squat neglected."),
        ("Wide-grip pulldown", "Lat pulldown (overhand)",
         "Same plane, same muscle, same grip type — only minor width "
         "tweak."),
        ("Conventional deadlift", "Romanian deadlift (heavy)",
         "Same hinge pattern. RDL is fine as accessory but not as "
         "secondary if you've already done conventional heavy."),
    ],
    "exception": (
        "Top set + variation pattern for high-absolute-strength athletes "
        "near competition (tjC6ilWMEMM @ 13:35). The secondary "
        "movement IS technically redundant (same muscle/pattern) but is "
        "a less-loadable variation (paused, tempo, deficit, etc.) — so "
        "you accumulate volume without the systemic fatigue cost of a "
        "second heavy set of the comp lift. Example: 1×1 conventional "
        "DL + 2×5 snatch-grip deficit DL @ ~60% TM (tjC6ilWMEMM @ 19:13)."
    ),
    "source": _src("tjC6ilWMEMM", "6:02"),
}


def is_same_category(ex_name_a: str, ex_name_b: str) -> str | None:
    """If both exercises fall in the same SEBASTIAN_EXERCISE_CATEGORIES
    category, return the category key. Otherwise None.

    Substring matching against each category's examples list — fuzzy by
    design (the LLM may pluralise / abbreviate names)."""
    a = (ex_name_a or "").lower().strip()
    b = (ex_name_b or "").lower().strip()
    if not a or not b:
        return None
    for cat, info in SEBASTIAN_EXERCISE_CATEGORIES.items():
        examples = [e.lower() for e in info["examples"]]
        in_a = any(ex.split("(")[0].strip() in a or a in ex.split("(")[0].strip()
                   for ex in examples)
        in_b = any(ex.split("(")[0].strip() in b or b in ex.split("(")[0].strip()
                   for ex in examples)
        if in_a and in_b:
            return cat
    return None


# NEGLECTED-MUSCLE TARGETS (tjC6ilWMEMM @ 35:50, 40:24, 39:30)
# Specific muscles Sebastian flags as undertrained in compound movements,
# with the isolation that addresses each gap. Use as a checklist when
# building a day's accessory work.

NEGLECTED_MUSCLE_TARGETS: dict[str, dict] = {
    "rectus_femoris": {
        "why_neglected": (
            "Biarticular quad (crosses hip + knee). In a compound squat "
            "the hip and knee move in conflicting actions, so rectus "
            "femoris is underdeveloped. The other three quad heads "
            "(vastus medialis/lateralis/intermedius) are monoarticular "
            "and ARE worked by squats."
        ),
        "isolation": "Leg extension",
        "alternates": ["Sissy squat", "Reverse Nordic"],
        "when_to_add": "Any squat-dominant day, especially hip-dominant.",
        "source": _src("tjC6ilWMEMM", "35:50"),
    },
    "short_head_biceps_femoris": {
        "why_neglected": (
            "One of the 4 hamstring heads. Monoarticular — only crosses "
            "knee, so its only function is knee flexion. Deadlifts / "
            "RDLs / good mornings / back extensions don't bend the knee, "
            "so this head is untouched."
        ),
        "isolation": "Lying leg curl",
        "alternates": ["Seated leg curl", "Nordic drops"],
        "when_to_add": "Any deadlift- or squat-dominant day.",
        "source": _src("tjC6ilWMEMM", "40:24"),
    },
    "glute_shortened_position": {
        "why_neglected": (
            "Deadlifts (and variations like snatch-grip / deficit) load "
            "the glute in its fully LENGTHENED position. The fully "
            "SHORTENED position (top of hip extension) is rarely "
            "loaded heavily without a dedicated movement."
        ),
        "isolation": "Hip thrust",
        "alternates": ["Glute bridge", "Cable kickback (single leg)"],
        "when_to_add": "Pair with deadlifts for full glute ROM.",
        "source": _src("tjC6ilWMEMM", "39:30"),
    },
    "lateral_delt_after_horizontal_press": {
        "why_neglected": (
            "Horizontal pressing (bench) trains anterior delt + pec but "
            "barely touches lateral delt. Lateral delt is the largest "
            "contributor to shoulder width."
        ),
        "isolation": "DB lateral raise",
        "alternates": ["Cable lateral raise", "Machine lateral raise"],
        "when_to_add": "Bench/push day closer.",
        "source": _src("tjC6ilWMEMM", "37:36"),
    },
}


# TRAINING SPLITS CATALOG (x_uhGTQGrAg)
# Sebastian's ranked recommendations per days-per-week available. Each
# entry: structure, when to pick it, when NOT to pick it, citation.

SEBASTIAN_TRAINING_SPLITS: dict[int, dict] = {
    2: {
        "preferred": "whole_body",
        "structure": [
            {"day": "Day 1", "focus": "Whole body (push + pull + squat + bend)",
             "vary": "hip-dominant squat + vertical push"},
            {"day": "Day 2", "focus": "Whole body (push + pull + squat + bend)",
             "vary": "knee-dominant squat + horizontal push"},
        ],
        "intensity": "Take to failure. 5 days of recovery between sessions.",
        "avoid": ("Upper/lower split for 2-day — saturates one body half "
                  "= junk volume in 2nd half of session."),
        "good_for": ["Busy professionals", "Non-strength sport athletes",
                     "Minimal-equipment lifters"],
        "source": _src("x_uhGTQGrAg", "17:53"),
    },
    3: {
        "preferred": "strength_system_3day",
        "alternatives": ["whole_body_each_session", "lilybridge",
                         "daily_undulating_periodization"],
        "structure": [
            {"day": "Day 1", "focus": "Squat day (quad focus)"},
            {"day": "Day 2", "focus": "Whole upper body"},
            {"day": "Day 3", "focus": "Deadlift + secondary quad accessory "
                                       "(e.g. leg extension)"},
        ],
        "avoid_ppl": ("PPL on 3 days gives only 1x/wk quad + 1x/wk upper "
                      "push — insufficient frequency."),
        "lilybridge_note": ("Squat + deadlift same day (heavy SQ + light "
                            "DL one week; reverse next). Works because "
                            "heavy spine-loading max once/wk for high "
                            "absolute strength."),
        "dup_note": ("Daily Undulating Periodization: SQ+BN+DL every "
                     "session with heavy/moderate/light rotation. Good "
                     "for skill acquisition; too much for elite high "
                     "absolute strength; needs added upper back work."),
        "good_for": ["Non-strength sport athletes",
                     "High absolute strength (Lilybridge style)",
                     "Strength specialists"],
        "source": _src("x_uhGTQGrAg", "21:04"),
    },
    4: {
        "preferred": "sebastian_4day",
        "structure": [
            {"day": "Day 1",
             "focus": "Upper body push/pull — HORIZONTAL focus"},
            {"day": "Day 2", "focus": "Lower body — QUAD focus"},
            {"day": "Day 3",
             "focus": "Upper body push/pull — VERTICAL focus"},
            {"day": "Day 4",
             "focus": "Lower body — POSTERIOR focus (with quad accessory)"},
        ],
        "rating": ("Sebastian's preferred split for everyone who can do "
                   "it — 'five gold stars'. Used by Hafthor + Sebastian "
                   "himself."),
        "female_variation": ("One upper body day + one whole-body day "
                             "for lifters requesting less upper, more "
                             "lower (user-requested, not blanket advice)."),
        "good_for": ["Anyone who wants the most out of resistance "
                     "training", "Strength + hypertrophy combined",
                     "Default for intermediate+ powerlifters"],
        "source": _src("x_uhGTQGrAg", "33:26"),
    },
    5: {
        "preferred": "sebastian_5day",
        "structure": [
            {"day": "Day 1", "focus": "Squat (heavy)"},
            {"day": "Day 2", "focus": "Upper body horizontal"},
            {"day": "Day 3", "focus": "Squat (lighter / variation)"},
            {"day": "Day 4", "focus": "Upper body vertical"},
            {"day": "Day 5", "focus": "Deadlift"},
        ],
        "rationale": ("2 squat + 1 deadlift + 2 upper. Deadlifts need "
                      "more recovery than squats — disagrees with "
                      "Poliquin's symmetric squat-DL priority."),
        "avoid_bro_split": ("Chest/back/legs/shoulders/arms = 1x/wk per "
                            "body part + 4 upper / 1 lower = chicken "
                            "legs. Not recommended."),
        "good_for": ["Light high-relative-strength athletes (e.g. <60kg "
                     "competitive)", "Lifters who genuinely have 5 "
                     "days available + good recovery"],
        "caveat": ("Too much for high absolute strength athletes "
                   "(e.g. Hafthor stays at 4 days)."),
        "source": _src("x_uhGTQGrAg", "44:38"),
    },
    6: {
        "preferred": "ppl_x2",
        "structure": [
            {"day": "Day 1", "focus": "Push (horizontal)"},
            {"day": "Day 2", "focus": "Pull (upper body pull)"},
            {"day": "Day 3", "focus": "Legs (quad-focused)"},
            {"day": "Day 4", "focus": "Push (vertical)"},
            {"day": "Day 5", "focus": "Pull (upper body pull)"},
            {"day": "Day 6", "focus": "Legs (posterior / deadlift)"},
        ],
        "important_note": ("Deadlift goes on a LOWER day, NOT pull day."),
        "rating": ("Not recommended optimally. Reserved for lifters who "
                   "need a daily gym routine (mental health, addiction "
                   "recovery, etc.). Same total volume as 4-day, just "
                   "spread thinner."),
        "good_for": ["Lifters needing rigid daily routine"],
        "source": _src("x_uhGTQGrAg", "47:43"),
    },
}


# SEBASTIAN'S 5-STEP PERIODIZATION MODEL (xZ2QTewSMuk @ 22:55)
# His interpretation/expansion of Bompa: prepends two beginner-friendly
# phases (learn-to-move, work-capacity) before the classical accumulation
# → transmutation → realization sequence.

SEBASTIAN_FIVE_STEP_MODEL: list[dict] = [
    {
        "phase": 1,
        "name": "Learn to move",
        "duration_weeks_range": (2, 4),
        "frequency_per_week": (2, 3),
        "description": (
            "One exercise per fundamental movement pattern (squat / hinge "
            "/ push / pull). Same session each time — low variation. "
            "Emphasis is movement literacy, not stimulus. No need to "
            "train to failure."
        ),
        "rep_ranges": {
            "complex (squat, deadlift)": (4, 6),
            "simpler (DB press, pulldown)": (8, 15),
        },
        "applies_to": ["beginner"],
        "source": _src("xZ2QTewSMuk", "24:01"),
    },
    {
        "phase": 2,
        "name": "Build work capacity",
        "duration_weeks_range": (3, 5),
        "description": (
            "Gradually add one set per week (2 → 3 → 4 → 5 sets of the "
            "main movement). Teaches the lifter what it feels like to "
            "work hard. Closer proximity to failure each week. Sets up "
            "for the high-volume hypertrophy phase that follows."
        ),
        "rep_ranges": {"main lift": (5, 10)},
        "applies_to": ["beginner", "returning lifter"],
        "source": _src("xZ2QTewSMuk", "28:23"),
    },
    {
        "phase": 3,
        "name": "Hypertrophy",
        "duration_weeks_range": (4, 8),
        "description": (
            "Build muscle. 10-20 sets per MUSCLE GROUP per week. Start "
            "at low end (10) and only increase if you plateau. Mechanical "
            "tension over absolute load. Target muscle = limiting "
            "factor of the movement."
        ),
        "volume_target": (10, 20),
        "volume_unit": "sets per muscle group per week",
        "rep_ranges": {
            "complex compound": (5, 10),
            "isolation": (8, 30),
        },
        "rep_range_note": ("Brad Schoenfeld meta-analysis: 5-30 reps all "
                           "produce equal hypertrophy IF taken to same "
                           "proximity to failure. Choose range that "
                           "suits the exercise's complexity."),
        "source": _src("xZ2QTewSMuk", "31:21"),
    },
    {
        "phase": 4,
        "name": "Strength",
        "duration_weeks_range": (3, 5),
        "description": (
            "Reduce reps, increase load. 80%+ 1RM on main lifts, ≤6 "
            "reps. 6-12 sets per MOVEMENT PATTERN per week (Rhea & "
            "colleagues 2009 — different denominator from hypertrophy). "
            "Conjugate style: accessory work stays in hypertrophy rep "
            "ranges, doesn't need 80%+."
        ),
        "volume_target": (6, 12),
        "volume_unit": "sets per movement pattern per week",
        "rep_ranges": {"main lift": (1, 6), "accessory": (8, 15)},
        "intensity_pct_min": 0.80,
        "source": _src("xZ2QTewSMuk", "38:32"),
    },
    {
        "phase": 5,
        "name": "Peaking",
        "duration_weeks_range": (4, 6),
        "description": (
            "Maximize performance in the specific task. Minimize volume "
            "by REMOVING non-specific work (bicep curls, calf raises, "
            "etc. — first to cut). Maintain intensity — peaking is NOT "
            "'go lighter', it's 'reduce fatigue by removing the volume "
            "that's not contributing to comp performance'."
        ),
        "rep_ranges": {"main lift": (1, 3)},
        "intensity_pct_min": 0.85,
        "applies_to": ["performance athletes only"],
        "skip_for": "aesthetics goals — peak is via nutrition / body comp",
        "source": _src("xZ2QTewSMuk", "41:45"),
    },
]


# LOADABILITY RULES (tjC6ilWMEMM @ 27:08, 28:00)
# Different movement profiles suit different rep × load combinations.
# Choosing the wrong combo (e.g. 1RM on a stiletto squat) leads to injury.

LOADABILITY_RULES: list[dict] = [
    {
        "profile": "compound_mid_range",
        "examples": ["Low-bar back squat", "Conventional deadlift",
                     "Powerlifting bench press"],
        "suitable_for": "Heavy / low reps (1-6). Load can be spread "
                        "across multiple joints + muscle groups.",
        "rep_range": (1, 6),
        "max_intensity_pct": 1.0,
        "source": _src("tjC6ilWMEMM", "27:08"),
    },
    {
        "profile": "compound_stretched_position",
        "examples": ["Flared-elbow bench (guillotine press)",
                     "Stiletto squat", "Sissy squat",
                     "Deficit deadlift", "Long-pause bench (2-3s)"],
        "suitable_for": "Moderate-to-high reps with LIGHTER loads. NOT "
                        "for 1RM. The muscle is at its weakest "
                        "contractile length — going heavy risks injury "
                        "(pec tears at full stretch, knee strain at "
                        "deep flexion).",
        "rep_range": (6, 12),
        "max_intensity_pct": 0.75,
        "source": _src("tjC6ilWMEMM", "28:00"),
    },
    {
        "profile": "isolation",
        "examples": ["Bicep curl", "Tricep pushdown", "Lateral raise",
                     "Leg extension", "Leg curl", "Face pull",
                     "Calf raise"],
        "suitable_for": "Higher reps (8-20+). Single muscle / single "
                        "joint — no other muscle to share load.",
        "rep_range": (8, 20),
        "max_intensity_pct": 0.50,  # vs related main-lift TM, not "1RM of the isolation"
        "source": _src("tjC6ilWMEMM", "27:30"),
    },
]


# VOLUME TARGETS BY MODE (xZ2QTewSMuk @ 31:21, 39:55; x_uhGTQGrAg @ 14:49)
# CRITICAL: hypertrophy is denominated per MUSCLE GROUP per week (10-20
# sets). Strength is denominated per MOVEMENT PATTERN per week (6-12 sets).
# These are different denominators and different ranges — the LLM should
# pick the right one based on the lifter's current block.

VOLUME_TARGETS_BY_MODE: dict[str, dict] = {
    "hypertrophy": {
        "sets_range": (10, 20),
        "unit": "sets per MUSCLE GROUP per week",
        "starting_recommendation": 10,
        "starting_note": ("Start at the low end (10). Only increase if "
                          "you plateau."),
        "applies_to_phases": ["learn_to_move", "build_work_capacity",
                              "hypertrophy"],
        "source": _src("xZ2QTewSMuk", "31:21"),
    },
    "strength": {
        "sets_range": (6, 12),
        "unit": "sets per MOVEMENT PATTERN per week",
        "starting_recommendation": 6,
        "starting_note": ("Strength uses heavier loads → more fatigue "
                          "per set → fewer total sets needed."),
        "frequency_per_pattern": (2, 3),
        "applies_to_phases": ["strength", "peaking"],
        "source": _src("xZ2QTewSMuk", "39:55"),
    },
}


# ===========================================================================
# ALEXANDER BROMLEY — accessory PROGRESSION methodology, movement-leverage
# classification, weakness-targeting decision framework, dormant-muscle
# wake-up. Citations: czEsWD56hCU (How to Progress Accessory Exercises),
# XogX2nvekME (Exercise Selection Tips). Transcripts:
# docs/bromley_transcripts/.
# ===========================================================================

# Bromley fills the "how do I drive accessories forward week-to-week" gap.
# Ben + Sebastian cover WHAT to pick; Bromley covers HOW to progress it.

BROMLEY_ACCESSORY_PROGRESSION: dict = {
    "core_principle": (
        "Accessory progression can't follow the main-lift template "
        "because fatigue accumulates unpredictably as you go down the "
        "workout. The progression scheme must ABSORB that variability "
        "instead of breaking on it. (Bromley czEsWD56hCU @ 0:55)"
    ),
    "methods": {
        "rep_range_method": {
            "name": "Rep-range method (preferred default)",
            "how_it_works": (
                "Fix SETS and a REP RANGE (e.g. 4 sets × 8-12). Weight "
                "floats within the session: hit a hard set, jump UP to "
                "try fewer reps next, drop DOWN when you can't hit the "
                "bottom of the range. As effort climbs across weeks, "
                "weight drifts down to stay in range. When all sets hit "
                "the TOP of the range, add weight next session."
            ),
            "example": (
                "Target 4 sets × 8-12. Set 1: 225×12 (hard). Set 2: "
                "jump to 235×10. Set 3: 235×9. Set 4: drop to 225×11. "
                "All sets stayed in range — total stress was high. "
                "Next session: try 230 as the starting weight."
            ),
            "best_for": ("Default for ALL accessory work past the "
                         "secondary main lift. Variable loading + "
                         "chase the range + chase stress."),
            "controls": [
                "Fix sets (e.g. 3 or 4).",
                "Fix rep range (e.g. 8-12, 6-8, 10-15).",
                "Float the weight within the session to stay in range.",
                "When all sets hit the top of the range, add load next time.",
            ],
            "source": _src("czEsWD56hCU", "9:14"),
        },
        "dynamic_double_progression": {
            "name": "Dynamic Double Progression (DDP)",
            "how_it_works": (
                "Fix a target: sets × reps × weight. Meet the rep count "
                "across all sets at the weight first, THEN add weight."
            ),
            "example": (
                "Target 4×12 @ 225. Week 1: 12/10/10/9 (failed total "
                "rep count) — keep weight same. Week 2: 12/11/12/10 — "
                "keep weight same. Week 3: 4×12 — add weight next time."
            ),
            "best_for": (
                "The FIRST accessory after the main + secondary main "
                "lifts, when you're still relatively fresh. Breaks "
                "down further into the workout because fatigue from "
                "upstream sets is unpredictable."
            ),
            "source": _src("czEsWD56hCU", "7:43"),
        },
        "block_level_shifting": {
            "name": "Block-level rep-range shift",
            "how_it_works": (
                "Run a rep range for ~4 weeks. Then either shift the "
                "range DOWN (8-12 → 6-8 → 4-6) for the next block, OR "
                "swap the exercise to refresh the stimulus. This sits "
                "ON TOP of either DDP or the rep-range method."
            ),
            "rationale": (
                "Avoids the 'forever 4x8' plateau — every 4 weeks the "
                "lifter has either heavier loads (lower rep range) or "
                "a novel exercise to chase."
            ),
            "source": _src("czEsWD56hCU", "10:45"),
        },
    },
    "effort_targets": {
        "starting_rpe_week_1": 7.0,
        "peak_rpe_block_end": 10.0,
        "rationale": (
            "Bromley: 'get to RPE 10 on your accessory workouts — "
            "RPE 7-8 is still by default a pretty good target range, "
            "but you can definitely allow yourself to get into the "
            "weeds' (czEsWD56hCU @ 12:24). The block ramps from "
            "easier early sets to maximal stress by the end."
        ),
        "why_safe": (
            "Accessories don't accrue fatigue at the rate main lifts "
            "do — hamstring curls, back extensions, DB RDLs don't "
            "wreck recovery like barbell deadlifts. So pushing to "
            "RPE 10 on isolations is fine. (czEsWD56hCU @ 13:25)"
        ),
        "exception": (
            "Spinal-loading or heavy compound accessories (deficit "
            "DL, good morning, heavy RDL) — DON'T push to RPE 10. "
            "The cost is real even though they're 'accessories'."
        ),
        "source": _src("czEsWD56hCU", "12:24"),
    },
    "fatigue_gradient_in_session": (
        "Order: main → secondary main → primary accessory (compound, "
        "still taxing) → secondary accessory (isolation, less taxing). "
        "Expect MORE fatigue at each step. DDP can work on exercise "
        "#3; breaks down by #5. Default to rep-range method everywhere "
        "after the secondary main lift. (czEsWD56hCU @ 3:46)"
    ),
    "go_ham_rule": (
        "Bromley: 'go ham, go nuts, the world is yours' (czEsWD56hCU "
        "@ 13:44). On lighter accessories, focus on stress accumulation, "
        "not precise progression. Stress is the goal, not the number "
        "on the bar."
    ),
    "source": _src("czEsWD56hCU", "9:14"),
}


BROMLEY_MOVEMENT_LEVERAGE: dict = {
    "framework_origin": (
        "Brian Carroll's 'mechanically similar movements' classification, "
        "expanded by Bromley with the advantaged/disadvantaged leverage "
        "axis. (XogX2nvekME @ 6:43)"
    ),
    "leverage_categories": {
        "advantaged": {
            "description": (
                "Better leverage, shorter range of motion, more weight "
                "moved per unit effort. CNS stimulation, force-"
                "production, neurological adaptation. Trains the body "
                "to handle heavier loads."
            ),
            "examples": [
                "Board press", "Pin press (above sticking point)",
                "Block pull", "Rack pull", "Trap bar deadlift",
                "Banded squat", "Banded bench", "Heavy walkout",
                "Partial squat (above parallel)",
            ],
            "best_for_phase": ["strength", "peaking"],
            "rep_range": (1, 5),
            "rationale": (
                "Shorter ROM + better leverage → can overload. Trains "
                "the nervous system to liberate more motor units, "
                "faster. Great for peak-phase CNS prep — old-school "
                "heavy walkouts before a meet (XogX2nvekME @ 10:50)."
            ),
            "source": _src("XogX2nvekME", "9:30"),
        },
        "disadvantaged": {
            "description": (
                "Longer range of motion, more time under tension, "
                "worse leverage. Lighter loads, inefficient — and "
                "that's exactly the point: high stress at low load = "
                "hypertrophy stimulus without main-lift fatigue cost."
            ),
            "examples": [
                "Deficit deadlift", "Stiff-leg deadlift",
                "Romanian deadlift (heavy ROM)",
                "Long-pause bench (3+s)", "Pause squat (3s in hole)",
                "Front squat", "Spoto press",
                "Flared / guillotine bench",
                "1.25 squat", "1.25 bench", "Snatch-grip deadlift",
                "Tempo squat (3s eccentric)",
            ],
            "best_for_phase": ["volume", "technique"],
            "rep_range": (5, 12),
            "rationale": (
                "Longer ROM + worse leverage → high stress per kg → "
                "high total stress at lighter loads = hypertrophy "
                "stimulus without burning main-lift recovery. Bromley: "
                "'disadvantaged is akin to being a newbie — high effort "
                "but inefficient, so systemic cost is low' "
                "(XogX2nvekME @ 12:04). This is how Westside maxed "
                "every week (XogX2nvekME @ 12:30)."
            ),
            "source": _src("XogX2nvekME", "11:40"),
        },
        "neutral_compound_accessory": {
            "description": (
                "Less mechanically similar to the main lift but trains "
                "the same muscles / movement pattern broadly. Lower "
                "impact on main-lift recovery than mechanically-similar "
                "variations because the neurological action is different."
            ),
            "examples": [
                "Good morning (for DL / squat)", "Weighted dip",
                "Bulgarian split squat", "Single-leg RDL",
                "JM press", "Barbell row", "Pendlay row",
                "T-bar row", "Chest-supported row",
                "Hip thrust", "Hack squat", "Leg press",
            ],
            "best_for_phase": ["volume", "technique", "strength"],
            "rep_range": (6, 12),
            "rationale": (
                "Hits the muscles without duplicating the main lift's "
                "exact fatigue cost. Sustainable across blocks. "
                "(XogX2nvekME @ 20:24)"
            ),
            "source": _src("XogX2nvekME", "8:09"),
        },
        "lighter_isolation": {
            "description": (
                "Machines, dumbbells, cables, isolation. Easy to recover "
                "from. Can train at high frequency (2-3x/wk per muscle). "
                "Default for the back half of every session."
            ),
            "examples": [
                "Cable tricep pushdown", "Band pushdown",
                "DB lateral raise", "DB front raise",
                "Lying leg curl", "Seated leg curl",
                "Leg extension", "Preacher curl",
                "Hammer curl", "Cable row (single arm)",
                "Face pull", "45° hyperextension (low load)",
                "DB fly", "Pec deck", "Cable kickback",
            ],
            "best_for_phase": ["volume", "technique", "strength"],
            "rep_range": (8, 20),
            "rationale": (
                "Smaller load + smaller muscle → easier recovery → "
                "tolerates high frequency and high effort (RPE 9-10 "
                "OK). Bromley: 'use these liberally — most of you "
                "aren't doing enough' (XogX2nvekME @ 22:46)."
            ),
            "source": _src("XogX2nvekME", "22:46"),
        },
    },
    "phase_mapping": {
        "volume": {
            "primary_leverage": "disadvantaged",
            "secondary_leverage": "neutral_compound_accessory",
            "tertiary_leverage": "lighter_isolation",
            "rationale": (
                "Volume/hypertrophy benefits from longer ROM + more "
                "time under tension at lighter loads. Build base "
                "broadly. (XogX2nvekME @ 14:20)"
            ),
        },
        "technique": {
            "primary_leverage": "disadvantaged",
            "secondary_leverage": "neutral_compound_accessory",
            "tertiary_leverage": "lighter_isolation",
            "rationale": (
                "Disadvantaged variations (paused, tempo, deficit) "
                "directly train technique. Same logic as volume phase."
            ),
        },
        "strength": {
            "primary_leverage": "neutral_compound_accessory",
            "secondary_leverage": "advantaged",
            "tertiary_leverage": "lighter_isolation",
            "rationale": (
                "Higher specificity to the main lift, more force "
                "production. Less time on disadvantaged variations; "
                "still some isolation for muscle balance."
            ),
        },
        "peaking": {
            "primary_leverage": "advantaged",
            "secondary_leverage": "lighter_isolation",
            "tertiary_leverage": "drop",
            "rationale": (
                "Bromley: 'usually you're gonna ditch this stuff going "
                "into peak prep — it's an easy way to eliminate extra "
                "stress' (XogX2nvekME @ 23:33). Cut high-volume "
                "accessory work, focus on overloaded specific patterns."
            ),
        },
    },
    "source": _src("XogX2nvekME", "6:43"),
}


BROMLEY_WEAKNESS_TARGETING: dict = {
    "decision_framework": (
        "When a lifter has a known weakness (sticking point, missed "
        "lockouts), two valid targeting strategies. Pick one or use "
        "both. (Bromley XogX2nvekME @ 24:40)"
    ),
    "pattern_targeting": {
        "description": (
            "Train the MOVEMENT PATTERN at the sticking point. Build "
            "motor unit recruitment + muscular endurance AROUND the "
            "failure spot."
        ),
        "examples": [
            {"weakness": "Bench lockout (top half slow)",
             "exercises": ["Pin press at lockout height (3-5 reps)",
                           "2-board press / 3-board press",
                           "Banded bench (overload at top)"]},
            {"weakness": "Bench off the chest (slow off the bottom)",
             "exercises": ["Long-pause bench (2-3s on chest, no bounce)",
                           "Pin press at chest height",
                           "Spoto press"]},
            {"weakness": "Squat out of the hole",
             "exercises": ["Pause squat (3s in the hole)",
                           "Pin squat at depth",
                           "Tempo squat (3s eccentric, no bounce)"]},
            {"weakness": "Deadlift off the floor",
             "exercises": ["Deficit deadlift (1-2 inches)",
                           "Paused deadlift (1s, 1\" off floor)",
                           "Snatch-grip deadlift"]},
            {"weakness": "Deadlift at the knee",
             "exercises": ["Pause deadlift just below the knee",
                           "Rack pull from just below the knee",
                           "Tempo deadlift (3s eccentric)"]},
        ],
        "when_to_use": (
            "Default for intermediate+ lifters who know their sticking "
            "point. Pattern-targeting attacks the failure spot directly. "
            "Works best at HEAVIER loads with LOWER reps (3-5)."
        ),
        "source": _src("XogX2nvekME", "25:50"),
    },
    "muscle_targeting": {
        "description": (
            "Reverse-engineer which MUSCLE failed first, then isolate "
            "it. Build the specific weak link without burning the whole "
            "movement."
        ),
        "examples": [
            {"weakness": "Bench lockout (triceps suspected)",
             "exercises": ["JM press (heavy)", "Skullcrusher",
                           "Cable pushdown (high volume)",
                           "Overhead tricep extension"]},
            {"weakness": "Bench off the chest (pec / anterior delt)",
             "exercises": ["Heavy incline DB press", "Pec deck",
                           "Cable fly", "DB bench"]},
            {"weakness": "Squat (quads suspected)",
             "exercises": ["Heavy leg press", "Front squat",
                           "Leg extension (heavy + high-rep)"]},
            {"weakness": "Squat (glutes / posterior suspected)",
             "exercises": ["Heavy good morning", "RDL",
                           "Hip thrust"]},
            {"weakness": "DL off the floor (quads / hips suspected)",
             "exercises": ["Pause squat", "Leg press",
                           "Deficit deadlift"]},
            {"weakness": "DL lockout (glutes / lower back suspected)",
             "exercises": ["Heavy hip thrust", "Back extension",
                           "Rack pull just below lockout"]},
        ],
        "when_to_use": (
            "When the lifter can identify WHICH muscle failed (rather "
            "than just where). Often used alongside pattern targeting — "
            "do pattern work on the main day, muscle isolation on a "
            "secondary or accessory slot."
        ),
        "source": _src("XogX2nvekME", "27:18"),
    },
    "novice_caveat": (
        "Bromley: 'focus targeting is only appropriate for mid-"
        "intermediates and up' (XogX2nvekME @ 16:55). Novices should "
        "just do general development — they don't have specific "
        "weaknesses yet, they're just untrained. Broad base first."
    ),
    "source": _src("XogX2nvekME", "24:40"),
}


BROMLEY_DORMANT_MUSCLE_PROTOCOL: dict = {
    "concept": (
        "Some muscles never recruit well because they've never been "
        "trained directly. Bromley's case study: his erectors stayed "
        "thin through years of barbell deadlifts; weeks of reverse "
        "hypers + RDLs and they popped. High-rep work (15-20 reps) "
        "at light loads teaches motor unit recruitment of dormant "
        "muscles. (XogX2nvekME @ 33:00)"
    ),
    "prescription": {
        "sets_range": (2, 3),
        "reps_range": (15, 20),
        "rpe_range": (8.0, 10.0),
        "load_guideline": ("Light enough to hit the full rep count and "
                           "still feel a deep burn — Bromley: 'high rep "
                           "shit that just burns can be the ticket to "
                           "waking up dormant muscles.' (XogX2nvekME @ "
                           "33:00) Quality > weight."),
    },
    "example_protocol_kickbacks": (
        "Tricep kickbacks with pink 5-lb dumbbells × 30 reps in a steep "
        "forward bend. Bromley: 'you won't be able to pick up the cell "
        "phone — it's good work.' (XogX2nvekME @ 33:48)"
    ),
    "common_dormant_muscles_by_lift": {
        "squat": [
            {"muscle": "rectus femoris",
             "exercise": "Leg extension (high-rep wake-up, 15-20)",
             "note": "Biarticular quad undertrained in compound squats."},
            {"muscle": "vastus medialis (VMO)",
             "exercise": "Knee-over-toe split squat (light, slow tempo)",
             "note": "Often dormant in lifters who avoid deep knee bend."},
        ],
        "bench": [
            {"muscle": "tricep long head",
             "exercise": "Overhead tricep extension (DB, 15-20 reps slow)",
             "note": "Crosses the shoulder; shortened in normal pressing."},
            {"muscle": "rear delt",
             "exercise": "Cable rear-delt fly (15-20 reps controlled)",
             "note": "Almost untrained in pure pressing programs."},
            {"muscle": "upper pec / clavicular head",
             "exercise": "Incline cable fly (15-20 reps)",
             "note": "Flat bench focuses on sternal head."},
        ],
        "deadlift": [
            {"muscle": "spinal erectors",
             "exercise": "Reverse hyper (light load, 15-20) OR 45° back "
                         "extension high-rep",
             "note": "Bromley's own case study — popped after weeks of "
                     "reverse hyper work he previously dismissed."},
            {"muscle": "glute (fully-shortened position)",
             "exercise": "Cable kickback / paused hip thrust at top",
             "note": "DL only loads lengthened glute; shortened needs "
                     "its own exercise."},
            {"muscle": "short head biceps femoris",
             "exercise": "Lying leg curl (15-20 reps, slow eccentric)",
             "note": "Knee flexion only target — never trained by DLs."},
        ],
    },
    "when_to_use": (
        "Add ONE high-rep wake-up exercise at the END of a session "
        "when a specific muscle group has been unresponsive to normal "
        "accessory work over 1-2 blocks. Not a permanent fixture — "
        "rotate in for 4-8 weeks, then back to normal accessory work "
        "once recruitment improves."
    ),
    "source": _src("XogX2nvekME", "33:00"),
}


BROMLEY_SESSION_ORDER: dict = {
    "principle": (
        "Performance precision DECREASES as you go down the workout. "
        "Track + plan precisely on the main lift and the mechanically "
        "similar movement. Chase stress (not specific numbers) on "
        "lighter accessories. (czEsWD56hCU @ 4:25)"
    ),
    "session_gradient": [
        {
            "position": 1,
            "name": "Main lift",
            "leverage": "specific (comp lift)",
            "progression_method": ("planned (RPE wave / top set + "
                                   "back-off per Ben's framework)"),
            "performance_precision": "high — every rep tracked",
            "rpe_target": ("block-specific — see Ben's "
                           "block_stacking_schedule"),
        },
        {
            "position": 2,
            "name": "Secondary main lift (mechanically similar)",
            "leverage": "advantaged or disadvantaged",
            "progression_method": ("planned — DDP works here OR rep-"
                                   "range method"),
            "performance_precision": "high — track sets and reps",
            "rpe_target": ("RPE 6-8 (Bromley) — Sebastian-style "
                           "lighter top set + back-off on a secondary "
                           "day"),
        },
        {
            "position": 3,
            "name": "Primary accessory (compound, still taxing)",
            "leverage": "neutral_compound_accessory",
            "progression_method": ("rep-range method (DDP starts to "
                                   "break down here)"),
            "performance_precision": "moderate — track total volume",
            "rpe_target": "RPE 7-9",
        },
        {
            "position": 4,
            "name": "Secondary accessory (lighter isolation)",
            "leverage": "lighter_isolation",
            "progression_method": "rep-range method",
            "performance_precision": ("low — chase stress, wing it, "
                                      "Bromley's 'go ham' zone"),
            "rpe_target": ("RPE 8-10 (close to failure on isolations "
                           "is OK)"),
        },
    ],
    "go_ham_rule": (
        "Bromley: 'go ham, go nuts, the world is yours' (czEsWD56hCU "
        "@ 13:44). On lighter accessories, focus on stress "
        "accumulation rather than precise progression. They don't "
        "accrue main-lift fatigue, so the cost is low."
    ),
    "source": _src("czEsWD56hCU", "4:25"),
}


# ===========================================================================
# Lookups
# ===========================================================================

def recommend_accessories(lift: str, target: str,
                          block_type: str | None = None,
                          count: int = 3) -> list[dict]:
    """Return up to N hypertrophy-accessory picks for (lift, target),
    ordered by tier. Falls back to the lift's 'general' pool if `target`
    isn't catalogued.

    When block_type is provided, picks' rep ranges are overridden via
    ACCESSORY_REP_RANGES_BY_BLOCK for back-compat with the engine's
    per-week wave. Time-based reps (e.g. '30-60s walk') are passed
    through unchanged.
    """
    by_lift = EXPERT_ACCESSORIES.get(lift, {})
    options = by_lift.get(target) or by_lift.get("general", [])
    out: list[dict] = []
    for opt in sorted(options, key=lambda o: o["tier"])[:count]:
        pick = dict(opt)
        if block_type and isinstance(pick.get("reps"), tuple):
            block_ranges = ACCESSORY_REP_RANGES_BY_BLOCK.get(block_type, {})
            override = block_ranges.get(pick["pattern"])
            if override:
                pick["reps"] = override
        out.append(pick)
    return out


def recommend_strength_variation(lift: str, weakness: str) -> list[dict]:
    """Return ranked strength-accessory variations for (lift, weakness).
    These are role='secondary' / dedicated-day movements with a specific
    weak-point purpose — pin squat for depth, pause DL for start
    position, long-pause bench for chest tightness, etc.
    """
    by_lift = STRENGTH_ACCESSORIES.get(lift, {})
    return list(by_lift.get(weakness, []))


def prescription_pattern(name: str) -> dict | None:
    """Fetch a top-set+backoff (primary/secondary) or straight-sets
    prescription pattern."""
    return PRESCRIPTION_PATTERNS.get(name)


def volume_landmark(lift: str) -> dict | None:
    """Ben's per-week main-lift set/session targets for a lift."""
    return VOLUME_LANDMARKS.get(lift)


def block_periodization() -> dict:
    """Ben's block-stacking + time-to-peak model."""
    return BLOCK_PERIODIZATION


def block_stacking_schedule(experience: str | None = None) -> dict:
    """Ben's per-position RPE schedule across a block stack (406rClFwNxY).
    Returns the whole dict if `experience` is None, or the specific
    experience-level slot if provided. Falls back to 'intermediate' for
    unrecognized values (the most common case)."""
    if experience is None:
        return BLOCK_STACKING
    return BLOCK_STACKING.get(experience, BLOCK_STACKING["intermediate"])


def time_to_peak() -> dict:
    """Ben's bottom-up programming process for finding personal block length."""
    return TIME_TO_PEAK


def rpe_usage_principles() -> dict:
    """Ben's RPE-usage principle (Klt4SLA64iE). RPE dictates the weight on
    the bar, not the other way round. Use this to ground coaching about
    how the lifter should USE the prescription on the day."""
    return RPE_USAGE


# ----- Sebastian Oreb lookups -----

def exercise_categories() -> dict:
    """Sebastian's 8-category structural balance model (tjC6ilWMEMM @
    22:34). Each category lists example exercises + the muscles trained +
    a movement-profile note. Use to avoid exercise redundancy when
    picking a secondary or accessory."""
    return SEBASTIAN_EXERCISE_CATEGORIES


def exercise_redundancy_rules() -> dict:
    """Sebastian's exercise-redundancy definition + examples + exception
    (tjC6ilWMEMM @ 6:02 / 13:35)."""
    return EXERCISE_REDUNDANCY_RULES


def check_redundancy(ex_name_a: str, ex_name_b: str) -> dict:
    """Detect whether two exercises are redundant per Sebastian's
    definition. Returns {"redundant": bool, "shared_category": str|None,
    "reason": str}.

    Substring match against SEBASTIAN_EXERCISE_CATEGORIES — fuzzy by
    design. False positives are acceptable; the LLM gets a hint that two
    exercises overlap and can either pick a different category or
    explicitly justify the redundancy (top-set + variation pattern)."""
    shared = is_same_category(ex_name_a, ex_name_b)
    if shared is None:
        return {"redundant": False, "shared_category": None,
                "reason": (f"'{ex_name_a}' and '{ex_name_b}' fall in "
                           f"different categories — not redundant.")}
    return {"redundant": True, "shared_category": shared,
            "reason": (f"Both '{ex_name_a}' and '{ex_name_b}' are in "
                       f"the '{shared}' category — same movement plane / "
                       f"muscle / resistance profile. To address what "
                       f"the first exercise neglected, pick the "
                       f"secondary from a DIFFERENT category. (Exception: "
                       f"top set + variation pattern for advanced "
                       f"lifters near comp — see "
                       f"exercise_redundancy_rules.)")}


def neglected_muscle_targets(main_lift: str | None = None) -> dict | list[dict]:
    """Sebastian's checklist of muscles that compound movements
    undertrain, with the isolation that fixes each (tjC6ilWMEMM @ 35:50 /
    39:30 / 40:24).

    If main_lift is squat or deadlift, returns the relevant subset
    (rectus femoris for squat-dominant; short head BF + glute-shortened
    for hinge-dominant). Without an argument, returns the whole dict."""
    if main_lift is None:
        return NEGLECTED_MUSCLE_TARGETS
    lift = main_lift.lower()
    keys_by_lift = {
        "squat": ["rectus_femoris", "short_head_biceps_femoris"],
        "deadlift": ["short_head_biceps_femoris", "glute_shortened_position"],
        "bench": ["lateral_delt_after_horizontal_press"],
    }
    keys = keys_by_lift.get(lift, list(NEGLECTED_MUSCLE_TARGETS.keys()))
    return [NEGLECTED_MUSCLE_TARGETS[k] | {"key": k} for k in keys
            if k in NEGLECTED_MUSCLE_TARGETS]


def training_splits(days_per_week: int | None = None) -> dict:
    """Sebastian's training-splits catalog (x_uhGTQGrAg). Returns the
    entry for the supplied days_per_week, or the whole dict if omitted.
    Clamps days_per_week to the [2, 6] range (1-day not catalogued, 7-day
    not recommended)."""
    if days_per_week is None:
        return SEBASTIAN_TRAINING_SPLITS
    days = max(2, min(6, int(days_per_week)))
    return SEBASTIAN_TRAINING_SPLITS[days]


def five_step_model(phase: int | None = None) -> list[dict] | dict:
    """Sebastian's 5-step periodization model (xZ2QTewSMuk @ 22:55).
    Returns the full list if `phase` is None, or the specified phase
    (1=learn-to-move, 2=work-capacity, 3=hypertrophy, 4=strength,
    5=peaking)."""
    if phase is None:
        return SEBASTIAN_FIVE_STEP_MODEL
    if 1 <= phase <= 5:
        return SEBASTIAN_FIVE_STEP_MODEL[phase - 1]
    raise ValueError(f"phase must be 1-5, got {phase}")


def loadability_rules() -> list[dict]:
    """Sebastian's rule of thumb mapping movement profile → suitable rep
    range + max intensity (tjC6ilWMEMM @ 27:08). Compound mid-range =
    heavy/low; stretched-position = light/moderate; isolation = high reps."""
    return LOADABILITY_RULES


def volume_targets(mode: str | None = None) -> dict:
    """Hypertrophy vs strength volume targets — CRITICAL distinction:
    hypertrophy is per MUSCLE GROUP, strength is per MOVEMENT PATTERN.
    Returns the whole dict if mode is None, or the mode entry directly."""
    if mode is None:
        return VOLUME_TARGETS_BY_MODE
    m = mode.lower()
    if m in VOLUME_TARGETS_BY_MODE:
        return VOLUME_TARGETS_BY_MODE[m]
    raise ValueError(f"mode must be 'hypertrophy' or 'strength', got {mode}")


# ----- Alexander Bromley lookups -----

def accessory_progression_methods() -> dict:
    """Bromley's accessory progression methodology (czEsWD56hCU): rep-range
    method (default), Dynamic Double Progression (for the first accessory),
    block-level rep-range shifting. Includes effort targets (start RPE 7,
    peak RPE 10 on isolations) and the fatigue gradient through a session."""
    return BROMLEY_ACCESSORY_PROGRESSION


def classify_movement_leverage(exercise_name: str) -> dict:
    """Return Bromley's leverage classification for an exercise.
    Substring match against example lists. Returns:
        {category: 'advantaged' | 'disadvantaged' |
                   'neutral_compound_accessory' | 'lighter_isolation' |
                   'unknown',
         exercise, best_for_phase, rep_range, reason, source}
    """
    name = (exercise_name or "").lower().strip()
    if not name:
        return {"category": "unknown", "exercise": exercise_name,
                "reason": "empty exercise name"}
    leverage = BROMLEY_MOVEMENT_LEVERAGE["leverage_categories"]
    for cat, info in leverage.items():
        for ex in info["examples"]:
            ex_lower = ex.lower()
            ex_base = ex_lower.split("(")[0].strip()
            # Bidirectional substring match — name contains ex_base OR
            # ex_base contains name (handles abbreviations + variants).
            if ex_base and (ex_base in name or name in ex_base):
                return {"category": cat,
                        "exercise": exercise_name,
                        "best_for_phase": info["best_for_phase"],
                        "rep_range": info["rep_range"],
                        "matched_example": ex,
                        "rationale": info["rationale"],
                        "source": info["source"]}
    return {"category": "unknown",
            "exercise": exercise_name,
            "reason": (f"'{exercise_name}' didn't match a known leverage "
                       f"category. Pick from advantaged / disadvantaged / "
                       f"neutral_compound_accessory / lighter_isolation "
                       f"based on the movement profile.")}


def leverage_for_block(block_type: str) -> dict:
    """Return Bromley's phase-mapping recommendation for which leverage
    categories to favor in this block type (volume/technique/strength/
    peaking). Returns the mapping dict directly."""
    bt = (block_type or "").lower()
    mapping = BROMLEY_MOVEMENT_LEVERAGE["phase_mapping"]
    if bt in mapping:
        return mapping[bt] | {"block_type": bt}
    return {"block_type": bt, "error": (
        f"unknown block_type {block_type!r}; valid: "
        f"{list(mapping.keys())}")}


def weakness_targeting_options() -> dict:
    """Bromley's pattern-vs-muscle decision framework for fixing a known
    weakness (XogX2nvekME @ 24:40). Returns both options + the novice
    caveat. Caller decides which to use (or both)."""
    return BROMLEY_WEAKNESS_TARGETING


def dormant_muscle_protocol(lift: str | None = None) -> dict:
    """Bromley's high-rep wake-up protocol for muscles that don't recruit
    well (XogX2nvekME @ 33:00). With `lift` (squat/bench/deadlift),
    returns the dormant-muscle picks for that lift's day."""
    if lift is None:
        return BROMLEY_DORMANT_MUSCLE_PROTOCOL
    lift_lower = lift.lower()
    out = dict(BROMLEY_DORMANT_MUSCLE_PROTOCOL)
    common = BROMLEY_DORMANT_MUSCLE_PROTOCOL["common_dormant_muscles_by_lift"]
    if lift_lower in common:
        out["recommended_for_lift"] = common[lift_lower]
    return out


def session_order_principle() -> dict:
    """Bromley's fatigue gradient through the session — track precisely
    on top, chase stress on bottom (czEsWD56hCU @ 4:25)."""
    return BROMLEY_SESSION_ORDER


def compute_macrocycle_sizing(
    weeks_to_comp: int,
    experience: str = "intermediate",
    lifter_status: str | None = None,
) -> dict:
    """Deterministic macrocycle sizing — given weeks until competition,
    return the recommended block sequence. The LLM should call this
    instead of guessing block counts; the engine math is reliable.

    Combines Ben's block-stacking (final block = PR / peaking) +
    Sebastian's block-length defaults (5 weeks productive + 1 deload
    week absorbed into the block, so each block ≈ 5 calendar weeks).

    `lifter_status` is a soft hint to bias the block-type sequence:
      - "rebuilding" / "post-injury" → start with extra volume blocks
      - "advanced" / "experienced"   → can skip directly to strength
      - default (None or anything else) → standard mix

    Returns:
        {
          weeks_to_comp: int,
          block_length: int,  # weeks per block including deload
          num_blocks: int,
          sequence: [{position, block_type, planned_weeks, role}, ...],
          rationale: str,
        }

    Source: Ben 406rClFwNxY @ 11:43 (work backwards from comp);
            Sebastian xZ2QTewSMuk @ 18:51 (4-5 week messycle default).
    """
    weeks_to_comp = max(4, int(weeks_to_comp))
    block_length = 5
    num_blocks = max(1, round(weeks_to_comp / block_length))

    # Allow ±2-week overshoot: short overshoots are fine (lifter starts a
    # couple weeks earlier than strictly needed). Only drop a block if
    # we'd overshoot by MORE than 2 weeks — otherwise we'd undersize
    # naturally-rounding cases like 13wk → 3 blocks (15wk total, +2).
    if num_blocks * block_length > weeks_to_comp + 2:
        num_blocks -= 1

    # Block-type sequence: always end on peaking. Earlier blocks are
    # developmental (volume / strength). Sequence biased by lifter_status.
    status = (lifter_status or "").lower()
    is_rebuilding = any(t in status for t in
                        ("rebuild", "rebuilding", "post-injury", "injury",
                         "returning", "deconditioned"))
    is_advanced = any(t in status for t in
                      ("advanced", "experienced", "elite"))

    def _seq(n: int) -> list[str]:
        if n <= 1:
            return ["peaking"]
        if n == 2:
            return ["strength", "peaking"]
        if n == 3:
            if is_rebuilding:
                return ["volume", "strength", "peaking"]
            if is_advanced:
                return ["strength", "strength", "peaking"]
            return ["volume", "strength", "peaking"]
        if n == 4:
            if is_rebuilding:
                return ["volume", "volume", "strength", "peaking"]
            if is_advanced:
                return ["strength", "strength", "strength", "peaking"]
            return ["volume", "strength", "strength", "peaking"]
        if n == 5:
            if is_rebuilding:
                return ["volume", "volume", "strength", "strength", "peaking"]
            if is_advanced:
                return ["volume", "strength", "strength", "strength", "peaking"]
            return ["volume", "strength", "strength", "strength", "peaking"]
        # 6+ blocks: pad with developmental blocks at the front.
        head = ["volume"] * (n - 4) + ["volume", "strength", "strength"]
        return head + ["peaking"]

    types = _seq(num_blocks)
    roles = ["deposit"] * (num_blocks - 1) + ["PR"]
    sequence = [
        {"position": i + 1, "block_type": t,
         "planned_weeks": block_length, "role": r}
        for i, (t, r) in enumerate(zip(types, roles))
    ]

    rationale = (
        f"{weeks_to_comp}wk to comp / {block_length}wk per block = "
        f"{num_blocks} blocks. Final block is peaking, ending on the "
        f"meet date. Earlier blocks are deposits per Ben's stacking "
        f"(406rClFwNxY @ 8:45). Sequence biased by lifter_status="
        f"{lifter_status!r}: {' → '.join(t.title() for t in types)}."
    )
    return {
        "weeks_to_comp": weeks_to_comp,
        "block_length": block_length,
        "num_blocks": num_blocks,
        "sequence": sequence,
        "rationale": rationale,
        "source": _src("406rClFwNxY", "11:43"),
    }


def hypertrophy_checklist() -> list[dict]:
    """Ben's default new-client hypertrophy accessory framework."""
    return list(HYPERTROPHY_CHECKLIST)


def is_avoided(exercise_name: str) -> bool:
    """True if this exercise is in the avoid list (normalised, lowercased)."""
    return exercise_name.lower().strip() in AVOID_AS_ACCESSORY


def accessory_rep_range(pattern: str, block_type: str) -> tuple[int, int]:
    """Back-compat: rep range for a pattern in a given block. Falls back
    to (8, 12), Ben's default hypertrophy-accessory zone."""
    return ACCESSORY_REP_RANGES_BY_BLOCK.get(block_type, {}).get(pattern, (8, 12))


def list_targets(lift: str) -> list[str]:
    """Available targets for a lift — useful for tool schemas / UI."""
    return list(EXPERT_ACCESSORIES.get(lift, {}).keys())
