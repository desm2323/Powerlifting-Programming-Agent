# Powerlifting Programming Agent

An intelligent software agent that designs and **auto-regulates** a powerlifting
program. You give it your training maxes and chat with it about what you're
training for; it diagnoses, picks the right block (volume / technique / strength
/ peaking), commits a full weekly plan, reads your session feedback (including
messy free text like *"grinded the last rep, felt heavy out of the hole"*),
tracks accumulated fatigue, and adjusts — progress, hold, or deload — within
hard safety limits.

> **2-minute demo video:** [ADD_YOUR_VIDEO_LINK_HERE]

---

## Why this exists

General-purpose chat models (GPT, Gemma, etc.) are unreliable powerlifting
programmers: they pick weights that are too heavy or too light, and they can't
manage fatigue. That isn't a prompt problem — it's structural. A chat model is:

- **stateless** — it never sees how training actually went, so it can't manage
  fatigue, which *is* an accumulated-history variable;
- **ungrounded** — it guesses loads from text patterns instead of computing them
  from your numbers;
- **unconstrained** — it has no concept of volume ceilings or sane progression.

This agent fixes each of those by *adding* the missing pieces: persistent state,
grounded computation, explicit domain constraints, a feedback loop, and a
curated knowledge base built from three real coaches' content (Ben L. Johnson,
Sebastian Oreb, and Alexander Bromley's YouTube videos, with timestamped
transcripts cached under `docs/`). The LLM is
used only for what it's good at — interpreting fuzzy feedback, designing the
week structure, diagnosing weaknesses, and explaining decisions. **It never
picks a weight.**

## Architecture

```
                 You (goal + session feedback, via CLI or browser chat)
                                       |
   +-----------------------------------v-----------------------------------+
   |  AGENT                                                                |
   |   LLM reasoning & interpretation (ReAct loop, OpenAI tool-calling)    |
   |     (diagnose · propose block · interpret feedback · explain)         |
   |        |          |           |             |             |           |
   |     Memory    Knowledge      Tools      Guardrails       Loop        |
   |  (state.json (blocks,       (loading   (TM caps,      (PERCEIVE ->   |
   |   sessions,   landmarks,     math,      MRV ceiling,   REASON ->     |
   |   PRs, block  accessories,   plates,    weekly-plan    GUARD ->      |
   |   history,    expert KB      RPE        validator,     DECIDE ->     |
   |   chat log)   tri-source)    model)     scope filter)  ACT)          |
   +-----------------------------------+-----------------------------------+
                                       |
                              Plan + adjustment
                                       |
                     (you run it, log results) ---> loop back
```

| Box | Module | Course concept |
|-----|--------|----------------|
| Reasoning | `coach/reasoning.py` | LLM agent / ReAct reason → act with typed tools |
| Memory | `coach/memory.py`, `coach/readiness.py` | short- and long-term memory; state estimation |
| Knowledge | `coach/blocks.py`, `coach/landmarks.py`, `coach/accessories.py`, `coach/expert_knowledge.py` | knowledge base (block types, volume landmarks, accessory catalog, tri-source coaching KB) |
| Tools | `coach/loading.py` | tool use / code execution (Program-of-Thoughts) |
| Guardrails | `coach/guardrails.py` | policy adherence / safety |
| Loop | `coach/loop.py` | PAGE perceive → reason → act cycle |
| Interface | `coach/chat.py`, `coach/chat_web.py`, `main.py` | CLI + REPL + Streamlit UI |

Fatigue management is treated as state: `state = (strength, readiness)`,
`action = progress | hold | deload`. The v1 policy in `coach/readiness.py` is
hand-written rules; see the roadmap for the learned-policy upgrade.

## Setup

```bash
git clone <your-repo-url>
cd powerlifting-agent
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The core engine runs without an API key in **deterministic offline mode**
(heuristic feedback parsing + templated explanations) so demos and tests are
fully reproducible. The LLM reasoning layer is optional enrichment:

```bash
cp .env.example .env       # then add your OPENAI_API_KEY
# or just: export OPENAI_API_KEY=sk-...
```

Default model is `gpt-4o-mini` (cheap, fast, supports tool calls); override with
`LLM_MODEL` in `.env`.

## Usage

There are three ways to drive the agent: **CLI** (one-shot commands),
**terminal chat REPL**, and **browser UI** (Streamlit). All share the same
state file at `data/state.json`.

### CLI (one-shot)

```bash
# 1. Record your maxes + lifter context. Block is NOT auto-picked — chat
#    with the coach so they can diagnose what you actually need.
python main.py init --name Sam --squat 150 --bench 100 --deadlift 190 \
               --bodyweight 85 --experience intermediate

# 2. See the current week of the active block (day-by-day prescription)
python main.py plan

# 3. Log a set — this triggers the perceive -> reason -> act loop
python main.py log --lift squat --weight 128 --reps 3 --rpe 9.5 \
               --notes "grinded the last rep, missed my 4th, felt heavy out of the hole"

# 4. Check state, or surface recurring weaknesses
python main.py status
python main.py diagnose

# 5. Ask the coach a free-text question (one-shot, uses ReAct + tools)
python main.py ask "how should I structure squat day this week?"
python main.py ask --trace "is bench frequency too low for my lifter profile?"
```

### Chat (terminal REPL)

```bash
python main.py chat
# you> just squatted 130x3 RPE 9, felt heavy
# you> what should I do today?
# you> week 1 felt light, the RPE 7 squat felt more like 5
#      (the engine nudges your working 1RM up ~2.5 kg per RPE point
#       and advances the week — same block, future weeks adjust)
# you> exit
```

### Browser UI

```bash
python main.py web        # launches Streamlit on localhost
```

A sidebar for setup + state, a live program panel rendering the committed
weekly plan as day-cards with role-grouped exercises (primary / secondary /
accessory), and a chat box that streams the LLM's tool-call trace.

## Tests

```bash
python -m pytest tests/ -q
```

The engine is covered by exact-answer ("verifiable") unit tests — the cheap,
reliable kind of evaluation. The test suite runs in forced-offline mode
(`tests/conftest.py` strips `OPENAI_API_KEY` so the LLM is never hit).

## Knowledge base — tri-source (Ben L. Johnson + Sebastian Oreb + Alexander Bromley)

`coach/expert_knowledge.py` is a structured KB extracted from three
independent authoritative voices in powerlifting programming. Every
entry carries a source citation in the form `video_id @ MM:SS` so any
recommendation is traceable back to the original content.

- **Ben L. Johnson** (`docs/ben_transcripts/`, 10 videos): top-set +
  back-off methodology, RPE-driven auto-regulation, volume landmarks,
  block stacking, hypertrophy accessory checklist, weak-point strength
  variations, AVOID list of non-progressive exercises.
- **Sebastian Oreb** (`docs/sebastian_transcripts/`, 3 videos):
  8-category structural balance model, exercise-redundancy detection,
  neglected-muscle targets, training-splits catalog (2-6 day), 5-step
  periodization model, loadability rules, hypertrophy vs strength volume
  denominators (per muscle group vs per movement pattern).
- **Alexander Bromley** (`docs/bromley_transcripts/`, 2 videos):
  accessory progression methodology (rep-range method + DDP +
  block-level shifting), advantaged/disadvantaged movement leverage
  classification with phase mapping, pattern-vs-muscle weakness
  targeting framework, dormant-muscle high-rep wake-up, session
  precision gradient.

The three voices are **complementary**: Ben covers what Sebastian +
Bromley de-emphasize (RPE-as-prescription, block-stack RPE), Sebastian
covers selection + redundancy + splits, Bromley covers the missing
"how do I drive accessories forward week-to-week" methodology. The
system prompt teaches the LLM to cite all three.

To refresh or extend the KB:

```bash
# Ben videos:
python scripts/ingest_ben_videos.py docs/ben_videos.txt

# Sebastian / Bromley / any other author:
python scripts/ingest_ben_videos.py docs/sebastian_videos.txt \
    --out-dir docs/sebastian_transcripts
python scripts/ingest_ben_videos.py docs/bromley_videos.txt \
    --out-dir docs/bromley_transcripts
```

The ingest script writes timestamped transcripts to the configured
directory and is build-time only; the agent never calls YouTube at
runtime.

## Roadmap (where this goes next)

- **Learned policy** — replace the rule-based `decide_progression()` with a
  contextual bandit, then a full RL policy, to tune progression to the
  individual lifter (the headline expansion; nothing else needs to change).
- **Velocity input** — accept bar-speed data as a second readiness signal.
- **Wearable / tracker integration** — auto-import sets from a strength-tracker
  app instead of typed `log` commands.
- **Multi-user** — generalize the single `data/state.json` into per-lifter
  stores so a coach could run multiple programs.

## Scope & safety

This is a training-load tool, not medical or nutrition advice. If session notes
mention pain, injury, or diet, the guardrail layer halts and defers to a coach,
physio, or dietitian. The weekly-plan validator additionally rejects plans that
violate basic powerlifting fatigue rules (e.g. >2 deadlift sessions per week,
primary + secondary of the same lift on the same day, or exercises on the
AVOID list — planks, push-ups, burpees, etc., which don't progressively
overload). See `coach/guardrails.py`.
