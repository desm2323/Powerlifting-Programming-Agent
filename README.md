# Powerlifting Programming Agent

An intelligent software agent that designs and **auto-regulates** a powerlifting
program. You give it your training maxes and a goal; it prescribes work, reads
your session feedback (including messy free text like *"grinded the last rep,
felt heavy out of the hole"*), tracks accumulated fatigue, and adjusts — progress,
hold, or deload — within hard safety limits.

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
grounded computation, explicit domain constraints, and a feedback loop. The LLM
is used only for what it's good at — interpreting fuzzy feedback, diagnosing
weaknesses, and explaining decisions. **It never picks a weight.**

## Architecture

```
                 You (goal + session feedback)
                              |
   +--------------------------v---------------------------+
   |  AGENT                                               |
   |   LLM reasoning & interpretation                     |
   |     (interpret feedback · diagnose · explain)        |
   |        |          |          |            |          |
   |     Memory    Knowledge     Tools      Guardrails    |
   |   (fatigue,  (periodiz.,  (loading    (caps, deload, |
   |    PRs)       landmarks)    math)       scope limit)  |
   +--------------------------+---------------------------+
                              |
                     Plan + adjustment
                              |
                   (you run it, log results) ---> loop back
```

| Box | Module | Course concept |
|-----|--------|----------------|
| Reasoning | `coach/reasoning.py` | LLM agent / ReAct-style reason→act |
| Memory | `coach/memory.py`, `coach/readiness.py` | short- and long-term memory; state |
| Knowledge | `coach/periodization.py`, `coach/landmarks.py` | knowledge base |
| Tools | `coach/loading.py` | tool use / code execution |
| Guardrails | `coach/guardrails.py` | policy adherence / safety |
| Loop | `coach/loop.py` | PAGE perceive→reason→act cycle |

Fatigue management is treated as state: `state = (strength, readiness)`,
`action = progress | hold | deload`. The v1 policy in `readiness.py` is
hand-written rules; see the roadmap for the learned-policy upgrade.

## Setup

```bash
git clone <your-repo-url>
cd powerlifting-agent
python3 -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
```

The core runs on the **standard library alone** — no API key needed. The LLM
reasoning layer is optional enrichment:

```bash
cp .env.example .env        # then add your key, or skip entirely
export ANTHROPIC_API_KEY=...   # enables LLM feedback interpretation/explanations
```

Without a key, the agent runs in deterministic offline mode (heuristic feedback
parsing + templated explanations), so demos are fully reproducible.

## Usage

```bash
# 1. Set maxes + goal; prints your first session
python main.py init --name Sam --squat 150 --bench 100 --deadlift 190 \
               --goal-lift squat --goal-target 10 --goal-weeks 10

# 2. See the next prescribed session for the current block
python main.py plan

# 3. Log a set — this triggers the perceive -> reason -> act loop
python main.py log --lift squat --weight 128 --reps 3 --rpe 9.5 \
               --notes "grinded the last rep, missed my 4th, felt heavy out of the hole"

# 4. Check state, or surface recurring weaknesses
python main.py status
python main.py diagnose
```

State persists in `data/state.json` between runs.

## Tests

```bash
python -m pytest tests/ -q
```

The engine is covered by exact-answer ("verifiable") unit tests — the cheap,
reliable kind of evaluation to build first.

## Roadmap (where this goes next)

- **Learned policy** — replace the rule-based `decide_progression()` with a
  contextual bandit, then a full RL policy, to tune progression to the individual
  lifter (the headline expansion; nothing else needs to change).
- **Per-session prescriptions** — track the exact issued prescription so "missed
  reps" compares against what was actually programmed, not the phase template.
- **Velocity input** — accept bar-speed data as a second readiness signal.
- **Web/app UI** and a tracker integration for daily use.

## Scope & safety

This is a training-load tool, not medical or nutrition advice. If session notes
mention pain, injury, or diet, the guardrail layer halts and defers to a coach,
physio, or dietitian. See `coach/guardrails.py`.
