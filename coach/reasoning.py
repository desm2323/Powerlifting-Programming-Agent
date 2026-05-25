"""
reasoning.py — the LLM layer (the "LLM reasoning & interpretation" box).

The LLM's ONLY jobs are the things it's actually good at:
  - turn fuzzy free-text feedback ("felt heavy, grinded the last single") into
    structured numbers the engine can use,
  - diagnose recurring weaknesses from the log,
  - explain WHY an adjustment was made, in plain language.

It never picks weights — those come from coach/loading.py.

If no API key (or the `anthropic` package) is available, every function falls
back to a deterministic heuristic so the agent still runs offline. That keeps
demos reproducible and reinforces the architecture: the LLM is enrichment on
top of a self-sufficient engine.
"""

from __future__ import annotations

import json
import os
import re

MODEL = os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest")


def _client():
    """Return an Anthropic client if usable, else None (offline mode)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:
        return None


def llm_available() -> bool:
    return _client() is not None


def _ask_json(system: str, user: str) -> dict | None:
    client = _client()
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=512, system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return None


# --- interpret feedback -----------------------------------------------------

def interpret_feedback(notes: str, prescribed_reps: int, performed_reps: int,
                       reported_rpe: float | None, rpe_cap: float) -> dict:
    """
    Produce: {effective_rpe, missed_reps, sticking_point, flags:[...]}.
    Tries the LLM first, falls back to a heuristic parse of `notes`.
    """
    system = ("You convert a powerlifter's session note into strict JSON with keys "
              "effective_rpe (number 5-10), missed_reps (int), sticking_point "
              "(string or null), flags (list; include 'pain' if any pain/injury). "
              "Return ONLY JSON.")
    user = (f"Prescribed reps: {prescribed_reps}. Performed reps: {performed_reps}. "
            f"Reported RPE: {reported_rpe}. Note: {notes!r}")
    out = _ask_json(system, user)
    if out and "effective_rpe" in out:
        out.setdefault("missed_reps", max(0, prescribed_reps - performed_reps))
        out.setdefault("flags", [])
        out.setdefault("sticking_point", None)
        out["rpe_cap"] = rpe_cap
        return out

    # --- offline heuristic ---
    text = (notes or "").lower()
    missed = max(0, prescribed_reps - performed_reps)
    rpe = reported_rpe
    if rpe is None:
        m = re.search(r"rpe\s*([0-9](?:\.5)?)", text)
        rpe = float(m.group(1)) if m else (10.0 if missed else rpe_cap)
    flags = ["pain"] if any(t in text for t in
                            ("pain", "sharp", "tweak", "pop", "hurt", "strain")) else []
    sticking = None
    if "lockout" in text or "lock out" in text:
        sticking = "lockout"
    elif "off the chest" in text or "off chest" in text:
        sticking = "off_the_chest"
    elif "hole" in text or "bottom" in text:
        sticking = "out_of_the_hole"
    return {"effective_rpe": float(rpe), "missed_reps": missed,
            "sticking_point": sticking, "flags": flags, "rpe_cap": rpe_cap}


# --- diagnose weaknesses ----------------------------------------------------

STICKING_FIXES = {
    "lockout": "Add lockout work (close-grip bench / board press for bench, "
               "block pulls for deadlift, pause/pin squats up top).",
    "off_the_chest": "Add bottom-range strength: paused bench and spoto press.",
    "out_of_the_hole": "Add bottom-position work: paused and tempo squats.",
}


def diagnose(history: list[dict]) -> list[dict]:
    """
    Rule-based weakness detection over recent sessions: if the same sticking
    point shows up repeatedly, surface a fix. (An LLM can enrich this later.)
    """
    points: dict[str, int] = {}
    for s in history:
        sp = s.get("sticking_point")
        if sp:
            points[sp] = points.get(sp, 0) + 1
    findings = []
    for sp, count in points.items():
        if count >= 2 and sp in STICKING_FIXES:
            findings.append({"weakness": sp, "occurrences": count,
                             "suggestion": STICKING_FIXES[sp]})
    return findings


# --- explain ----------------------------------------------------------------

def explain(decision: str, ctx: dict) -> str:
    """Plain-language rationale. LLM makes it nicer; template is the fallback."""
    system = ("You are a concise powerlifting coach. In 1-2 sentences, explain the "
              "adjustment decision to the lifter. Be direct and encouraging.")
    user = f"Decision: {decision}. Context: {json.dumps(ctx)}"
    out = _ask_json(system, '{"explanation": "..."}') if False else None  # JSON not needed
    client = _client()
    if client is not None:
        try:
            msg = client.messages.create(model=MODEL, max_tokens=160, system=system,
                                         messages=[{"role": "user", "content": user}])
            txt = "".join(b.text for b in msg.content
                          if getattr(b, "type", "") == "text").strip()
            if txt:
                return txt
        except Exception:
            pass
    templates = {
        "progress": ("Clean session under the effort cap with readiness high, so "
                     "I'm bumping your training max for next cycle."),
        "hold": ("Readiness is okay but the session ran at or above the planned "
                 "effort, so we repeat the load to consolidate before pushing."),
        "deload": ("Fatigue has built up (readiness {r}); taking a planned deload "
                   "to recover so the next block lands on fresh legs.").format(
                       r=ctx.get("readiness", "?")),
    }
    return templates.get(decision, "Holding steady.")
