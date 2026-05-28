"""
chat.py — natural-language REPL on top of the agent.

This is the "talkable" front end. Each user turn is classified into an intent
(log / plan / status / diagnose / ask / exit) and dispatched either to the
existing Agent methods or to coach_react (the ReAct LLM loop).

Intent parsing is deterministic-first: structured commands (`log squat 130x3`,
"what's next") are matched by regex so the agent works without an API key.
Anything ambiguous falls through to coach_react, which is where the LLM
actually earns its keep.
"""

from __future__ import annotations

import re
from typing import Optional

from . import reasoning
from .loop import Agent


# Map a root token (or any word starting with it) to the canonical lift name.
# Order matters for `deadlift` vs `dead` — check the longer root first.
_LIFT_ROOTS = [
    ("deadlift", "deadlift"), ("dead", "deadlift"), ("dl", "deadlift"),
    ("squat", "squat"), ("sq", "squat"),
    ("bench", "bench"), ("bp", "bench"), ("press", "bench"),
]


def _extract_lift(text: str) -> Optional[str]:
    """Match against word stems so 'squatted' / 'benched' / 'deadlifts' all hit."""
    for tok in re.findall(r"[a-z]+", text.lower()):
        for root, canonical in _LIFT_ROOTS:
            if tok.startswith(root):
                return canonical
    return None


def _extract_log(text: str) -> Optional[dict]:
    """Pull weight/reps/RPE from a free-text session log, e.g.
    "just squatted 130x3 at RPE 9, felt heavy" -> structured dict."""
    lift = _extract_lift(text)
    if not lift:
        return None
    # Weight x reps: 130x3, 130 x 3, 130 for 3, 130@3
    wxr = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|×|@|for)\s*(\d+)", text, re.I)
    if not wxr:
        return None
    weight = float(wxr.group(1))
    reps = int(wxr.group(2))
    rpe_match = re.search(r"rpe\s*(\d{1,2}(?:\.\d)?)", text, re.I)
    rpe = float(rpe_match.group(1)) if rpe_match else None
    return {"lift": lift, "weight": weight, "reps": reps, "rpe": rpe,
            "notes": text.strip()}


# PLAN intent = "show me what I'm doing today". DELIBERATELY does NOT match
# bare "plan" (too greedy: "plan my next block", "plan my season", "plan
# the strength block" should reach the LLM, not the current-block renderer).
PLAN_PHRASES = ("what should i do today", "what should i do now",
                "next session", "next workout",
                "what's next", "whats next",
                "today's workout", "todays workout",
                "today's session", "todays session")
# Exact-match shortcuts — typing literally "plan" still renders today's plan.
PLAN_EXACT = ("plan", "today's plan", "todays plan", "what now")

# Words that, when present alongside "plan", signal the user wants the LLM
# (multi-block / future / season planning), NOT the current-block renderer.
PLAN_LLM_HINTS = ("block", "blocks", "season", "macrocycle", "future", "next",
                  "comp", "competition", "meet", "month", "months", "weeks")

STATUS_PHRASES = ("status", "where am i", "training max", "current state",
                  "how am i doing", "summary")

DIAGNOSE_PHRASES = ("diagnose", "weakness", "weaknesses", "pattern",
                    "recurring", "trends")

EXIT_PHRASES = ("exit", "quit", "bye", "goodbye", "q", ":q")


def parse_intent(text: str) -> dict:
    """Classify a user message into one of: log / plan / status / diagnose /
    ask / exit / noop. Returns {"intent": ..., ...slots}."""
    raw = text.strip()
    lower = raw.lower()

    if not raw:
        return {"intent": "noop"}
    if lower in EXIT_PHRASES:
        return {"intent": "exit"}
    if any(p in lower for p in STATUS_PHRASES):
        return {"intent": "status"}
    if any(p in lower for p in DIAGNOSE_PHRASES):
        return {"intent": "diagnose"}

    log = _extract_log(raw)
    if log:
        return {"intent": "log", **log}

    # PLAN intent: only fires for unambiguous "show today's session" phrasings.
    # Multi-block / season / future queries — even when they contain "plan" —
    # go to the LLM so it can call get_season_plan, propose_season, etc.
    if any(p in lower for p in PLAN_PHRASES):
        return {"intent": "plan", "lift": _extract_lift(lower)}
    if lower in PLAN_EXACT:
        return {"intent": "plan", "lift": None}
    if (lower.startswith("plan ") and
            not any(h in lower for h in PLAN_LLM_HINTS)):
        # Bare "plan today" / "plan for this week" → render current.
        return {"intent": "plan", "lift": _extract_lift(lower)}

    return {"intent": "ask", "query": raw}


class Chat:
    """Natural-language REPL over the Agent."""

    BANNER = (
        "Powerlifting agent — chat mode.  Type 'exit' to quit.\n"
        "Try: \"what should I do today?\"  or  \"just squatted 130x3 "
        "RPE 9, felt heavy\"\n"
    )

    def __init__(self, state_path: str):
        self.agent = Agent(state_path)

    def handle(self, text: str, on_step=None) -> bool:
        """Dispatch a single message. Returns False if the loop should stop.
        Stores the most recent ReAct trace on self.last_trace for the UI.
        `on_step(message)` is called by coach_react after each LLM call so the
        UI can show progress instead of a silent spinner."""
        self.last_trace: list = []
        intent = parse_intent(text)
        kind = intent["intent"]

        if kind == "noop":
            return True
        if kind == "exit":
            print("ok, see you next session.")
            return False

        # If there's no active block, route EVERYTHING (even what looks like
        # a log) to coach_react so the LLM can drive onboarding and propose
        # the first block. The engine commands don't make sense before then.
        active_block = self.agent.state["block"].get("type")
        if active_block is None and kind not in ("status",):
            out = reasoning.coach_react(text, self.agent.state,
                                        agent=self.agent, on_step=on_step)
            self.last_trace = out.get("steps", [])
            print(f"coach> {self._with_commit_check(out)}")
            return True

        if kind == "status":
            self.agent.status()
            return True
        if kind == "diagnose":
            findings = reasoning.diagnose(self.agent.state["sessions"],
                                          self.agent.state)
            if not findings:
                print("No recurring weaknesses detected yet.")
            for f in findings:
                print(f"- {f['lift']} ({f['occurrences']}x {f['weakness']}): "
                      f"{f['suggestion']}")
            return True
        if kind == "plan":
            self.agent.plan_next(intent.get("lift"))
            return True
        if kind == "log":
            self.agent.log_session(
                intent["lift"], intent["weight"], intent["reps"],
                intent.get("rpe"), intent.get("notes", ""),
            )
            return True
        if kind == "ask":
            out = reasoning.coach_react(intent["query"], self.agent.state,
                                        agent=self.agent, on_step=on_step)
            self.last_trace = out.get("steps", [])
            print(f"coach> {self._with_commit_check(out)}")
            return True
        return True

    # --- post-turn integrity checks ----------------------------------------

    _COMMIT_VERBS = ("committed", "i've committed", "i have committed",
                     "i've set up", "i've created", "i've built",
                     "i've put together", "block starts", "starts monday",
                     "starts tuesday", "starts wednesday", "starts thursday",
                     "starts friday", "starts saturday", "starts sunday")

    # Phrases the LLM uses when claiming to have laid out a macrocycle.
    # If any appear in the reply but propose_season was NOT called this turn
    # AND no season_plan already exists, surface a warning so the user
    # isn't misled by an empty Season Plan tab.
    _SEASON_VERBS = ("macrocycle", "future blocks are planned",
                     "future blocks", "leading up to your meet",
                     "leading up to your competition",
                     "laid out the macrocycle", "laid out a macrocycle",
                     "i've laid out", "i have laid out",
                     "planned a macrocycle", "i've planned",
                     "the season plan", "the upcoming blocks",
                     "the next blocks", "after this block we'll",
                     "after this you'll", "after that you'll",
                     "next block will be", "next we'll",
                     "transition into", "ending in peaking",
                     "culminating in your competition")

    def _with_commit_check(self, out: dict) -> str:
        """Catch the case where the LLM claims to have committed a block
        but propose_block was never actually called (or failed). Appends
        a visible note so the user isn't misled."""
        text = out.get("text", "")
        steps = out.get("steps", [])
        lower = text.lower()
        claims_commit = any(v in lower for v in self._COMMIT_VERBS)
        if not claims_commit:
            return self._with_season_check(out)
        propose_calls = [s for s in steps if s[0] == "propose_block"]
        if not propose_calls:
            return (text + "\n\n⚠️ The agent said it committed a block but "
                    "didn't actually call propose_block this turn. Re-ask "
                    "with: 'Commit that as a block now.'")
        errors = [s for s in propose_calls
                  if isinstance(s[2], dict) and "error" in s[2]]
        if errors and self.agent.state["block"].get("type") is None:
            err = errors[-1][2]["error"]
            return (text + f"\n\n⚠️ propose_block was rejected — block NOT "
                    f"committed. Validator said: {err[:240]}")
        return self._with_season_check(out, base_text=text)

    def _with_season_check(self, out: dict, base_text: str | None = None) -> str:
        """Catch the case where the LLM mentioned planning future blocks /
        macrocycle / 'leading up to your meet' in text but did NOT actually
        call propose_season — and no season_plan exists yet. The lifter
        opens the Season Plan tab and sees nothing under Upcoming despite
        what the chat said."""
        text = base_text if base_text is not None else out.get("text", "")
        steps = out.get("steps", [])
        lower = text.lower()
        claims_season = any(v in lower for v in self._SEASON_VERBS)
        if not claims_season:
            return text
        season_calls = [s for s in steps
                        if s[0] in ("propose_season", "update_season_plan")]
        # If propose_season was called this turn, no warning needed.
        if season_calls:
            return text
        # Or if a season_plan already exists from an earlier turn, also OK.
        sp = self.agent.state.get("season_plan", {}) or {}
        if sp.get("blocks"):
            return text
        return (text + "\n\n⚠️ The agent mentioned future blocks / a "
                "macrocycle but didn't actually call propose_season this "
                "turn — the **Season Plan** tab will show only your "
                "current block (no Upcoming). Re-ask with: *'lay out the "
                "full macrocycle from now until my competition'*.")

    def repl(self) -> None:
        print(self.BANNER)
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            try:
                cont = self.handle(line)
            except Exception as e:
                print(f"(chat error: {type(e).__name__}: {e})")
                cont = True
            print()
            if not cont:
                return
