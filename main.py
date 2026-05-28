"""
main.py — command-line interface for the powerlifting agent.

Commands:
  init      set training maxes + a goal, and print the first session
  plan      print the next prescribed session(s) for the current block
  log       log a performed set; triggers the perceive->reason->act loop
  status    show maxes, block phase, readiness, PRs
  diagnose  surface recurring weaknesses from the log
  ask       one-shot natural-language coach question (ReAct, --trace shows tools)
  chat      interactive terminal chat
  web       browser chat UI via Streamlit

Examples:
  python main.py init --name Sam --squat 150 --bench 100 --deadlift 190 \\
                 --goal-lift squat --goal-target 10 --goal-weeks 10
  python main.py log --lift squat --weight 128 --reps 3 --rpe 9 \\
                 --notes "grinded the last rep, felt heavy"
  python main.py chat
  python main.py web
"""

from __future__ import annotations

import argparse

from coach import Agent


def main():
    p = argparse.ArgumentParser(description="Intelligent powerlifting programming agent")
    p.add_argument("--state", default="data/state.json", help="path to the state file")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="record maxes + lifter context. "
                                     "Goals & block are diagnosed via chat.")
    pi.add_argument("--name", required=True)
    pi.add_argument("--squat", type=float, required=True)
    pi.add_argument("--bench", type=float, required=True)
    pi.add_argument("--deadlift", type=float, required=True)
    pi.add_argument("--bodyweight", type=float, default=None,
                    help="bodyweight in kg")
    pi.add_argument("--experience",
                    choices=["novice", "intermediate", "advanced"],
                    default=None)

    pp = sub.add_parser("plan", help="print the next session")
    pp.add_argument("--lift", default=None)

    pl = sub.add_parser("log", help="log a performed set")
    pl.add_argument("--lift", required=True, choices=["squat", "bench", "deadlift"])
    pl.add_argument("--weight", type=float, required=True)
    pl.add_argument("--reps", type=int, required=True)
    pl.add_argument("--rpe", type=float, default=None)
    pl.add_argument("--notes", default="")

    sub.add_parser("status", help="show current state")
    sub.add_parser("diagnose", help="surface recurring weaknesses")
    sub.add_parser("season", help="show the macrocycle — past, current, "
                                  "and planned upcoming blocks")

    pa = sub.add_parser("ask", help="ask the coach a natural-language question")
    pa.add_argument("query", nargs="+", help="the question (free text)")
    pa.add_argument("--trace", action="store_true",
                    help="print the tool-call trace (ReAct steps)")

    sub.add_parser("chat", help="start a natural-language chat with the agent")
    sub.add_parser("web", help="launch the browser chat UI (Streamlit)")

    args = p.parse_args()
    agent = Agent(args.state)

    if args.cmd == "init":
        maxes = {"squat": args.squat, "bench": args.bench, "deadlift": args.deadlift}
        agent.init_program(args.name, maxes,
                           bodyweight_kg=args.bodyweight,
                           experience=args.experience)
    elif args.cmd == "plan":
        agent.plan_next(args.lift)
    elif args.cmd == "log":
        agent.log_session(args.lift, args.weight, args.reps, args.rpe, args.notes)
    elif args.cmd == "status":
        agent.status()
    elif args.cmd == "season":
        agent._print_season_plan(brief=False)
    elif args.cmd == "diagnose":
        from coach import reasoning, memory
        state = memory.load_state(args.state)
        findings = reasoning.diagnose(state["sessions"], state)
        if not findings:
            print("No recurring weaknesses detected yet.")
        for f in findings:
            print(f"- {f['lift']} ({f['occurrences']}x {f['weakness']}): "
                  f"{f['suggestion']}")
    elif args.cmd == "ask":
        from coach import reasoning
        query = " ".join(args.query)
        out = reasoning.coach_react(query, agent.state, agent=agent)
        if args.trace and out["steps"]:
            print("--- ReAct trace ---")
            for name, ins, res in out["steps"]:
                print(f"  -> {name}({ins}) = {res}")
            print("--- final answer ---")
        print(out["text"])
    elif args.cmd == "chat":
        from coach.chat import Chat
        Chat(args.state).repl()
    elif args.cmd == "web":
        import os
        import subprocess
        import sys as _sys
        here = os.path.dirname(os.path.abspath(__file__))
        app = os.path.join(here, "coach", "chat_web.py")
        subprocess.run([_sys.executable, "-m", "streamlit", "run", app],
                       cwd=here)


if __name__ == "__main__":
    main()
