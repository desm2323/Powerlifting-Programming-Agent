"""
main.py — command-line interface for the powerlifting agent.

Commands:
  init    set training maxes + a goal, and print the first session
  plan    print the next prescribed session(s) for the current block
  log     log a performed set; triggers the perceive->reason->act loop
  status  show maxes, block phase, readiness, PRs
  diagnose  surface recurring weaknesses from the log

Examples:
  python main.py init --name Sam --squat 150 --bench 100 --deadlift 190 \\
                 --goal-lift squat --goal-target 10 --goal-weeks 10
  python main.py plan
  python main.py log --lift squat --weight 128 --reps 3 --rpe 9 \\
                 --notes "grinded the last rep, felt heavy"
  python main.py status
"""

from __future__ import annotations

import argparse

from coach import Agent


def main():
    p = argparse.ArgumentParser(description="Intelligent powerlifting programming agent")
    p.add_argument("--state", default="data/state.json", help="path to the state file")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="set maxes and goal")
    pi.add_argument("--name", required=True)
    pi.add_argument("--squat", type=float, required=True)
    pi.add_argument("--bench", type=float, required=True)
    pi.add_argument("--deadlift", type=float, required=True)
    pi.add_argument("--goal-lift", default="squat")
    pi.add_argument("--goal-target", type=float, default=10)
    pi.add_argument("--goal-weeks", type=int, default=10)

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

    args = p.parse_args()
    agent = Agent(args.state)

    if args.cmd == "init":
        maxes = {"squat": args.squat, "bench": args.bench, "deadlift": args.deadlift}
        goal = {"lift": args.goal_lift, "target": args.goal_target,
                "weeks": args.goal_weeks}
        agent.init_program(args.name, maxes, goal)
    elif args.cmd == "plan":
        agent.plan_next(args.lift)
    elif args.cmd == "log":
        agent.log_session(args.lift, args.weight, args.reps, args.rpe, args.notes)
    elif args.cmd == "status":
        agent.status()
    elif args.cmd == "diagnose":
        from coach import reasoning, memory
        findings = reasoning.diagnose(memory.load_state(args.state)["sessions"])
        if not findings:
            print("No recurring weaknesses detected yet.")
        for f in findings:
            print(f"- {f['occurrences']}x {f['weakness']}: {f['suggestion']}")


if __name__ == "__main__":
    main()
