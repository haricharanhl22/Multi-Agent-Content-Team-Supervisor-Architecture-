"""
main.py — Run the multi-agent content team.

Usage:
  python main.py --task "Write a short report on the benefits of solar energy"
  python main.py --task "Calculate compound interest on 10000 at 5% for 10 years and explain it" 
  python main.py    # interactive mode

The team:
  supervisor → researcher → writer → reviewer → (revise if needed) → done
"""

import argparse
import sys
import os
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()


def check_env():
    missing = [k for k in ["OPENAI_API_KEY", "TAVILY_API_KEY"] if not os.getenv(k)]
    if missing:
        print(f"\nERROR: Missing API keys: {', '.join(missing)}")
        print("Add them to .env: OPENAI_API_KEY and TAVILY_API_KEY")
        sys.exit(1)


def run_team(task: str, max_steps: int = 12, max_revisions: int = 2):
    """Run the multi-agent team on a task, streaming each agent's work."""
    from src.graph import build_team_graph

    graph = build_team_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "next_agent": "",
        "research_notes": "",
        "draft": "",
        "review_feedback": "",
        "code_output": "",
        "revision_count": 0,
        "max_revisions": max_revisions,
        "approved": False,
        "steps_taken": 0,
        "max_steps": max_steps,
    }

    print(f"\nTeam task: {task}")
    print(f"Max steps: {max_steps} | Max revisions: {max_revisions}\n")
    print("=" * 60)

    final_state = None
    last_agent = None

    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        # Show supervisor routing decisions
        next_agent = event.get("next_agent", "")
        if next_agent and next_agent != last_agent:
            if next_agent == "finish":
                print(f"\n[Supervisor] Task complete — finishing.")
            else:
                print(f"\n[Supervisor] Routing to: {next_agent}")
            last_agent = next_agent

        # Show the latest agent message
        messages = event.get("messages", [])
        if messages:
            last = messages[-1]
            content = str(getattr(last, "content", ""))
            if content.startswith("["):  # Agent reports start with [AgentName]
                print(f"{content[:400]}")

        final_state = event

    print("\n" + "=" * 60)

    # Show the final deliverable
    if final_state:
        print("\n" + "=" * 60)
        print("FINAL DELIVERABLE")
        print("=" * 60)

        if final_state.get("draft"):
            print("\nDRAFT:\n")
            print(final_state["draft"])

        if final_state.get("code_output"):
            print("\nCODE OUTPUT:\n")
            print(final_state["code_output"])

        approved = final_state.get("approved", False)
        revisions = final_state.get("revision_count", 0)
        print(f"\n[Status: {'Approved by reviewer' if approved else 'Max revisions reached'} after {revisions} revision(s)]")


def interactive_mode(max_steps: int = 12, max_revisions: int = 2):
    print("\nMulti-Agent Content Team — Interactive Mode")
    print("Give the team a task (research + write + review), or 'quit'.\n")

    while True:
        try:
            task = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if task.lower() in ("quit", "exit", "q"):
            break
        if not task:
            continue
        run_team(task, max_steps=max_steps, max_revisions=max_revisions)


if __name__ == "__main__":
    check_env()

    parser = argparse.ArgumentParser(description="Multi-Agent Content Team")
    parser.add_argument("--task", type=str, help="Task for the team")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-revisions", type=int, default=2)
    args = parser.parse_args()

    if args.task:
        run_team(args.task, max_steps=args.max_steps, max_revisions=args.max_revisions)
    else:
        interactive_mode(max_steps=args.max_steps, max_revisions=args.max_revisions)
