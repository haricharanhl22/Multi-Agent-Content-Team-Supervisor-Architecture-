"""
The supervisor and the assembled multi-agent graph.

The supervisor is the orchestrator. It looks at the current state and decides
which specialist should act next — or whether the task is complete.

Graph flow:
  START → supervisor → [researcher | writer | reviewer | coder | END]
                ↑__________________________|
  (every specialist reports back to the supervisor)

The supervisor uses an LLM to make routing decisions, but with guardrails:
  - revision loops are capped (max_revisions)
  - total steps are capped (max_steps)
  - the reviewer's approval is respected
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import json

from src.state import TeamState
from src.agents import researcher_node, writer_node, reviewer_node, coder_node

load_dotenv()


SUPERVISOR_PROMPT = """You are a supervisor managing a team of specialist agents.
Your job is to decide which agent should work next, based on the current progress.

Your team:
- researcher: searches the web and gathers facts. Use FIRST for most tasks.
- coder: writes and runs Python code. Use only if the task needs computation.
- writer: turns research into a polished draft. Use AFTER research is done.
- reviewer: critiques the draft. Use AFTER a draft exists.
- FINISH: the task is complete, end the workflow.

Current progress:
- Research notes: {has_research}
- Code output: {has_code}
- Draft: {has_draft}
- Reviewer approved: {approved}
- Revisions so far: {revisions} / {max_revisions}

Task: {task}

Decision rules:
1. If no research notes yet → choose "researcher"
2. If task needs code and no code output yet → choose "coder"
3. If research done but no draft → choose "writer"
4. If draft exists but not reviewed → choose "reviewer"
5. If reviewer said NEEDS_REVISION and revisions < max → choose "writer"
6. If reviewer APPROVED, or max revisions hit → choose "FINISH"

Respond with ONLY ONE WORD: researcher, coder, writer, reviewer, or FINISH"""


def supervisor_node(state: TeamState) -> TeamState:
    """The orchestrator. Decides the next agent to run."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = SUPERVISOR_PROMPT.format(
        has_research="YES" if state.get("research_notes") else "NO",
        has_code="YES" if state.get("code_output") else "NO",
        has_draft="YES" if state.get("draft") else "NO",
        approved="YES" if state.get("approved") else "NO",
        revisions=state.get("revision_count", 0),
        max_revisions=state.get("max_revisions", 2),
        task=state["task"],
    )

    decision = llm.invoke([SystemMessage(content=prompt)]).content.strip().lower()

    # Parse decision — default to FINISH if unclear
    valid = ["researcher", "coder", "writer", "reviewer", "finish"]
    next_agent = "finish"
    for v in valid:
        if v in decision:
            next_agent = v
            break

    # Hard guardrails — override LLM if limits hit
    if state.get("steps_taken", 0) >= state.get("max_steps", 12):
        next_agent = "finish"
    if state.get("approved"):
        next_agent = "finish"
    if state.get("revision_count", 0) >= state.get("max_revisions", 2) and state.get("draft"):
        # Too many revisions — accept what we have
        if not state.get("approved"):
            next_agent = "finish"

    return {"next_agent": next_agent}


def route_to_agent(state: TeamState) -> str:
    """Reads the supervisor's decision and routes to that agent."""
    return state.get("next_agent", "finish")


def build_team_graph():
    """Assemble the multi-agent graph."""
    graph = StateGraph(TeamState)

    # Add all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("coder", coder_node)

    # Start at supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor routes to a specialist or finishes
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "writer":     "writer",
            "reviewer":   "reviewer",
            "coder":      "coder",
            "finish":     END,
        },
    )

    # Every specialist reports back to the supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")
    graph.add_edge("coder", "supervisor")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
