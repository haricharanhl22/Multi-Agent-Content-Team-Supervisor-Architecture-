"""
Shared state for the multi-agent team.

This state flows through the supervisor and every specialist agent.
Each agent reads what it needs and writes its contribution back.
The shared scratchpad (research_notes, draft, review_feedback) is how
agents communicate without talking to each other directly — everything
goes through the supervisor and the shared state.
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages


class TeamState(TypedDict):
    # Conversation history
    messages: Annotated[list, add_messages]

    # The original task from the user
    task: str

    # Which agent the supervisor wants to run next
    # (set by supervisor, read by the routing edge)
    next_agent: str

    # Shared scratchpad — agents read and write these
    research_notes: str      # filled by researcher
    draft: str               # filled by writer
    review_feedback: str     # filled by reviewer
    code_output: str         # filled by coder

    # Revision tracking — prevents infinite writer↔reviewer loops
    revision_count: int
    max_revisions: int

    # Whether the reviewer approved the draft
    approved: bool

    # Step guard
    steps_taken: int
    max_steps: int
