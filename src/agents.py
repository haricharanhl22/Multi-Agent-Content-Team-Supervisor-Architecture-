"""
The specialist agents. Each is a node in the graph with one clear job.

  researcher : searches the web, produces research_notes
  writer     : turns research_notes into a draft
  reviewer   : critiques the draft, sets approved=True/False + feedback
  coder      : writes and runs code when needed, produces code_output

Each agent only sees what it needs from the shared state, does its job,
and writes its contribution back. They never call each other — the
supervisor coordinates everything.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

from src.state import TeamState
from src.tools import web_search, run_python

load_dotenv()


def _llm(temperature=0):
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


# ── Researcher ───────────────────────────────────────────────────────────────
def researcher_node(state: TeamState) -> TeamState:
    """Searches the web and compiles research notes on the task."""
    llm = _llm().bind_tools([web_search])

    prompt = f"""You are a research specialist. Your job is to gather accurate,
relevant information on the following task by searching the web.

Task: {state['task']}

Search for the key facts needed. After searching, write a concise set of
research notes (bullet points) capturing the most important findings.
Always base notes on search results, never on assumptions."""

    messages = [SystemMessage(content=prompt)]
    response = llm.invoke(messages)

    # If the LLM wants to search, run the search and feed results back
    notes = ""
    if hasattr(response, "tool_calls") and response.tool_calls:
        search_results = []
        for tc in response.tool_calls:
            if tc["name"] == "web_search":
                result = web_search.invoke(tc["args"])
                search_results.append(result)

        # Now ask the LLM to summarize the search into notes
        summary_prompt = f"""Based on these search results, write concise research
notes (bullet points) for the task: {state['task']}

Search results:
{chr(10).join(search_results)}

Research notes:"""
        notes = _llm().invoke([HumanMessage(content=summary_prompt)]).content
    else:
        notes = response.content

    return {
        "research_notes": notes,
        "messages": [AIMessage(content=f"[Researcher] Compiled research notes:\n{notes[:500]}...")],
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


# ── Writer ───────────────────────────────────────────────────────────────────
def writer_node(state: TeamState) -> TeamState:
    """Turns research notes into a polished draft. Incorporates review feedback if any."""
    feedback_section = ""
    if state.get("review_feedback"):
        feedback_section = f"""
A reviewer gave this feedback on your previous draft. Address it in your revision:
{state['review_feedback']}

Your previous draft was:
{state.get('draft', '')}
"""

    prompt = f"""You are a writing specialist. Write a clear, well-structured piece
on the task using the research notes provided.

Task: {state['task']}

Research notes:
{state.get('research_notes', 'No research notes available.')}
{feedback_section}

Write a polished, well-organized draft. Use clear sections and accurate information."""

    draft = _llm(temperature=0.4).invoke([SystemMessage(content=prompt)]).content

    return {
        "draft": draft,
        "messages": [AIMessage(content=f"[Writer] Produced draft ({len(draft)} chars).")],
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


# ── Reviewer ─────────────────────────────────────────────────────────────────
def reviewer_node(state: TeamState) -> TeamState:
    """Critiques the draft. Sets approved=True if good, else gives feedback."""
    prompt = f"""You are a critical reviewer. Evaluate this draft against the task.

Task: {state['task']}

Draft:
{state.get('draft', '')}

Assess: Is it accurate, complete, well-structured, and does it fully address the task?

Respond in EXACTLY this format:
VERDICT: APPROVED or NEEDS_REVISION
FEEDBACK: <specific, actionable feedback — what to fix, or why it's approved>"""

    review = _llm().invoke([SystemMessage(content=prompt)]).content

    approved = "APPROVED" in review.split("FEEDBACK")[0].upper()
    feedback = review.split("FEEDBACK:")[-1].strip() if "FEEDBACK:" in review else review

    return {
        "approved": approved,
        "review_feedback": feedback,
        "revision_count": state.get("revision_count", 0) + (0 if approved else 1),
        "messages": [AIMessage(content=f"[Reviewer] Verdict: {'APPROVED' if approved else 'NEEDS REVISION'}\n{feedback[:300]}")],
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


# ── Coder ────────────────────────────────────────────────────────────────────
def coder_node(state: TeamState) -> TeamState:
    """Writes and runs Python code when the task needs computation."""
    llm = _llm().bind_tools([run_python])

    prompt = f"""You are a coding specialist. Write and run Python code to help
with the task. Use the run_python tool to execute and verify your code.

Task: {state['task']}

Research notes (if relevant):
{state.get('research_notes', '')}

Write code to solve the computational part of this task."""

    response = llm.invoke([SystemMessage(content=prompt)])

    output = ""
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "run_python":
                result = run_python.invoke(tc["args"])
                output += f"Code:\n{tc['args']['code']}\n\nOutput:\n{result}\n"
    else:
        output = response.content

    return {
        "code_output": output,
        "messages": [AIMessage(content=f"[Coder] {output[:400]}")],
        "steps_taken": state.get("steps_taken", 0) + 1,
    }
