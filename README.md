# Multi-Agent Content Team — Supervisor Architecture

A LangGraph multi-agent system where a supervisor coordinates four specialist
agents (researcher, writer, reviewer, coder) to complete content tasks. The
interesting piece is the writer ↔ reviewer feedback loop: the reviewer can
send a draft back for revision, and a hard cap prevents the loop from running
forever.

The project was built to explore the supervisor pattern specifically — how
control flow gets shared between an LLM-decided router and code-level
guardrails, and what goes wrong when you trust the LLM too much.

## How it works

The supervisor reads the current task state and picks which specialist runs
next. Each specialist does one job, writes its output to shared state, and
returns control to the supervisor. The supervisor decides whether to route
to another agent or finish.

The reviewer is the key node: if it rejects the writer's draft, the
supervisor routes back to the writer for a revision. This loop is capped by
`max_revisions` (default 3). After the cap, the supervisor accepts the
current draft and moves on, regardless of what the reviewer says.

## The team

| Agent      | Role                                | Tools         |
| ---------- | ----------------------------------- | ------------- |
| Supervisor | Routes work, decides what runs next | (LLM routing) |
| Researcher | Searches the web, compiles notes    | web_search    |
| Writer     | Drafts content from research        | —             |
| Reviewer   | Critiques, approves or rejects      | —             |
| Coder      | Writes and runs Python              | run_python    |

## Stack

- **LangGraph** — StateGraph with explicit nodes and conditional edges
- **OpenAI** `gpt-4o-mini` — every agent's LLM
- **Tavily** — web search (free tier)
- **MemorySaver** — checkpointing across turns

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY and TAVILY_API_KEY
```

## Usage

```bash
# Research + writing task
python main.py --task "Write a short report on the benefits of solar energy"

# Task that exercises the coder
python main.py --task "Calculate compound interest on €10,000 at 5% for 10 years and explain the result"

# Interactive
python main.py
```

## Example run

```
Task: Write a short report on the benefits of solar energy

[Supervisor] → researcher
[Researcher] Compiled research notes: solar reduces electricity bills...

[Supervisor] → writer
[Writer] Produced draft (1240 chars).

[Supervisor] → reviewer
[Reviewer] NEEDS REVISION — add specific cost figures and address
           environmental impact more directly.

[Supervisor] → writer
[Writer] Produced draft (1580 chars).

[Supervisor] → reviewer
[Reviewer] APPROVED.

[Supervisor] Task complete.
```

## Graph

```
START → supervisor → route_to_agent
                       ├── researcher → supervisor
                       ├── writer     → supervisor
                       ├── reviewer   → supervisor
                       ├── coder      → supervisor
                       └── finish     → END
```

## Guardrails

These are the parts that matter once you stop trusting the LLM:

- `max_steps` — hard cap on total agent actions across the whole run.
- `max_revisions` — caps the writer ↔ reviewer loop specifically.
- Code-enforced routing — even when the LLM picks the wrong next node, the
  guardrails override it.
- Explicit `APPROVED` check — the reviewer's verdict ends the loop
  cleanly rather than relying on the supervisor to notice.

Without these, the supervisor LLM will occasionally route in circles
(reviewer → writer → reviewer with no real progress) or skip the reviewer
entirely. The guardrails turn a "usually works" agent into one that
terminates predictably.

## Layout

```
.
├── main.py              # entry point
├── src/                 # agent nodes, state, graph assembly
├── .env.example
└── requirements.txt
```

## Notes

This is a learning project for the supervisor pattern. It's not deployed
anywhere — the value is in the graph design and the guardrail logic in `src/`.
