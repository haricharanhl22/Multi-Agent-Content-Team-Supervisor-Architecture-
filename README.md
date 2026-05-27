# Multi-Agent Content Team — Supervisor Architecture

A LangGraph multi-agent system where a supervisor orchestrates a team of
specialist agents (researcher, writer, reviewer, coder) to complete complex
tasks. Features a writer↔reviewer feedback loop with revision limits.

This is the frontier agentic AI pattern — the fastest-growing specialization
in the field.

---

## How it works

A supervisor agent reads the task and decides which specialist should work next.
Each specialist does one job, writes its result to shared state, and reports
back to the supervisor — which then decides the next step.

The key feature: if the reviewer rejects the writer's draft, the supervisor
routes it back to the writer for revision. This feedback loop is capped by
`max_revisions` to prevent infinite loops.

## The team

| Agent | Job | Tools |
|---|---|---|
| Supervisor | Routes work, decides next step | (LLM routing) |
| Researcher | Searches web, compiles notes | web_search |
| Writer | Turns research into a draft | — |
| Reviewer | Critiques draft, approves or rejects | — |
| Coder | Writes and runs Python | run_python |

## Stack

- LangGraph — multi-node agent graph
- OpenAI gpt-4o-mini — every agent's brain
- Tavily — web search (free tier)
- MemorySaver — state checkpointing

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY and TAVILY_API_KEY
mkdir src   # if not already there — move .py files into src/
```

## Usage

```bash
# A research + writing task
python main.py --task "Write a short report on the benefits of solar energy"

# A task that needs the coder
python main.py --task "Calculate compound interest on 10000 euros at 5% for 10 years and explain the result"

# Interactive
python main.py
```

## Example flow

```
Team task: Write a short report on the benefits of solar energy

[Supervisor] Routing to: researcher
[Researcher] Compiled research notes: - Solar reduces electricity bills...

[Supervisor] Routing to: writer
[Writer] Produced draft (1240 chars).

[Supervisor] Routing to: reviewer
[Reviewer] Verdict: NEEDS REVISION
Add specific cost figures and address environmental impact more directly.

[Supervisor] Routing to: writer
[Writer] Produced draft (1580 chars).

[Supervisor] Routing to: reviewer
[Reviewer] Verdict: APPROVED

[Supervisor] Task complete — finishing.
```

## Architecture

```
START → supervisor → route_to_agent()
                       ├── researcher → supervisor
                       ├── coder      → supervisor
                       ├── writer     → supervisor
                       ├── reviewer   → supervisor
                       └── finish     → END
```

## Guardrails (production essentials)

- `max_steps` — hard cap on total agent actions
- `max_revisions` — caps the writer↔reviewer loop
- Supervisor overrides — even if the LLM picks wrong, code enforces limits
- Approval respected — reviewer's APPROVED verdict ends the loop


deterministic guardrails in code that override it. Never trust the LLM alone
for control flow in production.
