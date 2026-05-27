"""
Tools used by the specialist agents.
  - web_search : researcher uses this
  - run_python : coder uses this (sandboxed exec)
"""

from langchain_core.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os, io, contextlib

load_dotenv()


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Returns titles, URLs, and summaries."""
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query=query, max_results=5)
        out = []
        for r in results.get("results", []):
            out.append(f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r['content'][:400]}")
        return "\n---\n".join(out) if out else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def run_python(code: str) -> str:
    """
    Execute Python code and return its output.
    Use this for calculations, data processing, or verifying logic.
    Only standard library and common packages are available.
    """
    try:
        buf = io.StringIO()
        # Restricted globals — basic safety
        safe_globals = {"__builtins__": __builtins__}
        with contextlib.redirect_stdout(buf):
            exec(code, safe_globals)
        output = buf.getvalue()
        return output if output else "Code ran successfully (no output)."
    except Exception as e:
        return f"Code error: {type(e).__name__}: {str(e)}"


RESEARCHER_TOOLS = [web_search]
CODER_TOOLS = [run_python]
