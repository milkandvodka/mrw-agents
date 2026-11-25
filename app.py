"""mrw agents: the demos on one site.

QBank Auditor's API lives under /qbank/api, Mentor Desk's under /mentor/api, and / serves the React app that switches
between them. LeakTrace runs as its own service; the site shows it in a frame at the address in LEAKTRACE_URL.
Run locally:  uvicorn app:app --port 8000   (and `npm run dev` in frontend/ for hot reload)
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mentor.server import app as mentor
from qbank.server import app as qbank

app = FastAPI(title="mrw agents")
app.mount("/qbank", qbank)
app.mount("/mentor", mentor)


@app.get("/api/config")
def config() -> dict:
    """Addresses the site needs at run time (kept in the host's settings, not in the code)."""
    return {"leaktrace_url": os.getenv("LEAKTRACE_URL", "").strip().rstrip("/")}


DIST = Path(__file__).parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="site")
