"""mrw agents: both demo agents on one site.

QBank Auditor's API lives under /qbank/api, Mentor Desk's under /mentor/api, and / serves the React app that switches between them.
Run locally:  uvicorn app:app --port 8000   (and `npm run dev` in frontend/ for hot reload)
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mentor.server import app as mentor
from qbank.server import app as qbank

app = FastAPI(title="mrw agents")
app.mount("/qbank", qbank)
app.mount("/mentor", mentor)

DIST = Path(__file__).parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="site")
