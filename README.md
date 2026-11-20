# mrw agents

Two agents for a medical exam-prep team, on one site. The switch at the top moves between them:

- **[QBank Auditor](qbank/README.md)** checks MCQs before students see them: questions you paste, or a demo bank of 13,758 real pharmacology questions. Every problem it finds goes to a review queue for faculty to confirm.
- **[Mentor Desk](mentor/README.md)** answers mentors' questions about their students: app use, lectures, tests, weak subjects, batch rank and test integrity. The cohort is synthetic.

Both run on LangChain agents with deepseek on Ollama Cloud. They answer from tool results, never from guesses. This is an independent prototype on public and synthetic data.

## Layout

| Path | What it is |
|---|---|
| `app.py` | One FastAPI server: QBank Auditor's API at `/qbank/api`, Mentor Desk's at `/mentor/api`, and the React app at `/` |
| `qbank/` | QBank Auditor: checks, bank, agent, evals and tests |
| `mentor/` | Mentor Desk: data generator, tools, agent, evals and tests |
| `frontend/` | One React app. `src/Shell.tsx` switches between `src/mentor/` and `src/qbank/`. The site opens on Mentor Desk; `#qbank` in the URL opens QBank Auditor |

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
echo OLLAMA_API_KEY=your-key > .env
python -m mentor.generate         # builds mentor/data/mentor.db (2 seconds)

cd frontend && npm install && npm run build && cd ..
uvicorn app:app --port 8000       # open http://127.0.0.1:8000
```

For UI work with hot reload, run `uvicorn app:app --reload` and, in `frontend/`, `npm run dev`. It proxies `/qbank` and `/mentor` to port 8000.

**Tests:**
- Backend: `pip install -r requirements-dev.txt`, then `pytest`. Set `MEDMCQA_PARQUET` to the MedMCQA train parquet to include the data-prep tests.
- Frontend: in `frontend/`, run `node --experimental-strip-types --test src/qbank/lib.test.ts src/mentor/lib.test.ts`.

**Evals** run the agents live against the Ollama key: `python -m qbank.evals` and `python -m mentor.evals`. Each package's README lists the suites and past results.

## Deploy on Railway

1. In Railway, create a project from this GitHub repo. Railway finds the `Dockerfile` and builds it: the React app, then Mentor Desk's database from its fixed seed, so the 12 MB file isn't committed.
2. In the service's **Variables**, add `OLLAMA_API_KEY` with your Ollama Cloud key. It stays in Railway, never in the repo.
3. Under **Settings → Networking**, generate a public domain. The app listens on the `PORT` Railway sets.

Every push to `main` redeploys. Every chat uses your Ollama quota. Chat memory lives in the server process, so a redeploy starts it fresh.
