# Build the React app, then serve it and both agents' APIs from one Python process.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY qbank/ qbank/
COPY mentor/ mentor/
# Mentor Desk's cohort is built from a fixed seed (same data every build), so the 12 MB database isn't committed.
RUN python -m mentor.generate
COPY --from=web /web/dist frontend/dist
# Railway (and most hosts) set PORT; this is the fallback for a plain `docker run`.
ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
