# Aline AI Assistant MVP

This monorepo contains a FastAPI backend, React frontend, Celery worker, and watchdog-based watcher for document ingestion and proposal review.

## Services
- **backend**: FastAPI + SQLAlchemy + Alembic
- **worker**: Celery worker for extraction and Ollama routing/extraction
- **watcher**: filesystem watcher that enqueues Celery tasks
- **frontend**: React + Vite + Tailwind

## Quickstart

```bash
docker compose up --build
```

Then open:
- Frontend: http://localhost:5173
- Backend docs: http://localhost:8000/docs

## Login
Seeded admin user:
- **email**: `admin@aline.local`
- **password**: `Admin123!` (change after first login)

## Workflow
1. Upload a document from **Documents** page.
2. Celery extracts text, calls Ollama router/extractor, and creates proposals.
3. Review proposals on **Proposals** page; edit fields, approve, or reject.
4. Approved proposals write to core tables and create audit logs.

## File Storage
Documents are stored on disk at `${FILE_STORAGE_ROOT}` (default: `/data/aline_docs`).
Extracted text is stored alongside each document version as `.txt`.

## Environment
Key env vars (set in `docker-compose.yml`):
- `DATABASE_URL`
- `JWT_SECRET`
- `FILE_STORAGE_ROOT`
- `OLLAMA_URL`
- `OLLAMA_MODEL`
- `REDIS_URL`
- `WATCH_PATHS`

## Optional: ngrok
Start ngrok for LAN access if desired:

```bash
ngrok http 5173
```

Update frontend API base to point to the backend host if exposing externally.

## Tests
```bash
cd backend
pytest
```
