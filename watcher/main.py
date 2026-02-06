import os
import time
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from celery import Celery

DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
FILE_STORAGE_ROOT = os.environ.get("FILE_STORAGE_ROOT", "/data/aline_docs")
WATCH_PATHS = os.environ.get("WATCH_PATHS", "/data/aline_docs")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
celery = Celery("aline", broker=REDIS_URL, backend=REDIS_URL)


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class Handler(FileSystemEventHandler):
    def on_created(self, event):
        self._handle(event)

    def on_modified(self, event):
        self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return
        path = event.src_path
        name = os.path.basename(path)
        if name.startswith("~") or name.startswith(".$") or name.endswith(".tmp"):
            return
        file_hash = sha256_file(path)
        session = SessionLocal()
        try:
            doc = session.execute(
                "SELECT id, file_hash FROM documents WHERE storage_path = :path",
                {"path": path},
            ).mappings().first()
            if doc and doc["file_hash"] == file_hash:
                return
            if not doc:
                session.execute(
                    "INSERT INTO documents (filename, mime, storage_path, uploader_id, folder_path_hint, file_hash, created_at) "
                    "VALUES (:filename, :mime, :storage_path, NULL, NULL, :file_hash, NOW())",
                    {
                        "filename": name,
                        "mime": "application/octet-stream",
                        "storage_path": path,
                        "file_hash": file_hash,
                    },
                )
                session.commit()
                doc = session.execute(
                    "SELECT id FROM documents WHERE storage_path = :path",
                    {"path": path},
                ).mappings().first()
            doc_id = doc["id"]
            version = session.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS next FROM document_versions WHERE doc_id = :doc_id",
                {"doc_id": doc_id},
            ).mappings().first()["next"]
            session.execute(
                "INSERT INTO document_versions (doc_id, version, created_at) VALUES (:doc_id, :version, NOW())",
                {"doc_id": doc_id, "version": version},
            )
            session.commit()
            doc_version = session.execute(
                "SELECT id FROM document_versions WHERE doc_id = :doc_id AND version = :version",
                {"doc_id": doc_id, "version": version},
            ).mappings().first()
            celery.send_task("app.worker.tasks.process_document_version", args=[doc_version["id"]])
        finally:
            session.close()


def main():
    observer = Observer()
    for path in WATCH_PATHS.split(","):
        if not path:
            continue
        os.makedirs(path, exist_ok=True)
        observer.schedule(Handler(), path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
