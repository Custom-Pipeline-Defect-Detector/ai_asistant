import os
import shutil
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.deps import get_db, require_roles
from app.models.models import Document, DocumentVersion
from app.schemas.schemas import DocumentOut
from app.services.extraction import sha256_file
from app.worker.tasks import process_document_version

router = APIRouter()


@router.post("/upload", response_model=DocumentOut)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _user=Depends(require_roles("Admin", "Manager", "PM", "Sales", "Engineer", "Technician", "QC")),
):
    os.makedirs(settings.file_storage_root, exist_ok=True)
    filename = file.filename
    storage_path = os.path.join(settings.file_storage_root, filename)
    with open(storage_path, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    file_hash = sha256_file(storage_path)

    document = Document(
        filename=filename,
        mime=file.content_type or "application/octet-stream",
        storage_path=storage_path,
        uploader_id=None,
        folder_path_hint=None,
        file_hash=file_hash,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    doc_version = DocumentVersion(doc_id=document.id, version=1)
    db.add(doc_version)
    db.commit()
    db.refresh(doc_version)

    process_document_version.delay(doc_version.id)
    return document


@router.get("/", response_model=list[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    _user=Depends(require_roles("Admin", "Manager", "PM", "Sales", "Engineer", "Technician", "QC", "Viewer")),
):
    return db.query(Document).order_by(Document.created_at.desc()).all()
