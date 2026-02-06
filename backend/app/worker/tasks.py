from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.models import Document, DocumentVersion, Proposal
from app.services.extraction import extract_text, save_extracted_text
from app.services.ollama import route_document, extract_document
from app.services.mapper import map_to_proposals
from app.worker.celery_app import celery


@celery.task
def process_document_version(doc_version_id: int) -> None:
    db: Session = SessionLocal()
    try:
        doc_version = db.get(DocumentVersion, doc_version_id)
        if not doc_version:
            return
        document = db.get(Document, doc_version.doc_id)
        if not document:
            return
        path = document.storage_path
        filename = document.filename
        text = extract_text(path)
        text_path = save_extracted_text(
            settings.file_storage_root,
            doc_version.doc_id,
            doc_version.version,
            text,
        )
        doc_version.extracted_text_path = text_path
        router = route_document(text)
        doc_version.router_json = router
        extractor = extract_document(text, router.get("doc_type", "unknown"))
        doc_version.extractor_json = extractor
        proposals = map_to_proposals(text, filename, extractor)
        for item in proposals:
            proposal = Proposal(
                doc_version_id=doc_version.id,
                proposed_action=item["proposed_action"],
                target_module=item["target_module"],
                target_table=item["target_table"],
                proposed_fields=item["proposed_fields"],
                field_confidence=item["field_confidence"],
                evidence=item["evidence"],
                questions=item["questions"],
                status="pending",
            )
            db.add(proposal)
        db.commit()
    finally:
        db.close()
