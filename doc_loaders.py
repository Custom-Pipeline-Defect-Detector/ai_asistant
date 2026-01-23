from pathlib import Path
from typing import List, Tuple

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_documents(file_paths: List[str], progress=None) -> List[Tuple[str, str]]:
    documents = []
    for path in file_paths:
        file_path = Path(path)
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if progress:
            progress(f"Loading {file_path.name}…")
        text = _extract_text(file_path)
        if text:
            documents.append((str(file_path), text))
    return documents


def _extract_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        return _extract_pdf_text(file_path)
    if file_path.suffix.lower() == ".docx":
        return _extract_docx_text(file_path)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError(
            f"Scanned PDF not supported (needs OCR): {file_path.name}"
        )
    return text


def _extract_docx_text(file_path: Path) -> str:
    doc = Document(str(file_path))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs).strip()
