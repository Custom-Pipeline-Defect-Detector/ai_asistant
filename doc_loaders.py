import json
import re
from pathlib import Path
from typing import Dict, List


def _clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _load_txt_like(path: Path) -> str:
    try:
        return _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return _clean_text(path.read_text(errors="ignore"))


def _load_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)
    return _clean_text("\n".join(parts))


def _load_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return _clean_text("\n".join(parts))


def _load_html(path: Path) -> str:
    raw = _load_txt_like(path)
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        return _clean_text(soup.get_text(separator="\n"))
    except Exception:
        raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
        raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.IGNORECASE)
        raw = re.sub(r"<[^>]+>", " ", raw)
        return _clean_text(raw)


def _load_csv(path: Path) -> str:
    import csv

    rows = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append("\t".join([c.strip() for c in row]))
    return _clean_text("\n".join(rows))


def _load_xlsx(path: Path) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    parts = []
    for ws in wb.worksheets:
        parts.append(f"=== Sheet: {ws.title} ===")
        for row in ws.iter_rows(values_only=True):
            vals = []
            for v in row:
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    vals.append(s)
            if vals:
                parts.append("\t".join(vals))
    return _clean_text("\n".join(parts))


def _load_pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts = []
    for i, slide in enumerate(prs.slides, start=1):
        parts.append(f"=== Slide {i} ===")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                t = shape.text.strip()
                if t:
                    parts.append(t)
    return _clean_text("\n".join(parts))


def _load_rtf(path: Path) -> str:
    raw = _load_txt_like(path)
    raw = re.sub(r"{\\[^}]+}", " ", raw)
    raw = re.sub(r"\\[a-zA-Z]+\d* ?", " ", raw)
    raw = raw.replace("{", " ").replace("}", " ")
    return _clean_text(raw)


def load_documents(paths: List[Path]) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []

    for p in paths:
        if not p.exists() or not p.is_file():
            continue

        ext = p.suffix.lower()
        text = ""

        if ext in {".txt", ".md", ".log", ".ini", ".xml"}:
            text = _load_txt_like(p)
        elif ext == ".pdf":
            text = _load_pdf_text(p)
        elif ext == ".docx":
            text = _load_docx(p)
        elif ext in {".html", ".htm"}:
            text = _load_html(p)
        elif ext == ".csv":
            text = _load_csv(p)
        elif ext == ".json":
            try:
                obj = json.loads(_load_txt_like(p))
                text = _clean_text(json.dumps(obj, ensure_ascii=False, indent=2))
            except Exception:
                text = _load_txt_like(p)
        elif ext == ".rtf":
            text = _load_rtf(p)
        elif ext == ".xlsx":
            try:
                text = _load_xlsx(p)
            except Exception:
                text = ""  # handled by rag_engine diagnostics
        elif ext == ".pptx":
            try:
                text = _load_pptx(p)
            except Exception:
                text = ""  # handled by rag_engine diagnostics

        if text and text.strip():
            docs.append({"source": str(p), "text": text})

    return docs
