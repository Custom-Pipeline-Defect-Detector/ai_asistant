from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_BASE_DIR = Path(r"D:\offline_doc_assistant")


def _expand_path(value: str) -> Path:
    expanded = os.path.expandvars(value.strip())
    return Path(expanded).expanduser()


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    auto_docs_dir: Path
    embed_model_dir: Path
    data_dir: Path
    index_dir: Path
    log_dir: Path
    app_icon_ico: Path
    app_icon_png: Path
    ui_logo_path: Path
    window_title: str
    always_on_top: bool
    min_width: int
    min_height: int
    logo_height_px: int
    ollama_base_url: str
    ollama_model: str
    enable_ocr: bool
    ocr_lang: str
    enable_libreoffice: bool
    poppler_path: Optional[str]
    tesseract_exe: Optional[str]
    soffice_path: Optional[str]
    pdf_ocr_max_pages: Optional[int]
    scanned_pdf_min_chars: int
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]
    top_k: int
    embed_batch_size: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int


def _default_config(base_dir: Path) -> dict[str, dict[str, str]]:
    data_dir = base_dir / "data"
    index_dir = data_dir / "index"
    log_dir = data_dir / "logs"
    return {
        "paths": {
            "base_dir": str(base_dir),
            "auto_docs_dir": str(base_dir / "index"),
            "embed_model_dir": str(base_dir / "models" / "embed"),
            "data_dir": str(data_dir),
            "index_dir": str(index_dir),
            "log_dir": str(log_dir),
            "app_icon_ico": str(base_dir / "app.ico"),
            "app_icon_png": str(base_dir / "logo.png"),
            "ui_logo_path": str(base_dir / "logo.png"),
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "qwen2.5:7b-instruct",
        },
        "indexing": {
            "chunk_size": "1100",
            "chunk_overlap": "220",
            "top_k": "6",
            "embed_batch_size": "64",
            "hnsw_m": "32",
            "hnsw_ef_construction": "200",
            "hnsw_ef_search": "64",
        },
        "ocr": {
            "enable_ocr": "true",
            "ocr_lang": "auto",
            "enable_libreoffice": "true",
            "poppler_path": str(base_dir / "poppler" / "Library" / "bin"),
            "tesseract_exe": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            "soffice_path": "",
            "pdf_ocr_max_pages": "",
            "scanned_pdf_min_chars": "60",
        },
        "ui": {
            "window_title": "Document AI Assistant",
            "always_on_top": "true",
            "min_width": "800",
            "min_height": "800",
            "logo_height_px": "65",
        },
    }


def _write_default_config(path: Path, base_dir: Path) -> None:
    defaults = _default_config(base_dir)
    config = ConfigParser()
    for section, values in defaults.items():
        config[section] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        config.write(f)


def _read_int(config: ConfigParser, section: str, option: str, fallback: int) -> int:
    try:
        return config.getint(section, option, fallback=fallback)
    except Exception:
        return fallback


def _read_bool(config: ConfigParser, section: str, option: str, fallback: bool) -> bool:
    try:
        return config.getboolean(section, option, fallback=fallback)
    except Exception:
        return fallback


def _read_optional_int(config: ConfigParser, section: str, option: str) -> Optional[int]:
    raw = config.get(section, option, fallback="").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def load_app_config(path: Path) -> AppConfig:
    if not path.exists():
        _write_default_config(path, DEFAULT_BASE_DIR)

    config = ConfigParser()
    config.read(path, encoding="utf-8")

    base_dir_raw = config.get("paths", "base_dir", fallback=str(DEFAULT_BASE_DIR))
    base_dir = _expand_path(base_dir_raw)

    data_dir = _expand_path(config.get("paths", "data_dir", fallback=str(base_dir / "data")))
    index_dir = _expand_path(config.get("paths", "index_dir", fallback=str(data_dir / "index")))
    log_dir = _expand_path(config.get("paths", "log_dir", fallback=str(data_dir / "logs")))

    auto_docs_dir = _expand_path(
        config.get("paths", "auto_docs_dir", fallback=str(base_dir / "index"))
    )
    embed_model_dir = _expand_path(
        config.get("paths", "embed_model_dir", fallback=str(base_dir / "models" / "embed"))
    )

    app_icon_ico = _expand_path(
        config.get("paths", "app_icon_ico", fallback=str(base_dir / "app.ico"))
    )
    app_icon_png = _expand_path(
        config.get("paths", "app_icon_png", fallback=str(base_dir / "logo.png"))
    )
    ui_logo_path = _expand_path(
        config.get("paths", "ui_logo_path", fallback=str(base_dir / "logo.png"))
    )

    window_title = config.get("ui", "window_title", fallback="Document AI Assistant")
    always_on_top = _read_bool(config, "ui", "always_on_top", True)
    min_width = _read_int(config, "ui", "min_width", 800)
    min_height = _read_int(config, "ui", "min_height", 800)
    logo_height_px = _read_int(config, "ui", "logo_height_px", 65)

    ollama_base_url = config.get("ollama", "base_url", fallback="http://localhost:11434")
    ollama_model = config.get("ollama", "model", fallback="qwen2.5:7b-instruct")

    enable_ocr = _read_bool(config, "ocr", "enable_ocr", True)
    ocr_lang = config.get("ocr", "ocr_lang", fallback="auto").strip() or "auto"
    enable_libreoffice = _read_bool(config, "ocr", "enable_libreoffice", True)

    poppler_path = config.get("ocr", "poppler_path", fallback="").strip() or None
    tesseract_exe = config.get("ocr", "tesseract_exe", fallback="").strip() or None
    soffice_path = config.get("ocr", "soffice_path", fallback="").strip() or None
    pdf_ocr_max_pages = _read_optional_int(config, "ocr", "pdf_ocr_max_pages")
    scanned_pdf_min_chars = _read_int(config, "ocr", "scanned_pdf_min_chars", 60)

    chunk_size = _read_optional_int(config, "indexing", "chunk_size")
    chunk_overlap = _read_optional_int(config, "indexing", "chunk_overlap")
    top_k = _read_int(config, "indexing", "top_k", 6)
    embed_batch_size = _read_int(config, "indexing", "embed_batch_size", 64)
    hnsw_m = _read_int(config, "indexing", "hnsw_m", 32)
    hnsw_ef_construction = _read_int(config, "indexing", "hnsw_ef_construction", 200)
    hnsw_ef_search = _read_int(config, "indexing", "hnsw_ef_search", 64)

    return AppConfig(
        base_dir=base_dir,
        auto_docs_dir=auto_docs_dir,
        embed_model_dir=embed_model_dir,
        data_dir=data_dir,
        index_dir=index_dir,
        log_dir=log_dir,
        app_icon_ico=app_icon_ico,
        app_icon_png=app_icon_png,
        ui_logo_path=ui_logo_path,
        window_title=window_title,
        always_on_top=always_on_top,
        min_width=min_width,
        min_height=min_height,
        logo_height_px=logo_height_px,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        enable_ocr=enable_ocr,
        ocr_lang=ocr_lang,
        enable_libreoffice=enable_libreoffice,
        poppler_path=poppler_path,
        tesseract_exe=tesseract_exe,
        soffice_path=soffice_path,
        pdf_ocr_max_pages=pdf_ocr_max_pages,
        scanned_pdf_min_chars=scanned_pdf_min_chars,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        embed_batch_size=embed_batch_size,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
    )


def apply_runtime_env(config: AppConfig) -> None:
    os.environ["APP_BASE_DIR"] = str(config.base_dir)
    os.environ["APP_LOGO_PATH"] = str(config.ui_logo_path)

    os.environ["ENABLE_OCR"] = "1" if config.enable_ocr else "0"
    os.environ["OCR_LANG"] = config.ocr_lang
    os.environ["ENABLE_LIBREOFFICE"] = "1" if config.enable_libreoffice else "0"

    if config.poppler_path:
        os.environ["POPPLER_PATH"] = config.poppler_path
    if config.tesseract_exe:
        os.environ["TESSERACT_EXE"] = config.tesseract_exe
    if config.soffice_path:
        os.environ["SOFFICE_PATH"] = config.soffice_path
    if config.pdf_ocr_max_pages is not None:
        os.environ["PDF_OCR_MAX_PAGES"] = str(config.pdf_ocr_max_pages)
    os.environ["SCANNED_PDF_MIN_CHARS"] = str(config.scanned_pdf_min_chars)

    if config.chunk_size is not None:
        os.environ["CHUNK_SIZE"] = str(config.chunk_size)
    if config.chunk_overlap is not None:
        os.environ["CHUNK_OVERLAP"] = str(config.chunk_overlap)

    os.environ["TOP_K"] = str(config.top_k)
    os.environ["EMBED_BATCH_SIZE"] = str(config.embed_batch_size)
    os.environ["HNSW_M"] = str(config.hnsw_m)
    os.environ["HNSW_EF_CONSTRUCTION"] = str(config.hnsw_ef_construction)
    os.environ["HNSW_EF_SEARCH"] = str(config.hnsw_ef_search)
