from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config_io import load_ini_with_backup, atomic_write_ini

DEFAULT_BASE_DIR = Path(r"D:\offline_doc_assistant")

DEFAULT_SUPPORTED_EXTS = [
    ".txt", ".md", ".mdx", ".log", ".ini", ".cfg", ".conf", ".env", ".xml",
    ".yml", ".yaml", ".tsv", ".tex",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs",
    ".go", ".rs", ".rb", ".php", ".sql", ".sh", ".bat", ".ps1",
    ".toml", ".gitignore",
    ".pdf",
    ".docx", ".doc", ".odt",
    ".xlsx", ".xls", ".xlsm", ".xltx", ".xltm", ".ods",
    ".pptx", ".ppt", ".ppsx", ".potx", ".odp",
    ".csv", ".json", ".jsonl", ".ndjson",
    ".html", ".htm",
    ".rtf",
    ".epub",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
]


def _expand_path(value: str) -> Path:
    expanded = os.path.expandvars(value.strip())
    return Path(expanded).expanduser()


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    data_dir: Path
    index_dir: Path
    log_dir: Path
    memory_dir: Path
    auto_docs_dir: Path
    embed_model_dir: Path
    icons_dir: Path
    app_icon_ico: Path
    app_icon_png: Path
    ui_logo_path: Path
    window_title: str
    always_on_top: bool
    min_width: int
    min_height: int
    logo_height_px: int
    status_template: str
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_s: int
    generation_temperature: float
    generation_top_p: float
    generation_num_predict: int
    index_sources: List[str]
    extra_sources: List[str]
    include_globs: List[str]
    exclude_globs: List[str]
    supported_exts: List[str]
    allow_any_path: bool
    max_file_mb: int
    max_docs_per_run: int
    chunk_size: int
    chunk_overlap: int
    embed_batch_size: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    chunk_summary_max_chars: int
    doc_chunk_cache_size: int
    retrieval_top_k: int
    retrieval_min_score: float
    retrieval_rerank: bool
    enable_ocr: bool
    ocr_lang: str
    pdf_dpi: int
    pdf_ocr_max_pages: Optional[int]
    scanned_pdf_min_chars: int
    enable_libreoffice: bool
    libreoffice_timeout_s: int
    poppler_path: Optional[str]
    tesseract_exe: Optional[str]
    soffice_path: Optional[str]
    agent_enable_control: bool
    agent_control_block_syntax: str
    agent_max_steps: int
    agent_auto_rebuild_index: bool
    agent_self_tune: bool


def default_app_config(base_dir: Path) -> Dict[str, Dict[str, str]]:
    data_dir = base_dir / "data"
    index_dir = data_dir / "index"
    log_dir = data_dir / "logs"
    memory_dir = data_dir / "memory"
    return {
        "paths": {
            "base_dir": str(base_dir),
            "embed_model_dir": str(base_dir / "models" / "embed"),
            "data_dir": str(data_dir),
            "index_dir": str(index_dir),
            "log_dir": str(log_dir),
            "memory_dir": str(memory_dir),
            "auto_docs_dir": str(base_dir / "index"),
            "icons_dir": str(base_dir),
            "app_icon_ico": str(base_dir / "app.ico"),
            "app_icon_png": str(base_dir / "logo.png"),
            "ui_logo_path": str(base_dir / "logo.png"),
        },
        "index": {
            "sources": str(base_dir / "index"),
            "extra_sources": "",
            "include_globs": "",
            "exclude_globs": "",
            "supported_exts": "; ".join(DEFAULT_SUPPORTED_EXTS),
            "allow_any_path": "true",
            "max_file_mb": "50",
            "max_docs_per_run": "0",
            "chunk_size": "1100",
            "chunk_overlap": "220",
            "embed_batch_size": "64",
            "hnsw_m": "32",
            "hnsw_ef_construction": "200",
            "hnsw_ef_search": "64",
            "chunk_summary_max_chars": "300",
            "doc_chunk_cache_size": "16",
        },
        "retrieval": {
            "top_k": "6",
            "min_score": "0.0",
            "rerank": "false",
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "qwen2.5:7b-instruct",
            "timeout_s": "600",
        },
        "generation": {
            "temperature": "0.2",
            "top_p": "0.9",
            "num_predict": "900",
        },
        "ocr": {
            "enable_ocr": "true",
            "lang": "auto",
            "pdf_dpi": "250",
            "max_pages": "",
            "poppler_path": str(base_dir / "poppler" / "Library" / "bin"),
            "tesseract_exe": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            "scanned_pdf_min_chars": "60",
        },
        "libreoffice": {
            "enable_libreoffice": "true",
            "timeout_s": "90",
            "soffice_path": "",
        },
        "ui": {
            "window_title": "Document AI Assistant",
            "always_on_top": "true",
            "min_width": "800",
            "min_height": "800",
            "logo_height_px": "65",
            "status_template": "Ready. Place documents in {auto_docs_dir} "
            "(OCR={ocr_status}, LibreOffice={libreoffice_status})",
        },
        "agent": {
            "enable_control": "true",
            "control_block_syntax": "<CONTROL>...</CONTROL>",
            "max_steps": "3",
            "auto_rebuild_index": "true",
            "self_tune": "true",
        },
    }


def write_default_config(path: Path, base_dir: Path) -> None:
    defaults = default_app_config(base_dir)
    config = ConfigParser()
    for section, values in defaults.items():
        config[section] = values
    atomic_write_ini(path, config)


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


def _read_float(config: ConfigParser, section: str, option: str, fallback: float) -> float:
    try:
        return config.getfloat(section, option, fallback=fallback)
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


def _split_list(raw: str) -> List[str]:
    parts = []
    for chunk in raw.replace("\n", ";").split(";"):
        cleaned = chunk.strip()
        if cleaned:
            parts.append(cleaned)
    return parts


def _normalize_exts(values: List[str]) -> List[str]:
    out = []
    for ext in values:
        e = ext.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        out.append(e)
    return sorted(set(out))


def load_app_config(path: Path) -> AppConfig:
    defaults = default_app_config(DEFAULT_BASE_DIR)
    config = load_ini_with_backup(path, defaults)

    base_dir_raw = config.get("paths", "base_dir", fallback=str(DEFAULT_BASE_DIR))
    base_dir = _expand_path(base_dir_raw)

    data_dir = _expand_path(config.get("paths", "data_dir", fallback=str(base_dir / "data")))
    index_dir = _expand_path(config.get("paths", "index_dir", fallback=str(data_dir / "index")))
    log_dir = _expand_path(config.get("paths", "log_dir", fallback=str(data_dir / "logs")))
    memory_dir = _expand_path(config.get("paths", "memory_dir", fallback=str(data_dir / "memory")))

    auto_docs_dir = _expand_path(
        config.get("paths", "auto_docs_dir", fallback=str(base_dir / "index"))
    )
    embed_model_dir = _expand_path(
        config.get("paths", "embed_model_dir", fallback=str(base_dir / "models" / "embed"))
    )
    icons_dir = _expand_path(config.get("paths", "icons_dir", fallback=str(base_dir)))

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
    status_template = config.get(
        "ui",
        "status_template",
        fallback="Ready. Place documents in {auto_docs_dir} (OCR={ocr_status}, LibreOffice={libreoffice_status})",
    )

    ollama_base_url = config.get("ollama", "base_url", fallback="http://localhost:11434")
    ollama_model = config.get("ollama", "model", fallback="qwen2.5:7b-instruct")
    ollama_timeout_s = _read_int(config, "ollama", "timeout_s", 600)

    generation_temperature = _read_float(config, "generation", "temperature", 0.2)
    generation_top_p = _read_float(config, "generation", "top_p", 0.9)
    generation_num_predict = _read_int(config, "generation", "num_predict", 900)

    index_sources = _split_list(config.get("index", "sources", fallback=str(auto_docs_dir)))
    if not index_sources:
        index_sources = [str(auto_docs_dir)]
    extra_sources = _split_list(config.get("index", "extra_sources", fallback=""))
    include_globs = _split_list(config.get("index", "include_globs", fallback=""))
    exclude_globs = _split_list(config.get("index", "exclude_globs", fallback=""))
    supported_exts = _normalize_exts(_split_list(config.get("index", "supported_exts", fallback="")))
    allow_any_path = _read_bool(config, "index", "allow_any_path", True)
    max_file_mb = _read_int(config, "index", "max_file_mb", 50)
    max_docs_per_run = _read_int(config, "index", "max_docs_per_run", 0)

    enable_ocr = _read_bool(config, "ocr", "enable_ocr", True)
    ocr_lang = config.get("ocr", "lang", fallback="auto").strip() or "auto"
    pdf_dpi = _read_int(config, "ocr", "pdf_dpi", 250)
    pdf_ocr_max_pages = _read_optional_int(config, "ocr", "max_pages")
    poppler_path = config.get("ocr", "poppler_path", fallback="").strip() or None
    tesseract_exe = config.get("ocr", "tesseract_exe", fallback="").strip() or None
    if not poppler_path:
        poppler_path = os.environ.get("POPPLER_PATH", "").strip() or None
    if not tesseract_exe:
        tesseract_exe = os.environ.get("TESSERACT_EXE", "").strip() or None
    scanned_pdf_min_chars = _read_int(config, "ocr", "scanned_pdf_min_chars", 60)

    enable_libreoffice = _read_bool(config, "libreoffice", "enable_libreoffice", True)
    libreoffice_timeout_s = _read_int(config, "libreoffice", "timeout_s", 90)
    soffice_path = config.get("libreoffice", "soffice_path", fallback="").strip() or None
    if not soffice_path:
        soffice_path = os.environ.get("SOFFICE_PATH", "").strip().strip('"') or None

    chunk_size = _read_int(config, "index", "chunk_size", 1100)
    chunk_overlap = _read_int(config, "index", "chunk_overlap", 220)
    embed_batch_size = _read_int(config, "index", "embed_batch_size", 64)
    hnsw_m = _read_int(config, "index", "hnsw_m", 32)
    hnsw_ef_construction = _read_int(config, "index", "hnsw_ef_construction", 200)
    hnsw_ef_search = _read_int(config, "index", "hnsw_ef_search", 64)
    chunk_summary_max_chars = _read_int(config, "index", "chunk_summary_max_chars", 300)
    doc_chunk_cache_size = _read_int(config, "index", "doc_chunk_cache_size", 16)

    retrieval_top_k = _read_int(config, "retrieval", "top_k", 6)
    retrieval_min_score = _read_float(config, "retrieval", "min_score", 0.0)
    retrieval_rerank = _read_bool(config, "retrieval", "rerank", False)

    agent_enable_control = _read_bool(config, "agent", "enable_control", True)
    agent_control_block_syntax = config.get(
        "agent",
        "control_block_syntax",
        fallback="<CONTROL>...</CONTROL>",
    ).strip()
    agent_max_steps = _read_int(config, "agent", "max_steps", 3)
    agent_auto_rebuild_index = _read_bool(config, "agent", "auto_rebuild_index", True)
    agent_self_tune = _read_bool(config, "agent", "self_tune", True)

    return AppConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        index_dir=index_dir,
        log_dir=log_dir,
        memory_dir=memory_dir,
        auto_docs_dir=auto_docs_dir,
        embed_model_dir=embed_model_dir,
        icons_dir=icons_dir,
        app_icon_ico=app_icon_ico,
        app_icon_png=app_icon_png,
        ui_logo_path=ui_logo_path,
        window_title=window_title,
        always_on_top=always_on_top,
        min_width=min_width,
        min_height=min_height,
        logo_height_px=logo_height_px,
        status_template=status_template,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        ollama_timeout_s=ollama_timeout_s,
        generation_temperature=generation_temperature,
        generation_top_p=generation_top_p,
        generation_num_predict=generation_num_predict,
        index_sources=index_sources,
        extra_sources=extra_sources,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        supported_exts=supported_exts,
        allow_any_path=allow_any_path,
        max_file_mb=max_file_mb,
        max_docs_per_run=max_docs_per_run,
        enable_ocr=enable_ocr,
        ocr_lang=ocr_lang,
        pdf_dpi=pdf_dpi,
        pdf_ocr_max_pages=pdf_ocr_max_pages,
        scanned_pdf_min_chars=scanned_pdf_min_chars,
        enable_libreoffice=enable_libreoffice,
        libreoffice_timeout_s=libreoffice_timeout_s,
        poppler_path=poppler_path,
        tesseract_exe=tesseract_exe,
        soffice_path=soffice_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_batch_size=embed_batch_size,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        chunk_summary_max_chars=chunk_summary_max_chars,
        doc_chunk_cache_size=doc_chunk_cache_size,
        retrieval_top_k=retrieval_top_k,
        retrieval_min_score=retrieval_min_score,
        retrieval_rerank=retrieval_rerank,
        agent_enable_control=agent_enable_control,
        agent_control_block_syntax=agent_control_block_syntax,
        agent_max_steps=agent_max_steps,
        agent_auto_rebuild_index=agent_auto_rebuild_index,
        agent_self_tune=agent_self_tune,
    )


def apply_runtime_env(config: AppConfig) -> None:
    os.environ["APP_BASE_DIR"] = str(config.base_dir)
    os.environ["APP_LOGO_PATH"] = str(config.ui_logo_path)
    os.environ["APP_LOG_DIR"] = str(config.log_dir)
    os.environ["APP_INDEX_DIR"] = str(config.index_dir)
    os.environ["APP_MEMORY_DIR"] = str(config.memory_dir)


APP_CONFIG_SCHEMA: Dict[str, Dict[str, str]] = {
    "paths": {
        "base_dir": "path",
        "data_dir": "path",
        "index_dir": "path",
        "log_dir": "path",
        "memory_dir": "path",
        "auto_docs_dir": "path",
        "embed_model_dir": "path",
        "icons_dir": "path",
        "app_icon_ico": "path",
        "app_icon_png": "path",
        "ui_logo_path": "path",
    },
    "index": {
        "sources": "list",
        "extra_sources": "list",
        "include_globs": "list",
        "exclude_globs": "list",
        "supported_exts": "list",
        "allow_any_path": "bool",
        "max_file_mb": "int",
        "max_docs_per_run": "int",
        "chunk_size": "int",
        "chunk_overlap": "int",
        "embed_batch_size": "int",
        "hnsw_m": "int",
        "hnsw_ef_construction": "int",
        "hnsw_ef_search": "int",
        "chunk_summary_max_chars": "int",
        "doc_chunk_cache_size": "int",
    },
    "retrieval": {
        "top_k": "int",
        "min_score": "float",
        "rerank": "bool",
    },
    "ollama": {
        "base_url": "str",
        "model": "str",
        "timeout_s": "int",
    },
    "generation": {
        "temperature": "float",
        "top_p": "float",
        "num_predict": "int",
    },
    "ocr": {
        "enable_ocr": "bool",
        "lang": "str",
        "pdf_dpi": "int",
        "max_pages": "optional_int",
        "poppler_path": "optional_path",
        "tesseract_exe": "optional_path",
        "scanned_pdf_min_chars": "int",
    },
    "libreoffice": {
        "enable_libreoffice": "bool",
        "timeout_s": "int",
        "soffice_path": "optional_path",
    },
    "ui": {
        "window_title": "str",
        "always_on_top": "bool",
        "min_width": "int",
        "min_height": "int",
        "logo_height_px": "int",
        "status_template": "str",
    },
    "agent": {
        "enable_control": "bool",
        "control_block_syntax": "str",
        "max_steps": "int",
        "auto_rebuild_index": "bool",
        "self_tune": "bool",
    },
}


def normalize_app_value(section: str, key: str, raw: str) -> Tuple[str, str]:
    section = section.lower()
    key = key.lower()
    value_type = APP_CONFIG_SCHEMA.get(section, {}).get(key)
    if value_type is None:
        raise ValueError(f"Unsupported config key: {section}.{key}")

    text = raw.strip()
    if value_type == "bool":
        if text.lower() in {"1", "true", "yes", "on"}:
            return section, "true"
        if text.lower() in {"0", "false", "no", "off"}:
            return section, "false"
        raise ValueError(f"Invalid boolean for {section}.{key}: {raw}")
    if value_type == "int":
        return section, str(int(text))
    if value_type == "float":
        return section, str(float(text))
    if value_type == "optional_int":
        if not text:
            return section, ""
        return section, str(int(text))
    if value_type == "path":
        if not text:
            raise ValueError(f"Empty path for {section}.{key}")
        return section, text
    if value_type == "optional_path":
        return section, text
    if value_type == "list":
        items = _split_list(text)
        if key == "supported_exts":
            items = _normalize_exts(items)
        return section, "; ".join(items)
    return section, text
