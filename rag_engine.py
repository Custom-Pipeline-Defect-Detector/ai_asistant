import json
import logging
import re
import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

from configparser import ConfigParser

from config_io import load_ini_with_backup, atomic_write_ini

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from doc_loaders import load_documents, LoadReportEntry


@dataclass
class ChunkLite:
    chunk_id: int
    doc_id: str
    source: str
    summary: str
    file: str
    idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkLite":
        base = {
            "chunk_id": data.get("chunk_id"),
            "doc_id": data.get("doc_id"),
            "source": data.get("source"),
            "summary": data.get("summary"),
            "file": data.get("file"),
            "idx": data.get("idx"),
        }
        missing = [k for k, v in base.items() if v is None]
        if missing:
            raise RAGEngineError(f"Invalid chunk entry missing {missing}")
        extra = {k: v for k, v in data.items() if k not in base}
        return cls(
            chunk_id=int(base["chunk_id"]),
            doc_id=str(base["doc_id"]),
            source=str(base["source"]),
            summary=str(base["summary"]),
            file=str(base["file"]),
            idx=int(base["idx"]),
            metadata=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "summary": self.summary,
            "file": self.file,
            "idx": self.idx,
        }
        payload.update(self.metadata)
        return payload


class RAGEngineError(Exception):
    pass


def default_assistant_config() -> Dict[str, Dict[str, str]]:
    return {
        "user": {
            "name": "Mir",
            "language": "auto",
        },
        "assistant": {
            "name": "Assistant",
            "tone": "professional_friendly",
            "personality": "Helpful, concise, accurate. Uses citations when possible. Asks follow-ups only if necessary.",
            "custom_rules": "",
            "system_prompt_template": "",
        },
        "preferences": {
            "time_format": "24h",
            "timezone": "local",
            "auto_save_facts": "true",
        },
        "agent": {
            "enable_control": "true",
            "control_block_syntax": "<CONTROL>...</CONTROL>",
            "max_steps": "3",
            "auto_rebuild_index": "true",
            "self_tune": "true",
        },
    }


ASSISTANT_SCHEMA: Dict[str, Dict[str, str]] = {
    "user": {
        "name": "str",
        "language": "str",
    },
    "assistant": {
        "name": "str",
        "tone": "str",
        "personality": "str",
        "custom_rules": "str",
        "system_prompt_template": "str",
    },
    "preferences": {
        "time_format": "str",
        "timezone": "str",
        "auto_save_facts": "bool",
    },
    "agent": {
        "enable_control": "bool",
        "control_block_syntax": "str",
        "max_steps": "int",
        "auto_rebuild_index": "bool",
        "self_tune": "bool",
    },
}


def normalize_assistant_value(section: str, key: str, raw: str) -> Tuple[str, str]:
    section = section.lower()
    key = key.lower()
    value_type = ASSISTANT_SCHEMA.get(section, {}).get(key)
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
    return section, text


class RAGEngine:
    """
    RAG + Ollama LLM + Persistent Memory.

    Indexing upgrades:
    - Incremental indexing via manifest.json
    - FAISS HNSW + IDMap2 (fast approximate search + supports remove_ids)
    - Automatic OCR decisions handled in doc_loaders (PDF auto OCR)
    """

    def __init__(
        self,
        base_dir: Path,
        embed_model_dir: Path,
        index_dir: Path,
        log_dir: Path,
        memory_dir: Path,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b-instruct",
        ollama_timeout_s: int = 600,
        generation_temperature: float = 0.2,
        generation_top_p: float = 0.9,
        generation_num_predict: int = 900,
        retrieval_min_score: float = 0.0,
        embed_batch_size: int = 64,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        enable_ocr: bool = True,
        ocr_lang: str = "auto",
        enable_libreoffice: bool = True,
        poppler_path: Optional[str] = None,
        tesseract_exe: Optional[str] = None,
        soffice_path: Optional[str] = None,
        pdf_ocr_max_pages: Optional[int] = None,
        scanned_pdf_min_chars: int = 60,
        pdf_dpi: int = 250,
        libreoffice_timeout_s: int = 90,
        chunk_summary_max_chars: int = 300,
        doc_chunk_cache_size: int = 16,
    ):
        self.base_dir = base_dir
        self.embed_model_dir = embed_model_dir
        self.index_dir = index_dir
        self.log_dir = log_dir

        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.ollama_timeout_s = ollama_timeout_s
        self.generation_temperature = generation_temperature
        self.generation_top_p = generation_top_p
        self.generation_num_predict = generation_num_predict
        self.retrieval_min_score = retrieval_min_score
        self.embed_batch_size = embed_batch_size
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.enable_ocr = enable_ocr
        self.ocr_lang = ocr_lang
        self.enable_libreoffice = enable_libreoffice
        self.poppler_path = poppler_path
        self.tesseract_exe = tesseract_exe
        self.soffice_path = soffice_path
        self.pdf_ocr_max_pages = pdf_ocr_max_pages
        self.scanned_pdf_min_chars = scanned_pdf_min_chars
        self.pdf_dpi = pdf_dpi
        self.libreoffice_timeout_s = libreoffice_timeout_s
        self.chunk_summary_max_chars = chunk_summary_max_chars
        self.doc_chunk_cache_size = doc_chunk_cache_size

        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "chunks.json"
        self.docs_path = self.index_dir / "docs.json"
        self.manifest_path = self.index_dir / "manifest.json"
        self.doc_chunks_dir = self.index_dir / "doc_chunks"

        self._embedder: Optional[SentenceTransformer] = None
        self._faiss: Optional[faiss.Index] = None
        self._chunks: List[ChunkLite] = []
        self._active_ids: Optional[set[int]] = None  # optional fast filter
        # fast lookup for chunk metadata by chunk_id (populated on load)
        self._chunk_map: Optional[Dict[int, ChunkLite]] = None
        self._doc_chunks_cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()

        # Persistent memory
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.ini_path = self.memory_dir / "assistant.ini"
        self.chat_log_path = self.memory_dir / "chat_history.jsonl"
        self.chat_summary_path = self.memory_dir / "chat_summary.txt"

        self._setup_logging()
        self._ensure_memory_files()

    # ---------------- logging ----------------
    def _setup_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / "app.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        )
        logging.info("RAGEngine initialized")

    # ---------------- memory (INI + chat log) ----------------
    def _ensure_memory_files(self):
        if not self.ini_path.exists():
            config = ConfigParser()
            for section, values in default_assistant_config().items():
                config[section] = values
            atomic_write_ini(self.ini_path, config)

        if not self.chat_summary_path.exists():
            self.chat_summary_path.write_text(
                "Conversation summary (short, important facts only):\n",
                encoding="utf-8",
            )

        if not self.chat_log_path.exists():
            self.chat_log_path.write_text("", encoding="utf-8")

    def apply_runtime_config(
        self,
        *,
        base_dir: Path,
        embed_model_dir: Path,
        index_dir: Path,
        log_dir: Path,
        memory_dir: Path,
        ollama_base_url: str,
        ollama_model: str,
        ollama_timeout_s: int,
        generation_temperature: float,
        generation_top_p: float,
        generation_num_predict: int,
        retrieval_min_score: float,
        embed_batch_size: int,
        hnsw_m: int,
        hnsw_ef_construction: int,
        hnsw_ef_search: int,
        enable_ocr: bool,
        ocr_lang: str,
        enable_libreoffice: bool,
        poppler_path: Optional[str],
        tesseract_exe: Optional[str],
        soffice_path: Optional[str],
        pdf_ocr_max_pages: Optional[int],
        scanned_pdf_min_chars: int,
        pdf_dpi: int,
        libreoffice_timeout_s: int,
        chunk_summary_max_chars: int,
        doc_chunk_cache_size: int,
    ) -> None:
        index_changed = index_dir != self.index_dir
        memory_changed = memory_dir != self.memory_dir

        self.base_dir = base_dir
        self.embed_model_dir = embed_model_dir
        self.index_dir = index_dir
        self.log_dir = log_dir
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.ollama_timeout_s = ollama_timeout_s
        self.generation_temperature = generation_temperature
        self.generation_top_p = generation_top_p
        self.generation_num_predict = generation_num_predict
        self.retrieval_min_score = retrieval_min_score
        self.embed_batch_size = embed_batch_size
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.enable_ocr = enable_ocr
        self.ocr_lang = ocr_lang
        self.enable_libreoffice = enable_libreoffice
        self.poppler_path = poppler_path
        self.tesseract_exe = tesseract_exe
        self.soffice_path = soffice_path
        self.pdf_ocr_max_pages = pdf_ocr_max_pages
        self.scanned_pdf_min_chars = scanned_pdf_min_chars
        self.pdf_dpi = pdf_dpi
        self.libreoffice_timeout_s = libreoffice_timeout_s
        self.chunk_summary_max_chars = chunk_summary_max_chars
        self.doc_chunk_cache_size = doc_chunk_cache_size

        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "chunks.json"
        self.docs_path = self.index_dir / "docs.json"
        self.manifest_path = self.index_dir / "manifest.json"
        self.doc_chunks_dir = self.index_dir / "doc_chunks"

        if memory_changed:
            self.memory_dir = memory_dir
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.ini_path = self.memory_dir / "assistant.ini"
            self.chat_log_path = self.memory_dir / "chat_history.jsonl"
            self.chat_summary_path = self.memory_dir / "chat_summary.txt"
            self._ensure_memory_files()

        if index_changed:
            self._faiss = None
            self._chunks = []
            self._active_ids = None
            self._chunk_map = None
            self._doc_chunks_cache = OrderedDict()

    def load_ini(self) -> Dict[str, Dict[str, str]]:
        config = load_ini_with_backup(self.ini_path, default_assistant_config())
        data: Dict[str, Dict[str, str]] = {}
        for section in config.sections():
            data[section] = {k: v for k, v in config.items(section)}
        return data

    def save_ini(self, data: Dict[str, Dict[str, str]]) -> None:
        config = ConfigParser()
        for section, kv in data.items():
            config[section] = kv
        atomic_write_ini(self.ini_path, config)

    def append_chat_log(self, role: str, text: str) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "text": text,
        }
        with self.chat_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_recent_chat(self, max_messages: int = 10) -> List[Dict[str, str]]:
        if not self.chat_log_path.exists():
            return []
        lines = self.chat_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        out = []
        for ln in lines[-max_messages:]:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return out

    def get_chat_summary(self) -> str:
        try:
            return self.chat_summary_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def set_chat_summary(self, summary: str) -> None:
        self.chat_summary_path.write_text(summary.strip() + "\n", encoding="utf-8")

    def clear_memory(self) -> None:
        self.chat_log_path.write_text("", encoding="utf-8")
        self.chat_summary_path.write_text("Conversation summary (short, important facts only):\n", encoding="utf-8")

    # ---------------- embed/index ----------------
    def _load_embedder(self):
        if self._embedder is None:
            if not self.embed_model_dir.exists():
                raise RAGEngineError(f"Embedding model folder missing: {self.embed_model_dir}")
            self._embedder = SentenceTransformer(str(self.embed_model_dir))

    def index_exists(self) -> bool:
        return self.index_path.exists() and self.chunks_path.exists() and self.docs_path.exists() and self.manifest_path.exists()

    def get_indexed_docs(self) -> List[str]:
        if not self.docs_path.exists():
            return []
        try:
            return json.loads(self.docs_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def clear_index(self) -> None:
        for p in [self.index_path, self.chunks_path, self.docs_path, self.manifest_path]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        if self.doc_chunks_dir.exists():
            for p in self.doc_chunks_dir.glob("*.json"):
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                self.doc_chunks_dir.rmdir()
            except Exception:
                pass
        self._faiss = None
        self._chunks = []
        self._active_ids = None
        self._chunk_map = None
        self._doc_chunks_cache = OrderedDict()

    # ---------- manifest ----------
    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": 1, "files": {}, "next_chunk_id": 0}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "files": {}, "next_chunk_id": 0}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _file_stat_fingerprint(path: Path) -> Dict[str, Any]:
        st = path.stat()
        return {
            "mtime": int(st.st_mtime),
            "size": int(st.st_size),
        }

    @staticmethod
    def _sha1_file(path: Path, max_bytes: int = 2_000_000) -> str:
        """
        Optional fallback: hash first N bytes + size to detect same-mtime weirdness.
        Keep it cheap.
        """
        h = hashlib.sha1()
        with path.open("rb") as f:
            h.update(f.read(max_bytes))
        h.update(str(path.stat().st_size).encode("utf-8"))
        return h.hexdigest()

    # ---------- index IO ----------
    def _save_index(self, index: faiss.Index, chunks: List[ChunkLite], docs: List[str]):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self.chunks_path.write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.docs_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index(self):
        if self._faiss is not None and self._chunks:
            return
        if not self.index_exists():
            raise RAGEngineError("Index not found. Please build the index first.")
        self._faiss = faiss.read_index(str(self.index_path))
        self._chunks = self._load_chunk_lites()

        # Build active id set from manifest (fast filtering)
        manifest = self._load_manifest()
        active_ids: set[int] = set()
        for _, meta in (manifest.get("files", {}) or {}).items():
            if not meta.get("active", True):
                continue
            for cid in meta.get("chunk_ids", []) or []:
                active_ids.add(int(cid))
        self._active_ids = active_ids
        # Build chunk map for fast retrieval lookup
        self._chunk_map = {c.chunk_id: c for c in self._chunks}

    @staticmethod
    def _normalize_path_for_id(path: str) -> str:
        resolved = Path(path).expanduser().resolve()
        normalized = resolved.as_posix()
        if os.name == "nt":
            normalized = normalized.lower()
        return normalized

    def _doc_id_for_source(self, path: str) -> str:
        normalized = self._normalize_path_for_id(path)
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def summarize_chunk(text: str, max_len: int = 300) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if max_len <= 0:
            return cleaned
        if len(cleaned) <= max_len:
            return cleaned
        if max_len == 1:
            return "…"
        trimmed = cleaned[: max_len - 1]
        return trimmed + "…"

    def _cache_doc_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]) -> None:
        if self.doc_chunk_cache_size <= 0:
            return
        if doc_id in self._doc_chunks_cache:
            self._doc_chunks_cache.move_to_end(doc_id)
        self._doc_chunks_cache[doc_id] = chunks
        while len(self._doc_chunks_cache) > self.doc_chunk_cache_size:
            self._doc_chunks_cache.popitem(last=False)

    def _load_doc_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        if doc_id in self._doc_chunks_cache:
            self._doc_chunks_cache.move_to_end(doc_id)
            return self._doc_chunks_cache[doc_id]
        path = self.doc_chunks_dir / f"{doc_id}.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            chunks = data.get("chunks") or []
        except Exception:
            chunks = []
        self._cache_doc_chunks(doc_id, chunks)
        return chunks

    def _resolve_chunk_text(self, chunk: ChunkLite) -> str:
        chunks = self._load_doc_chunks(chunk.doc_id)
        if 0 <= chunk.idx < len(chunks):
            candidate = chunks[chunk.idx]
            if candidate.get("chunk_id") == chunk.chunk_id:
                return candidate.get("text", "")
        for item in chunks:
            if item.get("chunk_id") == chunk.chunk_id:
                return item.get("text", "")
        return ""

    def _write_doc_chunks(self, doc_id: str, source: str, chunks: List[Dict[str, Any]]) -> None:
        self.doc_chunks_dir.mkdir(parents=True, exist_ok=True)
        payload = {"doc_id": doc_id, "source": source, "chunks": chunks}
        path = self.doc_chunks_dir / f"{doc_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if doc_id in self._doc_chunks_cache:
            self._doc_chunks_cache.pop(doc_id, None)

    def _migrate_legacy_chunks(self, raw: List[Dict[str, Any]]) -> None:
        grouped: Dict[str, Dict[str, Any]] = {}
        for entry in raw:
            if "text" not in entry:
                continue
            source = str(entry.get("source") or "")
            text = str(entry.get("text") or "")
            chunk_id = int(entry.get("chunk_id"))
            doc_id = self._doc_id_for_source(source)
            doc_info = grouped.setdefault(doc_id, {"source": source, "chunks": []})
            doc_info["chunks"].append({"chunk_id": chunk_id, "text": text})

        chunk_lites: List[ChunkLite] = []
        for doc_id, info in grouped.items():
            source = info["source"]
            chunks = info["chunks"]
            self._write_doc_chunks(doc_id, source, chunks)
            for idx, item in enumerate(chunks):
                summary = self.summarize_chunk(item.get("text", ""), self.chunk_summary_max_chars)
                chunk_lites.append(
                    ChunkLite(
                        chunk_id=int(item["chunk_id"]),
                        doc_id=doc_id,
                        source=source,
                        summary=summary,
                        file=f"doc_chunks/{doc_id}.json",
                        idx=idx,
                    )
                )

        self.chunks_path.write_text(
            json.dumps([c.to_dict() for c in chunk_lites], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest = self._load_manifest()
        files_meta = manifest.get("files", {}) or {}
        for path, meta in files_meta.items():
            if not meta.get("doc_id"):
                meta["doc_id"] = self._doc_id_for_source(path)
            files_meta[path] = meta
        manifest["files"] = files_meta
        self._save_manifest(manifest)

    def _load_chunk_lites(self) -> List[ChunkLite]:
        raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if any("text" in entry for entry in raw):
            self._migrate_legacy_chunks(raw)
            raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        return [ChunkLite.from_dict(c) for c in raw]
    # ---------- chunking ----------
    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1100, overlap: int = 220) -> List[str]:
        text = text.replace("\r\n", "\n")
        if not text.strip():
            return []
        # If chunk_size is zero or negative, or larger than the whole text, return a single full-document chunk.
        if chunk_size is None or chunk_size <= 0 or chunk_size >= len(text):
            s = text.strip()
            return [s] if s else []
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == n:
                break
            start = max(0, end - overlap)
        return chunks

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / norm

    def _new_faiss_index(self, dim: int) -> faiss.Index:
        """
        Fast ANN:
        - HNSW for speed on large chunk sets
        - IDMap2 to allow remove_ids
        """
        hnsw = faiss.IndexHNSWFlat(dim, int(self.hnsw_m), faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = int(self.hnsw_ef_construction)
        hnsw.hnsw.efSearch = int(self.hnsw_ef_search)

        return faiss.IndexIDMap2(hnsw)

    # ---------- incremental build ----------
    def build_index(
        self,
        file_paths: List[str],
        progress: Callable[[str], None] = lambda _: None,
        chunk_size: int = 1100,
        overlap: int = 220,
        # OCR knobs (auto behavior is in doc_loaders)
        enable_ocr: Optional[bool] = None,
        ocr_lang: Optional[str] = None,
        enable_libreoffice: Optional[bool] = None,
    ) -> Dict[str, object]:
        """
        Incremental indexing:
        - Detect added/changed/deleted files via manifest
        - Remove chunks of deleted/changed files from FAISS
        - Add chunks for newly processed files only
        """
        self._load_embedder()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Defaults: make it "automatic" without user choosing
        if enable_ocr is None:
            enable_ocr = self.enable_ocr
        if ocr_lang is None:
            ocr_lang = self.ocr_lang
        if enable_libreoffice is None:
            enable_libreoffice = self.enable_libreoffice

        poppler_path = self.poppler_path
        pdf_ocr_max_pages = self.pdf_ocr_max_pages
        scanned_pdf_min_chars = self.scanned_pdf_min_chars

        # Load existing index/manifest if present
        manifest = self._load_manifest()
        files_meta: Dict[str, Any] = manifest.get("files", {}) or {}
        next_chunk_id = int(manifest.get("next_chunk_id", 0))
        for path, meta in list(files_meta.items()):
            if not meta.get("doc_id"):
                meta["doc_id"] = self._doc_id_for_source(path)
                files_meta[path] = meta

        existing_paths_set = set(files_meta.keys())
        current_paths_set = set(file_paths)

        deleted_paths = sorted(existing_paths_set - current_paths_set)
        candidate_paths = sorted(current_paths_set)

        # Decide which changed
        changed_paths: List[str] = []
        for p in candidate_paths:
            pp = Path(p)
            if not pp.exists():
                continue
            fp = self._file_stat_fingerprint(pp)
            old = files_meta.get(p)
            if not old:
                changed_paths.append(p)
                continue
            if int(old.get("mtime", -1)) != int(fp["mtime"]) or int(old.get("size", -1)) != int(fp["size"]):
                changed_paths.append(p)

        # Load or initialize FAISS + chunks
        index: Optional[faiss.Index] = None
        chunks: List[ChunkLite] = []
        if self.index_path.exists() and self.chunks_path.exists():
            progress("Loading existing index…")
            index = faiss.read_index(str(self.index_path))
            chunks = self._load_chunk_lites()
        else:
            index = None
            chunks = []

        # Ensure index exists
        if index is None:
            progress("Creating new index…")
            # We need dim first -> embed one small string
            test = self._embedder.encode(["test"], convert_to_numpy=True).astype(np.float32)
            dim = int(test.shape[1])
            index = self._new_faiss_index(dim)

        # Remove chunks for deleted + changed files
        remove_paths = set(deleted_paths) | set(changed_paths)
        removed_chunk_ids: List[int] = []
        if remove_paths:
            progress(f"Removing outdated chunks ({len(remove_paths)} files)…")
            for p in remove_paths:
                meta = files_meta.get(p) or {}
                for cid in (meta.get("chunk_ids") or []):
                    removed_chunk_ids.append(int(cid))
                # mark inactive (we keep record for debug)
                meta["active"] = False
                meta["chunk_ids"] = []
                if not meta.get("doc_id"):
                    meta["doc_id"] = self._doc_id_for_source(p)
                files_meta[p] = meta

            if removed_chunk_ids:
                idsel = faiss.IDSelectorBatch(np.array(removed_chunk_ids, dtype=np.int64))
                index.remove_ids(idsel)
                removed_ids = set(removed_chunk_ids)
                chunks = [c for c in chunks if c.chunk_id not in removed_ids]

            for p in remove_paths:
                meta = files_meta.get(p) or {}
                doc_id = meta.get("doc_id") or self._doc_id_for_source(p)
                doc_path = self.doc_chunks_dir / f"{doc_id}.json"
                if doc_path.exists():
                    try:
                        doc_path.unlink()
                    except Exception:
                        pass
                self._doc_chunks_cache.pop(doc_id, None)

        # Extract+chunk only changed/new files
        progress(f"Loading documents (changed/new: {len(changed_paths)})…")
        docs, report = load_documents(
            [Path(p) for p in changed_paths],
            enable_ocr=enable_ocr,
            ocr_lang=ocr_lang,
            enable_libreoffice=enable_libreoffice,
            soffice_path=self.soffice_path,
            libreoffice_timeout_s=self.libreoffice_timeout_s,
            tesseract_exe=self.tesseract_exe,
            scanned_pdf_min_chars=scanned_pdf_min_chars,
            pdf_ocr_max_pages=pdf_ocr_max_pages,
            poppler_path=poppler_path,
            pdf_dpi=self.pdf_dpi,
        )

        attempted = len(report)
        succeeded = len(docs)
        failed = attempted - succeeded

        # Build new chunks for changed/new docs
        new_chunks: List[ChunkLite] = []
        new_chunk_ids: List[int] = []
        new_chunk_texts: List[str] = []
        per_file_chunk_ids: Dict[str, List[int]] = {}
        per_file_doc_ids: Dict[str, str] = {}

        for doc_i, d in enumerate(docs, start=1):
            progress(f"Chunking ({doc_i}/{len(docs)})…")
            source = d["source"]
            doc_id = self._doc_id_for_source(source)
            per_file_doc_ids[source] = doc_id
            chs = self._chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
            ids_for_file: List[int] = []
            doc_chunks: List[Dict[str, Any]] = []
            for idx, ch in enumerate(chs):
                cid = next_chunk_id
                next_chunk_id += 1
                ids_for_file.append(cid)
                summary = self.summarize_chunk(ch, self.chunk_summary_max_chars)
                new_chunks.append(
                    ChunkLite(
                        chunk_id=cid,
                        doc_id=doc_id,
                        source=source,
                        summary=summary,
                        file=f"doc_chunks/{doc_id}.json",
                        idx=idx,
                    )
                )
                new_chunk_ids.append(cid)
                new_chunk_texts.append(ch)
                doc_chunks.append({"chunk_id": cid, "text": ch})
            per_file_chunk_ids[source] = ids_for_file
            self._write_doc_chunks(doc_id, source, doc_chunks)

        # Embed only new chunks
        if new_chunks:
            progress("Embedding new chunks…")
            emb = self._embedder.encode(
                new_chunk_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=int(self.embed_batch_size),
            )
            emb = emb.astype(np.float32)
            emb = self._normalize(emb)

            progress("Adding to index…")
            ids = np.array([c.chunk_id for c in new_chunks], dtype=np.int64)
            index.add_with_ids(emb, ids)

            # Update chunks store: keep old chunks list (for source display) + append new
            # Note: removed chunks remain in chunks.json, but retrieval filters by active ids + FAISS remove.
            chunks.extend(new_chunks)

        # Update manifest metadata for changed/new files
        progress("Updating manifest…")
        for p in changed_paths:
            pp = Path(p)
            if not pp.exists():
                continue

            fp = self._file_stat_fingerprint(pp)
            meta = files_meta.get(p) or {}
            meta.update(fp)
            meta["active"] = True
            meta["chunk_ids"] = per_file_chunk_ids.get(p, [])
            meta["doc_id"] = per_file_doc_ids.get(p) or meta.get("doc_id") or self._doc_id_for_source(p)
            meta["last_indexed_at"] = datetime.now().isoformat(timespec="seconds")

            # store extractor report snippet if available
            rep = next((r for r in report if r.source == p), None)
            if rep:
                meta["method"] = rep.method
                meta["chars"] = rep.chars
                meta["warning"] = rep.warning
                meta["error"] = rep.error

            files_meta[p] = meta

        # Mark deleted paths inactive (keep record)
        for p in deleted_paths:
            meta = files_meta.get(p) or {}
            meta["active"] = False
            meta["deleted_at"] = datetime.now().isoformat(timespec="seconds")
            meta["chunk_ids"] = []
            if not meta.get("doc_id"):
                meta["doc_id"] = self._doc_id_for_source(p)
            files_meta[p] = meta

        manifest["files"] = files_meta
        manifest["next_chunk_id"] = next_chunk_id
        self._save_manifest(manifest)

        # Save docs list = current selected list (not only active)
        progress("Saving index…")
        self._save_index(index, chunks, file_paths)

        # refresh in-memory
        self._faiss = index
        self._chunks = chunks
        self._active_ids = None  # will rebuild on load_index() if needed
        # refresh chunk map for fast retrieval
        self._chunk_map = {c.chunk_id: c for c in chunks}

        progress("Index ready.")

        return {
            "docs": len(file_paths),
            "indexed_docs": len([p for p in file_paths if (files_meta.get(p) or {}).get("active", True)]),
            "attempted_docs": attempted,
            "failed_docs": failed,
            "chunks": len(new_chunks),
            "report": [r.to_dict() if isinstance(r, LoadReportEntry) else r for r in report],
            "settings": {
                "enable_ocr": enable_ocr,
                "ocr_lang": ocr_lang,
                "enable_libreoffice": enable_libreoffice,
                "incremental": True,
                "deleted_files": len(deleted_paths),
                "changed_files": len(changed_paths),
            },
        }

    def retrieve(self, query: str, top_k: int = 6) -> List[Tuple[ChunkLite, float]]:
        self._load_embedder()
        self._load_index()

        q = self._embedder.encode([query], show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        q = self._normalize(q)

        # Overfetch a bit because we may filter by active ids
        overfetch = max(top_k * 3, 20)
        scores, ids = self._faiss.search(q, overfetch)

        # Build chunk_id -> metadata index (fast)
        # Use cached map to avoid rebuilding on every query
        chunk_map: Dict[int, ChunkLite] = self._chunk_map or {}
        # Fallback in case chunk_map is missing
        if not chunk_map and self._chunks:
            chunk_map = {c.chunk_id: c for c in self._chunks}
            self._chunk_map = chunk_map

        results: List[Tuple[ChunkLite, float]] = []
        active = self._active_ids

        for cid, score in zip(ids[0].tolist(), scores[0].tolist()):
            if cid < 0:
                continue
            if score < self.retrieval_min_score:
                continue
            if active is not None and cid not in active:
                continue
            meta = chunk_map.get(int(cid))
            if not meta:
                continue
            results.append((meta, float(score)))
            if len(results) >= top_k:
                break

        return results

    # ---------------- tools (time) ----------------
    def tool_time_string(self) -> str:
        ini = self.load_ini()
        time_format = ini.get("preferences", {}).get("time_format", "24h").lower()

        now = datetime.now()
        if time_format == "12h":
            return now.strftime("%I:%M %p").lstrip("0")
        return now.strftime("%H:%M")

    def _looks_like_time_question(self, q: str) -> bool:
        ql = q.lower()
        return any(
            s in ql
            for s in [
                "what time is it",
                "tell me the time",
                "current time",
                "time now",
                "现在几点",
                "几点了",
                "وقت",
            ]
        )

    def _looks_like_doc_count_question(self, q: str) -> bool:
        """
        Detect queries asking how many files or documents the user has.
        We check for common phrases like "how many files do i have" or "how many documents".
        """
        ql = q.lower().strip()
        # Normalize punctuation
        ql = ql.replace("?", "").replace(".", "").replace(",", "")
        patterns = [
            "how many files do i have",
            "how many documents do i have",
            "how many docs do i have",
            "how many files",
            "how many documents",
            "how many docs",
        ]
        return any(phrase in ql for phrase in patterns)

    # ---------------- prompting ----------------
    def _build_prompt(
        self,
        question: str,
        retrieved: List[Tuple[ChunkLite, float]],
        ini: Dict[str, Dict[str, str]],
        chat_summary: str,
        recent_chat: List[Dict[str, str]],
        *,
        allow_control: bool = False,
    ) -> Tuple[str, List[str]]:
        ctx_lines = []
        sources = []
        for i, (chunk, score) in enumerate(retrieved, start=1):
            full_text = self._resolve_chunk_text(chunk) or chunk.summary
            ctx_lines.append(f"[{i}] Source: {Path(chunk.source).name}\n{full_text}\n")
            sources.append(f"[{i}] {Path(chunk.source).name}#chunk_{chunk.chunk_id} score={score:.3f}")

        context = "\n".join(ctx_lines).strip()

        user_name = ini.get("user", {}).get("name", "User")
        assistant_name = ini.get("assistant", {}).get("name", "Assistant")
        tone = ini.get("assistant", {}).get("tone", "professional_friendly")
        personality = ini.get("assistant", {}).get(
            "personality",
            "Helpful, concise, accurate. Uses citations when possible.",
        )
        custom_rules = ini.get("assistant", {}).get("custom_rules", "").strip()
        prompt_template = ini.get("assistant", {}).get("system_prompt_template", "").strip()
        agent_cfg = ini.get("agent", {})
        agent_enable_control = agent_cfg.get("enable_control", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        control_syntax = agent_cfg.get("control_block_syntax", "<CONTROL>...</CONTROL>").strip()

        pref_lang = ini.get("user", {}).get("language", "auto").lower()
        language_rule = (
            "Answer language:\n"
            "- If preference is 'en', answer in English.\n"
            "- If preference is 'zh', answer in Chinese.\n"
            "- If preference is 'th', answer in Thai.\n"
            "- If preference is 'auto', answer in the same language as the user's question.\n"
            "- even if the document is in other language. answer in user's preferred language. if require translate to user's preferred language.\n"
        )

        chat_snips = []
        for m in recent_chat:
            role = m.get("role", "")
            text = m.get("text", "")
            if not text:
                continue
            if role == "user":
                chat_snips.append(f"{user_name}: {text}")
            else:
                chat_snips.append(f"{assistant_name}: {text}")
        recent_chat_block = "\n".join(chat_snips[-10:]).strip()

        system_rules = (
            f"You are {assistant_name}, a document question-answering tool.\n"
            f"Tone: {tone}. Style: {personality}\n"
            f"User name: {user_name}\n\n"
            "HARD RULES (follow exactly):\n"
            "1) Use ONLY the provided CONTEXT to answer. If the answer is not in CONTEXT, say you cannot find it in the documents.\n"
            "2) Do NOT use outside knowledge. Do NOT guess.\n"
            "3) Do NOT mention AI, LLMs, system prompts, or hidden rules.\n"
            "4) Keep answers short and clear. Prefer bullets for steps.\n"
            "5) If the question is ambiguous, ask ONE short clarification question.\n"
            "6) Follow the user's language preference.\n\n"
            "CITATIONS (NORMAL MODE ONLY):\n"
            "- When you use information from CONTEXT, add citations like [1], [2] at the end of the sentence.\n"
            "- At the end, add a short 'Sources:' line listing file names used.\n"
            "- If you did not use CONTEXT, do not add citations.\n\n"
            "TABLE MODE (VERY STRICT):\n"
            "- Enter TABLE MODE when asks for as well as if its going to look better for user : CSV, export, spreadsheet, table, Excel. or anything that might be easier to explain in table format.\n"
            "- In TABLE MODE your output MUST be exactly:\n"
            "  (a) First line: TABLE_MODE\n"
            "  (b) Then a Markdown pipe table with header row and separator row.\n"
            "  (c) No other text before or after. No bullets. No explanation. No 'Sources' line. No citations.\n"
            "- If required fields are missing for a table, ask ONE clarification question IN NORMAL MODE.\n\n"
            "QUALITY RULES:\n"
            "- Prefer extracting structured fields from CONTEXT (names, dates, amounts, statuses).\n"
            "- For procedures: provide numbered steps.\n"
            "- For error codes mentioned by the user: give a likely cause + fix steps.\n"
            "- If user asks to translate and language is English: translate and add: Translated.\n"
            "- if user asks for any help or prompt anyhting. if its going to be easier for user to view them as table format. give in table format.\n"
            "- any data or information that is coming from an xlsx or csv or any table format. give in table format.\n"
            "- if users asks you to translate someting translate it properly suppose user says translate it to chinese you should translate what you responded in chinese , not the documnets. unless user asks for document translation.\n"
        )

        if custom_rules:
            system_rules += f"\nCUSTOM RULES:\n{custom_rules}\n"

        if allow_control and agent_enable_control:
            system_rules += (
                "\nCONTROL ACTIONS:\n"
                "- If the user asks you to operate the app (settings, sources, indexing), decide whether to take control and emit control actions.\n"
                "- If you need to change config or indexing, respond with ONLY a control block and no answer.\n"
                "- Control block syntax must be exactly:\n"
                "  <CONTROL>\n"
                "  SET section.key=value\n"
                "  ADD_SOURCE \"path\"\n"
                "  REMOVE_SOURCE \"path\"\n"
                "  REBUILD_INDEX\n"
                "  CLEAR_INDEX\n"
                "  RELOAD_CONFIG\n"
                "  LIST_SOURCES\n"
                "  RESCAN_DISK\n"
                "  </CONTROL>\n"
                "- If no actions are needed, answer normally and do NOT include a control block.\n"
                f"- Declared control syntax: {control_syntax}\n"
            )

        system_rules += f"\nUser language preference: {pref_lang}\n{language_rule}\n"

        if prompt_template:
            payload = {
                "assistant_name": assistant_name,
                "user_name": user_name,
                "tone": tone,
                "personality": personality,
                "custom_rules": custom_rules,
                "pref_lang": pref_lang,
                "language_rule": language_rule,
                "system_rules": system_rules,
                "chat_summary": chat_summary,
                "recent_chat": recent_chat_block,
                "context": context,
                "question": question,
            }
            try:
                prompt = prompt_template.format(**payload)
            except Exception:
                prompt = (
                    f"{system_rules}\n"
                    f"MEMORY SUMMARY (facts about the user / preferences):\n{chat_summary}\n\n"
                    f"RECENT CHAT:\n{recent_chat_block}\n\n"
                    f"CONTEXT (documents):\n{context}\n\n"
                    f"QUESTION: {question}\n"
                    "ANSWER:"
                )
        else:
            prompt = (
                f"{system_rules}\n"
                f"MEMORY SUMMARY (facts about the user / preferences):\n{chat_summary}\n\n"
                f"RECENT CHAT:\n{recent_chat_block}\n\n"
                f"CONTEXT (documents):\n{context}\n\n"
                f"QUESTION: {question}\n"
                "ANSWER:"
            )
        return prompt, sources

    @staticmethod
    def _detect_auto_language(text: str) -> str:
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[\u0E00-\u0E7F]", text):
            return "th"
        return "en"

    def _no_context_message(self, question: str, ini: Dict[str, Dict[str, str]]) -> str:
        pref_lang = ini.get("user", {}).get("language", "auto").lower()
        lang = pref_lang if pref_lang in {"en", "zh", "th"} else self._detect_auto_language(question)
        if lang == "zh":
            return "我在已索引的文档中找不到相关内容。请确认已添加正确的文件/文件夹并重新索引。"
        if lang == "th":
            return "ไม่พบข้อมูลในเอกสารที่จัดทำดัชนี โปรดเพิ่มไฟล์/โฟลเดอร์ที่ถูกต้องและทำการสร้างดัชนีใหม่."
        return "I couldn't find this in the indexed documents. Please add the right files/folders and reindex."

    # ---------------- continuation helpers ----------------
    @staticmethod
    def _tail(text: str, n_chars: int = 700) -> str:
        t = (text or "").strip()
        if len(t) <= n_chars:
            return t
        return t[-n_chars:]

    @staticmethod
    def _looks_truncated(text: str) -> bool:
        t = (text or "").rstrip()
        if not t:
            return False
        if t.count("```") % 2 == 1:
            return True
        if t.endswith("|") and "\n|" in t[-200:]:
            return True
        if re.search(r"[\w\u4e00-\u9fff]\s*$", t) and not re.search(r"[。！？.!?]\s*$", t):
            if len(t) > 80:
                return True
        if t.endswith(":") or t.endswith("-") or t.endswith("—"):
            return True
        return False

    def _build_continue_prompt(self, base_prompt: str, partial_answer: str) -> str:
        tail = self._tail(partial_answer, 700)
        return (
            base_prompt
            + "\n\n"
            + "The answer above may be incomplete due to output limits.\n"
            + "Continue the ANSWER from exactly where it stopped.\n"
            + "Rules:\n"
            + "- Do NOT repeat earlier text.\n"
            + "- Do NOT restart the answer.\n"
            + "- Keep the same format.\n"
            + "- If you were in TABLE MODE, continue the same table correctly.\n"
            + f"\nLast part of your answer (for reference, do not repeat):\n{tail}\n"
            + "\nANSWER (continue):"
        )

    # ---------------- Ollama generate ----------------
    def _ollama_generate(self, prompt: str, num_predict: Optional[int] = None) -> str:
        url = f"{self.ollama_base_url}/api/generate"
        if num_predict is None:
            num_predict = self.generation_num_predict
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(self.generation_temperature),
                "num_predict": int(num_predict),
                "top_p": float(self.generation_top_p),
            },
        }
        r = requests.post(url, json=payload, timeout=int(self.ollama_timeout_s))
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    def _ollama_generate_stream(
        self,
        prompt: str,
        on_token: Callable[[str], None],
        num_predict: Optional[int] = None,
    ) -> Tuple[str, str]:
        url = f"{self.ollama_base_url}/api/generate"
        if num_predict is None:
            num_predict = self.generation_num_predict
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": float(self.generation_temperature),
                "num_predict": int(num_predict),
                "top_p": float(self.generation_top_p),
            },
        }

        parts: List[str] = []
        done_reason = ""
        with requests.post(url, json=payload, stream=True, timeout=int(self.ollama_timeout_s)) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                tok = obj.get("response") or ""
                if tok:
                    parts.append(tok)
                    on_token(tok)
                if obj.get("done"):
                    done_reason = str(obj.get("done_reason") or "")
                    break

        return "".join(parts).strip(), done_reason

    # ---------------- learning (safe) ----------------
    def _extract_memory_updates(self, user_text: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        t = user_text.strip()

        m = re.search(r"\b(my name is|call me)\s+([A-Za-z][A-Za-z0-9 _-]{1,40})\b", t, flags=re.I)
        if m:
            out.append(("user.name", m.group(2).strip()))

        if re.search(r"\b(speak in chinese|use chinese|中文回答|用中文)\b", t, flags=re.I):
            out.append(("user.language", "zh"))
        if re.search(r"\b(speak in english|use english|英文回答|用英文)\b", t, flags=re.I):
            out.append(("user.language", "en"))

        if re.search(r"\b(short answers|be concise|简短回答|少说点)\b", t, flags=re.I):
            out.append(("assistant.tone", "concise"))
        if re.search(r"\b(more detail|be detailed|详细一点)\b", t, flags=re.I):
            out.append(("assistant.tone", "detailed"))

        return out

    def _apply_memory_updates(self, updates: List[Tuple[str, str]]) -> None:
        if not updates:
            return
        ini = self.load_ini()
        for key, val in updates:
            section, k = key.split(".", 1)
            ini.setdefault(section, {})
            ini[section][k] = val
        self.save_ini(ini)

    def _update_chat_summary(self, user_text: str, assistant_text: str) -> None:
        summary = self.get_chat_summary()
        lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
        if not lines:
            lines = ["Conversation summary (short, important facts only):"]

        updates = self._extract_memory_updates(user_text)
        for k, v in updates:
            pretty = f"- {k} = {v}"
            if pretty not in lines:
                lines.append(pretty)

        if len(lines) > 18:
            lines = lines[:1] + lines[-17:]

        self.set_chat_summary("\n".join(lines))

    def record_interaction(self, question: str, answer: str) -> None:
        self.append_chat_log("user", question)
        self.append_chat_log("assistant", answer)
        updates = self._extract_memory_updates(question)
        self._apply_memory_updates(updates)
        self._update_chat_summary(question, answer)

    # ---------------- main answering ----------------
    def generate_answer(
        self,
        question: str,
        progress: Callable[[str], None] = lambda _: None,
        top_k: int = 60,
        *,
        allow_control: bool = False,
    ) -> Dict[str, object]:
        # Handle special queries (time / file count) before retrieval
        if self._looks_like_time_question(question):
            t = self.tool_time_string()
            ans = f"The current local time is **{t}**."
            return {"answer": ans, "sources": []}

        if self._looks_like_doc_count_question(question):
            docs = self.get_indexed_docs()
            n = len(docs)
            header = f"You have {n} document" + ("s" if n != 1 else "") + "."
            if n > 0:
                lines = [f"{i}. {Path(d).name}" for i, d in enumerate(docs, 1)]
                answer = header + "\n" + "\n".join(lines)
            else:
                answer = header
            return {"answer": answer, "sources": []}

        progress("Retrieving…")
        retrieved = self.retrieve(question, top_k=top_k)

        ini = self.load_ini()
        if not retrieved and not allow_control:
            answer = self._no_context_message(question, ini)
            return {"answer": answer, "sources": []}

        summary = self.get_chat_summary()
        recent = self.load_recent_chat(max_messages=10)
        if not retrieved and not recent and not allow_control:
            answer = self._no_context_message(question, ini)
            return {"answer": answer, "sources": []}

        prompt, sources = self._build_prompt(
            question,
            retrieved,
            ini,
            summary,
            recent,
            allow_control=allow_control,
        )

        progress("Generating…")
        answer = self._ollama_generate(prompt)
        if not answer:
            answer = "I don’t know from the provided documents."

        return {"answer": answer, "sources": sources}

    def answer_question(
        self,
        question: str,
        progress: Callable[[str], None] = lambda _: None,
        top_k: int = 60,
    ) -> Dict[str, object]:
        res = self.generate_answer(question, progress=progress, top_k=top_k, allow_control=False)
        self.record_interaction(question, res.get("answer", ""))
        return res

    def answer_question_stream(
        self,
        question: str,
        progress: Callable[[str], None] = lambda _: None,
        top_k: int = 60,
        on_token: Callable[[str], None] = lambda _: None,
        max_continuations: int = 4,
        num_predict_each: int = 750,
    ) -> Dict[str, object]:
        # Handle special queries (time / file count) before retrieval
        if self._looks_like_time_question(question):
            t = self.tool_time_string()
            ans = f"The current local time is **{t}**."
            on_token(ans)
            self.append_chat_log("user", question)
            self.append_chat_log("assistant", ans)
            self._update_chat_summary(question, ans)
            return {"answer": ans, "sources": []}

        if self._looks_like_doc_count_question(question):
            # Return the exact number of indexed files and list them
            docs = self.get_indexed_docs()
            n = len(docs)
            header = f"You have {n} document" + ("s" if n != 1 else "") + "."
            if n > 0:
                lines = [f"{i}. {Path(d).name}" for i, d in enumerate(docs, 1)]
                answer = header + "\n" + "\n".join(lines)
            else:
                answer = header
            # Stream the answer as a whole once
            on_token(answer)
            self.append_chat_log("user", question)
            self.append_chat_log("assistant", answer)
            self._update_chat_summary(question, answer)
            return {"answer": answer, "sources": []}

        progress("Retrieving…")
        retrieved = self.retrieve(question, top_k=top_k)

        ini = self.load_ini()
        if not retrieved:
            answer = self._no_context_message(question, ini)
            on_token(answer)
            self.append_chat_log("user", question)
            self.append_chat_log("assistant", answer)
            self._update_chat_summary(question, answer)
            return {"answer": answer, "sources": []}

        summary = self.get_chat_summary()
        recent = self.load_recent_chat(max_messages=10)
        if not retrieved and not recent:
            answer = self._no_context_message(question, ini)
            on_token(answer)
            self.append_chat_log("user", question)
            self.append_chat_log("assistant", answer)
            self._update_chat_summary(question, answer)
            return {"answer": answer, "sources": []}

        base_prompt, sources = self._build_prompt(
            question,
            retrieved,
            ini,
            summary,
            recent,
            allow_control=False,
        )

        progress("Generating…")
        full_parts: List[str] = []

        chunk_text, done_reason = self._ollama_generate_stream(
            base_prompt,
            on_token=on_token,
            num_predict=num_predict_each,
        )
        if chunk_text:
            full_parts.append(chunk_text)

        for i in range(int(max_continuations)):
            current = "".join(full_parts).strip()
            needs_more = (done_reason.lower() == "length") or self._looks_truncated(current)
            if not needs_more:
                break

            progress(f"Continuing… ({i+1}/{max_continuations})")
            cont_prompt = self._build_continue_prompt(base_prompt, current)

            more_text, done_reason = self._ollama_generate_stream(
                cont_prompt,
                on_token=on_token,
                num_predict=num_predict_each,
            )
            if not more_text:
                break
            full_parts.append(more_text)

        answer = "".join(full_parts).strip()
        if not answer:
            answer = "I don’t know from the provided documents."

        self.append_chat_log("user", question)
        self.append_chat_log("assistant", answer)

        updates = self._extract_memory_updates(question)
        self._apply_memory_updates(updates)
        self._update_chat_summary(question, answer)

        return {"answer": answer, "sources": sources}
