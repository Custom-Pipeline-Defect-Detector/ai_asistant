import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from doc_loaders import load_documents


@dataclass
class ChunkMetadata:
    chunk_id: int
    source: str
    text: str


class RAGEngineError(Exception):
    pass


class RAGEngine:
    """
    RAG + Ollama LLM + Persistent Memory.
    Memory is stored in:
      D:\\offline_doc_assistant\\data\\memory\\assistant.ini
      D:\\offline_doc_assistant\\data\\memory\\chat_history.jsonl
      D:\\offline_doc_assistant\\data\\memory\\chat_summary.txt
    """

    def __init__(
        self,
        base_dir: Path,
        embed_model_dir: Path,
        index_dir: Path,
        log_dir: Path,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b-instruct",
    ):
        self.base_dir = base_dir
        self.embed_model_dir = embed_model_dir
        self.index_dir = index_dir
        self.log_dir = log_dir

        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model

        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "chunks.json"
        self.docs_path = self.index_dir / "docs.json"

        self._embedder: Optional[SentenceTransformer] = None
        self._faiss: Optional[faiss.IndexFlatIP] = None
        self._chunks: List[ChunkMetadata] = []

        # Persistent memory
        self.memory_dir = self.base_dir / "data" / "memory"
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
        """
        INI holds assistant persona + user prefs.
        Summary holds rolling "what matters" memory.
        Chat log holds raw conversation lines.
        """
        if not self.ini_path.exists():
            self.ini_path.write_text(
                "\n".join(
                    [
                        "[user]",
                        "name=Mir",
                        "language=auto",
                        "",
                        "[assistant]",
                        "name=Assistant",
                        "tone=professional_friendly",
                        "personality=Helpful, concise, accurate. Uses citations when possible. Asks follow-ups only if necessary.",
                        "custom_rules=",
                        "",
                        "[preferences]",
                        "time_format=24h",
                        "timezone=local",
                        "auto_save_facts=true",
                    ]
                ),
                encoding="utf-8",
            )

        if not self.chat_summary_path.exists():
            self.chat_summary_path.write_text(
                "Conversation summary (short, important facts only):\n",
                encoding="utf-8",
            )

        if not self.chat_log_path.exists():
            self.chat_log_path.write_text("", encoding="utf-8")

    def load_ini(self) -> Dict[str, Dict[str, str]]:
        """
        Minimal INI parser (no extra deps).
        """
        data: Dict[str, Dict[str, str]] = {}
        current = None
        for line in self.ini_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip().lower()
                data.setdefault(current, {})
                continue
            if "=" in line and current:
                k, v = line.split("=", 1)
                data[current][k.strip().lower()] = v.strip()
        return data

    def save_ini(self, data: Dict[str, Dict[str, str]]) -> None:
        lines = []
        for section, kv in data.items():
            lines.append(f"[{section}]")
            for k, v in kv.items():
                lines.append(f"{k}={v}")
            lines.append("")
        self.ini_path.write_text("\n".join(lines), encoding="utf-8")

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
        """
        Clears chat memory only (NOT the FAISS index).
        """
        self.chat_log_path.write_text("", encoding="utf-8")
        self.chat_summary_path.write_text("Conversation summary (short, important facts only):\n", encoding="utf-8")

    # ---------------- embed/index ----------------
    def _load_embedder(self):
        if self._embedder is None:
            if not self.embed_model_dir.exists():
                raise RAGEngineError(f"Embedding model folder missing: {self.embed_model_dir}")
            self._embedder = SentenceTransformer(str(self.embed_model_dir))

    def index_exists(self) -> bool:
        return self.index_path.exists() and self.chunks_path.exists() and self.docs_path.exists()

    def get_indexed_docs(self) -> List[str]:
        if not self.docs_path.exists():
            return []
        try:
            return json.loads(self.docs_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def clear_index(self) -> None:
        for p in [self.index_path, self.chunks_path, self.docs_path]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        self._faiss = None
        self._chunks = []

    def _save_index(self, index: faiss.IndexFlatIP, chunks: List[ChunkMetadata], docs: List[str]):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self.chunks_path.write_text(
            json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.docs_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index(self):
        if self._faiss is not None and self._chunks:
            return
        if not self.index_exists():
            raise RAGEngineError("Index not found. Please build the index first.")
        self._faiss = faiss.read_index(str(self.index_path))
        raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self._chunks = [ChunkMetadata(**c) for c in raw]

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1100, overlap: int = 220) -> List[str]:
        text = text.replace("\r\n", "\n")
        if not text.strip():
            return []
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

    def build_index(
        self,
        file_paths: List[str],
        progress: Callable[[str], None] = lambda _: None,
        chunk_size: int = 1100,
        overlap: int = 220,
    ) -> Dict[str, int]:
        self._load_embedder()

        progress("Loading documents…")
        docs = load_documents([Path(p) for p in file_paths])

        if not docs:
            raise RAGEngineError(
                "No text could be extracted.\n"
                "If PDFs are scanned images, OCR is required (not enabled).\n"
                "If indexing Excel/PPT/HTML ensure deps are installed."
            )

        all_chunks: List[ChunkMetadata] = []
        for doc_i, d in enumerate(docs, start=1):
            progress(f"Chunking ({doc_i}/{len(docs)})…")
            for ch in self._chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap):
                all_chunks.append(ChunkMetadata(chunk_id=len(all_chunks), source=d["source"], text=ch))

        if not all_chunks:
            raise RAGEngineError("No text chunks could be created.")

        progress("Embedding…")
        texts = [c.text for c in all_chunks]
        emb = self._embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        emb = emb.astype(np.float32)
        emb = self._normalize(emb)

        progress("Building index…")
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        progress("Saving…")
        self._save_index(index, all_chunks, file_paths)

        self._faiss = index
        self._chunks = all_chunks

        progress("Index ready.")
        return {"docs": len(file_paths), "chunks": len(all_chunks)}

    def retrieve(self, query: str, top_k: int = 6) -> List[Tuple[ChunkMetadata, float]]:
        self._load_embedder()
        self._load_index()

        q = self._embedder.encode([query], show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        q = self._normalize(q)

        scores, ids = self._faiss.search(q, top_k)
        results: List[Tuple[ChunkMetadata, float]] = []
        for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append((self._chunks[idx], float(score)))
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

    # ---------------- prompting ----------------
    @staticmethod
    def _escape_html(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _build_prompt(
        self,
        question: str,
        retrieved: List[Tuple[ChunkMetadata, float]],
        ini: Dict[str, Dict[str, str]],
        chat_summary: str,
        recent_chat: List[Dict[str, str]],
    ) -> Tuple[str, List[str]]:
        # Context pack
        ctx_lines = []
        sources = []
        for i, (chunk, score) in enumerate(retrieved, start=1):
            ctx_lines.append(f"[{i}] Source: {Path(chunk.source).name}\n{chunk.text}\n")
            sources.append(f"[{i}] {Path(chunk.source).name}#chunk_{chunk.chunk_id} score={score:.3f}")

        context = "\n".join(ctx_lines).strip()

        # Persona
        user_name = ini.get("user", {}).get("name", "User")
        assistant_name = ini.get("assistant", {}).get("name", "Assistant")
        tone = ini.get("assistant", {}).get("tone", "professional_friendly")
        personality = ini.get("assistant", {}).get(
            "personality",
            "Helpful, concise, accurate. Uses citations when possible.",
        )
        custom_rules = ini.get("assistant", {}).get("custom_rules", "").strip()

        # Recent chat (short)
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

        # Hard rules (robustness)
        system_rules = (
            f"You are {assistant_name}, an offline document assistant.\n"
            f"Tone: {tone}. Personality: {personality}\n"
            f"User name: {user_name}\n\n"
            "RULES (must follow):\n"
            "1) If the user asks for facts not supported by the provided CONTEXT, say: I don’t know from the provided documents.\n"
            "2) When you use context, cite sources like [1], [2].\n"
            "3) Be concise. Prefer bullet points for procedures.\n"
            "4) If the question is ambiguous, ask ONE short clarification.\n"
        )
        if custom_rules:
            system_rules += f"\nCUSTOM RULES:\n{custom_rules}\n"

        prompt = (
            f"{system_rules}\n"
            f"MEMORY SUMMARY (facts about the user / preferences):\n{chat_summary}\n\n"
            f"RECENT CHAT:\n{recent_chat_block}\n\n"
            f"CONTEXT (documents):\n{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )
        return prompt, sources

    def _ollama_generate(self, prompt: str) -> str:
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 650,
                "top_p": 0.9,
            },
        }
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    # ---------------- learning (safe) ----------------
    def _extract_memory_updates(self, user_text: str) -> List[Tuple[str, str]]:
        """
        Very conservative 'learning':
        only saves explicit preferences or identity the user tells the assistant.
        Example:
          "Call me Mir"
          "My name is Mir"
          "Speak in Chinese"
          "Use short answers"
        """
        out: List[Tuple[str, str]] = []
        t = user_text.strip()

        # name
        m = re.search(r"\b(my name is|call me)\s+([A-Za-z][A-Za-z0-9 _-]{1,40})\b", t, flags=re.I)
        if m:
            out.append(("user.name", m.group(2).strip()))

        # language preference
        if re.search(r"\b(speak in chinese|use chinese|中文回答|用中文)\b", t, flags=re.I):
            out.append(("user.language", "zh"))
        if re.search(r"\b(speak in english|use english|英文回答|用英文)\b", t, flags=re.I):
            out.append(("user.language", "en"))

        # brevity
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
        """
        Lightweight summary growth (fast, offline):
        - Keeps only a small number of lines
        - Adds only preference-like statements detected from user
        """
        summary = self.get_chat_summary()
        lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
        if not lines:
            lines = ["Conversation summary (short, important facts only):"]

        updates = self._extract_memory_updates(user_text)
        for k, v in updates:
            pretty = f"- {k} = {v}"
            if pretty not in lines:
                lines.append(pretty)

        # keep it short
        if len(lines) > 18:
            lines = lines[:1] + lines[-17:]

        self.set_chat_summary("\n".join(lines))

    # ---------------- main answering ----------------
    def answer_question(
        self,
        question: str,
        progress: Callable[[str], None] = lambda _: None,
        top_k: int = 6,
    ) -> Dict[str, object]:
        # time tool shortcut
        if self._looks_like_time_question(question):
            t = self.tool_time_string()
            ans = f"The current local time is **{t}**."
            self.append_chat_log("user", question)
            self.append_chat_log("assistant", ans)
            self._update_chat_summary(question, ans)
            return {"answer": ans, "sources": []}

        # normal RAG
        progress("Retrieving…")
        retrieved = self.retrieve(question, top_k=top_k)

        ini = self.load_ini()
        summary = self.get_chat_summary()
        recent = self.load_recent_chat(max_messages=10)

        prompt, sources = self._build_prompt(question, retrieved, ini, summary, recent)

        progress("Generating…")
        answer = self._ollama_generate(prompt)
        if not answer:
            answer = "I don’t know from the provided documents."

        # save memory + chat log
        self.append_chat_log("user", question)
        self.append_chat_log("assistant", answer)

        # safe learning (only explicit preferences)
        updates = self._extract_memory_updates(question)
        self._apply_memory_updates(updates)
        self._update_chat_summary(question, answer)

        return {"answer": answer, "sources": sources}
