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
    def _build_prompt(
        self,
        question: str,
        retrieved: List[Tuple[ChunkMetadata, float]],
        ini: Dict[str, Dict[str, str]],
        chat_summary: str,
        recent_chat: List[Dict[str, str]],
    ) -> Tuple[str, List[str]]:
        ctx_lines = []
        sources = []
        for i, (chunk, score) in enumerate(retrieved, start=1):
            ctx_lines.append(f"[{i}] Source: {Path(chunk.source).name}\n{chunk.text}\n")
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

        pref_lang = ini.get("user", {}).get("language", "auto").lower()
        language_rule = (
            "Answer language:\n"
            "- If preference is 'en', answer in English.\n"
            "- If preference is 'zh', answer in Chinese.\n"
            "- If preference is 'th', answer in Thai.\n"
            "- If preference is 'auto', answer in the same language as the user's question.\n"
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
            "1) Use ONLY the provided CONTEXT to answer. If the answer is not in CONTEXT, try to find near best match.\n"
            " \n"
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
            "- Enter TABLE MODE when asks for as well as if its going to look better for user : CSV, export, spreadsheet, table, Excel. or anything that might be easier to explain in table format. \n"
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

        system_rules += f"\nUser language preference: {pref_lang}\n{language_rule}\n"

        prompt = (
            f"{system_rules}\n"
            f"MEMORY SUMMARY (facts about the user / preferences):\n{chat_summary}\n\n"
            f"RECENT CHAT:\n{recent_chat_block}\n\n"
            f"CONTEXT (documents):\n{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )
        return prompt, sources

    # ---------------- continuation helpers ----------------
    @staticmethod
    def _tail(text: str, n_chars: int = 700) -> str:
        t = (text or "").strip()
        if len(t) <= n_chars:
            return t
        return t[-n_chars:]

    @staticmethod
    def _looks_truncated(text: str) -> bool:
        """
        Heuristics: if output ends in a suspicious way, treat as incomplete.
        """
        t = (text or "").rstrip()
        if not t:
            return False

        # Unclosed code fence
        if t.count("```") % 2 == 1:
            return True

        # Unclosed markdown table: ends with '|' but no closing row context
        if t.endswith("|") and "\n|" in t[-200:]:
            return True

        # Ends mid-sentence / mid-list marker (common truncation)
        if re.search(r"[\w\u4e00-\u9fff]\s*$", t) and not re.search(r"[。！？.!?]\s*$", t):
            # if it ends with a letter/hanzi and no terminal punctuation, likely cut
            if len(t) > 80:
                return True

        # Ends with colon or dash
        if t.endswith(":") or t.endswith("-") or t.endswith("—"):
            return True

        return False

    def _build_continue_prompt(self, base_prompt: str, partial_answer: str) -> str:
        """
        Build a continuation instruction while keeping the same system/context rules.
        We reuse the original prompt (system + context + QUESTION) and add a continuation directive.
        """
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

    # ---------------- Ollama generate (stream + non-stream) ----------------
    def _ollama_generate(self, prompt: str, num_predict: int = 650) -> str:
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": int(num_predict),
                "top_p": 0.9,
            },
        }
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    def _ollama_generate_stream(
        self,
        prompt: str,
        on_token: Callable[[str], None],
        num_predict: int = 650,
    ) -> Tuple[str, str]:
        """
        Streams tokens from Ollama and calls on_token(token) as text arrives.
        Returns (full_text, done_reason).
        """
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": int(num_predict),
                "top_p": 0.9,
            },
        }

        parts: List[str] = []
        done_reason = ""
        with requests.post(url, json=payload, stream=True, timeout=600) as r:
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

    # ---------------- main answering ----------------
    def answer_question(
        self,
        question: str,
        progress: Callable[[str], None] = lambda _: None,
        top_k: int = 6,
    ) -> Dict[str, object]:
        if self._looks_like_time_question(question):
            t = self.tool_time_string()
            ans = f"The current local time is **{t}**."
            self.append_chat_log("user", question)
            self.append_chat_log("assistant", ans)
            self._update_chat_summary(question, ans)
            return {"answer": ans, "sources": []}

        progress("Retrieving…")
        retrieved = self.retrieve(question, top_k=top_k)

        ini = self.load_ini()
        summary = self.get_chat_summary()
        recent = self.load_recent_chat(max_messages=10)

        prompt, sources = self._build_prompt(question, retrieved, ini, summary, recent)

        progress("Generating…")
        answer = self._ollama_generate(prompt, num_predict=900)
        if not answer:
            answer = "I don’t know from the provided documents."

        self.append_chat_log("user", question)
        self.append_chat_log("assistant", answer)

        updates = self._extract_memory_updates(question)
        self._apply_memory_updates(updates)
        self._update_chat_summary(question, answer)

        return {"answer": answer, "sources": sources}

    def answer_question_stream(
        self,
        question: str,
        progress: Callable[[str], None] = lambda _: None,
        top_k: int = 6,
        on_token: Callable[[str], None] = lambda _: None,
        max_continuations: int = 4,
        num_predict_each: int = 750,
    ) -> Dict[str, object]:
        """
        Streams tokens via on_token(token).
        Improvement: auto-continue if Ollama stops due to length/truncation.
        """
        if self._looks_like_time_question(question):
            t = self.tool_time_string()
            ans = f"The current local time is **{t}**."
            on_token(ans)
            self.append_chat_log("user", question)
            self.append_chat_log("assistant", ans)
            self._update_chat_summary(question, ans)
            return {"answer": ans, "sources": []}

        progress("Retrieving…")
        retrieved = self.retrieve(question, top_k=top_k)

        ini = self.load_ini()
        summary = self.get_chat_summary()
        recent = self.load_recent_chat(max_messages=10)

        base_prompt, sources = self._build_prompt(question, retrieved, ini, summary, recent)

        progress("Generating…")
        full_parts: List[str] = []

        # First generation
        chunk_text, done_reason = self._ollama_generate_stream(
            base_prompt,
            on_token=on_token,
            num_predict=num_predict_each,
        )
        if chunk_text:
            full_parts.append(chunk_text)

        # Auto-continue loop
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
