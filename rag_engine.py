import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from doc_loaders import load_documents


@dataclass
class ChunkMetadata:
    chunk_id: int
    source: str
    text: str


class RAGEngineError(Exception):
    pass


class RAGEngine:
    def __init__(
        self,
        base_dir: Path,
        embed_model_dir: Path,
        llm_model_path: Path,
        index_dir: Path,
        log_dir: Path,
    ):
        self.base_dir = base_dir
        self.embed_model_dir = embed_model_dir
        self.llm_model_path = llm_model_path
        self.index_dir = index_dir
        self.log_dir = log_dir

        self.index_path = self.index_dir / "index.faiss"
        self.chunks_path = self.index_dir / "chunks.json"
        self.docs_path = self.index_dir / "docs.json"

        self._embedder = None
        self._llm = None

        self._setup_logging()

    def _setup_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / "app.log"
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        logging.info("RAG Engine initialized.")

    def _load_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            if not self.embed_model_dir.exists():
                raise RAGEngineError(
                    f"Embedding model missing at {self.embed_model_dir}"
                )
            self._embedder = SentenceTransformer(str(self.embed_model_dir))
        return self._embedder

    def _load_llm(self) -> Llama:
        if self._llm is None:
            if not self.llm_model_path.exists():
                raise RAGEngineError(
                    f"LLM model missing at {self.llm_model_path}"
                )
            self._llm = Llama(
                model_path=str(self.llm_model_path),
                n_ctx=4096,
                n_threads=8,
            )
        return self._llm

    def index_exists(self) -> bool:
        return self.index_path.exists() and self.chunks_path.exists()

    def build_index(self, file_paths: List[str], progress=None) -> Dict[str, int]:
        if progress:
            progress("Loading documents…")
        try:
            docs = load_documents(file_paths, progress)
        except ValueError as exc:
            logging.warning("Document load warning: %s", exc)
            raise RAGEngineError(str(exc)) from exc
        if not docs:
            raise RAGEngineError("No valid text extracted from documents.")

        if progress:
            progress("Chunking documents…")
        chunks = self._chunk_documents(docs)
        if not chunks:
            raise RAGEngineError("No chunks created from documents.")

        if progress:
            progress("Embedding chunks…")
        embeddings = self._embed_chunks([chunk.text for chunk in chunks])
        embeddings = self._normalize_embeddings(embeddings)

        if progress:
            progress("Building FAISS index…")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        with self.chunks_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(chunk) for chunk in chunks], f, ensure_ascii=False, indent=2)

        with self.docs_path.open("w", encoding="utf-8") as f:
            json.dump(file_paths, f, ensure_ascii=False, indent=2)

        logging.info("Index built with %s chunks.", len(chunks))
        return {"chunks": len(chunks), "docs": len(file_paths)}

    def answer_question(self, question: str, progress=None) -> Dict[str, List[str]]:
        if not self.index_exists():
            raise RAGEngineError("Index not found. Build the index first.")

        if progress:
            progress("Loading index…")
        index = faiss.read_index(str(self.index_path))
        with self.chunks_path.open("r", encoding="utf-8") as f:
            chunks = [ChunkMetadata(**item) for item in json.load(f)]

        if progress:
            progress("Embedding query…")
        query_embedding = self._embed_chunks([question])
        query_embedding = self._normalize_embeddings(query_embedding)

        if progress:
            progress("Retrieving…")
        scores, indices = index.search(query_embedding, k=5)

        retrieved = []
        source_lines = []
        for rank, idx in enumerate(indices[0], start=1):
            if idx < 0:
                continue
            chunk = chunks[idx]
            score = float(scores[0][rank - 1])
            retrieved.append((rank, chunk, score))
            source_lines.append(
                f"[{rank}] {Path(chunk.source).name}#chunk_{chunk.chunk_id} score={score:.3f}"
            )

        context = "\n\n".join(
            [f"[{rank}] {chunk.text}" for rank, chunk, _ in retrieved]
        )

        if progress:
            progress("Generating…")
        prompt = (
            "You are a document assistant. Answer ONLY using the provided context. "
            "If the answer is not in the context, say: 'I don’t know from the provided documents.'\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        llm = self._load_llm()
        response = llm(
            prompt,
            max_tokens=512,
            temperature=0.2,
            stop=["\nQuestion:"],
        )
        answer = response["choices"][0]["text"].strip()
        return {
            "answer": answer,
            "sources": source_lines,
        }

    def _chunk_documents(self, documents: List[Tuple[str, str]]) -> List[ChunkMetadata]:
        chunks = []
        chunk_size = 1000
        overlap = 200
        for source, text in documents:
            start = 0
            chunk_id = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        ChunkMetadata(
                            chunk_id=chunk_id,
                            source=source,
                            text=chunk_text,
                        )
                    )
                    chunk_id += 1
                start = end - overlap
        return chunks

    def _embed_chunks(self, texts: List[str]) -> np.ndarray:
        embedder = self._load_embedder()
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype("float32")

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
