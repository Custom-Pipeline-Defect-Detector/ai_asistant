# Offline Document AI Assistant (Windows)

A fully offline Windows desktop application that lets you index local documents and ask questions using Retrieval-Augmented Generation (RAG). It uses a local Qwen GGUF model with `llama-cpp-python`, SentenceTransformer embeddings, FAISS, and a PyQt5 UI.

## Requirements
- Python 3.10+
- Windows machine with the models placed locally (no downloads at runtime)

## Setup
1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Model Placement
Place your models in the following paths:
```
D:\offline_doc_assistant\models\qwen_model.gguf
D:\offline_doc_assistant\models\embed\  (SentenceTransformer model folder)
```
The app will **not** attempt to download any models.

## Running the App
```bash
python app.py
```

## Notes
- **Scanned PDFs**: If a PDF has no extractable text, the app will warn: “Scanned PDF not supported (needs OCR)”. You must run OCR externally first.
- **Offline**: The application is fully offline and makes no web requests.

## Folder Structure
```
D:\offline_doc_assistant\
  app.py
  ui_main.py
  rag_engine.py
  doc_loaders.py
  worker.py
  requirements.txt
  README.md
  assets\
  models\
    qwen_model.gguf
    embed\
  data\
    index\
    logs\
```
