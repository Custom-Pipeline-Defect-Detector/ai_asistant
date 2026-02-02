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
By default, the app looks for models in:
```
D:\offline_doc_assistant\models\qwen_model.gguf
D:\offline_doc_assistant\models\embed\  (SentenceTransformer model folder)
```
You can change these paths (and other settings) via `app_config.ini`. The app will **not** attempt to download any models.

## Configuration (app_config.ini)
The app reads settings from `app_config.ini` (next to `app.py`). This file controls:
- Base directory and indexing folders
- OCR/Tesseract/Poppler/LibreOffice paths
- Chunking, retrieval, and FAISS tuning values
- Window/UI behavior
- Ollama endpoint/model

Update the values in `app_config.ini` and restart the app to apply changes.

## Running the App
```bash
python app.py
```

## Notes
- **Scanned PDFs**: The app can OCR scanned PDFs when OCR is enabled in `app_config.ini`.
- **Offline**: The application is fully offline and makes no web requests beyond the local Ollama endpoint you configure.

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
