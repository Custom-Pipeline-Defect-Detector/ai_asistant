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
D:\offline_doc_assistant\models\embed\  (SentenceTransformer model folder)
```
You can change these paths (and other settings) via `app_config.ini`. The app will **not** attempt to download any models.

## Configuration (app_config.ini + assistant.ini)
The app reads settings from `app_config.ini` (next to `app.py`) and `assistant.ini` (under `data/memory/`). These files control:
- **Paths**: base/data/index/logs/memory folders, UI assets
- **Indexing**: sources list, include/exclude globs, supported extensions, chunk size/overlap
- **Retrieval**: `top_k`, `min_score`
- **OCR**: enable/disable, language, PDF DPI/page limits
- **LibreOffice**: enable/disable, conversion timeout, optional soffice path
- **Ollama**: base URL, model, request timeout
- **Generation**: temperature, top_p, num_predict
- **UI defaults**: title, min size, always-on-top, status template
- **Agent controls**: control block syntax, auto rebuild, self-tune
- **Assistant behavior**: tone, personality, custom rules, prompt template

The app supports live reload of config changes when triggered by control actions (no restart required).

### app_config.ini schema
```
[paths] base_dir, data_dir, index_dir, log_dir, memory_dir, auto_docs_dir, embed_model_dir, icons_dir, app_icon_ico, app_icon_png, ui_logo_path
[index] sources, extra_sources, include_globs, exclude_globs, supported_exts, allow_any_path, max_file_mb, max_docs_per_run, chunk_size, chunk_overlap, embed_batch_size, hnsw_m, hnsw_ef_construction, hnsw_ef_search
[retrieval] top_k, min_score, rerank
[ollama] base_url, model, timeout_s
[generation] temperature, top_p, num_predict
[ocr] enable_ocr, lang, pdf_dpi, max_pages, poppler_path, tesseract_exe, scanned_pdf_min_chars
[libreoffice] enable_libreoffice, timeout_s, soffice_path
[ui] window_title, min_width, min_height, always_on_top, logo_height_px, status_template
[agent] enable_control, control_block_syntax, max_steps, auto_rebuild_index, self_tune
```

### assistant.ini schema
```
[user] name, language
[assistant] name, tone, personality, custom_rules, system_prompt_template
[preferences] time_format, timezone, auto_save_facts
[agent] enable_control, control_block_syntax, max_steps, auto_rebuild_index, self_tune
```

## Running the App
```bash
python app.py
```

## Control Blocks (Agent Actions)
The assistant can emit tool-like commands inside a `<CONTROL>` block. When detected, the app applies the actions, reloads config/index, and then answers again.

Example:
```
<CONTROL>
SET paths.auto_docs_dir="D:\docs"
ADD_SOURCE "E:\Team\Policies"
REMOVE_SOURCE "E:\Old"
REBUILD_INDEX
SET retrieval.top_k=12
</CONTROL>
```

Supported actions:
- `SET section.key=value`
- `ADD_SOURCE "path"`
- `REMOVE_SOURCE "path"`
- `REBUILD_INDEX`
- `CLEAR_INDEX`
- `RELOAD_CONFIG`
- `LIST_SOURCES`
- `RESCAN_DISK`

## Usage Examples
- **Index a new folder**: “Please index `E:\Docs`”
- **Add a source**: “Add `E:\Team\Policies` to the index”
- **Change model**: “Switch to model `llama3.1:8b`”
- **Rebuild**: “Rebuild the index”

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
    embed\
  data\
    index\
    logs\
    memory\
      assistant.ini
```
