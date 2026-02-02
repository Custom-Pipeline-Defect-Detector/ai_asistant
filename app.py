import fnmatch
import re
import hashlib
import json
import os
import shutil
import sys
import traceback
from difflib import get_close_matches
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

# --- ADD THIS near the top of app.py (imports) ---

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QPainter,
    QColor,
    QPen,
    QLinearGradient,
    QPainterPath,
    QIcon,
)
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QWidget,
    QToolButton,
)

from app_config import APP_CONFIG_SCHEMA, apply_runtime_env, default_app_config, load_app_config, normalize_app_value
from config_io import append_audit_log, config_to_dict, load_ini_with_backup, atomic_write_ini
from rag_engine import RAGEngine, default_assistant_config, normalize_assistant_value
from ui_main import MainWindow
from worker import WorkerThread

DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("app_config.ini")
CONFIG_PATH = Path(os.environ.get("DOC_ASSISTANT_CONFIG", str(DEFAULT_CONFIG_PATH)))


def set_windows_appusermodelid(app_id: str) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _matches_globs(path: Path, globs: Iterable[str]) -> bool:
    if not globs:
        return True
    normalized = str(path)
    name = path.name
    for pattern in globs:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(normalized, pattern):
            return True
    return False


def _walk_all_paths(root: Path) -> List[str]:
    """Utility for simple wildcard matching over the base_dir tree (best-effort)."""
    try:
        root = Path(root)
        if not root.exists():
            return []
        return [str(p) for p in root.rglob("*")]
    except Exception:
        return []


def collect_auto_docs(config) -> list[str]:
    sources = (config.index_sources or []) + (config.extra_sources or [])
    if not sources:
        sources = [str(config.auto_docs_dir)]

    include_globs = config.include_globs or []
    exclude_globs = config.exclude_globs or []
    supported_exts = {ext.lower() for ext in (config.supported_exts or [])}
    max_bytes = max(0, int(config.max_file_mb)) * 1024 * 1024 if config.max_file_mb else 0
    max_docs = int(config.max_docs_per_run) if config.max_docs_per_run else 0

    docs: list[str] = []

    def accept_path(p: Path) -> bool:
        if not p.is_file():
            return False
        if p.name.startswith("~$"):
            return False
        if not config.allow_any_path:
            try:
                p.resolve().relative_to(config.base_dir.resolve())
            except Exception:
                return False
        if supported_exts and p.suffix.lower() not in supported_exts:
            return False
        if include_globs and not _matches_globs(p, include_globs):
            return False
        if exclude_globs and _matches_globs(p, exclude_globs):
            return False
        if max_bytes and p.stat().st_size > max_bytes:
            return False
        return True

    for src in sources:
        p = _expand_path(src)
        if p.is_file():
            if accept_path(p):
                docs.append(str(p))
        elif p.is_dir():
            for file_path in p.rglob("*"):
                if accept_path(file_path):
                    docs.append(str(file_path))
                    if max_docs and len(docs) >= max_docs:
                        return docs

    return docs


def parse_control_blocks(text: str) -> Tuple[str, List[Dict[str, str]]]:
    actions: List[Dict[str, str]] = []
    cleaned = text
    while True:
        start = cleaned.find("<CONTROL>")
        end = cleaned.find("</CONTROL>")
        if start == -1 or end == -1 or end < start:
            break
        block = cleaned[start + len("<CONTROL>") : end]
        cleaned = (cleaned[:start] + cleaned[end + len("</CONTROL>") :]).strip()
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("SET "):
                _, rest = line.split(" ", 1)
                if "=" not in rest:
                    actions.append({"type": "ERROR", "detail": f"Invalid SET syntax: {line}"})
                    continue
                key, value = rest.split("=", 1)
                actions.append(
                    {
                        "type": "SET",
                        "key": key.strip(),
                        "value": _strip_quotes(value),
                    }
                )
                continue
            if line.upper().startswith("ADD_SOURCE"):
                _, _, value = line.partition(" ")
                actions.append({"type": "ADD_SOURCE", "value": _strip_quotes(value)})
                continue
            if line.upper().startswith("REMOVE_SOURCE"):
                _, _, value = line.partition(" ")
                actions.append({"type": "REMOVE_SOURCE", "value": _strip_quotes(value)})
                continue
            if line.upper() == "REBUILD_INDEX":
                actions.append({"type": "REBUILD_INDEX"})
                continue
            if line.upper() == "CLEAR_INDEX":
                actions.append({"type": "CLEAR_INDEX"})
                continue
            if line.upper() == "RELOAD_CONFIG":
                actions.append({"type": "RELOAD_CONFIG"})
                continue
            if line.upper() == "LIST_SOURCES":
                actions.append({"type": "LIST_SOURCES"})
                continue
            if line.upper() == "RESCAN_DISK":
                actions.append({"type": "RESCAN_DISK"})
                continue
            actions.append({"type": "ERROR", "detail": f"Unknown control action: {line}"})
    return cleaned.strip(), actions


class AppController(QWidget):
    def __init__(self):
        super().__init__()
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        self.config_path = CONFIG_PATH
        self.config = load_app_config(self.config_path)
        apply_runtime_env(self.config)

        window_flags = Qt.FramelessWindowHint | Qt.Window
        if self.config.always_on_top:
            window_flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(window_flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setMouseTracking(True)

        self.setWindowTitle(self.config.window_title)
        self.setMinimumSize(self.config.min_width, self.config.min_height)

        # ---------- resize config ----------
        self._resize_margin = 6
        self._frame_margin = self._resize_margin
        self._corner_radius = 22
        self._drag_allowed_height = 20
        self._drag_pos = None

        self._resizing = False
        self._resize_edge = None
        self._press_pos = None
        self._start_geom = None
        self._pending_file_choices: List[Path] = []

        # =========================
        # Child UI (MainWindow)
        # =========================
        self.ui = MainWindow()
        self.ui.setParent(self)
        self.ui.setWindowFlags(Qt.Widget)

        # =========================
        # Custom buttons
        # =========================
        btn_css = """
            QToolButton {
                border: none;
                border-radius: 14px;
                background: rgba(255, 255, 255, 30);
                color: white;
                font-size: 14px;
                font-weight: 700;
            }
            QToolButton:hover { background: rgba(255, 255, 255, 70); }
            QToolButton:pressed { background: rgba(255, 255, 255, 110); }
        """

        self.settings_btn = QToolButton(self)
        self.settings_btn.setText("⚙")
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.setFixedSize(30, 30)
        self.settings_btn.setStyleSheet(btn_css)
        self.settings_btn.clicked.connect(self.on_settings)

        self.min_btn = QToolButton(self)
        self.min_btn.setText("–")
        self.min_btn.setCursor(Qt.PointingHandCursor)
        self.min_btn.setFixedSize(28, 28)
        self.min_btn.setStyleSheet(btn_css)
        self.min_btn.clicked.connect(self.showMinimized)

        self.close_btn = QToolButton(self)
        self.close_btn.setText("✕")
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.setStyleSheet(btn_css)
        self.close_btn.clicked.connect(self._quit_app)

        # =========================
        # RAG / app logic
        # =========================
        self.rag = RAGEngine(
            base_dir=self.config.base_dir,
            embed_model_dir=self.config.embed_model_dir,
            index_dir=self.config.index_dir,
            log_dir=self.config.log_dir,
            memory_dir=self.config.memory_dir,
            ollama_base_url=self.config.ollama_base_url,
            ollama_model=self.config.ollama_model,
            ollama_timeout_s=self.config.ollama_timeout_s,
            generation_temperature=self.config.generation_temperature,
            generation_top_p=self.config.generation_top_p,
            generation_num_predict=self.config.generation_num_predict,
            retrieval_min_score=self.config.retrieval_min_score,
            embed_batch_size=self.config.embed_batch_size,
            hnsw_m=self.config.hnsw_m,
            hnsw_ef_construction=self.config.hnsw_ef_construction,
            hnsw_ef_search=self.config.hnsw_ef_search,
            enable_ocr=self.config.enable_ocr,
            ocr_lang=self.config.ocr_lang,
            enable_libreoffice=self.config.enable_libreoffice,
            poppler_path=self.config.poppler_path,
            tesseract_exe=self.config.tesseract_exe,
            soffice_path=self.config.soffice_path,
            pdf_ocr_max_pages=self.config.pdf_ocr_max_pages,
            scanned_pdf_min_chars=self.config.scanned_pdf_min_chars,
            pdf_dpi=self.config.pdf_dpi,
            libreoffice_timeout_s=self.config.libreoffice_timeout_s,
        )

        self._busy = False
        self._thread: WorkerThread | None = None

        ini = self.rag.load_ini()
        user_name = ini.get("user", {}).get("name", "Mir")
        lang = ini.get("user", {}).get("language", "en") or "en"
        self.ui.apply_language(lang, user_name=user_name)
        self.ui.set_logo_height(self.config.logo_height_px)

        self.ui.send_button.clicked.connect(self.on_send)
        self.ui.add_file_button.clicked.connect(self.on_add_files)
        self.ui.files_dropped.connect(self.on_files_dropped)

        self._update_status_ready()
        self._apply_layout()

    # =========================
    # Layout
    # =========================
    def _apply_layout(self):
        m = self._frame_margin
        self.ui.setGeometry(m, m, self.width() - 2 * m, self.height() - 2 * m)

        pad = 10
        gap = 8
        y = pad

        x_close = self.width() - self.close_btn.width() - pad
        self.close_btn.move(x_close, y)

        x_min = x_close - gap - self.min_btn.width()
        self.min_btn.move(x_min, y)

        x_set = x_min - gap - self.settings_btn.width()
        self.settings_btn.move(x_set, y)

    def _apply_window_flags(self):
        window_flags = Qt.FramelessWindowHint | Qt.Window
        if self.config.always_on_top:
            window_flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(window_flags)
        self.show()

    def _format_status(self, template: str) -> str:
        return template.format(
            auto_docs_dir=self.config.auto_docs_dir,
            ocr_status="ON" if self.config.enable_ocr else "OFF",
            libreoffice_status="ON" if self.config.enable_libreoffice else "OFF",
            source_count=len(self.config.index_sources) + len(self.config.extra_sources),
            index_dir=self.config.index_dir,
            data_dir=self.config.data_dir,
        )

    def _update_status_ready(self):
        self.ui.update_status(self._format_status(self.config.status_template))

    def _load_app_config_parser(self) -> ConfigParser:
        defaults = default_app_config(self.config.base_dir)
        return load_ini_with_backup(self.config_path, defaults)

    def _load_assistant_config_parser(self) -> ConfigParser:
        defaults = default_assistant_config()
        return load_ini_with_backup(self.rag.ini_path, defaults)

    def _hash_user_message(self, message: str) -> str:
        return hashlib.sha256(message.encode("utf-8")).hexdigest()

    def _apply_config_to_runtime(self, config) -> None:
        self.rag.apply_runtime_config(
            base_dir=config.base_dir,
            embed_model_dir=config.embed_model_dir,
            index_dir=config.index_dir,
            log_dir=config.log_dir,
            memory_dir=config.memory_dir,
            ollama_base_url=config.ollama_base_url,
            ollama_model=config.ollama_model,
            ollama_timeout_s=config.ollama_timeout_s,
            generation_temperature=config.generation_temperature,
            generation_top_p=config.generation_top_p,
            generation_num_predict=config.generation_num_predict,
            retrieval_min_score=config.retrieval_min_score,
            embed_batch_size=config.embed_batch_size,
            hnsw_m=config.hnsw_m,
            hnsw_ef_construction=config.hnsw_ef_construction,
            hnsw_ef_search=config.hnsw_ef_search,
            enable_ocr=config.enable_ocr,
            ocr_lang=config.ocr_lang,
            enable_libreoffice=config.enable_libreoffice,
            poppler_path=config.poppler_path,
            tesseract_exe=config.tesseract_exe,
            soffice_path=config.soffice_path,
            pdf_ocr_max_pages=config.pdf_ocr_max_pages,
            scanned_pdf_min_chars=config.scanned_pdf_min_chars,
            pdf_dpi=config.pdf_dpi,
            libreoffice_timeout_s=config.libreoffice_timeout_s,
        )

    def _apply_config_to_ui(self, config) -> None:
        self.config = config
        apply_runtime_env(self.config)
        self._apply_window_flags()
        self.setWindowTitle(self.config.window_title)
        self.setMinimumSize(self.config.min_width, self.config.min_height)
        self.ui.set_logo_height(self.config.logo_height_px)
        self._update_status_ready()

    def _apply_control_actions(
        self,
        actions: List[Dict[str, str]],
        user_message: str,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Tuple[List[str], bool, bool, Optional[List[str]]]:
        summaries: List[str] = []
        config_changed = False
        index_changed = False
        listed_sources: Optional[List[str]] = None
        errors: List[str] = []

        app_cfg = self._load_app_config_parser()
        assistant_cfg = self._load_assistant_config_parser()
        app_before = config_to_dict(app_cfg)
        assistant_before = config_to_dict(assistant_cfg)

        def emit(msg: str):
            if progress:
                progress(msg)

        for action in actions:
            action_type = action.get("type", "").upper()
            if action_type == "ERROR":
                errors.append(action.get("detail", "Invalid action"))
                continue
            if action_type == "SET":
                key = action.get("key", "")
                value = action.get("value", "")
                if "." not in key:
                    errors.append(f"Invalid SET key: {key}")
                    continue
                section, option = key.split(".", 1)
                section = section.strip().lower()
                option = option.strip().lower()
                try:
                    if section in APP_CONFIG_SCHEMA:
                        _, normalized = normalize_app_value(section, option, value)
                        if section not in app_cfg:
                            app_cfg[section] = {}
                        app_cfg[section][option] = normalized
                        config_changed = True
                        if section == "index" or (section == "paths" and option in {"auto_docs_dir", "index_dir"}):
                            index_changed = True
                        summaries.append(f"Set {section}.{option}")
                    else:
                        _, normalized = normalize_assistant_value(section, option, value)
                        if section not in assistant_cfg:
                            assistant_cfg[section] = {}
                        assistant_cfg[section][option] = normalized
                        config_changed = True
                        summaries.append(f"Set {section}.{option} (assistant)")
                except Exception as ex:
                    errors.append(str(ex))
                continue
            if action_type in {"ADD_SOURCE", "REMOVE_SOURCE"}:
                value = action.get("value", "").strip()
                if not value:
                    errors.append(f"Missing source for {action_type}")
                    continue
                sources_raw = app_cfg.get("index", "sources", fallback="")
                sources = [s for s in sources_raw.split(";") if s.strip()]
                if action_type == "ADD_SOURCE":
                    if value not in sources:
                        sources.append(value)
                        if "index" not in app_cfg:
                            app_cfg["index"] = {}
                        app_cfg["index"]["sources"] = "; ".join(sources)
                        config_changed = True
                        index_changed = True
                        summaries.append(f"Added source: {value}")
                else:
                    if value in sources:
                        sources = [s for s in sources if s != value]
                        if "index" not in app_cfg:
                            app_cfg["index"] = {}
                        app_cfg["index"]["sources"] = "; ".join(sources)
                        config_changed = True
                        index_changed = True
                        summaries.append(f"Removed source: {value}")
                continue
            if action_type == "REBUILD_INDEX":
                index_changed = True
                summaries.append("Rebuild index")
                continue
            if action_type == "CLEAR_INDEX":
                summaries.append("Clear index")
                index_changed = True
                self.rag.clear_index()
                continue
            if action_type == "RELOAD_CONFIG":
                summaries.append("Reload config")
                config_changed = True
                continue
            if action_type == "LIST_SOURCES":
                sources_raw = app_cfg.get("index", "sources", fallback="")
                listed_sources = [s.strip() for s in sources_raw.split(";") if s.strip()]
                summaries.append("Listed sources")
                continue
            if action_type == "RESCAN_DISK":
                index_changed = True
                summaries.append("Rescan disk")
                continue

        if errors:
            summaries.append("Errors: " + "; ".join(errors))

        if config_changed:
            atomic_write_ini(self.config_path, app_cfg)
            atomic_write_ini(self.rag.ini_path, assistant_cfg)

        audit_payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "user_message_hash": self._hash_user_message(user_message),
            "actions": actions,
            "errors": errors,
            "diff": {
                "app_config": {"before": app_before, "after": config_to_dict(app_cfg)},
                "assistant_ini": {"before": assistant_before, "after": config_to_dict(assistant_cfg)},
            },
        }
        append_audit_log(self.config.log_dir / "config_audit.log", audit_payload)

        if config_changed:
            emit("Reloading configuration…")

        return summaries, config_changed, index_changed, listed_sources

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_layout()

    # =========================
    # Paint
    # =========================
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        r = self._corner_radius
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)

        path = QPainterPath()
        path.addRoundedRect(rect, r, r)

        painter.fillPath(path, QColor(25, 25, 30, 210))

        gloss = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gloss.setColorAt(0.0, QColor(255, 255, 255, 90))
        gloss.setColorAt(0.25, QColor(255, 255, 255, 25))
        gloss.setColorAt(0.6, QColor(255, 255, 255, 0))
        painter.fillPath(path, gloss)

        painter.setPen(QPen(QColor(255, 255, 255, 70), 1.3))
        painter.drawPath(path)

        inner = QRectF(rect).adjusted(3, 3, -3, -3)
        inner_path = QPainterPath()
        inner_path.addRoundedRect(inner, max(0, r - 3), max(0, r - 3))
        painter.setPen(QPen(QColor(0, 0, 0, 120), 1.0))
        painter.drawPath(inner_path)

        painter.end()

    # =========================
    # Resize helpers
    # =========================
    def _hit_test_resize(self, pos):
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        m = self._resize_margin

        left = x <= m
        right = x >= w - m
        top = y <= m
        bottom = y >= h - m

        if top and left:
            return "top-left"
        if top and right:
            return "top-right"
        if bottom and left:
            return "bottom-left"
        if bottom and right:
            return "bottom-right"
        if left:
            return "left"
        if right:
            return "right"
        if top:
            return "top"
        if bottom:
            return "bottom"
        return None

    def _cursor_for_edge(self, edge):
        if not edge:
            return Qt.ArrowCursor
        mapping = {
            "left": Qt.SizeHorCursor,
            "right": Qt.SizeHorCursor,
            "top": Qt.SizeVerCursor,
            "bottom": Qt.SizeVerCursor,
            "top-left": Qt.SizeFDiagCursor,
            "bottom-right": Qt.SizeFDiagCursor,
            "top-right": Qt.SizeBDiagCursor,
            "bottom-left": Qt.SizeBDiagCursor,
        }
        return mapping.get(edge, Qt.ArrowCursor)

    def _perform_resize(self, global_pos):
        dx = global_pos.x() - self._press_pos.x()
        dy = global_pos.y() - self._press_pos.y()
        g = self._start_geom

        x, y, w, h = g.x(), g.y(), g.width(), g.height()

        if "left" in self._resize_edge:
            x += dx
            w -= dx
        if "right" in self._resize_edge:
            w += dx
        if "top" in self._resize_edge:
            y += dy
            h -= dy
        if "bottom" in self._resize_edge:
            h += dy

        w = max(self.minimumWidth(), w)
        h = max(self.minimumHeight(), h)
        self.setGeometry(x, y, w, h)

    # =========================
    # Mouse: resize + move
    # =========================
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            edge = self._hit_test_resize(event.pos())
            if edge:
                self._resizing = True
                self._resize_edge = edge
                self._press_pos = event.globalPos()
                self._start_geom = self.geometry()
                event.accept()
                return

            if event.pos().y() <= self._drag_allowed_height:
                self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing and (event.buttons() & Qt.LeftButton):
            self._perform_resize(event.globalPos())
            event.accept()
            return

        edge = self._hit_test_resize(event.pos())
        self.setCursor(self._cursor_for_edge(edge))

        if self._drag_pos is not None and (event.buttons() & Qt.LeftButton):
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        self._resizing = False
        self._resize_edge = None
        self._press_pos = None
        self._start_geom = None
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    # =========================
    # Ensure close = quit app
    # =========================
    def _quit_app(self):
        QApplication.instance().quit()

    def closeEvent(self, event):
        QApplication.instance().quit()
        event.accept()

    # =========================
    # Worker helpers
    # =========================
    def _start_worker(self, fn, args=(), kwargs=None, on_ok=None, on_err=None, on_partial=None):
        if self._busy:
            return
        self._busy = True
        self.ui.set_buttons_enabled(False)

        self._thread = WorkerThread(fn, args=args, kwargs=kwargs or {})
        self._thread.progress.connect(self.ui.update_status)

        if on_partial:
            self._thread.partial.connect(on_partial)

        def _ok(payload):
            self._busy = False
            self.ui.set_buttons_enabled(True)
            if on_ok:
                on_ok(payload)

        def _err(err_text):
            self._busy = False
            self.ui.set_buttons_enabled(True)
            if on_err:
                on_err(err_text)
            else:
                QMessageBox.critical(self, "Error", err_text)

        self._thread.result.connect(_ok)
        self._thread.error.connect(_err)
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.start()

    def _show_index_report_if_needed(self, res: dict):
        """
        Displays a small report when some docs fail extraction/indexing.
        Expects rag_engine.build_index() to return:
          - report: list[dict]
          - failed_docs/indexed_docs/attempted_docs (optional but recommended)
          - settings.enable_ocr (optional)
        """
        report = res.get("report", []) or []
        if not report:
            return

        failed = [r for r in report if not r.get("ok")]
        if not failed:
            return

        lines = []
        for r in failed[:20]:
            name = Path(r.get("source", "")).name or r.get("source", "")
            msg = r.get("error") or r.get("warning") or "Unknown failure"
            lines.append(f"- {name}: {msg}")

        settings = res.get("settings", {}) or {}
        hints = []
        if not settings.get("enable_ocr", self.config.enable_ocr):
            hints.append("Tip: enable OCR in app_config.ini to index scanned PDFs/images (after installing OCR deps).")
        if not settings.get("enable_libreoffice", self.config.enable_libreoffice):
            hints.append("Tip: enable LibreOffice in app_config.ini to index .doc/.odt/.ods/.odp/.ppt/.xls.")

        msg = "Some files were not indexed:\n\n" + "\n".join(lines)
        if hints:
            msg += "\n\n" + "\n".join(hints)

        QMessageBox.information(self, "Indexing report", msg)

    def _ensure_index(self, after_ok=None):
        files = collect_auto_docs(self.config)
        if not files:
            self.ui.update_status(f"No documents found in sources: {', '.join(self.config.index_sources)}")
            QMessageBox.warning(
                self,
                "No documents found",
                "No documents found in configured sources.\n\n"
                "Add sources in app_config.ini or use control commands to add paths.",
            )
            return

        if self.rag.index_exists():
            indexed = set(self.rag.get_indexed_docs())
            current = set(files)
            if indexed == current:
                self.ui.update_status(f"Index ready. Docs: {len(files)}")
                if after_ok:
                    after_ok({"docs": len(files), "chunks": -1})
                return

        def do_build():
            return self.rag.build_index(
                files,
                progress=lambda s: self._thread.progress.emit(s) if self._thread else None,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                enable_ocr=self.config.enable_ocr,
                ocr_lang=self.config.ocr_lang,
                enable_libreoffice=self.config.enable_libreoffice,
            )

        def on_ok(res):
            docs = res.get("docs", len(files))
            chunks = res.get("chunks", 0)

            indexed_docs = res.get("indexed_docs", docs)
            attempted_docs = res.get("attempted_docs", indexed_docs)
            failed_docs = res.get("failed_docs", max(0, attempted_docs - indexed_docs))

            self.ui.update_status(
                f"Index ready. Selected: {docs}, Indexed: {indexed_docs}/{attempted_docs}, "
                f"Failed: {failed_docs}, Chunks: {chunks}"
            )

            # NEW: show per-file report for partial failures
            if failed_docs > 0:
                self._show_index_report_if_needed(res)

            if after_ok:
                after_ok(res)

        self._start_worker(
            fn=do_build,
            on_ok=on_ok,
            on_err=lambda e: QMessageBox.critical(self, "Index Error", e),
        )

    # ---------- Agent-ish helpers (path handling + visible progress) ----------
    def on_files_dropped(self, paths: list):
        """Handle drag & drop from the UI."""
        if self._busy:
            return
        # treat dropped files like 'Add files' (copy into auto_docs_dir)
        copied: List[Path] = []
        failures: List[str] = []

        for p in paths:
            try:
                copied.append(self._copy_to_auto_docs(Path(p)))
            except Exception as exc:
                failures.append(f"{p}: {exc}")

        if failures:
            QMessageBox.warning(self, "Some files could not be added", "\n".join(failures[:30]))

        if not copied:
            return

        self.ui.append_user_message(f"[added {len(copied)} file(s)]")
        self.ui.update_status("Indexing added files…")
        self._ensure_index()

    def _handle_pending_file_choice(self, q: str) -> bool:
        """Resolve a user choice when we previously offered multiple file matches."""
        if not self._pending_file_choices:
            return False

        raw = (q or "").strip().lower()
        if raw in {"cancel", "stop", "never mind", "no"}:
            self._pending_file_choices.clear()
            self.ui.update_status("Cancelled.")
            self.ui.append_assistant_message("Okay — cancelled.")
            return True

        if raw in {"all", "a"}:
            picked = list(self._pending_file_choices)
            self._pending_file_choices.clear()
            self._index_specific_files(picked)
            return True

        # number selection (1-based)
        if raw.isdigit():
            i = int(raw)
            if 1 <= i <= len(self._pending_file_choices):
                picked = [self._pending_file_choices[i - 1]]
                self._pending_file_choices.clear()
                self._index_specific_files(picked)
                return True

        # not understood; re-show prompt
        rows = [[str(i + 1), str(p)] for i, p in enumerate(self._pending_file_choices[:50])]
        self.ui._show_grid_in_sheet(["#", "File"], rows)
        self.ui.append_assistant_message(
            "Please type a number (e.g., 1), or type **all** to index them all, or **cancel**."
        )
        return True

    def _index_specific_files(self, files: List[Path]) -> None:
        """Index a specific file list without changing configured source folders."""
        if not files:
            return

        # Copy into auto_docs_dir to keep everything local/portable
        copied: List[Path] = []
        failures: List[str] = []
        for p in files:
            try:
                copied.append(self._copy_to_auto_docs(p))
            except Exception as exc:
                failures.append(f"{p}: {exc}")

        if failures:
            QMessageBox.warning(self, "Some files could not be added", "\n".join(failures[:30]))

        if not copied:
            return

        self.ui.update_status("Indexing selected file(s)…")
        self._ensure_index()

    def _handle_path_message(self, q: str) -> bool:
        """Detects file/folder paths (or control commands) in the user message."""
        raw = (q or "").strip()
        if not raw:
            return False

        # Natural-language directory listing requests
        if re.search(r"\b(what|which|list|show)\b.*\b(files?|folders?|contents?)\b", raw.lower()) and self._extract_windows_path(raw):
            return self._handle_list_dir_request(raw)

        # --- Control commands ---
        if raw.startswith("/"):
            return self._handle_control_command(raw)

        # If user pasted a path inside text, try to treat the whole message as a path token.
        token = raw
        # common wrappers like quotes
        if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
            token = token[1:-1].strip()

        # Expand env vars and ~
        token = os.path.expandvars(os.path.expanduser(token))

        # Windows copy/paste sometimes includes trailing punctuation
        token = token.strip().strip("<>[](){}.,;")

        # Wildcards -> glob expand (best-effort)
        if any(ch in token for ch in ["*", "?", "["]):
            matches = []
            try:
                matches = [Path(p) for p in sorted(fnmatch.filter(_walk_all_paths(self.config.base_dir), token))]
            except Exception:
                matches = []
            if not matches:
                try:
                    matches = [Path(p) for p in sorted(Path().glob(token))]
                except Exception:
                    matches = []
            return self._handle_path_matches(matches, token)

        p = Path(token)
        if p.exists():
            if p.is_dir():
                return self._add_source_folder_and_reindex(p)
            if p.is_file():
                return self._handle_path_matches([p], token)

        return False

    def _handle_path_matches(self, matches: List[Path], token: str) -> bool:
        matches = [m for m in matches if m.exists()]
        if not matches:
            return False

        # If multiple: offer a small picker
        if len(matches) > 1:
            self._pending_file_choices = matches[:50]
            rows = [[str(i + 1), str(p)] for i, p in enumerate(self._pending_file_choices)]
            self.ui._show_grid_in_sheet(["#", f"Matches for {token}"], rows)
            self.ui.append_assistant_message(
                "I found multiple matching files. Reply with a number (e.g., 1), **all**, or **cancel**."
            )
            self.ui.update_status("Waiting for selection…")
            return True

        # Single file
        self._index_specific_files([matches[0]])
        return True

    def _add_source_folder_and_reindex(self, folder: Path) -> bool:
        """Persist a folder into [index].extra_sources and re-index."""
        folder = folder.resolve()
        app_cfg = self._load_app_config_parser()

        if not app_cfg.has_section("index"):
            app_cfg.add_section("index")

        current = app_cfg.get("index", "extra_sources", fallback="").strip()
        parts = [p.strip() for p in current.split(";") if p.strip()]
        if str(folder) not in parts:
            parts.append(str(folder))
            app_cfg.set("index", "extra_sources", "; ".join(parts))
            atomic_write_ini(self.config_path, app_cfg)
            self.config = load_app_config(self.config_path)
            apply_runtime_env(self.config)

        self.ui.append_user_message(f"[added folder source] {folder}")
        self.ui.update_status("Indexing folder…")
        self._ensure_index()
        return True

    def _handle_control_command(self, raw: str) -> bool:
        """A small command set to make the assistant more agent-like."""
        parts = raw.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"/help", "/?"}:
            self.ui.append_assistant_message(
                "Commands:\n"
                "• /help — show this\n"
                "• /sources — show current configured sources\n"
                "• /add <path> — add a folder as a source (persists to app_config.ini)\n"
                "• /reindex — rebuild index\n"
                "• /clear — clear chat\n"
                "You can also paste a file path to add+index it.\n"
                "• /ls <folder> — list files in a folder\n"
                "• /find <name or pattern> — search files in your configured sources"
            )
            self.ui.update_status("Ready")
            return True

        if cmd == "/clear":
            self.ui.clear_chat()
            self.ui.update_status("Ready")
            return True

        if cmd == "/sources":
            sources = (self.config.index_sources or []) + (self.config.extra_sources or [])
            if not sources:
                sources = [str(self.config.auto_docs_dir)]
            txt = "\n".join(f"• {s}" for s in sources)
            self.ui.append_assistant_message("Current sources:\n" + txt)
            self.ui.update_status("Ready")
            return True

        if cmd == "/reindex":
            self.ui.update_status("Re-indexing…")
            try:
                if self.config.index_dir.exists():
                    shutil.rmtree(self.config.index_dir, ignore_errors=True)
            except Exception:
                pass
            self._ensure_index()
            return True

        if cmd == "/add":
            if not arg:
                self.ui.append_assistant_message("Usage: /add <folder-path>")
                return True
            cleaned = arg.strip().strip('"').strip("'")
            p = Path(os.path.expandvars(os.path.expanduser(cleaned))).resolve()
            if not p.exists() or not p.is_dir():
                self.ui.append_assistant_message(f"That folder doesn't exist: {p}")
                return True
            return self._add_source_folder_and_reindex(p)

        if cmd in {"/ls", "/dir"}:
            if not arg:
                self.ui.append_assistant_message("Usage: /ls <folder-path>")
                return True
            return self._handle_list_dir_request(arg)

        if cmd == "/find":
            if not arg:
                self.ui.append_assistant_message("Usage: /find <filename or keywords>\nExample: /find invoice 2024.pdf")
                return True
            return self._handle_find_request(arg)

        return False

    # =========================
    # FIXED: these are now REAL methods on AppController
    # =========================
    def _extract_windows_path(self, text: str) -> Optional[str]:
        # Best-effort: grab a Windows path like C:\foo\bar or UNC paths.
        # Supports Chinese characters and parentheses.
        if not text:
            return None
        t = text.strip()

        # If the whole string is a path token already, return it.
        if re.search(r"^[A-Za-z]:\\", t) or t.startswith("\\"):
            return t

        # Look for the last drive path inside the string.
        m = re.search(r"([A-Za-z]:\\[^\n\r\t]+)", t)
        if m:
            return m.group(1).strip().strip('"').strip("'")

        # Look for UNC path
        m = re.search(r"(\\\\[^\n\r\t]+)", t)
        if m:
            return m.group(1).strip().strip('"').strip("'")

        return None

    def _format_dir_listing(self, folder: Path, entries: List[os.DirEntry], truncated: bool) -> str:
        lines = []
        lines.append(f"**Folder:** `{folder}`")
        lines.append("")
        dirs = [e for e in entries if e.is_dir(follow_symlinks=False)]
        files = [e for e in entries if e.is_file(follow_symlinks=False)]

        lines.append(
            f"**{len(dirs)} folder(s), {len(files)} file(s)**" + (" (showing first results)" if truncated else "")
        )
        lines.append("")
        if dirs:
            lines.append("**Folders**")
            for e in dirs:
                lines.append(f"• 📁 {e.name}")
            lines.append("")
        if files:
            lines.append("**Files**")
            for e in files:
                try:
                    size = e.stat(follow_symlinks=False).st_size
                except Exception:
                    size = None
                if size is None:
                    lines.append(f"• 📄 {e.name}")
                else:
                    lines.append(f"• 📄 {e.name} — {size:,} bytes")
        if not dirs and not files:
            lines.append("_Empty folder._")

        lines.append("")
        lines.append("Tip: paste a file path to add it to the index, or use `/add <folder>` to index the whole folder.")
        return "\n".join(lines)

    def _handle_list_dir_request(self, text: str) -> bool:
        # Accept direct folder path, or a natural-language request containing a path.
        raw = (text or "").strip()
        if not raw:
            return False

        # If user wrote "what files are in this location <path>", extract the path.
        candidate = self._extract_windows_path(raw)
        if candidate is None:
            # fallback: treat as-is
            candidate = raw

        candidate = candidate.strip().strip("<>[](){}.,;")
        candidate = os.path.expandvars(os.path.expanduser(candidate))
        folder = Path(candidate)

        if not folder.exists():
            self.ui.append_assistant_message(f"I can’t access that path (not found): `{folder}`")
            self.ui.update_status("Ready")
            return True
        if not folder.is_dir():
            self.ui.append_assistant_message(f"That isn’t a folder: `{folder}`")
            self.ui.update_status("Ready")
            return True

        # Respect allow_any_path: if disabled, only allow within base_dir
        if not self.config.allow_any_path:
            try:
                folder.resolve().relative_to(self.config.base_dir.resolve())
            except Exception:
                self.ui.append_assistant_message(
                    "That folder is outside your allowed base directory.\n"
                    f"Allowed base: `{self.config.base_dir}`\n\n"
                    "If you want to allow any path, set `allow_any_path = true` in app_config.ini."
                )
                self.ui.update_status("Ready")
                return True

        # List entries (limit for UI)
        max_entries = 200
        try:
            with os.scandir(str(folder)) as it:
                entries = sorted(list(it), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except Exception as e:
            self.ui.append_assistant_message(f"Couldn’t read that folder: `{folder}`\nError: {e}")
            self.ui.update_status("Ready")
            return True

        truncated = len(entries) > max_entries
        entries = entries[:max_entries]

        msg = self._format_dir_listing(folder, entries, truncated)
        self.ui.append_assistant_message(msg)
        self.ui.update_status("Ready")
        return True

    def _handle_find_request(self, query: str) -> bool:
        needle = (query or "").strip().strip('"').strip("'")
        if not needle:
            return False

        # Determine roots to search
        roots = []
        sources = (self.config.index_sources or []) + (self.config.extra_sources or [])
        if not sources:
            sources = [str(self.config.auto_docs_dir)]
        for src in sources:
            p = _expand_path(src)
            if p.exists():
                roots.append(p)

        # If allow_any_path is false, restrict to base_dir
        if not self.config.allow_any_path:
            base = self.config.base_dir.resolve()
            filtered = []
            for r in roots:
                try:
                    r.resolve().relative_to(base)
                    filtered.append(r)
                except Exception:
                    # skip roots outside base
                    continue
            roots = filtered or [base]

        # Search (best-effort) with limits
        include_globs = self.config.include_globs or []
        exclude_globs = self.config.exclude_globs or []
        supported_exts = {ext.lower() for ext in (self.config.supported_exts or [])}

        max_hits = 50
        max_scanned = 15000
        hits = []
        scanned = 0

        # simple tokenization for keyword matching
        tokens = [t for t in re.split(r"\s+", needle.lower()) if t]

        def is_excluded(p: Path) -> bool:
            if exclude_globs and _matches_globs(p, exclude_globs):
                return True
            if include_globs and not _matches_globs(p, include_globs):
                return True
            if supported_exts and p.suffix.lower() not in supported_exts:
                return True
            return False

        for root in roots:
            if scanned >= max_scanned or len(hits) >= max_hits:
                break
            if root.is_file():
                scanned += 1
                rp = root
                name = rp.name.lower()
                if all(t in name for t in tokens) and not is_excluded(rp):
                    hits.append(rp)
                continue

            try:
                for p in root.rglob("*"):
                    if scanned >= max_scanned or len(hits) >= max_hits:
                        break
                    scanned += 1
                    if not p.is_file():
                        continue
                    if is_excluded(p):
                        continue
                    name = p.name.lower()
                    if all(t in name for t in tokens):
                        hits.append(p)
            except Exception:
                continue

        if not hits:
            self.ui.append_assistant_message(
                f"No matches found for `{needle}` (scanned ~{scanned:,} paths).\n\n"
                "Try a shorter query or add more sources with `/add <folder>`."
            )
            self.ui.update_status("Ready")
            return True

        lines = [f"Found **{len(hits)}** match(es) for `{needle}` (scanned ~{scanned:,} paths):", ""]
        for p in hits[:max_hits]:
            lines.append(f"• `{p}`")
        if len(hits) >= max_hits:
            lines.append("")
            lines.append(f"_Showing first {max_hits} results._")

        lines.append("")
        lines.append("Tip: paste any file path above to add+index it automatically.")
        self.ui.append_assistant_message("\n".join(lines))
        self.ui.update_status("Ready")
        return True

    # =========================
    # AI Tool Router (agent control)
    # =========================
    @staticmethod
    def _extract_first_json(text: str) -> Optional[dict]:
        """Best-effort: extract the first JSON object from a model response."""
        if not text:
            return None
        # direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # try to find a JSON object in the text
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        return None
        return None

    def _agent_route(self, user_text: str) -> bool:
        """Let the LLM decide whether to invoke a local tool (filesystem / indexing / UI control).
        Returns True if a tool action was executed and we should stop normal Q&A for this turn.
        """
        if not getattr(self.config, "agent_enable_control", False):
            return False

        # Provide tool schema + context to the model
        allow_any = bool(getattr(self.config, "allow_any_path", False))
        base_dir = str(getattr(self.config, "base_dir", Path.cwd()))
        sources = (self.config.index_sources or []) + (self.config.extra_sources or [])
        if not sources:
            sources = [str(self.config.auto_docs_dir)]

        tools = [
            {"type": "read_file", "args": {"path": "string"}}, 
            {"type": "list_dir", "args": {"path": "string"}, "when": "User wants to know what files/folders are inside a directory."},
            {"type": "find_files", "args": {"query": "string"}, "when": "User wants you to locate a file on the computer by name/keywords/pattern."},
            {"type": "add_source", "args": {"path": "string"}, "when": "User asks to add a folder so it will be indexed going forward."},
            {"type": "reindex", "args": {}, "when": "User asks to rebuild/reindex."},
            {"type": "clear_chat", "args": {}, "when": "User asks to clear chat."},
            {"type": "answer", "args": {}, "when": "Normal document Q&A (no local tool needed)."},
        ]

        controller_prompt = f"""You are the local AI asistant for company information holder and answering according to the information.
Your job: decide whether to call a LOCAL TOOL (filesystem or app control) or just answer.

RULES:
-always give complete answer, no guessing always to the point from the provided document, if user provides a document name answer from that document only.
-no matter how long it might take, make sure to get all the data from the chunks before answering.
-dont behave like an AI, be professorional and straight to the point.
-You are an expert in understanding user requests about documents, company rules, and file management.
- If the user mentions a filename (e.g., "st10", "ST10.xlsx") but does NOT provide a folder path, answer accordingly .
- "what is in <file>" or "what are in <file>" means the user wants the file content, not the folder listing.
-always think step by step about what the user is asking before answering.
-if user ask about files or directories first think what it could be then answer.
-if user ask a question first check history then answer.
-dont assume that user is asking for files or directories, only do so when its clear from the context.
-if user ask a question first think what it could be then answer, dont go looking for files or directories unless its clear from the context.
- If the user asks about folder contents OR gives a folder path, choose list_dir.
-if the user asks what documents they have check the index folder, or show the files in index folder.
- If the user asks you to locate a file anywhere on the computer, choose find_files.
- If the user asks to add a folder to be indexed, choose add_source.
- If none apply, choose answer. 
-do not hallucinate.

SECURITY / SCOPE:
- allow_any_path = {str(allow_any).lower()}
- base_dir = {base_dir}
- configured_sources = {json.dumps(sources, ensure_ascii=False)}

TOOLS:
{json.dumps(tools, ensure_ascii=False)}

Return schema:
{{
  "action": "<one of: list_dir|find_files|add_source|reindex|clear_chat|answer>",
  "args": {{ ... }},
  "user_visible": "optional short status line to show while executing"
}}

User message:
{user_text}
"""

        # Ask the SAME local model (Ollama) to decide. We keep this fast (low tokens).
        try:
            raw = self.rag._ollama_generate(controller_prompt, num_predict=220)
        except Exception:
            return False

        obj = self._extract_first_json(raw)
        if not obj:
            return False

        action = str(obj.get("action", "")).strip()
        args = obj.get("args", {}) if isinstance(obj.get("args", {}), dict) else {}
        status = str(obj.get("user_visible", "")).strip()
        if status:
            self.ui.update_status(status)

        # Execute actions
        if action == "clear_chat":
            self.ui.clear_chat()
            self.ui.update_status("Ready")
            return True

        if action == "reindex":
            self.ui.update_status("Rebuilding index…")
            self._start_reindex_async()
            return True

        if action == "add_source":
            p = str(args.get("path", "")).strip()
            if not p:
                return False
            self.ui.update_status("Adding folder source…")
            return self._handle_add_source(Path(p))

        if action == "list_dir":
            p = str(args.get("path", "")).strip()
            if not p:
                # fall back: try to extract a path from the user text
                p = self._extract_windows_path(user_text) or ""
            if not p:
                return False
            return self._handle_list_dir_request(p)

        if action == "find_files":
            q = str(args.get("query", "")).strip() or user_text
            self.ui.update_status("Searching files…")
            return self._handle_find_request(q)

        # action == "answer" (or unknown): let normal pipeline handle it
        return False

    def _indexed_doc_tokens(self) -> Set[str]:
        docs = self.rag.get_indexed_docs()
        tokens: Set[str] = set()
        for doc in docs:
            p = Path(doc)
            if p.name:
                tokens.add(p.name.lower())
            if p.stem:
                tokens.add(p.stem.lower())
        return tokens

    def _has_indexed_doc_reference(self, user_text: str) -> bool:
        if not user_text:
            return False
        tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9_.-]+", user_text)}
        if not tokens:
            return False
        indexed_tokens = self._indexed_doc_tokens()
        return bool(tokens & indexed_tokens)

    def _looks_like_filename(self, user_text: str) -> bool:
        supported_exts = {ext.lower().lstrip(".") for ext in (self.config.supported_exts or [])}
        if not supported_exts:
            return False
        pattern = r"\b[\w\-.]+\.(?:" + "|".join(re.escape(ext) for ext in supported_exts) + r")\b"
        return bool(re.search(pattern, user_text, flags=re.IGNORECASE))

    def _should_attempt_agent_route(self, user_text: str) -> bool:
        if not getattr(self.config, "agent_enable_control", False):
            return False
        raw = (user_text or "").strip()
        if not raw:
            return False

        # If the user references an indexed doc, treat it as a content question.
        if self._has_indexed_doc_reference(raw):
            return False

        lowered = raw.lower()
        if raw.startswith("/"):
            return True
        if self._extract_windows_path(raw):
            return True
        if re.search(r"\b(add|index|reindex|rebuild|clear)\b", lowered):
            return True
        if re.search(r"\b(find|locate|search)\b.*\b(file|document|doc)\b", lowered):
            return True
        if re.search(r"\b(list|show|open|browse)\b.*\b(folder|directory|files?|contents?)\b", lowered):
            return True
        if self._looks_like_filename(raw):
            return True

        return False

    def _handle_indexed_docs_query(self, user_text: str) -> bool:
        raw = (user_text or "").strip()
        if not raw:
            return False
        lowered = raw.lower()
        if not re.search(r"\b(list|show|what)\b", lowered):
            return False
        if not re.search(r"\b(indexed|index)\b", lowered):
            return False
        if not re.search(r"\b(documents|docs|files)\b", lowered):
            return False

        docs = self.rag.get_indexed_docs()
        if not docs:
            self.ui.append_assistant_message("No documents are indexed yet. Add files or folders to start indexing.")
            self.ui.update_status("Ready")
            return True

        lines = [f"Indexed documents ({len(docs)}):", ""]
        for doc in docs:
            lines.append(f"• {doc}")
        self.ui.append_assistant_message("\n".join(lines))
        self.ui.update_status("Ready")
        return True

    def _start_reindex_async(self):
        """Kick off a background reindex using the existing WorkerThread."""
        def task():
            progress = lambda s: self._thread.progress.emit(s) if self._thread else None
            progress("Collecting files…")
            files = collect_auto_docs(self.config)
            if not files:
                progress("No files found to index.")
                return False
            progress(f"Indexing {len(files)} file(s)…")
            self.rag.build_index(
                files,
                progress=progress,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                enable_ocr=self.config.enable_ocr,
                ocr_lang=self.config.ocr_lang,
            )
            return True

        self._start_worker(task, on_ok=lambda _: self.ui.update_status("Ready"))

    def _handle_add_source(self, folder: Path) -> bool:
        """Persist a folder into app_config.ini as an extra source, then reindex."""
        folder = _expand_path(str(folder))
        if not folder.exists() or not folder.is_dir():
            self.ui.append_assistant_message(f"That folder doesn’t exist: {folder}")
            self.ui.update_status("Ready")
            return True

        # Scope restriction unless allow_any_path
        if not self.config.allow_any_path:
            try:
                folder.resolve().relative_to(self.config.base_dir.resolve())
            except Exception:
                self.ui.append_assistant_message(
                    "That folder is outside the allowed base directory.\n"
                    "Either move it under the app folder, or set allow_any_path=true in app_config.ini."
                )
                self.ui.update_status("Ready")
                return True

        # Add to extra_sources and persist
        extras = list(self.config.extra_sources or [])
        sf = str(folder)
        if sf not in extras:
            extras.append(sf)
            self.config.extra_sources = extras
            # write back
            ini = load_ini_with_backup(self.config_path)
            if not ini.has_section("app"):
                ini.add_section("app")
            ini.set("app", "extra_sources", json.dumps(extras, ensure_ascii=False))
            atomic_write_ini(self.config_path, ini)

        self.ui.append_assistant_message(f"Added folder source: {folder}\nRebuilding index…")
        self._start_reindex_async()
        return True

    def on_send(self):
        q = self.ui.get_user_text()
        if not q:
            return

        self.ui.clear_user_text()
        self.ui.append_user_message(q)
        self.ui.update_sources_text("")
        self.ui.update_status("Preparing…")

        if self._pending_file_choices:
            if self._handle_pending_file_choice(q):
                return

        # Fallback deterministic handlers (paths/commands)
        if self._handle_path_message(q):
            return



        # Let the model decide whether to invoke a local tool (filesystem/app control)
        if self._should_attempt_agent_route(q) and self._agent_route(q):
            return

        def _after_index(_):
            self.ui.begin_assistant_stream()

            def do_answer():
                progress = lambda s: self._thread.progress.emit(s) if self._thread else None
                on_token = lambda t: self._thread.partial.emit(t) if self._thread else None
                top_k_val = self.config.retrieval_top_k

                if self.config.agent_enable_control:
                    initial = self.rag.generate_answer(
                        q,
                        progress=progress,
                        top_k=top_k_val,
                        allow_control=True,
                    )

                    clean_text, actions = parse_control_blocks(initial.get("answer", ""))
                    if not actions:
                        streamed = self.rag.answer_question_stream(
                            q,
                            progress=progress,
                            top_k=top_k_val,
                            on_token=on_token,
                        )
                        return {
                            "answer": streamed.get("answer", ""),
                            "sources": streamed.get("sources", []),
                            "actions": [],
                            "config": None,
                        }

                    summaries, config_changed, index_changed, listed_sources = self._apply_control_actions(
                        actions,
                        q,
                        progress=progress,
                    )

                    updated_config = self.config
                    if config_changed:
                        updated_config = load_app_config(self.config_path)
                        self._apply_config_to_runtime(updated_config)

                    rebuild_requested = any(
                        action.get("type") in {"REBUILD_INDEX", "RESCAN_DISK"}
                        for action in actions
                    )
                    source_changed = any(
                        action.get("type") in {"ADD_SOURCE", "REMOVE_SOURCE"}
                        for action in actions
                    )
                    clear_requested = any(action.get("type") == "CLEAR_INDEX" for action in actions)

                    if index_changed and not clear_requested and (
                        rebuild_requested or (source_changed and updated_config.agent_auto_rebuild_index)
                    ):
                        files = collect_auto_docs(updated_config)
                        if files:
                            self.rag.build_index(
                                files,
                                progress=progress,
                                chunk_size=updated_config.chunk_size,
                                overlap=updated_config.chunk_overlap,
                                enable_ocr=updated_config.enable_ocr,
                                ocr_lang=updated_config.ocr_lang,
                                enable_libreoffice=updated_config.enable_libreoffice,
                            )

                    final = self.rag.answer_question_stream(
                        q,
                        progress=progress,
                        top_k=updated_config.retrieval_top_k,
                        on_token=on_token,
                    )
                    final_answer = final.get("answer", "")

                    if listed_sources:
                        summaries.append("Sources: " + "; ".join(listed_sources))

                    return {
                        "answer": final_answer,
                        "sources": final.get("sources", []),
                        "actions": summaries,
                        "config": updated_config,
                    }

                streamed = self.rag.answer_question_stream(
                    q,
                    progress=progress,
                    top_k=top_k_val,
                    on_token=on_token,
                )
                return {
                    "answer": streamed.get("answer", ""),
                    "sources": streamed.get("sources", []),
                    "actions": [],
                    "config": None,
                }

            def on_ok(res):
                ans = res.get("answer", "")
                sources = res.get("sources", [])
                actions = res.get("actions") or []
                if actions:
                    ans = ans + "\n\nActions taken: " + "; ".join(actions)
                self.ui.finalize_assistant_stream(ans)
                self.ui.update_sources_text("\n".join(sources))
                if res.get("config"):
                    self._apply_config_to_ui(res["config"])
                self.ui.update_status("Done.")

            def on_err(err):
                self.ui.finalize_assistant_stream("❌ Error while generating.\n\n" + err)
                QMessageBox.critical(self, "Answer Error", err)

            self._start_worker(
                fn=do_answer,
                on_ok=on_ok,
                on_err=on_err,
                on_partial=self.ui.update_assistant_stream,
            )

        self._ensure_index(after_ok=_after_index)

    def _copy_to_auto_docs(self, src: Path) -> Path:
        dest_dir = self.config.auto_docs_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            if src.resolve() == (dest_dir / src.name).resolve():
                return dest_dir / src.name
        except Exception:
            pass

        dest = dest_dir / src.name
        if dest.exists():
            stem = src.stem
            suffix = src.suffix
            counter = 1
            while True:
                candidate = dest_dir / f"{stem}_{counter}{suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                counter += 1

        shutil.copy2(src, dest)
        return dest

    def on_add_files(self):
        if self._busy:
            return

        start_dir = str(self.config.base_dir)
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Add files to index", start_dir, "All files (*.*)")
        if not file_paths:
            return

        copied: List[Path] = []
        failures: List[str] = []

        for file_path in file_paths:
            try:
                copied.append(self._copy_to_auto_docs(Path(file_path)))
            except Exception as exc:
                failures.append(f"{file_path}: {exc}")

        if failures:
            QMessageBox.warning(
                self,
                "File copy warning",
                "Some files could not be added:\n\n" + "\n".join(failures[:10]),
            )

        if copied:
            self.ui.update_status(f"Added {len(copied)} file(s). Indexing new files…")
            self._ensure_index()

    def on_settings(self):
        ini = self.rag.load_ini()
        current_lang = ini.get("user", {}).get("language", "en") or "en"
        user_name = ini.get("user", {}).get("name", "Mir")

        docs = collect_auto_docs(self.config)
        indexed = self.rag.get_indexed_docs()
        summary = self.rag.get_chat_summary()

        ocr_on = self.config.enable_ocr
        lo_on = self.config.enable_libreoffice
        ocr_lang = self.config.ocr_lang

        info = (
            f"自动文档文件夹： {self.config.auto_docs_dir}\n"
            f"目前找到的文件：{len(docs)}\n"
            f"已索引的文档： {len(indexed)}\n"
            f"OCR={'ON' if ocr_on else 'OFF'} (lang={ocr_lang})\n"
            f"LibreOffice={'ON' if lo_on else 'OFF'}\n"
            f"人工智能记忆摘要:\n{summary}\n"
        )

        dlg = self.ui.open_settings_dialog(info, current_lang)

        def do_rebuild():
            dlg.accept()
            self.rag.clear_index()
            self.ui.update_status("Index cleared. Rebuilding…")
            self._ensure_index()

        def do_clear_index():
            if QMessageBox.question(
                self,
                "Confirm Delete",
                "Delete ALL indexed memory?\n\nThis removes FAISS + chunks + docs list.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ) == QMessageBox.Yes:
                self.rag.clear_index()
                self.ui.update_sources_text("")
                self.ui.update_status("Index cleared. Place docs and send a message to rebuild.")
                dlg.accept()

        def do_clear_ai_memory():
            if QMessageBox.question(
                self,
                "Confirm Delete",
                "Delete AI chat memory?\n\n"
                "This removes:\n"
                "- chat_history.jsonl\n"
                "- chat_summary.txt\n\n"
                "Document index will NOT be affected.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ) == QMessageBox.Yes:
                self.rag.clear_memory()
                self.ui.update_status("AI memory cleared.")
                dlg.accept()

        def on_close_apply_language():
            selected = dlg.get_selected_language()
            ini2 = self.rag.load_ini()
            ini2.setdefault("user", {})
            ini2["user"]["language"] = selected
            self.rag.save_ini(ini2)

            self.ui.apply_language(selected, user_name=user_name)
            dlg.accept()

        dlg.rebuild_btn.clicked.connect(do_rebuild)
        dlg.clear_btn.clicked.connect(do_clear_index)
        dlg.clear_memory_btn.clicked.connect(do_clear_ai_memory)

        dlg.close_btn.clicked.disconnect()
        dlg.close_btn.clicked.connect(on_close_apply_language)

        dlg.exec_()


def excepthook(exc_type, exc, tb):
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        QMessageBox.critical(None, "Unhandled Exception", msg)
    except Exception:
        print(msg, file=sys.stderr)


def load_app_icon(config) -> QIcon:
    if config.app_icon_ico.exists():
        return QIcon(str(config.app_icon_ico))
    if config.app_icon_png.exists():
        return QIcon(str(config.app_icon_png))
    return QIcon()


def main():
    sys.excepthook = excepthook
    set_windows_appusermodelid("offline_doc_assistant.DocumentAIAssistant")

    app = QApplication(sys.argv)

    cfg = load_app_config(CONFIG_PATH)
    icon = load_app_icon(cfg)
    if not icon.isNull():
        app.setWindowIcon(icon)

    w = AppController()
    if not icon.isNull():
        w.setWindowIcon(icon)

    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
