import os
import sys
import traceback
from pathlib import Path

# --- ADD THIS near the top of app.py (imports) ---
import subprocess
import time
import requests

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
    QMessageBox,
    QWidget,
    QToolButton,
)

from app_config import apply_runtime_env, load_app_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("app_config.ini")
CONFIG_PATH = Path(os.environ.get("DOC_ASSISTANT_CONFIG", str(DEFAULT_CONFIG_PATH)))
APP_CONFIG = load_app_config(CONFIG_PATH)
apply_runtime_env(APP_CONFIG)

from rag_engine import RAGEngine
from ui_main import MainWindow
from worker import WorkerThread

AUTO_DOCS_DIR = APP_CONFIG.auto_docs_dir
BASE_DIR = APP_CONFIG.base_dir
EMBED_DIR = APP_CONFIG.embed_model_dir
DATA_DIR = APP_CONFIG.data_dir
INDEX_DIR = APP_CONFIG.index_dir
LOG_DIR = APP_CONFIG.log_dir

SUPPORTED_EXTS = {
    # text-like
    ".txt", ".md", ".mdx", ".log", ".ini", ".cfg", ".conf", ".env", ".xml",
    ".yml", ".yaml", ".tsv", ".tex",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs",
    ".go", ".rs", ".rb", ".php", ".sql", ".sh", ".bat", ".ps1",
    ".toml", ".gitignore",

    # office + common
    ".pdf",
    ".docx", ".doc", ".odt",
    ".xlsx", ".xls", ".xlsm", ".xltx", ".xltm", ".ods",
    ".pptx", ".ppt", ".ppsx", ".potx", ".odp",
    ".csv", ".json", ".jsonl", ".ndjson",
    ".html", ".htm",
    ".rtf",
    ".epub",

    # images (OCR optional)
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
}

APP_ICON_ICO = APP_CONFIG.app_icon_ico
APP_ICON_PNG = APP_CONFIG.app_icon_png


def set_windows_appusermodelid(app_id: str) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def collect_auto_docs() -> list[str]:
    if not AUTO_DOCS_DIR.exists():
        return []
    docs: list[str] = []
    for p in AUTO_DOCS_DIR.rglob("*"):
        if not p.is_file():
            continue
        # Skip temporary office files (e.g., ~\$ doc names)
        if p.name.startswith("~$"):
            continue
        # If extension is supported, include it; otherwise include anyway to attempt conversion
        docs.append(str(p))
    return docs


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "1" if default else "0").strip().lower()
    return v in {"1", "true", "yes", "on"}


class AppController(QWidget):
    def __init__(self):
        super().__init__()
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


        window_flags = Qt.FramelessWindowHint | Qt.Window
        if APP_CONFIG.always_on_top:
            window_flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(window_flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setMouseTracking(True)

        self.setWindowTitle(APP_CONFIG.window_title)
        self.setMinimumSize(APP_CONFIG.min_width, APP_CONFIG.min_height)

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
            base_dir=BASE_DIR,
            embed_model_dir=EMBED_DIR,
            index_dir=INDEX_DIR,
            log_dir=LOG_DIR,
            ollama_base_url=APP_CONFIG.ollama_base_url,
            ollama_model=APP_CONFIG.ollama_model,
        )

        self._busy = False
        self._thread: WorkerThread | None = None

        ini = self.rag.load_ini()
        user_name = ini.get("user", {}).get("name", "Mir")
        lang = ini.get("user", {}).get("language", "en") or "en"
        self.ui.apply_language(lang, user_name=user_name)
        self.ui.set_logo_height(APP_CONFIG.logo_height_px)

        self.ui.send_button.clicked.connect(self.on_send)

        # show OCR/LO settings hint in status (driven by env vars)
        ocr_on = _bool_env("ENABLE_OCR", True)
        lo_on = _bool_env("ENABLE_LIBREOFFICE", True)
        self.ui.update_status(
            f"Ready. Place documents in {AUTO_DOCS_DIR} "
            f"(OCR={'ON' if ocr_on else 'OFF'}, LibreOffice={'ON' if lo_on else 'OFF'})"
        )
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
        if not settings.get("enable_ocr", _bool_env("ENABLE_OCR", False)):
            hints.append("Tip: set ENABLE_OCR=1 to index scanned PDFs/images (after installing OCR deps).")
        if not settings.get("enable_libreoffice", _bool_env("ENABLE_LIBREOFFICE", True)):
            hints.append("Tip: set ENABLE_LIBREOFFICE=1 (and install LibreOffice) to index .doc/.odt/.ods/.odp/.ppt/.xls.")

        msg = "Some files were not indexed:\n\n" + "\n".join(lines)
        if hints:
            msg += "\n\n" + "\n".join(hints)

        QMessageBox.information(self, "Indexing report", msg)

    def _ensure_index(self, after_ok=None):
        files = collect_auto_docs()
        if not files:
            self.ui.update_status(f"No documents found in: {AUTO_DOCS_DIR}")
            QMessageBox.warning(
                self,
                "No documents found",
                f"No documents found in:\n{AUTO_DOCS_DIR}\n\n"
                f"Place files of any type into this folder to index them.",
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
            # Read chunk sizing from environment; default to full document (0) for chunk_size and 0 for overlap
            cs_env = os.environ.get("CHUNK_SIZE", "").strip()
            ov_env = os.environ.get("CHUNK_OVERLAP", "").strip()
            try:
                chunk_size_val = int(cs_env) if cs_env else 0
            except Exception:
                chunk_size_val = 0
            try:
                overlap_val = int(ov_env) if ov_env else 0
            except Exception:
                overlap_val = 0
            return self.rag.build_index(
                files,
                progress=lambda s: self._thread.progress.emit(s) if self._thread else None,
                chunk_size=chunk_size_val,
                overlap=overlap_val,
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

    def on_send(self):
        q = self.ui.get_user_text()
        if not q:
            return

        self.ui.clear_user_text()
        self.ui.append_user_message(q)
        self.ui.update_sources_text("")
        self.ui.update_status("Preparing…")

        def _after_index(_):
            self.ui.begin_assistant_stream()

            def do_answer_stream():
                # Use configurable TOP_K for retrieval; default to 6 for fast answering
                top_k_env = os.environ.get("TOP_K", "").strip()
                try:
                    top_k_val = int(top_k_env) if top_k_env else 6
                except Exception:
                    top_k_val = 6
                return self.rag.answer_question_stream(
                    q,
                    progress=lambda s: self._thread.progress.emit(s) if self._thread else None,
                    top_k=top_k_val,
                    on_token=lambda tok: self._thread.partial.emit(tok) if self._thread else None,
                )

            def on_ok(res):
                ans = res.get("answer", "")
                sources = res.get("sources", [])
                self.ui.finalize_assistant_stream(ans)
                self.ui.update_sources_text("\n".join(sources))
                self.ui.update_status("Done.")

            def on_err(err):
                self.ui.finalize_assistant_stream("❌ Error while generating.\n\n" + err)
                QMessageBox.critical(self, "Answer Error", err)

            self._start_worker(
                fn=do_answer_stream,
                on_ok=on_ok,
                on_err=on_err,
                on_partial=self.ui.update_assistant_stream,
            )

        self._ensure_index(after_ok=_after_index)

    def on_settings(self):
        ini = self.rag.load_ini()
        current_lang = ini.get("user", {}).get("language", "en") or "en"
        user_name = ini.get("user", {}).get("name", "Mir")

        docs = collect_auto_docs()
        indexed = self.rag.get_indexed_docs()
        summary = self.rag.get_chat_summary()

        ocr_on = _bool_env("ENABLE_OCR", True)
        lo_on = _bool_env("ENABLE_LIBREOFFICE", True)
        ocr_lang = os.environ.get("OCR_LANG", "eng")

        info = (
            f"自动文档文件夹： {AUTO_DOCS_DIR}\n"
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


def load_app_icon() -> QIcon:
    if APP_ICON_ICO.exists():
        return QIcon(str(APP_ICON_ICO))
    if APP_ICON_PNG.exists():
        return QIcon(str(APP_ICON_PNG))
    return QIcon()


def main():
    sys.excepthook = excepthook
    set_windows_appusermodelid("offline_doc_assistant.DocumentAIAssistant")

    app = QApplication(sys.argv)

    icon = load_app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    w = AppController()
    if not icon.isNull():
        w.setWindowIcon(icon)

    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
