import re
from pathlib import Path
import os
from PyQt5.QtCore import QEasingCurve, QEvent, QPropertyAnimation, Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
)

from formatters import to_rich_html, parse_json_rows, md_table_to_grid


LANG_OPTIONS = [
    ("auto", "Auto"),
    ("en", "English"),
    ("zh", "中文"),
    ("th", "ไทย"),
]


UI_STRINGS = {
    "en": {
        "title": "Document AI Assistant",
        "settings": "Settings",
        "greeting": "Nice to see you, {name}. What’s new?",
        "input_placeholder": "Ask me anything related to your documents, \nIf new documents were added, please rebuild the index from setting page.",
        "sources": "Sources",
        "status_start": "Starting…",
        "btn_rebuild": "Rebuild index now",
        "btn_clear_index": "Delete ALL indexed memory (FAISS + metadata)",
        "btn_clear_ai_memory": "Delete AI chat memory",
        "btn_close": "Close",
        "lang_label": "Language",
        "lang_hint": "UI + AI language",
        "sheet_title": "Sheet (CSV/Excel view)",
    },
    "zh": {
        "title": "文档 AI 助手",
        "settings": "设置",
        "greeting": "很高兴见到你，{name}。有什么新消息？",
        "input_placeholder": "您可以询问任何与您的文档相关的问题。\n如果添加了新文档，请从设置页面重建索引。",
        "sources": "来源",
        "status_start": "启动中…",
        "btn_rebuild": "立即重建索引",
        "btn_clear_index": "删除全部已索引数据（FAISS + 元数据）",
        "btn_clear_ai_memory": "删除 AI 对话记忆",
        "btn_close": "关闭",
        "lang_label": "语言",
        "lang_hint": "界面 + AI 语言",
        "sheet_title": "表格（CSV/Excel 视图）",
    },
    "th": {
        "title": "ผู้ช่วย AI เอกสาร",
        "settings": "การตั้งค่า",
        "greeting": "ยินดีที่ได้พบคุณ, {name}. มีอะไรใหม่บ้าง?",
        "input_placeholder": "คุณสามารถสอบถามฉันได้ทุกเรื่องที่เกี่ยวข้องกับเอกสารของคุณ\n หากมีการเพิ่มเอกสารใหม่ โปรดสร้างดัชนีใหม่จากหน้าการตั้งค่า",
        "sources": "แหล่งอ้างอิง",
        "status_start": "กำลังเริ่ม…",
        "btn_rebuild": "สร้างดัชนีใหม่ตอนนี้",
        "btn_clear_index": "ลบข้อมูลที่จัดทำดัชนีทั้งหมด (FAISS + metadata)",
        "btn_clear_ai_memory": "ลบหน่วยความจำแชท AI",
        "btn_close": "ปิด",
        "lang_label": "ภาษา",
        "lang_hint": "ภาษา UI + AI",
        "sheet_title": "ตาราง (มุมมอง CSV/Excel)",
    },
}


def _pick_strings(lang_code: str) -> dict:
    if lang_code in UI_STRINGS:
        return UI_STRINGS[lang_code]
    return UI_STRINGS["en"]


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setModal(True)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        self.setMinimumWidth(560)

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        self.title = QLabel("Settings")
        self.title.setObjectName("dlgTitle")

        self.info = QLabel("")
        self.info.setObjectName("dlgInfo")
        self.info.setWordWrap(True)

        lang_row = QFrame()
        lang_row.setObjectName("langRow")
        lang_row.setAttribute(Qt.WA_StyledBackground, True)
        lr = QHBoxLayout(lang_row)
        lr.setContentsMargins(12, 10, 12, 10)
        lr.setSpacing(10)

        self.lang_label = QLabel("Language")
        self.lang_label.setObjectName("langLabel")

        self.lang_hint = QLabel("UI + AI language")
        self.lang_hint.setObjectName("langHint")

        self.lang_combo = QComboBox()
        self.lang_combo.setObjectName("langCombo")
        for code, label in LANG_OPTIONS:
            self.lang_combo.addItem(label, code)
        self.lang_combo.setCurrentIndex(0)

        lr.addWidget(self.lang_label, 0)
        lr.addStretch(1)
        lr.addWidget(self.lang_hint, 0)
        lr.addSpacing(8)
        lr.addWidget(self.lang_combo, 0)

        self.rebuild_btn = QPushButton("Rebuild index now")
        self.rebuild_btn.setObjectName("dlgPrimary")

        self.clear_btn = QPushButton("Delete ALL indexed memory (FAISS + metadata)")
        self.clear_btn.setObjectName("dlgDanger")

        self.clear_memory_btn = QPushButton("Delete AI chat memory")
        self.clear_memory_btn.setObjectName("dlgDanger")

        self.close_btn = QPushButton("Close")
        self.close_btn.setObjectName("dlgGhost")
        self.close_btn.clicked.connect(self.accept)

        root.addWidget(self.title)
        root.addWidget(self.info)
        root.addSpacing(6)
        root.addWidget(lang_row)
        root.addSpacing(6)
        root.addWidget(self.rebuild_btn)
        root.addWidget(self.clear_btn)
        root.addWidget(self.clear_memory_btn)
        root.addSpacing(10)
        root.addWidget(self.close_btn)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QDialog {
                background: #0b0f19;
                color: #e8eefc;
                font-family: "Segoe UI";
            }
            QLabel#dlgTitle {
                font-size: 18px;
                font-weight: 800;
                color: #eef3ff;
            }
            QLabel#dlgInfo {
                color: rgba(232,238,252,0.82);
            }

            QFrame#langRow {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 14px;
            }
            QLabel#langLabel {
                font-weight: 800;
                color: #eef3ff;
                background: transparent;
                border: none;
            }
            QLabel#langHint {
                color: rgba(232,238,252,0.72);
                font-size: 12px;
                background: transparent;
                border: none;
            }
            QComboBox#langCombo {
                padding: 7px 12px;
                border-radius: 12px;
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.12);
                color: #eef3ff;
                min-width: 140px;
                font-weight: 700;
            }
            QComboBox#langCombo::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox#langCombo:hover {
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.18);
            }
            QComboBox QAbstractItemView {
                background: #0b0f19;
                color: #eef3ff;
                selection-background-color: rgba(0,120,212,0.35);
                border: 1px solid rgba(255,255,255,0.12);
                outline: 0px;
            }

            QPushButton {
                padding: 12px 14px;
                border-radius: 14px;
                font-weight: 700;
            }
            QPushButton#dlgPrimary {
                background: rgba(0, 120, 212, 0.22);
                border: 1px solid rgba(0, 120, 212, 0.45);
                color: #eef3ff;
            }
            QPushButton#dlgPrimary:hover {
                background: rgba(0, 120, 212, 0.30);
                border: 1px solid rgba(0, 120, 212, 0.60);
            }
            QPushButton#dlgDanger {
                background: rgba(220, 38, 38, 0.18);
                border: 1px solid rgba(220, 38, 38, 0.35);
                color: #eef3ff;
            }
            QPushButton#dlgDanger:hover {
                background: rgba(220, 38, 38, 0.26);
                border: 1px solid rgba(220, 38, 38, 0.52);
            }
            QPushButton#dlgGhost {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.10);
                color: #eef3ff;
            }
            QPushButton#dlgGhost:hover {
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.18);
            }
            """
        )

    def set_language_ui(self, lang_code: str):
        s = _pick_strings(lang_code)
        self.title.setText(s["settings"])
        self.lang_label.setText(s["lang_label"])
        self.lang_hint.setText(s["lang_hint"])
        self.rebuild_btn.setText(s["btn_rebuild"])
        self.clear_btn.setText(s["btn_clear_index"])
        self.clear_memory_btn.setText(s["btn_clear_ai_memory"])
        self.close_btn.setText(s["btn_close"])

    def set_selected_language(self, lang_code: str):
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == lang_code:
                self.lang_combo.setCurrentIndex(i)
                return
        self.lang_combo.setCurrentIndex(0)

    def get_selected_language(self) -> str:
        return str(self.lang_combo.currentData())


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._anims = []
        self._scale = 1.0
        self._base_font_pt = 14
        self._logo_pixmap = None
        self._lang = "en"
        self._user_name = "Mir"
        self._logo_height_px = 65

        # Chat rendering state (for streaming)
        self._chat_msgs = []  # list of dicts: {"role": "user"|"assistant", "text": str}
        self._stream_index = None  # index in _chat_msgs of active streaming assistant message

        # Streaming smoothing (prevents UI jitter / "shaking")
        self._is_streaming = False
        self._pending_render = False
        self._stream_render_timer: QTimer | None = None

        # ✅ Smart auto-scroll (do NOT force-scroll while user reads older messages)
        self._stick_to_bottom = True
        self._programmatic_scroll = False

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

        self._build_ui()

        # ✅ ENTER = Send (Shift+Enter = newline)
        self.input_box.installEventFilter(self)

        self._apply_style()
        self._apply_logo_size()

        self._apply_auto_scale()
        self._fade_in()

    # ✅ ENTER-to-send handler
    def eventFilter(self, obj, event):
        if obj is self.input_box and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Return, Qt.Key_Enter):
                # Shift+Enter => newline
                if event.modifiers() & Qt.ShiftModifier:
                    return False
                # Enter => send
                self.send_button.click()
                return True
        return super().eventFilter(obj, event)

    # ---------- Logo sizing ----------
    def _apply_logo_size(self):
        if self._logo_pixmap and not self._logo_pixmap.isNull():
            pm = self._logo_pixmap.scaledToHeight(self._logo_height_px, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pm)
            self.logo_label.setFixedSize(pm.size())
            self.logo_label.updateGeometry()

    def set_logo_height(self, height_px: int):
        self._logo_height_px = max(8, int(height_px))
        self._apply_logo_size()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.content = QFrame()
        self.content.setObjectName("contentCard")
        self.content.setAttribute(Qt.WA_StyledBackground, True)
        root.addWidget(self.content)

        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(26, 22, 26, 22)
        content_layout.setSpacing(18)

        top = QFrame()
        top.setObjectName("topBar")
        top.setAttribute(Qt.WA_StyledBackground, True)
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(12)

        self.logo_label = QLabel()
        self.logo_label.setObjectName("logoLabel")
        self.logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.logo_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.logo_label.setAttribute(Qt.WA_TranslucentBackground, True)

        logo_path = Path(r"D:\offline_doc_assistant\logo.png")
        if logo_path.exists():
            pm = QPixmap(str(logo_path))
            if not pm.isNull():
                self._logo_pixmap = pm
                self.logo_label.setPixmap(pm)

        self.greeting = QLabel("Nice to see you. What’s new?")
        self.greeting.setObjectName("greeting")
        self.greeting.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.settings_button = QPushButton("⚙")
        self.settings_button.setObjectName("settingsButton")
        self.settings_button.setFixedSize(46, 40)

        top_l.addWidget(self.logo_label, 0)
        top_l.addWidget(self.greeting, 1)
        # top_l.addWidget(self.settings_button, 0, Qt.AlignRight)

        # Transcript
        self.chat_transcript = QTextBrowser()
        self.chat_transcript.setObjectName("chatTranscript")
        self.chat_transcript.setOpenExternalLinks(False)
        self.chat_transcript.setAttribute(Qt.WA_StyledBackground, True)

        # ✅ Hide scrollbars but keep scrolling with wheel/touchpad
        self.chat_transcript.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_transcript.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # ✅ Detect user scrolling so we don't auto-jump while generating
        sb = self.chat_transcript.verticalScrollBar()
        sb.valueChanged.connect(self._on_chat_scroll_changed)

        # ===== Sheet (Excel-like grid inside window frame) =====
        self.sheet_card = QFrame()
        self.sheet_card.setObjectName("sheetCard")
        self.sheet_card.setAttribute(Qt.WA_StyledBackground, True)

        sheet_layout = QVBoxLayout(self.sheet_card)
        sheet_layout.setContentsMargins(12, 10, 12, 10)
        sheet_layout.setSpacing(8)

        self.sheet_title = QLabel("Sheet (CSV/Excel view)")
        self.sheet_title.setObjectName("sheetTitle")

        self.sheet_table = QTableWidget()
        self.sheet_table.setObjectName("sheetTable")
        self.sheet_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sheet_table.setAlternatingRowColors(True)
        self.sheet_table.setSortingEnabled(False)
        self.sheet_table.setWordWrap(False)
        self.sheet_table.setCornerButtonEnabled(False)
        self.sheet_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sheet_table.setSelectionBehavior(QTableWidget.SelectItems)
        self.sheet_table.setSelectionMode(QTableWidget.ExtendedSelection)

        self.sheet_table.horizontalHeader().setStretchLastSection(True)
        self.sheet_table.verticalHeader().setDefaultSectionSize(26)

        sheet_layout.addWidget(self.sheet_title, 0)
        sheet_layout.addWidget(self.sheet_table, 1)

        self.sheet_card.setVisible(False)

        # Input bar
        self.input_card = QFrame()
        self.input_card.setObjectName("inputCard")
        self.input_card.setAttribute(Qt.WA_StyledBackground, True)
        input_row = QHBoxLayout(self.input_card)
        input_row.setContentsMargins(18, 14, 18, 14)
        input_row.setSpacing(12)

        # ✅ FIX: use QTextEdit (IME-friendly) + remove focus frame (no white border)
        self.input_box = QTextEdit()
        self.input_box.setObjectName("inputBox")
        self.input_box.setFrameStyle(QFrame.NoFrame)
        self.input_box.setFocusPolicy(Qt.StrongFocus)
        self.input_box.setInputMethodHints(Qt.ImhNone)
        self.input_box.setFixedHeight(64)
        self.input_box.setPlaceholderText("Message Copilot")
        self.input_box.setAttribute(Qt.WA_StyledBackground, True)
        self.input_box.setAcceptRichText(False)

        self.send_button = QPushButton("➤")
        self.send_button.setObjectName("sendButton")
        self.send_button.setFixedSize(54, 44)

        input_row.addWidget(self.input_box, 1)
        input_row.addWidget(self.send_button, 0, Qt.AlignVCenter)

        self.sources_label = QLabel("")
        self.sources_label.setObjectName("sourcesLabel")
        self.sources_label.setWordWrap(True)

        self.status_label = QLabel("Starting…")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)

        content_layout.addWidget(top)
        content_layout.addWidget(self.chat_transcript, 1)
        content_layout.addWidget(self.sheet_card, 1)
        content_layout.addWidget(self.input_card, 0)
        content_layout.addWidget(self.status_label, 0)

    # ✅ Smart-scroll hook
    def _on_chat_scroll_changed(self, value: int):
        if self._programmatic_scroll:
            return
        sb = self.chat_transcript.verticalScrollBar()
        # "near bottom" threshold (px)
        self._stick_to_bottom = value >= (sb.maximum() - 24)

    def _apply_style(self):
        pal = self.palette()
        pal.setColor(QPalette.Window, Qt.transparent)
        pal.setColor(QPalette.Base, Qt.transparent)
        pal.setColor(QPalette.AlternateBase, Qt.transparent)
        pal.setColor(QPalette.Button, Qt.transparent)
        pal.setColor(QPalette.Text, Qt.white)
        pal.setColor(QPalette.WindowText, Qt.white)
        self.setPalette(pal)

        self.setStyleSheet(
            """
            QWidget {
                background: transparent;
                color: #e8eefc;
                font-family: "Segoe UI";
            }

            QFrame#contentCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 rgba(11,15,25,205),
                                            stop:0.55 rgba(10,19,38,195),
                                            stop:1 rgba(7,10,18,205));
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 18px;
            }

            QFrame#topBar { background: transparent; border: none; }

            QLabel#greeting {
                font-weight: 800;
                font-size: 20px;
                color: #eef3ff;
                background: transparent;
                border: none;
            }

            QLabel#logoLabel { background: transparent; border: none; }

            QTextBrowser#chatTranscript {
                background: transparent;
                border: none;
                padding: 8px 2px;
                color: #dbe7ff;
            }

            QFrame#inputCard {
                background: rgba(28, 33, 48, 0.72);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 26px;
            }

            /* ✅ FIX: style QTextEdit (not QTextBrowser) + kill focus outline */
            QTextEdit#inputBox {
                background: transparent;
                border: none;
                outline: none;
                color: #eef3ff;
                font-size: 16px;
                padding: 6px 6px;
            }
            QTextEdit#inputBox:focus {
                border: none;
                outline: none;
            }

            QPushButton#sendButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 rgba(123, 97, 255, 0.70),
                                            stop:0.55 rgba(0, 120, 212, 0.75),
                                            stop:1 rgba(52, 198, 255, 0.65));
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 16px;
                color: #ffffff;
                font-size: 20px;
                font-weight: 900;
            }
            QPushButton#sendButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 rgba(123, 97, 255, 0.85),
                                            stop:0.55 rgba(0, 120, 212, 0.90),
                                            stop:1 rgba(52, 198, 255, 0.78));
                border: 1px solid rgba(255,255,255,0.26);
            }
            QPushButton#sendButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 rgba(123, 97, 255, 0.92),
                                            stop:0.55 rgba(0, 120, 212, 0.95),
                                            stop:1 rgba(52, 198, 255, 0.82));
            }
            QPushButton#sendButton:disabled {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.10);
                color: rgba(232, 238, 252, 0.55);
            }

            QLabel#statusLabel {
                color: rgba(232, 238, 252, 0.82);
                font-size: 12px;
                margin-top: 1px;
                background: transparent;
                border: none;
            }

            QFrame#sheetCard {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 16px;
            }

            QLabel#sheetTitle {
                font-weight: 900;
                color: rgba(238,243,255,0.92);
                background: transparent;
                border: none;
            }

            QTableWidget#sheetTable {
                background: rgba(0,0,0,0.18);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 12px;
                gridline-color: rgba(255,255,255,0.14);
                color: #eef3ff;
                selection-background-color: rgba(0,120,212,0.28);
                selection-color: #ffffff;
                outline: none;
            }

            QHeaderView::section {
                background: rgba(255,255,255,0.06);
                color: rgba(238,243,255,0.95);
                padding: 6px 8px;
                border: 1px solid rgba(255,255,255,0.10);
                font-weight: 900;
            }
            """
        )

    def _fade_in(self):
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        effect.setOpacity(0.0)

        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(240)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.finished.connect(lambda: self.setGraphicsEffect(None))
        self._anims.append(anim)
        anim.start()

    def _animate_scroll_to_bottom(self):
        sb = self.chat_transcript.verticalScrollBar()
        start = sb.value()
        end = sb.maximum()
        anim = QPropertyAnimation(sb, b"value", self)
        anim.setDuration(220)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anims.append(anim)
        anim.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_auto_scale()

    def _apply_auto_scale(self):
        w = max(900, self.width())
        h = max(520, self.height())
        target_w, target_h = 1200, 700
        scale = min(w / target_w, h / target_h)
        scale = max(1.0, min(1.7, scale))

        if abs(scale - self._scale) < 0.02:
            return
        self._scale = scale

        f = QFont()
        f.setPointSizeF(self._base_font_pt * scale)
        self.setFont(f)

        self.input_box.setFixedHeight(int(64 * scale))
        self.send_button.setFixedSize(int(54 * scale), int(44 * scale))

    # ===== Sheet helpers =====
    def _hide_sheet(self):
        self.sheet_card.setVisible(False)
        self.sheet_table.clear()
        self.sheet_table.setRowCount(0)
        self.sheet_table.setColumnCount(0)

    def _show_grid_in_sheet(self, headers, rows):
        self.sheet_card.setVisible(True)
        self.sheet_table.setUpdatesEnabled(False)

        self.sheet_table.clear()
        self.sheet_table.setRowCount(len(rows))
        self.sheet_table.setColumnCount(len(headers))
        self.sheet_table.setHorizontalHeaderLabels([str(h) for h in headers])

        for r, row in enumerate(rows):
            for c in range(len(headers)):
                val = row[c] if c < len(row) else ""
                item = QTableWidgetItem(str(val))
                self.sheet_table.setItem(r, c, item)

        self.sheet_table.resizeColumnsToContents()
        self.sheet_table.setUpdatesEnabled(True)

    # ===== Streaming render throttle =====
    def _schedule_render(self):
        if self._pending_render:
            return
        self._pending_render = True

        if self._stream_render_timer is None:
            self._stream_render_timer = QTimer(self)
            self._stream_render_timer.setSingleShot(True)
            self._stream_render_timer.timeout.connect(self._flush_render)

        self._stream_render_timer.start(33)

    def _flush_render(self):
        self._pending_render = False
        self._render_chat()

    @staticmethod
    def _escape_user_html(message: str) -> str:
        return (
            message.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

    def _render_chat(self):
        # capture scroll position BEFORE changing html
        sb = self.chat_transcript.verticalScrollBar()
        prev_val = sb.value()
        should_stick = self._stick_to_bottom

        blocks = []
        for m in self._chat_msgs:
            if m["role"] == "user":
                safe = self._escape_user_html(m["text"])
                blocks.append(
                    f"""
                    <div style="margin:10px 0; display:flex; justify-content:flex-end;">
                      <div style="
                          max-width: 82%;
                          padding:12px 14px;
                          border-radius:18px;
                          background: rgba(0, 120, 212, 0.16);
                          border: 1px solid rgba(0, 120, 212, 0.22);
                      ">
                        <div style="font-weight:800; color:#eef3ff;">You</div>
                        <div style="margin-top:6px; color:#dbe7ff;">{safe}</div>
                      </div>
                    </div>
                    """
                )
            else:
                rich = to_rich_html(m["text"])
                blocks.append(
                    f"""
                    <div style="margin:10px 0; display:flex; justify-content:flex-start;">
                      <div style="
                          max-width: 92%;
                          padding:12px 14px;
                          border-radius:18px;
                          background: rgba(255,255,255,0.06);
                          border: 1px solid rgba(255,255,255,0.10);
                      ">
                        <div style="font-weight:800; color:#eef3ff;">Assistant</div>
                        <div style="margin-top:6px; color:#dbe7ff;">{rich}</div>
                      </div>
                    </div>
                    """
                )

        self.chat_transcript.setHtml("\n".join(blocks))

        # ✅ Smart scroll behavior:
        self._programmatic_scroll = True
        try:
            if should_stick:
                if self._is_streaming:
                    sb.setValue(sb.maximum())
                else:
                    self._animate_scroll_to_bottom()
            else:
                sb.setValue(min(prev_val, sb.maximum()))
        finally:
            self._programmatic_scroll = False

    def apply_language(self, lang_code: str, user_name: str = "Mir"):
        self._lang = lang_code if lang_code in UI_STRINGS else "en"
        self._user_name = user_name or "Mir"
        s = _pick_strings(self._lang)

        self.greeting.setText(s["greeting"].format(name=self._user_name))
        self.input_box.setPlaceholderText(s["input_placeholder"])
        self.sheet_title.setText(s.get("sheet_title", "Sheet (CSV/Excel view)"))

        if not self.status_label.text().strip():
            self.status_label.setText(s["status_start"])

    def set_buttons_enabled(self, enabled: bool):
        self.send_button.setEnabled(enabled)
        self.settings_button.setEnabled(enabled)

    def update_status(self, message: str):
        self.status_label.setText(message)

    def update_sources_text(self, sources_text: str):
        self.sources_label.setText(sources_text)

    def get_user_text(self) -> str:
        return self.input_box.toPlainText().strip()

    def clear_user_text(self):
        self.input_box.clear()

    def append_user_message(self, message: str):
        self._chat_msgs.append({"role": "user", "text": message})
        self._render_chat()

    def begin_assistant_stream(self):
        self._hide_sheet()
        self._chat_msgs.append({"role": "assistant", "text": ""})
        self._stream_index = len(self._chat_msgs) - 1
        self._is_streaming = True
        self._render_chat()

    def update_assistant_stream(self, delta: str):
        if self._stream_index is None:
            self.begin_assistant_stream()
        self._chat_msgs[self._stream_index]["text"] += delta
        self._schedule_render()

    def finalize_assistant_stream(self, full_message: str):
        if self._stream_index is None:
            self._chat_msgs.append({"role": "assistant", "text": ""})
            self._stream_index = len(self._chat_msgs) - 1

        self._hide_sheet()
        self._chat_msgs[self._stream_index]["text"] = full_message
        self._stream_index = None
        self._is_streaming = False

        if self._pending_render:
            return
        self._render_chat()

    def append_assistant_message(self, message: str):
        self.begin_assistant_stream()
        self.finalize_assistant_stream(message)

    def clear_chat(self):
        self._chat_msgs.clear()
        self._stream_index = None
        self.chat_transcript.clear()
        self.sources_label.clear()
        self.input_box.clear()
        self._hide_sheet()

        self._is_streaming = False
        self._pending_render = False
        if self._stream_render_timer is not None:
            self._stream_render_timer.stop()

        # reset smart-scroll defaults
        self._stick_to_bottom = True
        self._programmatic_scroll = False

    def open_settings_dialog(self, index_info_text: str, lang_code: str):
        dlg = SettingsDialog(self)
        dlg.info.setText(index_info_text)
        dlg.set_selected_language(lang_code)
        dlg.set_language_ui(lang_code)
        return dlg
