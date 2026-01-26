from pathlib import Path

from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PyQt5.QtGui import QFont, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(520)

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

        self.rebuild_btn = QPushButton("Rebuild index now")
        self.rebuild_btn.setObjectName("dlgPrimary")

        self.clear_btn = QPushButton("Delete ALL indexed memory (FAISS + metadata)")
        self.clear_btn.setObjectName("dlgDanger")

        self.close_btn = QPushButton("Close")
        self.close_btn.setObjectName("dlgGhost")

        root.addWidget(self.title)
        root.addWidget(self.info)

        root.addSpacing(6)
        root.addWidget(self.rebuild_btn)
        root.addWidget(self.clear_btn)

        root.addSpacing(10)
        root.addWidget(self.close_btn)

        self.close_btn.clicked.connect(self.accept)

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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._anims = []
        self._scale = 1.0
        self._base_font_pt = 14
        self._logo_pixmap = None

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)

        self._build_ui()
        self._apply_style()
        self._apply_auto_scale()
        self._fade_in()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 22)
        root.setSpacing(18)

        # Top bar
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

        self.greeting = QLabel("🛠️")
        self.greeting.setObjectName("greeting")
        self.greeting.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.settings_button = QPushButton("⚙")
        self.settings_button.setObjectName("settingsButton")
        self.settings_button.setFixedSize(46, 40)

        top_l.addWidget(self.logo_label, 0)
        top_l.addWidget(self.greeting, 1)
        top_l.addWidget(self.settings_button, 0, Qt.AlignRight)

        # Chat transcript
        self.chat_transcript = QTextBrowser()
        self.chat_transcript.setObjectName("chatTranscript")
        self.chat_transcript.setOpenExternalLinks(False)
        self.chat_transcript.setAttribute(Qt.WA_StyledBackground, True)

        # Input bar
        self.input_card = QFrame()
        self.input_card.setObjectName("inputCard")
        self.input_card.setAttribute(Qt.WA_StyledBackground, True)
        input_row = QHBoxLayout(self.input_card)
        input_row.setContentsMargins(18, 14, 18, 14)
        input_row.setSpacing(12)

        self.input_box = QTextBrowser()
        self.input_box.setObjectName("inputBox")
        self.input_box.setFixedHeight(64)
        self.input_box.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.input_box.setPlaceholderText("Ask about the errors and and whats in the documetns...")
        self.input_box.setAttribute(Qt.WA_StyledBackground, True)

        self.send_button = QPushButton("➤")
        self.send_button.setObjectName("sendButton")
        self.send_button.setFixedSize(54, 44)

        input_row.addWidget(self.input_box, 1)
        input_row.addWidget(self.send_button, 0, Qt.AlignVCenter)

        # Status + sources
        self.sources_label = QLabel("")
        self.sources_label.setObjectName("sourcesLabel")
        self.sources_label.setWordWrap(True)

        self.status_label = QLabel("Starting…")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)

        root.addWidget(top)
        root.addWidget(self.chat_transcript, 1)
        root.addWidget(self.input_card, 0)
        root.addWidget(self.sources_label, 0)
        root.addWidget(self.status_label, 0)

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
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #0b0f19, stop:0.55 #0a1326, stop:1 #070a12);
                color: #e8eefc;
                font-family: "Segoe UI";
            }

            QFrame#topBar {
                background: transparent;
                border: none;
            }

            QLabel#greeting {
                font-weight: 800;
                font-size: 28px;
                background: transparent;
                border: none;
                color: #eef3ff;
            }

            QLabel#logoLabel {
                background: transparent;
                border: none;
            }

            QPushButton#settingsButton {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 14px;
                color: #eef3ff;
                font-size: 18px;
                font-weight: 800;
            }
            QPushButton#settingsButton:hover {
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.18);
            }

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

            QTextBrowser#inputBox {
                background: transparent;
                border: none;
                color: #eef3ff;
                font-size: 16px;
                padding: 6px 6px;
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

            QLabel#sourcesLabel {
                color: rgba(232, 238, 252, 0.72);
                font-size: 12px;
                background: transparent;
                margin-top: 6px;
            }

            QLabel#statusLabel {
                color: rgba(232, 238, 252, 0.82);
                font-size: 12px;
                background: transparent;
                margin-top: 2px;
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

        if self._logo_pixmap and not self._logo_pixmap.isNull():
            logo_h = int(44 * scale)
            pm = self._logo_pixmap.scaledToHeight(logo_h, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pm)
            self.logo_label.setFixedSize(pm.width(), pm.height())

        self.input_box.setFixedHeight(int(64 * scale))
        self.send_button.setFixedSize(int(54 * scale), int(44 * scale))

    # ===== API used by app.py =====
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
        safe = (
            message.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        self.chat_transcript.append(
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
        self._animate_scroll_to_bottom()

    def append_assistant_message(self, message: str):
        safe = (
            message.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        self.chat_transcript.append(
            f"""
            <div style="margin:10px 0; display:flex; justify-content:flex-start;">
              <div style="
                  max-width: 82%;
                  padding:12px 14px;
                  border-radius:18px;
                  background: rgba(255,255,255,0.06);
                  border: 1px solid rgba(255,255,255,0.10);
              ">
                <div style="font-weight:800; color:#eef3ff;">Assistant</div>
                <div style="margin-top:6px; color:#dbe7ff;">{safe}</div>
              </div>
            </div>
            """
        )
        self._animate_scroll_to_bottom()

    def clear_chat(self):
        self.chat_transcript.clear()
        self.sources_label.clear()
        self.input_box.clear()

    def open_settings_dialog(self, index_info_text: str):
        dlg = SettingsDialog(self)
        dlg.info.setText(index_info_text)
        return dlg
