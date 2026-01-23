from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        self.add_docs_button = QPushButton("Add Documents")
        self.build_index_button = QPushButton("Build/Update Index")
        self.doc_list = QListWidget()
        self.status_label = QLabel("Status: Idle")
        self.status_label.setWordWrap(True)

        left_panel.addWidget(self.add_docs_button)
        left_panel.addWidget(self.build_index_button)
        left_panel.addWidget(QLabel("Selected Documents:"))
        left_panel.addWidget(self.doc_list, stretch=1)
        left_panel.addWidget(self.status_label)

        self.chat_transcript = QTextBrowser()
        self.input_box = QPlainTextEdit()
        self.input_box.setFixedHeight(90)
        self.send_button = QPushButton("Send")
        self.clear_button = QPushButton("Clear Chat")

        input_buttons = QHBoxLayout()
        input_buttons.addWidget(self.send_button)
        input_buttons.addWidget(self.clear_button)

        self.sources_panel = QListWidget()
        self.sources_panel.setMaximumHeight(140)

        right_panel.addWidget(QLabel("Chat"))
        right_panel.addWidget(self.chat_transcript, stretch=1)
        right_panel.addWidget(QLabel("Your Question"))
        right_panel.addWidget(self.input_box)
        right_panel.addLayout(input_buttons)
        right_panel.addWidget(QLabel("Sources"))
        right_panel.addWidget(self.sources_panel)

        splitter = QFrame()
        splitter_layout = QHBoxLayout(splitter)
        splitter_layout.setContentsMargins(0, 0, 0, 0)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        splitter_layout.addWidget(left_widget, 1)
        splitter_layout.addWidget(right_widget, 2)

        layout.addWidget(splitter)

    def set_buttons_enabled(self, enabled: bool):
        self.add_docs_button.setEnabled(enabled)
        self.build_index_button.setEnabled(enabled)
        self.send_button.setEnabled(enabled)

    def update_status(self, message: str):
        self.status_label.setText(f"Status: {message}")

    def append_user_message(self, message: str):
        self.chat_transcript.append(
            f"<p><b>User:</b> {message}</p>"
        )

    def append_assistant_message(self, message: str):
        self.chat_transcript.append(
            f"<p><b>Assistant:</b> {message}</p>"
        )

    def update_sources(self, sources):
        self.sources_panel.clear()
        for source in sources:
            self.sources_panel.addItem(source)

    def clear_chat(self):
        self.chat_transcript.clear()
        self.sources_panel.clear()
        self.input_box.clear()
