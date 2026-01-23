import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)

from rag_engine import RAGEngine, RAGEngineError
from ui_main import MainWindow
from worker import Worker

BASE_DIR = Path(r"D:\offline_doc_assistant")
MODELS_DIR = BASE_DIR / "models"
EMBED_MODEL_DIR = MODELS_DIR / "embed"
LLM_MODEL_PATH = MODELS_DIR / "qwen_model.gguf"
INDEX_DIR = BASE_DIR / "data" / "index"
LOG_DIR = BASE_DIR / "data" / "logs"


class AppController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = MainWindow()
        self.setCentralWidget(self.ui)
        self.setWindowTitle("Offline Document AI Assistant")
        self.resize(1100, 700)

        self.rag_engine = RAGEngine(
            base_dir=BASE_DIR,
            embed_model_dir=EMBED_MODEL_DIR,
            llm_model_path=LLM_MODEL_PATH,
            index_dir=INDEX_DIR,
            log_dir=LOG_DIR,
        )

        self.selected_docs = []

        self.ui.add_docs_button.clicked.connect(self.on_add_documents)
        self.ui.build_index_button.clicked.connect(self.on_build_index)
        self.ui.send_button.clicked.connect(self.on_send)
        self.ui.clear_button.clicked.connect(self.on_clear_chat)

        self.ui.update_status("Ready. Add documents to begin.")
        self.refresh_doc_list()

    def on_add_documents(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Documents",
            str(BASE_DIR),
            "Documents (*.pdf *.docx *.txt *.md)",
        )
        if not files:
            return
        for path in files:
            if path not in self.selected_docs:
                self.selected_docs.append(path)
        self.refresh_doc_list()
        self.ui.update_status(f"Selected {len(self.selected_docs)} document(s).")

    def refresh_doc_list(self):
        self.ui.doc_list.clear()
        for path in self.selected_docs:
            self.ui.doc_list.addItem(path)

    def on_build_index(self):
        if not self.selected_docs:
            QMessageBox.warning(self, "No Documents", "Please add documents first.")
            return
        if not EMBED_MODEL_DIR.exists():
            QMessageBox.critical(
                self,
                "Embedding Model Missing",
                f"Place the embedding model in {EMBED_MODEL_DIR}",
            )
            return
        self.ui.update_status("Indexing…")
        self.ui.set_buttons_enabled(False)

        worker = Worker(self.rag_engine.build_index, self.selected_docs)
        worker.progress.connect(self.ui.update_status)
        worker.error.connect(self.on_worker_error)
        worker.finished.connect(self.on_index_finished)
        worker.start()

    def on_index_finished(self, result):
        self.ui.set_buttons_enabled(True)
        if result:
            self.ui.update_status(
                f"Index ready. {result['chunks']} chunks from {result['docs']} documents."
            )
        else:
            self.ui.update_status("Indexing complete.")

    def on_send(self):
        question = self.ui.input_box.toPlainText().strip()
        if not question:
            return
        if not self.rag_engine.index_exists():
            QMessageBox.information(
                self,
                "Index Missing",
                "Please build the index before asking questions.",
            )
            return
        if not LLM_MODEL_PATH.exists():
            QMessageBox.critical(
                self,
                "LLM Model Missing",
                f"Place the Qwen GGUF model at {LLM_MODEL_PATH}",
            )
            return

        self.ui.append_user_message(question)
        self.ui.input_box.clear()
        self.ui.update_status("Retrieving…")
        self.ui.set_buttons_enabled(False)

        worker = Worker(self.rag_engine.answer_question, question)
        worker.progress.connect(self.ui.update_status)
        worker.error.connect(self.on_worker_error)
        worker.finished.connect(self.on_answer_ready)
        worker.start()

    def on_answer_ready(self, result):
        self.ui.set_buttons_enabled(True)
        if not result:
            self.ui.update_status("Ready.")
            return
        self.ui.append_assistant_message(result["answer"])
        self.ui.update_sources(result["sources"])
        self.ui.update_status("Ready.")

    def on_clear_chat(self):
        self.ui.clear_chat()

    def on_worker_error(self, message):
        self.ui.set_buttons_enabled(True)
        self.ui.update_status("Error occurred.")
        QMessageBox.critical(self, "Error", message)


def main():
    app = QApplication(sys.argv)
    controller = AppController()
    controller.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
