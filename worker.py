import traceback
from typing import Any, Callable, Optional

from PyQt5.QtCore import QThread, pyqtSignal


class WorkerThread(QThread):
    progress = pyqtSignal(str)   # status messages
    partial = pyqtSignal(str)    # streaming tokens
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn: Callable[..., Any], args=(), kwargs: Optional[dict] = None):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}

    def run(self):
        try:
            out = self.fn(*self.args, **self.kwargs)
            self.result.emit(out)
        except Exception:
            self.error.emit(traceback.format_exc())
