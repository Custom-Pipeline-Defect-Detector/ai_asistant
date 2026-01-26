import traceback
from typing import Any, Callable

from PyQt5.QtCore import QThread, pyqtSignal


class WorkerThread(QThread):
    progress = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn: Callable[..., Any], args=()):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self):
        try:
            out = self.fn(*self.args)
            self.result.emit(out)
        except Exception:
            self.error.emit(traceback.format_exc())
