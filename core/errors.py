from PyQt6.QtWidgets import QMessageBox, QWidget


class AppError(Exception):
    def __init__(self, msg: str, title: str = ""):
        self.msg = msg
        self.title = title

    def show_warning(self, parent: QWidget):
        QMessageBox.warning(parent, self.title, self.msg)
