"""Qt matrix panel."""
from __future__ import annotations

from .qt_compat import QtWidgets


__all__ = ["MatrixPanel"]


class MatrixPanel(QtWidgets.QWidget):
    """Placeholder for the matrix view."""

    def __init__(self, window, session):
        super().__init__(window)
        self.window = window
        self.session = session
        self.setObjectName("tab_matrixDisplay")

    def reinit(self):
        pass

    def redraw(self):
        pass
