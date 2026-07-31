"""PyQt5 and Matplotlib Qt backend imports used by ncv."""
from __future__ import annotations

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

    QT_AVAILABLE = True
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    QtCore = QtGui = QtWidgets = None
    FigureCanvasQTAgg = NavigationToolbar2QT = None
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = exc
else:
    QT_IMPORT_ERROR = None


def require_qt() -> None:
    if not QT_AVAILABLE:
        raise RuntimeError("ncv requires PyQt5 to run the Qt viewer.") from QT_IMPORT_ERROR
