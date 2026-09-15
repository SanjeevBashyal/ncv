"""PyQt6 and pyqtgraph imports used by ncv."""
from __future__ import annotations

try:
    from PyQt6 import QtCore, QtGui, QtWidgets, uic
    import pyqtgraph as pg

    QT_AVAILABLE = True
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    QtCore = QtGui = QtWidgets = uic = pg = None
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = exc
else:
    QT_IMPORT_ERROR = None
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major",
                        background="w", foreground="k")


def require_qt() -> None:
    if not QT_AVAILABLE:
        raise RuntimeError(
            "ncv requires PyQt6 and pyqtgraph to run the Qt viewer."
        ) from QT_IMPORT_ERROR
