"""PyQt6 and pyqtgraph imports used by ncv."""
from __future__ import annotations

try:
    from PyQt6 import QtCore, QtGui, QtWidgets, uic
    import pyqtgraph as pg

    QT_AVAILABLE = True
# ImportError, not ModuleNotFoundError: a mismatched system library (conda
# mixing pip PyQt6 with its own libfreetype/libharfbuzz) raises a bare
# ImportError from the C loader, which used to escape as a raw traceback.
except ImportError as exc:  # pragma: no cover - environment specific
    QtCore = QtGui = QtWidgets = uic = pg = None
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = exc
else:
    QT_IMPORT_ERROR = None
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major",
                        background="w", foreground="k")


def require_qt() -> None:
    if QT_AVAILABLE:
        return
    message = [
        "ncv requires PyQt6 and pyqtgraph to run the Qt viewer.",
        f"Import failed with: {QT_IMPORT_ERROR}",
    ]
    if "undefined symbol" in str(QT_IMPORT_ERROR):
        message += [
            "",
            "That is a system-library mismatch, not a missing package -- it "
            "usually means a pip-installed PyQt6 is loading conda's older "
            "libfreetype/libharfbuzz. Align them from one channel:",
            "    conda install -c conda-forge freetype harfbuzz",
            "    conda install -c conda-forge pyqt   # or take Qt from conda",
        ]
    raise RuntimeError("\n".join(message)) from QT_IMPORT_ERROR
