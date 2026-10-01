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
    # Both fixes take Qt from conda-forge. Its package is `pyqt6`: conda-forge
    # `pyqt` is PyQt5. pip goes first, as both write to site-packages/PyQt6.
    use_conda_qt = [
        "    pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip",
        "    conda install -c conda-forge pyqt6",
    ]
    if "undefined symbol" in str(QT_IMPORT_ERROR):
        message += [
            "",
            "That is a library mismatch, not a missing package. Typically the",
            "pip PyQt6 wheel (its Qt libraries search only their own folder,",
            "RUNPATH=$ORIGIN) picks up the system's older libfreetype, while",
            "conda supplies a newer libharfbuzz that needs symbols the old",
            "FreeType lacks. Updating conda's freetype/harfbuzz does not help;",
            "use Qt from conda-forge instead of the pip wheel:",
            *use_conda_qt,
            "or, keeping the wheel, preload conda's FreeType:",
            "    LD_PRELOAD=$CONDA_PREFIX/lib/libfreetype.so.6 ncv FILE.nc",
        ]
    elif "cannot open shared object file" in str(QT_IMPORT_ERROR):
        message += [
            "",
            "A system library Qt needs is missing on this machine (commonly on",
            "HPC nodes). Use Qt from conda-forge, which brings its own libraries:",
            *use_conda_qt,
            "or ask the administrators for the library named above.",
        ]
    raise RuntimeError("\n".join(message)) from QT_IMPORT_ERROR
