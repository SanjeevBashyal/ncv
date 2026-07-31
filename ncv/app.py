"""PyQt5 application and main-window orchestration for ncv."""
from __future__ import annotations

import sys

import numpy as np

from . import ncvmap as _ncvmap
from .ncvcommon import resource_path
from .ncvcontour import ContourPanel
from .ncvmap import MapPanel, MapUnavailablePanel
from .ncvmatrix import MatrixPanel
from .ncvscatter import ScatterPanel
from .ncvutils import selvar, vardim2var
from .pyui.ui_main_window import Ui_NcvMainWindow
from .qt_compat import QtGui, QtWidgets, require_qt
from .session import NcvSession, normalize_files


HAVE_CARTOPY = _ncvmap.HAVE_CARTOPY
CARTOPY_IMPORT_ERROR = _ncvmap.CARTOPY_IMPORT_ERROR


def ensure_cartopy():
    """Retry Cartopy import and keep compatibility exports current."""
    global HAVE_CARTOPY, CARTOPY_IMPORT_ERROR

    HAVE_CARTOPY = _ncvmap.ensure_cartopy()
    CARTOPY_IMPORT_ERROR = _ncvmap.CARTOPY_IMPORT_ERROR
    return HAVE_CARTOPY


def _window_geometry() -> tuple[int, int, int, int]:
    screen = QtWidgets.QApplication.primaryScreen()
    if screen is None:
        return 1000, 800, 100, 50
    geom = screen.availableGeometry()
    width = geom.width()
    height = geom.height()
    w = width if width < 1000 else max(2 * width // 5, 1000)
    h = height if height < 800 else max(4 * height // 5, 800)
    x = geom.x() if width < 1000 else geom.x() + max((width - w) // 2, 0)
    return w, h, x, geom.y()


class NcvMainWindow(QtWidgets.QMainWindow, Ui_NcvMainWindow):
    """Top-level window that owns the four independent Qt panels."""

    instances = []

    def __init__(self, session: NcvSession, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.session = session
        self._children = []

        pages = (
            (self.tab_scatterPlot, "Scatter/Line"),
            (self.tab_contourPlot, "Contour"),
            (self.tab_mapPlot, "Map"),
            (self.tab_matrixDisplay, "Matrix"),
        )
        labels = []
        for page, fallback in pages:
            index = self.tabWidget_main.indexOf(page)
            labels.append(
                self.tabWidget_main.tabText(index) if index >= 0 else fallback)
        for page, _fallback in pages:
            index = self.tabWidget_main.indexOf(page)
            if index >= 0:
                self.tabWidget_main.removeTab(index)

        self.scatter = ScatterPanel(self, session)
        self.contour = ContourPanel(self, session)
        if ensure_cartopy():
            self.map = MapPanel(self, session)
        else:
            self.map = MapUnavailablePanel(CARTOPY_IMPORT_ERROR)
        self.matrix = MatrixPanel(self, session)
        self.tab_matrixDisplay = self.matrix

        for index, (panel, label) in enumerate(zip(
                (self.scatter, self.contour, self.map, self.matrix), labels)):
            self.tabWidget_main.insertTab(index, panel, label)
        self.tabWidget_main.setCurrentIndex(0)
        self.tabWidget_main.currentChanged.connect(self._tab_changed)

        self.setWindowIcon(QtGui.QIcon(
            resource_path("images", "ncvue_icon.png")))
        w, h, x, y = _window_geometry()
        self.setGeometry(x, y, w, h)
        self.update_title()
        self.select_default_tab()
        NcvMainWindow.instances.append(self)

    def update_title(self):
        self.setWindowTitle(f"ncv {self.session.title_suffix}".strip())

    def select_default_tab(self):
        if not _ncvmap.HAVE_CARTOPY:
            return
        mapfirst = False
        try:
            if self.session.usex:
                for attr in (self.session.latvar, self.session.lonvar):
                    if attr:
                        variable = attr[0:attr.rfind("(")].rstrip()
                        mapfirst = (
                            mapfirst
                            or np.prod(self.session.fi[variable].shape) > 1
                        )
            else:
                for attr in (self.session.latvar, self.session.lonvar):
                    if any(attr):
                        item = next(value for value in attr if value)
                        _group, variable = vardim2var(
                            item, self.session.groups)
                        mapfirst = (
                            mapfirst
                            or np.prod(selvar(self.session, variable).shape) > 1
                        )
        except Exception:
            mapfirst = False
        if mapfirst:
            self.tabWidget_main.setCurrentIndex(2)

    def _tab_changed(self, _index):
        panel = self.tabWidget_main.currentWidget()
        if hasattr(panel, "redraw"):
            panel.redraw()

    def open_file_dialog(self, use_xarray):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Choose netcdf file(s)",
            "",
            "NetCDF files (*.nc *.nc4 *.cdf);;All files (*)",
        )
        if files:
            self.open_files(files, use_xarray)

    def open_files(self, files, use_xarray=False):
        try:
            self.session.open(files, use_xarray=use_xarray)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "ncv", str(exc))
            return
        for window in list(NcvMainWindow.instances):
            if window.session is self.session:
                window.on_session_changed()

    def on_session_changed(self):
        self.update_title()
        for panel in (self.scatter, self.contour, self.map, self.matrix):
            panel.reinit()
        self.select_default_tab()
        current = self.tabWidget_main.currentWidget()
        if hasattr(current, "redraw"):
            current.redraw()

    def create_secondary_window(self):
        child = NcvMainWindow(self.session)
        child.move(self.x() + 50, self.y() + 50)
        self._children.append(child)
        child.show()

    def closeEvent(self, event):
        if self in NcvMainWindow.instances:
            NcvMainWindow.instances.remove(self)
        if not any(
                window.session is self.session
                for window in NcvMainWindow.instances):
            self.session.close()
        super().closeEvent(event)


def ncv(ncfile=None, miss=None, usex=False):
    """Launch the Qt ncv application."""
    require_qt()
    if miss is None:
        miss = np.nan
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    session = NcvSession(miss=miss)
    files = normalize_files(ncfile)
    if files:
        session.open(files, use_xarray=usex)
    window = NcvMainWindow(session)
    window.show()
    if owns_app:
        app.aboutToQuit.connect(session.close)
        return app.exec_()
    return window


__all__ = [
    "ContourPanel",
    "HAVE_CARTOPY",
    "MapPanel",
    "MapUnavailablePanel",
    "MatrixPanel",
    "NcvMainWindow",
    "ScatterPanel",
    "ensure_cartopy",
    "ncv",
]
