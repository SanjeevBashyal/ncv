"""PyQt5 application for ncv."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import netCDF4 as nc
import numpy as np


def _import_cartopy():
    try:
        import cartopy.crs as imported_ccrs
        import cartopy.feature as imported_cfeature
    except Exception as exc:  # pragma: no cover - depends on environment
        return None, None, exc
    return imported_ccrs, imported_cfeature, None


ccrs, cfeature, CARTOPY_IMPORT_ERROR = _import_cartopy()
HAVE_CARTOPY = CARTOPY_IMPORT_ERROR is None


def ensure_cartopy():
    """Retry Cartopy import and update module-level availability state."""
    global ccrs, cfeature, CARTOPY_IMPORT_ERROR, HAVE_CARTOPY

    if HAVE_CARTOPY:
        return True
    ccrs, cfeature, CARTOPY_IMPORT_ERROR = _import_cartopy()
    HAVE_CARTOPY = CARTOPY_IMPORT_ERROR is None
    return HAVE_CARTOPY

from .dimensions import dimension_specs, dimension_values, empty_dimension_specs
from .ncvmethods import get_miss
from .ncvutils import add_cyclic, format_coord_contour, format_coord_map
from .ncvutils import format_coord_scatter, get_slice_values, parse_entry
from .ncvutils import selvar, set_axis_label, set_miss, vardim2var
from .pyui.ui_contour_panel import Ui_ContourPanel
from .pyui.ui_main_window import Ui_NcvMainWindow
from .pyui.ui_map_panel import Ui_MapPanel
from .pyui.ui_map_unavailable import Ui_MapUnavailablePanel
from .pyui.ui_scatter_panel import Ui_ScatterPanel
from .qt_compat import FigureCanvasQTAgg, NavigationToolbar2QT
from .qt_compat import QtCore, QtGui, QtWidgets, require_qt
from .session import HAVE_XARRAY, NcvSession, normalize_files


def _resource_path(*parts: str) -> str:
    return str(Path(__file__).resolve().parent.joinpath(*parts))


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
    y = geom.y()
    return w, h, x, y


def _maybe_color(value: str):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    return parsed if isinstance(parsed, tuple) else value


def _float_or_none(value: str):
    if value == "None":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _minmax_ylim(ylim, ylim2):
    ymin = None
    ymax = None
    if isinstance(ylim, (list, tuple)) and isinstance(ylim2, (list, tuple)):
        vals = list(ylim) + list(ylim2)
        if all(v is not None for v in vals):
            ymin = min(vals)
            ymax = max(vals)
    return ymin, ymax


def _set_combo_items(combo, values, current=None):
    combo.blockSignals(True)
    combo.clear()
    combo.addItems([str(value) for value in values])
    if current is not None and str(current) in [combo.itemText(i) for i in range(combo.count())]:
        combo.setCurrentText(str(current))
    combo.blockSignals(False)


class DimensionControlRow(QtWidgets.QWidget):
    """A row of label + combo dimension selectors."""

    changed = QtCore.pyqtSignal()

    def __init__(self, maxdim=1, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.labels = []
        self.selectors = []
        self.set_count(maxdim)

    def set_count(self, count):
        while len(self.selectors) < max(count, 1):
            label = QtWidgets.QLabel(str(len(self.selectors)))
            selector = QtWidgets.QComboBox()
            selector.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            selector.currentIndexChanged.connect(self.changed)
            self.labels.append(label)
            self.selectors.append(selector)
            self.layout.addWidget(label)
            self.layout.addWidget(selector)
        for i, selector in enumerate(self.selectors):
            visible = i < max(count, 1)
            self.labels[i].setVisible(visible)
            selector.setVisible(visible)

    def set_specs(self, specs):
        self.set_count(len(specs))
        for label, selector, spec in zip(self.labels, self.selectors, specs):
            label.setText(spec.label)
            label.setToolTip(spec.tooltip)
            selector.blockSignals(True)
            selector.clear()
            selector.addItems(spec.values)
            selector.setCurrentText(spec.value)
            selector.setEnabled(spec.enabled)
            selector.setToolTip(spec.tooltip)
            selector.blockSignals(False)

    def values(self):
        return dimension_values(self.selectors)

    def set_value(self, index, value):
        if 0 <= index < len(self.selectors):
            self.selectors[index].setCurrentText(str(value))


class PlotPanel(QtWidgets.QWidget):
    """Common Qt panel helpers."""

    def __init__(self, window, session: NcvSession, name: str):
        super().__init__(window)
        self.window = window
        self.session = session
        self.name = name
        self._updating = False
        self._copy_session()

    def _copy_session(self):
        self.usex = self.session.usex
        self.fi = self.session.fi
        self.groups = self.session.groups
        self.miss = self.session.miss
        self.dunlim = self.session.dunlim
        self.time = self.session.time
        self.tname = self.session.tname
        self.tvar = self.session.tvar
        self.dtime = self.session.dtime
        self.latvar = self.session.latvar
        self.lonvar = self.session.lonvar
        self.latdim = self.session.latdim
        self.londim = self.session.londim
        self.maxdim = self.session.maxdim
        self.cols = self.session.cols

    def columns(self):
        return [""] + list(self.session.cols)

    def connect_file_controls(self):
        """Connect the common file buttons supplied by a Designer form."""
        self.pushButton_openFile.clicked.connect(
            lambda: self.window.open_file_dialog(False))
        self.pushButton_openXarray.setVisible(HAVE_XARRAY)
        if HAVE_XARRAY:
            self.pushButton_openXarray.clicked.connect(
                lambda: self.window.open_file_dialog(True))
        self.pushButton_newWindow.clicked.connect(
            self.window.create_secondary_window)

    def populate_cmap_combo(self, combo):
        """Populate a Designer-owned combo with Matplotlib colormaps."""
        combo.clear()
        for cmap in sorted([c for c in plt.colormaps() if not c.endswith("_r")]):
            icon_path = _resource_path("images", f"{cmap}.png")
            if os.path.exists(icon_path):
                combo.addItem(QtGui.QIcon(icon_path), cmap)
            else:
                combo.addItem(cmap)
        combo.setCurrentText("RdYlBu")

    def slice_miss(self, dim_controls: DimensionControlRow, variable):
        miss = get_miss(self, variable)
        values = dim_controls.values()
        values.extend(["0"] * max(0, variable.ndim - len(values)))
        out = get_slice_values(values, variable)
        if out.ndim > 1:
            out = out.squeeze()
        out = set_miss(miss, out)
        try:
            _ = out.shape[0]
        except IndexError:
            out = np.array([np.nan])
        return out

    def reinit(self):
        self._copy_session()

    def redraw(self):
        raise NotImplementedError


class ScatterPanel(PlotPanel, Ui_ScatterPanel):
    def __init__(self, window, session):
        super().__init__(window, session, "Scatter/Line")
        self.line_y = []
        self.line_y2 = []
        self._build_ui()
        self.reinit()

    def _build_ui(self):
        self.setupUi(self)
        self.connect_file_controls()
        self.figure = Figure(facecolor="white", figsize=(1, 1))
        self.axes = self.figure.add_subplot(111)
        self.axes2 = self.axes.twinx()
        self.axes2.yaxis.set_label_position("right")
        self.axes2.yaxis.tick_right()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plotLayout.addWidget(self.canvas, 1)
        self.plotLayout.addWidget(self.toolbar)

        self.xd = DimensionControlRow(self.maxdim)
        self.yd = DimensionControlRow(self.maxdim)
        self.y2d = DimensionControlRow(self.maxdim)
        self.xDimensionsLayout.addWidget(self.xd)
        self.yDimensionsLayout.addWidget(self.yd)
        self.y2DimensionsLayout.addWidget(self.y2d)

        c = list(plt.rcParams["axes.prop_cycle"])
        col1 = c[0]["color"]
        col2 = c[3]["color"]
        for entry in (self.lineEdit_lineColorY1,
                      self.lineEdit_markerFillColorY1,
                      self.lineEdit_markerEdgeColorY1):
            entry.setText(col1)
        for entry in (self.lineEdit_lineColorY2,
                      self.lineEdit_markerFillColorY2,
                      self.lineEdit_markerEdgeColorY2):
            entry.setText(col2)

        self.comboBox_x.currentIndexChanged.connect(self.selected_x)
        self.checkBox_invX.stateChanged.connect(self.checked_x)
        self.comboBox_y.currentIndexChanged.connect(self.selected_y)
        self.checkBox_invY.stateChanged.connect(self.checked_y)
        self.pushButton_redraw.clicked.connect(self.redraw)
        self.xd.changed.connect(self.spinned_x)
        self.yd.changed.connect(self.spinned_y)
        for entry in (self.lineEdit_lineStyleY1,
                      self.lineEdit_lineWidthY1,
                      self.lineEdit_lineColorY1,
                      self.lineEdit_markerStyleY1,
                      self.lineEdit_markerSizeY1,
                      self.lineEdit_markerFillColorY1,
                      self.lineEdit_markerEdgeColorY1,
                      self.lineEdit_markerEdgeWidthY1,
                      self.lineEdit_xlim, self.lineEdit_ylim):
            entry.editingFinished.connect(self.entered_y)
        self.comboBox_y2.currentIndexChanged.connect(self.selected_y2)
        self.checkBox_invY2.stateChanged.connect(self.checked_y2)
        self.checkBox_sameYaxis.stateChanged.connect(self.checked_yy2)
        self.y2d.changed.connect(self.spinned_y2)
        for entry in (self.lineEdit_lineStyleY2,
                      self.lineEdit_lineWidthY2,
                      self.lineEdit_lineColorY2,
                      self.lineEdit_markerStyleY2,
                      self.lineEdit_markerSizeY2,
                      self.lineEdit_markerFillColorY2,
                      self.lineEdit_markerEdgeColorY2,
                      self.lineEdit_markerEdgeWidthY2,
                      self.lineEdit_y2lim):
            entry.editingFinished.connect(self.entered_y2)
        self.pushButton_quit.clicked.connect(QtWidgets.QApplication.quit)

    def reinit(self):
        super().reinit()
        self._updating = True
        columns = self.columns()
        for combo in (self.comboBox_x, self.comboBox_y, self.comboBox_y2):
            _set_combo_items(combo, columns, "")
        for dims in (self.xd, self.yd, self.y2d):
            dims.set_specs(empty_dimension_specs(self.maxdim))
        self.lineEdit_xlim.setText("None")
        self.lineEdit_ylim.setText("None")
        self.lineEdit_y2lim.setText("None")
        self._updating = False

    def checked_x(self):
        if not self._updating:
            self.redraw_y()
            self.redraw_y2()

    def checked_y(self):
        if not self._updating:
            self.redraw_y()

    def checked_y2(self):
        if not self._updating:
            self.redraw_y2()

    def checked_yy2(self):
        if not self._updating:
            self.lineEdit_ylim.setText("None")
            self.lineEdit_y2lim.setText("None")
            self.redraw_y()
            self.redraw_y2()

    def entered_y(self):
        if not self._updating:
            self.redraw_y()

    def entered_y2(self):
        if not self._updating:
            self.redraw_y2()

    def selected_x(self):
        if self._updating:
            return
        self.xd.set_specs(
            dimension_specs(self, self.comboBox_x.currentText(), "x"))
        self.lineEdit_xlim.setText("None")
        self.redraw()

    def selected_y(self):
        if self._updating:
            return
        self.yd.set_specs(
            dimension_specs(self, self.comboBox_y.currentText(), "y"))
        self.lineEdit_ylim.setText("None")
        self.redraw()

    def selected_y2(self):
        if self._updating:
            return
        self.y2d.set_specs(
            dimension_specs(self, self.comboBox_y2.currentText(), "y2"))
        self.lineEdit_y2lim.setText("None")
        self.redraw()

    def spinned_x(self):
        if not self._updating:
            self.redraw()

    def spinned_y(self):
        if not self._updating:
            self.redraw()

    def spinned_y2(self):
        if not self._updating:
            self.redraw()

    def redraw_y(self):
        y = self.comboBox_y.currentText()
        if not y:
            return
        inv_y = self.checkBox_invY.isChecked()
        ylim = parse_entry(self.lineEdit_ylim.text())
        ylim2 = parse_entry(self.lineEdit_y2lim.text())
        ls = self.lineEdit_lineStyleY1.text()
        lw = float(self.lineEdit_lineWidthY1.text())
        color = _maybe_color(self.lineEdit_lineColorY1.text())
        marker = self.lineEdit_markerStyleY1.text()
        ms = float(self.lineEdit_markerSizeY1.text())
        mfc = _maybe_color(self.lineEdit_markerFillColorY1.text())
        mec = _maybe_color(self.lineEdit_markerEdgeColorY1.text())
        mew = float(self.lineEdit_markerEdgeWidthY1.text())
        y2 = self.comboBox_y2.currentText()
        same_y = self.checkBox_sameYaxis.isChecked()
        pargs = {
            "linestyle": ls,
            "linewidth": lw,
            "marker": marker,
            "markersize": ms,
            "markerfacecolor": mfc,
            "markeredgecolor": mec,
            "markeredgewidth": mew,
        }
        gy, vy = vardim2var(y, self.groups)
        tname = self.tname if self.usex else self.tname[gy]
        if vy == tname:
            ylab = "Date"
            pargs["color"] = color
        else:
            ylab = set_axis_label(selvar(self, vy))
            if len(self.line_y) == 1:
                pargs["color"] = color
        for line in self.line_y:
            plt.setp(line, **pargs)
        if "color" in pargs and pargs["color"] != "None":
            self.axes.spines["left"].set_color(pargs["color"])
            self.axes.tick_params(axis="y", colors=pargs["color"])
            self.axes.yaxis.label.set_color(pargs["color"])
        self.axes.yaxis.set_label_text(ylab)
        if not isinstance(ylim, list):
            ylim = self.axes.get_ylim()
        if not isinstance(ylim2, list):
            ylim2 = self.axes2.get_ylim()
        if same_y and y2:
            ymin, ymax = _minmax_ylim(ylim, ylim2)
            if ymin is not None and ymax is not None:
                ylim = [ymin, ymax]
                ylim2 = [ymin, ymax]
            self.axes.set_ylim(ylim)
            self.axes2.set_ylim(ylim2)
        if inv_y and ylim[0] is not None:
            if ylim[0] < ylim[1]:
                ylim = ylim[::-1]
            self.axes.set_ylim(ylim)
        else:
            if ylim[1] < ylim[0]:
                ylim = ylim[::-1]
            self.axes.set_ylim(ylim)
        self._apply_x_limits()
        self.canvas.draw()
        self.toolbar.update()

    def redraw_y2(self):
        y2 = self.comboBox_y2.currentText()
        if not y2:
            return
        y = self.comboBox_y.currentText()
        inv_y2 = self.checkBox_invY2.isChecked()
        same_y = self.checkBox_sameYaxis.isChecked()
        ylim = parse_entry(self.lineEdit_ylim.text())
        ylim2 = parse_entry(self.lineEdit_y2lim.text())
        pargs = {
            "linestyle": self.lineEdit_lineStyleY2.text(),
            "linewidth": float(self.lineEdit_lineWidthY2.text()),
            "marker": self.lineEdit_markerStyleY2.text(),
            "markersize": float(self.lineEdit_markerSizeY2.text()),
            "markerfacecolor": _maybe_color(
                self.lineEdit_markerFillColorY2.text()),
            "markeredgecolor": _maybe_color(
                self.lineEdit_markerEdgeColorY2.text()),
            "markeredgewidth": float(
                self.lineEdit_markerEdgeWidthY2.text()),
        }
        color = _maybe_color(self.lineEdit_lineColorY2.text())
        gy, vy = vardim2var(y2, self.groups)
        tname = self.tname if self.usex else self.tname[gy]
        if vy == tname:
            ylab = "Date"
            pargs["color"] = color
        else:
            ylab = set_axis_label(selvar(self, vy))
            if len(self.line_y2) == 1:
                pargs["color"] = color
        for line in self.line_y2:
            plt.setp(line, **pargs)
        if "color" in pargs and pargs["color"] != "None":
            self.axes2.spines["right"].set_color(pargs["color"])
            self.axes2.tick_params(axis="y", colors=pargs["color"])
            self.axes2.yaxis.label.set_color(pargs["color"])
        self.axes2.yaxis.set_label_text(ylab)
        if not isinstance(ylim, list):
            ylim = self.axes.get_ylim()
        if not isinstance(ylim2, list):
            ylim2 = self.axes2.get_ylim()
        if same_y and y:
            ymin, ymax = _minmax_ylim(ylim, ylim2)
            if ymin is not None and ymax is not None:
                ylim = [ymin, ymax]
                ylim2 = [ymin, ymax]
            self.axes.set_ylim(ylim)
            self.axes2.set_ylim(ylim2)
        ylim = ylim2
        if inv_y2 and ylim[0] is not None:
            if ylim[0] < ylim[1]:
                ylim = ylim[::-1]
            self.axes2.set_ylim(ylim)
        else:
            if ylim[1] < ylim[0]:
                ylim = ylim[::-1]
            self.axes2.set_ylim(ylim)
        self._apply_x_limits()
        self.canvas.draw()
        self.toolbar.update()

    def _apply_x_limits(self):
        inv_x = self.checkBox_invX.isChecked()
        xlim = parse_entry(self.lineEdit_xlim.text())
        if not isinstance(xlim, list):
            xlim = self.axes.get_xlim()
        if inv_x and xlim[0] is not None:
            if xlim[0] < xlim[1]:
                xlim = xlim[::-1]
            self.axes.set_xlim(xlim)
        else:
            if xlim[1] < xlim[0]:
                xlim = xlim[::-1]
            self.axes.set_xlim(xlim)

    def redraw(self):
        x = self.comboBox_x.currentText()
        y = self.comboBox_y.currentText()
        y2 = self.comboBox_y2.currentText()
        self.axes.clear()
        self.axes2.clear()
        self.axes2.yaxis.set_label_position("right")
        self.axes2.yaxis.tick_right()
        vx = vy = vy2 = "None"
        if y or y2:
            if y:
                gy, vy = vardim2var(y, self.groups)
                tname = self.tname if self.usex else self.tname[gy]
                if vy == tname:
                    yy = self.time if self.usex else self.time[gy]
                    ylab = "Date"
                else:
                    yy = selvar(self, vy)
                    ylab = set_axis_label(yy)
                yy = self.slice_miss(self.yd, yy)
            if y2:
                gy2, vy2 = vardim2var(y2, self.groups)
                tname = self.tname if self.usex else self.tname[gy2]
                if vy2 == tname:
                    yy2 = self.time if self.usex else self.time[gy2]
                    ylab2 = "Date"
                else:
                    yy2 = selvar(self, vy2)
                    ylab2 = set_axis_label(yy2)
                yy2 = self.slice_miss(self.y2d, yy2)
            if x:
                gx, vx = vardim2var(x, self.groups)
                tname = self.tname if self.usex else self.tname[gx]
                if vx == tname:
                    xx = self.time if self.usex else self.time[gx]
                    xlab = "Date"
                else:
                    xx = selvar(self, vx)
                    xlab = set_axis_label(xx)
                xx = self.slice_miss(self.xd, xx)
            else:
                nx = yy.shape[0] if y else yy2.shape[0]
                xx = np.arange(nx)
                xlab = ""
            if not y:
                yy = np.ones_like(xx, dtype="float") * np.nan
                ylab = ""
            if not y2:
                yy2 = np.ones_like(xx, dtype="float") * np.nan
                ylab2 = ""
            try:
                self.line_y = self.axes.plot(xx, yy)
            except Exception:
                print(f"Scatter: x ({vx}) and y ({vy}) shapes do not match:", xx.shape, yy.shape)
                return
            try:
                self.line_y2 = self.axes2.plot(xx, yy2)
            except Exception:
                print(f"Scatter: x ({vx}) and y2 ({vy2}) shapes do not match:", xx.shape, yy2.shape)
                return
            self.axes.xaxis.set_label_text(xlab)
            self.axes.yaxis.set_label_text(ylab)
            self.axes2.xaxis.set_label_text(xlab)
            self.axes2.yaxis.set_label_text(ylab2)
            self.axes2.format_coord = lambda x0, y0: format_coord_scatter(
                x0, y0, self.axes, self.axes2, xx.dtype, yy.dtype, yy2.dtype)
            self.redraw_y()
            self.redraw_y2()
        self.canvas.draw()
        self.toolbar.update()


class ContourPanel(PlotPanel, Ui_ContourPanel):
    def __init__(self, window, session):
        super().__init__(window, session, "Contour")
        self._build_ui()
        self.reinit()

    def _build_ui(self):
        self.setupUi(self)
        self.connect_file_controls()
        self.figure = Figure(facecolor="white", figsize=(1, 1))
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plotLayout.addWidget(self.canvas, 1)
        self.plotLayout.addWidget(self.toolbar)

        self.zd = DimensionControlRow(self.maxdim)
        self.xd = DimensionControlRow(self.maxdim)
        self.yd = DimensionControlRow(self.maxdim)
        self.zDimensionsLayout.addWidget(self.zd)
        self.xDimensionsLayout.addWidget(self.xd)
        self.yDimensionsLayout.addWidget(self.yd)
        self.populate_cmap_combo(self.comboBox_cmap)

        self.pushButton_prevZ.clicked.connect(self.prev_z)
        self.pushButton_nextZ.clicked.connect(self.next_z)
        self.comboBox_z.currentIndexChanged.connect(self.selected_z)
        self.checkBox_transZ.stateChanged.connect(self.checked)
        self.lineEdit_zmin.editingFinished.connect(self.entered_z)
        self.lineEdit_zmax.editingFinished.connect(self.entered_z)
        self.zd.changed.connect(self.spinned_z)
        self.comboBox_x.currentIndexChanged.connect(self.selected_x)
        self.checkBox_invX.stateChanged.connect(self.checked)
        self.comboBox_y.currentIndexChanged.connect(self.selected_y)
        self.checkBox_invY.stateChanged.connect(self.checked)
        self.xd.changed.connect(self.spinned_x)
        self.yd.changed.connect(self.spinned_y)
        self.comboBox_cmap.currentIndexChanged.connect(self.selected_cmap)
        for check in (self.checkBox_revCmap, self.checkBox_mesh,
                      self.checkBox_grid):
            check.stateChanged.connect(self.checked)
        self.pushButton_quit.clicked.connect(QtWidgets.QApplication.quit)

    def reinit(self):
        super().reinit()
        self._updating = True
        columns = self.columns()
        for combo in (self.comboBox_z, self.comboBox_x, self.comboBox_y):
            _set_combo_items(combo, columns, "")
        for dims in (self.zd, self.xd, self.yd):
            dims.set_specs(empty_dimension_specs(self.maxdim))
        self.lineEdit_zmin.setText("None")
        self.lineEdit_zmax.setText("None")
        self._updating = False

    def checked(self):
        if not self._updating:
            self.redraw()

    def entered_z(self):
        if not self._updating:
            self.redraw()

    def spinned_x(self):
        self.checked()

    def spinned_y(self):
        self.checked()

    def spinned_z(self):
        self.checked()

    def selected_cmap(self):
        self.checked()

    def _move_z(self, step):
        idx = self.comboBox_z.currentIndex() + step
        if 0 < idx < self.comboBox_z.count():
            self.comboBox_z.setCurrentIndex(idx)

    def next_z(self):
        self._move_z(1)

    def prev_z(self):
        self._move_z(-1)

    def selected_x(self):
        if self._updating:
            return
        self.checkBox_invX.setChecked(False)
        self.xd.set_specs(
            dimension_specs(self, self.comboBox_x.currentText(), "x"))
        self.redraw()

    def selected_y(self):
        if self._updating:
            return
        self.checkBox_invY.setChecked(False)
        self.yd.set_specs(
            dimension_specs(self, self.comboBox_y.currentText(), "y"))
        self.redraw()

    def selected_z(self):
        if self._updating:
            return
        self.comboBox_x.setCurrentText("")
        self.comboBox_y.setCurrentText("")
        self.checkBox_invX.setChecked(False)
        self.checkBox_invY.setChecked(False)
        self.lineEdit_zmin.setText("None")
        self.lineEdit_zmax.setText("None")
        self.xd.set_specs(empty_dimension_specs(self.maxdim))
        self.yd.set_specs(empty_dimension_specs(self.maxdim))
        self.zd.set_specs(
            dimension_specs(self, self.comboBox_z.currentText(), "z"))
        self.redraw()

    def redraw(self):
        z = self.comboBox_z.currentText()
        trans_z = self.checkBox_transZ.isChecked()
        zmin = _float_or_none(self.lineEdit_zmin.text())
        zmax = _float_or_none(self.lineEdit_zmax.text())
        x = self.comboBox_x.currentText()
        y = self.comboBox_y.currentText()
        inv_x = self.checkBox_invX.isChecked()
        inv_y = self.checkBox_invY.isChecked()
        cmap = self.comboBox_cmap.currentText()
        if self.checkBox_revCmap.isChecked():
            cmap += "_r"
        mesh = self.checkBox_mesh.isChecked()
        grid = self.checkBox_grid.isChecked()
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        vx = vy = vz = "None"
        if z:
            gz, vz = vardim2var(z, self.groups)
            tname = self.tname if self.usex else self.tname[gz]
            if vz == tname:
                zz = self.dtime if self.usex else self.dtime[gz]
                zlab = "Year" if mesh else "Date"
                if not mesh:
                    zz = self.time if self.usex else self.time[gz]
            else:
                zz = selvar(self, vz)
                zlab = set_axis_label(zz)
            zz = self.slice_miss(self.zd, zz)
            if not trans_z:
                zz = zz.T
        else:
            zlab = ""
        if y:
            gy, vy = vardim2var(y, self.groups)
            tname = self.tname if self.usex else self.tname[gy]
            if vy == tname:
                yy = self.dtime if mesh and self.usex else self.time if self.usex else self.dtime[gy] if mesh else self.time[gy]
                ylab = "Year" if mesh else "Date"
            else:
                yy = selvar(self, vy)
                ylab = set_axis_label(yy)
            yy = self.slice_miss(self.yd, yy)
        else:
            ylab = ""
        if x:
            gx, vx = vardim2var(x, self.groups)
            tname = self.tname if self.usex else self.tname[gx]
            if vx == tname:
                xx = self.dtime if mesh and self.usex else self.time if self.usex else self.dtime[gx] if mesh else self.time[gx]
                xlab = "Year" if mesh else "Date"
            else:
                xx = selvar(self, vx)
                xlab = set_axis_label(xx)
            xx = self.slice_miss(self.xd, xx)
        else:
            xlab = ""
        if not z:
            nx = xx.shape[0] if x else 1
            ny = yy.shape[0] if y else 1
            zz = np.ones((ny, nx)) * np.nan
        if zz.ndim < 2:
            print(f"Contour: z ({vz}) is not 2-dimensional:", zz.shape)
            return
        if not x:
            xx = np.arange(zz.shape[1])
        if not y:
            yy = np.arange(zz.shape[0])
        extend = "neither"
        if zmin is not None:
            zz = np.maximum(zz, zmin)
            extend = "min" if zmax is None else "both"
        if zmax is not None:
            zz = np.minimum(zz, zmax)
            extend = "max" if zmin is None else "both"
        try:
            if mesh:
                cc = self.axes.pcolormesh(xx, yy, zz, vmin=zmin, vmax=zmax,
                                          cmap=cmap, shading="nearest")
                cb = self.figure.colorbar(cc, fraction=0.05, shrink=0.75,
                                          extend=extend)
            else:
                cc = self.axes.contourf(xx, yy, zz, vmin=zmin, vmax=zmax,
                                        cmap=cmap, extend=extend)
                cb = self.figure.colorbar(cc, fraction=0.05, shrink=0.75)
        except Exception:
            print(f"Contour: x ({vx}), y ({vy}), z ({vz}) shapes do not match:",
                  xx.shape, yy.shape, zz.shape)
            return
        cb.set_label(zlab)
        self.axes.xaxis.set_label_text(xlab)
        self.axes.yaxis.set_label_text(ylab)
        self.axes.format_coord = lambda x0, y0: format_coord_contour(
            x0, y0, self.axes, xx, yy, zz)
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        if inv_x:
            self.axes.set_xlim(xlim[::-1])
        if inv_y:
            self.axes.set_ylim(ylim[::-1])
        if grid:
            self.axes.grid(True, color="white", linewidth=0.5)
        self.canvas.draw()
        self.toolbar.update()


class MapUnavailablePanel(QtWidgets.QWidget, Ui_MapUnavailablePanel):
    def __init__(self, error=None):
        super().__init__()
        self.setupUi(self)
        message = "Map view is unavailable because Cartopy could not be imported."
        if error is not None:
            message += f"\n\n{type(error).__name__}: {error}"
        message += f"\n\nPython executable: {sys.executable}"
        self.label_message.setText(message)

    def reinit(self):
        pass

    def redraw(self):
        pass


class MapPanel(PlotPanel, Ui_MapPanel):
    def __init__(self, window, session):
        if not ensure_cartopy():
            raise RuntimeError("Cartopy is required for MapPanel") from CARTOPY_IMPORT_ERROR
        super().__init__(window, session, "Map")
        self.iunlim = -1
        self.nunlim = 0
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(lambda: self.update_frame(False))
        self.anim_inc = 1
        self._updating = True
        self._build_ui()
        self._updating = False
        self.reinit()

    def _build_ui(self):
        self.setupUi(self)
        self.connect_file_controls()
        self.figure = Figure(facecolor="white", figsize=(1, 1))
        self.axes = self.figure.add_subplot(111, projection=ccrs.PlateCarree())
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plotLayout.addWidget(self.canvas, 1)
        self.plotLayout.addWidget(self.toolbar)

        self.vd = DimensionControlRow(self.maxdim)
        self.lond = DimensionControlRow(self.maxdim)
        self.latd = DimensionControlRow(self.maxdim)
        self.vDimensionsLayout.addWidget(self.vd)
        self.lonDimensionsLayout.addWidget(self.lond)
        self.latDimensionsLayout.addWidget(self.latd)
        self.populate_cmap_combo(self.comboBox_cmap)

        self.projs = ["AlbersEqualArea", "AzimuthalEquidistant", "EckertI",
                      "EckertII", "EckertIII", "EckertIV", "EckertV",
                      "EckertVI", "EqualEarth", "EquidistantConic",
                      "InterruptedGoodeHomolosine",
                      "LambertAzimuthalEqualArea", "LambertConformal",
                      "LambertCylindrical", "Mercator", "Miller", "Mollweide",
                      "NorthPolarStereo", "PlateCarree", "Robinson",
                      "Sinusoidal", "SouthPolarStereo", "Stereographic"]
        self.iprojs = [ccrs.AlbersEqualArea, ccrs.AzimuthalEquidistant,
                       ccrs.EckertI, ccrs.EckertII, ccrs.EckertIII,
                       ccrs.EckertIV, ccrs.EckertV, ccrs.EckertVI,
                       ccrs.EqualEarth, ccrs.EquidistantConic,
                       ccrs.InterruptedGoodeHomolosine,
                       ccrs.LambertAzimuthalEqualArea, ccrs.LambertConformal,
                       ccrs.LambertCylindrical, ccrs.Mercator, ccrs.Miller,
                       ccrs.Mollweide, ccrs.NorthPolarStereo, ccrs.PlateCarree,
                       ccrs.Robinson, ccrs.Sinusoidal, ccrs.SouthPolarStereo,
                       ccrs.Stereographic]
        self.comboBox_projection.clear()
        self.comboBox_projection.addItems(self.projs)
        self.comboBox_projection.setCurrentText("PlateCarree")

        self.horizontalSlider_timeStep.valueChanged.connect(self.tstep_t)
        self.pushButton_firstTime.clicked.connect(self.first_t)
        self.pushButton_prevTime.clicked.connect(self.prev_t)
        self.pushButton_runBackward.clicked.connect(self.prun_t)
        self.pushButton_runForward.clicked.connect(self.nrun_t)
        self.pushButton_nextTime.clicked.connect(self.next_t)
        self.pushButton_lastTime.clicked.connect(self.last_t)
        self.comboBox_repeat.currentIndexChanged.connect(self.repeat_t)
        self.pushButton_prevVariable.clicked.connect(self.prev_v)
        self.pushButton_nextVariable.clicked.connect(self.next_v)
        self.comboBox_variable.currentIndexChanged.connect(self.selected_v)
        self.checkBox_transVariable.stateChanged.connect(self.checked)
        self.lineEdit_vmin.editingFinished.connect(self.entered_v)
        self.lineEdit_vmax.editingFinished.connect(self.entered_v)
        self.checkBox_allValues.stateChanged.connect(self.checked_all)
        self.vd.changed.connect(self.spinned_v)
        self.comboBox_lon.currentIndexChanged.connect(self.selected_lon)
        self.checkBox_invLon.stateChanged.connect(self.checked)
        self.checkBox_shiftLon.stateChanged.connect(self.checked)
        self.comboBox_lat.currentIndexChanged.connect(self.selected_lat)
        self.checkBox_invLat.stateChanged.connect(self.checked)
        self.lond.changed.connect(self.spinned_lon)
        self.latd.changed.connect(self.spinned_lat)
        self.comboBox_cmap.currentIndexChanged.connect(self.selected_cmap)
        for check in (self.checkBox_revCmap, self.checkBox_mesh,
                      self.checkBox_global, self.checkBox_coast,
                      self.checkBox_borders, self.checkBox_rivers,
                      self.checkBox_lakes, self.checkBox_grid):
            check.stateChanged.connect(self.checked)
        self.comboBox_projection.currentIndexChanged.connect(self.selected_proj)
        self.lineEdit_centralLon.editingFinished.connect(self.entered_clon)
        self.pushButton_quit.clicked.connect(QtWidgets.QApplication.quit)

    def reinit(self):
        super().reinit()
        self._updating = True
        self.iunlim = -1
        self.nunlim = 0
        columns = self.columns()
        for combo in (self.comboBox_variable, self.comboBox_lon,
                      self.comboBox_lat):
            _set_combo_items(combo, columns, "")
        for dims in (self.vd, self.lond, self.latd):
            dims.set_specs(empty_dimension_specs(self.maxdim))
        self.lineEdit_vmin.setText("None")
        self.lineEdit_vmax.setText("None")
        self.horizontalSlider_timeStep.setRange(0, 1)
        self.horizontalSlider_timeStep.setValue(0)
        self.comboBox_repeat.setCurrentText("repeat")
        if self.usex:
            if self.lonvar:
                self.comboBox_lon.setCurrentText(self.lonvar)
                self.lond.set_specs(dimension_specs(
                    self, self.comboBox_lon.currentText(), "lon"))
            if self.latvar:
                self.comboBox_lat.setCurrentText(self.latvar)
                self.latd.set_specs(dimension_specs(
                    self, self.comboBox_lat.currentText(), "lat"))
        else:
            if any(self.lonvar):
                lon = next(item for item in self.lonvar if item)
                self.comboBox_lon.setCurrentText(lon)
                self.lond.set_specs(dimension_specs(
                    self, self.comboBox_lon.currentText(), "lon"))
            if any(self.latvar):
                lat = next(item for item in self.latvar if item)
                self.comboBox_lat.setCurrentText(lat)
                self.latd.set_specs(dimension_specs(
                    self, self.comboBox_lat.currentText(), "lat"))
        self._updating = False

    def checked(self):
        if not self._updating:
            self.redraw()

    def checked_all(self):
        if self._updating:
            return
        vmin, vmax = self.get_vminmax()
        self.lineEdit_vmin.setText(str(vmin))
        self.lineEdit_vmax.setText(str(vmax))
        self.redraw()

    def entered_clon(self):
        self.checked()

    def entered_v(self):
        self.checked()

    def selected_cmap(self):
        self.checked()

    def selected_proj(self):
        self.checked()

    def repeat_t(self):
        pass

    def first_t(self):
        self.set_tstep(0)
        self.update_frame(True)

    def last_t(self):
        self.set_tstep(max(self.nunlim - 1, 0))
        self.update_frame(True)

    def nrun_t(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pushButton_runForward.setText(">")
        else:
            self.anim_inc = 1
            self.pushButton_runBackward.setText("<")
            self.pushButton_runForward.setText("||")
            self.timer.start()

    def prun_t(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pushButton_runBackward.setText("<")
        else:
            self.anim_inc = -1
            self.pushButton_runForward.setText(">")
            self.pushButton_runBackward.setText("||")
            self.timer.start()

    def next_t(self):
        it = self._current_time_index()
        if it < self.nunlim - 1:
            it += 1
        elif self.comboBox_repeat.currentText() == "repeat":
            it = 0
        elif self.comboBox_repeat.currentText() == "reflect" and it > 0:
            it -= 1
        self.set_tstep(it)
        self.update_frame(True)

    def prev_t(self):
        it = self._current_time_index()
        if it > 0:
            it -= 1
        elif self.comboBox_repeat.currentText() == "repeat":
            it = max(self.nunlim - 1, 0)
        elif (self.comboBox_repeat.currentText() == "reflect"
              and self.nunlim > 1):
            it += 1
        self.set_tstep(it)
        self.update_frame(True)

    def _move_v(self, step):
        idx = self.comboBox_variable.currentIndex() + step
        if 0 < idx < self.comboBox_variable.count():
            self.comboBox_variable.setCurrentIndex(idx)

    def next_v(self):
        self._move_v(1)

    def prev_v(self):
        self._move_v(-1)

    def selected_lat(self):
        if self._updating:
            return
        self.checkBox_invLat.setChecked(False)
        self.latd.set_specs(
            dimension_specs(self, self.comboBox_lat.currentText(), "lat"))
        self.redraw()

    def selected_lon(self):
        if self._updating:
            return
        self.checkBox_invLon.setChecked(False)
        self.checkBox_shiftLon.setChecked(False)
        self.lond.set_specs(
            dimension_specs(self, self.comboBox_lon.currentText(), "lon"))
        self.redraw()

    def selected_v(self):
        if self._updating:
            return
        v = self.comboBox_variable.currentText()
        if not v:
            self.redraw()
            return
        self.set_unlim(v)
        self.horizontalSlider_timeStep.setRange(0, max(self.nunlim - 1, 0))
        self.set_tstep(0)
        vmin, vmax = self.get_vminmax()
        self.lineEdit_vmin.setText(str(vmin))
        self.lineEdit_vmax.setText(str(vmax))
        self.vd.set_specs(dimension_specs(self, v, "var"))
        self.redraw()

    def spinned_lon(self):
        self.checked()

    def spinned_lat(self):
        self.checked()

    def spinned_v(self):
        try:
            self.set_tstep(int(self.vd.values()[self.iunlim]))
        except (ValueError, IndexError):
            pass
        self.checked()

    def tstep_t(self, step):
        if self._updating:
            return
        self.set_tstep(int(step))
        self.update_frame(True)

    def _current_time_index(self):
        try:
            return int(self.vd.values()[self.iunlim])
        except (ValueError, IndexError):
            return 0

    def get_vminmax(self):
        v = self.comboBox_variable.currentText()
        if not v:
            return 0, 1
        gz, vz = vardim2var(v, self.groups)
        tname = self.tname if self.usex else self.tname[gz]
        if vz == tname:
            return 0, 1
        vv = selvar(self, vz)
        imiss = get_miss(self, vv)
        if self.checkBox_allValues.isChecked() or (np.sum(vv.shape[:-2]) < 50):
            arr = set_miss(imiss, vv)
            return np.nanmin(arr), np.nanmax(arr)
        rng = np.random.default_rng()
        vmin = np.inf
        vmax = -np.inf
        for _ in range(50):
            ss = []
            for i in range(vv.ndim):
                if i < vv.ndim - 2:
                    idim = rng.integers(0, vv.shape[i])
                    ss.append(slice(idim, idim + 1))
                else:
                    ss.append(slice(0, vv.shape[i]))
            arr = set_miss(imiss, vv[tuple(ss)])
            vmin = min(vmin, np.nanmin(arr))
            vmax = max(vmax, np.nanmax(arr))
        return vmin, vmax

    def set_tstep(self, it):
        v = self.comboBox_variable.currentText()
        if not v:
            return
        gz, vz = vardim2var(v, self.groups)
        if self.usex:
            dunlim = self.dunlim
            time = self.time
        else:
            dunlim = self.dunlim[gz]
            time = self.time[gz]
        try:
            zz = selvar(self, vz)
            dims = zz.dims if self.usex else zz.dimensions
            has_unlim = dunlim in dims
        except Exception:
            has_unlim = False
        if dunlim and has_unlim and self.iunlim >= 0:
            self.vd.set_value(self.iunlim, it)
            self.horizontalSlider_timeStep.blockSignals(True)
            self.horizontalSlider_timeStep.setValue(int(it))
            self.horizontalSlider_timeStep.blockSignals(False)
            if self.usex:
                self.label_timeValue.setText(str(time.values[it]))
            else:
                try:
                    self.label_timeValue.setText(str(np.around(time[it], 4)))
                except TypeError:
                    self.label_timeValue.setText(str(time[it]))

    def set_unlim(self, v):
        gz, vz = vardim2var(v, self.groups)
        if self.usex:
            tname = self.tname
            time = self.time
            dunlim = self.dunlim
        else:
            tname = self.tname[gz]
            time = self.time[gz]
            dunlim = self.dunlim[gz]
        if vz == tname:
            self.iunlim = 0
            self.nunlim = time.size
        else:
            zz = selvar(self, vz)
            dims = zz.dims if self.usex else zz.dimensions
            self.iunlim = dims.index(dunlim) if dunlim and dunlim in dims else 0
            self.nunlim = zz.shape[self.iunlim] if zz.ndim > 0 else 0

    def redraw(self):
        self.timer.stop()
        self.pushButton_runForward.setText(">")
        self.pushButton_runBackward.setText("<")
        v = self.comboBox_variable.currentText()
        trans_v = self.checkBox_transVariable.isChecked()
        vmin = _float_or_none(self.lineEdit_vmin.text())
        vmax = _float_or_none(self.lineEdit_vmax.text())
        x = self.comboBox_lon.currentText()
        y = self.comboBox_lat.currentText()
        inv_lon = self.checkBox_invLon.isChecked()
        inv_lat = self.checkBox_invLat.isChecked()
        shift_lon = self.checkBox_shiftLon.isChecked()
        cmap = self.comboBox_cmap.currentText()
        if self.checkBox_revCmap.isChecked():
            cmap += "_r"
        mesh = self.checkBox_mesh.isChecked()
        self.iiglobal = self.checkBox_global.isChecked()
        coast = self.checkBox_coast.isChecked()
        borders = self.checkBox_borders.isChecked()
        rivers = self.checkBox_rivers.isChecked()
        lakes = self.checkBox_lakes.isChecked()
        grid = self.checkBox_grid.isChecked()
        proj_name = self.comboBox_projection.currentText()
        self.iproj = self.iprojs[self.projs.index(proj_name)]
        clon = self.lineEdit_centralLon.text()
        vx = vy = vz = "None"
        if v:
            gz, vz = vardim2var(v, self.groups)
            tname = self.tname if self.usex else self.tname[gz]
            if vz == tname:
                vv = self.dtime if mesh and self.usex else self.time if self.usex else self.dtime[gz] if mesh else self.time[gz]
                vlab = "Year" if mesh else "Date"
            else:
                vv = selvar(self, vz)
                vlab = set_axis_label(vv)
            vv = self.slice_miss(self.vd, vv)
            if trans_v:
                vv = vv.T
            if shift_lon:
                vv = np.roll(vv, vv.shape[1] // 2, axis=1)
        else:
            vlab = ""
        if y:
            gy, vy = vardim2var(y, self.groups)
            tname = self.tname if self.usex else self.tname[gy]
            if vy == tname:
                yy = self.dtime if mesh and self.usex else self.time if self.usex else self.dtime[gy] if mesh else self.time[gy]
                ylab = "Year" if mesh else "Date"
            else:
                yy = selvar(self, vy)
                ylab = set_axis_label(yy)
            yy = self.slice_miss(self.latd, yy)
        else:
            ylab = ""
        if x:
            gx, vx = vardim2var(x, self.groups)
            tname = self.tname if self.usex else self.tname[gx]
            if vx == tname:
                xx = self.dtime if mesh and self.usex else self.time if self.usex else self.dtime[gx] if mesh else self.time[gx]
                xlab = "Year" if mesh else "Date"
            else:
                xx = selvar(self, vx)
                xlab = set_axis_label(xx)
            xx = self.slice_miss(self.lond, xx)
            xx360 = (xx + 360.0) % 360.0 if np.any(np.isfinite(xx)) else xx
            if xx.size > 1:
                if xx.ndim > 1:
                    x0 = xx[:, 0].mean()
                    x1 = xx[:, -2].mean() if self.iiglobal else xx[:, -1].mean()
                else:
                    x0 = xx[0]
                    x1 = xx[-2] if self.iiglobal else xx[-1]
                self.ixxmean = 0.5 * (x1 + x0)
                if self.iiglobal:
                    self.ixxmean = np.around(self.ixxmean / 180.0, 0) * 180.0
            else:
                self.ixxmean = xx360[0]
            if self.ixxmean > 180.0:
                self.ixxmean -= 360.0
        else:
            xlab = ""
            self.ixxmean = 0.0
        self.iclon = float(clon) if clon != "None" else self.ixxmean
        self.figure.clear()
        self.axes = self.figure.add_subplot(
            111, projection=self.iproj(central_longitude=self.iclon))
        if v:
            if vv.ndim < 2:
                print(f"Map: var ({vz}) is not 2-dimensional:", vv.shape)
                return
            if not x:
                nx = vv.shape[1]
                xx = -180.0 + np.arange(nx) / float(nx) * 360.0
                xx += 0.5 * (xx[1] - xx[0])
                xlab = ""
            if not y:
                ny = vv.shape[0]
                yy = -90.0 + np.arange(ny) / float(ny) * 180.0
                yy += 0.5 * (yy[1] - yy[0])
                ylab = ""
            extend = "neither"
            if vmin is not None:
                vv = np.maximum(vv, vmin)
                extend = "min" if vmax is None else "both"
            if vmax is not None:
                vv = np.minimum(vv, vmax)
                extend = "max" if vmin is None else "both"
            if xx.ndim == 1 and yy.ndim == 1:
                self.ixx, self.iyy = np.meshgrid(xx, yy)
            elif xx.ndim == 1 and yy.ndim == 2:
                self.ixx, _tmp = np.meshgrid(xx, yy[:, 0])
                self.iyy = yy
            elif xx.ndim == 2 and yy.ndim == 1:
                self.ixx, self.iyy = np.meshgrid(xx, yy)
                _tmp, self.iyy = np.meshgrid(xx[0, :], yy)
                self.ixx = xx
            elif xx.ndim == 2 and yy.ndim == 2:
                self.ixx = xx
                self.iyy = yy
            else:
                print(f"Map: lon ({vx}), lat ({vy}) dimensions not 1D or 2D:",
                      xx.shape, yy.shape)
                return
            if inv_lon:
                self.ixx = np.fliplr(self.ixx)
            if inv_lat:
                self.iyy = np.flipud(self.iyy)
            self.ivv = vv
            if self.iiglobal:
                self.ivvc, self.ixxc, self.iyyc = add_cyclic(
                    self.ivv, x=self.ixx, y=self.iyy)
            else:
                self.ivvc = self.ivv
                self.ixxc = self.ixx
                self.iyyc = self.iyy
            self.itrans = ccrs.PlateCarree()
            self.ivmin = vmin
            self.ivmax = vmax
            self.icmap = cmap
            self.ncmap = mpl.colormaps[self.icmap].N
            self.ncmap = self.ncmap if self.ncmap < 256 else 15
            self.iextend = extend
            try:
                if mesh:
                    self.cc = self.axes.pcolormesh(
                        self.ixx, self.iyy, self.ivv,
                        vmin=self.ivmin, vmax=self.ivmax,
                        cmap=self.icmap, shading="nearest",
                        transform=self.itrans)
                    self.cb = self.figure.colorbar(
                        self.cc, fraction=0.05, shrink=0.75, pad=0.07,
                        extend=self.iextend)
                else:
                    self.cc = self.axes.contourf(
                        self.ixxc, self.iyyc, self.ivvc, self.ncmap,
                        vmin=self.ivmin, vmax=self.ivmax, cmap=self.icmap,
                        extend=self.iextend, transform=self.itrans)
                    self.cb = self.figure.colorbar(
                        self.cc, fraction=0.05, shrink=0.75, pad=0.07)
            except Exception:
                print(f"Map: lon ({vx}), lat ({vy}), var ({vz}) shapes do not match:",
                      self.ixx.shape, self.iyy.shape, self.ivv.shape)
                return
            self.cb.set_label(vlab)
            self.axes.format_coord = lambda x0, y0: format_coord_map(
                x0, y0, self.axes, self.ixx, self.iyy, self.ivv)
        if self.iiglobal:
            self.axes.set_global()
        if coast:
            self.axes.add_feature(cfeature.COASTLINE)
            self.axes.gridlines(draw_labels=True, linewidth=0,
                                x_inline=False, y_inline=False)
        if borders:
            self.axes.add_feature(cfeature.BORDERS, edgecolor="grey")
        if rivers:
            self.axes.add_feature(cfeature.RIVERS)
        if lakes:
            self.axes.add_feature(cfeature.LAKES, alpha=0.5)
        self.axes.xaxis.set_label_text(xlab)
        self.axes.yaxis.set_label_text(ylab)
        if grid:
            self.axes.gridlines(draw_labels=False, x_inline=False, y_inline=False)
        self.canvas.draw()
        self.toolbar.update()

    def update_frame(self, isframe=False):
        v = self.comboBox_variable.currentText()
        if not v:
            return
        trans_v = self.checkBox_transVariable.isChecked()
        mesh = self.checkBox_mesh.isChecked()
        rep = self.comboBox_repeat.currentText()
        shift_lon = self.checkBox_shiftLon.isChecked()
        gz, vz = vardim2var(v, self.groups)
        if self.usex:
            if vz == self.tname:
                vz = self.tvar
        else:
            if vz == self.tname[gz]:
                vz = self.tvar[gz]
        vv = selvar(self, vz)
        it = self._current_time_index()
        if not isframe:
            if self.anim_inc == 1 and it == self.nunlim - 1:
                if rep == "repeat":
                    it = 0
                elif rep == "reflect":
                    self.anim_inc = -1
                    it += self.anim_inc
                else:
                    self.timer.stop()
                    self.pushButton_runForward.setText(">")
            elif self.anim_inc == -1 and it == 0:
                if rep == "repeat":
                    it = self.nunlim - 1
                elif rep == "reflect":
                    self.anim_inc = 1
                    it += self.anim_inc
                else:
                    self.timer.stop()
                    self.pushButton_runBackward.setText("<")
            else:
                it += self.anim_inc
        self.set_tstep(it)
        vv = self.slice_miss(self.vd, vv)
        if vv.ndim < 2:
            self.timer.stop()
            return
        if trans_v:
            vv = vv.T
        if shift_lon:
            vv = np.roll(vv, vv.shape[1] // 2, axis=1)
        self.ivv = vv
        if mesh:
            self.cc.remove()
            self.cc = self.axes.pcolormesh(
                self.ixx, self.iyy, self.ivv, vmin=self.ivmin, vmax=self.ivmax,
                cmap=self.icmap, shading="nearest", transform=self.itrans)
        else:
            self.cc.remove()
            if self.iiglobal:
                self.ivvc, self.ixxc = add_cyclic(self.ivv, x=self.ixx)
            else:
                self.ivvc = self.ivv
            self.cc = self.axes.contourf(
                self.ixxc, self.iyyc, self.ivvc, self.ncmap,
                vmin=self.ivmin, vmax=self.ivmax, cmap=self.icmap,
                extend=self.iextend, transform=self.itrans)
        self.canvas.draw_idle()


class NcvMainWindow(QtWidgets.QMainWindow, Ui_NcvMainWindow):
    instances = []

    def __init__(self, session: NcvSession, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.session = session
        self._children = []

        def tab_text(page, fallback):
            index = self.tabWidget_main.indexOf(page)
            return self.tabWidget_main.tabText(index) if index >= 0 else fallback

        scatter_label = tab_text(self.tab_scatterPlot, "Scatter/Line")
        contour_label = tab_text(self.tab_contourPlot, "Contour")
        map_label = tab_text(self.tab_mapPlot, "Map")
        for page in (self.tab_scatterPlot, self.tab_contourPlot,
                     self.tab_mapPlot):
            index = self.tabWidget_main.indexOf(page)
            if index >= 0:
                self.tabWidget_main.removeTab(index)

        self.scatter = ScatterPanel(self, session)
        self.contour = ContourPanel(self, session)
        if ensure_cartopy():
            self.map = MapPanel(self, session)
        else:
            self.map = MapUnavailablePanel(CARTOPY_IMPORT_ERROR)
        self.tabWidget_main.insertTab(0, self.scatter, scatter_label)
        self.tabWidget_main.insertTab(1, self.contour, contour_label)
        self.tabWidget_main.insertTab(2, self.map, map_label)
        self.tabWidget_main.setCurrentIndex(0)
        self.tabWidget_main.currentChanged.connect(self._tab_changed)
        self.setWindowIcon(QtGui.QIcon(_resource_path("images", "ncvue_icon.png")))
        w, h, x, y = _window_geometry()
        self.setGeometry(x, y, w, h)
        self.update_title()
        self.select_default_tab()
        NcvMainWindow.instances.append(self)

    def update_title(self):
        suffix = self.session.title_suffix
        self.setWindowTitle(f"ncv {suffix}".strip())

    def select_default_tab(self):
        if not HAVE_CARTOPY:
            return
        mapfirst = False
        try:
            if self.session.usex:
                for attr in (self.session.latvar, self.session.lonvar):
                    if attr:
                        ivar = attr[0:attr.rfind("(")].rstrip()
                        mapfirst = mapfirst or np.prod(self.session.fi[ivar].shape) > 1
            else:
                for attr in (self.session.latvar, self.session.lonvar):
                    if any(attr):
                        first = next(item for item in attr if item)
                        _group, var = vardim2var(first, self.session.groups)
                        mapfirst = mapfirst or np.prod(selvar(self.session, var).shape) > 1
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
            self, "Choose netcdf file(s)", "",
            "NetCDF files (*.nc *.nc4 *.cdf);;All files (*)")
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
        for panel in (self.scatter, self.contour, self.map):
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
        if not any(window.session is self.session for window in NcvMainWindow.instances):
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
