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

    def add_file_controls(self, layout):
        open_file = QtWidgets.QPushButton("Open File")
        open_file.setToolTip("Open new netcdf file(s)")
        open_file.clicked.connect(lambda: self.window.open_file_dialog(False))
        layout.addWidget(open_file)

        if HAVE_XARRAY:
            open_xarray = QtWidgets.QPushButton("Open xarray")
            open_xarray.setToolTip("Open new netcdf file(s) with xarray")
            open_xarray.clicked.connect(lambda: self.window.open_file_dialog(True))
            layout.addWidget(open_xarray)

        new_window = QtWidgets.QPushButton("New Window")
        new_window.setToolTip("Open secondary ncv window")
        new_window.clicked.connect(self.window.create_secondary_window)
        layout.addWidget(new_window)
        layout.addStretch(1)

    def make_combo(self, values=None, callback=None, tooltip=""):
        combo = QtWidgets.QComboBox()
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        combo.addItems([str(value) for value in (values or [])])
        combo.setToolTip(tooltip)
        if callback is not None:
            combo.currentIndexChanged.connect(callback)
        return combo

    def make_entry(self, text="", callback=None, width=70, tooltip=""):
        entry = QtWidgets.QLineEdit(str(text))
        entry.setFixedWidth(width)
        entry.setToolTip(tooltip)
        if callback is not None:
            entry.editingFinished.connect(callback)
        return entry

    def make_check(self, label, checked=False, callback=None, tooltip=""):
        check = QtWidgets.QCheckBox(label)
        check.setChecked(bool(checked))
        check.setToolTip(tooltip)
        if callback is not None:
            check.stateChanged.connect(callback)
        return check

    def make_cmap_combo(self, callback=None):
        combo = self.make_combo(callback=None, tooltip="Choose colormap")
        for cmap in sorted([c for c in plt.colormaps() if not c.endswith("_r")]):
            icon_path = _resource_path("images", f"{cmap}.png")
            if os.path.exists(icon_path):
                combo.addItem(QtGui.QIcon(icon_path), cmap)
            else:
                combo.addItem(cmap)
        combo.setCurrentText("RdYlBu")
        if callback is not None:
            combo.currentIndexChanged.connect(callback)
        return combo

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


class ScatterPanel(PlotPanel):
    def __init__(self, window, session):
        super().__init__(window, session, "Scatter/Line")
        self.line_y = []
        self.line_y2 = []
        self._build_ui()
        self.reinit()

    def _build_ui(self):
        main = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        self.add_file_controls(row)
        main.addLayout(row)

        self.figure = Figure(facecolor="white", figsize=(1, 1))
        self.axes = self.figure.add_subplot(111)
        self.axes2 = self.axes.twinx()
        self.axes2.yaxis.set_label_position("right")
        self.axes2.yaxis.tick_right()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        main.addWidget(self.canvas, 1)
        main.addWidget(self.toolbar)

        controls = QtWidgets.QVBoxLayout()
        main.addLayout(controls)

        rowxy = QtWidgets.QHBoxLayout()
        rowxy.addWidget(QtWidgets.QLabel("x"))
        self.x = self.make_combo(callback=self.selected_x,
                                 tooltip='Choose variable of x-axis. Empty uses index.')
        rowxy.addWidget(self.x)
        self.inv_x = self.make_check("invert x", callback=self.checked_x,
                                     tooltip="Invert x-axis")
        rowxy.addWidget(self.inv_x)
        rowxy.addSpacing(16)
        rowxy.addWidget(QtWidgets.QLabel("y"))
        rowxy.addWidget(self._nav_button("<", self.prev_y))
        rowxy.addWidget(self._nav_button(">", self.next_y))
        self.y = self.make_combo(callback=self.selected_y,
                                 tooltip="Choose variable of y-axis")
        rowxy.addWidget(self.y)
        self.inv_y = self.make_check("invert y", callback=self.checked_y,
                                     tooltip="Invert y-axis")
        rowxy.addWidget(self.inv_y)
        self.redraw_button = QtWidgets.QPushButton("Redraw")
        self.redraw_button.clicked.connect(self.redraw)
        rowxy.addWidget(self.redraw_button)
        controls.addLayout(rowxy)

        dimrow = QtWidgets.QHBoxLayout()
        self.xd = DimensionControlRow(self.maxdim)
        self.xd.changed.connect(self.spinned_x)
        self.yd = DimensionControlRow(self.maxdim)
        self.yd.changed.connect(self.spinned_y)
        dimrow.addWidget(self.xd)
        dimrow.addSpacing(16)
        dimrow.addWidget(self.yd)
        controls.addLayout(dimrow)

        style1 = QtWidgets.QHBoxLayout()
        c = list(plt.rcParams["axes.prop_cycle"])
        col1 = c[0]["color"]
        col2 = c[3]["color"]
        self.ls = self._entry_with_label(style1, "ls", "-", self.entered_y, 55)
        self.lw = self._entry_with_label(style1, "lw", "1", self.entered_y, 40)
        self.lc = self._entry_with_label(style1, "c", col1, self.entered_y, 80)
        self.marker = self._entry_with_label(style1, "marker", "None", self.entered_y, 60)
        self.ms = self._entry_with_label(style1, "ms", "1", self.entered_y, 40)
        self.mfc = self._entry_with_label(style1, "mfc", col1, self.entered_y, 80)
        self.mec = self._entry_with_label(style1, "mec", col1, self.entered_y, 80)
        self.mew = self._entry_with_label(style1, "mew", "1", self.entered_y, 40)
        style1.addStretch(1)
        controls.addLayout(style1)

        limits = QtWidgets.QHBoxLayout()
        self.xlim = self._entry_with_label(limits, "xlim", "None", self.entered_y, 110)
        self.ylim = self._entry_with_label(limits, "ylim", "None", self.entered_y, 110)
        limits.addStretch(1)
        controls.addLayout(limits)

        rowy2 = QtWidgets.QHBoxLayout()
        rowy2.addWidget(QtWidgets.QLabel("y2"))
        rowy2.addWidget(self._nav_button("<", self.prev_y2))
        rowy2.addWidget(self._nav_button(">", self.next_y2))
        self.y2 = self.make_combo(callback=self.selected_y2,
                                  tooltip="Choose variable for right-hand-side y-axis")
        rowy2.addWidget(self.y2)
        self.inv_y2 = self.make_check("invert y2", callback=self.checked_y2,
                                      tooltip="Invert right-hand-side y-axis")
        rowy2.addWidget(self.inv_y2)
        self.same_y = self.make_check("same y-axes", callback=self.checked_yy2,
                                      tooltip="Same limits for both y-axes")
        rowy2.addWidget(self.same_y)
        rowy2.addStretch(1)
        controls.addLayout(rowy2)

        dimrow2 = QtWidgets.QHBoxLayout()
        self.y2d = DimensionControlRow(self.maxdim)
        self.y2d.changed.connect(self.spinned_y2)
        dimrow2.addWidget(self.y2d)
        dimrow2.addStretch(1)
        controls.addLayout(dimrow2)

        style2 = QtWidgets.QHBoxLayout()
        self.ls2 = self._entry_with_label(style2, "ls", "-", self.entered_y2, 55)
        self.lw2 = self._entry_with_label(style2, "lw", "1", self.entered_y2, 40)
        self.lc2 = self._entry_with_label(style2, "c", col2, self.entered_y2, 80)
        self.marker2 = self._entry_with_label(style2, "marker", "None", self.entered_y2, 60)
        self.ms2 = self._entry_with_label(style2, "ms", "1", self.entered_y2, 40)
        self.mfc2 = self._entry_with_label(style2, "mfc", col2, self.entered_y2, 80)
        self.mec2 = self._entry_with_label(style2, "mec", col2, self.entered_y2, 80)
        self.mew2 = self._entry_with_label(style2, "mew", "1", self.entered_y2, 40)
        style2.addStretch(1)
        controls.addLayout(style2)

        rowquit = QtWidgets.QHBoxLayout()
        self.y2lim = self._entry_with_label(rowquit, "y2lim", "None", self.entered_y2, 110)
        rowquit.addStretch(1)
        quit_button = QtWidgets.QPushButton("Quit")
        quit_button.clicked.connect(QtWidgets.QApplication.quit)
        rowquit.addWidget(quit_button)
        controls.addLayout(rowquit)

    def _entry_with_label(self, layout, label, text, callback, width):
        layout.addWidget(QtWidgets.QLabel(label))
        entry = self.make_entry(text, callback=callback, width=width)
        layout.addWidget(entry)
        return entry

    def _nav_button(self, label, callback):
        button = QtWidgets.QPushButton(label)
        button.setFixedWidth(34)
        button.clicked.connect(callback)
        return button

    def reinit(self):
        super().reinit()
        self._updating = True
        columns = self.columns()
        for combo in (self.x, self.y, self.y2):
            _set_combo_items(combo, columns, "")
        for dims in (self.xd, self.yd, self.y2d):
            dims.set_specs(empty_dimension_specs(self.maxdim))
        self.xlim.setText("None")
        self.ylim.setText("None")
        self.y2lim.setText("None")
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
            self.ylim.setText("None")
            self.y2lim.setText("None")
            self.redraw_y()
            self.redraw_y2()

    def entered_y(self):
        if not self._updating:
            self.redraw_y()

    def entered_y2(self):
        if not self._updating:
            self.redraw_y2()

    def _move_combo(self, combo, step):
        idx = combo.currentIndex() + step
        if 0 < idx < combo.count():
            combo.setCurrentIndex(idx)

    def next_y(self):
        self._move_combo(self.y, 1)

    def prev_y(self):
        self._move_combo(self.y, -1)

    def next_y2(self):
        self._move_combo(self.y2, 1)

    def prev_y2(self):
        self._move_combo(self.y2, -1)

    def selected_x(self):
        if self._updating:
            return
        self.xd.set_specs(dimension_specs(self, self.x.currentText(), "x"))
        self.xlim.setText("None")
        self.redraw()

    def selected_y(self):
        if self._updating:
            return
        self.yd.set_specs(dimension_specs(self, self.y.currentText(), "y"))
        self.ylim.setText("None")
        self.redraw()

    def selected_y2(self):
        if self._updating:
            return
        self.y2d.set_specs(dimension_specs(self, self.y2.currentText(), "y2"))
        self.y2lim.setText("None")
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
        y = self.y.currentText()
        if not y:
            return
        inv_y = self.inv_y.isChecked()
        ylim = parse_entry(self.ylim.text())
        ylim2 = parse_entry(self.y2lim.text())
        ls = self.ls.text()
        lw = float(self.lw.text())
        color = _maybe_color(self.lc.text())
        marker = self.marker.text()
        ms = float(self.ms.text())
        mfc = _maybe_color(self.mfc.text())
        mec = _maybe_color(self.mec.text())
        mew = float(self.mew.text())
        y2 = self.y2.currentText()
        same_y = self.same_y.isChecked()
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
        y2 = self.y2.currentText()
        if not y2:
            return
        y = self.y.currentText()
        inv_y2 = self.inv_y2.isChecked()
        same_y = self.same_y.isChecked()
        ylim = parse_entry(self.ylim.text())
        ylim2 = parse_entry(self.y2lim.text())
        pargs = {
            "linestyle": self.ls2.text(),
            "linewidth": float(self.lw2.text()),
            "marker": self.marker2.text(),
            "markersize": float(self.ms2.text()),
            "markerfacecolor": _maybe_color(self.mfc2.text()),
            "markeredgecolor": _maybe_color(self.mec2.text()),
            "markeredgewidth": float(self.mew2.text()),
        }
        color = _maybe_color(self.lc2.text())
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
        inv_x = self.inv_x.isChecked()
        xlim = parse_entry(self.xlim.text())
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
        x = self.x.currentText()
        y = self.y.currentText()
        y2 = self.y2.currentText()
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


class ContourPanel(PlotPanel):
    def __init__(self, window, session):
        super().__init__(window, session, "Contour")
        self._build_ui()
        self.reinit()

    def _build_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.add_file_controls(top)
        main.addLayout(top)

        self.figure = Figure(facecolor="white", figsize=(1, 1))
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        main.addWidget(self.canvas, 1)
        main.addWidget(self.toolbar)

        controls = QtWidgets.QVBoxLayout()
        main.addLayout(controls)

        rowz = QtWidgets.QHBoxLayout()
        rowz.addWidget(QtWidgets.QLabel("z"))
        rowz.addWidget(self._nav_button("<", self.prev_z))
        rowz.addWidget(self._nav_button(">", self.next_z))
        self.z = self.make_combo(callback=self.selected_z, tooltip="Choose variable")
        rowz.addWidget(self.z)
        self.trans_z = self.make_check("transpose z", callback=self.checked,
                                       tooltip="Transpose matrix")
        rowz.addWidget(self.trans_z)
        self.zmin = self._entry_with_label(rowz, "zmin", "None", self.entered_z, 110)
        self.zmax = self._entry_with_label(rowz, "zmax", "None", self.entered_z, 110)
        rowz.addStretch(1)
        controls.addLayout(rowz)

        self.zd = DimensionControlRow(self.maxdim)
        self.zd.changed.connect(self.spinned_z)
        controls.addWidget(self.zd)

        rowxy = QtWidgets.QHBoxLayout()
        rowxy.addWidget(QtWidgets.QLabel("x"))
        self.x = self.make_combo(callback=self.selected_x,
                                 tooltip='Choose variable of x-axis. Empty uses index.')
        rowxy.addWidget(self.x)
        self.inv_x = self.make_check("invert x", callback=self.checked,
                                     tooltip="Invert x-axis")
        rowxy.addWidget(self.inv_x)
        rowxy.addSpacing(16)
        rowxy.addWidget(QtWidgets.QLabel("y"))
        self.y = self.make_combo(callback=self.selected_y,
                                 tooltip='Choose variable of y-axis. Empty uses index.')
        rowxy.addWidget(self.y)
        self.inv_y = self.make_check("invert y", callback=self.checked,
                                     tooltip="Invert y-axis")
        rowxy.addWidget(self.inv_y)
        rowxy.addStretch(1)
        controls.addLayout(rowxy)

        dimrow = QtWidgets.QHBoxLayout()
        self.xd = DimensionControlRow(self.maxdim)
        self.xd.changed.connect(self.spinned_x)
        self.yd = DimensionControlRow(self.maxdim)
        self.yd.changed.connect(self.spinned_y)
        dimrow.addWidget(self.xd)
        dimrow.addSpacing(16)
        dimrow.addWidget(self.yd)
        controls.addLayout(dimrow)

        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(QtWidgets.QLabel("cmap"))
        self.cmap = self.make_cmap_combo(callback=self.selected_cmap)
        opts.addWidget(self.cmap)
        self.rev_cmap = self.make_check("reverse cmap", callback=self.checked)
        self.mesh = self.make_check("mesh", checked=True, callback=self.checked)
        self.grid = self.make_check("grid", callback=self.checked)
        opts.addWidget(self.rev_cmap)
        opts.addWidget(self.mesh)
        opts.addWidget(self.grid)
        opts.addStretch(1)
        quit_button = QtWidgets.QPushButton("Quit")
        quit_button.clicked.connect(QtWidgets.QApplication.quit)
        opts.addWidget(quit_button)
        controls.addLayout(opts)

    def _entry_with_label(self, layout, label, text, callback, width):
        layout.addWidget(QtWidgets.QLabel(label))
        entry = self.make_entry(text, callback=callback, width=width)
        layout.addWidget(entry)
        return entry

    def _nav_button(self, label, callback):
        button = QtWidgets.QPushButton(label)
        button.setFixedWidth(34)
        button.clicked.connect(callback)
        return button

    def reinit(self):
        super().reinit()
        self._updating = True
        columns = self.columns()
        for combo in (self.z, self.x, self.y):
            _set_combo_items(combo, columns, "")
        for dims in (self.zd, self.xd, self.yd):
            dims.set_specs(empty_dimension_specs(self.maxdim))
        self.zmin.setText("None")
        self.zmax.setText("None")
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
        idx = self.z.currentIndex() + step
        if 0 < idx < self.z.count():
            self.z.setCurrentIndex(idx)

    def next_z(self):
        self._move_z(1)

    def prev_z(self):
        self._move_z(-1)

    def selected_x(self):
        if self._updating:
            return
        self.inv_x.setChecked(False)
        self.xd.set_specs(dimension_specs(self, self.x.currentText(), "x"))
        self.redraw()

    def selected_y(self):
        if self._updating:
            return
        self.inv_y.setChecked(False)
        self.yd.set_specs(dimension_specs(self, self.y.currentText(), "y"))
        self.redraw()

    def selected_z(self):
        if self._updating:
            return
        self.x.setCurrentText("")
        self.y.setCurrentText("")
        self.inv_x.setChecked(False)
        self.inv_y.setChecked(False)
        self.zmin.setText("None")
        self.zmax.setText("None")
        self.xd.set_specs(empty_dimension_specs(self.maxdim))
        self.yd.set_specs(empty_dimension_specs(self.maxdim))
        self.zd.set_specs(dimension_specs(self, self.z.currentText(), "z"))
        self.redraw()

    def redraw(self):
        z = self.z.currentText()
        trans_z = self.trans_z.isChecked()
        zmin = _float_or_none(self.zmin.text())
        zmax = _float_or_none(self.zmax.text())
        x = self.x.currentText()
        y = self.y.currentText()
        inv_x = self.inv_x.isChecked()
        inv_y = self.inv_y.isChecked()
        cmap = self.cmap.currentText()
        if self.rev_cmap.isChecked():
            cmap += "_r"
        mesh = self.mesh.isChecked()
        grid = self.grid.isChecked()
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


class MapUnavailablePanel(QtWidgets.QWidget):
    def __init__(self, error=None):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        message = "Map view is unavailable because Cartopy could not be imported."
        if error is not None:
            message += f"\n\n{type(error).__name__}: {error}"
        message += f"\n\nPython executable: {sys.executable}"
        label = QtWidgets.QLabel(message)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)

    def reinit(self):
        pass

    def redraw(self):
        pass


class MapPanel(PlotPanel):
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
        main = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.add_file_controls(top)
        top.addWidget(QtWidgets.QLabel("Time:"))
        self.timelbl = QtWidgets.QLabel("")
        top.addWidget(self.timelbl)
        main.addLayout(top)

        self.figure = Figure(facecolor="white", figsize=(1, 1))
        self.axes = self.figure.add_subplot(111, projection=ccrs.PlateCarree())
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        main.addWidget(self.canvas, 1)
        main.addWidget(self.toolbar)

        controls = QtWidgets.QVBoxLayout()
        main.addLayout(controls)

        rowt = QtWidgets.QHBoxLayout()
        rowt.addWidget(QtWidgets.QLabel("step"))
        self.tstep = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.tstep.setRange(0, 1)
        self.tstep.valueChanged.connect(self.tstep_t)
        rowt.addWidget(self.tstep)
        self.first_time = self._button("|<<", self.first_t)
        self.prev_time = self._button("|<", self.prev_t)
        self.prun_time = self._button("<", self.prun_t)
        self.nrun_time = self._button(">", self.nrun_t)
        self.next_time = self._button(">|", self.next_t)
        self.last_time = self._button(">>|", self.last_t)
        for button in (self.first_time, self.prev_time, self.prun_time,
                       self.nrun_time, self.next_time, self.last_time):
            rowt.addWidget(button)
        rowt.addWidget(QtWidgets.QLabel("repeat"))
        self.repeat = self.make_combo(["once", "repeat", "reflect"], self.repeat_t)
        self.repeat.setCurrentText("repeat")
        rowt.addWidget(self.repeat)
        controls.addLayout(rowt)

        rowv = QtWidgets.QHBoxLayout()
        rowv.addWidget(QtWidgets.QLabel("var"))
        rowv.addWidget(self._button("<", self.prev_v))
        rowv.addWidget(self._button(">", self.next_v))
        self.v = self.make_combo(callback=self.selected_v, tooltip="Choose variable")
        rowv.addWidget(self.v)
        self.trans_v = self.make_check("transpose var", callback=self.checked)
        rowv.addWidget(self.trans_v)
        self.vmin = self._entry_with_label(rowv, "vmin", "0", self.entered_v, 110)
        self.vmax = self._entry_with_label(rowv, "vmax", "1", self.entered_v, 110)
        self.vall = self.make_check("all", callback=self.checked_all)
        rowv.addWidget(self.vall)
        rowv.addStretch(1)
        controls.addLayout(rowv)

        self.vd = DimensionControlRow(self.maxdim)
        self.vd.changed.connect(self.spinned_v)
        controls.addWidget(self.vd)

        rowll = QtWidgets.QHBoxLayout()
        rowll.addWidget(QtWidgets.QLabel("lon"))
        self.lon = self.make_combo(callback=self.selected_lon)
        rowll.addWidget(self.lon)
        self.inv_lon = self.make_check("invert lon", callback=self.checked)
        self.shift_lon = self.make_check("shift lon/2", callback=self.checked)
        rowll.addWidget(self.inv_lon)
        rowll.addWidget(self.shift_lon)
        rowll.addSpacing(16)
        rowll.addWidget(QtWidgets.QLabel("lat"))
        self.lat = self.make_combo(callback=self.selected_lat)
        rowll.addWidget(self.lat)
        self.inv_lat = self.make_check("invert lat", callback=self.checked)
        rowll.addWidget(self.inv_lat)
        rowll.addStretch(1)
        controls.addLayout(rowll)

        dimrow = QtWidgets.QHBoxLayout()
        self.lond = DimensionControlRow(self.maxdim)
        self.latd = DimensionControlRow(self.maxdim)
        self.lond.changed.connect(self.spinned_lon)
        self.latd.changed.connect(self.spinned_lat)
        dimrow.addWidget(self.lond)
        dimrow.addSpacing(16)
        dimrow.addWidget(self.latd)
        controls.addLayout(dimrow)

        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(QtWidgets.QLabel("cmap"))
        self.cmap = self.make_cmap_combo(callback=self.selected_cmap)
        opts.addWidget(self.cmap)
        self.rev_cmap = self.make_check("reverse cmap", callback=self.checked)
        self.mesh = self.make_check("mesh", checked=True, callback=self.checked)
        self.iglobal = self.make_check("global", callback=self.checked)
        self.coast = self.make_check("coast", checked=True, callback=self.checked)
        self.borders = self.make_check("borders", callback=self.checked)
        self.rivers = self.make_check("rivers", callback=self.checked)
        self.lakes = self.make_check("lakes", callback=self.checked)
        self.grid = self.make_check("grid", callback=self.checked)
        for widget in (self.rev_cmap, self.mesh, self.iglobal, self.coast,
                       self.borders, self.rivers, self.lakes, self.grid):
            opts.addWidget(widget)
        opts.addStretch(1)
        controls.addLayout(opts)

        projrow = QtWidgets.QHBoxLayout()
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
        projrow.addWidget(QtWidgets.QLabel("projection"))
        self.proj = self.make_combo(self.projs, self.selected_proj)
        self.proj.setCurrentText("PlateCarree")
        projrow.addWidget(self.proj)
        self.clon = self._entry_with_label(projrow, "central lon", "None",
                                           self.entered_clon, 80)
        projrow.addStretch(1)
        quit_button = QtWidgets.QPushButton("Quit")
        quit_button.clicked.connect(QtWidgets.QApplication.quit)
        projrow.addWidget(quit_button)
        controls.addLayout(projrow)

    def _button(self, text, callback):
        button = QtWidgets.QPushButton(text)
        button.setFixedWidth(38)
        button.clicked.connect(callback)
        return button

    def _entry_with_label(self, layout, label, text, callback, width):
        layout.addWidget(QtWidgets.QLabel(label))
        entry = self.make_entry(text, callback=callback, width=width)
        layout.addWidget(entry)
        return entry

    def reinit(self):
        super().reinit()
        self._updating = True
        self.iunlim = -1
        self.nunlim = 0
        columns = self.columns()
        for combo in (self.v, self.lon, self.lat):
            _set_combo_items(combo, columns, "")
        for dims in (self.vd, self.lond, self.latd):
            dims.set_specs(empty_dimension_specs(self.maxdim))
        self.vmin.setText("None")
        self.vmax.setText("None")
        self.tstep.setRange(0, 1)
        self.tstep.setValue(0)
        self.repeat.setCurrentText("repeat")
        if self.usex:
            if self.lonvar:
                self.lon.setCurrentText(self.lonvar)
                self.lond.set_specs(dimension_specs(self, self.lon.currentText(), "lon"))
            if self.latvar:
                self.lat.setCurrentText(self.latvar)
                self.latd.set_specs(dimension_specs(self, self.lat.currentText(), "lat"))
        else:
            if any(self.lonvar):
                lon = next(item for item in self.lonvar if item)
                self.lon.setCurrentText(lon)
                self.lond.set_specs(dimension_specs(self, self.lon.currentText(), "lon"))
            if any(self.latvar):
                lat = next(item for item in self.latvar if item)
                self.lat.setCurrentText(lat)
                self.latd.set_specs(dimension_specs(self, self.lat.currentText(), "lat"))
        self._updating = False

    def checked(self):
        if not self._updating:
            self.redraw()

    def checked_all(self):
        if self._updating:
            return
        vmin, vmax = self.get_vminmax()
        self.vmin.setText(str(vmin))
        self.vmax.setText(str(vmax))
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
            self.nrun_time.setText(">")
        else:
            self.anim_inc = 1
            self.prun_time.setText("<")
            self.nrun_time.setText("||")
            self.timer.start()

    def prun_t(self):
        if self.timer.isActive():
            self.timer.stop()
            self.prun_time.setText("<")
        else:
            self.anim_inc = -1
            self.nrun_time.setText(">")
            self.prun_time.setText("||")
            self.timer.start()

    def next_t(self):
        it = self._current_time_index()
        if it < self.nunlim - 1:
            it += 1
        elif self.repeat.currentText() == "repeat":
            it = 0
        elif self.repeat.currentText() == "reflect" and it > 0:
            it -= 1
        self.set_tstep(it)
        self.update_frame(True)

    def prev_t(self):
        it = self._current_time_index()
        if it > 0:
            it -= 1
        elif self.repeat.currentText() == "repeat":
            it = max(self.nunlim - 1, 0)
        elif self.repeat.currentText() == "reflect" and self.nunlim > 1:
            it += 1
        self.set_tstep(it)
        self.update_frame(True)

    def _move_v(self, step):
        idx = self.v.currentIndex() + step
        if 0 < idx < self.v.count():
            self.v.setCurrentIndex(idx)

    def next_v(self):
        self._move_v(1)

    def prev_v(self):
        self._move_v(-1)

    def selected_lat(self):
        if self._updating:
            return
        self.inv_lat.setChecked(False)
        self.latd.set_specs(dimension_specs(self, self.lat.currentText(), "lat"))
        self.redraw()

    def selected_lon(self):
        if self._updating:
            return
        self.inv_lon.setChecked(False)
        self.shift_lon.setChecked(False)
        self.lond.set_specs(dimension_specs(self, self.lon.currentText(), "lon"))
        self.redraw()

    def selected_v(self):
        if self._updating:
            return
        v = self.v.currentText()
        if not v:
            self.redraw()
            return
        self.set_unlim(v)
        self.tstep.setRange(0, max(self.nunlim - 1, 0))
        self.set_tstep(0)
        vmin, vmax = self.get_vminmax()
        self.vmin.setText(str(vmin))
        self.vmax.setText(str(vmax))
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
        v = self.v.currentText()
        if not v:
            return 0, 1
        gz, vz = vardim2var(v, self.groups)
        tname = self.tname if self.usex else self.tname[gz]
        if vz == tname:
            return 0, 1
        vv = selvar(self, vz)
        imiss = get_miss(self, vv)
        if self.vall.isChecked() or (np.sum(vv.shape[:-2]) < 50):
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
        v = self.v.currentText()
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
            self.tstep.blockSignals(True)
            self.tstep.setValue(int(it))
            self.tstep.blockSignals(False)
            if self.usex:
                self.timelbl.setText(str(time.values[it]))
            else:
                try:
                    self.timelbl.setText(str(np.around(time[it], 4)))
                except TypeError:
                    self.timelbl.setText(str(time[it]))

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
        self.nrun_time.setText(">")
        self.prun_time.setText("<")
        v = self.v.currentText()
        trans_v = self.trans_v.isChecked()
        vmin = _float_or_none(self.vmin.text())
        vmax = _float_or_none(self.vmax.text())
        x = self.lon.currentText()
        y = self.lat.currentText()
        inv_lon = self.inv_lon.isChecked()
        inv_lat = self.inv_lat.isChecked()
        shift_lon = self.shift_lon.isChecked()
        cmap = self.cmap.currentText()
        if self.rev_cmap.isChecked():
            cmap += "_r"
        mesh = self.mesh.isChecked()
        self.iiglobal = self.iglobal.isChecked()
        coast = self.coast.isChecked()
        borders = self.borders.isChecked()
        rivers = self.rivers.isChecked()
        lakes = self.lakes.isChecked()
        grid = self.grid.isChecked()
        proj_name = self.proj.currentText()
        self.iproj = self.iprojs[self.projs.index(proj_name)]
        clon = self.clon.text()
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
        v = self.v.currentText()
        if not v:
            return
        trans_v = self.trans_v.isChecked()
        mesh = self.mesh.isChecked()
        rep = self.repeat.currentText()
        shift_lon = self.shift_lon.isChecked()
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
                    self.nrun_time.setText(">")
            elif self.anim_inc == -1 and it == 0:
                if rep == "repeat":
                    it = self.nunlim - 1
                elif rep == "reflect":
                    self.anim_inc = 1
                    it += self.anim_inc
                else:
                    self.timer.stop()
                    self.prun_time.setText("<")
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


class NcvMainWindow(QtWidgets.QMainWindow):
    instances = []

    def __init__(self, session: NcvSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._children = []
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.scatter = ScatterPanel(self, session)
        self.contour = ContourPanel(self, session)
        self.tabs.addTab(self.scatter, "Scatter/Line")
        self.tabs.addTab(self.contour, "Contour")
        if ensure_cartopy():
            self.map = MapPanel(self, session)
        else:
            self.map = MapUnavailablePanel(CARTOPY_IMPORT_ERROR)
        self.tabs.addTab(self.map, "Map")
        self.tabs.currentChanged.connect(self._tab_changed)
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
            self.tabs.setCurrentIndex(2)

    def _tab_changed(self, _index):
        panel = self.tabs.currentWidget()
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
        current = self.tabs.currentWidget()
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
