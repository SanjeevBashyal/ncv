"""Shared Qt helpers for ncv panels."""
from __future__ import annotations

import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from .dimensions import dimension_values
from .ncvmethods import get_miss
from .ncvutils import (
    get_slice_values,
    parse_entry,
    selvar,
    set_miss,
    vardim2var,
)
from .qt_compat import QtCore, QtGui, QtWidgets
from .session import HAVE_XARRAY, NcvSession


__all__ = [
    "DimensionControlRow",
    "PlotPanel",
    "TimeControlMixin",
    "float_or_none",
    "parse_limits",
    "resource_path",
    "set_combo_items",
]


def resource_path(*parts: str) -> str:
    return str(Path(__file__).resolve().parent.joinpath(*parts))


def float_or_none(value: str):
    if value is None or str(value).strip() == "None":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_limits(text):
    """Return a ``(minimum, maximum)`` pair from a limit entry."""
    value = str(text).strip()
    if value == "None":
        return None, None
    if value[:1] in "([" and value[-1:] in ")]":
        value = value[1:-1]
    parts = value.split(",")
    if len(parts) != 2:
        return None, None
    return tuple(parse_entry(part.strip()) for part in parts)


def set_combo_items(combo, values, current=None):
    combo.blockSignals(True)
    combo.clear()
    combo.addItems([str(value) for value in values])
    items = [combo.itemText(i) for i in range(combo.count())]
    if current is not None and str(current) in items:
        combo.setCurrentText(str(current))
    combo.blockSignals(False)


class DimensionControlRow(QtWidgets.QWidget):
    """A row of label and combo-box dimension selectors."""

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
            selector = self.selectors[index]
            blocked = selector.blockSignals(True)
            try:
                selector.setCurrentText(str(value))
            finally:
                selector.blockSignals(blocked)


class TimeControlMixin:
    """Shared time navigation for panels using the standard time widgets."""

    def init_time_controls(self, variable_combo, dimension_row):
        self._time_variable_combo = variable_combo
        self._time_dimension_row = dimension_row
        self.iunlim = -1
        self.nunlim = 0
        self.anim_inc = 1
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(lambda: self.update_frame(False))

        self.horizontalSlider_timeStep.valueChanged.connect(self.tstep_t)
        self.pushButton_firstTime.clicked.connect(self.first_t)
        self.pushButton_prevTime.clicked.connect(self.prev_t)
        self.pushButton_runBackward.clicked.connect(self.prun_t)
        self.pushButton_runForward.clicked.connect(self.nrun_t)
        self.pushButton_nextTime.clicked.connect(self.next_t)
        self.pushButton_lastTime.clicked.connect(self.last_t)

    def first_t(self):
        self._show_frame(0)

    def last_t(self):
        self._show_frame(self.nunlim - 1)

    def nrun_t(self):
        self._toggle_animation(1)

    def prun_t(self):
        self._toggle_animation(-1)

    def next_t(self):
        self._show_frame(self._next_time_index(1)[0])

    def prev_t(self):
        self._show_frame(self._next_time_index(-1)[0])

    def _show_frame(self, index):
        if self.nunlim > 0:
            self.set_tstep(index)
            self.update_frame(True)

    def _stop_animation(self):
        self.timer.stop()
        self.pushButton_runBackward.setText("<")
        self.pushButton_runForward.setText(">")

    def _set_animation_direction(self, direction):
        self.anim_inc = direction
        self.pushButton_runBackward.setText("||" if direction < 0 else "<")
        self.pushButton_runForward.setText("||" if direction > 0 else ">")

    def _toggle_animation(self, direction):
        if self.nunlim <= 1:
            self._stop_animation()
        elif self.timer.isActive() and self.anim_inc == direction:
            self._stop_animation()
        else:
            self._set_animation_direction(direction)
            self.timer.start()

    def _next_time_index(self, direction):
        if self.nunlim <= 1:
            return 0, direction, True
        current = self._current_time_index()
        target = current + direction
        if 0 <= target < self.nunlim:
            return target, direction, False
        repeat = self.comboBox_repeat.currentText()
        if repeat == "repeat":
            return (0 if direction > 0 else self.nunlim - 1), direction, False
        if repeat == "reflect":
            direction *= -1
            return current + direction, direction, False
        return current, direction, True

    def tstep_t(self, step):
        if self._updating:
            return
        self.set_tstep(int(step))
        self.update_frame(True)

    def _current_time_index(self):
        if self.iunlim < 0 or self.nunlim <= 0:
            return 0
        try:
            value = int(self._time_dimension_row.values()[self.iunlim])
        except (ValueError, IndexError):
            return 0
        return min(max(value, 0), self.nunlim - 1)

    def set_tstep(self, it):
        if self.iunlim < 0 or self.nunlim <= 0:
            return
        it = min(max(int(it), 0), self.nunlim - 1)
        self._time_dimension_row.set_value(self.iunlim, it)
        blocked = self.horizontalSlider_timeStep.blockSignals(True)
        self.horizontalSlider_timeStep.setValue(it)
        self.horizontalSlider_timeStep.blockSignals(blocked)
        details = self._time_details(self._time_variable_combo.currentText())
        if details is None:
            return
        _variable, time, _axis, _count = details
        value = time.values[it] if self.usex else time[it]
        if not self.usex:
            try:
                value = np.around(value, 4)
            except TypeError:
                pass
        self.label_timeValue.setText(str(value))

    def set_unlim(self, vardim):
        details = self._time_details(vardim)
        if details is None:
            self.iunlim = -1
            self.nunlim = 0
        else:
            _variable, _time, self.iunlim, self.nunlim = details
        self._sync_time_controls()

    def _time_details(self, vardim):
        if not vardim:
            return None
        group, variable_name = vardim2var(vardim, self.groups)
        if self.usex:
            tname, tvar = self.tname, self.tvar
            time, time_dim = self.time, self.dunlim
        else:
            tname, tvar = self.tname[group], self.tvar[group]
            time, time_dim = self.time[group], self.dunlim[group]
        if time is None or not tvar:
            return None
        if variable_name == tname:
            variable_name = tvar
        try:
            variable = selvar(self, variable_name)
            time_variable = selvar(self, tvar)
            dims = variable.dims if self.usex else variable.dimensions
            time_dims = (time_variable.dims if self.usex
                         else time_variable.dimensions)
        except Exception:
            return None
        if time_dim not in time_dims:
            time_dim = time_dims[0] if time_dims else None
        if time_dim not in dims:
            return None
        axis = dims.index(time_dim)
        count = min(int(variable.shape[axis]), int(time.size))
        if count <= 0:
            return None
        return variable, time, axis, count

    def _sync_time_controls(self):
        self._stop_animation()
        enabled = self.iunlim >= 0 and self.nunlim > 0
        for widget in (
            self.label_step,
            self.horizontalSlider_timeStep,
            self.pushButton_firstTime,
            self.pushButton_prevTime,
            self.pushButton_nextTime,
            self.pushButton_lastTime,
            self.label_repeat,
            self.comboBox_repeat,
        ):
            widget.setEnabled(enabled)
        can_run = enabled and self.nunlim > 1
        self.pushButton_runBackward.setEnabled(can_run)
        self.pushButton_runForward.setEnabled(can_run)
        blocked = self.horizontalSlider_timeStep.blockSignals(True)
        self.horizontalSlider_timeStep.setRange(0, max(self.nunlim - 1, 0))
        self.horizontalSlider_timeStep.setValue(0)
        self.horizontalSlider_timeStep.blockSignals(blocked)
        if not enabled:
            self.label_timeValue.clear()


class PlotPanel(QtWidgets.QWidget):
    """Common behavior for Qt plotting panels."""

    def __init__(self, window, session: NcvSession, name: str):
        super().__init__(window)
        self.window = window
        self.session = session
        self.name = name
        self._updating = False
        self._copy_session()

    def _copy_session(self):
        for name in (
            "usex", "fi", "groups", "miss", "dunlim", "time", "tname",
            "tvar", "dtime", "latvar", "lonvar", "latdim", "londim",
            "maxdim", "cols",
        ):
            setattr(self, name, getattr(self.session, name))

    def columns(self):
        return [""] + list(self.session.cols)

    def time_values(self, group, decimal=False):
        values = self.dtime if decimal else self.time
        return values if self.usex else values[group]

    def connect_file_controls(self):
        self.pushButton_openFile.clicked.connect(
            lambda: self.window.open_file_dialog(False))
        self.pushButton_openXarray.setVisible(HAVE_XARRAY)
        if HAVE_XARRAY:
            self.pushButton_openXarray.clicked.connect(
                lambda: self.window.open_file_dialog(True))
        self.pushButton_newWindow.clicked.connect(
            self.window.create_secondary_window)

    def populate_cmap_combo(self, combo):
        combo.clear()
        colormaps = sorted(c for c in plt.colormaps() if not c.endswith("_r"))
        for cmap in colormaps:
            icon_path = resource_path("images", f"{cmap}.png")
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
