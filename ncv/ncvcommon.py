"""Shared Qt helpers for ncv panels."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .dimensions import dimension_values
from .ncvmethods import get_miss
from .ncvutils import (
    chunk_shape,
    get_slice_values,
    overview_stride,
    parse_entry,
    selvar,
    set_axis_label,
    set_miss,
    vardim2var,
)
from .qt_compat import QtCore, QtGui, QtWidgets, pg, uic
from .session import HAVE_XARRAY, NcvSession


__all__ = [
    "MAX_CELLS",
    "DimensionControlRow",
    "ScrollableView",
    "display_axes",
    "PlotPanel",
    "TimeControlMixin",
    "cursor_label",
    "float_or_none",
    "load_ui",
    "parse_limits",
    "resource_path",
    "set_combo_items",
    "to_plot_values",
]


# One read budget for every panel: how many cells may be held at once.
MAX_CELLS = 4000000


def display_axes(dim_values):
    """Axis indices shown as a 2-D image, i.e. the ones set to ``all``."""
    return [i for i, value in enumerate(dim_values) if str(value) == "all"]


def resource_path(*parts: str) -> str:
    return str(Path(__file__).resolve().parent.joinpath(*parts))


def load_ui(name: str, widget) -> None:
    """Load ``ncv/ui/<name>.ui`` onto an existing widget."""
    uic.loadUi(resource_path("ui", f"{name}.ui"), widget)


def to_plot_values(values):
    """Return ``(numeric array, is_datetime)`` that pyqtgraph can plot.

    ``datetime64`` becomes POSIX seconds, which is what ``pg.DateAxisItem``
    expects.  Everything else is coerced to float.
    """
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype("datetime64[s]").astype("float64"), True
    return array.astype(float), False


def cursor_label(plot_widget, layout, formatter):
    """Add a read-out label under ``plot_widget`` fed by mouse position."""
    label = QtWidgets.QLabel("")
    layout.addWidget(label)
    view = plot_widget.plotItem.vb

    def moved(pos):
        if plot_widget.plotItem.sceneBoundingRect().contains(pos):
            point = view.mapSceneToView(pos)
            label.setText(formatter(point.x(), point.y()))

    plot_widget.scene().sigMouseMoved.connect(moved)
    return label


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


def _cmap_icon(cmap, width=64, height=12):
    """Build a gradient swatch icon from a pyqtgraph colormap."""
    lut = cmap.getLookupTable(nPts=width, alpha=False)
    image = QtGui.QImage(width, 1, QtGui.QImage.Format.Format_RGB888)
    for i, (red, green, blue) in enumerate(lut):
        image.setPixel(i, 0, QtGui.qRgb(int(red), int(green), int(blue)))
    return QtGui.QIcon(QtGui.QPixmap.fromImage(image.scaled(width, height)))


def set_combo_items(combo, values, current=None):
    combo.blockSignals(True)
    combo.clear()
    combo.addItems([str(value) for value in values])
    items = [combo.itemText(i) for i in range(combo.count())]
    if current is not None and str(current) in items:
        combo.setCurrentText(str(current))
    combo.blockSignals(False)


class ScrollableView(QtWidgets.QWidget):
    """A view with scroll bars aligned to its own edges.

    The bars sit in a grid beside and below the view, so their length always
    matches the view extent.  Range is the full data size and page step is the
    loaded span, so the thumb shows how much of the variable is in memory.
    """

    windowChanged = QtCore.pyqtSignal()

    def __init__(self, view, parent=None):
        super().__init__(parent)
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        self.view = view
        self.vbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        self.hbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        grid.addWidget(view, 0, 0)
        grid.addWidget(self.vbar, 0, 1)
        grid.addWidget(self.hbar, 1, 0)
        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)
        for bar in (self.vbar, self.hbar):
            bar.valueChanged.connect(lambda _value: self.windowChanged.emit())
        self.set_extent(0, 0, 0, 0)

    def set_extent(self, nrows, ncols, row_span, col_span):
        """Full data size and the span currently loaded from it."""
        for bar, total, span in ((self.vbar, nrows, row_span),
                                 (self.hbar, ncols, col_span)):
            blocked = bar.blockSignals(True)
            bar.setRange(0, max(0, int(total) - int(span)))
            bar.setPageStep(max(1, int(span)))
            bar.setSingleStep(max(1, int(span) // 10))
            bar.setEnabled(total > span)
            bar.blockSignals(blocked)

    def offsets(self):
        """Current (row, column) scroll offsets."""
        return self.vbar.value(), self.hbar.value()


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
            selector.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
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
        for name in sorted(pg.colormap.listMaps()):
            combo.addItem(_cmap_icon(pg.colormap.get(name)), name)
        combo.setCurrentText("viridis")

    def selected_cmap_object(self, combo, reverse_check):
        cmap = pg.colormap.get(combo.currentText() or "viridis")
        if reverse_check.isChecked():
            # pg caches colormaps and reverse() mutates in place
            cmap = copy.deepcopy(cmap)
            cmap.reverse()
        return cmap

    def slice_miss(self, dim_controls: DimensionControlRow, variable,
                   window=None):
        miss = get_miss(self, variable)
        values = dim_controls.values()
        values.extend(["0"] * max(0, variable.ndim - len(values)))
        out = get_slice_values(values, variable, window=window)
        if out.ndim > 1:
            out = out.squeeze()
        out = set_miss(miss, out)
        try:
            _ = out.shape[0]
        except IndexError:
            out = np.array([np.nan])
        return out

    def _series(self, vardim, dim_controls, window=None):
        """Return (values, label, is_datetime) for one combo selection."""
        group, name = vardim2var(vardim, self.groups)
        tname = self.tname if self.usex else self.tname[group]
        if name == tname:
            values, label = self.time_values(group), "Date"
        else:
            values = selvar(self, name)
            label = set_axis_label(values)
        values, is_date = to_plot_values(
            self.slice_miss(dim_controls, values, window=window))
        return values, label, is_date

    def _set_axis(self, axis, is_date):
        """Swap an axis between plain and date ticks when the dtype changes."""
        if isinstance(self.item.getAxis(axis), pg.DateAxisItem) == is_date:
            return
        self.item.setAxisItems({axis: (
            pg.DateAxisItem(orientation=axis) if is_date
            else pg.AxisItem(orientation=axis))})

    def read_window(self, variable, dim_values, offsets=(0, 0), overview=True):
        """Return ``(window, full_shape, span)`` for the two displayed axes.

        ``window`` feeds ``get_slice_values``, so only the chosen region is
        read from disk.  In overview mode the whole extent is read at a stride
        chosen from the variable's chunking; otherwise a native-resolution
        block of at most ``MAX_CELLS`` cells is read at ``offsets``.
        """
        axes = display_axes(dim_values)
        if len(axes) != 2:
            return None, None, None
        shape = tuple(int(variable.shape[a]) for a in axes)
        side = max(1, int(np.sqrt(MAX_CELLS)))
        span = tuple(min(side, shape[i]) for i in range(2))
        if overview:
            stride = overview_stride(shape, chunk_shape(variable),
                                     max_cells=MAX_CELLS)
            window = {a: (0, shape[i], stride[i]) for i, a in enumerate(axes)}
            if stride == (1, 1):
                # the whole variable is loaded at native resolution, so there
                # is nothing further to scroll to
                span = shape
        else:
            window = {}
            for i, axis in enumerate(axes):
                start = min(max(int(offsets[i]), 0), shape[i] - span[i])
                window[axis] = (start, start + span[i], 1)
        return window, shape, span

    def reinit(self):
        self._copy_session()

    def redraw(self):
        raise NotImplementedError
