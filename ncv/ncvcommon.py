"""Shared Qt helpers for ncv panels."""
from __future__ import annotations

import copy
import os
import warnings
from pathlib import Path

import numpy as np

from .dimensions import dimension_values, resolve_selected_variable
from .ncvmethods import get_miss
from .ncvutils import (
    DIMMETHODS,
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
from .session import NcvSession


__all__ = [
    "DimensionControlRow",
    "ScrollableView",
    "display_axes",
    "PlotPanel",
    "TimeControlMixin",
    "color_levels",
    "cursor_label",
    "float_or_none",
    "load_ui",
    "memory_budget_cells",
    "native_levels",
    "set_no_data_colors",
    "parse_limits",
    "resource_path",
    "set_combo_items",
    "to_plot_values",
]


def memory_budget_cells(fraction=0.2, bytes_per_cell=32):
    """Cells that may be held at once: a fraction of currently available RAM.

    32 bytes per cell covers the float64 read, the clamp copy and pyqtgraph's
    RGBA image.  Linux reports MemAvailable, which counts reclaimable cache;
    other POSIX systems fall back to free pages; anything else to 1 GB.
    """
    available = None
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if available is None:
        try:
            available = (os.sysconf("SC_AVPHYS_PAGES")
                         * os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, ValueError, OSError):
            available = 1 << 30
    return max(1, int(available * fraction / bytes_per_cell))


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
    return np.asarray(array, dtype=float), False   # no copy if already float


def color_levels(data, low=None, high=None):
    """Colour-scale limits: the given ones, else the data's range.

    nanmin/nanmax rather than ``data[np.isfinite(data)]``: boolean indexing a
    chunk of tens of millions of cells builds an index array first, which cost
    0.8 s per redraw.  The data is only scanned when a limit is missing.
    """
    if low is None or high is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN chunk
            lo, hi = np.nanmin(data), np.nanmax(data)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            lo, hi = 0.0, 1.0
        low = lo if low is None else low
        high = hi if high is None else high
    return low, high


def native_levels(data, data_range, low=None, high=None):
    """Colour range for an integer chunk whose masked cells hold the dtype min.

    ``data_range`` is the valid (unmasked) min/max.  Valid values below a
    user-given ``low`` are raised to it in place, so that only the sentinel
    falls under the range and lands on the transparent colour-table entry.
    """
    low = data_range[0] if low is None else low
    high = data_range[1] if high is None else high
    if low > data_range[0]:
        sentinel = np.iinfo(data.dtype).min
        floor = int(np.ceil(low))
        np.copyto(data, floor, where=(data < floor) & (data != sentinel))
    return low, high


def set_no_data_colors(image, cmap, low, high, entries=256):
    """Colour an integer image without NaN: entry 0 of the colour table is
    transparent and the levels start one step below ``low``, so the real
    minimum maps to entry 1 and the dtype-minimum sentinel clips to entry 0.
    """
    lut = np.vstack(([0, 0, 0, 0],
                     cmap.getLookupTable(nPts=entries - 1, alpha=True)))
    step = (high - low) / float(entries - 2) if high > low else 1.0
    image.setLookupTable(lut.astype(np.ubyte))
    image.setLevels((low - step, high))


def cursor_label(plot_widget, label, formatter):
    """Feed the form's ``label`` with the value under the mouse."""
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

    windowChanged = QtCore.pyqtSignal()   # debounced: one per pause in a drag
    moved = QtCore.pyqtSignal()           # immediate: for cheap in-memory pans

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
        self._settle = QtCore.QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(40)
        self._settle.timeout.connect(self.windowChanged)
        for bar in (self.vbar, self.hbar):
            bar.valueChanged.connect(self._value_changed)
        self.set_extent(0, 0, 0, 0)

    def _value_changed(self, _value):
        self.moved.emit()
        self._settle.start()

    def set_extent(self, nrows, ncols, row_span, col_span):
        """Full data size and the span shown, keeping the current position."""
        self.show_view(self.vbar.value(), self.hbar.value(),
                       row_span, col_span, nrows, ncols)

    def show_view(self, row, col, row_span, col_span, nrows, ncols):
        """Put the thumbs on the displayed region (in cells) without emitting.

        Position and length of each thumb are the shown region's position and
        size within the whole variable.
        """
        for bar, start, span, total in ((self.vbar, row, row_span, nrows),
                                        (self.hbar, col, col_span, ncols)):
            total = int(total)
            span = int(min(max(1, round(span)), max(total, 1)))
            blocked = bar.blockSignals(True)
            bar.setRange(0, max(0, total - span))
            bar.setPageStep(span)
            bar.setSingleStep(max(1, span // 10))
            bar.setValue(int(min(max(0, round(start)), max(0, total - span))))
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
        self._zwindow = None
        self._chunk_notified = None
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
                   window=None, native=False):
        """Read the selection with missing values removed.

        Normally returns a float array with NaN where data is missing.  With
        ``native`` an unscaled signed-integer chunk is kept in its own dtype
        instead - no float copy - and ``(data, (min, max))`` is returned, with
        missing cells set to the dtype minimum (see ``set_no_data_colors``).
        """
        miss = get_miss(self, variable)
        values = dim_controls.values()
        values.extend(["0"] * max(0, variable.ndim - len(values)))
        out = get_slice_values(values, variable, window=window)
        if out.ndim > 1:
            out = out.squeeze()
        if native and np.issubdtype(out.dtype, np.signedinteger) and out.ndim == 2:
            kept = self._native_chunk(out)
            if kept is not None:
                return kept
        if (isinstance(out, np.ma.MaskedArray)
                and np.issubdtype(out.dtype, np.number)):
            # netCDF4 already masked _FillValue / missing_value / the default
            # fill: turn that mask into NaN once, and scan only for a user -m
            # value instead of re-finding all of them in the whole chunk
            mask = np.ma.getmaskarray(out)
            out = np.asarray(out.data, dtype=float)
            np.copyto(out, np.nan, where=mask)
            miss = [self.miss]
        out = set_miss(miss, out)
        try:
            _ = out.shape[0]
        except IndexError:
            out = np.array([np.nan])
        return out

    def _native_chunk(self, out):
        """``(data, (min, max))`` with missing cells at the dtype minimum, or
        None when that value could collide with real data."""
        data = np.ma.getdata(out)
        mask = np.ma.getmaskarray(out)
        try:
            if not np.isnan(self.miss):
                mask = mask | (data == self.miss)
        except TypeError:
            pass
        info = np.iinfo(data.dtype)
        sentinel = info.min
        # reductions with where= instead of np.ma.min/max: those fill (copy)
        # the whole chunk first, twice - 0.45 s on a 40M-cell chunk
        valid = ~mask
        if not valid.any():
            low, high = 0, 1               # nothing valid in this chunk
        else:
            low = data.min(where=valid, initial=info.max)
            high = data.max(where=valid, initial=info.min)
            if low == sentinel:
                return None                # the sentinel is a real value here
        if not data.flags.writeable:
            data = data.copy()
        np.copyto(data, sentinel, where=mask)
        return data, (low, high)

    def _series(self, vardim, dim_controls, window=None, native=False):
        """Return ``(values, label, is_datetime, native_range)``.

        ``native_range`` is the valid min/max when ``native`` kept an integer
        chunk in its own dtype, else None and ``values`` is float.
        """
        group, name = vardim2var(vardim, self.groups)
        tname = self.tname if self.usex else self.tname[group]
        if name == tname:
            values, label = self.time_values(group), "Date"
            native = False
        else:
            values = selvar(self, name)
            label = set_axis_label(values)
        out = self.slice_miss(dim_controls, values, window=window, native=native)
        if isinstance(out, tuple):
            return out[0], label, False, out[1]
        values, is_date = to_plot_values(out)
        return values, label, is_date, None

    def _set_axis(self, axis, is_date):
        """Swap an axis between plain and date ticks when the dtype changes."""
        if isinstance(self.item.getAxis(axis), pg.DateAxisItem) == is_date:
            return
        self.item.setAxisItems({axis: (
            pg.DateAxisItem(orientation=axis) if is_date
            else pg.AxisItem(orientation=axis))})

    def read_window(self, variable, dim_values, offsets=(0, 0), coarse=False,
                    center=None):
        """Return ``(window, full_shape, span, mode)`` for the displayed axes.

        ``window`` feeds ``get_slice_values`` so only that region is read.
        ``mode`` is ``coarse`` (whole extent, strided - only on request),
        ``full`` (the whole slice fits in memory) or ``chunk`` (a
        full-resolution block at ``offsets`` - or centred on ``center`` -
        sized from available memory).
        """
        axes = display_axes(dim_values)
        if len(axes) != 2:
            return None, None, None, None
        shape = tuple(int(variable.shape[a]) for a in axes)
        budget = memory_budget_cells()
        # axes reduced by mean/std/... are read in full before reducing
        reduced = [int(variable.shape[i]) for i, v in enumerate(dim_values)
                   if str(v) in DIMMETHODS]
        budget = max(1, budget // max(1, int(np.prod(reduced))))
        if coarse:
            stride = overview_stride(shape, chunk_shape(variable),
                                     max_cells=budget)
            window = {a: (0, shape[i], stride[i]) for i, a in enumerate(axes)}
            return window, shape, shape, "coarse"
        if shape[0] * shape[1] <= budget:
            window = {a: (0, shape[i], 1) for i, a in enumerate(axes)}
            return window, shape, shape, "full"
        side = max(1, int(np.sqrt(budget)))
        span = tuple(min(side, n) for n in shape)
        if center is not None:
            offsets = [int(center[i]) - span[i] // 2 for i in range(2)]
        window = {}
        for i, axis in enumerate(axes):
            start = min(max(int(offsets[i]), 0), shape[i] - span[i])
            window[axis] = (start, start + span[i], 1)
        return window, shape, span, "chunk"

    def scroll_window(self, vardim, dim_controls, offsets=None, center=None):
        """Window to read for ``vardim``; alerts once when it only fits in chunks.

        ``center`` (variable axis order) centres the chunk on a cell; otherwise
        it starts at ``offsets``.  Stores ``self._zwindow`` (the two displayed
        axes' windows) and ``self._full_shape`` for the panel's scroll bars.
        """
        self._zwindow = self._full_shape = None
        try:
            _group, _name, variable = resolve_selected_variable(self, vardim)
        except Exception:
            variable = None
        if getattr(variable, "ndim", 0) < 2:
            self.scroll.set_extent(0, 0, 0, 0)
            return None
        values = dim_controls.values()
        values.extend(["0"] * max(0, variable.ndim - len(values)))
        window, shape, span, mode = self.read_window(
            variable, values, (0, 0) if offsets is None else offsets,
            self.checkBox_fullCoarse.isChecked(), center)
        if window is None:
            self.scroll.set_extent(0, 0, 0, 0)
            return None
        axes = sorted(window)
        self._zwindow = (window[axes[0]], window[axes[1]])
        self._full_shape = shape
        if mode == "chunk" and vardim != self._chunk_notified:
            self._chunk_notified = vardim
            self._alert_chunked(vardim, shape, span)
        return window

    # ---- plots: the scroll bars mirror the view; a read happens only when the
    # view leaves the loaded chunk (Contour, and Map's flat-projection image)

    def init_view_sync(self):
        self._geom = None
        self._syncing = False
        self._reload_center = None
        self._reload = QtCore.QTimer(self)
        self._reload.setSingleShot(True)
        self._reload.setInterval(150)
        self._reload.timeout.connect(self._reload_chunk)
        self.scroll.moved.connect(self._bars_moved)
        self.item.vb.sigRangeChanged.connect(self._view_moved)

    def set_view_geometry(self, rect, transposed):
        """Record how the drawn chunk maps cells to plot coordinates.

        ``rect`` is the QRectF the chunk image was placed in.  Everything is
        kept in display orientation: rows run along the image's y axis.
        """
        if self._zwindow is None:
            self._geom = None
            return
        (r0, r1, _sr), (c0, c1, _sc) = self._zwindow
        full = self._full_shape
        if transposed:
            (r0, r1), (c0, c1) = (c0, c1), (r0, r1)
            full = full[::-1]
        dx = rect.width() / max(1, c1 - c0)
        dy = rect.height() / max(1, r1 - r0)
        self._geom = (full, (r0, r1, c0, c1), rect.x(), rect.y(), dx, dy,
                      transposed)

    def _view_cells(self):
        full, (r0, r1, c0, c1), x0, y0, dx, dy, _t = self._geom
        xr, yr = self.item.vb.viewRange()
        cols = sorted(c0 + (x - x0) / dx for x in xr)
        rows = sorted(r0 + (y - y0) / dy for y in yr)
        return rows, cols

    def initial_view(self):
        """Whole extent when fully loaded; else the chunk's central half, so
        there is room to pan either way before the next read."""
        if self._geom is None:
            return
        full, (r0, r1, c0, c1), x0, y0, dx, dy, _t = self._geom
        whole = r0 == 0 and c0 == 0 and r1 >= full[0] and c1 >= full[1]
        rows = (r1 - r0) * (1.0 if whole else 0.5)
        cols = (c1 - c0) * (1.0 if whole else 0.5)
        vb = self.item.vb
        if not whole and vb.state["aspectLocked"] and vb.height() > 0:
            # an aspect-locked view (Map) widens one axis to the widget's shape;
            # pick spans with that shape so the view still fits in the chunk
            ratio = (vb.width() / vb.height()) * abs(dy) / abs(dx)  # cols/row
            rows = min(rows, cols / ratio)
            cols = rows * ratio
        rc, cc = (r1 - r0) / 2.0, (c1 - c0) / 2.0     # chunk centre, local
        self._syncing = True
        vb.setRange(
            xRange=sorted((x0 + (cc - cols / 2) * dx, x0 + (cc + cols / 2) * dx)),
            yRange=sorted((y0 + (rc - rows / 2) * dy, y0 + (rc + rows / 2) * dy)),
            padding=0)
        self._syncing = False
        self._view_moved()

    def _view_moved(self, *_args):
        """Mouse pan/zoom (or a programmatic range): move the thumbs."""
        if self._syncing or self._geom is None:
            return
        rows, cols = self._view_cells()
        self.scroll.show_view(rows[0], cols[0], rows[1] - rows[0],
                              cols[1] - cols[0], *self._geom[0])
        self._maybe_reload(rows, cols)

    def _bars_moved(self):
        """A thumb moved: pan the view there - no read while inside the chunk."""
        if self._geom is None:
            return
        full, (r0, r1, c0, c1), x0, y0, dx, dy, _t = self._geom
        r, c = self.scroll.offsets()
        rs, cs = self.scroll.vbar.pageStep(), self.scroll.hbar.pageStep()
        self._syncing = True
        self.item.vb.setRange(
            xRange=sorted((x0 + (c - c0) * dx, x0 + (c + cs - c0) * dx)),
            yRange=sorted((y0 + (r - r0) * dy, y0 + (r + rs - r0) * dy)),
            padding=0)
        self._syncing = False
        self._maybe_reload((r, r + rs), (c, c + cs))

    def _maybe_reload(self, rows, cols):
        full, (r0, r1, c0, c1), *_rest = self._geom
        whole = r0 == 0 and c0 == 0 and r1 >= full[0] and c1 >= full[1]
        inside = (r0 <= rows[0] + 0.5 and rows[1] - 0.5 <= r1
                  and c0 <= cols[0] + 0.5 and cols[1] - 0.5 <= c1)
        # a view wider than a chunk (zoomed out, or 'global') can never be
        # inside one: reload only if the chunk centred on it would differ,
        # otherwise the timer would re-read the same chunk forever
        sr, sc = r1 - r0, c1 - c0
        nr0 = min(max(int((rows[0] + rows[1]) / 2) - sr // 2, 0), full[0] - sr)
        nc0 = min(max(int((cols[0] + cols[1]) / 2) - sc // 2, 0), full[1] - sc)
        if whole or inside or (nr0, nc0) == (r0, c0):
            self._reload.stop()
        else:
            self._reload.start()

    def view_center(self):
        """Centre of the view in variable-axis cells, for the next chunk."""
        if self._geom is None:
            return None
        rows, cols = self._view_cells()
        center = ((rows[0] + rows[1]) / 2.0, (cols[0] + cols[1]) / 2.0)
        return center[::-1] if self._geom[-1] else center

    def _reload_chunk(self):
        self._reload_center = self.view_center()
        self.redraw()
        self._reload_center = None

    def coord_window(self, vardim, is_x, transposed=False):
        """The data window, restricted to a 1-D or 2-D coordinate variable.

        A 2-D coordinate shares the data's own axes; a 1-D one follows the
        displayed axis it labels, which swaps when the data is transposed.
        """
        if self._zwindow is None or not vardim:
            return None
        try:
            _group, _name, variable = resolve_selected_variable(self, vardim)
        except Exception:
            return None
        rows_w, cols_w = self._zwindow
        if getattr(variable, "ndim", 0) == 1:
            return {0: cols_w if is_x != transposed else rows_w}
        return {0: rows_w, 1: cols_w}

    def _alert_chunked(self, vardim, shape, span):
        _group, name = vardim2var(vardim, self.groups)
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        box.setWindowTitle("ncv")
        box.setText(f"{name} is too large to load at once.")
        box.setInformativeText(
            f"Full size {shape[0]:,} x {shape[1]:,}. It is loaded in chunks "
            f"of {span[0]:,} x {span[1]:,} at full resolution.\n\n"
            "Use the scroll bars to move through it, or tick "
            "'full (coarse)' to see all of it at reduced resolution.")
        # open(), not exec(): non-blocking, so neither the UI nor the
        # offscreen tests stall on it
        box.open()
        self._chunk_alert = box

    def reinit(self):
        self._chunk_notified = None
        self._copy_session()

    def redraw(self):
        raise NotImplementedError
