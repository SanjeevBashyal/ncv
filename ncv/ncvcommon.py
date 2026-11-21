"""Shared Qt helpers for ncv panels."""
from __future__ import annotations

import copy
import os
import threading
import warnings
from pathlib import Path

import numpy as np

from .dimensions import dimension_values, resolve_selected_variable
from .ncvmethods import get_miss
from .ncvutils import (
    DIMMETHODS,
    RESOLUTIONS,
    STATISTICS,
    aggregate_series,
    chunk_shape,
    coarser_resolutions,
    get_slice_values,
    overview_stride,
    parse_entry,
    period_label,
    period_starts,
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
    "coordinate_variable",
    "cursor_label",
    "read_axis",
    "float_or_none",
    "load_ui",
    "CHUNK_MAX_CELLS",
    "cgroup_available",
    "memory_budget_cells",
    "native_levels",
    "set_palette_image",
    "parse_limits",
    "read_responsive",
    "reading",
    "resource_path",
    "set_combo_items",
    "to_plot_values",
]


_READ_LOCK = threading.Lock()   # netCDF-C is not thread-safe: one read at a time
_READING = 0


def reading():
    """True while ``read_responsive`` waits for a read: timers that would
    start another redraw re-arm themselves instead."""
    return _READING > 0


def read_responsive(fn):
    """Run a blocking read on a worker thread while the event loop keeps
    painting and answering the window manager (netCDF4 releases the GIL), so
    a slow disk never turns the window 'Not Responding'.  User input waits
    until the read ends; the caller still gets the result synchronously.
    """
    global _READING
    app = QtWidgets.QApplication.instance()
    if app is None or QtCore.QThread.currentThread() is not app.thread():
        return fn()
    box, loop = {}, QtCore.QEventLoop()

    def work():
        try:
            with _READ_LOCK:
                box["out"] = fn()
        except BaseException as exc:
            box["exc"] = exc
        QtCore.QMetaObject.invokeMethod(
            loop, "quit", QtCore.Qt.ConnectionType.QueuedConnection)

    _READING += 1
    try:
        threading.Thread(target=work, daemon=True).start()
        loop.exec(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    finally:
        _READING -= 1
    if "exc" in box:
        raise box["exc"]
    return box["out"]


# Chunk size that keeps a switch under ~0.5 s (about 9 ns per cell measured on
# Contour and Matrix). Beyond this, more memory only buys slower switches.
CHUNK_MAX_CELLS = 36_000_000


def cgroup_available(root="/sys/fs/cgroup", proc="/proc/self/cgroup"):
    """Bytes left under this process's tightest cgroup v2 ``memory.max``.

    HPC login nodes and batch jobs cap each user well below the node's RAM,
    which is all ``/proc/meminfo`` reports.  None when unlimited or unknown.
    """
    try:
        with open(proc) as cgroups:
            path = next(line.split("::", 1)[1].strip()
                        for line in cgroups if line.startswith("0::"))
    except (OSError, StopIteration):
        return None
    left = None
    folder = os.path.join(root, path.lstrip("/"))
    while True:
        try:
            with open(os.path.join(folder, "memory.max")) as limit:
                cap = limit.read().strip()
            if cap != "max":
                with open(os.path.join(folder, "memory.current")) as used:
                    room = int(cap) - int(used.read())
                left = room if left is None else min(left, room)
        except (OSError, ValueError):
            pass
        if os.path.abspath(folder) == os.path.abspath(root):
            return None if left is None else max(0, left)
        folder = os.path.dirname(folder)


def memory_budget_cells(fraction=0.2, bytes_per_cell=32, cap=CHUNK_MAX_CELLS):
    """Cells that may be held at once: a fraction of currently available RAM,
    capped at ``cap`` (switch time for an image; None for no cap).

    32 bytes per cell covers the float64 read, the clamp copy and pyqtgraph's
    RGBA image.  Linux reports MemAvailable, which counts reclaimable cache;
    other POSIX systems fall back to free pages; anything else to 1 GB.  A
    cgroup memory limit (HPC login node or job) lowers it further.
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
    limit = cgroup_available()
    if limit is not None:
        available = min(available, limit)
    cells = int(available * fraction / bytes_per_cell)
    return max(1, cells if cap is None else min(cap, cells))


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


BANDS = 10      # colour bands of the "Banded" scale
EQ_BINS = 65536  # value bins of the "Equalised" scale's lookup table


def color_scale(scale, low, high, data=None):
    """``f(x)`` mapping float32 values (in place) to colour positions 0..1
    for a ``comboBox_cscale`` choice, or None for Linear (the fast path).

    Log: log10 between ``low`` and ``high``.  Low / High: ncview's stretch of
    the low / high end.  Banded: ``BANDS`` flat bands, like filled contours.
    Equalised: positions follow the quantiles of a sample of ``data``.
    """
    if scale not in ("Log", "Low", "High", "Banded", "Equalised") or \
            not high > low:
        return None
    if scale == "Equalised":
        sample = np.asarray(data, dtype=np.float32).ravel()
        sample = sample[::max(1, sample.size // 100_000)]
        sample = sample[(sample >= low) & (sample <= high)]  # no NaN, sentinel
        if sample.size == 0:
            return None
        # histogram equalisation: a CDF table over EQ_BINS linear bins, looked
        # up per cell (~0.2 s on 36M cells; np.interp on quantiles took 2.5 s)
        counts = np.histogram(sample, bins=EQ_BINS, range=(low, high))[0]
        table = ((np.cumsum(counts) - counts / 2) / sample.size).astype(np.float32)

        def f(x):
            nan = np.isnan(x)
            x[nan] = low                    # restored below; no NaN cast
            x -= low
            x *= EQ_BINS / (high - low)
            np.clip(x, 0, EQ_BINS - 1, out=x)
            x[...] = np.take(table, x.astype(np.int32), mode="clip")
            x[nan] = np.nan
            return x
        return f
    if scale == "Log":
        # ponytail: data reaching 0 shows six decades below the maximum
        floor = low if low > 0 else high * 1e-6
        if not floor > 0:
            return None
        lo, span = np.log10(floor), np.log10(high / floor)
    else:
        lo, span = low, high - low

    def f(x):
        if scale == "Log":
            np.maximum(x, floor, out=x)
            np.log10(x, out=x)
        x -= lo
        x /= span
        np.clip(x, 0.0, 1.0, out=x)
        if scale == "Low":                  # log10(1 + 9t)
            x *= 9.0
            x += 1.0
            np.log10(x, out=x)
        elif scale == "High":               # 1 - log10(1 + 9(1 - t))
            x *= -9.0
            x += 10.0
            np.log10(x, out=x)
            np.subtract(1.0, x, out=x)
        elif scale == "Banded":
            x *= BANDS
            np.floor(x, out=x)
            np.minimum(x, BANDS - 1, out=x)
            x += 0.5
            x /= BANDS
        return x
    return f


def scaled_colormap(cmap, scale_fn, low, high):
    """``cmap`` re-spaced by ``scale_fn`` for the colour bar: its value axis
    stays linear and each value shows the colour the image gives it."""
    if scale_fn is None:
        return cmap
    steps = np.linspace(0.0, 1.0, 256)
    pos = scale_fn(np.linspace(low, high, 256, dtype=np.float32))
    return pg.ColorMap(steps, cmap.map(pos.astype(float), mode="byte"))


def set_palette_image(image, data, cmap, low, high, scale_fn=None):
    """Show ``data`` on ``image`` as uint8 colour indices.

    Qt scales an indexed image itself at each paint (~20 ms per zoom step),
    where pyqtgraph's autoDownsample re-coloured and mean-averaged the whole
    chunk on every zoom level (0.2-1.5 s on 36M cells) - and averaged class
    codes into classes that do not exist.  ``low`` maps to index 1 and
    ``high`` to 255 (through ``scale_fn``, see ``color_scale``); index 0 is
    transparent: NaN, or the dtype-minimum sentinel of a native integer chunk.
    """
    data = np.asarray(data)
    x = data.astype(np.float32)
    if data.dtype.kind == "i":
        missing = data == np.iinfo(data.dtype).min
    else:
        missing = ~np.isfinite(x)
    if scale_fn is None:
        x -= low
        x *= 254.0 / (high - low) if high > low else 0.0
        np.clip(x, 0.0, 254.0, out=x)
    else:
        scale_fn(x)
        x *= 254.0
    x += 1.5                       # rounds on the cast: [low, high] -> 1..255
    x[missing] = 0.0
    lut = np.vstack(([0, 0, 0, 0], cmap.getLookupTable(nPts=255, alpha=True)))
    image.setImage(x.astype(np.uint8), levels=None, autoLevels=False)
    image.setLookupTable(lut.astype(np.ubyte))


def coordinate_variable(variable, dim):
    """The 1-D CF coordinate variable of ``dim`` (same name), or None.
    Works for a netCDF4 Variable and an xarray DataArray."""
    try:
        if hasattr(variable, "coords"):                      # xarray
            coord = variable.coords.get(dim)
        else:                                                 # netCDF4
            coord = variable.group().variables.get(dim)
    except Exception:
        return None
    if coord is None or getattr(coord, "ndim", 0) != 1:
        return None
    dims = getattr(coord, "dims", None) or getattr(coord, "dimensions", ())
    return coord if tuple(dims) == (dim,) else None


def read_axis(coord, window):
    """``(values, is_datetime)`` of a 1-D coordinate over ``(start, stop,
    step)`` - only that range is read."""
    start, stop, step = window
    values = (coord.values[start:stop:step] if hasattr(coord, "coords")
              else coord[start:stop:step])
    values = np.ma.asanyarray(values)
    if values.dtype.kind in "fiu":
        values = np.ma.filled(values.astype(float), np.nan)
    return to_plot_values(np.asarray(values))


def cursor_label(plot_widget, label, formatter, checkbox):
    """Feed the form's ``label`` with the value under the mouse, only while
    ``checkbox`` is ticked: each update repaints the label, which costs a
    round trip on a forwarded X display."""
    view = plot_widget.plotItem.vb

    def moved(pos):
        if not checkbox.isChecked():
            return
        if plot_widget.plotItem.sceneBoundingRect().contains(pos):
            point = view.mapSceneToView(pos)
            label.setText(formatter(point.x(), point.y()))

    plot_widget.scene().sigMouseMoved.connect(moved)
    checkbox.toggled.connect(lambda on: on or label.clear())
    return label


def resolution_starts(combo, times):
    """Period starts of ``times`` for the resolution chosen in ``combo``, or
    None; only resolutions coarser than the data's step stay selectable."""
    allowed = coarser_resolutions(times)
    for i in range(combo.count()):
        text = combo.itemText(i)
        combo.model().item(i).setEnabled(text == "None" or text in allowed)
    resolution = combo.currentText()
    return period_starts(times, resolution) if resolution in allowed else None


def is_discrete(variable):
    """Integer data that is not packed (no scale_factor / add_offset):
    classes or counts, where a mode means something."""
    attrs = getattr(variable, "attrs", None)
    if attrs is None:
        attrs = getattr(variable, "ncattrs", lambda: [])()
    return (np.dtype(variable.dtype).kind in "iub"
            and not {"scale_factor", "add_offset"} & set(attrs))


def allow_mode(combo, discrete):
    """Mode only for discrete data: continuous values are almost all unique,
    so their mode would just be the period's minimum."""
    combo.model().item(combo.findText("Mode")).setEnabled(discrete)
    if not discrete and combo.currentText() == "Mode":
        blocked = combo.blockSignals(True)
        combo.setCurrentText("Mean")
        combo.blockSignals(blocked)


def cell_double_click(plot_widget, callback, checkbox):
    """Call ``callback(x, y)`` with the view coordinates of a double-click,
    only while ``checkbox`` (the cursor read-out box) is ticked."""
    view = plot_widget.plotItem.vb

    def clicked(event):
        if not (event.double() and checkbox.isChecked()):
            return
        pos = event.scenePos()
        if view.sceneBoundingRect().contains(pos):
            point = view.mapSceneToView(pos)
            callback(point.x(), point.y())

    plot_widget.scene().sigMouseClicked.connect(clicked)


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
        self._settle.timeout.connect(self._settled)
        self._shown = None
        for bar in (self.vbar, self.hbar):
            bar.valueChanged.connect(self._value_changed)
            bar.sliderReleased.connect(self._settle.start)
        self.set_extent(0, 0, 0, 0)

    def _value_changed(self, _value):
        self.moved.emit()
        # a drag loads once, on release: loading at every pause *during* a
        # drag queued second-long reads and froze the window
        if not (self.vbar.isSliderDown() or self.hbar.isSliderDown()):
            self._settle.start()

    def _settled(self):
        if reading():                         # one read at a time
            self._settle.start()
            return
        if self.offsets() != self._shown:     # only a real move reads
            self._shown = self.offsets()
            self.windowChanged.emit()

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
        self._shown = self.offsets()

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
            # not AdjustToContents: that measures every item on each relayout,
            # 0.7 s for a 216000-entry dimension
            selector.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy
                .AdjustToMinimumContentsLengthWithIcon)
            selector.setMinimumContentsLength(7)
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
        # skip a frame while the previous one is still being read
        self.timer.timeout.connect(
            lambda: reading() or self.update_frame(False))

        self.horizontalSlider_timeStep.valueChanged.connect(self.tstep_t)
        self.pushButton_firstTime.clicked.connect(self.first_t)
        self.pushButton_prevTime.clicked.connect(self.prev_t)
        self.pushButton_runBackward.clicked.connect(self.prun_t)
        self.pushButton_runForward.clicked.connect(self.nrun_t)
        self.pushButton_nextTime.clicked.connect(self.next_t)
        self.pushButton_lastTime.clicked.connect(self.last_t)
        # temporal aggregation: the time controls then step through periods
        self._periods = None    # each period's first step + the end, or None
        self.comboBox_tres.currentIndexChanged.connect(self._temporal_changed)
        self.comboBox_tstat.currentIndexChanged.connect(self._temporal_changed)

    def _temporal_changed(self):
        self.set_unlim(self._time_variable_combo.currentText())
        self.set_tstep(self._current_time_index())  # the period of the step
        self.update_frame(True)

    def _time_periods(self, time, count):
        """Period starts for the chosen resolution, or None; only the
        resolutions coarser than the data's step stay selectable."""
        times = np.asarray(time.values if self.usex else time)[:count]
        starts = resolution_starts(self.comboBox_tres, times)
        return None if starts is None else np.r_[starts, times.size]  # + end

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

    def _time_value(self, axis):
        values = self._time_dimension_row.values()
        return values[axis] if 0 <= axis < len(values) else ""

    def _raw_time_index(self):
        """The time selector's step (a raw index, never a period)."""
        try:
            return int(self._time_value(self.iunlim))
        except ValueError:
            return 0

    def _current_time_index(self):
        """The frame shown: a time step, or a period while aggregating."""
        if self.iunlim < 0 or self.nunlim <= 0:
            return 0
        value = self._raw_time_index()
        if self._periods is not None:
            value = int(np.searchsorted(self._periods, value, side="right")) - 1
        return min(max(value, 0), self.nunlim - 1)

    def set_tstep(self, it):
        if self.iunlim < 0 or self.nunlim <= 0:
            return
        it = min(max(int(it), 0), self.nunlim - 1)
        raw = int(self._periods[it]) if self._periods is not None else it
        self._time_dimension_row.set_value(self.iunlim, raw)
        blocked = self.horizontalSlider_timeStep.blockSignals(True)
        self.horizontalSlider_timeStep.setValue(it)
        self.horizontalSlider_timeStep.blockSignals(blocked)
        details = self._time_details(self._time_variable_combo.currentText())
        if details is None:
            return
        _variable, time, _axis, _count = details
        value = time.values[raw] if self.usex else time[raw]
        if self._periods is not None:
            value = period_label(value, self.comboBox_tres.currentText())
        elif not self.usex:
            try:
                value = np.around(value, 4)
            except TypeError:
                pass
        self.label_timeValue.setText(str(value))

    def set_unlim(self, vardim):
        details = self._time_details(vardim)
        self._periods = None
        values = self._time_dimension_row.values()
        if (details is not None and not str(self._time_value(details[2])).isdigit()
                and sum(str(v) == "all" for v in values) >= 2):
            details = None      # time is an axis of the 2-D view: left alone
        if details is None:
            self.iunlim = -1
            self.nunlim = 0
        else:
            variable, time, self.iunlim, count = details
            allow_mode(self.comboBox_tstat, is_discrete(variable))
            self._periods = self._time_periods(time, count)
            self.nunlim = (count if self._periods is None
                           else self._periods.size - 1)
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

    # set by TimeControlMixin on panels with time controls
    iunlim = -1
    _periods = None

    def __init__(self, window, session: NcvSession, name: str):
        super().__init__(window)
        self.window = window
        self.session = session
        self.name = name
        self._updating = False
        self._zwindow = None
        self._chunk_notified = None
        self._series_win = None
        self._series_tile = None
        self._agg_cache = None      # (key, aggregated frame), see read_values
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
        """Fill ``combo`` once the window shows: parsing pyqtgraph's colour
        maps for the icons takes ~0.6 s.  Until then the empty combo means
        viridis (see ``selected_cmap_object``)."""
        def fill():
            blocked = combo.blockSignals(True)   # no redraw for the fill
            try:
                combo.clear()
                for name in sorted(pg.colormap.listMaps()):
                    combo.addItem(_cmap_icon(pg.colormap.get(name)), name)
                combo.setCurrentText("viridis")
            finally:
                combo.blockSignals(blocked)

        timer = QtCore.QTimer(combo)    # dies with the combo, fill and all
        timer.setSingleShot(True)
        timer.timeout.connect(fill)
        timer.start(0)

    def selected_cmap_object(self, combo, reverse_check):
        cmap = pg.colormap.get(combo.currentText() or "viridis")
        if reverse_check.isChecked():
            # pg caches colormaps and reverse() mutates in place
            cmap = copy.deepcopy(cmap)
            cmap.reverse()
        return cmap

    def slice_miss(self, dim_controls: DimensionControlRow, variable,
                   window=None, native=False, aggregate=False):
        """Read the selection with missing values removed.

        Normally returns a float array with NaN where data is missing.  With
        ``native`` an unscaled signed-integer chunk is kept in its own dtype
        instead - no float copy - and ``(data, (min, max))`` is returned, with
        missing cells set to the dtype minimum (see ``set_palette_image``).
        ``aggregate``: see ``read_values``.
        """
        values = (list(dim_controls) if isinstance(dim_controls, list)
                  else dim_controls.values())
        values.extend(["0"] * max(0, variable.ndim - len(values)))
        out = self.read_values(values, variable, window, aggregate)
        if out.ndim > 1:
            out = out.squeeze()
        if native and np.issubdtype(out.dtype, np.signedinteger) and out.ndim == 2:
            kept = self._native_chunk(out)
            if kept is not None:
                return kept
        return self._to_nan(out, get_miss(self, variable))

    def read_values(self, values, variable, window=None, aggregate=False):
        """``get_slice_values`` on the worker thread.  With ``aggregate`` and a
        temporal resolution chosen (``TimeControlMixin``), the statistic of
        the current period replaces its single time step - the time axis
        keeps length 1, as a single step's read does."""
        axis = self.iunlim
        if not (aggregate and self._periods is not None
                and 0 <= axis < len(values) and str(values[axis]).isdigit()):
            return read_responsive(
                lambda: get_slice_values(values, variable, window=window))
        k = self._current_time_index()
        p0, p1 = int(self._periods[k]), int(self._periods[k + 1])
        stat = self.comboBox_tstat.currentText()
        # one frame cached: colour, limit and repeated redraws reuse it
        key = (self.session.generation, getattr(variable, "name", None),
               variable.shape, tuple(values),
               tuple(sorted((window or {}).items())), p0, p1, stat)
        if self._agg_cache is None or self._agg_cache[0] != key:
            self._agg_cache = (key, self._aggregate(
                values, variable, window, axis, p0, p1, stat))
        return self._agg_cache[1]

    def _aggregate(self, values, variable, window, axis, p0, p1, stat):
        """``stat`` over steps ``[p0, p1)`` of ``axis``, reading contiguous
        time hyperslabs (netCDF pays per chunk decompressed, so no chunk is
        read twice) with memory bounded by ``memory_budget_cells``."""
        values = list(values)
        values[axis] = "all"
        window = dict(window or {})
        miss = get_miss(self, variable)
        reduced = [str(v) in DIMMETHODS for v in values]
        taxis = axis - sum(reduced[:axis])      # time's place in a read
        cells, rows_axis = 1, None              # values per step; a split axis
        for a, v in enumerate(values):
            if a != axis and (v == "all" or reduced[a]):
                start, stop, step = window.get(a, (0, variable.shape[a], 1))
                cells *= len(range(start, min(stop, variable.shape[a]), step))
                if v == "all" and rows_axis is None:
                    rows_axis = a

        def read():
            out = read_responsive(
                lambda: get_slice_values(values, variable, window=window))
            return np.moveaxis(np.asarray(self._to_nan(out, miss), dtype=float),
                               taxis, 0)

        if stat in ("Sum", "Mean", "Minimum", "Maximum"):
            # stream whole time chunks; memory stays at one block
            tchunk = (chunk_shape(variable, (axis,)) or (1,))[0]
            block = max(1, memory_budget_cells() // max(cells, 1))
            block = max(tchunk, block // tchunk * tchunk)
            out = count = None
            for t0 in range(p0, p1, block):
                window[axis] = (t0, min(t0 + block, p1), 1)
                part = read()
                if stat in ("Sum", "Mean"):
                    valid = np.isfinite(part)
                    total = np.where(valid, part, 0.0).sum(axis=0)
                    n = valid.sum(axis=0)
                    out = total if out is None else out + total
                    count = n if count is None else count + n
                else:
                    pick = np.fmin if stat == "Minimum" else np.fmax
                    m = pick.reduce(part, axis=0)
                    out = m if out is None else pick(out, m)
            if count is not None:
                if stat == "Mean":
                    out = out / np.maximum(count, 1)
                out[count == 0] = np.nan
            return np.expand_dims(out, taxis)

        # Median, Mode need the whole period: split rows if it is too big
        window[axis] = (p0, p1, 1)
        steps = p1 - p0
        budget = memory_budget_cells(bytes_per_cell=48, cap=None)
        whole = np.zeros(1, dtype=np.intp)
        if rows_axis is None or steps * cells <= budget:
            return np.expand_dims(aggregate_series(read(), whole, stat)[0], taxis)
        start, stop, step = window.get(rows_axis, (0, variable.shape[rows_axis], 1))
        nrows = len(range(start, min(stop, variable.shape[rows_axis]), step))
        unit = max(1, (chunk_shape(variable, (rows_axis,)) or (1,))[0] // step)
        k = max(1, budget // max(steps * (cells // nrows), 1))
        k = max(unit, k // unit * unit)      # whole chunks: none read twice
        parts = []
        for i in range(0, nrows, k):
            window[rows_axis] = (start + i * step,
                                 min(stop, start + (i + k) * step), step)
            parts.append(aggregate_series(read(), whole, stat)[0])
        rpos = rows_axis - sum(reduced[:rows_axis])
        rpos -= taxis < rpos                 # the time axis is reduced away
        return np.expand_dims(np.concatenate(parts, axis=rpos), taxis)

    def _to_nan(self, out, miss):
        """Float copy of a read with NaN where missing."""
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

    def _plot_cell_series(self, vardim, dim_controls, row, col, transposed,
                          where):
        """Add the shown cell ``row``/``col`` (labelled ``where``) along time,
        else the first other dimension longer than one, to this panel's
        series window, as ncview does.  Reads one value per step, never the
        variable."""
        if self._zwindow is None:
            return
        group, name, variable = resolve_selected_variable(self, vardim)
        dims = tuple(getattr(variable, "dims", None) or variable.dimensions)
        values = dim_controls.values()
        values.extend(["0"] * max(0, variable.ndim - len(values)))
        shown = display_axes(values)
        if len(shown) != 2:
            return
        if transposed:
            row, col = col, row
        for axis, i, (start, _stop, step) in zip(shown, (row, col),
                                                 self._zwindow):
            values[axis] = str(start + i * step)
        free = [a for a in range(variable.ndim)
                if a not in shown and variable.shape[a] > 1]
        if not free:
            self.label_cursor.setText(f"{name}: no time dimension to plot")
            return
        unlimited = self.dunlim if self.usex else self.dunlim[group]
        axis = (dims.index(unlimited) if unlimited in dims
                and dims.index(unlimited) in free else free[0])
        label = f"{name} @ {where} [" + ", ".join(
            f"{dims[a]}={values[a]}" for a in range(variable.ndim)
            if a != axis) + "]"
        win = self._series_win
        if win is not None and label in {
                line.name() for line in win.plot.plotItem.listDataItems()}:
            win.raise_()        # already plotted: nothing to read
            return
        values[axis] = "all"
        win = self._series_window()     # up before the read: the click answers
        win.setWindowTitle(f"ncv {self.name}: reading {name}…")
        try:
            y = self._cell_series(vardim, variable, values, shown, axis)
        finally:
            win.setWindowTitle(f"ncv {self.name}: time series")
        x, is_date, xlabel = np.arange(y.size, dtype=float), False, dims[axis]
        try:    # time from the session, else the dim's coordinate variable
            if dims[axis] == unlimited:
                found, found_label = (
                    to_plot_values(self.time_values(group)), "Date")
            else:
                coord = coordinate_variable(variable, dims[axis])
                found, found_label = (read_axis(coord, (0, y.size, 1)),
                                      set_axis_label(coord))
            if found[0].size == y.size:
                (x, is_date), xlabel = found, found_label
        except Exception:
            pass
        if x.size > 1 and x[0] > x[-1]:     # clipToView wants x ascending
            x, y = x[::-1], y[::-1]

        item = win.plot.plotItem
        if isinstance(item.getAxis("bottom"), pg.DateAxisItem) != is_date:
            item.clear()    # dates and plain numbers don't share an axis
            win.ylabels.clear()
            item.setAxisItems({"bottom": (
                pg.DateAxisItem(orientation="bottom") if is_date
                else pg.AxisItem(orientation="bottom"))})
        color = pg.intColor(len(item.listDataItems()), hues=9)
        # a marker is one pixmap blit per point per repaint (4.6 s to add a
        # 1e6-step series): only short series get them
        markers = ({"symbol": "o", "symbolSize": 5, "symbolBrush": color}
                   if y.size <= 2000 else {})
        line = item.plot(x, y, pen=color, connect="finite", name=label,
                         **markers)
        line.raw = (x, y)       # the dialog's aggregation works from these
        line.discrete = is_discrete(variable)
        self._aggregate_lines(win)
        win.ylabels.add(set_axis_label(variable))
        item.setLabel("left", next(iter(win.ylabels))
                      if len(win.ylabels) == 1 else "")
        item.setLabel("bottom", xlabel)

    def _aggregate_lines(self, win):
        """Show every line of the series dialog aggregated as its combos say,
        from the raw series kept on the line: no read (~16 ms per 244k)."""
        dates = isinstance(win.plot.plotItem.getAxis("bottom"), pg.DateAxisItem)
        for combo in (win.tres, win.tstat):
            combo.setEnabled(dates)
        lines = win.plot.plotItem.listDataItems()
        allow_mode(win.tstat, all(line.discrete for line in lines))
        for line in lines:
            x, y = line.raw
            starts = (resolution_starts(
                win.tres, x.astype(np.int64).astype("datetime64[s]"))
                if dates else None)
            if starts is not None:
                x, y = x[starts], aggregate_series(
                    y, starts, win.tstat.currentText())
            line.setData(x, y)

    def _series_window(self):
        """This panel's series dialog, created on first use and shown."""
        win = self._series_win
        if win is None:
            win = self._series_win = QtWidgets.QDialog(self)
            win.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            win.plot = pg.PlotWidget()
            win.plot.showGrid(x=True, y=True, alpha=0.3)
            win.plot.addLegend()
            # ~2 points per pixel (spikes kept), visible range only: a 1e6-step
            # series pans like a short one
            win.plot.setDownsampling(auto=True, mode="peak")
            win.plot.setClipToView(True)
            win.ylabels = set()
            # temporal aggregation of the lines (combos made in code: the
            # dialog has no form)
            win.tres, win.tstat = QtWidgets.QComboBox(), QtWidgets.QComboBox()
            win.tres.addItems(["None", *RESOLUTIONS])
            win.tstat.addItems(STATISTICS)
            win.tstat.setCurrentText("Mean")
            win.tres.setToolTip("Temporal resolution: each line's statistic "
                                "per period (only periods longer than the "
                                "data's time step)")
            win.tstat.setToolTip("Statistic over each period's time steps")
            for combo in (win.tres, win.tstat):
                combo.currentIndexChanged.connect(
                    lambda _i: self._aggregate_lines(win))
            bar = QtWidgets.QHBoxLayout()
            bar.addWidget(QtWidgets.QLabel("aggregate"))
            bar.addWidget(win.tres)
            bar.addWidget(win.tstat)
            bar.addStretch(1)
            layout = QtWidgets.QVBoxLayout(win)
            layout.addLayout(bar)
            layout.addWidget(win.plot)
            def forget(*_):             # closing frees the cached tile too
                self._series_win = self._series_tile = None
            win.destroyed.connect(forget)
            win.resize(720, 420)
        win.show()
        win.raise_()
        return win

    def _cell_series(self, vardim, variable, values, shown, axis):
        """The series selected by ``values`` (``all`` on ``axis``, indices on
        ``shown``).  netCDF pays per chunk decompressed (~11 µs each: 2.7 s
        for one cell of a 244k-step hourly file stored one chunk per step), so
        the cell's whole spatial chunk tile is read in that same pass and
        kept: later clicks inside the tile read nothing."""
        idx = [int(values[a]) for a in shown]
        chunks = chunk_shape(variable, shown)
        steps = variable.shape[axis]
        if chunks:
            tile = [(i // c * c, min(i // c * c + c, variable.shape[a]))
                    for i, c, a in zip(idx, chunks, shown)]
            rows, cols = (t1 - t0 for t0, t1 in tile)
            key = (self.session.generation, vardim, axis, tuple(tile),
                   tuple(v for a, v in enumerate(values) if a not in shown))
            # a cached tile is used whatever the budget says now: holding it
            # is what lowered the available memory
            hit = self._series_tile is not None and self._series_tile[0] == key
            if hit or steps * rows * cols <= memory_budget_cells(
                    bytes_per_cell=8, cap=None):
                if not hit:
                    tile_values = list(values)
                    for a in shown:
                        tile_values[a] = "all"
                    window = {a: (t0, t1, 1) for a, (t0, t1) in zip(shown, tile)}
                    # kept as read (float32 stays float32): only the clicked
                    # series is turned into float with NaN
                    data = read_responsive(lambda: get_slice_values(
                        tile_values, variable, window=window))
                    order = sorted((axis, *shown))
                    size = {axis: steps, shown[0]: rows, shown[1]: cols}
                    data = np.moveaxis(np.reshape(data, [size[a] for a in order]),
                                       order.index(axis), 0)
                    self._series_tile = (key, data)
                series = self._series_tile[1][:, idx[0] - tile[0][0],
                                              idx[1] - tile[1][0]]
                return np.asarray(self._to_nan(series, get_miss(self, variable)),
                                  dtype=float)
        return np.asarray(self.slice_miss(values, variable), dtype=float).ravel()

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

    def _series(self, vardim, dim_controls, window=None, native=False,
                aggregate=False):
        """Return ``(values, label, is_datetime, native_range)``.

        ``native_range`` is the valid min/max when ``native`` kept an integer
        chunk in its own dtype, else None and ``values`` is float.
        ``aggregate``: see ``read_values``.
        """
        group, name = vardim2var(vardim, self.groups)
        tname = self.tname if self.usex else self.tname[group]
        if name == tname:
            values, label = self.time_values(group), "Date"
            native = False
        else:
            values = selvar(self, name)
            label = set_axis_label(values)
        out = self.slice_miss(dim_controls, values, window=window,
                              native=native, aggregate=aggregate)
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
            stride = overview_stride(shape, chunk_shape(variable, axes),
                                     max_cells=budget)
            window = {a: (0, shape[i], stride[i]) for i, a in enumerate(axes)}
            return window, shape, shape, "coarse"
        if shape[0] * shape[1] <= budget:
            window = {a: (0, shape[i], 1) for i, a in enumerate(axes)}
            return window, shape, shape, "full"
        side = max(1, int(np.sqrt(budget)))
        span = [min(side, n) for n in shape]
        chunks = chunk_shape(variable, axes)
        if chunks:
            for i in (0, 1):
                # a touched chunk is read whole: when one is longer than the
                # window, take all of it and fit the other axis to the budget,
                # e.g. one-row chunks (1, nx) -> full-width strips of
                # budget // nx rows instead of a square reading 36x its size
                if min(chunks[i], shape[i]) > span[i]:
                    span[i] = min(chunks[i], shape[i], budget)
                    span[1 - i] = max(1, min(shape[1 - i], budget // span[i]))
                    break
        span = tuple(span)
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
        # a mouse drag reads once, on release (like the scroll bars), and
        # never while another read is running
        if reading() or (QtWidgets.QApplication.mouseButtons()
                         != QtCore.Qt.MouseButton.NoButton):
            self._reload.start()
            return
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
