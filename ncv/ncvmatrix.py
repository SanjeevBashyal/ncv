"""Qt matrix panel."""
from __future__ import annotations

import html
import re
import warnings

import numpy as np

from .dimensions import (
    dimension_specs,
    empty_dimension_specs,
    resolve_selected_variable,
)
from .ncvcommon import (
    DimensionControlRow,
    PlotPanel,
    ScrollableView,
    TimeControlMixin,
    load_ui,
    set_combo_items,
)
from .ncvmethods import get_miss
from .ncvutils import get_slice_values, set_miss, vardim2var
from .qt_compat import QtCore, QtWidgets


_ATTR = re.compile(r"^(\s*)([\w.][\w. ]*):(.*)$")


def _metadata_html(text):
    """Monospace, attribute names bold, netCDF4's <class ...> noise dropped."""
    out = []
    for line in text.splitlines():
        if line.startswith("<class "):
            continue
        match = _ATTR.match(line)
        if match:
            indent, key, rest = match.groups()
            out.append(f"{indent}<b>{html.escape(key)}:</b>{html.escape(rest)}")
        else:
            out.append(html.escape(line))
    return "<pre>" + "\n".join(out) + "</pre>"


def _format_value(value, number_format):
    if np.ma.is_masked(value):
        return ""
    array = np.asarray(value)
    try:
        if np.issubdtype(array.dtype, np.datetime64):
            return "" if np.isnat(value) else str(value)
        if np.issubdtype(array.dtype, np.number):
            if np.issubdtype(array.dtype, np.floating) and np.isnan(value):
                return ""
            return number_format % value
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


class _ArrayTableModel(QtCore.QAbstractTableModel):
    """Lazy table view over a two-dimensional NumPy array."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = np.empty((0, 0))
        self.xheaders = None
        self.yheaders = None
        self.data_format = "%.1f"
        self.header_format = "%.1f"
        self.flip_vertical = False
        self.flip_horizontal = False
        self.show_indices = False
        self.row_index = None
        self.col_index = None
        self.missing = None

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.values.shape[0]

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.values.shape[1]

    def _source_row(self, row):
        return self.rowCount() - row - 1 if self.flip_vertical else row

    def _source_column(self, column):
        return self.columnCount() - column - 1 if self.flip_horizontal else column

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            row = self._source_row(index.row())
            column = self._source_column(index.column())
            value = self.values[row, column]
            if self.missing is not None and value == self.missing:
                return ""                  # native integer chunk: no data
            return _format_value(value, self.data_format)
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return (QtCore.Qt.AlignmentFlag.AlignRight |
                    QtCore.Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        horizontal = orientation == QtCore.Qt.Orientation.Horizontal
        source = (self._source_column(section) if horizontal
                  else self._source_row(section))
        if self.show_indices:
            # position in the variable, not in the loaded chunk
            index = self.col_index if horizontal else self.row_index
            if index is not None and source < len(index):
                return str(index[source])
            return str(source)
        headers = self.xheaders if horizontal else self.yheaders
        if headers is None or source >= len(headers):
            return str(source)
        return _format_value(headers[source], self.header_format)

    def set_table(self, values=None, xheaders=None, yheaders=None,
                  row_index=None, col_index=None, missing=None):
        self.beginResetModel()
        self.missing = missing
        self.values = (np.empty((0, 0)) if values is None
                       else np.asarray(values))
        self.xheaders = xheaders
        self.yheaders = yheaders
        self.row_index = row_index
        self.col_index = col_index
        self.endResetModel()

    def set_options(self, data_format, header_format, flip_vertical,
                    flip_horizontal, show_indices):
        self.beginResetModel()
        self.data_format = data_format
        self.header_format = header_format
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.show_indices = show_indices
        self.endResetModel()


class MatrixPanel(TimeControlMixin, PlotPanel):
    """Array table and NetCDF metadata tab."""

    def __init__(self, window, session):
        super().__init__(window, session, "Matrix")
        self._updating = True
        self._build_ui()
        self._updating = False
        self.reinit()

    def _build_ui(self):
        load_ui("matrix_panel", self)
        self.connect_file_controls()

        self.model = _ArrayTableModel(self)
        self.tableView_showMatrix.setModel(self.model)
        # runtime-only: the scroll bars live in a grid around the table, which
        # Designer cannot express. Everything static (splitter, scroll-bar
        # policies, hidden header, stretch, size policies) is in the .ui.
        self._header_splitter = self.splitter_matrixAndHeader
        self.scroll = ScrollableView(self.tableView_showMatrix)
        self.scroll.windowChanged.connect(self._scrolled)
        self._header_splitter.insertWidget(0, self.scroll)
        self._header_splitter.setStretchFactor(0, 3)
        self._header_splitter.setStretchFactor(1, 1)
        self._header_splitter.setCollapsible(1, True)
        self._header_sizes = None
        self._native_range = None
        self.pushButton_metadata.toggled.connect(self._toggle_metadata)
        self.checkBox_fullCoarse.stateChanged.connect(lambda *_: self.redraw())
        self.zd = DimensionControlRow(self.maxdim)
        self.xd = DimensionControlRow(self.maxdim)
        self.yd = DimensionControlRow(self.maxdim)
        self.vDimensionsLayout.addWidget(self.zd)
        self.lonDimensionsLayout.addWidget(self.xd)
        self.latDimensionsLayout.addWidget(self.yd)
        self.init_time_controls(self.comboBox_z, self.zd)

        self.lineEdit_min.setReadOnly(True)
        self.lineEdit_max.setReadOnly(True)
        self.comboBox_z.currentIndexChanged.connect(self.selected_z)
        self.comboBox_x.currentIndexChanged.connect(self.selected_x)
        self.comboBox_y.currentIndexChanged.connect(self.selected_y)
        self.zd.changed.connect(self.spinned_z)
        self.xd.changed.connect(self.redraw)
        self.yd.changed.connect(self.redraw)
        self.checkBox_transVariable.stateChanged.connect(
            lambda *_: self.redraw())
        self.checkBox_allValues.stateChanged.connect(self._update_statistics)
        self.comboBox_dataFormat.currentIndexChanged.connect(
            self._update_model_options)
        self.comboBox_dataFormat.currentIndexChanged.connect(
            self._update_statistics)
        self.comboBox_rowColHeaderFormat.currentIndexChanged.connect(
            self._update_model_options)
        self.checkBox_flipTableTopBottom.stateChanged.connect(
            self._update_model_options)
        self.checkBox_flipTableLeftRight.stateChanged.connect(
            self._update_model_options)
        self.checkBox_showCellIndices.stateChanged.connect(
            self._update_model_options)
        self.pushButton_quit.clicked.connect(QtWidgets.QApplication.quit)
        self._update_model_options()

    def _toggle_metadata(self, shown):
        splitter = self._header_splitter
        if shown:
            self.textBrowser_showHeader.setVisible(True)
            width = max(splitter.width(), 1)
            splitter.setSizes(self._header_sizes
                              or [int(width * 0.7), int(width * 0.3)])
        else:
            # remember the width while it is still visible, so reopening
            # restores it instead of resetting to the default split
            self._header_sizes = splitter.sizes()
            self.textBrowser_showHeader.setVisible(False)
        self.pushButton_metadata.setText(
            "Metadata: \u25bc" if shown else "Metadata: \u25b6")

    def _visible_cells(self):
        """(rows, columns) the table viewport can show at once."""
        table = self.tableView_showMatrix
        rows = table.viewport().height() // max(1, table.verticalHeader().defaultSectionSize())
        cols = table.viewport().width() // max(1, table.horizontalHeader().defaultSectionSize())
        return max(1, rows), max(1, cols)

    def _chunk_center(self):
        """Cell (variable axis order) the next chunk is centred on: the middle
        of what the table shows, so scrolling either way stays inside it."""
        (r, c), (vr, vc) = self.scroll.offsets(), self._visible_cells()
        center = (r + vr / 2.0, c + vc / 2.0)
        return center[::-1] if self.checkBox_transVariable.isChecked() else center

    def _size_bars(self):
        """Thumbs = the visible cells, positioned within the whole variable."""
        if self._full_shape is None:
            self.scroll.set_extent(0, 0, 0, 0)
            return
        full = self._full_shape
        if self.checkBox_transVariable.isChecked():
            full = full[::-1]
        (r, c), (vr, vc) = self.scroll.offsets(), self._visible_cells()
        self.scroll.show_view(r, c, vr, vc, *full)

    def _show_offsets(self):
        """Scroll the table so the bar position is its top-left cell.

        Returns False when that region is not inside the loaded chunk.
        """
        if self._zwindow is None:
            return False
        (r, c), (vr, vc) = self.scroll.offsets(), self._visible_cells()
        (r0, r1, sr), (c0, c1, sc) = self._zwindow
        if self.checkBox_transVariable.isChecked():
            (r0, r1, sr), (c0, c1, sc) = (c0, c1, sc), (r0, r1, sr)
        if sr != 1 or sc != 1:
            return True          # full or coarse: nothing further to load
        if not (r0 <= r and r + vr <= r1 and c0 <= c and c + vc <= c1):
            return False
        table = self.tableView_showMatrix
        table.verticalScrollBar().setValue(r - r0)
        table.horizontalScrollBar().setValue(c - c0)
        return True

    def _scrolled(self):
        if self._updating:
            return
        # inside the loaded chunk: just move the table, no read at all
        if not self._show_offsets():
            self._refresh_table(update_statistics=False)

    def reinit(self):
        super().reinit()
        self._updating = True
        columns = self.columns()
        for combo in (self.comboBox_z, self.comboBox_x, self.comboBox_y):
            set_combo_items(combo, columns, "")
        for dimensions in (self.zd, self.xd, self.yd):
            dimensions.set_specs(empty_dimension_specs(self.maxdim))
        self.lineEdit_min.setText("None")
        self.lineEdit_max.setText("None")
        self.comboBox_repeat.setCurrentText("repeat")
        self.set_unlim("")

        x = self._default_coordinate(self.lonvar)
        y = self._default_coordinate(self.latvar)
        if x:
            self.comboBox_x.setCurrentText(x)
            self.xd.set_specs(dimension_specs(self, x, "var"))
        if y:
            self.comboBox_y.setCurrentText(y)
            self.yd.set_specs(dimension_specs(self, y, "var"))
        self.model.set_table()
        self._show_dataset_metadata()
        self._update_model_options()
        self._updating = False

    def _default_coordinate(self, values):
        if self.usex:
            return values or ""
        return next((value for value in values if value), "")

    def selected_z(self):
        if self._updating:
            return
        z = self.comboBox_z.currentText()
        if not z:
            self.zd.set_specs(empty_dimension_specs(self.maxdim))
            self.set_unlim("")
            self.model.set_table()
            self._set_statistics(None)
            self._show_dataset_metadata()
            return
        self.zd.set_specs(dimension_specs(self, z, "var"))
        self.set_unlim(z)
        self.set_tstep(0)
        self.redraw()

    def selected_x(self):
        if self._updating:
            return
        x = self.comboBox_x.currentText()
        self.xd.set_specs(
            dimension_specs(self, x, "var") if x
            else empty_dimension_specs(self.maxdim))
        self.redraw()

    def selected_y(self):
        if self._updating:
            return
        y = self.comboBox_y.currentText()
        self.yd.set_specs(
            dimension_specs(self, y, "var") if y
            else empty_dimension_specs(self.maxdim))
        self.redraw()

    def spinned_z(self):
        if self.iunlim >= 0:
            try:
                self.set_tstep(int(self.zd.values()[self.iunlim]))
            except (ValueError, IndexError):
                pass
        self.redraw()

    def set_tstep(self, index):
        super().set_tstep(index)
        current = self._current_time_index()
        for combo, dimensions in (
            (self.comboBox_x, self.xd),
            (self.comboBox_y, self.yd),
        ):
            details = self._time_details(combo.currentText())
            if details is not None:
                _variable, _time, axis, count = details
                dimensions.set_value(axis, min(current, count - 1))

    def redraw(self):
        if self._updating:
            return
        self._stop_animation()
        self._refresh_table()

    def update_frame(self, isframe=False):
        z = self.comboBox_z.currentText()
        if self._time_details(z) is None or self.nunlim <= 0:
            self._stop_animation()
            return
        index = self._current_time_index()
        if not isframe:
            index, direction, stop = self._next_time_index(self.anim_inc)
            if stop:
                self._stop_animation()
                return
            if direction != self.anim_inc:
                self._set_animation_direction(direction)
        self.set_tstep(index)
        self._refresh_table(
            update_statistics=not self.checkBox_allValues.isChecked())

    def _selected_values(self, vardim, dimensions, window=None, native=False):
        """``(variable, values)``; with ``native`` it is
        ``(variable, values, native_range)`` and a signed-integer 2-D chunk
        keeps its dtype, missing cells at the dtype minimum (no float copy).
        """
        group, selected_name = vardim2var(vardim, self.groups)
        _group, _physical_name, variable = resolve_selected_variable(
            self, vardim)
        tname = self.tname if self.usex else self.tname[group]
        synthetic_time = selected_name == tname
        source = self.time_values(group) if synthetic_time else variable
        if source.ndim == 0:
            out = source.values if self.usex else source[...]
        else:
            values = dimensions.values()
            values.extend(["0"] * max(0, source.ndim - len(values)))
            out = get_slice_values(values, source, window=window)
        if native and not synthetic_time:
            chunk = np.ma.asanyarray(out).squeeze()
            if np.issubdtype(chunk.dtype, np.signedinteger) and chunk.ndim == 2:
                kept = self._native_chunk(chunk)
                if kept is not None:
                    return variable, kept[0], kept[1]
        if not synthetic_time:
            out = self._replace_missing(variable, out)
        out = np.asanyarray(out).squeeze()
        return (variable, out, None) if native else (variable, out)

    def _replace_missing(self, variable, values):
        array = np.ma.asarray(values)
        if np.ma.is_masked(array):
            if np.issubdtype(array.dtype, np.number):
                # one conversion, then fill in place - filled() would copy again
                mask = np.ma.getmaskarray(array)
                array = np.array(array.data, dtype=float)
                array[mask] = np.nan
            else:
                fill = (np.datetime64("NaT")
                        if np.issubdtype(array.dtype, np.datetime64) else "")
                array = array.filled(fill)
        else:
            array = np.asarray(array)
        try:
            return set_miss(get_miss(self, variable), array)
        except (TypeError, ValueError):
            return np.asarray(array)

    def _refresh_table(self, update_statistics=True):
        z = self.comboBox_z.currentText()
        if not z:
            self.model.set_table()
            self._set_statistics(None)
            self._show_dataset_metadata()
            return
        window = self.scroll_window(z, self.zd, center=self._chunk_center())
        try:
            variable, raw, native_range = self._selected_values(
                z, self.zd, window=window, native=True)
        except Exception as exc:
            self.model.set_table()
            self._show_variable_metadata(z, f"Unable to read selection: {exc}")
            return

        array = np.asarray(raw)
        if array.ndim == 2 and self.checkBox_transVariable.isChecked():
            array = array.T
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            xlength = self._header_length(
                self.comboBox_x.currentText(), self.xd, True)
            ylength = self._header_length(
                self.comboBox_y.currentText(), self.yd, False)
            if ylength == array.size and xlength != array.size:
                array = array.reshape(-1, 1)
            else:
                array = array.reshape(1, -1)
        elif array.ndim > 2:
            self.model.set_table()
            if update_statistics:
                self._refresh_statistics(raw)
            self._show_variable_metadata(
                z, "Select dimension values until Z is at most two-dimensional.")
            return

        xheaders = self._headers(
            self.comboBox_x.currentText(), self.xd, array.shape, True)
        yheaders = self._headers(
            self.comboBox_y.currentText(), self.yd, array.shape, False)
        row_index = col_index = None
        if self._zwindow is not None and array.ndim == 2:
            row_index, col_index = (np.arange(*w) for w in self._zwindow)
            if self.checkBox_transVariable.isChecked():
                row_index, col_index = col_index, row_index
        self._native_range = native_range
        missing = (np.iinfo(array.dtype).min if native_range is not None
                   else None)
        self.model.set_table(array, xheaders, yheaders, row_index, col_index,
                             missing)
        self._size_bars()
        self._show_offsets()
        if update_statistics:
            self._refresh_statistics(raw, native_range)
        self._show_variable_metadata(z)

    def _refresh_statistics(self, current_values, value_range=None):
        if self.checkBox_allValues.isChecked():
            self._update_statistics()
        else:
            self._set_statistics(current_values, value_range)

    def _headers(self, vardim, dimensions, shape, horizontal):
        if not vardim:
            return None
        window = self.coord_window(vardim, horizontal, self.checkBox_transVariable.isChecked())
        try:
            _variable, values = self._selected_values(
                vardim, dimensions, window=window)
        except Exception:
            return None
        values = np.asarray(values)
        expected = shape[1] if horizontal else shape[0]
        if values.ndim == 0 and expected == 1:
            return values.reshape(1)
        if values.ndim == 1 and values.size == expected:
            return values
        if values.ndim == 2 and values.T.shape == shape:
            values = values.T   # the data was transposed for display
        if values.ndim == 2 and values.shape == shape:
            return values[0, :] if horizontal else values[:, 0]
        return None

    def _header_length(self, vardim, dimensions, horizontal):
        if not vardim:
            return None
        try:
            _variable, values = self._selected_values(
                vardim, dimensions,
                window=self.coord_window(vardim, horizontal, self.checkBox_transVariable.isChecked()))
        except Exception:
            return None
        values = np.asarray(values)
        return values.size if values.ndim <= 1 else None

    def _update_model_options(self, *_args):
        if not hasattr(self, "model"):
            return
        self.model.set_options(
            self.comboBox_dataFormat.currentText(),
            self.comboBox_rowColHeaderFormat.currentText(),
            self.checkBox_flipTableTopBottom.isChecked(),
            self.checkBox_flipTableLeftRight.isChecked(),
            self.checkBox_showCellIndices.isChecked(),
        )

    def _update_statistics(self, *_args):
        if self._updating:
            return
        z = self.comboBox_z.currentText()
        if not z:
            self._set_statistics(None)
            return
        if self.checkBox_allValues.isChecked():
            try:
                group, selected_name = vardim2var(z, self.groups)
                _group, _name, variable = resolve_selected_variable(self, z)
                tname = self.tname if self.usex else self.tname[group]
                source = (self.time_values(group) if selected_name == tname
                          else variable)
                values = source.values if self.usex else source[...]
                if selected_name != tname:
                    values = self._replace_missing(variable, values)
                self._set_statistics(values)
            except Exception:
                self._set_statistics(None)
        else:
            # what is displayed is what is summarised; this used to re-read Z
            # with no window - the whole variable - on every format change
            self._set_statistics(self.model.values, self._native_range)

    def _set_statistics(self, values, value_range=None):
        minimum = maximum = None
        if value_range is not None:
            minimum, maximum = value_range
        elif values is not None:
            array = np.asarray(values)
            try:
                if np.issubdtype(array.dtype, np.datetime64):
                    array = array[~np.isnat(array)]
                    if array.size:
                        minimum, maximum = np.min(array), np.max(array)
                elif np.issubdtype(array.dtype, np.number):
                    # nanmin/nanmax: boolean-indexing a large chunk builds an
                    # index array first, which costs seconds
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        low, high = np.nanmin(array), np.nanmax(array)
                    if np.isfinite(low):
                        minimum, maximum = low, high
                elif array.size:
                    minimum, maximum = np.min(array), np.max(array)
            except (TypeError, ValueError):
                pass
        number_format = self.comboBox_dataFormat.currentText()
        self.lineEdit_min.setText(
            "None" if minimum is None
            else _format_value(minimum, number_format))
        self.lineEdit_max.setText(
            "None" if maximum is None
            else _format_value(maximum, number_format))

    def _show_dataset_metadata(self):
        if not self.session.has_data:
            text = "No NetCDF file loaded."
        elif self.usex:
            text = str(self.fi)
        else:
            parts = []
            for index, dataset in enumerate(self.fi):
                name = (self.session.files[index]
                        if index < len(self.session.files) else str(index))
                groups = [dataset]
                for group in groups:
                    groups.extend(group.groups.values())
                header = "\n\n".join(str(group) for group in groups)
                parts.append(f"File: {name}\n{header}")
            text = "\n\n".join(parts)
        self.textBrowser_showHeader.setHtml(_metadata_html(text))

    def _show_variable_metadata(self, vardim, warning=""):
        try:
            _group, selected_name = vardim2var(vardim, self.groups)
            _group, physical_name, variable = resolve_selected_variable(
                self, vardim)
            title = f"Variable: {selected_name}"
            if physical_name != selected_name:
                title += f" (derived from {physical_name})"
            text = f"{title}\n{variable}"
        except Exception as exc:
            text = f"Variable: {vardim}\nUnable to read metadata: {exc}"
        if warning:
            text += f"\n\n{warning}"
        self.textBrowser_showHeader.setHtml(_metadata_html(text))


__all__ = ["MatrixPanel"]
