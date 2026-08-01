"""Qt matrix panel."""
from __future__ import annotations

import numpy as np

from .dimensions import (
    dimension_specs,
    empty_dimension_specs,
    resolve_selected_variable,
)
from .ncvcommon import (
    DimensionControlRow,
    PlotPanel,
    TimeControlMixin,
    set_combo_items,
)
from .ncvmethods import get_miss
from .ncvutils import get_slice_values, set_miss, vardim2var
from .pyui.ui_matrix_panel import Ui_widget_matrixPanel
from .qt_compat import QtCore, QtWidgets


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

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.values.shape[0]

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.values.shape[1]

    def _source_row(self, row):
        return self.rowCount() - row - 1 if self.flip_vertical else row

    def _source_column(self, column):
        return self.columnCount() - column - 1 if self.flip_horizontal else column

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == QtCore.Qt.DisplayRole:
            row = self._source_row(index.row())
            column = self._source_column(index.column())
            return _format_value(self.values[row, column], self.data_format)
        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        horizontal = orientation == QtCore.Qt.Horizontal
        source = (self._source_column(section) if horizontal
                  else self._source_row(section))
        if self.show_indices:
            return str(source)
        headers = self.xheaders if horizontal else self.yheaders
        if headers is None or source >= len(headers):
            return str(source)
        return _format_value(headers[source], self.header_format)

    def set_table(self, values=None, xheaders=None, yheaders=None):
        self.beginResetModel()
        self.values = (np.empty((0, 0)) if values is None
                       else np.asarray(values))
        self.xheaders = xheaders
        self.yheaders = yheaders
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


class MatrixPanel(TimeControlMixin, PlotPanel, Ui_widget_matrixPanel):
    """Array table and NetCDF metadata tab."""

    def __init__(self, window, session):
        super().__init__(window, session, "Matrix")
        self._updating = True
        self._build_ui()
        self._updating = False
        self.reinit()

    def _build_ui(self):
        self.setupUi(self)
        self.connect_file_controls()

        self.model = _ArrayTableModel(self)
        self.tableView_showMatrix.setModel(self.model)
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
        self.checkBox_allValues.stateChanged.connect(self._update_statistics)
        self.comboBox_dataFormat.currentIndexChanged.connect(
            self._update_model_options)
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

    def _selected_values(self, vardim, dimensions):
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
            out = get_slice_values(values, source)
        if not synthetic_time:
            out = self._replace_missing(variable, out)
        return variable, np.asanyarray(out).squeeze()

    def _replace_missing(self, variable, values):
        array = np.ma.asarray(values)
        if np.ma.is_masked(array):
            if np.issubdtype(array.dtype, np.datetime64):
                fill = np.datetime64("NaT")
            elif np.issubdtype(array.dtype, np.number):
                if not np.issubdtype(array.dtype, np.inexact):
                    array = array.astype(float)
                fill = np.nan
            else:
                fill = ""
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
        try:
            variable, raw = self._selected_values(z, self.zd)
        except Exception as exc:
            self.model.set_table()
            self._show_variable_metadata(z, f"Unable to read selection: {exc}")
            return

        array = np.asarray(raw)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            xlength = self._header_length(
                self.comboBox_x.currentText(), self.xd)
            ylength = self._header_length(
                self.comboBox_y.currentText(), self.yd)
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
        self.model.set_table(array, xheaders, yheaders)
        if update_statistics:
            self._refresh_statistics(raw)
        self._show_variable_metadata(z)

    def _refresh_statistics(self, current_values):
        if self.checkBox_allValues.isChecked():
            self._update_statistics()
        else:
            self._set_statistics(current_values)

    def _headers(self, vardim, dimensions, shape, horizontal):
        if not vardim:
            return None
        try:
            _variable, values = self._selected_values(vardim, dimensions)
        except Exception:
            return None
        values = np.asarray(values)
        expected = shape[1] if horizontal else shape[0]
        if values.ndim == 0 and expected == 1:
            return values.reshape(1)
        if values.ndim == 1 and values.size == expected:
            return values
        if values.ndim == 2 and values.shape == shape:
            return values[0, :] if horizontal else values[:, 0]
        return None

    def _header_length(self, vardim, dimensions):
        if not vardim:
            return None
        try:
            _variable, values = self._selected_values(vardim, dimensions)
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
            try:
                _variable, values = self._selected_values(z, self.zd)
            except Exception:
                values = None
            self._set_statistics(values)

    def _set_statistics(self, values):
        minimum = maximum = None
        if values is not None:
            array = np.asarray(values)
            try:
                if np.issubdtype(array.dtype, np.datetime64):
                    array = array[~np.isnat(array)]
                elif np.issubdtype(array.dtype, np.number):
                    array = array[np.isfinite(array)]
                if array.size:
                    minimum, maximum = np.min(array), np.max(array)
            except (TypeError, ValueError):
                pass
        self.lineEdit_min.setText("None" if minimum is None else str(minimum))
        self.lineEdit_max.setText("None" if maximum is None else str(maximum))

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
        self.textBrowser_showHeader.setPlainText(text)

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
        self.textBrowser_showHeader.setPlainText(text)


__all__ = ["MatrixPanel"]
