"""Shared Qt helpers for ncv panels."""
from __future__ import annotations

import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from .dimensions import dimension_values
from .ncvmethods import get_miss
from .ncvutils import get_slice_values, parse_entry, set_miss
from .qt_compat import QtCore, QtGui, QtWidgets
from .session import HAVE_XARRAY, NcvSession


__all__ = [
    "DimensionControlRow",
    "PlotPanel",
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
