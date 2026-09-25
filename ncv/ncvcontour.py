"""Qt contour plotting panel."""
from __future__ import annotations

import numpy as np

from .dimensions import (
    dimension_specs,
    empty_dimension_specs,
    resolve_selected_variable,
)
from .ncvcommon import DimensionControlRow, PlotPanel, ScrollableView
from .ncvcommon import cursor_label, load_ui
from .ncvcommon import parse_limits, set_combo_items
from .ncvutils import cell_edges, format_coord_contour
from .qt_compat import QtCore, QtWidgets, pg


class ContourPanel(PlotPanel):
    """Contour plot tab, drawn as a heat map."""

    def __init__(self, window, session):
        super().__init__(window, session, "Contour")
        self._build_ui()
        self.reinit()

    def _build_ui(self):
        load_ui("contour_panel", self)
        self.connect_file_controls()

        self.plot = pg.PlotWidget()
        self.item = self.plot.plotItem
        self.image = pg.ImageItem()
        self.item.addItem(self.image)
        self.colorbar = pg.ColorBarItem(interactive=False)
        self.colorbar.setImageItem(self.image, insert_in=self.item)
        self.scroll = ScrollableView(self.plot)
        self.scroll.windowChanged.connect(self._scrolled)
        self.plotLayout.addWidget(self.scroll, 1)
        cursor_label(self.plot, self.plotLayout, self._format_cursor)
        self._overview = True
        self._xx = self._yy = self._zz = None
        self._zwindow = None
        self._xdate = self._ydate = False

        self.zd = DimensionControlRow(self.maxdim)
        self.xd = DimensionControlRow(self.maxdim)
        self.yd = DimensionControlRow(self.maxdim)
        self.zDimensionsLayout.addWidget(self.zd)
        self.xDimensionsLayout.addWidget(self.xd)
        self.yDimensionsLayout.addWidget(self.yd)
        self.populate_cmap_combo(self.comboBox_cmap)

        self.comboBox_z.currentIndexChanged.connect(self.selected_z)
        self.checkBox_transposeZ.stateChanged.connect(self.checked)
        self.lineEdit_zlim.editingFinished.connect(self.entered_z)
        self.zd.changed.connect(self.spinned_z)
        self.comboBox_x.currentIndexChanged.connect(self.selected_x)
        self.checkBox_invX.stateChanged.connect(self.checked)
        self.comboBox_y.currentIndexChanged.connect(self.selected_y)
        self.checkBox_invY.stateChanged.connect(self.checked)
        self.xd.changed.connect(self.spinned_x)
        self.yd.changed.connect(self.spinned_y)
        self.comboBox_cmap.currentIndexChanged.connect(self.selected_cmap)
        for check in (self.checkBox_revCmap, self.checkBox_grid):
            check.stateChanged.connect(self.checked)
        self.pushButton_quit.clicked.connect(QtWidgets.QApplication.quit)

    def reinit(self):
        super().reinit()
        self._updating = True
        columns = self.columns()
        for combo in (self.comboBox_z, self.comboBox_x, self.comboBox_y):
            set_combo_items(combo, columns, "")
        for dimensions in (self.zd, self.xd, self.yd):
            dimensions.set_specs(empty_dimension_specs(self.maxdim))
        self._reset_z_limits()
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

    def _reset_z_limits(self):
        self.lineEdit_zlim.setText("None")

    def _z_limits(self):
        return parse_limits(self.lineEdit_zlim.text())

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
        self._reset_z_limits()
        self._overview = True
        self._zwindow = None
        self.xd.set_specs(empty_dimension_specs(self.maxdim))
        self.yd.set_specs(empty_dimension_specs(self.maxdim))
        self.zd.set_specs(
            dimension_specs(self, self.comboBox_z.currentText(), "z"))
        self.redraw()

    def _scrolled(self):
        if not self._updating:
            self._overview = False
            self.redraw()

    def _scroll_window(self, vardim):
        """Window for the current scroll position; sizes the bars to match."""
        try:
            _group, _name, variable = resolve_selected_variable(self, vardim)
        except Exception:
            return None
        if getattr(variable, "ndim", 0) < 2:
            self.scroll.set_extent(0, 0, 0, 0)
            return None
        values = self.zd.values()
        values.extend(["0"] * max(0, variable.ndim - len(values)))
        window, shape, span = self.read_window(
            variable, values, self.scroll.offsets(), self._overview)
        if shape is None:
            self.scroll.set_extent(0, 0, 0, 0)
            return None
        self.scroll.set_extent(shape[0], shape[1], span[0], span[1])
        axes = sorted(window)
        self._zwindow = (window[axes[0]], window[axes[1]])
        return window

    def _axis_windows(self):
        """z's (row, column) windows, accounting for the transpose above."""
        if self._zwindow is None:
            return None, None
        rows_w, cols_w = self._zwindow
        if not self.checkBox_transposeZ.isChecked():
            rows_w, cols_w = cols_w, rows_w   # zz was transposed
        return rows_w, cols_w

    def _coord_window(self, vardim, rows_w, cols_w, is_x):
        """Same region/stride as z, for a 1-D or 2-D coordinate variable."""
        if rows_w is None:
            return None
        try:
            _group, _name, variable = resolve_selected_variable(self, vardim)
        except Exception:
            return None
        if getattr(variable, "ndim", 0) == 1:
            return {0: cols_w if is_x else rows_w}
        return {0: rows_w, 1: cols_w}

    def redraw(self):
        z = self.comboBox_z.currentText()
        x = self.comboBox_x.currentText()
        y = self.comboBox_y.currentText()
        zmin, zmax = self._z_limits()

        if not z:
            self.image.clear()
            self._zz = None
            return

        zz, zlabel, _zdate = self._series(z, self.zd,
                                          window=self._scroll_window(z))
        if not self.checkBox_transposeZ.isChecked():
            zz = zz.T
        if zz.ndim < 2:
            print(f"Contour: z ({z}) is not 2-dimensional:", zz.shape)
            return

        rows_w, cols_w = self._axis_windows()
        if x:
            xx, xlabel, self._xdate = self._series(
                x, self.xd, window=self._coord_window(x, rows_w, cols_w, True))
        else:
            xx, xlabel, self._xdate = np.arange(zz.shape[1], dtype=float), "", False
        if y:
            yy, ylabel, self._ydate = self._series(
                y, self.yd, window=self._coord_window(y, rows_w, cols_w, False))
        else:
            yy, ylabel, self._ydate = np.arange(zz.shape[0], dtype=float), "", False

        xx = xx[0, :] if xx.ndim > 1 else xx
        yy = yy[:, 0] if yy.ndim > 1 else yy
        if zz.shape != (yy.size, xx.size):
            print(f"Contour: x ({x}), y ({y}), z ({z}) shapes do not match:",
                  xx.shape, yy.shape, zz.shape)
            self.image.clear()
            self._zz = None
            return

        if zmin is not None:
            zz = np.maximum(zz, zmin)
        if zmax is not None:
            zz = np.minimum(zz, zmax)
        finite = zz[np.isfinite(zz)]
        levels = (
            zmin if zmin is not None else (finite.min() if finite.size else 0.0),
            zmax if zmax is not None else (finite.max() if finite.size else 1.0),
        )

        # ponytail: image cells are evenly spaced across the x/y extent;
        # switch to pg.PColorMeshItem if irregular grids need exact spacing.
        xedges, yedges = cell_edges(xx), cell_edges(yy)
        self.image.setImage(zz, autoLevels=False)
        self.image.setRect(QtCore.QRectF(
            xedges[0], yedges[0],
            xedges[-1] - xedges[0], yedges[-1] - yedges[0]))
        self.colorbar.setColorMap(
            self.selected_cmap_object(self.comboBox_cmap,
                                      self.checkBox_revCmap))
        self.colorbar.setLevels(low=levels[0], high=levels[1])
        self.colorbar.setLabel("right", zlabel)

        self._set_axis("bottom", self._xdate)
        self._set_axis("left", self._ydate)
        self.item.setLabel("bottom", xlabel)
        self.item.setLabel("left", ylabel)
        self.item.showGrid(x=self.checkBox_grid.isChecked(),
                           y=self.checkBox_grid.isChecked(), alpha=0.5)
        self.item.vb.invertX(self.checkBox_invX.isChecked())
        self.item.vb.invertY(self.checkBox_invY.isChecked())
        self.item.vb.autoRange(padding=0)
        self._xx, self._yy, self._zz = xx, yy, zz

    def _format_cursor(self, x, y):
        if self._zz is None:
            return ""
        return format_coord_contour(x, y, self._xx, self._yy, self._zz,
                                    self._xdate, self._ydate)


__all__ = ["ContourPanel"]
