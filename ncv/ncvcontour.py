"""Qt contour plotting panel."""
from __future__ import annotations

import numpy as np

from .dimensions import dimension_specs, empty_dimension_specs
from .ncvcommon import DimensionControlRow, PlotPanel, ScrollableView
from .ncvcommon import color_levels, cursor_label, load_ui
from .ncvcommon import native_levels, set_no_data_colors
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
        # paint at screen resolution, not the chunk's: 2.8 s -> 0.2 s on 34M cells
        self.image = pg.ImageItem(autoDownsample=True)
        self.item.addItem(self.image)
        self.colorbar = pg.ColorBarItem(interactive=False)
        self.colorbar.setImageItem(self.image, insert_in=self.item)
        self.scroll = ScrollableView(self.plot)
        self.init_view_sync()
        self._view_key = None
        self.plotLayout.addWidget(self.scroll, 1)
        cursor_label(self.plot, self.plotLayout, self._format_cursor)
        self._xx = self._yy = self._zz = None
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
        for check in (self.checkBox_revCmap, self.checkBox_grid,
                      self.checkBox_fullCoarse):
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
        self.xd.set_specs(empty_dimension_specs(self.maxdim))
        self.yd.set_specs(empty_dimension_specs(self.maxdim))
        self.zd.set_specs(
            dimension_specs(self, self.comboBox_z.currentText(), "z"))
        self.redraw()

    def redraw(self):
        z = self.comboBox_z.currentText()
        x = self.comboBox_x.currentText()
        y = self.comboBox_y.currentText()
        zmin, zmax = self._z_limits()

        if not z:
            self.image.clear()
            self._zz = None
            return

        # native=True: an integer chunk stays in its own dtype (no float copy)
        zz, zlabel, _zdate, zrange = self._series(
            z, self.zd,
            window=self.scroll_window(z, self.zd, center=self._reload_center),
            native=True)
        if not self.checkBox_transposeZ.isChecked():
            zz = zz.T
        if zz.ndim < 2:
            print(f"Contour: z ({z}) is not 2-dimensional:", zz.shape)
            return

        transposed = not self.checkBox_transposeZ.isChecked()
        # without a coordinate, label the axes with the chunk's real indices
        rows_w = cols_w = None
        if self._zwindow is not None:
            rows_w, cols_w = self._zwindow[::-1] if transposed else self._zwindow
        if x:
            xx, xlabel, self._xdate, _ = self._series(
                x, self.xd, window=self.coord_window(x, True, transposed))
        else:
            xx = (np.arange(*cols_w, dtype=float) if cols_w
                  else np.arange(zz.shape[1], dtype=float))
            xlabel, self._xdate = "", False
        if y:
            yy, ylabel, self._ydate, _ = self._series(
                y, self.yd, window=self.coord_window(y, False, transposed))
        else:
            yy = (np.arange(*rows_w, dtype=float) if rows_w
                  else np.arange(zz.shape[0], dtype=float))
            ylabel, self._ydate = "", False

        xx = xx[0, :] if xx.ndim > 1 else xx
        yy = yy[:, 0] if yy.ndim > 1 else yy
        if zz.shape != (yy.size, xx.size):
            print(f"Contour: x ({x}), y ({y}), z ({z}) shapes do not match:",
                  xx.shape, yy.shape, zz.shape)
            self.image.clear()
            self._zz = None
            return

        # no clamping: the image levels already saturate out-of-range values
        levels = (color_levels(zz, zmin, zmax) if zrange is None
                  else native_levels(zz, zrange, zmin, zmax))

        # ponytail: image cells are evenly spaced across the x/y extent;
        # switch to pg.PColorMeshItem if irregular grids need exact spacing.
        xedges, yedges = cell_edges(xx), cell_edges(yy)
        self.image.setImage(zz, autoLevels=False)
        rect = QtCore.QRectF(xedges[0], yedges[0],
                             xedges[-1] - xedges[0], yedges[-1] - yedges[0])
        self.image.setRect(rect)
        self.set_view_geometry(rect, transposed)
        cmap = self.selected_cmap_object(self.comboBox_cmap,
                                         self.checkBox_revCmap)
        self.colorbar.setColorMap(cmap)
        self.colorbar.setLevels(low=levels[0], high=levels[1])
        if zrange is not None:
            # after the colorbar, which pushes its own colour table onto the
            # image: missing cells (dtype minimum) get the transparent entry
            set_no_data_colors(self.image, cmap, *levels)
        self.colorbar.setLabel("right", zlabel)

        self._set_axis("bottom", self._xdate)
        self._set_axis("left", self._ydate)
        self.item.setLabel("bottom", xlabel)
        self.item.setLabel("left", ylabel)
        self.item.showGrid(x=self.checkBox_grid.isChecked(),
                           y=self.checkBox_grid.isChecked(), alpha=0.5)
        self.item.vb.invertX(self.checkBox_invX.isChecked())
        self.item.vb.invertY(self.checkBox_invY.isChecked())
        self._xx, self._yy, self._zz = xx, yy, zz
        # reset the view only when *what* is shown changes; a chunk reload or a
        # colour/grid tweak keeps the user's view and just re-syncs the bars
        key = (z, x, y, transposed, self.checkBox_fullCoarse.isChecked(),
               tuple(self.zd.values()))
        if key != self._view_key:
            self._view_key = key
            self.initial_view()
        else:
            self._view_moved()

    def _format_cursor(self, x, y):
        if self._zz is None:
            return ""
        return format_coord_contour(x, y, self._xx, self._yy, self._zz,
                                    self._xdate, self._ydate)


__all__ = ["ContourPanel"]
