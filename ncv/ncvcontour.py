"""Qt contour plotting panel."""
from __future__ import annotations

from matplotlib.figure import Figure
import numpy as np

from .dimensions import dimension_specs, empty_dimension_specs
from .ncvcommon import DimensionControlRow, PlotPanel
from .ncvcommon import float_or_none, set_combo_items
from .ncvutils import format_coord_contour, parse_entry, selvar
from .ncvutils import set_axis_label, vardim2var
from .pyui.ui_contour_panel import Ui_ContourPanel
from .qt_compat import FigureCanvasQTAgg, NavigationToolbar2QT
from .qt_compat import QtWidgets


class ContourPanel(PlotPanel, Ui_ContourPanel):
    """Contour plot tab."""

    def __init__(self, window, session):
        super().__init__(window, session, "Contour")
        self._build_ui()
        self.reinit()

    def _build_ui(self):
        self.setupUi(self)
        self.connect_file_controls()
        self.figure = Figure(facecolor="white", figsize=(1, 1))
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plotLayout.addWidget(self.canvas, 1)
        self.plotLayout.addWidget(self.toolbar)

        self.zd = DimensionControlRow(self.maxdim)
        self.xd = DimensionControlRow(self.maxdim)
        self.yd = DimensionControlRow(self.maxdim)
        self.zDimensionsLayout.addWidget(self.zd)
        self.xDimensionsLayout.addWidget(self.xd)
        self.yDimensionsLayout.addWidget(self.yd)
        self.populate_cmap_combo(self.comboBox_cmap)

        for name, callback in (
            ("pushButton_prevZ", self.prev_z),
            ("pushButton_nextZ", self.next_z),
        ):
            button = getattr(self, name, None)
            if button is not None:
                button.clicked.connect(callback)
        self.comboBox_z.currentIndexChanged.connect(self.selected_z)
        self.checkBox_transZ.stateChanged.connect(self.checked)
        for name in ("lineEdit_zlim", "lineEdit_zmin", "lineEdit_zmax"):
            entry = getattr(self, name, None)
            if entry is not None:
                entry.editingFinished.connect(self.entered_z)
        self.zd.changed.connect(self.spinned_z)
        self.comboBox_x.currentIndexChanged.connect(self.selected_x)
        self.checkBox_invX.stateChanged.connect(self.checked)
        self.comboBox_y.currentIndexChanged.connect(self.selected_y)
        self.checkBox_invY.stateChanged.connect(self.checked)
        self.xd.changed.connect(self.spinned_x)
        self.yd.changed.connect(self.spinned_y)
        self.comboBox_cmap.currentIndexChanged.connect(self.selected_cmap)
        for check in (
            self.checkBox_revCmap,
            self.checkBox_mesh,
            self.checkBox_grid,
        ):
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

    def _move_z(self, step):
        index = self.comboBox_z.currentIndex() + step
        if 0 < index < self.comboBox_z.count():
            self.comboBox_z.setCurrentIndex(index)

    def next_z(self):
        self._move_z(1)

    def prev_z(self):
        self._move_z(-1)

    def _reset_z_limits(self):
        for name in ("lineEdit_zlim", "lineEdit_zmin", "lineEdit_zmax"):
            entry = getattr(self, name, None)
            if entry is not None:
                entry.setText("None")

    def _z_limits(self):
        entry = getattr(self, "lineEdit_zlim", None)
        if entry is not None:
            limits = parse_entry(entry.text())
            if isinstance(limits, (list, tuple)) and len(limits) == 2:
                return limits
            return None, None
        return (
            float_or_none(self.lineEdit_zmin.text()),
            float_or_none(self.lineEdit_zmax.text()),
        )

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
        trans_z = self.checkBox_transZ.isChecked()
        zmin, zmax = self._z_limits()
        x = self.comboBox_x.currentText()
        y = self.comboBox_y.currentText()
        inv_x = self.checkBox_invX.isChecked()
        inv_y = self.checkBox_invY.isChecked()
        cmap = self.comboBox_cmap.currentText()
        if self.checkBox_revCmap.isChecked():
            cmap += "_r"
        mesh = self.checkBox_mesh.isChecked()
        grid = self.checkBox_grid.isChecked()
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        vx = vy = vz = "None"
        if z:
            group_z, vz = vardim2var(z, self.groups)
            tname = self.tname if self.usex else self.tname[group_z]
            if vz == tname:
                zz = self.time_values(group_z, decimal=mesh)
                zlabel = "Year" if mesh else "Date"
            else:
                zz = selvar(self, vz)
                zlabel = set_axis_label(zz)
            zz = self.slice_miss(self.zd, zz)
            if not trans_z:
                zz = zz.T
        else:
            zlabel = ""
        if y:
            group_y, vy = vardim2var(y, self.groups)
            tname = self.tname if self.usex else self.tname[group_y]
            if vy == tname:
                yy = self.time_values(group_y, decimal=mesh)
                ylabel = "Year" if mesh else "Date"
            else:
                yy = selvar(self, vy)
                ylabel = set_axis_label(yy)
            yy = self.slice_miss(self.yd, yy)
        else:
            ylabel = ""
        if x:
            group_x, vx = vardim2var(x, self.groups)
            tname = self.tname if self.usex else self.tname[group_x]
            if vx == tname:
                xx = self.time_values(group_x, decimal=mesh)
                xlabel = "Year" if mesh else "Date"
            else:
                xx = selvar(self, vx)
                xlabel = set_axis_label(xx)
            xx = self.slice_miss(self.xd, xx)
        else:
            xlabel = ""
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
                contour = self.axes.pcolormesh(
                    xx, yy, zz, vmin=zmin, vmax=zmax,
                    cmap=cmap, shading="nearest")
                colorbar = self.figure.colorbar(
                    contour, fraction=0.05, shrink=0.75, extend=extend)
            else:
                contour = self.axes.contourf(
                    xx, yy, zz, vmin=zmin, vmax=zmax,
                    cmap=cmap, extend=extend)
                colorbar = self.figure.colorbar(
                    contour, fraction=0.05, shrink=0.75)
        except Exception:
            print(
                f"Contour: x ({vx}), y ({vy}), z ({vz}) shapes do not match:",
                xx.shape, yy.shape, zz.shape)
            return
        colorbar.set_label(zlabel)
        self.axes.xaxis.set_label_text(xlabel)
        self.axes.yaxis.set_label_text(ylabel)
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


__all__ = ["ContourPanel"]
