"""Qt scatter and line plotting panel."""
from __future__ import annotations

import ast

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .dimensions import dimension_specs, empty_dimension_specs
from .ncvcommon import DimensionControlRow, PlotPanel, set_combo_items
from .ncvutils import format_coord_scatter, parse_entry
from .ncvutils import selvar, set_axis_label, vardim2var
from .pyui.ui_scatter_panel import Ui_ScatterPanel
from .qt_compat import FigureCanvasQTAgg, NavigationToolbar2QT
from .qt_compat import QtWidgets


def _maybe_color(value: str):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    return parsed if isinstance(parsed, tuple) else value


def _minmax_ylim(ylim, ylim2):
    ymin = None
    ymax = None
    if isinstance(ylim, (list, tuple)) and isinstance(ylim2, (list, tuple)):
        values = list(ylim) + list(ylim2)
        if all(value is not None for value in values):
            ymin = min(values)
            ymax = max(values)
    return ymin, ymax


class ScatterPanel(PlotPanel, Ui_ScatterPanel):
    """Scatter and line plot tab."""

    def __init__(self, window, session):
        super().__init__(window, session, "Scatter/Line")
        self.line_y = []
        self.line_y2 = []
        self._build_ui()
        self.reinit()

    def _build_ui(self):
        self.setupUi(self)
        self.connect_file_controls()
        self.figure = Figure(facecolor="white", figsize=(1, 1))
        self.axes = self.figure.add_subplot(111)
        self.axes2 = self.axes.twinx()
        self.axes2.yaxis.set_label_position("right")
        self.axes2.yaxis.tick_right()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plotLayout.addWidget(self.canvas, 1)
        self.plotLayout.addWidget(self.toolbar)

        self.xd = DimensionControlRow(self.maxdim)
        self.yd = DimensionControlRow(self.maxdim)
        self.y2d = DimensionControlRow(self.maxdim)
        self.xDimensionsLayout.addWidget(self.xd)
        self.yDimensionsLayout.addWidget(self.yd)
        self.y2DimensionsLayout.addWidget(self.y2d)

        colors = list(plt.rcParams["axes.prop_cycle"])
        col1 = colors[0]["color"]
        col2 = colors[3]["color"]
        for entry in (
            self.lineEdit_lineColorY1,
            self.lineEdit_markerFillColorY1,
            self.lineEdit_markerEdgeColorY1,
        ):
            entry.setText(col1)
        for entry in (
            self.lineEdit_lineColorY2,
            self.lineEdit_markerFillColorY2,
            self.lineEdit_markerEdgeColorY2,
        ):
            entry.setText(col2)

        self.comboBox_x.currentIndexChanged.connect(self.selected_x)
        self.checkBox_invX.stateChanged.connect(self.checked_x)
        self.comboBox_y.currentIndexChanged.connect(self.selected_y)
        self.checkBox_invY.stateChanged.connect(self.checked_y)
        self.pushButton_redraw.clicked.connect(self.redraw)
        self.xd.changed.connect(self.spinned_x)
        self.yd.changed.connect(self.spinned_y)
        for entry in (
            self.lineEdit_lineStyleY1,
            self.lineEdit_lineWidthY1,
            self.lineEdit_lineColorY1,
            self.lineEdit_markerStyleY1,
            self.lineEdit_markerSizeY1,
            self.lineEdit_markerFillColorY1,
            self.lineEdit_markerEdgeColorY1,
            self.lineEdit_markerEdgeWidthY1,
            self.lineEdit_xlim,
            self.lineEdit_ylim,
        ):
            entry.editingFinished.connect(self.entered_y)
        self.comboBox_y2.currentIndexChanged.connect(self.selected_y2)
        self.checkBox_invY2.stateChanged.connect(self.checked_y2)
        self.checkBox_sameYaxis.stateChanged.connect(self.checked_yy2)
        self.y2d.changed.connect(self.spinned_y2)
        for entry in (
            self.lineEdit_lineStyleY2,
            self.lineEdit_lineWidthY2,
            self.lineEdit_lineColorY2,
            self.lineEdit_markerStyleY2,
            self.lineEdit_markerSizeY2,
            self.lineEdit_markerFillColorY2,
            self.lineEdit_markerEdgeColorY2,
            self.lineEdit_markerEdgeWidthY2,
            self.lineEdit_y2lim,
        ):
            entry.editingFinished.connect(self.entered_y2)
        self.pushButton_quit.clicked.connect(QtWidgets.QApplication.quit)

    def reinit(self):
        super().reinit()
        self._updating = True
        columns = self.columns()
        for combo in (self.comboBox_x, self.comboBox_y, self.comboBox_y2):
            set_combo_items(combo, columns, "")
        for dimensions in (self.xd, self.yd, self.y2d):
            dimensions.set_specs(empty_dimension_specs(self.maxdim))
        self.lineEdit_xlim.setText("None")
        self.lineEdit_ylim.setText("None")
        self.lineEdit_y2lim.setText("None")
        self._updating = False

    def checked_x(self):
        if not self._updating:
            self.redraw_y()
            self.redraw_y2()

    def checked_y(self):
        if not self._updating:
            self.redraw_y()

    def checked_y2(self):
        if not self._updating:
            self.redraw_y2()

    def checked_yy2(self):
        if not self._updating:
            self.lineEdit_ylim.setText("None")
            self.lineEdit_y2lim.setText("None")
            self.redraw_y()
            self.redraw_y2()

    def entered_y(self):
        if not self._updating:
            self.redraw_y()

    def entered_y2(self):
        if not self._updating:
            self.redraw_y2()

    def selected_x(self):
        if self._updating:
            return
        self.xd.set_specs(
            dimension_specs(self, self.comboBox_x.currentText(), "x"))
        self.lineEdit_xlim.setText("None")
        self.redraw()

    def selected_y(self):
        if self._updating:
            return
        self.yd.set_specs(
            dimension_specs(self, self.comboBox_y.currentText(), "y"))
        self.lineEdit_ylim.setText("None")
        self.redraw()

    def selected_y2(self):
        if self._updating:
            return
        self.y2d.set_specs(
            dimension_specs(self, self.comboBox_y2.currentText(), "y2"))
        self.lineEdit_y2lim.setText("None")
        self.redraw()

    def spinned_x(self):
        if not self._updating:
            self.redraw()

    def spinned_y(self):
        if not self._updating:
            self.redraw()

    def spinned_y2(self):
        if not self._updating:
            self.redraw()

    def redraw_y(self):
        y = self.comboBox_y.currentText()
        if not y:
            return
        inv_y = self.checkBox_invY.isChecked()
        ylim = parse_entry(self.lineEdit_ylim.text())
        ylim2 = parse_entry(self.lineEdit_y2lim.text())
        line_style = self.lineEdit_lineStyleY1.text()
        line_width = float(self.lineEdit_lineWidthY1.text())
        color = _maybe_color(self.lineEdit_lineColorY1.text())
        marker = self.lineEdit_markerStyleY1.text()
        marker_size = float(self.lineEdit_markerSizeY1.text())
        marker_fill = _maybe_color(self.lineEdit_markerFillColorY1.text())
        marker_edge = _maybe_color(self.lineEdit_markerEdgeColorY1.text())
        marker_edge_width = float(self.lineEdit_markerEdgeWidthY1.text())
        y2 = self.comboBox_y2.currentText()
        same_y = self.checkBox_sameYaxis.isChecked()
        plot_args = {
            "linestyle": line_style,
            "linewidth": line_width,
            "marker": marker,
            "markersize": marker_size,
            "markerfacecolor": marker_fill,
            "markeredgecolor": marker_edge,
            "markeredgewidth": marker_edge_width,
        }
        group, variable = vardim2var(y, self.groups)
        tname = self.tname if self.usex else self.tname[group]
        if variable == tname:
            ylabel = "Date"
            plot_args["color"] = color
        else:
            ylabel = set_axis_label(selvar(self, variable))
            if len(self.line_y) == 1:
                plot_args["color"] = color
        for line in self.line_y:
            plt.setp(line, **plot_args)
        if "color" in plot_args and plot_args["color"] != "None":
            self.axes.spines["left"].set_color(plot_args["color"])
            self.axes.tick_params(axis="y", colors=plot_args["color"])
            self.axes.yaxis.label.set_color(plot_args["color"])
        self.axes.yaxis.set_label_text(ylabel)
        if not isinstance(ylim, list):
            ylim = self.axes.get_ylim()
        if not isinstance(ylim2, list):
            ylim2 = self.axes2.get_ylim()
        if same_y and y2:
            ymin, ymax = _minmax_ylim(ylim, ylim2)
            if ymin is not None and ymax is not None:
                ylim = [ymin, ymax]
                ylim2 = [ymin, ymax]
            self.axes.set_ylim(ylim)
            self.axes2.set_ylim(ylim2)
        if inv_y and ylim[0] is not None:
            if ylim[0] < ylim[1]:
                ylim = ylim[::-1]
            self.axes.set_ylim(ylim)
        else:
            if ylim[1] < ylim[0]:
                ylim = ylim[::-1]
            self.axes.set_ylim(ylim)
        self._apply_x_limits()
        self.canvas.draw()
        self.toolbar.update()

    def redraw_y2(self):
        y2 = self.comboBox_y2.currentText()
        if not y2:
            return
        y = self.comboBox_y.currentText()
        inv_y2 = self.checkBox_invY2.isChecked()
        same_y = self.checkBox_sameYaxis.isChecked()
        ylim = parse_entry(self.lineEdit_ylim.text())
        ylim2 = parse_entry(self.lineEdit_y2lim.text())
        plot_args = {
            "linestyle": self.lineEdit_lineStyleY2.text(),
            "linewidth": float(self.lineEdit_lineWidthY2.text()),
            "marker": self.lineEdit_markerStyleY2.text(),
            "markersize": float(self.lineEdit_markerSizeY2.text()),
            "markerfacecolor": _maybe_color(
                self.lineEdit_markerFillColorY2.text()),
            "markeredgecolor": _maybe_color(
                self.lineEdit_markerEdgeColorY2.text()),
            "markeredgewidth": float(
                self.lineEdit_markerEdgeWidthY2.text()),
        }
        color = _maybe_color(self.lineEdit_lineColorY2.text())
        group, variable = vardim2var(y2, self.groups)
        tname = self.tname if self.usex else self.tname[group]
        if variable == tname:
            ylabel = "Date"
            plot_args["color"] = color
        else:
            ylabel = set_axis_label(selvar(self, variable))
            if len(self.line_y2) == 1:
                plot_args["color"] = color
        for line in self.line_y2:
            plt.setp(line, **plot_args)
        if "color" in plot_args and plot_args["color"] != "None":
            self.axes2.spines["right"].set_color(plot_args["color"])
            self.axes2.tick_params(axis="y", colors=plot_args["color"])
            self.axes2.yaxis.label.set_color(plot_args["color"])
        self.axes2.yaxis.set_label_text(ylabel)
        if not isinstance(ylim, list):
            ylim = self.axes.get_ylim()
        if not isinstance(ylim2, list):
            ylim2 = self.axes2.get_ylim()
        if same_y and y:
            ymin, ymax = _minmax_ylim(ylim, ylim2)
            if ymin is not None and ymax is not None:
                ylim = [ymin, ymax]
                ylim2 = [ymin, ymax]
            self.axes.set_ylim(ylim)
            self.axes2.set_ylim(ylim2)
        ylim = ylim2
        if inv_y2 and ylim[0] is not None:
            if ylim[0] < ylim[1]:
                ylim = ylim[::-1]
            self.axes2.set_ylim(ylim)
        else:
            if ylim[1] < ylim[0]:
                ylim = ylim[::-1]
            self.axes2.set_ylim(ylim)
        self._apply_x_limits()
        self.canvas.draw()
        self.toolbar.update()

    def _apply_x_limits(self):
        inv_x = self.checkBox_invX.isChecked()
        xlim = parse_entry(self.lineEdit_xlim.text())
        if not isinstance(xlim, list):
            xlim = self.axes.get_xlim()
        if inv_x and xlim[0] is not None:
            if xlim[0] < xlim[1]:
                xlim = xlim[::-1]
            self.axes.set_xlim(xlim)
        else:
            if xlim[1] < xlim[0]:
                xlim = xlim[::-1]
            self.axes.set_xlim(xlim)

    def redraw(self):
        x = self.comboBox_x.currentText()
        y = self.comboBox_y.currentText()
        y2 = self.comboBox_y2.currentText()
        self.axes.clear()
        self.axes2.clear()
        self.axes2.yaxis.set_label_position("right")
        self.axes2.yaxis.tick_right()
        vx = vy = vy2 = "None"
        if y or y2:
            if y:
                group, vy = vardim2var(y, self.groups)
                tname = self.tname if self.usex else self.tname[group]
                if vy == tname:
                    yy = self.time_values(group)
                    ylabel = "Date"
                else:
                    yy = selvar(self, vy)
                    ylabel = set_axis_label(yy)
                yy = self.slice_miss(self.yd, yy)
            if y2:
                group2, vy2 = vardim2var(y2, self.groups)
                tname = self.tname if self.usex else self.tname[group2]
                if vy2 == tname:
                    yy2 = self.time_values(group2)
                    ylabel2 = "Date"
                else:
                    yy2 = selvar(self, vy2)
                    ylabel2 = set_axis_label(yy2)
                yy2 = self.slice_miss(self.y2d, yy2)
            if x:
                group_x, vx = vardim2var(x, self.groups)
                tname = self.tname if self.usex else self.tname[group_x]
                if vx == tname:
                    xx = self.time_values(group_x)
                    xlabel = "Date"
                else:
                    xx = selvar(self, vx)
                    xlabel = set_axis_label(xx)
                xx = self.slice_miss(self.xd, xx)
            else:
                nx = yy.shape[0] if y else yy2.shape[0]
                xx = np.arange(nx)
                xlabel = ""
            if not y:
                yy = np.ones_like(xx, dtype="float") * np.nan
                ylabel = ""
            if not y2:
                yy2 = np.ones_like(xx, dtype="float") * np.nan
                ylabel2 = ""
            try:
                self.line_y = self.axes.plot(xx, yy)
            except Exception:
                print(
                    f"Scatter: x ({vx}) and y ({vy}) shapes do not match:",
                    xx.shape, yy.shape)
                return
            try:
                self.line_y2 = self.axes2.plot(xx, yy2)
            except Exception:
                print(
                    f"Scatter: x ({vx}) and y2 ({vy2}) shapes do not match:",
                    xx.shape, yy2.shape)
                return
            self.axes.xaxis.set_label_text(xlabel)
            self.axes.yaxis.set_label_text(ylabel)
            self.axes2.xaxis.set_label_text(xlabel)
            self.axes2.yaxis.set_label_text(ylabel2)
            self.axes2.format_coord = lambda x0, y0: format_coord_scatter(
                x0, y0, self.axes, self.axes2,
                xx.dtype, yy.dtype, yy2.dtype)
            self.redraw_y()
            self.redraw_y2()
        self.canvas.draw()
        self.toolbar.update()


__all__ = ["ScatterPanel"]
