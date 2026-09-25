"""Qt scatter and line plotting panel."""
from __future__ import annotations

import ast

import numpy as np

from .dimensions import dimension_specs, empty_dimension_specs
from .ncvcommon import DimensionControlRow, PlotPanel, cursor_label, load_ui
from .ncvcommon import parse_limits, set_combo_items
from .ncvutils import datetime_str
from .qt_compat import QtCore, QtWidgets, pg


# matplotlib style strings kept in the .ui, mapped onto Qt/pyqtgraph
PEN_STYLES = {
    "-": QtCore.Qt.PenStyle.SolidLine,
    "--": QtCore.Qt.PenStyle.DashLine,
    "-.": QtCore.Qt.PenStyle.DashDotLine,
    ":": QtCore.Qt.PenStyle.DotLine,
}
SYMBOLS = {
    "o": "o", "s": "s", "d": "d", "D": "d", "p": "p", "h": "h",
    "+": "+", "x": "x", "*": "star",
    "^": "t1", "v": "t", "<": "t2", ">": "t3",
}
# PlotDataItem.setData(**style) with no x/y is a DATA call and blanks the curve;
# these setters change style in place.
SETTERS = {
    "pen": "setPen",
    "symbol": "setSymbol",
    "symbolSize": "setSymbolSize",
    "symbolBrush": "setSymbolBrush",
    "symbolPen": "setSymbolPen",
}


def _maybe_color(value: str):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    return parsed if isinstance(parsed, tuple) else value


def _float_or(value, default=1.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ScatterPanel(PlotPanel):
    """Scatter and line plot tab."""

    def __init__(self, window, session):
        super().__init__(window, session, "Scatter/Line")
        self.curves_y = []
        self.curves_y2 = []
        self._build_ui()
        self.reinit()

    def _build_ui(self):
        load_ui("scatter_panel", self)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.item = self.plot.plotItem
        self.item.showAxis("right")
        # second y-axis: a view box sharing the x range (pyqtgraph's twinx)
        self.vb2 = pg.ViewBox()
        self.item.scene().addItem(self.vb2)
        self.item.getAxis("right").linkToView(self.vb2)
        self.vb2.setXLink(self.item)
        self.item.vb.sigResized.connect(
            lambda: self.vb2.setGeometry(self.item.vb.sceneBoundingRect()))
        self.plotLayout.addWidget(self.plot, 1)
        cursor_label(self.plot, self.label_cursor, self._format_cursor)

        self.xd = DimensionControlRow(self.maxdim)
        self.yd = DimensionControlRow(self.maxdim)
        self.y2d = DimensionControlRow(self.maxdim)
        self.xDimensionsLayout.addWidget(self.xd)
        self.yDimensionsLayout.addWidget(self.yd)
        self.y2DimensionsLayout.addWidget(self.y2d)

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
        self._xdate = self._ydate = False
        self._updating = False

    # ---------------------------------------------------------------- events

    def checked_x(self):
        if not self._updating:
            self._apply_limits()

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

    # ----------------------------------------------------------------- style

    def _style(self, suffix):
        """Pen and symbol options from the Y1/Y2 style entries."""
        text = lambda name: getattr(self, f"lineEdit_{name}{suffix}").text()
        style = PEN_STYLES.get(text("lineStyle"))
        color = _maybe_color(text("lineColor"))
        pen = None
        if style is not None and color != "None":
            pen = pg.mkPen(color=color, width=_float_or(text("lineWidth")),
                           style=style)
        symbol = SYMBOLS.get(text("markerStyle"))
        options = {"pen": pen, "symbol": symbol}
        if symbol is not None:
            fill = _maybe_color(text("markerFillColor"))
            edge = _maybe_color(text("markerEdgeColor"))
            options["symbolSize"] = _float_or(text("markerSize"))
            options["symbolBrush"] = (
                None if fill == "None" else pg.mkBrush(fill))
            options["symbolPen"] = (
                None if edge == "None" else
                pg.mkPen(color=edge,
                         width=_float_or(text("markerEdgeWidth"))))
        return options, color

    def _restyle(self, curves, suffix, axis):
        options, color = self._style(suffix)
        for curve in curves:
            for key, value in options.items():
                getattr(curve, SETTERS[key])(value)
        if color != "None":
            self.item.getAxis(axis).setPen(pg.mkPen(color))
            self.item.getAxis(axis).setTextPen(pg.mkPen(color))

    # ------------------------------------------------------------- redrawing

    def redraw_y(self):
        if not self.comboBox_y.currentText():
            return
        self._restyle(self.curves_y, "Y1", "left")
        self._apply_limits()

    def redraw_y2(self):
        if not self.comboBox_y2.currentText():
            return
        self._restyle(self.curves_y2, "Y2", "right")
        self._apply_limits()

    def _range(self, text, view, invert, axis):
        """Apply one axis limit entry; auto-range when it is None."""
        low, high = parse_limits(text)
        if low is None or high is None:
            # enableAutoRange only rescales when the flag CHANGES, and pyqtgraph
            # defaults it to on - so ask for the rescale explicitly.
            view.enableAutoRange(axis=axis)
            view.updateAutoRange()
        else:
            view.setRange(**{f"{axis}Range": sorted((low, high))}, padding=0)
        setter = view.invertX if axis == "x" else view.invertY
        setter(bool(invert))

    def _apply_limits(self):
        same_y = (self.checkBox_sameYaxis.isChecked()
                  and self.comboBox_y.currentText()
                  and self.comboBox_y2.currentText())
        self._range(self.lineEdit_xlim.text(), self.item.vb,
                    self.checkBox_invX.isChecked(), "x")
        if same_y:
            # union of both y ranges on both axes
            bounds = [v for view in (self.item.vb, self.vb2)
                      for v in view.childrenBounds()[1] or []]
            if bounds:
                span = (min(bounds), max(bounds))
                self.item.vb.setYRange(*span, padding=0)
                self.vb2.setYRange(*span, padding=0)
        else:
            self._range(self.lineEdit_ylim.text(), self.item.vb, False, "y")
            self._range(self.lineEdit_y2lim.text(), self.vb2, False, "y")
        self.item.vb.invertY(self.checkBox_invY.isChecked())
        self.vb2.invertY(self.checkBox_invY2.isChecked())

    def _add_curves(self, view, xx, yy, suffix):
        options, _color = self._style(suffix)
        columns = yy.T if yy.ndim > 1 else [yy]
        curves = []
        for column in columns:
            curve = pg.PlotDataItem(xx, column, **options)
            view.addItem(curve)
            curves.append(curve)
        return curves

    def redraw(self):
        x = self.comboBox_x.currentText()
        y = self.comboBox_y.currentText()
        y2 = self.comboBox_y2.currentText()
        self.item.clear()
        self.vb2.clear()
        self.curves_y = []
        self.curves_y2 = []
        if not (y or y2):
            return

        yy = yy2 = None
        ylabel = ylabel2 = ""
        if y:
            yy, ylabel, self._ydate, _ = self._series(y, self.yd)
        if y2:
            yy2, ylabel2, _y2date, _ = self._series(y2, self.y2d)
        if x:
            xx, xlabel, self._xdate, _ = self._series(x, self.xd)
        else:
            xx = np.arange((yy if y else yy2).shape[0], dtype=float)
            xlabel, self._xdate = "", False

        self._set_axis("bottom", self._xdate)
        self._set_axis("left", self._ydate)
        self.item.setLabel("bottom", xlabel)
        self.item.setLabel("left", ylabel)
        self.item.setLabel("right", ylabel2)

        for name, values, view, suffix, target in (
            (y, yy, self.item, "Y1", "curves_y"),
            (y2, yy2, self.vb2, "Y2", "curves_y2"),
        ):
            if not name:
                continue
            if values.shape[0] != xx.shape[0]:
                print(f"Scatter: x and {name} shapes do not match:",
                      xx.shape, values.shape)
                continue
            setattr(self, target, self._add_curves(view, xx, values, suffix))

        self.redraw_y()
        self.redraw_y2()
        self._apply_limits()

    def _format_cursor(self, x, y):
        xstr = datetime_str(x) if self._xdate else f"{x:.6g}"
        ystr = datetime_str(y) if self._ydate else f"{y:.6g}"
        return f"x={xstr}, y={ystr}"


__all__ = ["ScatterPanel"]
