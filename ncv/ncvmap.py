"""Qt map panels with optional Cartopy support."""
from __future__ import annotations

import sys

import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np

from .dimensions import dimension_specs, empty_dimension_specs
from .ncvcommon import (
    DimensionControlRow,
    PlotPanel,
    TimeControlMixin,
    float_or_none,
    set_combo_items,
)
from .ncvmethods import get_miss
from .ncvutils import (
    add_cyclic,
    format_coord_map,
    selvar,
    set_axis_label,
    set_miss,
    vardim2var,
)
from .pyui.ui_map_panel import Ui_MapPanel
from .pyui.ui_map_unavailable import Ui_MapUnavailablePanel
from .qt_compat import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
    QtWidgets,
)


__all__ = [
    "CARTOPY_IMPORT_ERROR",
    "HAVE_CARTOPY",
    "MapPanel",
    "MapUnavailablePanel",
    "ensure_cartopy",
]


def _import_cartopy():
    try:
        import cartopy.crs as imported_ccrs
        import cartopy.feature as imported_cfeature
    except Exception as exc:  # pragma: no cover - depends on environment
        return None, None, exc
    return imported_ccrs, imported_cfeature, None


ccrs, cfeature, CARTOPY_IMPORT_ERROR = _import_cartopy()
HAVE_CARTOPY = CARTOPY_IMPORT_ERROR is None


def ensure_cartopy():
    """Retry Cartopy import and update module-level availability state."""
    global ccrs, cfeature, CARTOPY_IMPORT_ERROR, HAVE_CARTOPY

    if HAVE_CARTOPY:
        return True
    ccrs, cfeature, CARTOPY_IMPORT_ERROR = _import_cartopy()
    HAVE_CARTOPY = CARTOPY_IMPORT_ERROR is None
    return HAVE_CARTOPY


class MapUnavailablePanel(QtWidgets.QWidget, Ui_MapUnavailablePanel):
    def __init__(self, error=None):
        super().__init__()
        self.setupUi(self)
        message = "Map view is unavailable because Cartopy could not be imported."
        if error is not None:
            message += f"\n\n{type(error).__name__}: {error}"
        message += f"\n\nPython executable: {sys.executable}"
        self.label_message.setText(message)

    def reinit(self):
        pass

    def redraw(self):
        pass


class MapPanel(TimeControlMixin, PlotPanel, Ui_MapPanel):
    def __init__(self, window, session):
        if not ensure_cartopy():
            raise RuntimeError("Cartopy is required for MapPanel") from CARTOPY_IMPORT_ERROR
        super().__init__(window, session, "Map")
        self._updating = True
        self._build_ui()
        self._updating = False
        self.reinit()

    def _build_ui(self):
        self.setupUi(self)
        self.connect_file_controls()
        self.figure = Figure(facecolor="white", figsize=(1, 1))
        self.axes = self.figure.add_subplot(111, projection=ccrs.PlateCarree())
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plotLayout.addWidget(self.canvas, 1)
        self.plotLayout.addWidget(self.toolbar)

        self.vd = DimensionControlRow(self.maxdim)
        self.lond = DimensionControlRow(self.maxdim)
        self.latd = DimensionControlRow(self.maxdim)
        self.vDimensionsLayout.addWidget(self.vd)
        self.lonDimensionsLayout.addWidget(self.lond)
        self.latDimensionsLayout.addWidget(self.latd)
        self.init_time_controls(self.comboBox_variable, self.vd)
        self.populate_cmap_combo(self.comboBox_cmap)

        self.projs = ["AlbersEqualArea", "AzimuthalEquidistant", "EckertI",
                      "EckertII", "EckertIII", "EckertIV", "EckertV",
                      "EckertVI", "EqualEarth", "EquidistantConic",
                      "InterruptedGoodeHomolosine",
                      "LambertAzimuthalEqualArea", "LambertConformal",
                      "LambertCylindrical", "Mercator", "Miller", "Mollweide",
                      "NorthPolarStereo", "PlateCarree", "Robinson",
                      "Sinusoidal", "SouthPolarStereo", "Stereographic"]
        self.iprojs = [ccrs.AlbersEqualArea, ccrs.AzimuthalEquidistant,
                       ccrs.EckertI, ccrs.EckertII, ccrs.EckertIII,
                       ccrs.EckertIV, ccrs.EckertV, ccrs.EckertVI,
                       ccrs.EqualEarth, ccrs.EquidistantConic,
                       ccrs.InterruptedGoodeHomolosine,
                       ccrs.LambertAzimuthalEqualArea, ccrs.LambertConformal,
                       ccrs.LambertCylindrical, ccrs.Mercator, ccrs.Miller,
                       ccrs.Mollweide, ccrs.NorthPolarStereo, ccrs.PlateCarree,
                       ccrs.Robinson, ccrs.Sinusoidal, ccrs.SouthPolarStereo,
                       ccrs.Stereographic]
        self.comboBox_projection.clear()
        self.comboBox_projection.addItems(self.projs)
        self.comboBox_projection.setCurrentText("PlateCarree")

        self.comboBox_variable.currentIndexChanged.connect(self.selected_v)
        self.checkBox_transVariable.stateChanged.connect(self.checked)
        self.lineEdit_min.editingFinished.connect(self.entered_v)
        self.lineEdit_max.editingFinished.connect(self.entered_v)
        self.checkBox_allValues.stateChanged.connect(self.checked_all)
        self.vd.changed.connect(self.spinned_v)
        self.comboBox_longitude.currentIndexChanged.connect(self.selected_lon)
        self.checkBox_invLongitude.stateChanged.connect(self.checked)
        self.checkBox_shiftLongitude.stateChanged.connect(self.checked)
        self.comboBox_latitude.currentIndexChanged.connect(self.selected_lat)
        self.checkBox_invLatitude.stateChanged.connect(self.checked)
        self.lond.changed.connect(self.spinned_lon)
        self.latd.changed.connect(self.spinned_lat)
        self.comboBox_cmap.currentIndexChanged.connect(self.selected_cmap)
        for check in (self.checkBox_revCmap, self.checkBox_mesh,
                      self.checkBox_global, self.checkBox_coast,
                      self.checkBox_borders, self.checkBox_rivers,
                      self.checkBox_lakes, self.checkBox_grid):
            check.stateChanged.connect(self.checked)
        self.comboBox_projection.currentIndexChanged.connect(self.selected_proj)
        self.lineEdit_centralLon.editingFinished.connect(self.entered_clon)
        self.pushButton_quit.clicked.connect(QtWidgets.QApplication.quit)

    def reinit(self):
        super().reinit()
        self._updating = True
        self.iunlim = -1
        self.nunlim = 0
        columns = self.columns()
        for combo in (self.comboBox_variable, self.comboBox_longitude,
                      self.comboBox_latitude):
            set_combo_items(combo, columns, "")
        for dims in (self.vd, self.lond, self.latd):
            dims.set_specs(empty_dimension_specs(self.maxdim))
        self.lineEdit_min.setText("None")
        self.lineEdit_max.setText("None")
        self.horizontalSlider_timeStep.setRange(0, 0)
        self.horizontalSlider_timeStep.setValue(0)
        self.comboBox_repeat.setCurrentText("repeat")
        if self.usex:
            if self.lonvar:
                self.comboBox_longitude.setCurrentText(self.lonvar)
                self.lond.set_specs(dimension_specs(
                    self, self.comboBox_longitude.currentText(), "lon"))
            if self.latvar:
                self.comboBox_latitude.setCurrentText(self.latvar)
                self.latd.set_specs(dimension_specs(
                    self, self.comboBox_latitude.currentText(), "lat"))
        else:
            if any(self.lonvar):
                lon = next(item for item in self.lonvar if item)
                self.comboBox_longitude.setCurrentText(lon)
                self.lond.set_specs(dimension_specs(
                    self, self.comboBox_longitude.currentText(), "lon"))
            if any(self.latvar):
                lat = next(item for item in self.latvar if item)
                self.comboBox_latitude.setCurrentText(lat)
                self.latd.set_specs(dimension_specs(
                    self, self.comboBox_latitude.currentText(), "lat"))
        self._sync_time_controls()
        self._updating = False

    def checked(self):
        if not self._updating:
            self.redraw()

    def checked_all(self):
        if self._updating:
            return
        vmin, vmax = self.get_vminmax()
        self.lineEdit_min.setText(str(vmin))
        self.lineEdit_max.setText(str(vmax))
        self.redraw()

    def entered_clon(self):
        self.checked()

    def entered_v(self):
        self.checked()

    def selected_cmap(self):
        self.checked()

    def selected_proj(self):
        self.checked()

    def selected_lat(self):
        if self._updating:
            return
        self.checkBox_invLatitude.setChecked(False)
        self.latd.set_specs(
            dimension_specs(
                self, self.comboBox_latitude.currentText(), "lat"))
        self.redraw()

    def selected_lon(self):
        if self._updating:
            return
        self.checkBox_invLongitude.setChecked(False)
        self.checkBox_shiftLongitude.setChecked(False)
        self.lond.set_specs(
            dimension_specs(
                self, self.comboBox_longitude.currentText(), "lon"))
        self.redraw()

    def selected_v(self):
        if self._updating:
            return
        v = self.comboBox_variable.currentText()
        if not v:
            self.iunlim = -1
            self.nunlim = 0
            self.vd.set_specs(empty_dimension_specs(self.maxdim))
            self._sync_time_controls()
            self.redraw()
            return
        self.vd.set_specs(dimension_specs(self, v, "var"))
        self.set_unlim(v)
        self.set_tstep(0)
        vmin, vmax = self.get_vminmax()
        self.lineEdit_min.setText(str(vmin))
        self.lineEdit_max.setText(str(vmax))
        self.redraw()

    def spinned_lon(self):
        self.checked()

    def spinned_lat(self):
        self.checked()

    def spinned_v(self):
        if self.iunlim >= 0:
            try:
                self.set_tstep(int(self.vd.values()[self.iunlim]))
            except (ValueError, IndexError):
                pass
        self.checked()

    def get_vminmax(self):
        v = self.comboBox_variable.currentText()
        if not v:
            return 0, 1
        gz, vz = vardim2var(v, self.groups)
        tname = self.tname if self.usex else self.tname[gz]
        if vz == tname:
            return 0, 1
        vv = selvar(self, vz)
        imiss = get_miss(self, vv)
        if self.checkBox_allValues.isChecked() or (np.sum(vv.shape[:-2]) < 50):
            arr = set_miss(imiss, vv)
            return np.nanmin(arr), np.nanmax(arr)
        rng = np.random.default_rng()
        vmin = np.inf
        vmax = -np.inf
        for _ in range(50):
            ss = []
            for i in range(vv.ndim):
                if i < vv.ndim - 2:
                    idim = rng.integers(0, vv.shape[i])
                    ss.append(slice(idim, idim + 1))
                else:
                    ss.append(slice(0, vv.shape[i]))
            arr = set_miss(imiss, vv[tuple(ss)])
            vmin = min(vmin, np.nanmin(arr))
            vmax = max(vmax, np.nanmax(arr))
        return vmin, vmax

    def redraw(self):
        self._stop_animation()
        v = self.comboBox_variable.currentText()
        trans_v = self.checkBox_transVariable.isChecked()
        vmin = float_or_none(self.lineEdit_min.text())
        vmax = float_or_none(self.lineEdit_max.text())
        x = self.comboBox_longitude.currentText()
        y = self.comboBox_latitude.currentText()
        inv_lon = self.checkBox_invLongitude.isChecked()
        inv_lat = self.checkBox_invLatitude.isChecked()
        shift_lon = self.checkBox_shiftLongitude.isChecked()
        cmap = self.comboBox_cmap.currentText()
        if self.checkBox_revCmap.isChecked():
            cmap += "_r"
        mesh = self.checkBox_mesh.isChecked()
        self.iiglobal = self.checkBox_global.isChecked()
        coast = self.checkBox_coast.isChecked()
        borders = self.checkBox_borders.isChecked()
        rivers = self.checkBox_rivers.isChecked()
        lakes = self.checkBox_lakes.isChecked()
        grid = self.checkBox_grid.isChecked()
        proj_name = self.comboBox_projection.currentText()
        self.iproj = self.iprojs[self.projs.index(proj_name)]
        clon = self.lineEdit_centralLon.text()
        vx = vy = vz = "None"
        if v:
            gz, vz = vardim2var(v, self.groups)
            tname = self.tname if self.usex else self.tname[gz]
            if vz == tname:
                vv = self.time_values(gz, decimal=mesh)
                vlab = "Year" if mesh else "Date"
            else:
                vv = selvar(self, vz)
                vlab = set_axis_label(vv)
            vv = self.slice_miss(self.vd, vv)
            if trans_v:
                vv = vv.T
            if shift_lon:
                vv = np.roll(vv, vv.shape[1] // 2, axis=1)
        else:
            vlab = ""
        if y:
            gy, vy = vardim2var(y, self.groups)
            tname = self.tname if self.usex else self.tname[gy]
            if vy == tname:
                yy = self.time_values(gy, decimal=mesh)
                ylab = "Year" if mesh else "Date"
            else:
                yy = selvar(self, vy)
                ylab = set_axis_label(yy)
            yy = self.slice_miss(self.latd, yy)
        else:
            ylab = ""
        if x:
            gx, vx = vardim2var(x, self.groups)
            tname = self.tname if self.usex else self.tname[gx]
            if vx == tname:
                xx = self.time_values(gx, decimal=mesh)
                xlab = "Year" if mesh else "Date"
            else:
                xx = selvar(self, vx)
                xlab = set_axis_label(xx)
            xx = self.slice_miss(self.lond, xx)
            xx360 = (xx + 360.0) % 360.0 if np.any(np.isfinite(xx)) else xx
            if xx.size > 1:
                if xx.ndim > 1:
                    x0 = xx[:, 0].mean()
                    x1 = xx[:, -2].mean() if self.iiglobal else xx[:, -1].mean()
                else:
                    x0 = xx[0]
                    x1 = xx[-2] if self.iiglobal else xx[-1]
                self.ixxmean = 0.5 * (x1 + x0)
                if self.iiglobal:
                    self.ixxmean = np.around(self.ixxmean / 180.0, 0) * 180.0
            else:
                self.ixxmean = xx360[0]
            if self.ixxmean > 180.0:
                self.ixxmean -= 360.0
        else:
            xlab = ""
            self.ixxmean = 0.0
        self.iclon = float(clon) if clon != "None" else self.ixxmean
        self.figure.clear()
        self.axes = self.figure.add_subplot(
            111, projection=self.iproj(central_longitude=self.iclon))
        if v:
            if vv.ndim < 2:
                print(f"Map: var ({vz}) is not 2-dimensional:", vv.shape)
                return
            if not x:
                nx = vv.shape[1]
                xx = -180.0 + np.arange(nx) / float(nx) * 360.0
                xx += 0.5 * (xx[1] - xx[0])
                xlab = ""
            if not y:
                ny = vv.shape[0]
                yy = -90.0 + np.arange(ny) / float(ny) * 180.0
                yy += 0.5 * (yy[1] - yy[0])
                ylab = ""
            extend = "neither"
            if vmin is not None:
                vv = np.maximum(vv, vmin)
                extend = "min" if vmax is None else "both"
            if vmax is not None:
                vv = np.minimum(vv, vmax)
                extend = "max" if vmin is None else "both"
            if xx.ndim == 1 and yy.ndim == 1:
                self.ixx, self.iyy = np.meshgrid(xx, yy)
            elif xx.ndim == 1 and yy.ndim == 2:
                self.ixx, _tmp = np.meshgrid(xx, yy[:, 0])
                self.iyy = yy
            elif xx.ndim == 2 and yy.ndim == 1:
                self.ixx, self.iyy = np.meshgrid(xx, yy)
                _tmp, self.iyy = np.meshgrid(xx[0, :], yy)
                self.ixx = xx
            elif xx.ndim == 2 and yy.ndim == 2:
                self.ixx = xx
                self.iyy = yy
            else:
                print(f"Map: lon ({vx}), lat ({vy}) dimensions not 1D or 2D:",
                      xx.shape, yy.shape)
                return
            if inv_lon:
                self.ixx = np.fliplr(self.ixx)
            if inv_lat:
                self.iyy = np.flipud(self.iyy)
            self.ivv = vv
            if self.iiglobal:
                self.ivvc, self.ixxc, self.iyyc = add_cyclic(
                    self.ivv, x=self.ixx, y=self.iyy)
            else:
                self.ivvc = self.ivv
                self.ixxc = self.ixx
                self.iyyc = self.iyy
            self.itrans = ccrs.PlateCarree()
            self.ivmin = vmin
            self.ivmax = vmax
            self.icmap = cmap
            self.ncmap = mpl.colormaps[self.icmap].N
            self.ncmap = self.ncmap if self.ncmap < 256 else 15
            self.iextend = extend
            try:
                if mesh:
                    self.cc = self.axes.pcolormesh(
                        self.ixx, self.iyy, self.ivv,
                        vmin=self.ivmin, vmax=self.ivmax,
                        cmap=self.icmap, shading="nearest",
                        transform=self.itrans)
                    self.cb = self.figure.colorbar(
                        self.cc, fraction=0.05, shrink=0.75, pad=0.07,
                        extend=self.iextend)
                else:
                    self.cc = self.axes.contourf(
                        self.ixxc, self.iyyc, self.ivvc, self.ncmap,
                        vmin=self.ivmin, vmax=self.ivmax, cmap=self.icmap,
                        extend=self.iextend, transform=self.itrans)
                    self.cb = self.figure.colorbar(
                        self.cc, fraction=0.05, shrink=0.75, pad=0.07)
            except Exception:
                print(f"Map: lon ({vx}), lat ({vy}), var ({vz}) shapes do not match:",
                      self.ixx.shape, self.iyy.shape, self.ivv.shape)
                return
            self.cb.set_label(vlab)
            self.axes.format_coord = lambda x0, y0: format_coord_map(
                x0, y0, self.axes, self.ixx, self.iyy, self.ivv)
        if self.iiglobal:
            self.axes.set_global()
        if coast:
            self.axes.add_feature(cfeature.COASTLINE)
            self.axes.gridlines(draw_labels=True, linewidth=0,
                                x_inline=False, y_inline=False)
        if borders:
            self.axes.add_feature(cfeature.BORDERS, edgecolor="grey")
        if rivers:
            self.axes.add_feature(cfeature.RIVERS)
        if lakes:
            self.axes.add_feature(cfeature.LAKES, alpha=0.5)
        self.axes.xaxis.set_label_text(xlab)
        self.axes.yaxis.set_label_text(ylab)
        if grid:
            self.axes.gridlines(draw_labels=False, x_inline=False, y_inline=False)
        self.canvas.draw()
        self.toolbar.update()

    def update_frame(self, isframe=False):
        v = self.comboBox_variable.currentText()
        details = self._time_details(v)
        if details is None or self.nunlim <= 0:
            self._stop_animation()
            return
        trans_v = self.checkBox_transVariable.isChecked()
        mesh = self.checkBox_mesh.isChecked()
        shift_lon = self.checkBox_shiftLongitude.isChecked()
        vv = details[0]
        it = self._current_time_index()
        if not isframe:
            it, direction, stop = self._next_time_index(self.anim_inc)
            if stop:
                self._stop_animation()
                return
            if direction != self.anim_inc:
                self._set_animation_direction(direction)
        self.set_tstep(it)
        vv = self.slice_miss(self.vd, vv)
        if vv.ndim < 2:
            self._stop_animation()
            return
        if trans_v:
            vv = vv.T
        if shift_lon:
            vv = np.roll(vv, vv.shape[1] // 2, axis=1)
        self.ivv = vv
        if mesh:
            self.cc.remove()
            self.cc = self.axes.pcolormesh(
                self.ixx, self.iyy, self.ivv, vmin=self.ivmin, vmax=self.ivmax,
                cmap=self.icmap, shading="nearest", transform=self.itrans)
        else:
            self.cc.remove()
            if self.iiglobal:
                self.ivvc, self.ixxc = add_cyclic(self.ivv, x=self.ixx)
            else:
                self.ivvc = self.ivv
            self.cc = self.axes.contourf(
                self.ixxc, self.iyyc, self.ivvc, self.ncmap,
                vmin=self.ivmin, vmax=self.ivmax, cmap=self.icmap,
                extend=self.iextend, transform=self.itrans)
        self.canvas.draw_idle()
