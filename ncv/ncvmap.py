"""Qt map panel drawn with pyqtgraph, projected with cartopy.

Only ``cartopy.crs`` and ``cartopy.feature`` are used, for projection maths and
Natural Earth geometry.  Neither pulls in matplotlib; all rendering is
pyqtgraph.
"""
from __future__ import annotations

import sys

import numpy as np

from .dimensions import dimension_specs, empty_dimension_specs
from .ncvcommon import (
    DimensionControlRow,
    PlotPanel,
    TimeControlMixin,
    cursor_label,
    float_or_none,
    load_ui,
    set_combo_items,
)
from .ncvmethods import get_miss
from .ncvutils import (
    add_cyclic,
    cell_edges,
    format_coord_map,
    selvar,
    set_axis_label,
    set_miss,
    vardim2var,
)
from .qt_compat import QtCore, QtGui, QtWidgets, pg


__all__ = [
    "CARTOPY_IMPORT_ERROR",
    "HAVE_CARTOPY",
    "MapPanel",
    "MapUnavailablePanel",
    "ensure_cartopy",
]


# PColorMeshItem costs ~3.5 us per cell; keep a frame near a second
MAX_CELLS = 250_000
# projections whose x depends only on longitude and y only on latitude,
# so a plain image with a rectangle extent is exact (and far faster)
SEPARABLE = ("PlateCarree", "Mercator", "Miller", "LambertCylindrical")
FEATURES = {
    "coast": ("COASTLINE", "#000000", None),
    "borders": ("BORDERS", "#808080", None),
    "rivers": ("RIVERS", "#3b7bbf", None),
    "lakes": ("LAKES", "#3b7bbf", (59, 123, 191, 128)),
}


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


def _rings(geom):
    """Yield every coordinate ring of a shapely geometry as an (n, 2) array."""
    kind = geom.geom_type
    if kind.startswith("Multi") or kind == "GeometryCollection":
        for part in geom.geoms:
            yield from _rings(part)
    elif kind == "Polygon":
        yield np.asarray(geom.exterior.coords)
        for interior in geom.interiors:
            yield np.asarray(interior.coords)
    elif kind in ("LineString", "LinearRing"):
        coords = np.asarray(geom.coords)
        if coords.size:
            yield coords


_FEATURE_CACHE = {}


def projected_rings(name, proj):
    """Natural Earth rings for ``name``, projected into ``proj``.

    ``project_geometry`` is what clips the dateline and interrupted
    projections correctly, and costs ~0.2 s, so results are cached.
    """
    key = (name, proj.proj4_init)
    if key not in _FEATURE_CACHE:
        attribute = FEATURES[name][0]
        rings = []
        for geom in getattr(cfeature, attribute).geometries():
            try:
                rings.extend(_rings(proj.project_geometry(
                    geom, ccrs.PlateCarree())))
            except Exception:
                continue
        _FEATURE_CACHE[key] = rings
    return _FEATURE_CACHE[key]


def _corner_grid(xx, yy):
    """Cell corners for 2-D curvilinear coordinate arrays."""
    def corners(values):
        # odd reflection extrapolates linearly past the edges
        padded = np.pad(np.asarray(values, dtype=float), 1,
                        mode="reflect", reflect_type="odd")
        return 0.25 * (padded[:-1, :-1] + padded[1:, :-1] +
                       padded[:-1, 1:] + padded[1:, 1:])
    return corners(xx), corners(yy)


def _nan_joined(rings):
    """Concatenate rings into one polyline separated by NaN."""
    if not rings:
        return np.empty(0), np.empty(0)
    separator = np.array([[np.nan, np.nan]])
    parts = []
    for ring in rings:
        parts.extend((ring, separator))
    joined = np.vstack(parts)
    return joined[:, 0], joined[:, 1]


def decimation_stride(shape, full_resolution=False):
    """Stride that keeps a grid under ``MAX_CELLS`` cells."""
    if full_resolution:
        return 1
    cells = int(shape[0]) * int(shape[1])
    if cells <= MAX_CELLS:
        return 1
    return int(np.ceil(np.sqrt(cells / MAX_CELLS)))


def spans_globe(lon, tol=1.0):
    """True when a longitude grid covers the full 360 degrees.

    ``add_cyclic`` appends ``lon[0] + 360`` whenever no wrap point exists, which
    smears a regional grid across the whole world.  Only wrap genuinely global
    data.
    """
    lon = np.asarray(lon)
    lon = lon[0, :] if lon.ndim > 1 else lon
    if lon.size < 2:
        return False
    step = np.abs(np.diff(lon)).mean()
    return (lon.max() - lon.min()) + step >= 360.0 - tol


class MapUnavailablePanel(QtWidgets.QWidget):
    def __init__(self, error=None):
        super().__init__()
        load_ui("map_unavailable", self)
        message = "Map view is unavailable because Cartopy could not be imported."
        if error is not None:
            message += f"\n\n{type(error).__name__}: {error}"
        message += f"\n\nPython executable: {sys.executable}"
        self.label_message.setText(message)

    def reinit(self):
        pass

    def redraw(self):
        pass


class MapPanel(TimeControlMixin, PlotPanel):
    def __init__(self, window, session):
        if not ensure_cartopy():
            raise RuntimeError("Cartopy is required for MapPanel") from CARTOPY_IMPORT_ERROR
        super().__init__(window, session, "Map")
        self._updating = True
        self._build_ui()
        self._updating = False
        self.reinit()

    def _build_ui(self):
        load_ui("map_panel", self)
        self.connect_file_controls()

        self.plot = pg.PlotWidget()
        self.item = self.plot.plotItem
        self.item.setAspectLocked(True)
        self.item.hideAxis("bottom")
        self.item.hideAxis("left")
        self.colorbar = pg.ColorBarItem(interactive=False)
        self._colorbar_added = False
        self.plotLayout.addWidget(self.plot, 1)
        cursor_label(self.plot, self.plotLayout, self._format_cursor)
        self._overlays = []
        self.data_item = None
        self.iproj = None
        self.ixx = self.iyy = self.ivv = None
        self._base_xx = self._base_yy = None
        self._cyclic = False

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
                      "Sinusoidal", "SouthPolarStereo", "Stereographic",
                      "TransverseMercator"]
        self.iprojs = [getattr(ccrs, name) for name in self.projs]
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
        for check in (self.checkBox_revCmap, self.checkBox_fullRes,
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
        self._set_limits(None, None)
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

    def _set_limits(self, vmin, vmax):
        for line_edit, value in (
            (self.lineEdit_min, vmin),
            (self.lineEdit_max, vmax),
        ):
            line_edit.setText(str(value))
            line_edit.setCursorPosition(0)

    # ---------------------------------------------------------------- events

    def checked(self):
        if not self._updating:
            self.redraw()

    def checked_all(self):
        if self._updating:
            return
        vmin, vmax = self.get_vminmax()
        self._set_limits(vmin, vmax)
        self.redraw()

    def entered_clon(self):
        self.checked()

    def entered_v(self):
        for line_edit in (self.lineEdit_min, self.lineEdit_max):
            line_edit.setCursorPosition(0)
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
        self._set_limits(vmin, vmax)
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

    # -------------------------------------------------------------- painting

    def _clear_overlays(self):
        for overlay in self._overlays:
            self.item.removeItem(overlay)
        self._overlays = []

    def _add_overlay(self, overlay):
        self.item.addItem(overlay)
        self._overlays.append(overlay)

    def _draw_features(self, proj):
        for name, (_attr, color, fill) in FEATURES.items():
            if not getattr(self, f"checkBox_{name}").isChecked():
                continue
            rings = projected_rings(name, proj)
            if not rings:
                continue
            if fill is not None:
                path = QtGui.QPainterPath()
                for ring in rings:
                    path.moveTo(ring[0, 0], ring[0, 1])
                    for px, py in ring[1:]:
                        path.lineTo(px, py)
                    path.closeSubpath()
                shape = QtWidgets.QGraphicsPathItem(path)
                shape.setBrush(QtGui.QBrush(QtGui.QColor(*fill)))
                shape.setPen(pg.mkPen(color, width=0.5))
                self._add_overlay(shape)
            else:
                xs, ys = _nan_joined(rings)
                self._add_overlay(pg.PlotDataItem(
                    xs, ys, connect="finite", pen=pg.mkPen(color, width=0.8)))

    def _draw_graticule(self, proj, labels, step=30.0):
        pen = pg.mkPen("#9a9a9a", width=0.5, style=QtCore.Qt.PenStyle.DotLine)
        lats = np.linspace(-89.5, 89.5, 180)
        lons = np.linspace(-180.0, 180.0, 361)
        for lon in np.arange(-180.0, 180.1, step):
            pts = proj.transform_points(
                ccrs.PlateCarree(), np.full_like(lats, lon), lats)
            self._add_overlay(pg.PlotDataItem(
                pts[:, 0], pts[:, 1], connect="finite", pen=pen))
            if labels:
                self._add_label(pts, f"{lon:g}°", anchor=(0.5, 0.0))
        for lat in np.arange(-90.0, 90.1, step):
            pts = proj.transform_points(
                ccrs.PlateCarree(), lons, np.full_like(lons, lat))
            self._add_overlay(pg.PlotDataItem(
                pts[:, 0], pts[:, 1], connect="finite", pen=pen))
            if labels:
                self._add_label(pts, f"{lat:g}°", anchor=(1.0, 0.5),
                                westmost=True)

    def _add_label(self, points, text, anchor, westmost=False):
        finite = points[np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])]
        if not finite.size:
            return
        index = finite[:, 0].argmin() if westmost else finite[:, 1].argmin()
        label = pg.TextItem(text, color="#606060", anchor=anchor)
        label.setPos(finite[index, 0], finite[index, 1])
        self._add_overlay(label)

    def _draw_data(self, xx, yy, vv, proj, cmap, levels):
        """Image for separable projections, quad mesh otherwise."""
        separable = type(proj).__name__ in SEPARABLE
        if separable and xx.ndim == 1 and yy.ndim == 1:
            xedges, yedges = cell_edges(xx), cell_edges(yy)
            edges = proj.transform_points(
                ccrs.PlateCarree(),
                np.array([xedges[0], xedges[-1]]),
                np.array([yedges[0], yedges[-1]]))
            image = pg.ImageItem(vv)
            image.setRect(QtCore.QRectF(
                edges[0, 0], edges[0, 1],
                edges[1, 0] - edges[0, 0], edges[1, 1] - edges[0, 1]))
            image.setColorMap(cmap)
            image.setLevels(levels)
            self._add_overlay(image)
            return image

        if xx.ndim == 1 and yy.ndim == 1:
            cx, cy = np.meshgrid(cell_edges(xx), cell_edges(yy))
        else:
            shape = vv.shape
            cx, cy = _corner_grid(
                xx if xx.ndim == 2 else np.broadcast_to(xx, shape),
                yy if yy.ndim == 2 else np.broadcast_to(yy[:, None], shape))
        pts = proj.transform_points(ccrs.PlateCarree(), cx, cy)
        mesh = pg.PColorMeshItem(
            pts[:, :, 0], pts[:, :, 1], vv,
            colorMap=cmap, levels=levels, enableAutoLevels=False)
        self._add_overlay(mesh)
        return mesh

    def redraw(self):
        self._stop_animation()
        self._clear_overlays()
        self.data_item = None
        v = self.comboBox_variable.currentText()
        x = self.comboBox_longitude.currentText()
        y = self.comboBox_latitude.currentText()
        vmin = float_or_none(self.lineEdit_min.text())
        vmax = float_or_none(self.lineEdit_max.text())
        self.iiglobal = self.checkBox_global.isChecked()
        clon = self.lineEdit_centralLon.text()
        proj_name = self.comboBox_projection.currentText()
        projection = self.iprojs[self.projs.index(proj_name)]

        vv = xx = yy = None
        vlab = ""
        if v:
            vv, vlab = self._variable_values(v)
        if x:
            xx = self._coordinate_values(x, self.lond)
        if y:
            yy = self._coordinate_values(y, self.latd)

        self.ixxmean = self._central_longitude(xx)
        self.iclon = float(clon) if clon != "None" else self.ixxmean
        self.iproj = projection(central_longitude=self.iclon)

        if vv is not None:
            if vv.ndim < 2:
                print(f"Map: var ({v}) is not 2-dimensional:", vv.shape)
                return
            if xx is None:
                nx = vv.shape[1]
                xx = -180.0 + (np.arange(nx) + 0.5) / float(nx) * 360.0
            if yy is None:
                ny = vv.shape[0]
                yy = -90.0 + (np.arange(ny) + 0.5) / float(ny) * 180.0
            self._plot_variable(xx, yy, vv, vmin, vmax, vlab)

        self._draw_features(self.iproj)
        if self.checkBox_coast.isChecked() or self.checkBox_grid.isChecked():
            self._draw_graticule(
                self.iproj, labels=self.checkBox_coast.isChecked())
        if self.iiglobal:
            self.item.setRange(
                xRange=self.iproj.x_limits, yRange=self.iproj.y_limits,
                padding=0)
        else:
            self.item.vb.autoRange(padding=0.02)

    def _plot_variable(self, xx, yy, vv, vmin, vmax, vlab):
        if self.checkBox_invLongitude.isChecked():
            xx = np.flip(xx, axis=-1)
        if self.checkBox_invLatitude.isChecked():
            yy = np.flip(yy, axis=0)
        nx = xx.shape[-1] if xx.ndim > 1 else xx.shape[0]
        if vv.shape != (yy.shape[0], nx):
            print("Map: lon, lat, var shapes do not match:",
                  np.shape(xx), np.shape(yy), vv.shape)
            return
        self._base_xx, self._base_yy = xx, yy
        self._cyclic = self.iiglobal and spans_globe(xx)
        if self._cyclic:
            vv, xx, yy = add_cyclic(vv, x=xx, y=yy)
        if vmin is not None:
            vv = np.maximum(vv, vmin)
        if vmax is not None:
            vv = np.minimum(vv, vmax)

        stride = decimation_stride(vv.shape, self.checkBox_fullRes.isChecked())
        if stride > 1:
            vv = vv[::stride, ::stride]
            xx = xx[..., ::stride] if xx.ndim > 1 else xx[::stride]
            yy = yy[::stride] if yy.ndim == 1 else yy[::stride, ::stride]

        finite = vv[np.isfinite(vv)]
        levels = (
            vmin if vmin is not None else (finite.min() if finite.size else 0.0),
            vmax if vmax is not None else (finite.max() if finite.size else 1.0),
        )
        cmap = self.selected_cmap_object(self.comboBox_cmap,
                                         self.checkBox_revCmap)
        try:
            self.data_item = self._draw_data(
                xx, yy, vv, self.iproj, cmap, levels)
        except Exception as exc:
            print("Map: lon, lat, var shapes do not match:",
                  np.shape(xx), np.shape(yy), vv.shape, exc)
            return
        self.colorbar.setColorMap(cmap)
        self.colorbar.setLevels(low=levels[0], high=levels[1])
        self.colorbar.setLabel("right", vlab)
        if not self._colorbar_added:
            self.colorbar.setImageItem(self.data_item, insert_in=self.item)
            self._colorbar_added = True
        self.ixx, self.iyy, self.ivv = xx, yy, vv

    def _variable_values(self, v):
        gz, vz = vardim2var(v, self.groups)
        tname = self.tname if self.usex else self.tname[gz]
        if vz == tname:
            values, label = self.time_values(gz, decimal=True), "Year"
        else:
            values = selvar(self, vz)
            label = set_axis_label(values)
        values = np.asarray(self.slice_miss(self.vd, values), dtype=float)
        if self.checkBox_transVariable.isChecked():
            values = values.T
        if self.checkBox_shiftLongitude.isChecked() and values.ndim > 1:
            values = np.roll(values, values.shape[1] // 2, axis=1)
        return values, label

    def _coordinate_values(self, vardim, dim_controls):
        group, name = vardim2var(vardim, self.groups)
        tname = self.tname if self.usex else self.tname[group]
        values = (self.time_values(group, decimal=True) if name == tname
                  else selvar(self, name))
        return np.asarray(self.slice_miss(dim_controls, values), dtype=float)

    def _central_longitude(self, xx):
        if xx is None or np.size(xx) == 0:
            return 0.0
        flat = np.asarray(xx).ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return 0.0
        if finite.size == 1:
            mean = (finite[0] + 360.0) % 360.0
        else:
            mean = 0.5 * (finite.min() + finite.max())
            if self.iiglobal:
                mean = np.around(mean / 180.0, 0) * 180.0
        return float(mean - 360.0 if mean > 180.0 else mean)

    def update_frame(self, isframe=False):
        v = self.comboBox_variable.currentText()
        details = self._time_details(v)
        if details is None or self.nunlim <= 0:
            self._stop_animation()
            return
        it = self._current_time_index()
        if not isframe:
            it, direction, stop = self._next_time_index(self.anim_inc)
            if stop:
                self._stop_animation()
                return
            if direction != self.anim_inc:
                self._set_animation_direction(direction)
        self.set_tstep(it)
        vv, _vlab = self._variable_values(v)
        if vv.ndim < 2:
            self._stop_animation()
            return
        if self._cyclic:
            vv, _x, _y = add_cyclic(vv, x=self._base_xx, y=self._base_yy)
        stride = decimation_stride(vv.shape, self.checkBox_fullRes.isChecked())
        if stride > 1:
            vv = vv[::stride, ::stride]
        self.ivv = vv
        if isinstance(self.data_item, pg.ImageItem):
            self.data_item.setImage(vv, autoLevels=False)
        elif self.data_item is not None:
            self.data_item.setData(z=vv)

    def _format_cursor(self, x, y):
        if self.iproj is None:
            return ""
        return format_coord_map(x, y, self.iproj, self.ixx, self.iyy, self.ivv)
