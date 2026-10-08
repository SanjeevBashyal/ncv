import ast
import os
from pathlib import Path
import subprocess
import sys

import netCDF4 as nc
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def make_sample(path, *, fixed_time=False):
    ds = nc.Dataset(path, "w")
    ds.title = "Matrix test dataset"
    ds.createDimension("time", 4 if fixed_time else None)
    ds.createDimension("lat", 3)
    ds.createDimension("lon", 4)

    time = ds.createVariable("time", "f8", ("time",))
    time.units = "days since 2000-01-01 00:00:00"
    time.calendar = "standard"

    lat = ds.createVariable("lat", "f4", ("lat",))
    lat.units = "degrees_north"
    lat.standard_name = "latitude"

    lon = ds.createVariable("lon", "f4", ("lon",))
    lon.units = "degrees_east"
    lon.standard_name = "longitude"

    temp = ds.createVariable("temp", "f4", ("time", "lat", "lon"))
    temp.units = "K"
    temp.long_name = "temperature"
    surface = ds.createVariable("surface", "f4", ("lat", "lon"))

    time[:] = [0, 1, 2, 3]
    lat[:] = [-45, 0, 45]
    lon[:] = [0, 90, 180, 270]
    temp[:] = np.arange(4 * 3 * 4).reshape(4, 3, 4)
    surface[:] = np.arange(3 * 4).reshape(3, 4)
    ds.close()


def test_import_is_cartopy_safe():
    import ncv

    assert callable(ncv.ncv)
    assert not hasattr(ncv, "ncvue")


def test_import_does_not_require_tk_or_cartopy():
    code = """
import sys
for name in ('tkinter', 'customtkinter', 'cartopy'):
    sys.modules[name] = None
import ncv
from ncv.app import HAVE_CARTOPY, NcvMainWindow
from ncv.ncvmap import MapUnavailablePanel
from ncv.qt_compat import QtWidgets
from ncv.session import NcvSession
assert callable(ncv.ncv)
assert not HAVE_CARTOPY
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
window = NcvMainWindow(NcvSession())
assert isinstance(window.map, MapUnavailablePanel)
window.close()
app.processEvents()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_package_has_no_tk_imports():
    package_dir = Path(__file__).parents[1] / "ncv"
    legacy_paths = {
        "ncvmain.py",
        "ncvscreen.py",
        "ncvwidgets.py",
        "tooltip.py",
        "themes",
    }
    assert not any((package_dir / path).exists() for path in legacy_paths)

    forbidden = {"tkinter", "customtkinter", "matplotlib"}
    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".", 1)[0] for alias in node.names}
                assert names.isdisjoint(forbidden), path
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden, path


def test_session_open_and_analyse_netcdf(tmp_path):
    from ncv.session import NcvSession

    path = tmp_path / "sample.nc"
    make_sample(path)

    session = NcvSession()
    session.open([str(path)])

    assert session.has_data
    assert session.maxdim == 3
    assert any(col.startswith("temp ") for col in session.cols)
    assert session.latvar[0].startswith("lat ")
    assert session.lonvar[0].startswith("lon ")
    assert session.dunlim[0] == "time"
    session.close()


def test_session_rejects_multiple_files_when_one_has_groups(tmp_path):
    from ncv.session import NcvSession

    path1 = tmp_path / "one.nc"
    path2 = tmp_path / "two.nc"
    make_sample(path1)
    make_sample(path2)
    ds = nc.Dataset(path2, "a")
    ds.createGroup("grouped")
    ds.close()

    with pytest.raises(ValueError):
        NcvSession().open([str(path1), str(path2)])


def test_get_slice_values_reducers():
    from ncv.ncvutils import get_slice_values

    data = np.arange(6).reshape(2, 3)

    out = get_slice_values(["all", "mean"], data)

    assert np.allclose(out, [1, 4])


def test_qt_window_smoke_with_generated_netcdf(tmp_path):
    from ncv.app import HAVE_CARTOPY, NcvMainWindow
    from ncv.ncvcontour import ContourPanel
    from ncv.ncvmap import MapPanel, MapUnavailablePanel
    from ncv.ncvmatrix import MatrixPanel
    from ncv.ncvscatter import ScatterPanel
    from ncv.qt_compat import QtWidgets
    from ncv.session import NcvSession

    path = tmp_path / "sample.nc"
    make_sample(path)
    session = NcvSession()
    session.open([str(path)])

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = NcvMainWindow(session)
    temp = next(col for col in session.cols if col.startswith("temp "))

    assert win.objectName() == "MainWindow"   # root name set in main_window.ui
    assert win.tabWidget_main.count() == 4
    assert win.tabWidget_main.widget(0) is win.scatter
    assert win.tabWidget_main.widget(1) is win.contour
    assert win.tabWidget_main.widget(3) is win.matrix
    assert isinstance(win.scatter, ScatterPanel)
    assert isinstance(win.contour, ContourPanel)
    assert isinstance(win.matrix, MatrixPanel)
    assert win.scatter.comboBox_y.objectName() == "comboBox_y"
    assert win.scatter.lineEdit_xlim.objectName() == "lineEdit_xlim"
    assert win.contour.comboBox_z.objectName() == "comboBox_z"
    assert win.contour.checkBox_transposeZ.objectName() == "checkBox_transposeZ"
    assert win.contour.lineEdit_zlim.objectName() == "lineEdit_zlim"

    win.contour.lineEdit_zlim.setText("(1.5, 8)")
    assert tuple(win.contour._z_limits()) == (1.5, 8.0)

    win.scatter.comboBox_y.setCurrentText(temp)
    win.scatter.selected_y()
    win.scatter.redraw()
    assert len(win.scatter.curves_y) == 1
    # _restyle must not blank the curve (setData(**style) used to wipe it)
    xdata = win.scatter.curves_y[0].getData()[0]
    assert xdata is not None and len(xdata) > 1
    # ...and the x-axis must actually rescale onto that data
    xlo, xhi = win.scatter.item.vb.viewRange()[0]
    assert xlo <= xdata.min() and xhi >= xdata.max()

    win.contour.comboBox_z.setCurrentText(temp)
    win.contour.selected_z()
    assert win.contour._zz is not None

    if not HAVE_CARTOPY:
        assert isinstance(win.map, MapUnavailablePanel)
    else:
        assert isinstance(win.map, MapPanel)
        assert win.map.lineEdit_min.objectName() == "lineEdit_min"
        assert win.map.lineEdit_max.objectName() == "lineEdit_max"
        assert win.map.comboBox_longitude.objectName() == "comboBox_longitude"
        assert win.map.comboBox_latitude.objectName() == "comboBox_latitude"
        assert (
            win.map.checkBox_invLongitude.objectName()
            == "checkBox_invLongitude"
        )
        assert win.map.checkBox_invLatitude.objectName() == "checkBox_invLatitude"
        assert (
            win.map.checkBox_shiftLongitude.objectName()
            == "checkBox_shiftLongitude"
        )

    win.close()
    app.processEvents()


@pytest.fixture(scope="session")
def qt_app():
    from ncv.qt_compat import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def make_map_panel(path, qt_app, *, fixed_time=False):
    from ncv.ncvmap import HAVE_CARTOPY, MapPanel
    from ncv.qt_compat import QtWidgets
    from ncv.session import NcvSession

    if not HAVE_CARTOPY:
        pytest.skip("Cartopy is not installed")

    make_sample(path, fixed_time=fixed_time)
    session = NcvSession()
    session.open([str(path)])

    class PanelWindow(QtWidgets.QWidget):
        def open_file_dialog(self, _use_xarray):
            pass

        def create_secondary_window(self):
            pass

    window = PanelWindow()
    panel = MapPanel(window, session)
    for check_box in (
        panel.checkBox_coast,
        panel.checkBox_borders,
        panel.checkBox_rivers,
        panel.checkBox_lakes,
    ):
        blocked = check_box.blockSignals(True)
        check_box.setChecked(False)
        check_box.blockSignals(blocked)
    return panel, window, session


def select_map_variable(panel, session, name):
    item = next(column for column in session.cols if column.startswith(f"{name} "))
    panel.comboBox_variable.setCurrentText(item)
    assert panel.comboBox_variable.currentText() == item


def assert_map_frame(panel, session, index):
    assert panel.iunlim >= 0
    assert panel.vd.selectors[panel.iunlim].currentText() == str(index)
    assert panel.horizontalSlider_timeStep.value() == index
    assert panel.label_timeValue.text() == str(session.time[0][index])


def test_map_time_navigation_and_animation(tmp_path, qt_app):
    panel, window, session = make_map_panel(
        tmp_path / "map-time.nc", qt_app)
    try:
        select_map_variable(panel, session, "temp")
        assert panel.nunlim == 4
        assert panel.horizontalSlider_timeStep.maximum() == 3
        assert_map_frame(panel, session, 0)
        assert panel.lineEdit_min.cursorPosition() == 0
        assert panel.lineEdit_max.cursorPosition() == 0

        panel.lineEdit_min.setText("0.1234567890123456789")
        panel.lineEdit_min.editingFinished.emit()
        assert panel.lineEdit_min.cursorPosition() == 0

        panel.pushButton_lastTime.click()
        assert_map_frame(panel, session, 3)
        panel.pushButton_prevTime.click()
        assert_map_frame(panel, session, 2)
        panel.pushButton_firstTime.click()
        assert_map_frame(panel, session, 0)
        panel.pushButton_nextTime.click()
        assert_map_frame(panel, session, 1)
        panel.horizontalSlider_timeStep.setValue(2)
        assert_map_frame(panel, session, 2)

        # A normal timer tick must not be stopped by dimension-selector signals.
        panel.horizontalSlider_timeStep.setValue(0)
        panel.pushButton_runForward.click()
        assert panel.timer.isActive()
        assert panel.pushButton_runForward.text() == "||"
        assert panel.pushButton_runBackward.text() == "<"
        panel.update_frame()
        assert panel.timer.isActive()
        assert_map_frame(panel, session, 1)
        panel.pushButton_runForward.click()
        assert not panel.timer.isActive()
        assert panel.pushButton_runForward.text() == ">"
        assert panel.pushButton_runBackward.text() == "<"

        panel.comboBox_repeat.setCurrentText("once")
        panel.horizontalSlider_timeStep.setValue(3)
        panel.pushButton_runForward.click()
        panel.update_frame()
        assert_map_frame(panel, session, 3)
        assert not panel.timer.isActive()
        assert panel.pushButton_runForward.text() == ">"
        assert panel.pushButton_runBackward.text() == "<"

        panel.comboBox_repeat.setCurrentText("repeat")
        panel.pushButton_runForward.click()
        panel.update_frame()
        assert_map_frame(panel, session, 0)
        assert panel.timer.isActive()
        panel.pushButton_runForward.click()
        assert not panel.timer.isActive()

        panel.comboBox_repeat.setCurrentText("reflect")
        panel.horizontalSlider_timeStep.setValue(3)
        panel.pushButton_runForward.click()
        panel.update_frame()
        assert_map_frame(panel, session, 2)
        assert panel.timer.isActive()
        assert panel.anim_inc == -1
        assert panel.pushButton_runForward.text() == ">"
        assert panel.pushButton_runBackward.text() == "||"

        # The opposite run button switches direction; the active one pauses.
        panel.pushButton_runForward.click()
        assert panel.timer.isActive()
        assert panel.anim_inc == 1
        assert panel.pushButton_runForward.text() == "||"
        assert panel.pushButton_runBackward.text() == "<"
        panel.pushButton_runForward.click()
        assert not panel.timer.isActive()
        assert panel.pushButton_runForward.text() == ">"
        assert panel.pushButton_runBackward.text() == "<"
    finally:
        panel.timer.stop()
        panel.close()
        window.close()
        session.close()


def test_map_fixed_time_and_static_variable_controls(tmp_path, qt_app):
    panel, window, session = make_map_panel(
        tmp_path / "map-fixed-time.nc", qt_app, fixed_time=True)
    try:
        assert session.dunlim[0] == ""

        select_map_variable(panel, session, "temp")
        assert panel.iunlim == 0
        assert panel.nunlim == 4
        assert panel.horizontalSlider_timeStep.isEnabled()
        assert panel.pushButton_runForward.isEnabled()
        panel.pushButton_nextTime.click()
        assert_map_frame(panel, session, 1)

        # The displayed synthetic datetime item resolves to the physical time
        # variable even when the time dimension is fixed-size.
        select_map_variable(panel, session, "datetime")
        assert panel.iunlim == 0
        assert panel.nunlim == 4
        assert panel.horizontalSlider_timeStep.maximum() == 3

        select_map_variable(panel, session, "surface")
        assert panel.iunlim == -1
        assert panel.nunlim == 0
        assert panel.horizontalSlider_timeStep.maximum() == 0
        assert not any(widget.isEnabled() for widget in (
            panel.horizontalSlider_timeStep,
            panel.pushButton_firstTime,
            panel.pushButton_prevTime,
            panel.pushButton_runBackward,
            panel.pushButton_runForward,
            panel.pushButton_nextTime,
            panel.pushButton_lastTime,
            panel.comboBox_repeat,
        ))
    finally:
        panel.timer.stop()
        panel.close()
        window.close()
        session.close()


def make_matrix_panel(path, qt_app, *, fixed_time=False):
    from ncv.ncvmatrix import MatrixPanel
    from ncv.qt_compat import QtWidgets
    from ncv.session import NcvSession

    make_sample(path, fixed_time=fixed_time)
    session = NcvSession()
    session.open([str(path)])

    window = _panel_window()
    panel = MatrixPanel(window, session)
    return panel, window, session


def _panel_window():
    """Stand-in for NcvMainWindow: panels only call these two methods."""
    from ncv.qt_compat import QtWidgets

    class PanelWindow(QtWidgets.QWidget):
        def open_file_dialog(self, _use_xarray):
            pass

        def create_secondary_window(self):
            pass

    return PanelWindow()


def select_matrix_variable(panel, session, name):
    item = next(column for column in session.cols if column.startswith(f"{name} "))
    panel.comboBox_z.setCurrentText(item)
    assert panel.comboBox_z.currentText() == item


def matrix_display(panel, row, column):
    from ncv.qt_compat import QtCore

    model = panel.tableView_showMatrix.model()
    return model.data(model.index(row, column), QtCore.Qt.ItemDataRole.DisplayRole)


def matrix_header(panel, section, orientation):
    from ncv.qt_compat import QtCore

    model = panel.tableView_showMatrix.model()
    return model.headerData(section, orientation, QtCore.Qt.ItemDataRole.DisplayRole)


def test_matrix_table_metadata_formats_and_flips(tmp_path, qt_app):
    from ncv.qt_compat import QtCore

    panel, window, session = make_matrix_panel(tmp_path / "matrix.nc", qt_app)
    try:
        dataset_header = panel.textBrowser_showHeader.toPlainText().lower()
        assert "matrix test dataset" in dataset_header
        assert "dimensions" in dataset_header
        assert "temp" in dataset_header

        select_matrix_variable(panel, session, "temp")
        model = panel.tableView_showMatrix.model()
        assert model.rowCount() == 3
        assert model.columnCount() == 4
        assert panel.comboBox_x.currentText().startswith("lon ")
        assert panel.comboBox_y.currentText().startswith("lat ")
        assert matrix_display(panel, 0, 0) == "0.0"
        assert matrix_display(panel, 2, 3) == "11.0"
        assert matrix_header(panel, 1, QtCore.Qt.Orientation.Horizontal) == "90.0"
        assert matrix_header(panel, 0, QtCore.Qt.Orientation.Vertical) == "-45.0"
        assert panel.lineEdit_min.text() == "0.0"
        assert panel.lineEdit_max.text() == "11.0"

        panel.checkBox_allValues.setChecked(True)
        assert float(panel.lineEdit_min.text()) == 0
        assert float(panel.lineEdit_max.text()) == 47
        panel.checkBox_allValues.setChecked(False)
        assert float(panel.lineEdit_max.text()) == 11

        variable_header = panel.textBrowser_showHeader.toPlainText().lower()
        assert "temp" in variable_header
        assert "temperature" in variable_header
        assert "units" in variable_header
        assert "k" in variable_header

        panel.comboBox_dataFormat.setCurrentText("%.2E")
        panel.comboBox_rowColHeaderFormat.setCurrentText("%.0f")
        assert matrix_display(panel, 1, 2) == "6.00E+00"
        assert panel.lineEdit_min.text() == "0.00E+00"
        assert panel.lineEdit_max.text() == "1.10E+01"
        assert matrix_header(panel, 1, QtCore.Qt.Orientation.Horizontal) == "90"
        assert matrix_header(panel, 0, QtCore.Qt.Orientation.Vertical) == "-45"

        panel.checkBox_flipTableLeftRight.setChecked(True)
        assert matrix_display(panel, 0, 0) == "3.00E+00"
        assert matrix_header(panel, 0, QtCore.Qt.Orientation.Horizontal) == "270"

        panel.checkBox_flipTableTopBottom.setChecked(True)
        assert matrix_display(panel, 0, 0) == "1.10E+01"
        assert matrix_header(panel, 0, QtCore.Qt.Orientation.Vertical) == "45"

        panel.checkBox_showCellIndices.setChecked(True)
        assert matrix_header(panel, 0, QtCore.Qt.Orientation.Horizontal) == "3"
        assert matrix_header(panel, 0, QtCore.Qt.Orientation.Vertical) == "2"
    finally:
        panel.timer.stop()
        panel.close()
        window.close()
        session.close()


def assert_matrix_frame(panel, session, index, first_value):
    assert panel.horizontalSlider_timeStep.value() == index
    assert panel.label_timeValue.text() == str(session.time[0][index])
    assert matrix_display(panel, 0, 0) == first_value


def test_matrix_time_navigation_animation_and_static_variable(
        tmp_path, qt_app):
    panel, window, session = make_matrix_panel(
        tmp_path / "matrix-time.nc", qt_app)
    try:
        select_matrix_variable(panel, session, "temp")
        assert panel.iunlim == 0
        assert panel.nunlim == 4
        assert_matrix_frame(panel, session, 0, "0.0")

        panel.pushButton_nextTime.click()
        assert_matrix_frame(panel, session, 1, "12.0")
        panel.horizontalSlider_timeStep.setValue(2)
        assert_matrix_frame(panel, session, 2, "24.0")
        panel.pushButton_lastTime.click()
        assert_matrix_frame(panel, session, 3, "36.0")
        panel.pushButton_firstTime.click()
        assert_matrix_frame(panel, session, 0, "0.0")

        panel.pushButton_runForward.click()
        assert panel.timer.isActive()
        panel.update_frame()
        assert panel.timer.isActive()
        assert_matrix_frame(panel, session, 1, "12.0")
        panel.pushButton_runForward.click()
        assert not panel.timer.isActive()

        select_matrix_variable(panel, session, "surface")
        assert panel.iunlim == -1
        assert panel.nunlim == 0
        assert panel.label_timeValue.text() == ""
        assert panel.horizontalSlider_timeStep.maximum() == 0
        assert not any(widget.isEnabled() for widget in (
            panel.horizontalSlider_timeStep,
            panel.pushButton_firstTime,
            panel.pushButton_prevTime,
            panel.pushButton_runBackward,
            panel.pushButton_runForward,
            panel.pushButton_nextTime,
            panel.pushButton_lastTime,
            panel.comboBox_repeat,
        ))
    finally:
        panel.timer.stop()
        panel.close()
        window.close()
        session.close()


def test_matrix_fixed_time_dimension_remains_active(tmp_path, qt_app):
    panel, window, session = make_matrix_panel(
        tmp_path / "matrix-fixed-time.nc", qt_app, fixed_time=True)
    try:
        assert session.dunlim[0] == ""
        select_matrix_variable(panel, session, "temp")
        assert panel.iunlim == 0
        assert panel.nunlim == 4
        assert panel.horizontalSlider_timeStep.isEnabled()
        assert panel.pushButton_runForward.isEnabled()
        panel.pushButton_nextTime.click()
        assert_matrix_frame(panel, session, 1, "12.0")
    finally:
        panel.timer.stop()
        panel.close()
        window.close()
        session.close()


def test_qt_designer_forms_load(qt_app):
    from PyQt6 import QtWidgets, uic

    ui_dir = Path(__file__).parents[1] / "ncv" / "ui"
    forms = {
        "main_window.ui",
        "scatter_panel.ui",
        "contour_panel.ui",
        "map_panel.ui",
        "map_unavailable.ui",
        "matrix_panel.ui",
    }

    assert {path.name for path in ui_dir.glob("*.ui")} == forms
    for form in forms:
        base = (QtWidgets.QMainWindow if form == "main_window.ui"
                else QtWidgets.QWidget)
        widget = base()
        uic.loadUi(str(ui_dir / form), widget)
        assert widget.children()
        if form.endswith("_panel.ui") and form != "map_unavailable.ui":
            # the read-out bar is static, so Designer shows it
            assert widget.horizontalLayout_status is not None
            assert widget.label_cursor is not None


def test_to_plot_values_and_cell_edges():
    from ncv.ncvcommon import to_plot_values
    from ncv.ncvutils import cell_edges

    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ms]")
    values, is_date = to_plot_values(times)
    assert is_date
    # POSIX seconds, one day apart
    assert values[1] - values[0] == 86400.0
    assert to_plot_values(np.arange(3)) [1] is False

    assert np.allclose(cell_edges([1.0, 2.0, 3.0]), [0.5, 1.5, 2.5, 3.5])
    assert np.allclose(cell_edges([5.0]), [4.5, 5.5])
    assert len(cell_edges(np.arange(7.0))) == 8


def test_scatter_style_tables_cover_the_form_defaults():
    from ncv.ncvscatter import PEN_STYLES, SYMBOLS

    # every style string the .ui ships must map onto Qt/pyqtgraph
    assert set("- -- -. :".split()) <= set(PEN_STYLES)
    assert set("o s ^ v d + x *".split()) <= set(SYMBOLS)
    # "None" deliberately maps to nothing, which disables the line/marker
    assert "None" not in PEN_STYLES and "None" not in SYMBOLS


def test_overview_stride_respects_chunk_and_cell_budgets():
    from ncv.ncvutils import overview_stride

    # fits in memory -> read everything
    assert overview_stride((90, 216), (600, 600)) == (1, 1)

    # a stride below the chunk size costs a full read, so it must snap to a
    # multiple of the chunk shape and keep touched chunks under budget
    sy, sx = overview_stride((90001, 216001), (600, 600), max_chunks=4000)
    assert sy % 600 == 0 and sx % 600 == 0
    touched = -(-90001 // sy) * (-(-216001 // sx))
    assert touched <= 4000
    assert touched * 1.0 > 1000          # not needlessly coarse

    # row chunks (1, nx): scaling x saves nothing, so the budget must come
    # from y alone, and the whole-row chunk is sampled along x, not 1 column
    sy, sx = overview_stride((84000, 216000), (1, 216000), max_chunks=500,
                             max_cells=36_000_000)
    assert -(-84000 // sy) <= 500
    assert sx < 216000

    # contiguous variables have no chunks; fall back to the cell budget
    sy, sx = overview_stride((90001, 216001), None, max_cells=250000)
    assert (-(-90001 // sy)) * (-(-216001 // sx)) <= 250000


def test_get_slice_values_window_matches_direct_read(tmp_path):
    from ncv.ncvutils import get_slice_values

    path = tmp_path / "chunked.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("y", 40)
        ds.createDimension("x", 60)
        var = ds.createVariable("v", "f8", ("y", "x"), chunksizes=(10, 10))
        var[:] = np.arange(40 * 60, dtype=float).reshape(40, 60)

    with nc.Dataset(path) as ds:
        var = ds["v"]
        full = get_slice_values(["all", "all"], var)
        windowed = get_slice_values(
            ["all", "all"], var, window={0: (8, 24, 2), 1: (10, 50, 5)})
        assert np.array_equal(windowed, np.asarray(full)[8:24:2, 10:50:5])
        # and the window really is what was read
        assert windowed.shape == (8, 8)


def test_scrollable_view_bar_geometry(qt_app):
    from PyQt6 import QtWidgets
    from ncv.ncvcommon import ScrollableView

    view = ScrollableView(QtWidgets.QWidget())
    # whole variable loaded -> nothing to scroll
    view.set_extent(100, 200, 100, 200)
    assert not view.vbar.isEnabled() and not view.hbar.isEnabled()
    # a window into a larger variable -> thumb shows the loaded fraction
    view.set_extent(90001, 216001, 500, 500)
    assert view.vbar.isEnabled() and view.vbar.maximum() == 89501
    assert view.vbar.pageStep() == 500
    assert view.hbar.maximum() == 215501
    view.vbar.setValue(40000)
    assert view.offsets() == (40000, 0)


def test_metadata_html_formatting():
    from ncv.ncvmatrix import _metadata_html

    out = _metadata_html("<class 'netCDF4.Variable'>\n    units: Pa\nfilling on")
    assert "<b>units:</b> Pa" in out
    assert "<class" not in out          # netCDF4 repr noise dropped
    assert "filling on" in out          # non-attribute lines survive
    assert "&lt;" not in out.replace("&lt;/pre&gt;", "")


def test_map_spans_globe():
    from ncv.ncvmap import spans_globe

    # a global grid wraps; a regional one must not (else add_cyclic smears it)
    assert spans_globe(np.linspace(-180.0, 177.5, 144))
    assert not spans_globe(np.linspace(11.0, 11.5, 6))
    assert not spans_globe(np.array([11.0]))


def test_gl_usable_and_mesh_cap(qt_app, monkeypatch):
    import ncv.ncvmap as nmap
    import ncv.qt_compat as compat

    monkeypatch.delenv("NCV_OPENGL", raising=False)
    assert compat.gl_usable() is False              # offscreen: no GL viewport
    assert nmap.mesh_max_cells() == nmap.MESH_MAX_CELLS_CPU
    monkeypatch.setenv("NCV_OPENGL", "1")
    assert compat.gl_usable() and nmap.mesh_max_cells() == nmap.MESH_MAX_CELLS
    monkeypatch.setenv("NCV_OPENGL", "0")
    assert compat.gl_usable() is False


def _map_file(path, rotate):
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("y", 40)
        ds.createDimension("x", 60)
        r, c = np.mgrid[0:40, 0:60].astype(float)
        lon = ds.createVariable("lon", "f8", ("y", "x"))
        lon.units = "degrees_east"
        lat = ds.createVariable("lat", "f8", ("y", "x"))
        lat.units = "degrees_north"
        # north-first rows, like most rasters; optionally sheared (curvilinear)
        lon[:] = 80.0 + 0.1 * c + (0.05 * r if rotate else 0.0)
        lat[:] = 30.0 - 0.1 * r + (0.03 * c if rotate else 0.0)
        ds.createVariable("v", "i4", ("y", "x"))[:] = (r * 60 + c).astype(int)


@pytest.mark.parametrize("rotate, item", [(False, "ImageItem"), (True, "PColorMeshItem")])
def test_map_rectilinear_2d_coords_draw_as_image(tmp_path, qt_app, rotate, item):
    from ncv.ncvmap import HAVE_CARTOPY
    if not HAVE_CARTOPY:
        pytest.skip("cartopy unavailable")
    from ncv.ncvmap import MapPanel
    from ncv.session import NcvSession

    path = tmp_path / "grid.nc"
    _map_file(path, rotate)
    session = NcvSession()
    session.open([str(path)])
    panel = MapPanel(_panel_window(), session)
    panel.resize(900, 600)
    panel.show()
    pick = lambda name: next(c for c in session.cols if c.startswith(name + " "))
    panel.comboBox_longitude.setCurrentText(pick("lon"))
    panel.comboBox_latitude.setCurrentText(pick("lat"))
    panel.comboBox_variable.setCurrentText(pick("v"))
    qt_app.processEvents()
    assert type(panel.data_item).__name__ == item
    # GL is only ever used for the mesh; never for an image
    assert type(panel.plot.viewport()).__name__ == "QWidget"
    if not rotate:
        assert panel.ixx.ndim == 1 and panel.iyy.ndim == 1   # read as 1-D axes

        # bbox: the view frames the variable (80-85.9 E, 26.1-30 N)
        panel.checkBox_boundingBox.setChecked(True)
        qt_app.processEvents()
        import cartopy.crs as ccrs
        (x0, x1), (y0, y1) = panel.item.vb.viewRange()
        corners = panel.iproj.transform_points(
            ccrs.PlateCarree(), np.array([80.0, 85.9]), np.array([26.1, 30.0]))
        assert x0 <= corners[:, 0].min() and x1 >= corners[:, 0].max()
        assert y0 <= corners[:, 1].min() and y1 >= corners[:, 1].max()
        assert x1 - x0 < 30 and y1 - y0 < 30                # not the world
    session.close()


def test_map_curvilinear_cursor_is_fast_and_exact():
    import time
    import cartopy.crs as ccrs
    from ncv.ncvutils import format_coord_map

    n = 2000                                              # 4M cells
    r, c = np.mgrid[0:n, 0:n].astype(float)
    lon = 80 + c * 0.005 + 0.3 * np.sin(r / 200.0)
    lat = 20 + r * 0.005 + 0.2 * np.sin(c / 250.0)
    zz = np.arange(n * n, dtype=float).reshape(n, n)
    proj = ccrs.PlateCarree()
    rng = np.random.default_rng(1)
    for x, y in rng.uniform((80.5, 20.5), (89.5, 29.5), (60, 2)):
        brute = zz.flat[np.abs((((lon + 360) % 360) - ((x + 360) % 360))**2
                               + (lat - y)**2).argmin()]
        assert format_coord_map(x, y, proj, lon, lat, zz).endswith(f"z={brute:.6g}")
    t0 = time.perf_counter()
    format_coord_map(85.0, 25.0, proj, lon, lat, zz)
    assert time.perf_counter() - t0 < 0.05


def test_contour_shows_spatial_data_north_up_as_stored(tmp_path, qt_app):
    from ncv.ncvcontour import ContourPanel
    from ncv.session import NcvSession

    path = tmp_path / "raster.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", None)
        ds.createDimension("y", 4)
        ds.createDimension("x", 6)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 2000-01-01"
        t[:] = [0.0, 1.0, 2.0]
        y = ds.createVariable("y", "f8", ("y",))
        y[:] = [30.0, 29.0, 28.0, 27.0]                     # north first
        x = ds.createVariable("x", "f8", ("x",))
        x[:] = 80.0 + np.arange(6.0)
        ds.createVariable("elev", "f8", ("y", "x"))[:] = np.arange(24.0).reshape(4, 6)
        ds.createVariable("series", "f8", ("time", "y"))[:] = np.arange(12.0).reshape(3, 4)
    session = NcvSession()
    session.open([str(path)])
    panel = ContourPanel(_panel_window(), session)
    pick = lambda name: next(c for c in session.cols if c.startswith(name + " "))

    panel.comboBox_z.setCurrentText(pick("elev"))
    assert not panel.checkBox_transposeZ.isChecked()        # shown as stored
    assert panel._zz.shape == (4, 6)
    assert list(panel._yy) == [30.0, 29.0, 28.0, 27.0]      # CF coordinate used
    assert list(panel._xx) == list(80.0 + np.arange(6.0))
    rect = panel.image.mapRectToView(panel.image.boundingRect())
    assert rect.top() < rect.bottom() and panel.image.height() == 4
    # row 0 (y = 30) sits at the top: the image rect runs downward from 30.5
    assert panel.image.mapToView(panel.image.boundingRect().topLeft()).y() > 30.0
    assert "z=0" in panel._format_cursor(80.0, 30.0)        # NW corner value
    assert "z=23" in panel._format_cursor(85.0, 27.0)       # SE corner value

    panel.comboBox_z.setCurrentText(pick("series"))          # (time, y)
    assert panel.checkBox_transposeZ.isChecked()             # time along x
    assert panel._zz.shape == (4, 3)
    session.close()


def test_map_mesh_stride():
    from ncv.ncvmap import MESH_MAX_CELLS, decimation_stride

    # only the curved-projection quad mesh is capped; it costs ~3.5 us/cell
    assert decimation_stride((100, 100)) == 1
    stride = decimation_stride((1440, 720))
    assert (1440 // stride) * (720 // stride) <= MESH_MAX_CELLS


def test_memory_budget_and_read_modes(tmp_path, qt_app):
    from ncv.ncvcommon import PlotPanel, memory_budget_cells
    from ncv.session import NcvSession

    budget = memory_budget_cells()
    assert budget > 0
    # scales with memory below the cap
    assert memory_budget_cells(fraction=0.002) > memory_budget_cells(fraction=0.001)

    path = tmp_path / "grid.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("y", 40)
        ds.createDimension("x", 60)
        ds.createVariable("v", "f8", ("y", "x"))[:] = 1.0
        ds.createVariable("r", "f8", ("y", "x"), chunksizes=(1, 60))[:] = 1.0
    session = NcvSession()
    session.open([str(path)])
    window = _panel_window()
    panel = PlotPanel(window, session, "t")
    var = session.fi[0]["v"]
    rows = session.fi[0]["r"]

    # fits in memory -> the whole slice, full resolution
    window, shape, span, mode = panel.read_window(var, ["all", "all"])
    assert mode == "full" and span == shape == (40, 60)
    # coarse only when asked for
    assert panel.read_window(var, ["all", "all"], coarse=True)[3] == "coarse"
    # too big for the budget -> a full-resolution chunk at the offsets
    import ncv.ncvcommon as common
    real = common.memory_budget_cells
    common.memory_budget_cells = lambda *a, **k: 100
    try:
        window, shape, span, mode = panel.read_window(
            var, ["all", "all"], offsets=(20, 30))
        # one-row chunks are read whole: full-width strip, not a square
        strip = panel.read_window(rows, ["all", "all"], offsets=(20, 30))
    finally:
        common.memory_budget_cells = real
    assert mode == "chunk" and span == (10, 10)
    assert window == {0: (20, 30, 1), 1: (30, 40, 1)}
    assert strip[3] == "chunk" and strip[2] == (1, 60)
    assert strip[0] == {0: (20, 21, 1), 1: (0, 60, 1)}
    session.close()


def test_matrix_chunk_headers_and_indices(tmp_path, qt_app):
    import ncv.ncvcommon as common
    from ncv.qt_compat import QtCore
    from ncv.ncvmatrix import MatrixPanel
    from ncv.session import NcvSession

    path = tmp_path / "chunked.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", 50)
        ds.createDimension("lon", 80)
        lat = ds.createVariable("lat", "f8", ("lat",))
        lat.units = "degrees_north"
        lat[:] = np.arange(50.0)
        lon = ds.createVariable("lon", "f8", ("lon",))
        lon.units = "degrees_east"
        lon[:] = 100.0 + np.arange(80.0)
        ds.createVariable("v", "f8", ("lat", "lon"))[:] = np.arange(4000.0).reshape(50, 80)

    real = common.memory_budget_cells
    common.memory_budget_cells = lambda *a, **k: 400        # 20 x 20 chunks
    try:
        session = NcvSession()
        session.open([str(path)])
        window = _panel_window()
        panel = MatrixPanel(window, session)
        vardim = next(c for c in session.cols if c.startswith("v "))
        panel.comboBox_z.setCurrentText(vardim)
        panel.scroll.hbar.setValue(30)
        panel._refresh_table()
        model = panel.model
        horizontal = QtCore.Qt.Orientation.Horizontal
        cols = model.col_index
        # the loaded chunk covers the bar position (it is centred on it)
        assert cols is not None and len(cols) == 20
        assert cols[0] <= 30 <= cols[-1]
        # headers are the chunk's slice of lon, not blank
        assert model.xheaders is not None
        assert np.allclose(model.xheaders, 100.0 + cols)
        # cell indices are positions in the variable, not in the chunk
        panel.checkBox_showCellIndices.setChecked(True)
        assert model.headerData(0, horizontal) == str(cols[0])
        assert model.headerData(19, horizontal) == str(cols[-1])
        # values line up with those indices
        assert float(model.values[0, 0]) == float(cols[0])
        session.close()
    finally:
        common.memory_budget_cells = real


def test_contour_bars_mirror_view_and_read_only_at_edges(tmp_path, qt_app):
    import ncv.ncvcommon as common
    from ncv.ncvcontour import ContourPanel
    from ncv.session import NcvSession

    path = tmp_path / "big.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("y", 200)
        ds.createDimension("x", 300)
        ds.createVariable("v", "f8", ("y", "x"))[:] = np.arange(60000.0).reshape(200, 300)

    real = common.memory_budget_cells
    common.memory_budget_cells = lambda *a, **k: 2500      # 50 x 50 chunks
    try:
        session = NcvSession()
        session.open([str(path)])
        panel = ContourPanel(_panel_window(), session)
        panel.resize(800, 600)
        panel.show()
        qt_app.processEvents()
        panel.comboBox_z.setCurrentText(
            next(c for c in session.cols if c.startswith("v ")))
        qt_app.processEvents()
        reads = []
        original = panel.redraw
        panel.redraw = lambda *a, **k: (reads.append(1), original(*a, **k))

        # thumbs = the displayed region within the whole variable
        rows, cols = panel._view_cells()
        assert panel.scroll.hbar.pageStep() == round(cols[1] - cols[0])
        assert panel.scroll.vbar.pageStep() == round(rows[1] - rows[0])
        assert panel.scroll.hbar.maximum() == 300 - panel.scroll.hbar.pageStep()

        # a small move stays inside the chunk: the view pans, nothing is read
        panel.scroll.hbar.setValue(panel.scroll.hbar.value() + 5)
        qt_app.processEvents()
        assert round(panel._view_cells()[1][0]) == panel.scroll.hbar.value()
        panel._reload.stop()
        assert reads == []

        # past the edge: one read, and the new chunk covers the view
        panel.scroll.hbar.setValue(250)
        qt_app.processEvents()
        assert panel._reload.isActive()
        panel._reload_chunk()
        _full, (_r0, _r1, c0, c1), *_ = panel._geom
        assert len(reads) == 1 and c0 <= 250 < c1
        session.close()
    finally:
        common.memory_budget_cells = real


def test_slice_miss_uses_netcdf_mask_without_rescanning(tmp_path, monkeypatch, qt_app):
    from ncv.ncvcommon import DimensionControlRow, PlotPanel
    from ncv.session import NcvSession

    path = tmp_path / "masked.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("x", 4)
        var = ds.createVariable("v", "i2", ("x",), fill_value=-9999)
        var[:] = np.ma.masked_equal([1, -9999, 3, 4], -9999)
    session = NcvSession()
    session.open([str(path)])
    panel = PlotPanel(_panel_window(), session, "t")
    dims = DimensionControlRow(1)
    dims.set_count(1)
    dims.selectors[0].addItems(["all"])
    calls = []
    monkeypatch.setattr(np, "isin", lambda *a, **k: calls.append(1))
    out = panel.slice_miss(dims, session.fi[0]["v"])
    assert np.isnan(out[1]) and out[0] == 1 and out[3] == 4
    assert calls == []          # netCDF4 already masked the fill value
    session.close()


def test_native_integer_image_is_transparent_where_missing(qt_app):
    import pyqtgraph as pg
    from ncv.ncvcommon import native_levels, set_palette_image

    sentinel = np.iinfo(np.int16).min
    data = np.array([[2, 9, 16], [sentinel, 5, sentinel]], dtype=np.int16)
    levels = native_levels(data, (2, 16))
    assert levels == (2, 16)
    image = pg.ImageItem()
    set_palette_image(image, data, pg.colormap.get("viridis"), *levels)
    # uint8 indices: Qt scales them itself, nothing re-rendered on zoom
    assert image.image.dtype == np.uint8 and image.levels is None
    assert image.image.tolist() == [[1, 128, 255], [0, 55, 0]]
    image.render()
    qimage = image.qimage
    alpha = lambda row, col: qimage.pixelColor(col, row).alpha()
    assert alpha(1, 0) == 0 and alpha(1, 2) == 0            # missing: clear
    assert alpha(0, 0) == alpha(0, 2) == alpha(1, 1) == 255   # valid: opaque
    # the real minimum and maximum take the two ends of the colormap
    ends = pg.colormap.get("viridis").getLookupTable(nPts=255, alpha=False)
    low, high = qimage.pixelColor(0, 0), qimage.pixelColor(2, 0)
    assert (low.red(), low.green(), low.blue()) == tuple(int(v) for v in ends[0])
    assert (high.red(), high.green(), high.blue()) == tuple(int(v) for v in ends[-1])

    # a user range above the data minimum raises valid cells, not the sentinel
    clamped = data.copy()
    native_levels(clamped, (2, 16), low=5)
    assert clamped[0, 0] == 5 and clamped[1, 0] == sentinel

    # floats: NaN is the transparent index, out-of-range values saturate
    set_palette_image(image, np.array([[np.nan, -1.0, 0.5, 9.0]]),
                      pg.colormap.get("viridis"), 0.0, 1.0)
    assert image.image.tolist() == [[0, 1, 128, 255]]


def test_contour_defaults_to_lat_lon_slice(tmp_path, qt_app):
    from ncv.dimensions import dimension_specs
    from ncv.session import NcvSession

    path = tmp_path / "zyx.nc"
    with nc.Dataset(path, "w") as ds:
        for name, size in (("z", 3), ("lat", 4), ("lon", 5)):
            ds.createDimension(name, size)
        ds.createVariable("lat", "f8", ("lat",))[:] = np.linspace(10, 40, 4)
        ds["lat"].units = "degrees_north"
        ds.createVariable("lon", "f8", ("lon",))[:] = np.linspace(0, 40, 5)
        ds["lon"].units = "degrees_east"
        ds.createVariable("v", "f4", ("z", "lat", "lon"))[:] = 1.0
    session = NcvSession()
    session.open([str(path)])
    vardim = next(c for c in session.cols if c.startswith("v "))
    # not depth x lat at lon 0: that slice decompresses every chunk along lat
    assert [s.value for s in dimension_specs(session, vardim, "z")[:3]] == \
        ["0", "all", "all"]
    session.close()


def test_import_ncv_does_not_load_xarray():
    code = "import sys, ncv; assert 'xarray' not in sys.modules, 'xarray'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_contour_keeps_integer_dtype(tmp_path, qt_app):
    from ncv.ncvcontour import ContourPanel
    from ncv.session import NcvSession

    path = tmp_path / "ints.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("y", 20)
        ds.createDimension("x", 30)
        ints = ds.createVariable("classes", "i2", ("y", "x"), fill_value=-9999)
        data = np.ma.masked_array(np.arange(600).reshape(20, 30) % 17 + 2,
                                  mask=np.zeros((20, 30), bool))
        data.mask[:5, :] = True
        ints[:] = data
        ds.createVariable("reals", "f8", ("y", "x"))[:] = 1.5
    session = NcvSession()
    session.open([str(path)])
    panel = ContourPanel(_panel_window(), session)
    pick = lambda name: next(c for c in session.cols if c.startswith(name + " "))

    panel.comboBox_z.setCurrentText(pick("classes"))
    assert panel._zz.dtype == np.int16                       # no float copy
    assert tuple(panel.colorbar.levels()) == (2, 18)         # fill excluded
    assert (panel._zz[:5] == np.iinfo(np.int16).min).all()   # masked rows
    assert "nan" in panel._format_cursor(0.0, 1.0)           # no -32768 shown

    panel.comboBox_z.setCurrentText(pick("reals"))
    assert panel._zz.dtype == np.float64                     # float path as before
    session.close()


def test_matrix_keeps_integer_dtype(tmp_path, qt_app):
    from ncv.qt_compat import QtCore
    from ncv.ncvmatrix import MatrixPanel
    from ncv.session import NcvSession

    path = tmp_path / "ints.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("y", 4)
        ds.createDimension("x", 5)
        var = ds.createVariable("classes", "i2", ("y", "x"), fill_value=-9999)
        data = np.ma.masked_array(np.arange(20).reshape(4, 5) + 3, mask=False)
        data.mask = np.zeros((4, 5), bool)
        data.mask[0, 0] = True
        var[:] = data
    session = NcvSession()
    session.open([str(path)])
    panel = MatrixPanel(_panel_window(), session)
    panel.comboBox_z.setCurrentText(
        next(c for c in session.cols if c.startswith("classes ")))
    model = panel.model
    display = QtCore.Qt.ItemDataRole.DisplayRole
    assert model.values.dtype == np.int16                     # no float copy
    assert model.data(model.index(0, 0), display) == ""       # missing cell
    assert model.data(model.index(0, 1), display) != ""
    # min/max exclude the missing cell (4 is the smallest valid value)
    assert panel.lineEdit_min.text().startswith("4")
    assert panel.lineEdit_max.text().startswith("22")
    # changing the number format must not re-read the variable
    panel.comboBox_dataFormat.setCurrentIndex(
        (panel.comboBox_dataFormat.currentIndex() + 1)
        % panel.comboBox_dataFormat.count())
    assert panel.lineEdit_min.text().startswith("4")
    session.close()


def test_map_opens_global_and_mouse_never_reads(qt_app):
    from ncv.ncvmap import HAVE_CARTOPY
    if not HAVE_CARTOPY:
        pytest.skip("cartopy unavailable")
    import ncv.ncvcommon as common
    import netCDF4 as ncdf
    from ncv.ncvmap import MapPanel
    from ncv.session import NcvSession
    import tempfile, os

    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "grid.nc")
    with ncdf.Dataset(path, "w") as ds:
        ds.createDimension("lat", 90)
        ds.createDimension("lon", 180)
        lat = ds.createVariable("lat", "f8", ("lat",))
        lat.units = "degrees_north"
        lat[:] = np.linspace(-89, 89, 90)
        lon = ds.createVariable("lon", "f8", ("lon",))
        lon.units = "degrees_east"
        lon[:] = np.linspace(-179, 179, 180)
        ds.createVariable("v", "f8", ("lat", "lon"))[:] = 1.0

    real = common.memory_budget_cells
    common.memory_budget_cells = lambda *a, **k: 400        # 20 x 20 chunks
    try:
        session = NcvSession()
        session.open([path])
        panel = MapPanel(_panel_window(), session)
        panel.resize(900, 600)
        panel.show()
        qt_app.processEvents()
        panel.comboBox_variable.setCurrentText(
            next(c for c in session.cols if c.startswith("v ")))
        qt_app.processEvents()
        # opens at the world extent, with the patch drawn inside it
        xr, _yr = panel.item.vb.viewRange()
        assert xr[0] <= -170 and xr[1] >= 170
        assert panel.scroll.hbar.pageStep() == 20        # thumb = the patch
        assert panel.ivv.ndim == 2                        # tiny on screen, still drawn
        reads = []
        original = panel.redraw
        panel.redraw = lambda *a, **k: (reads.append(1), original(*a, **k))
        for _ in range(5):                                # mouse navigation
            panel.item.vb.translateBy(x=10)
            qt_app.processEvents()
        assert reads == []
        panel.scroll.hbar.setValue(100)                   # a bar move reads once
        panel._scrolled()
        assert len(reads) == 1
        assert panel._zwindow[1][0] == 100
        session.close()
    finally:
        common.memory_budget_cells = real


def test_map_drag_reads_once_and_at_screen_resolution(tmp_path, qt_app):
    from ncv.ncvmap import HAVE_CARTOPY
    if not HAVE_CARTOPY:
        pytest.skip("cartopy unavailable")
    from PyQt6.QtTest import QTest
    import ncv.ncvcommon as common
    from ncv.ncvmap import MapPanel
    from ncv.session import NcvSession

    path = tmp_path / "fine.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", 900)
        ds.createDimension("lon", 1800)
        lat = ds.createVariable("lat", "f8", ("lat",))
        lat.units = "degrees_north"
        lat[:] = np.linspace(-89.9, 89.9, 900)
        lon = ds.createVariable("lon", "f8", ("lon",))
        lon.units = "degrees_east"
        lon[:] = np.linspace(-179.9, 179.9, 1800)
        ds.createVariable("v", "i2", ("lat", "lon"))[:] = 7

    real = common.memory_budget_cells
    common.memory_budget_cells = lambda *a, **k: 250000     # 500 x 500 patch
    try:
        session = NcvSession()
        session.open([str(path)])
        panel = MapPanel(_panel_window(), session)
        panel.resize(900, 600)
        panel.show()
        qt_app.processEvents()
        panel.comboBox_variable.setCurrentText(
            next(c for c in session.cols if c.startswith("v ")))
        qt_app.processEvents()
        # at world view the screen can't show every cell: read coarser
        assert panel._read_stride > 1
        assert panel.ivv.shape[1] < 500

        reads = []
        original = panel.redraw
        panel.redraw = lambda *a, **k: (reads.append(1), original(*a, **k))
        bar = panel.scroll.hbar
        bar.setSliderDown(True)
        for _ in range(10):                   # a held drag with pauses
            bar.setSliderPosition(bar.value() + 50)
            QTest.qWait(60)
        assert reads == []                    # nothing read while dragging
        bar.setSliderDown(False)              # emits sliderReleased
        QTest.qWait(80)
        assert len(reads) == 1                # one read, on release

        # zoom right in: once settled, re-read at full resolution
        box = panel.data_item.mapRectToView(panel.data_item.boundingRect())
        panel.item.vb.setRange(
            xRange=(box.center().x() - 0.5, box.center().x() + 0.5),
            yRange=(box.center().y() - 0.5, box.center().y() + 0.5), padding=0)
        QTest.qWait(350)
        assert panel._read_stride == 1 and len(reads) == 2
        session.close()
    finally:
        common.memory_budget_cells = real


def test_map_tiled_patch_is_read_once_at_full_resolution(tmp_path, qt_app):
    from ncv.ncvmap import HAVE_CARTOPY
    if not HAVE_CARTOPY:
        pytest.skip("cartopy unavailable")
    from PyQt6.QtTest import QTest
    import ncv.ncvcommon as common
    from ncv.ncvmap import MapPanel
    from ncv.session import NcvSession

    path = tmp_path / "tiled.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", 900)
        ds.createDimension("lon", 1800)
        lat = ds.createVariable("lat", "f8", ("lat",))
        lat.units = "degrees_north"
        lat[:] = np.linspace(-89.9, 89.9, 900)
        lon = ds.createVariable("lon", "f8", ("lon",))
        lon.units = "degrees_east"
        lon[:] = np.linspace(-179.9, 179.9, 1800)
        ds.createVariable("v", "i2", ("lat", "lon"), chunksizes=(100, 100),
                          zlib=True)[:] = 7

    real = common.memory_budget_cells
    common.memory_budget_cells = lambda *a, **k: 250000     # 500 x 500 patch
    try:
        session = NcvSession()
        session.open([str(path)])
        panel = MapPanel(_panel_window(), session)
        panel.resize(900, 600)
        panel.show()
        qt_app.processEvents()
        panel.comboBox_variable.setCurrentText(
            next(c for c in session.cols if c.startswith("v ")))
        qt_app.processEvents()
        # a stride inside a 100 x 100 tile decompresses every tile anyway
        assert panel._full_patch and panel._read_stride == 1
        assert panel.ivv.shape == (500, 500)

        reads = []
        original = panel.redraw
        panel.redraw = lambda *a, **k: (reads.append(1), original(*a, **k))
        vb = panel.item.vb
        for factor in (0.2, 5.0, 3.0):        # zoom in, back out, out more
            vb.scaleBy((factor, factor))
            QTest.qWait(350)                  # let each zoom settle
        assert reads == []                    # Qt rescales; nothing re-read
        session.close()
    finally:
        common.memory_budget_cells = real


def test_map_cursor_is_fast_and_matches_brute_force():
    import time
    import cartopy.crs as ccrs
    from ncv.ncvutils import format_coord_map

    proj = ccrs.PlateCarree()
    lon = np.linspace(-179.5, 179.5, 360)
    lat = np.linspace(-89.5, 89.5, 180)
    zz = np.arange(360 * 180, dtype=float).reshape(180, 360)
    gx, gy = np.meshgrid(lon, lat)
    rng = np.random.default_rng(0)
    for x, y in list(rng.uniform((-180, -90), (180, 90), (200, 2))) + [(179.9, 0.0), (-179.9, 0.0)]:
        brute = zz.flat[np.abs((((gx + 360.) % 360.) - ((x + 360.) % 360.))**2
                               + (gy - y)**2).argmin()]
        assert format_coord_map(x, y, proj, lon, lat, zz).endswith(f"z={brute:.6g}")

    # a full-resolution patch must not stall the mouse
    big = np.zeros((5000, 5000), dtype=np.int16)
    t0 = time.perf_counter()
    format_coord_map(1.0, 1.0, proj, np.linspace(0, 5, 5000),
                     np.linspace(0, 5, 5000), big)
    assert time.perf_counter() - t0 < 0.05


def test_file_menu_actions(tmp_path, qt_app):
    from ncv.app import NcvMainWindow
    from ncv.session import NcvSession

    window = NcvMainWindow(NcvSession())
    calls = []
    window.open_file_dialog = lambda use_xarray: calls.append(use_xarray)
    # the lambdas were connected before the patch; route through them
    window.actionOpen_File.triggered.disconnect()
    window.actionOpen_xarray.triggered.disconnect()
    window.actionOpen_File.triggered.connect(lambda: window.open_file_dialog(False))
    window.actionOpen_xarray.triggered.connect(lambda: window.open_file_dialog(True))
    window.actionOpen_File.trigger()
    window.actionOpen_xarray.trigger()
    assert calls == [False, True]
    before = len(NcvMainWindow.instances)
    window.actionNew_Window.trigger()
    assert len(NcvMainWindow.instances) == before + 1
    for win in list(NcvMainWindow.instances):
        win.close()


def test_file_menu_is_wired_at_startup(qt_app, monkeypatch):
    from ncv.app import NcvMainWindow
    from ncv.session import NcvSession

    calls = []
    monkeypatch.setattr(NcvMainWindow, "open_file_dialog",
                        lambda self, use_xarray: calls.append(use_xarray))
    window = NcvMainWindow(NcvSession())
    window.actionOpen_File.trigger()
    window.actionOpen_xarray.trigger()
    assert calls == [False, True]
    window.close()


def test_matrix_outer_bar_picks_chunk_inner_slides_table(tmp_path, qt_app):
    import ncv.ncvcommon as common
    from PyQt6 import QtCore as QtC
    from ncv.ncvmatrix import MatrixPanel
    from ncv.session import NcvSession

    path = tmp_path / "grid.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("y", 400)
        ds.createDimension("x", 400)
        ds.createVariable("v", "f8", ("y", "x"))[:] = np.arange(160000.0).reshape(400, 400)
    real = common.memory_budget_cells
    common.memory_budget_cells = lambda *a, **k: 10000       # 100 x 100 chunks
    try:
        session = NcvSession()
        session.open([str(path)])
        panel = MatrixPanel(_panel_window(), session)
        panel.resize(900, 700)
        panel.show()
        qt_app.processEvents()
        panel.comboBox_z.setCurrentText(next(c for c in session.cols if c.startswith("v ")))
        qt_app.processEvents()
        table = panel.tableView_showMatrix
        assert table.verticalScrollBarPolicy() != QtC.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        reads = []
        original = panel._refresh_table
        panel._refresh_table = lambda *a, **k: (reads.append(1), original(*a, **k))
        from PyQt6.QtTest import QTest
        outer = panel.scroll.vbar
        # outer thumb = the chunk (100 of 400 rows), inner range = the chunk
        assert outer.pageStep() == 100 and outer.maximum() == 300
        inner = table.verticalScrollBar()
        assert inner.maximum() > 0

        # inner: slides the table, leaves the outer bar and the chunk alone
        inner.setValue(30)
        QTest.qWait(60)
        assert table.rowAt(0) == 30
        assert outer.value() == 0 and reads == []

        # outer: loads the chunk starting there, table back at its top
        outer.setValue(150)
        QTest.qWait(60)
        assert len(reads) == 1 and panel._zwindow[0][0] == 150
        assert inner.value() == 0

        # a held outer drag reads nothing until release, then once
        outer.setSliderDown(True)
        for step in range(5):
            outer.setSliderPosition(160 + 10 * step)
            QTest.qWait(60)
        assert len(reads) == 1
        outer.setSliderDown(False)
        QTest.qWait(60)
        assert len(reads) == 2 and panel._zwindow[0][0] == 200
        session.close()
    finally:
        common.memory_budget_cells = real


def test_matrix_hover_readout(tmp_path, qt_app):
    from ncv.ncvmatrix import MatrixPanel
    from ncv.session import NcvSession

    path = tmp_path / "small.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("y", 3)
        ds.createDimension("x", 4)
        ds.createVariable("v", "f8", ("y", "x"))[:] = np.arange(12.0).reshape(3, 4)
    session = NcvSession()
    session.open([str(path)])
    panel = MatrixPanel(_panel_window(), session)
    panel.comboBox_z.setCurrentText(next(c for c in session.cols if c.startswith("v ")))
    panel.checkBox_showCellIndices.setChecked(True)
    table = panel.tableView_showMatrix
    # 'Cursor' unticked: no tracking, no read-out
    assert not table.viewport().hasMouseTracking()
    table.entered.emit(panel.model.index(2, 3))
    assert panel.label_cursor.text() == ""
    panel.checkBox_cursor.setChecked(True)
    assert table.viewport().hasMouseTracking()
    table.entered.emit(panel.model.index(2, 3))
    assert panel.label_cursor.text() == "x=3, y=2, z=11.0"
    panel.checkBox_cursor.setChecked(False)
    assert panel.label_cursor.text() == ""
    session.close()


def test_read_responsive_runs_off_the_gui_thread(qt_app):
    import threading
    from ncv.ncvcommon import read_responsive, reading

    seen = {}

    def read():
        seen["main"] = threading.current_thread() is threading.main_thread()
        seen["busy"] = reading()
        return 42

    assert read_responsive(read) == 42
    assert seen == {"main": False, "busy": True}
    assert not reading()

    def fail():
        raise ValueError("bad read")

    with pytest.raises(ValueError, match="bad read"):
        read_responsive(fail)
    assert not reading()


def test_metadata_toggle_restores_width(qt_app):
    from ncv.ncvmatrix import MatrixPanel
    from ncv.session import NcvSession

    window = _panel_window()
    panel = MatrixPanel(window, NcvSession())
    panel.resize(1000, 600)
    panel.show()
    qt_app.processEvents()
    button = panel.pushButton_metadata
    assert button.text().endswith("\u25b6")
    button.setChecked(True)
    qt_app.processEvents()
    assert button.text().endswith("\u25b2")
    panel._header_splitter.setSizes([600, 380])
    qt_app.processEvents()
    chosen = panel._header_splitter.sizes()
    button.setChecked(False)
    button.setChecked(True)
    qt_app.processEvents()
    assert panel._header_splitter.sizes() == chosen

@pytest.mark.parametrize("error, reason, advice", [
    # an HPC node without Qt's system libraries
    ("libEGL.so.1: cannot open shared object file", "libEGL.so.1",
     "conda install -c conda-forge pyqt6"),
    # pip PyQt6 wheel + system FreeType 2.10 + conda HarfBuzz
    ("libharfbuzz.so.0: undefined symbol: FT_Get_Colorline_Stops",
     "FT_Get_Colorline_Stops", "LD_PRELOAD=$CONDA_PREFIX/lib/libfreetype.so.6"),
])
def test_broken_pyqt6_reports_the_real_reason(error, reason, advice):
    code = """
import builtins, sys
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.startswith("PyQt6"):
        raise ImportError(%r)
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import ncv                              # the package itself stays importable
try:
    import ncv.app
except RuntimeError as exc:
    print("RUNTIME:", exc)
except Exception as exc:
    print("OTHER:", type(exc).__name__, exc)
""" % error
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=Path(__file__).parents[1])
    out = result.stdout
    assert "RUNTIME: ncv requires PyQt6" in out, out + result.stderr
    assert reason in out                                # the real reason
    assert advice in out
    assert "pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip" in out
    assert "conda-forge pyqt\n" not in out             # that one is PyQt5
    assert "NoneType" not in out


def test_chunk_budget_is_capped_and_respects_cgroup(tmp_path, monkeypatch):
    import ncv.ncvcommon as common

    # plenty of memory: the interactivity cap decides
    assert common.memory_budget_cells(fraction=1e9) == common.CHUNK_MAX_CELLS

    # an HPC-style cgroup tree: the parent caps 2 GiB, 1 GiB already in use
    gib = 1 << 30
    leaf = tmp_path / "user.slice" / "user-1.slice"
    leaf.mkdir(parents=True)
    (tmp_path / "user.slice" / "memory.max").write_text(str(2 * gib))
    (tmp_path / "user.slice" / "memory.current").write_text(str(gib))
    (leaf / "memory.max").write_text("max")               # unlimited level
    (leaf / "memory.current").write_text("123")
    proc = tmp_path / "cgroup"
    proc.write_text("0::/user.slice/user-1.slice\n")
    assert common.cgroup_available(str(tmp_path), str(proc)) == gib

    # nothing readable -> unknown, today's behaviour
    assert common.cgroup_available(str(tmp_path), str(tmp_path / "nope")) is None

    # the budget follows the cgroup room when it is tighter than the node's
    # MemAvailable (compare against this machine's own, unlimited, budget)
    monkeypatch.setattr(common, "cgroup_available", lambda: None)
    node = common.memory_budget_cells()
    monkeypatch.setattr(common, "cgroup_available", lambda: gib // 8)
    assert common.memory_budget_cells() == min(node, int(gib // 8 * 0.2 / 32))
    monkeypatch.setattr(common, "cgroup_available", lambda: 0)
    assert common.memory_budget_cells() == 1               # no room at all


def test_cli_help_uses_ncv_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "ncv", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "netcdf_file" in result.stdout
    assert "ncv file.nc" in result.stdout                  # how to open a file
    assert "xcb-util-cursor" in result.stdout              # X11 forwarding fix

    # --help must work where Qt can't load (a broken HPC environment)
    code = (
        "import builtins, sys\n"
        "real = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name.startswith('PyQt6'): raise ImportError('no Qt here')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "sys.argv = ['ncv', '--help']\n"
        "from ncv.__main__ import main\n"
        "main()\n")
    blocked = subprocess.run([sys.executable, "-c", code], text=True,
                             capture_output=True, cwd=Path(__file__).parents[1])
    assert blocked.returncode == 0, blocked.stderr
    assert "examples:" in blocked.stdout
