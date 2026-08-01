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

    forbidden = {"tkinter", "customtkinter"}
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

    assert win.objectName() == "NcvMainWindow"
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
    assert len(win.scatter.line_y) == 1

    win.contour.comboBox_z.setCurrentText(temp)
    win.contour.selected_z()
    assert len(win.contour.figure.axes) >= 1

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

    class PanelWindow(QtWidgets.QWidget):
        def open_file_dialog(self, _use_xarray):
            pass

        def create_secondary_window(self):
            pass

    window = PanelWindow()
    panel = MatrixPanel(window, session)
    return panel, window, session


def select_matrix_variable(panel, session, name):
    item = next(column for column in session.cols if column.startswith(f"{name} "))
    panel.comboBox_z.setCurrentText(item)
    assert panel.comboBox_z.currentText() == item


def matrix_display(panel, row, column):
    from ncv.qt_compat import QtCore

    model = panel.tableView_showMatrix.model()
    return model.data(model.index(row, column), QtCore.Qt.DisplayRole)


def matrix_header(panel, section, orientation):
    from ncv.qt_compat import QtCore

    model = panel.tableView_showMatrix.model()
    return model.headerData(section, orientation, QtCore.Qt.DisplayRole)


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
        assert matrix_header(panel, 1, QtCore.Qt.Horizontal) == "90.0"
        assert matrix_header(panel, 0, QtCore.Qt.Vertical) == "-45.0"
        assert float(panel.lineEdit_min.text()) == 0
        assert float(panel.lineEdit_max.text()) == 11

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
        assert matrix_header(panel, 1, QtCore.Qt.Horizontal) == "90"
        assert matrix_header(panel, 0, QtCore.Qt.Vertical) == "-45"

        panel.checkBox_flipTableLeftRight.setChecked(True)
        assert matrix_display(panel, 0, 0) == "3.00E+00"
        assert matrix_header(panel, 0, QtCore.Qt.Horizontal) == "270"

        panel.checkBox_flipTableTopBottom.setChecked(True)
        assert matrix_display(panel, 0, 0) == "1.10E+01"
        assert matrix_header(panel, 0, QtCore.Qt.Vertical) == "45"

        panel.checkBox_showCellIndices.setChecked(True)
        assert matrix_header(panel, 0, QtCore.Qt.Horizontal) == "3"
        assert matrix_header(panel, 0, QtCore.Qt.Vertical) == "2"
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


def test_qt_designer_forms_compile():
    from PyQt5 import uic
    from ncv.pyui.ui_contour_panel import Ui_widget_ContourPanel
    from ncv.pyui.ui_main_window import Ui_NcvMainWindow
    from ncv.pyui.ui_map_panel import Ui_MapPanel
    from ncv.pyui.ui_map_unavailable import Ui_MapUnavailablePanel
    from ncv.pyui.ui_matrix_panel import Ui_widget_matrixPanel
    from ncv.pyui.ui_scatter_panel import Ui_ScatterPanel

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
        form_class, base_class = uic.loadUiType(str(ui_dir / form))
        assert form_class is not None
        assert base_class is not None

    assert all(ui_class is not None for ui_class in (
        Ui_NcvMainWindow,
        Ui_ScatterPanel,
        Ui_widget_ContourPanel,
        Ui_MapPanel,
        Ui_MapUnavailablePanel,
        Ui_widget_matrixPanel,
    ))


def test_cli_help_uses_ncv_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "ncv", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "netcdf_file" in result.stdout
