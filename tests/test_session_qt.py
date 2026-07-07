import os
import subprocess
import sys

import netCDF4 as nc
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def make_sample(path):
    ds = nc.Dataset(path, "w")
    ds.createDimension("time", None)
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

    time[:] = [0, 1]
    lat[:] = [-45, 0, 45]
    lon[:] = [0, 90, 180, 270]
    temp[:] = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    ds.close()


def test_import_is_cartopy_safe():
    import ncv

    assert callable(ncv.ncv)
    assert not hasattr(ncv, "ncvue")


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
    from ncv.app import HAVE_CARTOPY, MapPanel, MapUnavailablePanel, NcvMainWindow
    from ncv.qt_compat import QtWidgets
    from ncv.session import NcvSession

    path = tmp_path / "sample.nc"
    make_sample(path)
    session = NcvSession()
    session.open([str(path)])

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = NcvMainWindow(session)
    temp = next(col for col in session.cols if col.startswith("temp "))

    win.scatter.y.setCurrentText(temp)
    win.scatter.selected_y()
    win.scatter.redraw()
    assert len(win.scatter.line_y) == 1

    win.contour.z.setCurrentText(temp)
    win.contour.selected_z()
    assert len(win.contour.figure.axes) >= 1

    if not HAVE_CARTOPY:
        assert isinstance(win.map, MapUnavailablePanel)
    else:
        assert isinstance(win.map, MapPanel)

    win.close()
    app.processEvents()


def test_cli_help_uses_ncv_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "ncv", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "netcdf_file" in result.stdout
