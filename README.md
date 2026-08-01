# ncv

`ncv` is a PyQt5 desktop application for quickly inspecting NetCDF files.
It combines interactive dimension selection, Matplotlib plots, an optional
Cartopy map view, and a formatted array/metadata viewer in one application.

The viewer can be started with one or more files from the command line, or
without arguments so that files can be opened from the interface.

## Features

- NetCDF4 is the default file-reading backend.
- Optional xarray loading for single-file and multi-file datasets.
- Variables are listed together with their dimension names and sizes.
- Dimensions can be retained, indexed, or reduced with common statistical
  operations.
- Missing values are recognized from NetCDF metadata and an optional
  command-line value.
- CF-style time coordinates are decoded when possible.
- Recognized longitude and latitude variables are selected automatically.
- Four coordinated views are available:
  - Scatter/Line
  - Contour
  - Map
  - Matrix and metadata
- New windows can share the currently opened dataset.
- The graphical interface is maintained as editable Qt Designer forms.

## Requirements

- Python 3.9 or newer
- A graphical desktop environment capable of running Qt applications

The core installation includes:

- Matplotlib
- netCDF4
- NumPy
- PyQt5

The following components are optional:

| Extra | Adds |
| --- | --- |
| `map` | Cartopy and the fully functional Map tab |
| `xarray` | The xarray file-reading path |
| `full` | Both Cartopy and xarray |
| `test` | pytest for development and validation |

Without Cartopy, `ncv` still starts normally. The Map tab remains present and
shows an explanation that mapping is unavailable. Without xarray, the normal
NetCDF4 reader remains available and the **Open xarray** button is hidden.

## Installation

### Install the complete application

Installing the `full` extra enables both the Map tab and xarray support:

```bash
python3 -m pip install "ncv[full]"
```

For the smaller core installation:

```bash
python3 -m pip install ncv
```

Individual optional features can be installed with:

```bash
python3 -m pip install "ncv[map]"
python3 -m pip install "ncv[xarray]"
```

Using a virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install "ncv[full]"
```

### Install as an isolated application with pipx

If `pipx` is available, it can keep the application separate from other
Python environments:

```bash
pipx install "ncv[full]"
pipx ensurepath
```

### Install the current source checkout

To install the current repository version:

```bash
git clone https://github.com/SanjeevBashyal/ncv.git
cd ncv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install ".[full]"
```

Use the source installation when you need changes that have not yet reached a
published package release.

For an editable development installation with tests:

```bash
python3 -m pip install -e ".[full,test]"
```

## Starting ncv

Open a NetCDF file directly:

```bash
ncv sample.nc
```

Start without a file and use **Open File** in the interface:

```bash
ncv
```

Open several files with the default NetCDF4 backend:

```bash
ncv run_01.nc run_02.nc
```

Use xarray instead of the default reader:

```bash
ncv --xarray sample.nc
```

Treat an additional numeric value as missing data:

```bash
ncv --miss -9999 sample.nc
```

The module launcher is equivalent and is useful when the console command is
not on `PATH`:

```bash
python3 -m ncv sample.nc
```

### Command-line reference

```text
ncv [-h] [-m missing_value] [-x] [netcdf_file ...]
```

| Argument | Description |
| --- | --- |
| `netcdf_file ...` | Zero or more NetCDF paths to open |
| `-m`, `--miss` | Additional floating-point missing value; default is `NaN` |
| `-x`, `--xarray` | Read through xarray instead of netCDF4 |
| `-h`, `--help` | Show command-line help |

## Basic workflow

1. Open one or more datasets from the command line or with **Open File**.
2. Choose one of the four tabs.
3. Select the variables to display.
4. Use the dimension controls beside each variable to select the required
   slice or reduction.
5. Adjust the plot or table display controls.

Each dimension selector can contain:

- `all` to retain the dimension;
- a zero-based integer to select one element; or
- `mean`, `std`, `min`, `max`, `ptp`, `sum`, `median`, or `var` to reduce that
  dimension.

Select or reduce enough dimensions for the requested display. Scatter data
must reduce to compatible one-dimensional arrays. Contour, Map, and Matrix
data should have no more than two displayed dimensions.

The **New Window** button opens another viewer backed by the same session.
Opening a new file in one of those windows refreshes all windows sharing that
session.

## Scatter/Line tab

The Scatter/Line tab displays one or two series against a shared X axis.

- X is optional; leaving it empty uses the sample index.
- Y is drawn on the left axis.
- Y2 is drawn on an independent right axis.
- Either Y axis and the X axis can be inverted.
- Y and Y2 can use independent limits or a common range.
- Line style, width, color, marker, marker size, marker fill, marker edge, and
  marker edge width can be set independently for Y and Y2.
- Datetime coordinates are supported on plot axes.
- The embedded Matplotlib toolbar provides pan, zoom, navigation, and image
  saving.

The `xlim`, `ylim`, and `y2lim` fields accept either `min, max` or
`(min, max)`. Examples:

```text
None
(0, 100)
None, 100
2025-01-01, 2025-12-31
```

`None` enables automatic scaling, and either bound can independently be
`None`.

## Contour tab

The Contour tab displays a two-dimensional Z slice.

- X and Y coordinates are optional; indices are used when they are empty.
- Z can be transposed.
- X and Y axes can be inverted.
- Filled-contour and gridded mesh modes are available.
- Matplotlib colormaps can be selected and reversed.
- Grid lines can be enabled.
- The Matplotlib navigation toolbar is included.

The `zlim` field uses the same `min, max` syntax as the Scatter limits. The
bounds control the displayed color range and clip values outside that range.

## Map tab

The Map tab displays a two-dimensional variable with Cartopy. Install the
`map` or `full` extra to enable it.

- Recognized longitude and latitude variables are selected automatically.
- One-dimensional and two-dimensional coordinate arrays are supported.
- If coordinates are left empty, a regular global longitude/latitude grid is
  generated from the data shape.
- The plotted variable can be transposed.
- Longitude and latitude can be inverted.
- Longitude data can be shifted by half the grid width.
- Minimum and maximum color limits can be entered manually.
- The **all** option forces the range calculation to inspect the complete
  variable; large variables may otherwise be sampled when calculating their
  initial range.
- Smooth contour and gridded mesh modes are available.
- Colormaps can be selected and reversed.
- Global/cyclic display, coastlines, borders, rivers, lakes, and grid lines
  can be enabled independently.
- Central longitude can be detected automatically or entered explicitly.
- Projection choices include Plate Carrée, Mercator, Robinson, Mollweide,
  Lambert projections, polar stereographic projections, Eckert I–VI, and
  several other Cartopy projections.
- The Matplotlib navigation toolbar is included.

Cartopy may retrieve Natural Earth feature data the first time coastlines or
other geographic features are requested.

## Matrix tab

The Matrix tab combines a read-only array table with NetCDF metadata.

- Z supplies the table cells.
- X supplies the horizontal column headers.
- Y supplies the vertical row headers.
- Recognized longitude and latitude variables are used as default X and Y
  coordinates.
- Scalar, one-dimensional, and two-dimensional selections are supported.
- If more than two dimensions remain, the table asks for additional
  selection or reduction instead of reshaping the data implicitly.
- Coordinate-size mismatches fall back to zero-based source indices.
- Missing and `NaN` values are displayed as blank cells.
- Non-numeric and datetime values are displayed as text.
- Cell values and coordinate headers have independent fixed-point or
  scientific-notation formats with zero to eight decimal places.
- The table can be flipped top-to-bottom or left-to-right. Data and headers
  remain synchronized.
- **Show cell indices** replaces coordinate headers with zero-based source
  indices.
- The read-only minimum and maximum fields show the current Z slice. Enabling
  **all** calculates the range from the complete variable.

When no Z variable is selected, the text browser shows dataset metadata,
including dimensions, variables, groups, and global attributes. Selecting Z
changes it to the selected variable's dimensions, shape, data type, and
attributes.

## Time controls

Map and Matrix provide shared time navigation whenever the selected variable
contains the detected time dimension. Both unlimited and fixed-size time
dimensions are supported. The controls are disabled for static variables.

| Control | Action |
| --- | --- |
| `|<<` | First frame |
| `|<` | Previous frame |
| `<` | Run backward; click the active button again to pause |
| `>` | Run forward; click the active button again to pause |
| `>|` | Next frame |
| `>>|` | Last frame |

The boundary mode determines what animation does at the first or last frame:

- `once` stops;
- `repeat` wraps to the other end; and
- `reflect` reverses direction.

The slider, dimension selector, displayed frame, and `Time:` label remain
synchronized. In Matrix, time-dependent X and Y headers follow the selected Z
frame.

## File-reading behavior

The default NetCDF4 backend supports either:

- one file, including a file containing NetCDF groups; or
- multiple files without groups.

Group names and multi-file identifiers are included in the variable list.
Opening multiple files when one contains groups is rejected with a clear
error, because combining those two naming schemes would be ambiguous.

The xarray path uses `xarray.open_dataset` for one file and
`xarray.open_mfdataset` for multiple files. Multi-file xarray workflows may
require additional packages used by the local xarray installation, such as
Dask.

Missing-data handling combines:

- the value passed with `--miss`;
- `_FillValue`;
- `missing_value`; and
- the default NetCDF fill value for the variable type.

## Python launcher

The viewer can also be started from Python:

```python
from ncv import ncv

ncv("sample.nc")
```

Multiple files and the optional arguments are also accepted:

```python
ncv(["run_01.nc", "run_02.nc"], miss=-9999, usex=False)
```

## Limitations

- `ncv` is a viewer; the Matrix table does not edit NetCDF values.
- Plot images can be saved through the Matplotlib toolbar, but there is no
  general data-export command.
- High-dimensional variables must be sliced or reduced to the dimensionality
  required by the selected view.
- Full-variable range calculations can be expensive for very large arrays.
- The Map tab requires Cartopy and may require Cartopy's geographic feature
  data on first use.

## Editing the interface with Qt Designer

Qt Designer is only required when modifying the interface. The editable forms
are stored in [`ncv/ui`](ncv/ui):

- `main_window.ui`
- `scatter_panel.ui`
- `contour_panel.ui`
- `map_panel.ui`
- `map_unavailable.ui`
- `matrix_panel.ui`

Generated PyQt5 modules are stored in [`ncv/pyui`](ncv/pyui). Do not edit a
generated `ui_*.py` file manually.

Open a form with Qt Designer, for example:

```bash
designer ncv/ui/matrix_panel.ui
```

The executable may be named `designer` or `qt5-designer`, depending on the
platform installation.

After saving any form, regenerate all Python UI modules from the repository
root:

```bash
./convert-ui-to-py.sh
```

The conversion script requires `pyuic5`. Matplotlib canvases, toolbars,
dimension selectors, Cartopy axes, and the Matrix model are attached to the
generated forms at runtime.

Widget object names are the contract between the Designer files and the
controllers. If an object is intentionally renamed or removed, update its
Python references and tests at the same time.

The main implementation modules are:

| Module | Responsibility |
| --- | --- |
| `ncv/app.py` | Main window, tabs, file dialogs, and shared windows |
| `ncv/session.py` | File lifecycle and session state |
| `ncv/dimensions.py` | Toolkit-neutral dimension selector specifications |
| `ncv/ncvcommon.py` | Shared Qt panel and time-control behavior |
| `ncv/ncvscatter.py` | Scatter/Line tab |
| `ncv/ncvcontour.py` | Contour tab |
| `ncv/ncvmap.py` | Map tab and optional-Cartopy fallback |
| `ncv/ncvmatrix.py` | Matrix table and metadata tab |

## Development and validation

Install the project in editable mode with all optional features and tests:

```bash
python3 -m pip install -e ".[full,test]"
```

Compile the Python modules and run the offscreen Qt test suite:

```bash
python3 -m py_compile ncv/*.py ncv/pyui/*.py
QT_QPA_PLATFORM=offscreen python3 -m pytest
```

Run the application normally for a manual check:

```bash
python3 -m ncv sample.nc
```

If the `ncv` command is not found after installation, activate the environment
where it was installed, inspect `command -v ncv`, or use
`python3 -m ncv` from that environment.
