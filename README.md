# ncv

`ncv` is a PyQt6 desktop application for quickly inspecting NetCDF files.
It combines interactive dimension selection, pyqtgraph plots, an optional
Cartopy-projected map view, and a formatted array/metadata viewer in one
application.

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
- Temporal aggregation: sum, mean, mode, median, minimum or maximum per hour,
  day, month, year or the entire period.
- Double-click a Contour or Map cell to plot its time series.
- Colour scales: linear, log, low/high stretch, banded and equalised.
- The Matrix can follow the Map's variable, time and position.
- New windows can share the currently opened dataset.
- The graphical interface is maintained as editable Qt Designer forms.

## Requirements

- Python 3.9 or newer
- A graphical desktop environment capable of running Qt applications

The installation includes everything needed for all four tabs:

- Cartopy
- netCDF4
- NumPy
- PyQt6
- pyqtgraph
- xarray

The only extra is `test`, which adds pytest for development and validation.

`ncv` still starts if Cartopy or xarray fail to import on a broken system. The
Map tab remains present and explains that mapping is unavailable, and the
**Open xarray** button is hidden.

## Installation

```bash
python3 -m pip install ncv
```

Using a virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install ncv
```

### Install into a conda environment

Install PyQt6 from conda-forge **before** ncv, so pip doesn't fetch the PyQt6
wheel. That wheel's Qt libraries only search their own folder for system
libraries, so in a conda environment they can mix the system's older FreeType
with conda's newer HarfBuzz and fail with
`undefined symbol: FT_Get_Colorline_Stops`.

```bash
conda install -c conda-forge pyqt6      # conda-forge "pyqt" is PyQt5
python3 -m pip install ncv
```

If the wheel is already installed, swap it out (pip first, as both write to
`site-packages/PyQt6`):

```bash
pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
conda install -c conda-forge pyqt6
```

### Install as an isolated application with pipx

If `pipx` is available, it can keep the application separate from other
Python environments:

```bash
pipx install ncv
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
python3 -m pip install .
```

Use the source installation when you need changes that have not yet reached a
published package release.

For an editable development installation with tests:

```bash
python3 -m pip install -e ".[test]"
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

**File > New Window** opens another viewer showing the same dataset. Opening
a file in either window changes only that window; the other keeps its file.

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
- pyqtgraph provides pan, zoom, and a right-click export menu for image
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

When X is the time variable, the **aggregate** combos resample Y and Y2: each
point is the statistic over one period, drawn at the period's start. See
[Temporal aggregation](#temporal-aggregation).

## Contour tab

The Contour tab displays a two-dimensional Z slice.

- X and Y coordinates are optional; indices are used when they are empty.
- Z can be transposed.
- X and Y axes can be inverted.
- The field is drawn as a heat map, one image cell per grid cell.
- pyqtgraph colormaps can be selected and reversed, with a
  [colour scale](#colour-scales).
- Grid lines can be enabled.
- Drag to pan, scroll to zoom; right-click for the export menu.
- [Time controls](#time-controls) step the time dimension while it is a
  selected index; they are off while time is one of the plotted axes.
- With **Cursor** ticked, double-clicking a cell plots its
  [time series](#cell-time-series).

The `zlim` field uses the same `min, max` syntax as the Scatter limits. The
bounds control the displayed color range and clip values outside that range.

## Map tab

The Map tab displays a two-dimensional variable with Cartopy, which is
installed with ncv. If Cartopy cannot be imported, the tab explains why and
the other tabs keep working.

- Recognized longitude and latitude variables are selected automatically.
- One-dimensional and two-dimensional coordinate arrays are supported.
- If coordinates are left empty, a regular global longitude/latitude grid is
  generated from the data shape.
- The plotted variable can be transposed.
- Longitude and latitude can be inverted.
- Longitude data can be shifted by half the grid width.
- Minimum and maximum color limits can be entered manually.
- The **all** option recalculates the colour range from the whole variable:
  read in full when it fits the memory budget, otherwise from a sample of
  whole chunks spread over it.
- Large grids are downsampled to stay responsive; `full res` draws every
  cell.
- Colormaps can be selected and reversed, with a
  [colour scale](#colour-scales).
- With **Cursor** ticked, double-clicking a cell plots its
  [time series](#cell-time-series).
- With **matrix** ticked, switching to the Matrix tab shows the same variable,
  selections, time step and aggregation, centred on the cell at the middle of
  the map view.
- Global/cyclic display, coastlines, borders, rivers, lakes, and grid lines
  can be enabled independently.
- Central longitude can be detected automatically or entered explicitly.
- Projection choices include Plate Carrée, Mercator, Robinson, Mollweide,
  Lambert projections, polar stereographic projections, Eckert I–VI, and
  several other Cartopy projections.
- Drag to pan, scroll to zoom; right-click for the export menu.

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
  **all** calculates the range from the complete variable, which reads all
  of it: slow, and memory-hungry, on large variables.
- Stepping through time keeps the table's scroll position; loading another
  chunk with the outer scroll bars starts at its top-left corner.

When no Z variable is selected, the text browser shows dataset metadata,
including dimensions, variables, groups, and global attributes. Selecting Z
changes it to the selected variable's dimensions, shape, data type, and
attributes.

## Time controls

Contour, Map and Matrix provide shared time navigation whenever the selected
variable contains the detected time dimension. Both unlimited and fixed-size
time dimensions are supported. The controls are disabled for static variables.
With a [temporal aggregation](#temporal-aggregation) chosen they step through
periods (e.g. months) instead of single time steps.

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

## Temporal aggregation

Every tab has two **aggregate** combos: a resolution (None, Hourly, Daily,
Monthly, Annual, Entire period) and a statistic (Sum, Mean, Mode, Median,
Minimum, Maximum). Missing values are skipped.

- Only resolutions longer than the data's time step can be chosen; daily data
  offers Monthly, Annual and Entire period.
- Mode is offered only for discrete data (unpacked integers such as classes
  or counts). On continuous values almost every value is unique, so a mode
  would just be the minimum.
- Contour, Map and Matrix show one period's statistic per frame; the time
  controls step through periods and the `Time:` label shows the period (e.g.
  `2001-03`).
- Scatter aggregates Y and Y2 when X is the time variable; the time-series
  dialog aggregates its lines.

Frames are computed from the period's time steps as they are read, so memory
stays bounded. On an hourly variable stored one chunk per step, a monthly
frame takes about 10 ms and an annual one about 150 ms. **Entire period**
reads every time step once, which takes as long as reading the whole time
axis. Changing the aggregation of a Scatter plot or of the time-series dialog
does not read the file again.

## Cell time series

With **Cursor** ticked, double-clicking a data cell in the Contour or Map tab
plots that cell's values through time in a separate window. Each further
double-click adds a line, labelled with the cell's coordinates and indices;
clicks beside the grid or on missing cells are ignored. The window's own
**aggregate** combos resample all lines.

The first double-click reads the time series of every cell in the clicked
cell's file chunk (it costs the same as one cell), so further clicks in that
chunk need no read. That memory is released when the window closes.

## Colour scales

Contour and Map offer a colour scale next to the colormap:

| Scale | Colour spacing |
| --- | --- |
| Linear | Even between the limits |
| Log | Logarithmic (positive data; values at or below zero get the lowest colour) |
| Low | More colours for the low end of the range |
| High | More colours for the high end of the range |
| Banded | Ten flat bands, like filled contours |
| Equalised | Each colour covers about the same number of cells |

The colour bar keeps a linear value axis and shows where each colour falls.

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
- Plot images can be saved through the pyqtgraph export menu, but there is no
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

The forms are loaded at runtime with `PyQt6.uic.loadUi`, so there is no
generation step: save the form and restart the application.

Open a form with Qt Designer, for example:

```bash
designer ncv/ui/matrix_panel.ui
```

The executable may be named `designer` or `qt6-designer`, depending on the
platform installation.

pyqtgraph plot widgets, dimension selectors, and the Matrix model are attached
to the loaded forms at runtime.

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
| `ncv/ncvmap.py` | Map tab (pyqtgraph rendering, Cartopy projections) and optional-Cartopy fallback |
| `ncv/ncvmatrix.py` | Matrix table and metadata tab |

## Development and validation

Install the project in editable mode with all optional features and tests:

```bash
python3 -m pip install -e ".[test]"
```

Compile the Python modules and run the offscreen Qt test suite:

```bash
python3 -m py_compile ncv/*.py
QT_QPA_PLATFORM=offscreen python3 -m pytest
```

Run the application normally for a manual check:

```bash
python3 -m ncv sample.nc
```

If the `ncv` command is not found after installation, activate the environment
where it was installed, inspect `command -v ncv`, or use
`python3 -m ncv` from that environment.
