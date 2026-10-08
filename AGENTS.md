# AGENTS.md

## Project Direction

`ncv` is a PyQt6 NetCDF viewer that plots with pyqtgraph. Main-window
orchestration lives in `ncv/app.py`, shared panel code in `ncv/ncvcommon.py`,
and each tab implementation in its corresponding `ncv/ncv*.py` module
(Scatter, Contour, Map, Matrix). It must stay usable on variables far larger
than memory (tens of billions of cells) and on HPC nodes over SSH X11.

Package imports must not depend on Tk, CustomTkinter, or matplotlib. `import
ncv` must not touch Qt or Cartopy.

## Layout

| Module | Role |
|---|---|
| `app.py` | Main window, File menu actions, tabs. Calls `require_qt()` **before** importing panels (they subclass Qt widgets at import time). |
| `qt_compat.py` | PyQt6/pyqtgraph import guard, `require_qt()` with install hints, `gl_usable()`. |
| `ncvcommon.py` | `PlotPanel` base, `ScrollableView`, memory budget, chunk windows, threaded reads, colour helpers. |
| `ncvutils.py`, `dimensions.py`, `session.py`, `ncvmethods.py` | Toolkit-neutral NetCDF / slicing / session logic. |
| `ui/*.ui` | Qt Designer forms, loaded at runtime with `uic.loadUi` (no generated Python). |

## UI Rules

- **Static properties belong in the `.ui`**, so Designer is an accurate
  preview: size policies, stretch, spacing, scroll-bar policies, checkbox
  defaults, widget text, the Map projection list, the status-bar row.
- **Code adds only what Designer cannot express**: pyqtgraph widgets,
  `DimensionControlRow`s, `ScrollableView`, colormap items with generated icons,
  signal connections.
- Widget object names are the contract between forms and code. Rename both
  together. Read the current `.ui` before editing, because the user edits forms
  in Designer.
- Each panel has a static status row `horizontalLayout_status`: `label_cursor`
  on the left; on the right, the Map has the time labels and the Matrix has
  `pushButton_metadata` and the time labels. The cursor read-out runs only while
  `checkBox_cursor` is ticked.
- Open File / Open xarray / New Window are `menuFile` actions in
  `main_window.ui`, not panel buttons.

## Large-Data Model

- **Never read a whole variable.** All reads go through `get_slice_values(...,
  window={axis: (start, stop, step)})` so netCDF4 reads only that hyperslab.
  Missing values, statistics, colour ranges and cursor look-ups must use the
  loaded window or a sample, never the full variable.
- **Chunk size** comes from `memory_budget_cells()`: 20% of available memory at
  32 bytes per cell, where available is `MemAvailable` limited by the tightest
  cgroup v2 `memory.max` (HPC login nodes and jobs). It is capped by
  `CHUNK_MAX_CELLS` (36M) because switch time grows ~9 ns per cell.
- **Load order**: whole variable if it fits; otherwise full-resolution chunks;
  coarse (`overview_stride`, chunk-aligned) only when `checkBox_fullCoarse` is
  ticked. netCDF4 cost is per **chunk decompressed**, not per value returned, so
  sub-chunk strides do not help.
- **Integer chunks stay native** (no float64 copy): missing cells hold the
  dtype minimum. Floats use NaN.
- **Images are uint8 palettes** (`set_palette_image`): values map to colour
  indices 1-255, missing to 0 (transparent), and Qt scales the indexed image
  at paint time (~20 ms per zoom step). Do not use pyqtgraph `autoDownsample`:
  it re-renders and mean-averages the whole chunk at every zoom level (0.2-1.5
  s on 36M cells) and averages class codes into classes that don't exist. The
  `ColorBarItem` is placed with `setImageItem([], insert_in=...)`, not linked:
  a linked bar pushes its levels onto the image.
- **Startup**: xarray is only located (`find_spec`) at import, and imported
  with `--xarray`: it pulls in pandas, ~3 s on a cold file system. Dimension
  combos use `AdjustToMinimumContentsLengthWithIcon`: `AdjustToContents`
  measures every item on each relayout (0.7 s for a 216000-entry dimension).
  Colormap combos fill after the window shows.
- **Blocking reads** run through `read_responsive()` on a worker thread (one at
  a time — netCDF-C is not thread-safe), while the event loop keeps painting.
  Timers that would start another redraw check `reading()` and re-arm.

## Panel Behaviour

- **`ScrollableView`**: outer scroll bars in a grid around the view.
  `windowChanged` fires once per real move and, during a drag, only on release.
- **Contour**: the view drives loading. Panning inside the loaded chunk never
  reads; leaving it reloads a chunk centred on the view. The bars mirror the
  displayed region. `transpose z` means what it says: data is shown as stored,
  except the box auto-ticks to put time on x. With X/Y empty, an axis uses its
  dimension's 1-D CF coordinate variable, so north-first latitude draws
  north-up. A variable with lat and lon dimensions defaults to the lat x lon
  slice: the first two dims (e.g. depth x lat at lon 0) are a thin slice that
  decompresses every chunk along lat (27 s on a 6x84000x216000 tiled variable).
- **Map**: navigation and loading are decoupled. It opens at the world extent,
  and the bars choose the loaded patch, which is read at the screen's
  resolution (re-read only when a settled zoom needs ≥2× finer or coarser data).
  When that stride falls inside one chunk, every chunk is decompressed anyway:
  the patch is read once at full resolution (`_full_patch`) and zooms re-read
  nothing. Contiguous and row-chunked files keep strided reads.
  With no variable selected, no coordinates are read.
  The view re-ranges only on first draw, a projection or central-longitude
  change, `global`, or `bbox` (fits the variable's lon/lat extent).
- **Map rendering**: rectilinear grids (1-D lon/lat, or 2-D lon/lat with
  identical rows and columns) draw as an `ImageItem` on the CPU viewport.
  Curvilinear grids draw as a `PColorMeshItem`, and the Map switches to the
  OpenGL viewport only then: GPU ~30 ms per frame at 4M cells, CPU ~10 s.
  `MESH_MAX_CELLS` is 4M with OpenGL and `MESH_MAX_CELLS_CPU` 250k without.
  Do not put `ImageItem` on a GL viewport: in pyqtgraph 0.14 it stopped
  refining `autoDownsample` on zoom (not re-checked since images became uint8
  palettes). `NCV_OPENGL=0/1` forces the choice. To update mesh
  values, call `setData(None, None, z)` — `setData(z=...)` clears the mesh.
- **Matrix**: the outer bars pick the loaded chunk (thumb = the chunk); the
  table's own bars scroll within it without reading. Headers and
  `showCellIndices` follow the chunk's window. The metadata pane is a collapsible
  splitter (`Metadata: ▶` / `▲`) that remembers its width.

## Development Notes

- Use `python3`, not `python`, in this repository.
- Keep `ncv` as the primary public name. Do not add an `ncvue` compatibility
  command or function; `ncv` is the only public launcher name.
- `pip install ncv` installs everything, including Cartopy and xarray. Code
  must still degrade gracefully if either fails to import: the Map tab shows
  `MapUnavailablePanel`, and the `--xarray` path fails with a clear message.
  The Map uses only `cartopy.crs` and `cartopy.feature`, which don't import
  matplotlib. All rendering is pyqtgraph.
- In conda environments, PyQt6 should come from conda-forge (`pyqt6`;
  conda-forge `pyqt` is PyQt5). The pip wheel's Qt can load the system's older
  FreeType next to conda's HarfBuzz (`undefined symbol: FT_Get_Colorline_Stops`).
- Prefer toolkit-neutral helpers for NetCDF/session/slicing logic. Qt widgets
  should consume plain specs and values.
- Measure before optimising. Profile on a real large file (e.g. GLIM
  `geology_class`, 90001×216001). Offscreen runs cannot create a GL context and
  `QWidget.grab()` does not capture a GL viewport, so check OpenGL behaviour on
  a real display.

## Useful Commands

```bash
python3 -m py_compile ncv/*.py
QT_QPA_PLATFORM=offscreen python3 -m pytest
QT_QPA_PLATFORM=offscreen python3 -m ncv
python3 -c "from ncv.ncvcommon import memory_budget_cells as m, cgroup_available as c; print(m(), c())"
```

## Testing Expectations

- Test `import ncv` without Cartopy installed, and that a broken PyQt6 import
  reports the real reason (not `'NoneType' object has no attribute 'QWidget'`).
- Test that no package module imports Tk, CustomTkinter, or matplotlib.
- Test `NcvSession.open()` with generated NetCDF files.
- Test Qt window creation with `QT_QPA_PLATFORM=offscreen`.
- Test that every `ncv/ui/*.ui` form loads under PyQt6 and has its static
  status row.
- Test that the Map tab is present but disabled/informational when Cartopy is
  unavailable.
- For large-data behaviour, generate small files and shrink the budget by
  patching `ncv.ncvcommon.memory_budget_cells`. Assert on the reads that happen
  (count calls to `redraw` / `_refresh_table`), not just on what is shown.
