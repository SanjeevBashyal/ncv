# Graph Report - ncv  (2026-09-15)

## Corpus Check
- Corpus is ~21,177 words - fits in a single context window. You may not need a graph.

## Summary
- 394 nodes · 932 edges · 19 communities (17 shown, 2 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.9)
- Token cost: 103,037 input · 0 output

## Community Hubs (Navigation)
- Main Window and Session Lifecycle
- Project Documentation and Policy
- Dimension Selection and Matrix Tab
- Map Panel Rendering
- Scatter Panel and Styling
- Toolkit-Neutral Utilities
- Module Surface and Shared Helpers
- Contour Panel
- Time Navigation and Animation
- Cartopy Guard and Dimension Widgets
- NetCDF Analysis Backend
- Plot Panel Base and Colormaps
- Array Table Model
- Launcher and Entry Points
- Variable Selection and Axis Labels
- Application Icon Branding
- Missing Value Handling
- pyqtgraph Data Conversion
- ncv Package Root

## God Nodes (most connected - your core abstractions)
1. `MapPanel` - 40 edges
2. `ScatterPanel` - 35 edges
3. `MatrixPanel` - 33 edges
4. `ContourPanel` - 27 edges
5. `dimension_specs()` - 25 edges
6. `TimeControlMixin` - 24 edges
7. `PlotPanel` - 21 edges
8. `vardim2var()` - 21 edges
9. `NcvSession` - 21 edges
10. `NcvMainWindow` - 20 edges

## Surprising Connections (you probably didn't know these)
- `test_qt_window_smoke_with_generated_netcdf()` --uses--> `ContourPanel`  [INFERRED]
  tests/test_session_qt.py → ncv/ncvcontour.py
- `test_qt_window_smoke_with_generated_netcdf()` --uses--> `MapPanel`  [INFERRED]
  tests/test_session_qt.py → ncv/ncvmap.py
- `test_qt_window_smoke_with_generated_netcdf()` --uses--> `MatrixPanel`  [INFERRED]
  tests/test_session_qt.py → ncv/ncvmatrix.py
- `test_qt_window_smoke_with_generated_netcdf()` --uses--> `ScatterPanel`  [INFERRED]
  tests/test_session_qt.py → ncv/ncvscatter.py
- `test_get_slice_values_reducers()` --calls--> `get_slice_values()`  [EXTRACTED]
  tests/test_session_qt.py → ncv/ncvutils.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Four Coordinated Views Sharing One Selection Model** — readme_scatter_line_tab, readme_contour_tab, readme_map_tab, readme_matrix_tab, readme_dimension_selection, readme_session_sharing [EXTRACTED 1.00]
- **Optional-Dependency Graceful Degradation Pattern** — agents_optional_cartopy_policy, agents_optional_xarray_policy, agents_dependency_exclusion_policy, readme_optional_extras, readme_map_tab, readme_xarray_backend [EXTRACTED 1.00]
- **Runtime-Loaded UI Contract** — agents_runtime_ui_loading, readme_qt_designer_forms, readme_widget_object_name_contract, agents_testing_expectations, agents_offscreen_qt_testing [INFERRED 0.85]
- **Map Rendering Elements Composing the ncvue Icon** — ncv_images_ncvue_icon_global_map_preview, ncv_images_ncvue_icon_colorbar_legend, ncv_images_ncvue_icon_rdylbu_colormap, ncv_images_ncvue_icon_netcdf_map_visualization [INFERRED 0.85]

## Communities (19 total, 2 thin omitted)

### Community 0 - "Main Window and Session Lifecycle"
Cohesion: 0.07
Nodes (28): fixture, NcvMainWindow, Top-level window that owns the four independent Qt panels., _window_geometry(), MapUnavailablePanel, NcvSession, Mutable state for one ncv application session., Close currently opened files/datasets. (+20 more)

### Community 1 - "Project Documentation and Policy"
Cohesion: 0.09
Nodes (42): Forbidden GUI/Plotting Dependencies, ncv Architecture Layout, NcvSession.open(), Offscreen Qt Testing Convention, Optional Cartopy Graceful Degradation, Optional xarray Backend Policy, Runtime .ui Loading with uic.loadUi, Single Public Launcher Name (+34 more)

### Community 2 - "Dimension Selection and Matrix Tab"
Cohesion: 0.12
Nodes (16): _as_text_values(), _dim_names(), _dim_tooltip(), dimension_specs(), DimensionSpec, empty_dimension_specs(), Toolkit-neutral dimension selector helpers., State needed by one dimension selector control. (+8 more)

### Community 3 - "Map Panel Rendering"
Cohesion: 0.10
Nodes (13): _corner_grid(), decimation_stride(), MapPanel, _nan_joined(), projected_rings(), Natural Earth rings for ``name``, projected into ``proj``. ``project_geometry``…, Cell corners for 2-D curvilinear coordinate arrays., Concatenate rings into one polyline separated by NaN. (+5 more)

### Community 4 - "Scatter Panel and Styling"
Cohesion: 0.11
Nodes (8): cursor_label(), Add a read-out label under ``plot_widget`` fed by mouse position., _float_or(), _maybe_color(), Pen and symbol options from the Y1/Y2 style entries., Apply one axis limit entry; auto-range when it is None., Scatter and line plot tab., ScatterPanel

### Community 5 - "Toolkit-Neutral Utilities"
Cohesion: 0.12
Nodes (24): ncv: a quick PyQt6 NetCDF viewer., add_cyclic(), _add_cyclic_data(), _add_cyclic_x(), datetime_str(), format_coord_contour(), format_coord_map(), get_slice() (+16 more)

### Community 6 - "Module Surface and Shared Helpers"
Cohesion: 0.20
Nodes (16): PyQt6 application and main-window orchestration for ncv., float_or_none(), load_ui(), parse_limits(), Shared Qt helpers for ncv panels., Load ``ncv/ui/<name>.ui`` onto an existing widget., Return a ``(minimum, maximum)`` pair from a limit entry., resource_path() (+8 more)

### Community 7 - "Contour Panel"
Cohesion: 0.18
Nodes (3): ContourPanel, Return (values, label, is_datetime) for one combo selection., Contour plot tab, drawn as a heat map.

### Community 9 - "Cartopy Guard and Dimension Widgets"
Cohesion: 0.14
Nodes (9): ensure_cartopy(), Retry Cartopy import and keep compatibility exports current., dimension_values(), Return current selector text from Qt combo boxes or compatible objects., DimensionControlRow, A row of label and combo-box dimension selectors., ensure_cartopy(), _import_cartopy() (+1 more)

### Community 10 - "NetCDF Analysis Backend"
Cohesion: 0.18
Nodes (15): analyse_netcdf(), analyse_netcdf_ncvue(), analyse_netcdf_xarray(), NetCDF analysis and missing-value helpers for ncv. This module was written by…, Call analyse_netcdf_xarray or analyse_netcdf_ncvue depending on self.usex…, Analyse netcdf file(s) opened with xarray for time (= unlimited), variables,…, Analyse netcdf file(s) for the unlimited dimension, calculating datetime,…, get_standard_name() (+7 more)

### Community 11 - "Plot Panel Base and Colormaps"
Cohesion: 0.19
Nodes (4): _cmap_icon(), PlotPanel, Common behavior for Qt plotting panels., Build a gradient swatch icon from a pyqtgraph colormap.

### Community 12 - "Array Table Model"
Cohesion: 0.27
Nodes (3): _ArrayTableModel, _format_value(), Lazy table view over a two-dimensional NumPy array.

### Community 13 - "Launcher and Entry Points"
Cohesion: 0.20
Nodes (9): ncv(), Launch the Qt ncv application., ncv(), Launch the Qt ncv application., main(), Command line entry point for ncv., normalize_files(), Application session and NetCDF loading for ncv. This module is intentionally… (+1 more)

### Community 14 - "Variable Selection and Axis Labels"
Cohesion: 0.33
Nodes (5): Return (values, label, is_datetime) for one combo selection., Extract variable from correct file. Parameters ---------- self : class ncv…, Set label plotting axis from name and unit of given variable `ncvar`.…, selvar(), set_axis_label()

### Community 15 - "Application Icon Branding"
Cohesion: 0.47
Nodes (6): ncvue Application Icon, ncvue Branding: Product Screenshot Over Wordmark, Colorbar Legend with Units (umol/m2/s), Global Geographic Map Preview (Gross Primary Production), NetCDF Geo-referenced Variable Visualization, RdYlBu-style Diverging Colormap Rendering

### Community 16 - "Missing Value Handling"
Cohesion: 0.40
Nodes (4): get_miss(), Get list of missing values, i.e. self.miss, x._FillValue, x.missing_value, and…, Set `x` to NaN or NaT for all values in miss. Parameters ---------- miss :…, set_miss()

### Community 17 - "pyqtgraph Data Conversion"
Cohesion: 0.40
Nodes (5): Return ``(numeric array, is_datetime)`` that pyqtgraph can plot. ``datetime64``…, to_plot_values(), cell_edges(), Return ``n + 1`` cell edges for ``n`` cell centers., test_to_plot_values_and_cell_edges()

## Ambiguous Edges - Review These
- `Matrix and Metadata Tab` → `Group and Multi-File Naming Conflict Rejection`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to

## Knowledge Gaps
- **3 isolated node(s):** `ncv`, `Offscreen Qt Testing Convention`, `CF-Style Time Coordinate Decoding`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 106 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Matrix and Metadata Tab` and `Group and Multi-File Naming Conflict Rejection`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `MapPanel` connect `Map Panel Rendering` to `Main Window and Session Lifecycle`, `Dimension Selection and Matrix Tab`, `Toolkit-Neutral Utilities`, `Module Surface and Shared Helpers`, `Time Navigation and Animation`, `Cartopy Guard and Dimension Widgets`, `Plot Panel Base and Colormaps`, `Variable Selection and Axis Labels`, `Missing Value Handling`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `ScatterPanel` connect `Scatter Panel and Styling` to `Main Window and Session Lifecycle`, `Module Surface and Shared Helpers`, `Cartopy Guard and Dimension Widgets`, `Plot Panel Base and Colormaps`, `Variable Selection and Axis Labels`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `MatrixPanel` connect `Dimension Selection and Matrix Tab` to `Main Window and Session Lifecycle`, `Module Surface and Shared Helpers`, `Time Navigation and Animation`, `Cartopy Guard and Dimension Widgets`, `Plot Panel Base and Colormaps`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MapPanel` (e.g. with `NcvMainWindow` and `DimensionControlRow`) actually correct?**
  _`MapPanel` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ScatterPanel` (e.g. with `NcvMainWindow` and `DimensionControlRow`) actually correct?**
  _`ScatterPanel` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `MatrixPanel` (e.g. with `NcvMainWindow` and `DimensionControlRow`) actually correct?**
  _`MatrixPanel` has 3 INFERRED edges - model-reasoned connections that need verification._