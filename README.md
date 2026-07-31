# ncv

A PyQt5 viewer for NetCDF files.

## Editing the interface with Qt Designer

The editable Qt Designer sources are in [`ncv/ui`](ncv/ui). Generated PyQt5
classes live in [`ncv/pyui`](ncv/pyui) and are imported by the application.
Never edit a generated `ui_*.py` module by hand.

Open the forms with Qt Designer (the executable may be named `designer` or
`qt5-designer` on your system):

```bash
designer ncv/ui/main_window.ui
designer ncv/ui/scatter_panel.ui
designer ncv/ui/contour_panel.ui
designer ncv/ui/map_panel.ui
designer ncv/ui/map_unavailable.ui
```

After saving a form, regenerate the Python forms from the repository root:

```bash
./convert-ui-to-py.sh
```

`main_window.ui` controls the window and tab shell. The panel forms contain the
visible plot controls. Matplotlib canvases, toolbars, NetCDF dimension
selectors, and the Cartopy axes are inserted at runtime into the named host
layouts in those forms.

Widget object names are the interface between Designer and the panel modules:
`ncvscatter.py`, `ncvcontour.py`, `ncvmap.py`, and `ncvmatrix.py`. Shared panel
code lives in `ncvcommon.py`, while `app.py` assembles the main window. Keep an
existing object name when moving or styling a widget. If you intentionally
rename or remove one, update the corresponding Python reference and tests.

Validate UI changes from the repository root with:

```bash
python3 -m py_compile ncv/*.py ncv/pyui/*.py
QT_QPA_PLATFORM=offscreen python3 -m pytest
QT_QPA_PLATFORM=offscreen python3 -m ncv
```
