# ncv

A PyQt5 viewer for NetCDF files.

## Editing the interface with Qt Designer

The editable Qt Designer sources are in [`ncv/ui`](ncv/ui). The application
loads these files directly at runtime, so saving a form in Designer is enough;
do not generate or edit a `pyuic5` Python module.

Open the forms with Qt Designer (the executable may be named `designer` or
`qt5-designer` on your system):

```bash
designer ncv/ui/main_window.ui
designer ncv/ui/scatter_panel.ui
designer ncv/ui/contour_panel.ui
designer ncv/ui/map_panel.ui
designer ncv/ui/map_unavailable.ui
```

`main_window.ui` controls the window and tab shell. The panel forms contain the
visible controls for each tab. Matplotlib canvases, toolbars, NetCDF dimension
selectors, and the Cartopy axes are inserted at runtime into the named host
layouts in those forms.

Widget object names are the interface between Designer and `ncv/app.py`. Keep
an existing object name when moving or styling a widget. If you intentionally
rename or remove one, update the corresponding Python reference and tests.

Validate UI changes from the repository root with:

```bash
python3 -m py_compile ncv/*.py
QT_QPA_PLATFORM=offscreen python3 -m pytest
QT_QPA_PLATFORM=offscreen python3 -m ncv
```
