# AGENTS.md

## Project Direction

`ncv` is the PyQt5 port of a historical Tkinter-based NetCDF viewer.
New public GUI work should target the Qt implementation in `ncv/app.py`.

The legacy Tk modules are kept as reference material during the port, but new
imports should not make `import ncv` depend on Tk, CustomTkinter, or Cartopy.

## Development Notes

- Use `python3`, not `python`, in this repository.
- Keep `ncv` as the primary public name.
- Do not add an `ncvue` compatibility command/function; `ncv` is the only
  public launcher name.
- Cartopy is optional. If it is missing, the Map tab must degrade gracefully
  instead of breaking package import or app startup.
- xarray is optional. The `--xarray` path should fail with a clear runtime
  message only when requested and unavailable.
- Prefer toolkit-neutral helpers for NetCDF/session/slicing logic. Qt widgets
  should consume plain specs and values instead of imitating Tk control
  variables.

## Useful Commands

```bash
python3 -m py_compile ncv/*.py
QT_QPA_PLATFORM=offscreen python3 -m pytest
QT_QPA_PLATFORM=offscreen python3 -m ncv
```

## Testing Expectations

- Test `import ncv` without Cartopy installed.
- Test `NcvSession.open()` with generated NetCDF files.
- Test Qt window creation with `QT_QPA_PLATFORM=offscreen`.
- Test that the Map tab is present but disabled/informational when Cartopy is
  unavailable.
