# AGENTS.md

## Project Direction

`ncv` is a PyQt5 NetCDF viewer. Main-window orchestration lives in `ncv/app.py`,
shared panel code in `ncv/ncvcommon.py`, and each tab implementation in its
corresponding `ncv/ncv*.py` module.

Package imports must not depend on Tk, CustomTkinter, or Cartopy.

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
  should consume plain specs and values.

## Useful Commands

```bash
python3 -m py_compile ncv/*.py
QT_QPA_PLATFORM=offscreen python3 -m pytest
QT_QPA_PLATFORM=offscreen python3 -m ncv
```

## Testing Expectations

- Test `import ncv` without Cartopy installed.
- Test that no package module imports Tk or CustomTkinter.
- Test `NcvSession.open()` with generated NetCDF files.
- Test Qt window creation with `QT_QPA_PLATFORM=offscreen`.
- Test that the Map tab is present but disabled/informational when Cartopy is
  unavailable.
