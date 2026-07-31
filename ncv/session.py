"""
Application session and NetCDF loading for ncv.

This module is intentionally GUI-toolkit neutral and keeps application state
in an explicit object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import netCDF4 as nc
import numpy as np

try:
    import xarray as xr
    HAVE_XARRAY = True
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    xr = None
    HAVE_XARRAY = False

from .ncvmethods import analyse_netcdf


@dataclass
class NcvSession:
    """Mutable state for one ncv application session."""

    miss: float = np.nan
    usex: bool = False
    fi: object = field(default_factory=list)
    groups: list = field(default_factory=list)
    dunlim: object = field(default_factory=list)
    time: object = field(default_factory=list)
    tname: object = field(default_factory=list)
    tvar: object = field(default_factory=list)
    dtime: object = field(default_factory=list)
    latvar: object = field(default_factory=list)
    lonvar: object = field(default_factory=list)
    latdim: object = field(default_factory=list)
    londim: object = field(default_factory=list)
    maxdim: int = 1
    cols: list = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    generation: int = 0

    def reset_metadata(self) -> None:
        self.groups = []
        self.dunlim = []
        self.time = []
        self.tname = []
        self.tvar = []
        self.dtime = []
        self.latvar = []
        self.lonvar = []
        self.latdim = []
        self.londim = []
        self.maxdim = 1
        self.cols = []

    def close(self) -> None:
        """Close currently opened files/datasets."""
        if self.usex:
            if self.fi is not None and not isinstance(self.fi, list):
                close = getattr(self.fi, "close", None)
                if close is not None:
                    close()
        else:
            for handle in self.fi or []:
                close = getattr(handle, "close", None)
                if close is not None:
                    close()
        self.fi = []
        self.files = []

    def open(self, paths: Sequence[str] | str | None,
             use_xarray: bool = False) -> None:
        """Open NetCDF files and analyse their variables."""
        if paths is None:
            paths = []
        if isinstance(paths, (str, Path)):
            paths = [str(paths)]
        paths = [str(path) for path in paths if str(path)]

        self.close()
        self.reset_metadata()
        self.files = list(paths)
        self.usex = bool(use_xarray and HAVE_XARRAY)

        if not paths:
            self.generation += 1
            return

        if use_xarray and not HAVE_XARRAY:
            raise RuntimeError("xarray support was requested but xarray is not installed.")

        if self.usex:
            if len(paths) > 1:
                self.fi = xr.open_mfdataset(paths)
            else:
                self.fi = xr.open_dataset(paths[0])
        else:
            self.fi = []
            for ii, path in enumerate(paths):
                self.fi.append(nc.Dataset(path, "r"))
                if len(paths) > 1:
                    width = int(np.ceil(np.log10(len(paths))))
                    self.groups.append(f"file{ii:0{width}d}")
            if len(paths) == 1:
                self.groups = list(self.fi[0].groups.keys())
            else:
                for ii, path in enumerate(paths):
                    if list(self.fi[ii].groups.keys()):
                        self.close()
                        raise ValueError(
                            "Either multiple files or one file with groups are "
                            f"allowed as input. Multiple files were given but "
                            f"{path} has groups."
                        )

        analyse_netcdf(self)
        self.generation += 1

    @property
    def has_data(self) -> bool:
        if self.usex:
            return self.fi is not None and not isinstance(self.fi, list)
        return bool(self.fi)

    @property
    def title_suffix(self) -> str:
        if len(self.files) == 1:
            return self.files[0]
        if len(self.files) > 1:
            return f"{len(self.files)} files"
        return ""


def normalize_files(ncfile: Iterable[str] | str | None) -> list[str]:
    """Normalize user-facing file arguments to a list of strings."""
    if ncfile is None:
        return []
    if isinstance(ncfile, (str, Path)):
        return [str(ncfile)]
    return [str(path) for path in ncfile]
