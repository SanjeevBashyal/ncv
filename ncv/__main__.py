#!/usr/bin/env python3
"""Command line entry point for ncv."""
from __future__ import annotations

import argparse

import numpy as np

EPILOG = """\
examples:
  ncv file.nc                open one NetCDF file
  ncv a.nc b.nc              open several files; each is a group (file0/, file1/)
  ncv data/*.nc              open every file a shell wildcard matches
  ncv -x a.nc b.nc           read with xarray: the files merge into one dataset
  ncv -m -9999 file.nc       also treat -9999 as missing
  ncv                        start empty, then use File > Open File

remote machines / HPC over X11 forwarding:
  ssh -X user@host           (or ssh -Y), then run ncv as above
  If Qt says 'could not load the Qt platform plugin "xcb"', Qt >= 6.5 needs
  libxcb-cursor. Install it with one of:
    conda install -c conda-forge xcb-util-cursor    (conda environment)
    sudo apt install libxcb-cursor0                 (Debian / Ubuntu)
    sudo dnf install xcb-util-cursor                (RHEL / Fedora)
  QT_DEBUG_PLUGINS=1 ncv file.nc   names any other missing library
  NCV_OPENGL=0 ncv file.nc         if map drawing misbehaves over X11
"""


def main():
    parser = argparse.ArgumentParser(
        prog="ncv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="A minimal GUI for a quick view of NetCDF files.",
        epilog=EPILOG,
    )
    parser.add_argument(
        "-m", "--miss", action="store", type=float,
        default=np.nan, dest="miss", metavar="missing_value",
        help="Additional value to treat as missing (default: NaN)",
    )
    parser.add_argument(
        "-x", "--xarray", action="store_true",
        default=False, dest="usex",
        help="Read with xarray; several files merge into one dataset",
    )
    parser.add_argument(
        "ncfile", nargs="*", default=None, metavar="netcdf_file",
        help="NetCDF file(s) to open",
    )
    args = parser.parse_args()
    # imported here, so --help works even where Qt is missing or broken
    from .app import ncv

    return ncv(ncfile=args.ncfile, miss=args.miss, usex=args.usex)


if __name__ == "__main__":
    raise SystemExit(main())
