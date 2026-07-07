#!/usr/bin/env python3
"""Command line entry point for ncv."""
from __future__ import annotations

import argparse

import numpy as np

from .app import ncv


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="A minimal GUI for a quick view of netcdf files.",
    )
    parser.add_argument(
        "-m", "--miss", action="store", type=float,
        default=np.nan, dest="miss", metavar="missing_value",
        help="Set value to missing value (default: numpy.nan)",
    )
    parser.add_argument(
        "-x", "--xarray", action="store_true",
        default=False, dest="usex", help="Use xarray to read input files",
    )
    parser.add_argument(
        "ncfile", nargs="*", default=None, metavar="netcdf_file",
        help="netcdf file",
    )
    args = parser.parse_args()
    return ncv(ncfile=args.ncfile, miss=args.miss, usex=args.usex)


if __name__ == "__main__":
    raise SystemExit(main())
