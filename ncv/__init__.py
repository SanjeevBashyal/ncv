"""ncv: a quick PyQt5 NetCDF viewer."""
from __future__ import annotations

from .ncvutils import DIMMETHODS
from .ncvutils import add_cyclic, has_cyclic
from .ncvutils import format_coord_contour, format_coord_map
from .ncvutils import format_coord_scatter, get_slice, get_slice_values
from .ncvutils import get_standard_name, get_units
from .ncvutils import list_intersection, parse_entry, selvar, set_axis_label
from .ncvutils import set_miss, spinbox_values, vardim2var
from .ncvutils import xzip_dim_name_length, zip_dim_name_length
from .session import HAVE_XARRAY, NcvSession

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__author__ = "Matthias Cuntz"


def ncv(ncfile=None, miss=None, usex=False):
    """Launch the Qt ncv application."""
    from .app import ncv as _ncv

    return _ncv(ncfile=ncfile, miss=miss, usex=usex)


__all__ = [
    "DIMMETHODS",
    "HAVE_XARRAY",
    "NcvSession",
    "add_cyclic",
    "format_coord_contour",
    "format_coord_map",
    "format_coord_scatter",
    "get_slice",
    "get_slice_values",
    "get_standard_name",
    "get_units",
    "has_cyclic",
    "list_intersection",
    "ncv",
    "parse_entry",
    "selvar",
    "set_axis_label",
    "set_miss",
    "spinbox_values",
    "vardim2var",
    "xzip_dim_name_length",
    "zip_dim_name_length",
]
