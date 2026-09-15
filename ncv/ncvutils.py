#!/usr/bin/env python3
"""
Toolkit-neutral utility functions for ncv.

The utility functions do not depend on the GUI classes.
Functions depending on the class are in ncvmethods.

This module was written by Matthias Cuntz while at Institut National de
Recherche pour l'Agriculture, l'Alimentation et l'Environnement (INRAE), Nancy,
France.

:copyright: Copyright 2020- Matthias Cuntz - mc (at) macu (dot) de
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   DIMMETHODS
   add_cyclic
   format_coord_contour
   format_coord_map
   get_slice
   get_standard_name
   get_units
   list_intersection
   parse_entry
   selvar
   set_axis_label
   set_miss
   spinbox_values
   vardim2var
   xzip_dim_name_length
   zip_dim_name_length

History
   * Written Nov-Dec 2020 by Matthias Cuntz (mc (at) macu (dot) de)
   * General get_slice function from individual methods for x, y, y2, z,
     Dec 2020, Matthias Cuntz
   * Added arithmetics to apply on axis/dimensions such as mean, std, etc.,
     Dec 2020, Matthias Cuntz
   * Added SEPCHAR and DIMMETHODS, Jan 2021, Matthias Cuntz
   * Set correct missing value for date variable in numpy's datetime64[ms]
     format May 2021, Matthias Cuntz
   * Added format_coord functions for scatter, contour, and map,
     May 2021, Matthias Cuntz
   * Replaced add_cyclic_point with add_cyclic as submitted to cartopy,
     Jun 2021, Matthias Cuntz
   * Removed SEPCHAR, Jun 2021, Matthias Cuntz
   * Final add_cyclic, has_cyclic from cartopy v0.20.1,
     Nov 2021, Matthias Cuntz
   * Address fi.variables[name] directly by fi[name], Jan 2024, Matthias Cuntz
   * Add selvar to allow multiple netcdf files, Jan 2024, Matthias Cuntz
   * Remove [ms] from check for datetime in format_coord on axes,
     Oct 2024, Matthias Cuntz
   * Increased digits in format_coord_scatter, Jan 2025, Matthias Cuntz
   * Increased digits in format_coord_contour and format_coord_map,
     Jan 2025, Matthias Cuntz
   * Use f-string in zip_dim_name_length, Feb 2025, Matthias Cuntz
   * Added xzip_dim_name_length, get_standard_name, get_units,
     Feb 2025, Matthias Cuntz
   * Allow xarray in selvar, Feb 2025, Matthias Cuntz
   * Add parse_entry from dfvutils, Jun 2025, Matthias Cuntz
   * Copy parse_entry from dfvutils again, deducing datetime string,
     Nov 2025, Matthias Cuntz

"""
from math import isfinite
import numpy as np
try:
    import cartopy.crs as ccrs
except ModuleNotFoundError:
    ccrs = None
try:
    import xarray as xr
    ihavex = True
except ModuleNotFoundError:
    ihavex = False


__all__ = ['DIMMETHODS',
           'add_cyclic', 'has_cyclic',
           'cell_edges', 'datetime_str',
           'format_coord_contour', 'format_coord_map',
           'get_slice', 'get_slice_values', 'get_standard_name', 'get_units',
           'list_intersection', 'parse_entry',
           'selvar', 'set_axis_label', 'set_miss',
           'spinbox_values', 'vardim2var',
           'xzip_dim_name_length', 'zip_dim_name_length']


DIMMETHODS = ('mean', 'std', 'min', 'max', 'ptp', 'sum', 'median', 'var')
"""
Arithmetic methods implemented on dimensions.

mean - average
std - standard deviation
min - minimum
max - maximum
ptp - point-to-point amplitude = max - min
sum - sum
median - 50-percentile
var - variance

"""


def _add_cyclic_data(data, axis=-1):
    """
    Add a cyclic point to a data array.

    Parameters
    ----------
    data : ndarray
        An n-dimensional array of data to add a cyclic point to.
    axis : int, optional
        Specifies the axis of the data array to add the cyclic point to.
        Defaults to the right-most axis.

    Returns
    -------
    The data array with a cyclic point added.

    """
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError(
            'The specified axis does not correspond to an array dimension.')
    npc = np.ma if np.ma.is_masked(data) else np
    return npc.concatenate((data, data[tuple(slicer)]), axis=axis)


def _add_cyclic_x(x, axis=-1, cyclic=360):
    """
    Add a cyclic point to a x/longitude coordinate array.

    Parameters
    ----------
    x : ndarray
        An array which specifies the x-coordinate values for
        the dimension the cyclic point is to be added to.
    axis : int, optional
        Specifies the axis of the x-coordinate array to add the cyclic point
        to. Defaults to the right-most axis.
    cyclic : float, optional
        Width of periodic domain (default: 360)

    Returns
    -------
    The coordinate array ``x`` with a cyclic point added.

    """
    npc = np.ma if np.ma.is_masked(x) else np
    # get cyclic x-coordinates
    # cx is the code from basemap (addcyclic)
    # https://github.com/matplotlib/basemap/blob/master/lib/mpl_toolkits/basemap/__init__.py
    cx = (np.take(x, [0], axis=axis) +
          cyclic * np.sign(np.diff(np.take(x, [0, -1], axis=axis),
                                   axis=axis)))
    # basemap ensures that the values do not exceed cyclic
    # (next code line). We do not do this to deal with rotated grids that
    # might have values not exactly 0.
    #     cx = npc.where(cx <= cyclic, cx, np.mod(cx, cyclic))
    return npc.concatenate((x, cx), axis=axis)


def has_cyclic(x, axis=-1, cyclic=360, precision=1e-4):
    """
    Check if x/longitude coordinates already have a cyclic point.

    Checks all differences between the first and last
    x-coordinates along ``axis`` to be less than ``precision``.

    Parameters
    ----------
    x : ndarray
        An array with the x-coordinate values to be checked for cyclic points.
    axis : int, optional
        Specifies the axis of the ``x`` array to be checked.
        Defaults to the right-most axis.
    cyclic : float, optional
        Width of periodic domain (default: 360).
    precision : float, optional
        Maximal difference between first and last x-coordinate to detect
        cyclic point (default: 1e-4).

    Returns
    -------
    True if a cyclic point was detected along the given axis,
    False otherwise.

    Examples
    --------
    Check for cyclic x-coordinate in one dimension.
    >>> import numpy as np
    >>> lons = np.arange(0, 360, 60)
    >>> clons = np.arange(0, 361, 60)
    >>> print(has_cyclic(lons))
    False
    >>> print(has_cyclic(clons))
    True

    Check for cyclic x-coordinate in two dimensions.
    >>> lats = np.arange(-90, 90, 30)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> clon2d, clat2d = np.meshgrid(clons, lats)
    >>> print(has_cyclic(lon2d))
    False
    >>> print(has_cyclic(clon2d))
    True

    """
    npc = np.ma if np.ma.is_masked(x) else np
    # transform to 0-cyclic, assuming e.g. -180 to 180 if any < 0
    x1 = np.mod(npc.where(x < 0, x + cyclic, x), cyclic)
    dd = np.diff(np.take(x1, [0, -1], axis=axis), axis=axis)
    if npc.all(np.abs(dd) < precision):
        return True
    else:
        return False


def add_cyclic(data, x=None, y=None, axis=-1,
               cyclic=360, precision=1e-4):
    """
    Add a cyclic point to an array and optionally corresponding
    x/longitude and y/latitude coordinates.

    Checks all differences between the first and last
    x-coordinates along ``axis`` to be less than ``precision``.

    Parameters
    ----------
    data : ndarray
        An n-dimensional array of data to add a cyclic point to.
    x : ndarray, optional
        An n-dimensional array which specifies the x-coordinate values
        for the dimension the cyclic point is to be added to, i.e. normally the
        longitudes. Defaults to None.

        If ``x`` is given then *add_cyclic* checks if a cyclic point is
        already present by checking all differences between the first and last
        coordinates to be less than ``precision``.
        No point is added if a cyclic point was detected.

        If ``x`` is 1- or 2-dimensional, ``x.shape[-1]`` must equal
        ``data.shape[axis]``, otherwise ``x.shape[axis]`` must equal
        ``data.shape[axis]``.
    y : ndarray, optional
        An n-dimensional array with the values of the y-coordinate, i.e.
        normally the latitudes.
        The cyclic point simply copies the last column. Defaults to None.

        No cyclic point is added if ``y`` is 1-dimensional.
        If ``y`` is 2-dimensional, ``y.shape[-1]`` must equal
        ``data.shape[axis]``, otherwise ``y.shape[axis]`` must equal
        ``data.shape[axis]``.
    axis : int, optional
        Specifies the axis of the arrays to add the cyclic point to,
        i.e. axis with changing x-coordinates. Defaults to the right-most axis.
    cyclic : int or float, optional
        Width of periodic domain (default: 360).
    precision : float, optional
        Maximal difference between first and last x-coordinate to detect
        cyclic point (default: 1e-4).

    Returns
    -------
    cyclic_data
        The data array with a cyclic point added.
    cyclic_x
        The x-coordinate with a cyclic point, only returned if the ``x``
        keyword was supplied.
    cyclic_y
        The y-coordinate with the last column of the cyclic axis duplicated,
        only returned if ``x`` was 2- or n-dimensional and the ``y``
        keyword was supplied.

    Examples
    --------
    Adding a cyclic point to a data array, where the cyclic dimension is
    the right-most dimension.

    >>> import numpy as np
    >>> data = np.ones([5, 6]) * np.arange(6)
    >>> cyclic_data = add_cyclic(data)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]]

    Adding a cyclic point to a data array and an associated x-coordinate.

    >>> lons = np.arange(0, 360, 60)
    >>> cyclic_data, cyclic_lons = add_cyclic(data, x=lons)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lons)
    [  0  60 120 180 240 300 360]

    Adding a cyclic point to a data array and an associated 2-dimensional
    x-coordinate.

    >>> lons = np.arange(0, 360, 60)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d = add_cyclic(data, x=lon2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lon2d)
    [[  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]]

    Adding a cyclic point to a data array and the associated 2-dimensional
    x- and y-coordinates.

    >>> lons = np.arange(0, 360, 60)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(
    ...     data, x=lon2d, y=lat2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lon2d)
    [[  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]]
    >>> print(cyclic_lat2d)
    [[-90. -90. -90. -90. -90. -90. -90.]
     [-54. -54. -54. -54. -54. -54. -54.]
     [-18. -18. -18. -18. -18. -18. -18.]
     [ 18.  18.  18.  18.  18.  18.  18.]
     [ 54.  54.  54.  54.  54.  54.  54.]]

    Not adding a cyclic point if cyclic point detected in x.

    >>> lons = np.arange(0, 361, 72)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(
    ...     data, x=lon2d, y=lat2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 1. 2. 3. 4. 5.]
     [0. 1. 2. 3. 4. 5.]
     [0. 1. 2. 3. 4. 5.]
     [0. 1. 2. 3. 4. 5.]
     [0. 1. 2. 3. 4. 5.]]
    >>> print(cyclic_lon2d)
    [[  0  72 144 216 288 360]
     [  0  72 144 216 288 360]
     [  0  72 144 216 288 360]
     [  0  72 144 216 288 360]
     [  0  72 144 216 288 360]]
    >>> print(cyclic_lat2d)
    [[-90. -90. -90. -90. -90. -90.]
     [-54. -54. -54. -54. -54. -54.]
     [-18. -18. -18. -18. -18. -18.]
     [ 18.  18.  18.  18.  18.  18.]
     [ 54.  54.  54.  54.  54.  54.]]

    """
    if x is None:
        return _add_cyclic_data(data, axis=axis)
    # if x was given
    if x.ndim > 2:
        xaxis = axis
    else:
        xaxis = -1
    if x.shape[xaxis] != data.shape[axis]:
        estr = (f'x.shape[{xaxis}] does not match the size of the'
                f' corresponding dimension of the data array:'
                f' x.shape[{xaxis}] = {x.shape[xaxis]},'
                f' data.shape[{axis}] = {data.shape[axis]}.')
        raise ValueError(estr)
    if has_cyclic(x, axis=xaxis, cyclic=cyclic, precision=precision):
        if y is None:
            return data, x
        # if y was given
        return data, x, y
    # if not has_cyclic, add cyclic points to data and x
    out_data = _add_cyclic_data(data, axis=axis)
    out_x = _add_cyclic_x(x, axis=xaxis, cyclic=cyclic)
    if y is None:
        return out_data, out_x
    # if y was given
    if y.ndim == 1:
        return out_data, out_x, y
    if y.ndim > 2:
        yaxis = axis
    else:
        yaxis = -1
    if y.shape[yaxis] != data.shape[axis]:
        estr = (f'y.shape[{yaxis}] does not match the size of the'
                f' corresponding dimension of the data array:'
                f' y.shape[{yaxis}] = {y.shape[yaxis]},'
                f' data.shape[{axis}] = {data.shape[axis]}.')
        raise ValueError(estr)
    out_y = _add_cyclic_data(y, axis=yaxis)
    return out_data, out_x, out_y


def datetime_str(value):
    """Format POSIX seconds as a date string."""
    return str(np.datetime64(int(value), 's')).replace('T', ' ')


def cell_edges(centers):
    """Return ``n + 1`` cell edges for ``n`` cell centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.array([0.0, 1.0])
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    middle = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate((
        [centers[0] - (middle[0] - centers[0])],
        middle,
        [centers[-1] + (centers[-1] - middle[-1])]))


def format_coord_contour(x, y, xx, yy, zz, xdate=False, ydate=False):
    """
    Cursor read-out for a contour/image plot, including the nearest cell value.

    Parameters
    ----------
    x, y : float
        Data coordinates of the cursor.
    xx, yy, zz : ndarray
        Numeric x-values, y-values and z-values.
    xdate, ydate : bool
        Whether `x`/`y` are POSIX seconds and should be shown as dates.

    Returns
    -------
    String with coordinates.

    """
    xarr = xx[0, :] if xx.ndim > 1 else xx
    yarr = yy[:, 0] if yy.ndim > 1 else yy
    if ((x > xarr.min()) & (x <= xarr.max()) &
            (y > yarr.min()) & (y <= yarr.max())):
        col = np.searchsorted(xarr, x) - 1
        row = np.searchsorted(yarr, y) - 1
        xout, yout, zout = xarr[col], yarr[row], zz[row, col]
    else:
        xout, yout, zout = x, y, np.nan

    xstr = datetime_str(xout) if xdate else f'{xout:.6g}'
    ystr = datetime_str(yout) if ydate else f'{yout:.6g}'
    return f'x={xstr}, y={ystr}, z={zout:.6g}'


def format_coord_map(x, y, proj, xx, yy, zz):
    """
    Cursor read-out for a map, including the nearest cell value.

    Parameters
    ----------
    x, y : float
        Projected coordinates of the cursor.
    proj : cartopy.crs.CRS
        Projection that `x`, `y` are given in.
    xx, yy, zz : ndarray
        Numpy arrays with x-values, y-values, and z-values.

    Returns
    -------
    String with coordinates.

    """
    if ccrs is None:
        return f'x={x:.6g}, y={y:.6g}'

    lon, lat = ccrs.PlateCarree().transform_point(x, y, proj)
    if not (isfinite(lon) and isfinite(lat)):
        return ''
    ns = 'N' if lat >= 0. else 'S'
    ew = 'E' if lon >= 0. else 'W'
    out = (f'{abs(lon):.4f} \u00b0{ew}, {abs(lat):.4f} \u00b0{ns}')
    if zz is None or zz.size == 0:
        return out

    # nearest grid cell
    gx, gy = (np.meshgrid(xx, yy) if (np.ndim(xx) == 1 and np.ndim(yy) == 1)
              else (xx, yy))
    if np.shape(gx) != np.shape(zz):
        return out
    idx = np.abs((((gx + 360.) % 360.) - ((lon + 360.) % 360.))**2 +
                 (gy - lat)**2).argmin()
    return f'{out}, z={np.asarray(zz).flat[idx]:.6g}'


def get_slice_values(dim_values, y):
    """
    Get slice of variable `y` from dimension selector values.

    Parameters
    ----------
    dim_values : list
        List of dimension selector values. Each value is either ``all``,
        one of ``DIMMETHODS``, or a numeric dimension index.
    y : ndarray or netCDF4._netCDF4.Variable
        Input array or netcdf variable

    Returns
    -------
    ndarray
        Slice of `y` chosen with spinboxes.

    Examples
    --------
    >>> gy, vy = vardim2var(y, self.groups)
    >>> yy = selvar(self, vy)
    >>> miss = get_miss(self, yy)
    >>> yy = get_slice_values(['all', '0'], yy).squeeze()
    >>> yy = set_miss(miss, yy)

    """
    isxarray = False
    if ihavex:
        if isinstance(y, xr.DataArray):
            isxarray = True
    methods = ['all']
    methods.extend(DIMMETHODS)
    dd = []
    ss = []
    for i in range(y.ndim):
        dim = str(dim_values[i])
        if dim in methods:
            s = slice(0, y.shape[i])
        else:
            idim = int(dim)
            s = slice(idim, idim + 1)
        dd.append(dim)
        ss.append(s)
    if len(ss) > 0:
        imeth = list_intersection(dd, DIMMETHODS)
        if len(imeth) > 0:
            if isxarray:
                sargs = dict(zip(y.dims, ss))
                yout = y.isel(**sargs).to_numpy()
            else:
                yout = y[tuple(ss)]
            ii = [ i for i, d in enumerate(dd) if d in imeth ]
            ii.reverse()  # last axis first
            for i in ii:
                if dd[i] == 'mean':
                    yout = np.ma.mean(yout, axis=i)
                elif dd[i] == 'std':
                    yout = np.ma.std(yout, axis=i)
                elif dd[i] == 'min':
                    yout = np.ma.min(yout, axis=i)
                elif dd[i] == 'max':
                    yout = np.ma.max(yout, axis=i)
                elif dd[i] == 'ptp':
                    yout = np.ma.ptp(yout, axis=i)
                elif dd[i] == 'sum':
                    yout = np.ma.sum(yout, axis=i)
                elif dd[i] == 'median':
                    yout = np.ma.median(yout, axis=i)
                elif dd[i] == 'var':
                    yout = np.ma.var(yout, axis=i)
            return yout
        else:
            if isxarray:
                sargs = dict(zip(y.dims, ss))
                yout = y.isel(**sargs).to_numpy()
            else:
                yout = y[tuple(ss)]
            return yout
    else:
        return np.array([], dtype=y.dtype)


def get_slice(dimspins, y):
    """
    Get slice of variable `y` inquiring dimension selector widgets.

    This compatibility wrapper accepts objects exposing ``get()``. New code
    should pass plain values to ``get_slice_values``.

    """
    return get_slice_values([dimspins[i].get() for i in range(y.ndim)], y)


def get_standard_name(ivar):
    """
    Get standard_name, long_name, or name attribute of netcdf variable

    Parameters
    ----------
    ivar : netCDF4._netCDF4.Variable
        netcdf variable

    Returns
    -------
    string
        standard_name, long_name, or name of variable `ivar`

    Examples
    --------
    >>> name = get_standard_name(self.fi['Tair])

    """
    if hasattr(ivar, 'standard_name'):
        return ivar.standard_name
    elif hasattr(ivar, 'long_name'):
        return ivar.long_name
    else:
        return ivar.name


def get_units(ivar):
    """
    Get units attribute of netcdf variable

    Parameters
    ----------
    ivar : netCDF4._netCDF4.Variable
        netcdf variable

    Returns
    -------
    string
        units or empty string

    Examples
    --------
    >>> units = get_units(self.fi['Tair])

    """
    if hasattr(ivar, 'units'):
        return ivar.units
    else:
        return ''


def list_intersection(lst1, lst2):
    """
    Intersection of two lists.

    From:
    https://stackoverflow.com/questions/3697432/how-to-find-list-intersection
    Using list comprehension for small lists and set() method with builtin
    intersection for longer lists.

    Parameters
    ----------
    lst1, lst2 : list
        Python lists

    Returns
    -------
    list
        List with common elements in both input lists.

    Examples
    --------
    >>> lst1 = [ 4, 9, 1, 17, 11, 26, 28, 28, 26, 66, 91]
    >>> lst2 = [9, 9, 74, 21, 45, 11, 63]
    >>> print(Intersection(lst1, lst2))
    [9, 11]

    """
    if (len(lst1) > 10) or (len(lst2) > 10):
        return list(set(lst1).intersection(lst2))
    else:
        return [ ll for ll in lst1 if ll in lst2 ]


def parse_entry(text):
    """
    Convert text string to correct data type

    Parse an entry field to None, bool, int, float, datetime, list, dict

    Parameters
    ----------
    text : str
        String from entry field

    Returns
    -------
    None, bool, int, float, datetime, list, dict

    Examples
    --------
    >>> parse_entry('7')
    7
    >>> parse_entry('7,3')
    [7, 3]

    """
    if ',' in text:
        # # list or str
        # try:
        #     tt = eval(f'[{text}]')
        # except SyntaxError:
        #     tt = text
        # parse each element
        stext = text.split(',')
        tt = [ parse_entry(ss) for ss in stext ]
    elif text == 'None':
        # None
        tt = None
    elif text == 'True':
        # bool True
        tt = True
    elif text == 'False':
        # bool False
        tt = False
    elif ':' in text:
        # dict, datetime, or str
        try:
            tt = eval(f'{{{text}}}')
        except SyntaxError:
            try:
                tt = np.datetime64(text)
            except ValueError:
                tt = text
    elif text.count('-') == 2:
        # datetime or str
        try:
            tt = np.datetime64(text)
        except ValueError:
            tt = text
    else:
        tt = text

    # if above gave str, check for scalars
    if tt == text:
        try:
            # int
            tt = int(text)
        except ValueError:
            try:
                # float
                tt = float(text)
            except ValueError:
                # str
                tt = text
            try:
                if not isfinite(tt):
                    # keep NaN and Inf string
                    tt = text
            except TypeError:
                pass
    return tt


def selvar(self, var):
    """
    Extract variable from correct file.

    Parameters
    ----------
    self : class
        ncv session or panel having fi and groups
    var : string
        Variable name including group or file identifier,
        such as 'Qle', 'group1/Qle', or 'file0001/Qle'

    Returns
    -------
    netcdf variable
        Pointer to variable in file

    Examples
    --------
    selvar(self, var)

    """
    if self.usex:
        return self.fi[var]
    if len(self.fi) == 0:
        return
    elif len(self.fi) == 1:
        fil = self.fi[0]
        return fil[var]
    else:
        if '/' in var:
            f = var[0:var.find('/')].rstrip()
            idxf = list(self.groups).index(f)
            vv = var[var.find('/') + 1:].rstrip()
            fil = self.fi[idxf]
            return fil[vv]
        else:
            fil = self.fi[0]
            return fil[var]


def set_axis_label(ncvar):
    """
    Set label plotting axis from name and unit of given variable
    `ncvar`.

    Parameters
    ----------
    ncvar : netCDF4._netCDF4.Variable
        netcdf variables class

    Returns
    -------
    str
        Label string: name (unit)

    Examples
    --------
    >>> ylab = set_axis_label(fi['w_soil'])

    """
    try:
        lab = ncvar.long_name
    except AttributeError:
        try:
            lab = ncvar.standard_name
        except AttributeError:
            lab = ncvar.name
    try:
        lab += ' (' + ncvar.units + ')'
    except AttributeError:
        pass
    return lab


def set_miss(miss, x):
    """
    Set `x` to NaN or NaT for all values in miss.

    Parameters
    ----------
    miss : iterable
        values which shall be set to np.nan or np.datetime64('NaT') in `x`
    x : ndarray
        numpy array

    Returns
    -------
    ndarray
        `x` with all values set np.nan or np.datetime64('NaT')
        that are equal to any value in miss.

    Examples
    --------
    >>> x = fi['time']
    >>> miss = get_miss(self, x)
    >>> x = set_miss(miss, x)

    """
    if np.issubdtype(x.dtype, np.datetime64):
        default = np.datetime64('NaT')
    else:
        default = np.nan
    for mm in miss:
        x = np.where(x == mm, default, x)
    return x


def spinbox_values(ndim):
    """
    Tuple for Spinbox values with 'all' before range(`ndim`) and
    'mean', 'std', etc. after range(`ndim`) if `ndim`>1,
    otherwise single entry (0,).

    Parameters
    ----------
    ndim : int
        Size of dimension.

    Returns
    -------
    tuple
        (('all',) + tuple(range(ndim)) + ('mean','std',...)) if ndim > 1
        (0,) else

    Examples
    --------
    >>> self.xd0.config(values=spinbox_values(xx.shape[0]))

    """
    if ndim > 1:
        return (('all',) + tuple(range(ndim)) + DIMMETHODS)
    else:
        return (0,)


def vardim2var(vardim, groups=[]):
    """
    Extract group index and variable name from 'group/variable (dim1=ndim1,)'.

    Parameters
    ----------
    vardim : string
        Variable name with dimensions, such as 'latitude (lat=32,lon=64)'
        or with group 'file1/latitude (lat=32,lon=64)'

    Returns
    -------
    int, string
        group index, variable name

    Examples
    --------
    >>> vardim2var('latitude (lat=32,lon=64)')
    0 latitude

    """
    var = vardim[0:vardim.rfind('(')].rstrip()
    if '/' in var:
        g = var[0:var.find('/')].rstrip()
        ig = list(groups).index(g)
    else:
        ig = 0
    return ig, var


def xzip_dim_name_length(ncvar):
    """
    Combines dimension names and length of xarray variable in list of strings.

    Parameters
    ----------
    ncvar : xarray.DataArray
        xarray variable

    Returns
    -------
    list
        List of dimension name and length in the form 'dim=len'.

    Examples
    --------
    >>> import xarray as xr
    >>> ifile = 'test.nc'
    >>> fi = xr.open_dataset(ifile)
    >>> w_soil = fi['w_soil']
    >>> print(zip_dim_name_length(w_soil))
    ('ntime=17520', 'nsoil=30')

    """
    out = []
    for i in range(ncvar.ndim):
        out.append(f'{ncvar.dims[i]}={ncvar.shape[i]}')
    return out


def zip_dim_name_length(ncvar):
    """
    Combines dimension names and length of netcdf variable in list of strings.

    Parameters
    ----------
    ncvar : netCDF4._netCDF4.Variable
        netcdf variables class

    Returns
    -------
    list
        List of dimension name and length in the form 'dim=len'.

    Examples
    --------
    >>> import netCDF4 as nc
    >>> ifile = 'test.nc'
    >>> fi = nc.Dataset(ifile, 'r')
    >>> w_soil = fi['w_soil']
    >>> print(zip_dim_name_length(w_soil))
    ('ntime=17520', 'nsoil=30')

    """
    out = []
    for i in range(ncvar.ndim):
        out.append(f'{ncvar.dimensions[i]}={ncvar.shape[i]}')
    return out
