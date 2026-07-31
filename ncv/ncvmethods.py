#!/usr/bin/env python3
"""NetCDF analysis and missing-value helpers for ncv.

This module was written by Matthias Cuntz while at Institut National de
Recherche pour l'Agriculture, l'Alimentation et l'Environnement (INRAE), Nancy,
France.

:copyright: Copyright 2020-2021 Matthias Cuntz - mc (at) macu (dot) de
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

"""
import numpy as np
from .ncvutils import selvar
from .ncvutils import xzip_dim_name_length, zip_dim_name_length
from .ncvutils import get_standard_name, get_units
import netCDF4 as nc
# nc.default_fillvals but with keys as variables['var'].dtype
nctypes = [ np.dtype(i) for i in nc.default_fillvals ]
ncfill  = dict(zip(nctypes, list(nc.default_fillvals.values())))
ncfill.update({np.dtype('O'): np.nan})
ncfill.update({np.dtype('<M8[ms]'): np.datetime64('NaT')})
ncfill.update({np.dtype('<M8[ns]'): np.datetime64('NaT')})


__all__ = ['analyse_netcdf', 'get_miss']


#
# Analyse netcdf file
#

def analyse_netcdf(self):
    """
    Call analyse_netcdf_xarray or analyse_netcdf_ncvue depending on self.usex

    Parameters
    ----------
    self : class
        ncv session or panel

    Returns
    -------
    Set variables:
        self.dunlim,
        self.time, self.tname, self.tvar, self.dtime,
        self.cols,
        self.latvar, self.lonvar, self.latdim, self.londim

    Examples
    --------
    >>> analyse_netcdf(self)

    """
    if self.usex:
        analyse_netcdf_xarray(self)
    else:
        analyse_netcdf_ncvue(self)


def analyse_netcdf_ncvue(self):
    """
    Analyse netcdf file(s) for the unlimited dimension, calculating datetime,
    variables, latitudes/longitudes variables and dimensions.

    Parameters
    ----------
    self : class
        ncv session or panel

    Returns
    -------
    Set variables:
        self.dunlim,
        self.time, self.tname, self.tvar, self.dtime,
        self.cols,
        self.latvar, self.lonvar, self.latdim, self.londim

    Examples
    --------
    >>> analyse_netcdf(self)

    """
    import datetime as dt
    try:
        import cftime as cf
    except ModuleNotFoundError:
        import netCDF4 as cf
    #
    ngroups = len(self.groups)
    for ig in range(max(ngroups, 1)):
        if len(self.fi) == 1:
            if ngroups > 0:
                ffi = self.fi[0]
                fi = ffi[self.groups[ig]]
                gname = self.groups[ig] + '/'
            else:
                fi = self.fi[ig]
                gname = ''
        else:
            fi = self.fi[ig]
            gname = self.groups[ig] + '/'
        #
        # search unlimited dimension
        self.dunlim.append('')
        for dd in fi.dimensions:
            if fi.dimensions[dd].isunlimited():
                self.dunlim[ig] = dd
                break
        #
        # search for time variable and make datetime variable
        self.time.append(None)
        self.tname.append('')
        self.tvar.append('')
        self.dtime.append(None)
        for vv in fi.variables:
            isunlim = False
            if self.dunlim[ig]:
                if vv.lower() == fi.dimensions[self.dunlim[ig]].name.lower():
                    isunlim = True
            if ( isunlim or vv.lower().startswith('time_') or
                 (vv.lower() == 'time') or (vv.lower() == 'datetime') or
                 (vv.lower() == 'date') ):
                self.tvar[ig] = gname + vv
                if vv.lower() == 'datetime':
                    self.tname[ig] = gname + 'date'
                else:
                    self.tname[ig] = gname + 'datetime'
                try:
                    ivar = selvar(self, self.tvar[ig])
                    tunit = ivar.units
                except AttributeError:
                    tunit = ''
                # assure 01, etc. if values < 1000, 10, 10 in year, month, day
                if tunit.find('since') > 0:
                    tt = tunit.split()
                    dd = tt[2].split('-')
                    tt[2] = (f'{int(dd[0]):04d}-{int(dd[1]):02d}-'
                             f'{int(dd[2]):02d}')
                    tunit = ' '.join(tt)
                try:
                    ivar = selvar(self, self.tvar[ig])
                    tcal = ivar.calendar
                except AttributeError:
                    tcal = 'standard'
                ivar = selvar(self, self.tvar[ig])
                time = ivar[:]
                # time dimension "day as %Y%m%d.%f" from cdo.
                if ' as ' in tunit:
                    itunit = tunit.split()[2]
                    dtime = []
                    for tt in time:
                        stt = str(tt).split('.')
                        sstt = ('00' + stt[0])[-8:] + '.' + stt[1]
                        dtime.append(dt.datetime.strptime(sstt, itunit))
                    ntime = cf.date2num(dtime,
                                        'days since 0001-01-01 00:00:00')
                    self.dtime[ig] = cf.num2date(
                        ntime,
                        'days since 0001-01-01 00:00:00')
                else:
                    try:
                        self.dtime[ig] = cf.num2date(time, tunit,
                                                     calendar=tcal)
                    except ValueError:
                        self.dtime[ig] = None
                if self.dtime[ig] is not None:
                    ntime = len(self.dtime[ig])
                    if (tcal == '360_day'):
                        ndays = [360.] * ntime
                    elif (tcal == '365_day'):
                        ndays = [365.] * ntime
                    elif (tcal == 'noleap'):
                        ndays = [365.] * ntime
                    elif (tcal == '366_day'):
                        ndays = [366.] * ntime
                    elif (tcal == 'all_leap'):
                        ndays = [366.] * ntime
                    else:
                        ndays = [ 365. +
                                  float((((t.year % 4) == 0) &
                                         ((t.year % 100) != 0)) |
                                        ((t.year % 400) == 0))
                                  for t in self.dtime[ig] ]
                    self.dtime[ig] = np.array([
                        t.year +
                        (t.dayofyr - 1 + t.hour / 24. +
                         t.minute / 1440 + t.second / 86400.) / ndays[i]
                        for i, t in enumerate(self.dtime[ig]) ])
                # make datetime variable
                if self.time[ig] is None:
                    try:
                        ttime = cf.num2date(
                            time, tunit, calendar=tcal,
                            only_use_cftime_datetimes=False,
                            only_use_python_datetimes=True)
                        self.time[ig] = np.array([ dd.isoformat()
                                                   for dd in ttime ],
                                                 dtype='datetime64[ms]')
                    except:
                        self.time[ig] = None
                if self.time[ig] is None:
                    try:
                        ttime = cf.num2date(time, tunit,
                                            calendar=tcal)
                        self.time[ig] = np.array([ dd.isoformat()
                                                   for dd in ttime ],
                                                 dtype='datetime64[ms]')
                    except:
                        self.time[ig] = None
                if self.time[ig] is None:
                    # if not possible use decimal year
                    self.time[ig] = self.dtime[ig]
                if self.time[ig] is None:
                    # could not interpret time at all,
                    # e.g. if units = "months since ..."
                    self.time[ig] = time
                    self.dtime[ig] = time
                break
        #
        # construct list of variable names with dimensions
        if self.time[ig] is not None:
            ivar = selvar(self, self.tvar[ig])
            addt = [self.tname[ig] + ' ' +
                    str(tuple(zip_dim_name_length(ivar)))]
            self.cols += addt
        ivars = []
        for vv in fi.variables:
            vname = gname + vv
            ss = tuple(zip_dim_name_length(fi[vv]))
            self.maxdim = max(self.maxdim, len(ss))
            ivars.append((vname, ss, len(ss)))
        self.cols += sorted([ vv[0] + ' ' + str(vv[1])
                              for vv in ivars ])
        #
        # search for lat/lon variables
        self.latvar.append('')
        self.lonvar.append('')
        # first sweep: *name must be "latitude" and
        #              units must be "degrees_north"
        if not self.latvar[ig]:
            for vv in fi.variables:
                sname = get_standard_name(fi[vv])
                if sname.lower() == 'latitude':
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees_north':
                        self.latvar[ig] = gname + vv
        if not self.lonvar[ig]:
            for vv in fi.variables:
                sname = get_standard_name(fi[vv])
                if sname.lower() == 'longitude':
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees_east':
                        self.lonvar[ig] = gname + vv
        # second sweep: name must start with lat and
        #               units must be "degrees_north"
        if not self.latvar[ig]:
            for vv in fi.variables:
                sname = fi[vv].name
                if sname[0:3].lower() == 'lat':
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees_north':
                        self.latvar[ig] = gname + vv
        if not self.lonvar[ig]:
            for vv in fi.variables:
                sname = fi[vv].name
                if sname[0:3].lower() == 'lon':
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees_east':
                        self.lonvar[ig] = gname + vv
        # third sweep: name must contain lat and
        #              units must be "degrees_north"
        if not self.latvar[ig]:
            for vv in fi.variables:
                sname = fi[vv].name
                sname = sname.lower()
                if sname.find('lat') >= 0:
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees_north':
                        self.latvar[ig] = gname + vv
        if not self.lonvar[ig]:
            for vv in fi.variables:
                sname = fi[vv].name
                sname = sname.lower()
                if sname.find('lon') >= 0:
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees_east':
                        self.lonvar[ig] = gname + vv
        # fourth sweep: axis is 'Y' or 'y'
        if not self.latvar[ig]:
            for vv in fi.variables:
                try:
                    saxis = fi[vv].axis
                except AttributeError:
                    saxis = ''
                if saxis.lower() == 'y':
                    self.latvar[ig] = gname + vv
        if not self.lonvar[ig]:
            for vv in fi.variables:
                try:
                    saxis = fi[vv].axis
                except AttributeError:
                    saxis = ''
                if saxis.lower() == 'x':
                    self.lonvar[ig] = gname + vv
        # fifth sweep: same as first but units can be simply "degrees"
        if not self.latvar[ig]:
            for vv in fi.variables:
                sname = get_standard_name(fi[vv])
                if sname.lower() == 'latitude':
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees':
                        self.latvar[ig] = gname + vv
        if not self.lonvar[ig]:
            for vv in fi.variables:
                sname = get_standard_name(fi[vv])
                if sname.lower() == 'longitude':
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees':
                        self.lonvar[ig] = gname + vv
        # sixth sweep: same as second but units can be simply "degrees"
        if not self.latvar[ig]:
            for vv in fi.variables:
                sname = fi[vv].name
                if sname[0:3].lower() == 'lat':
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees':
                        self.latvar[ig] = gname + vv
        if not self.lonvar[ig]:
            for vv in fi.variables:
                sname = fi[vv].name
                if sname[0:3].lower() == 'lon':
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees':
                        self.lonvar[ig] = gname + vv
        # seventh sweep: same as third but units can be simply "degrees"
        if not self.latvar[ig]:
            for vv in fi.variables:
                sname = fi[vv].name
                sname = sname.lower()
                if sname.find('lat') >= 0:
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees':
                        self.latvar[ig] = gname + vv
        if not self.lonvar[ig]:
            for vv in fi.variables:
                sname = fi[vv].name
                sname = sname.lower()
                if sname.find('lon') >= 0:
                    sunit = get_units(fi[vv])
                    if sunit.lower() == 'degrees':
                        self.lonvar[ig] = gname + vv
        #
        # determine lat/lon dimensions
        self.latdim.append('')
        self.londim.append('')
        if self.latvar[ig]:
            ivar = selvar(self, self.latvar[ig])
            latshape = ivar.shape
            if (len(latshape) < 1) or (len(latshape) > 2):
                print('Something went wrong determining lat/lon:'
                      ' latitude variable is not 1D or 2D.\n'
                      'latitude variable with dimensions:',
                      self.latvar[ig], ivar.dimensions)
                self.latvar[ig] = ''
            else:
                self.latdim[ig] = ivar.dimensions[0]
        if self.lonvar[ig]:
            ivar = selvar(self, self.lonvar[ig])
            lonshape = ivar.shape
            if len(lonshape) == 1:
                self.londim[ig] = ivar.dimensions[0]
            elif len(lonshape) == 2:
                self.londim[ig] = ivar.dimensions[1]
            else:
                print('Something went wrong determining lat/lon:'
                      ' longitude variable is not 1D or 2D.\n'
                      'longitude variable with dimensions:',
                      self.lonvar[ig], ivar.dimensions)
                self.lonvar[ig] = ''
        #
        # add units to lat/lon name
        if self.latvar[ig]:
            ivar = selvar(self, self.latvar[ig])
            idim = tuple(zip_dim_name_length(ivar))
            self.latvar[ig] = self.latvar[ig] + ' ' + str(idim)
        if self.lonvar[ig]:
            ivar = selvar(self, self.lonvar[ig])
            idim = tuple(zip_dim_name_length(ivar))
            self.lonvar[ig] = self.lonvar[ig] + ' ' + str(idim)


def analyse_netcdf_xarray(self):
    """
    Analyse netcdf file(s) opened with xarray for time (= unlimited),
    variables, latitudes/longitudes variables and dimensions.

    Parameters
    ----------
    self : class
        ncv session or panel

    Returns
    -------
    Set variables:
        self.dunlim,
        self.time, self.tname, self.tvar, self.dtime,
        self.cols,
        self.latvar, self.lonvar, self.latdim, self.londim

    Examples
    --------
    >>> analyse_netcdf_xarray(self)

    """
    import xarray as xr

    #
    # search time
    self.dunlim = None
    self.time = None
    self.tname = ''
    self.tvar = None
    self.dtime = None
    for cc in self.fi.coords:
        if np.issubdtype(self.fi.coords[cc].dtype, np.datetime64):
            self.dunlim = cc
            break
    if self.dunlim is None:
        if 'time' in self.fi.coords:
            self.dunlim = 'time'
    if self.dunlim is not None:
        self.time = self.fi.coords[self.dunlim]
        self.tname = self.dunlim
        self.tvar = self.dunlim
        self.dtime = self.time

    #
    # construct list of variable names with dimensions
    if self.time is not None:
        addt = [self.tname + ' ' +
                str(tuple(xzip_dim_name_length(self.fi[self.tvar])))]
        self.cols += addt
    ivars = []
    for vv in self.fi.variables:
        if vv != self.tname:
            ss = tuple(xzip_dim_name_length(self.fi[vv]))
            ndim = self.fi[vv].ndim
            self.maxdim = max(self.maxdim, ndim)
            ivars.append((vv, ss, ndim))
    self.cols += sorted([ vv[0] + ' ' + str(vv[1])
                          for vv in ivars ])

    #
    # search for lat/lon variables
    # first sweep: units must be "degrees_north" and "degrees_east"
    self.latvar = ''
    self.lonvar = ''
    for cc in self.fi.coords:
        ccoord = self.fi.coords[cc]
        cunits = get_units(ccoord)
        if cunits == 'degrees_north':
            self.latvar = cc
        if cunits == 'degrees_east':
            self.lonvar = cc
    # second sweep: axis is 'x', 'y', or 'X', 'Y'
    if not self.latvar:
        for cc in self.fi.coords:
            if cc.lower() == 'y':
                self.latvar = cc
    if not self.lonvar:
        for cc in self.fi.coords:
            if cc.lower() == 'x':
                self.lonvar = cc

    #
    # determine lat/lon dimensions
    self.latdim = ''
    self.londim = ''

    #
    # add units to lat/lon name
    if self.latvar:
        idim = tuple(xzip_dim_name_length(self.fi[self.latvar]))
        self.latvar = self.latvar + ' ' + str(idim)
    if self.lonvar:
        idim = tuple(xzip_dim_name_length(self.fi[self.lonvar]))
        self.lonvar = self.lonvar + ' ' + str(idim)


#
# Get list of missing values
#

def get_miss(self, x):
    """
    Get list of missing values, i.e. self.miss, x._FillValue,
    x.missing_value, and from netcdf4.default_fillvals.

    Parameters
    ----------
    self : class
        ncv session or panel
    x : netCDF4._netCDF4.Variable
        netcdf variable

    Returns
    -------
    list
        List with missing values self.miss, x._FillValue,
        x.missing_value if present,  and from netcdf4.default_fillvals

    Examples
    --------
    >>> x = fi['time']
    >>> miss = get_miss(self, x)

    """
    try:
        out = [ncfill[x.dtype]]
    except KeyError:
        out = []
    try:
        if x.dtype != np.dtype('<M8[ms]'):
            out += [self.miss]
    except AttributeError:
        pass
    try:
        out += [x._FillValue]
    except AttributeError:
        pass
    try:
        out += [x.missing_value]
    except AttributeError:
        pass
    return out
