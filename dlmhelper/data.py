# -*- coding: utf-8 -*-

"""This module provides two classes which are used to handle data from DLM results.

- Grid: Container for gridded data which can be easily converted to a TimeSeries object
- TimeSeries: Container for timeseries data which is used for all dlm fit functions
- DLMResult: Handles single results
- DLMResultList: Acts as a container for all DLMResults generated for a given time series


"""

import tarfile
import os
import json
import datetime


from dataclasses import dataclass, fields, asdict, field
from typing import Union, List, Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
import matplotlib.gridspec 

from tabulate import tabulate

import dlmhelper.spatio_temporal

from statsmodels.tsa.statespace.structural import UnobservedComponentsResults

from dlmhelper import *

def _to_datetime64(time: ArrayLike, time_unit: str, reference_time: Union[np.datetime64,datetime.datetime]) -> np.ndarray:
    """
    Convert an array of time values given as fractionl values with unit [time_unit] to datetime64 values
    using the given reference_time

    :param time: Array of floats
    :type time: ArrayLike
    :param time_unit: time_unit the floats from `time` correspond to
    :type time_unit: str 
    :param reference_time:
    :type reference_time: np.datetime64 | datetime.datetime
    :returns: Array of datetime64 values
    :rtype: np.ndarray[np.datetime64]
    """
    
    time = np.asarray(time)
    reference_time = np.datetime64(reference_time)
    
    if len(time.shape)>1:
        raise ValueError("time needs to be one-dimensional!")
    
    if time_unit in MILLISECOND_ALIASES:
        time_unit = 'ms'
    elif time_unit in SECOND_ALIASES:
        time_unit = 's'
    elif time_unit in MINUTE_ALIASES:
        time_unit = 'm'
    elif time_unit in HOUR_ALIASES:
        time_unit = 'h'
    elif time_unit in DAY_ALIASES:
        time_unit = 'D'
    elif time_unit in WEEK_ALIASES:
        time_unit = 'W'
    elif time_unit in MONTH_ALIASES:
        time_unit = 'M'
    elif time_unit in YEAR_ALIASES:
        time_unit =  'Y'
    else:
        raise ValueError(f"time_unit {time_unit} not found!")
        
    out_time = np.empty(time.shape, dtype='datetime64[ms]')    
    
    for i, t in enumerate(time):
        out_time[i] = reference_time + t*np.timedelta64(1,time_unit).astype('timedelta64[ms]')
        
        
    return out_time

def _datetime64_to_delta(time: np.ndarray, time_unit: str) -> np.ndarray:
    """
    Convert an array containing datetime64 values to an array 
    with integers of `time_unit` since 1970-01-01.
    
    :param time: Array of datetime64
    :type time: np.ndarray
    :param time_unit: time_unit the floats from `time` correspond to
    :type time_unit: str 
    
    :returns: Array of int values which correspond to the time_units
        since epoch (1970-01-01)
    :rtype: np.ndarray[int]
    """
    
    if len(time.shape)>1:
        raise ValueError("time needs to be one-dimensional!")
    
    if time_unit in MILLISECOND_ALIASES:
        time_unit = 'ms'
    elif time_unit in SECOND_ALIASES:
        time_unit = 's'
    elif time_unit in MINUTE_ALIASES:
        time_unit = 'm'
    elif time_unit in HOUR_ALIASES:
        time_unit = 'h'
    elif time_unit in DAY_ALIASES:
        time_unit = 'D'
    elif time_unit in WEEK_ALIASES:
        time_unit = 'W'
    elif time_unit in MONTH_ALIASES:
        time_unit = 'M'
    elif time_unit in YEAR_ALIASES:
        time_unit =  'Y'
    else:
        raise ValueError(f"time_unit {time_unit} not found!")
        
    out_time = np.empty(time.shape, dtype=int)    
    
    for i, t in enumerate(time):
        out_time[i] = (time[i] - np.datetime64('1970-01-01')).astype(f'timedelta64[{time_unit}]').astype(int)
        
        
    return out_time


def get_grid_dim(lat_min: float, lat_max: float, lon_min: float, lon_max: float, 
             lat_step: float, lon_step: float) -> dict:
    """
    Returns a dictionary containing the specified grid dimensions.
    Used by various functions in this package.
    
    :param lat_min: Minimum latitude
    :type lat_min: float
    :param lat_max: Maximum latitude
    :type lat_max: float
    :param lon_min: Minimum longitude
    :type lon_min: float
    :param lon_max: Maximum longitude
    :type lon_max: float
    :param lat_step: Latitude step-size
    :type lat_step: float
    :param lon_step: Longitude step-size
    :type lon:step: float 
    :returns: dictionary
    :rtype: dict
    """
    grid_dim = {
        'LAT_LOW': lat_min,
        'LAT_HIGH': lat_max, 
        'LON_LOW': lon_min,
        'LON_HIGH': lon_max,
        'LAT_STEP': lat_step,
        'LON_STEP': lon_step
    }
    
    return grid_dim

def _is_grid_dim(g: dict) -> bool:
    """
    Checks whether a dictionary is a dictionary defined by
    :py:func:`dlmhelper.dlm_data.grid_dim'.
    
    :param g: The dict to test
    :type: g: dict
    :returns: A boolean
    :rtype: bool

    """
    if len(g.keys()) != 6:
        return False
    
    keys = get_grid_dim(0,0,0,0,0,0).keys()
    
    for k in keys:
        if k not in g.keys():
            return False
        
    return True

    
@dataclass
class TimeSeries:
    """This class acts as a container for timeseries data.
    When creating a TimeSeries object either `time` or `time64`
    AND `time_unit` and `reference_time` has to be specified.
    
    The data is then automatically sorted and missing time steps
    of `time_unit` are added to the arrays (e.g. for daily data all
    missing days are added, if `time_unit=hours` all missing hours are
    added) and filled with NaN's.
    
    
    :param data: The timeseries data of shape (time)
    :type data: np.ndarray
    :param time: Time values corresponding to `time_unit` since
        `reference_time` of shape (time)
    :type time: np.ndarray, optional
    :param time_unit: Unit of the values from `time`.
     Possible values are listed in :data:`dlmhelper.TIME_ALIASES`
    :type time_unit: str
    :param reference_time: Reference time for the values from
        time array, defaults to Unix-epoch
    :type reference_time: np.datetime64, optional
    :param error: Errors for the timeseries data
        of shape (time)
    :type error: np.ndarray, optional
    :param N: Number of datapoints averaged for each
        timestep of shape (time)
    :type N: np.ndarray, optional
    :param product_type: Identifier of the data used
    :type product_type: str, optional
    :param grid_dim: Dimensions of the averaged
        area, can be created with :func:`dlmhelper.data.grid_dim`
    :type grid_dim: dict, optional
    """
    data: np.ndarray
    time: np.ndarray = None
    time_unit: str = "day"
    time64: np.ndarray = None
    reference_time: np.datetime64 = np.datetime64('1970-01-01')
    error: np.ndarray = None
    N: np.ndarray = None
    product_type: str = None
    grid_dim: dict = None
    
    
    #@classmethod
    #def create(cls,name, timeseries: TimeSeries, result: UnobservedComponentsResults, score = None):
    #    pass
    
    def __post_init__(self):
        
        if self.data is None:
            raise ValueError("The data field must be provided during initialization")
        if (self.reference_time is None or self.time_unit is None) or (self.time is None and self.time64 is None):
            raise ValueError("Either the time or the time64 and the reference_time and time_unit field need to be provided during initialization!")
            
        self.data = np.asarray(self.data)
        self.reference_time = np.datetime64(self.reference_time)
        
        if self.error is None:
            self.error = np.full(self.data.shape, np.nan)
        else:
            self.error = np.asarray(self.error)
        
        if self.N is None:
            self.N = np.full(self.data.shape, np.nan)
        else:
            self.N = np.asarray(self.N)
                 
        if self.time64 is None:
            self.time64 = np.full(self.data.shape, np.nan)
        else:
            self.time64 = np.asarray(self.time64)
        
        if self.time is None:
            self.time = _datetime64_to_delta(self.time64,self.time_unit)
        else:
            self.time = np.asarray(self.time)
            
        if len(self.data.shape)>1 or len(self.error.shape)>1 or len(self.time.shape)>1 or len(self.N.shape)>1  or len(self.time64.shape)>1:
            raise ValueError("data, error, N and time field need to be one-dimensional!")
        
        if self.data.shape != self.error.shape or self.data.shape != self.time.shape or self.data.shape != self.N.shape or self.data.shape != self.time64.shape:
            raise ValueError("data, error, N and time need to have the same size!")
            
        if self.product_type is not None:
            if type(self.product_type)!=str:
                raise TypeError("product_type must be string!")
        if self.grid_dim is not None:
            if not _is_grid_dim(self.grid_dim):
                raise TypeError("grid_dim has the wrong structure. Use grid_dim function to initialize dict!")
            
         #Sort time series data and add missing days (filled with NaNs)    
        _t = self.time
        _d = self.data
        _e = self.error
        _N = self.N
    
        start = np.nanmin(_t)
        end = np.nanmax(_t)
        size = int(end-start+1)

        data = np.full((size),np.nan)
        error = np.full((size), np.nan)
        N = np.zeros((size))

        time = np.zeros((size))

        for i in range(size):
            time[i]=np.nanmin(_t)+i
            if (_d[_t==start+i].shape[0]>0):
                data[i] = _d[_t==start+i][0]
                error[i] = _e[_t==start+i][0]
                if _N is not None: N[i] =_N[_t==start+i][0]

            else:
                data[i] = np.nan
                error[i] = np.nan
                N[i] = 0
        
        self.time = time
        self.data = data
        self.error = error
        
        if not np.all(np.isnan(self.N)):
            self.N = N
        else:
            self.N = np.full(self.data.shape, np.nan)
        
        
        self.time64 = _to_datetime64(self.time, self.time_unit, self.reference_time)
        
            
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for field_name, field_type in self.__annotations__.items():
            if field_type == np.ndarray:
                if not np.array_equal(getattr(self, field_name), getattr(other, field_name),equal_nan=True):
                    return False
            elif field_name == "time64":
                if not np.all(np.equal(getattr(self, field_name), getattr(other, field_name))):
                    return False
            else:    
                if getattr(self, field_name) != getattr(other, field_name):
                    return False
        return True
    
    @classmethod
    def load(cls,path):
        """Load TimeSeries object from .json

        :param path: path to .json file
        :type path: str
        :returns: TimeSeries object
        :rtype: TimeSeries
        """
        with open(path,'r') as f:
            s = f.read()
            argDict = json.loads(s, object_hook=JSONDecoder)
            fieldSet = {f.name for f in fields(cls) if f.init}
            filteredArgDict = {k : v for k, v in argDict.items() if k in fieldSet}
            
            filteredArgDict["reference_time"] = np.datetime64(filteredArgDict["reference_time"])
            
            return cls(**filteredArgDict)
            
    def save(self,path,fname, verbose=1):
        """Save TimeSeries object as .json
        Filename will be 'TimeSeries_fname.json'

        :param path: Path to .json file
        :type path: str
        :param fname: fname
        :type fname: str
        """
        s=json.dumps(asdict(self),cls=NpDecoder, indent=2)
        cpath = path+'TimeSeries_'+fname+'.json'
        with open(cpath, 'w') as f:
            f.write(s)
            if verbose>=1:
                print("Saved data at:", cpath)
                
                
    def plot(self, ax=None):
        """
        Plot the time series. If `ax` is not specified, create a
        new figure. Returns the figure and axis.

        :param ax: (Optional) The axis the plot should be drawn on,
            defaults to None
        :type ax: matplotlib.axes, optional
        :returns: The axis and figure
        :rtype: matplotlib.axes, matlotlib.figure
        """
        if self.time64 is not None:
            time = self.time64
        else:
            time = self.time

        if ax is None: fig, ax = plt.subplots()

        ax.scatter(time, self.data,color=C4, marker='.')

        ax.text(0.05,0.9,f"{self.product_type}",transform=ax.transAxes)
        ax.tick_params(axis='y', labelrotation=45)
        return ax.get_figure(), ax

@dataclass
class Grid:
    """This class acts as a container for gridded data.
    When creating a Grid object either `time` or `time64`
    AND `time_unit` and `reference_time` has to be specified.
    
    The data is then automatically sorted and missing time steps
    of `time_unit` are added to the arrays (e.g. for daily data all
    missing days are added, if `time_unit=hours` all missing hours are
    added) and filled with NaN's. 
    
    """
    data: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray = None
    time64: np.ndarray = None
    time_unit: str = 'day'
    reference_time: np.datetime64 = np.datetime64('1970-01-01')
    
    error: np.ndarray = None
    N: np.ndarray = None
    
    product_type: str = None
    grid_dim: dict = None
    
    def __post_init__(self):
        
        if self.data is None or self.lat is None or self.lon is None:
            raise ValueError("Data, lat and lon field need to be provided during initialization")
        
        if (self.reference_time is None or self.time_unit is None) or (self.time is None and self.time64 is None):
            raise ValueError("Either the time or the time64 and the reference_time and time_unit field need to be provided during initialization!")
        
        self.data = np.asarray(self.data)
        self.lat = np.asarray(self.lat)
        self.lon = np.asarray(self.lon)
        
            
        if self.N is not None:
            self.N = np.asarray(self.N)
            
        if self.error is not None:
            self.error = np.asarray(self.error)
            
        if len(self.lat.shape)>1 or len(self.lon.shape)>1 or len(self.time.shape)>1:
            raise ValueError("lat, lon and time field need to be one-dimensional!")
            
        if self.lat.size>1:
            if not np.all(np.diff(self.lat)==np.diff(self.lat)[0]):
                raise ValueError("latitude grid needs to be evenly spaced!")
                
        if self.lon.size>1:
            if not np.all(np.diff(self.lon)==np.diff(self.lon)[0]):
                raise ValueError("longitude grid needs to be evenly spaced!")
            
        if self.data.shape != (self.lat.shape[0], self.lon.shape[0], self.time.shape[0]):
            raise ValueError(f"data array shape must be ({self.lat.shape[0]}, {self.lon.shape[0]}, {self.time.shape[0]}) but is {self.data.shape}")
        
        if self.error is not None:
            if self.error.shape != (self.lat.shape[0], self.lon.shape[0], self.time.shape[0]):
                raise ValueError(f"error array shape must be ({self.lat.shape[0]}, {self.lon.shape[0]}, {self.time.shape[0]}) but is {self.error.shape}")
            
        if self.N is not None:
            if self.N.shape != (self.lat.shape[0], self.lon.shape[0], self.time.shape[0]):
                raise ValueError(f"N array shape must be ({self.lat.shape[0]}, {self.lon.shape[0]}, {self.time.shape[0]}) but is {self.N.shape}")
        
        if self.grid_dim is not None:
            if not _is_grid_dim(self.grid_dim):
                raise TypeError("grid_dim has the wrong structure. Use grid_dim function to initialize dict!")
       
        if self.time is None:
            self.time = _datetime64_to_delta(self.time64,self.time_unit)
        else:
            self.time = np.asarray(self.time)
            
        #Sort gridded data and add missing days (filled with NaNs)    
        _t = self.time
        _d = self.data
        _e = self.error
        _N = self.N
    
        start = np.nanmin(_t)
        end = np.nanmax(_t)
        size = int(end-start+1)

        data = np.full((_d.shape[0],_d.shape[1],size), np.nan)
        if _e is not None: error = np.full((_d.shape[0],_d.shape[1],size), np.nan)
        if _N is not None: N = np.zeros((_d.shape[0],_d.shape[1],size))

        time = np.zeros((size))

        for i in range(size):
            time[i]=np.nanmin(_t)+i
            if (_d[:,:,_t==start+i].shape[2]>0):
                data[:,:,i] = _d[:,:,_t==start+i][:,:,0]
                if _e is not None: error[:,:,i] = _e[:,:,_t==start+i][:,:,0]
                if _N is not None: N[:,:,i] =_N[:,:,_t==start+i][:,:,0]

            else:
                data[:,:,i] = np.nan
                if _e is not None: error[:,:,i] = np.nan
                if _N is not None: N[:,:,i] = 0
        
        self.time = time
        self.data = data
        if _e is not None: self.error = error
        if _N is not None: self.N = N
        
        self.time64 = _to_datetime64(self.time, self.time_unit, self.reference_time)
        
    def inhomogeneity_spatial(self, scale_lat: float = None, scale_lon: float = None):
        """Return the spatial inhomogeneity of the data using
        :func:`dlmhelper.spatio_temporal.inhomogeneity_spatial`

        :param scale_lat: Weight of the latitudinal part of the spatial
            inhomogeneity, if not specified lat and lon part will be
            equally weighted, defaults to `None`
        :type scale_lat: float, optional
        :param scale_lon: Weight of the longitudinal part of the spatial
            inhomogeneity, if not specified lat and lon part will be
            equally weighted, defaults to `None`
        :type scale_lon: float, optional
        :returns: Array of shape (time, 3), aach row contains the
            inhomogeneity, asymmetry component, and entropy component
            for the corresponding time step in N.
        :rtype: np.ndarray
             """
        if self.N is None:
            raise ValueError("Spatial inhomogeneity can only be calculated if number of data points per grid cell (N) is available!")
        return dlmhelper.spatio_temporal.inhomogeneity_spatial(self.lat, self.lon, self.N, scale_lat = scale_lat, scale_lon = scale_lon)

    def inhomogeneity_temporal(self):
        """ Return the temporal inhomogeneity of the data using
        :func:`dlmhelper.spatio_temporal.inhomogeneity_temporal`
        
        :returns: Array of temporal homogeneity values at each grid point,
            with shape (lat, lon, 3).
            The last dimension contains the temporal inhomogeneity, 
            asymmetry component, and entropy component.
        :rtype: np.ndarray:
        """
        if self.N is None:
            raise ValueError("Temporal inhomogeneity can only be calculated if number of data points per grid cell (N) is available!")
        return dlmhelper.spatio_temporal.inhomogeneity_temporal(self.lat, self.lon, self.time, self.N)
    
    def filter_inhomogeneity_spatial(self, hs_lim: float = None, scale_lat: float = None, scale_lon: float = None):
        """Filter the data using the spatial inhomogeneity (`hs`)
        calculated by
        :func:`dlmhelper.spatio_temporal.inhomogeneity_spatial`.
        Filters days with `hs>hs_lim`, if `hs_lim` is not specified
        it will be calculated using the following formula::
        hs_lim = median(hs) + 2*std(hs)

        :param hs_lim:  Maximum spatial inhomogeneity
        :type hs_lim: float, optional
        :param scale_lat: Weight of the latitudinal part of the spatial
            inhomogeneity, if not specified lat and lon part will be
            equally weighted, defaults to `None`
        :type scale_lat: float, optional
        :param scale_lon: Weight of the longitudinal part of the spatial
            inhomogeneity, if not specified lat and lon part will be
            equally weighted, defaults to `None`
        :type scale_lon: float, optional

        """
        if self.N is None:
            raise ValueError("Spatial inhomogeneity can only be calculated if number of data points per grid cell (N) is available!")
            
        hs = dlmhelper.spatio_temporal.inhomogeneity_spatial(self.lat, self.lon, self.N, scale_lat = scale_lat, scale_lon = scale_lon)
        
        if hs_lim is None:
            hs_lim= np.nanmedian(hs[:,0])+2*np.nanstd(hs[:,0])
            
        self.data[...,hs[...,0]>hs_lim]=np.nan
        self.error[...,hs[...,0]>hs_lim]=np.nan
        if self.N is not None:
            self.N[...,hs[...,0]>hs_lim]=0
        
    def filter_inhomogeneity_temporal(self, ht_lim: float = 0.5):
        """Filter the data using the temporal inhomogeneity (`ht`)
        calculated by
        :func:`dlmhelper.spatio_temporal.inhomogeneity_temporal`.
        Filters grid cells with `ht>ht_lim`.

        :param ht_lim: Limit for temporal inhomogeneity,
            defaults to 0.5
        :type ht_lim: float
        """
        if self.N is None:
            raise ValueError("Temporal inhomogeneity can only be calculated if number of data points per grid cell (N) is available!")
        ht = dlmhelper.spatio_temporal.inhomogeneity_temporal(self.lat, self.lon, self.time, self.N)
        
        self.data[ht[...,0]>ht_lim,:]=np.nan
        self.error[ht[...,0]>ht_lim,:]=np.nan
        if self.N is not None:
            self.N[ht[...,0]>ht_lim,:]=0
            
    def to_TimeSeries(self, zonal_avg: bool = False):
        """Creates a TimeSeries object from the gridded data
        by calculating the area-weighted average.
        If `zonal_avg=True` the data is first averaged over the
        longitudes and only then over the latitudes. This can help
        with the sampling bias in certain cases (e.g. if the data
        represents an atmospheric trace gas which is zonally 
        well-mixed).

        :param zonal_avg: Whether to average zonally first
        :type zonal_avg: bool
        :returns: TimeSeries object from the data
        :rtype: TimeSeries
        """
        avg_data, avg_error = dlmhelper.spatio_temporal._area_weighted_average(self.data, self.error, self.lat, self.lon, zonal_avg = zonal_avg)
        if self.N is not None: 
            avg_N = np.zeros(self.time.size)
            avg_N = np.nansum(self.N, axis=(0,1),out=avg_N, where=(~np.isnan(np.nanmean(self.N, axis=(0,1)))))
        else:
            avg_N = None
    
        return TimeSeries(avg_data, self.time, self.time_unit, error =  avg_error, N = avg_N, product_type = self.product_type, reference_time = self.reference_time, grid_dim = self.grid_dim)
        
        

@dataclass
class DLMResult:
    """This class is a container for results from dlm fits.
    Objects from this class are created by 
    :func:`dlmhelper.tools.dlm_fit` or 
    :func:`dlmhelper.tools.dlm_ensemble`
    or when loading saved results from a .json file using
    :func:`DLMResult.load`.
    
    :param name: Name/Identifier of object
    :type name: str
    :param timeseries: TimeSeries object that has been fitted
    :type timeseries: TimeSeries
    :param dlm_specification: Dict containing the dlm configuration
    :type dlm_specification: dict
    :param dlm_fit_params: Dict containing the values of the
        underlying parameters fitted by the dlm
    :type dlm_fit_params: dict
    :param dlm_fit_rating: Dict containing the various metrics
        which aid in comparing diffrent dlm fits
    :type dlm_fit_rating: dict
    
    :param level: level component of shape (time)
    :type level: np.ndarray
    :param level_cov: covariance of level component of
        shape (time)
    :type level_cov: np.ndarray
    :param trend: trend component of shape (time)
    :type trend: np.ndarray
    :param trend_cov: covariance of trend component of
        shape (time)
    :type trend_cov: np.ndarray
    :param seas: seasonal components of shape (time, n) where
        `n` is the number of seasonal components
    :type seas: np.ndarray
    :param seas_cov: covariance of seasonal components of
        shape (time,n)
    :type seas_cov: np.ndarray
    :param ar: auto-regressive component of shape (time)
    :type ar: np.ndarray
    :param ar_cov: covariance of auto-regressive component of
        shape (time)
    :type ar_cov: np.ndarray
    :param resid: Residual of shape (time)
    :type resi: np.ndarray
    :param _loglikeburn: 
        UnobservedComponentsResults.loglikelihood_burn
    :type _loglikeburn: int
    """
    name: str 
    timeseries: TimeSeries 
    
    dlm_specification: dict
    dlm_fit_params: dict
    dlm_fit_rating: dict
    
    level: np.ndarray
    level_cov: np.ndarray
    trend: np.ndarray
    trend_cov: np.ndarray
    seas: np.ndarray
    seas_cov: np.ndarray
    ar: np.ndarray
    ar_cov: np.ndarray
    resid: np.ndarray
    
    _loglikeburn: int
    
    
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for field_name, field_type in self.__annotations__.items():
            if field_type == np.ndarray:
                if not np.array_equal(getattr(self, field_name), getattr(other, field_name),equal_nan=True):
                    return False
            elif field_type == TimeSeries:
                if not getattr(self, field_name).__eq__(getattr(other, field_name)):
                    return False
            else:
                if getattr(self, field_name) != getattr(other, field_name):
                    return False
        return True
    
    
    def __post_init__(self):
        # add aggregated rating
        _cov_level = self.dlm_fit_rating["cov_level"]
        _cov_seas = self.dlm_fit_rating["cov_seas"]
        _cv_amse = self.dlm_fit_rating["cv_amse"]
        self.dlm_fit_rating["agg"] = _cov_level+_cov_seas+_cv_amse
        
    @classmethod
    def create(cls,name, timeseries: TimeSeries, result: UnobservedComponentsResults, score = None):
        """Creates a DLMResult object from a TimeSeries object
        and an UnobservedComponentsResults object

        :param name: Identifier
        :type name: str
        :param timeseries: Fitted timeseries
        :type timeseries: TimeSeries
        :param result: Fit results
        :type resutl: UnobservedComponentsResults
        :param score: dictionary of cross validation scores (this will
            change in the future)
        :type score: dict
        :returns: DLMResult object
        :rtype: DLMResult
        """
        res = result
        
        size = timeseries.time.size
        if res.level is not None:
            lvl = res.level['smoothed']
            lvl_cov = res.level['smoothed_cov']
        else:
            lvl_cov = lvl = np.zeros(time.shape)

        if res.trend is not None:
            trend = res.trend['smoothed']
            trend_cov = res.trend['smoothed_cov']
        else:
            trend_cov = trend = np.zeros(time.shape)

        if res.freq_seasonal is not None:
            seas = np.empty([size, len(res.freq_seasonal)])
            seas_cov = np.empty([size, len(res.freq_seasonal)])
            for i in range(len(res.freq_seasonal)):
                seas[:,i] = res.freq_seasonal[i]['smoothed']
                seas_cov[:,i] = res.freq_seasonal[i]['smoothed_cov']
        else:
            seas_cov = seas = np.zeros(size)

        if res.autoregressive is not None:
            ar = res.autoregressive['smoothed']
            ar_cov = res.autoregressive['smoothed_cov']
        else:
            ar_cov = ar = np.zeros(size)
            
        
        #resid = res.resid <- this seems to be different from the line below
        # TODO: find out why, but this works for now
        resid = timeseries.data - (lvl+np.sum(seas, axis=1)+ar)
        spec = res.specification
        
        ex_score = np.nan
        if score is not None:
            if cls._name_from_spec(spec) in score:
                ex_score = score[cls._name_from_spec(spec)]
            
        
        _dicts = [dict(zip(["param", "std_err"], [res.params[i], np.sqrt(np.diag(res.cov_params()))[i]])) for i in range(res.params.shape[0])]
        dlm_fit_params = dict(zip(res.param_names, _dicts))
        
        _nb = res.loglikelihood_burn
        
        
        dlm_fit_rating = {
            "converged": res.mle_retvals['converged'],
            "aic": res.aic,
            "ll": res.llf,
            "ssr": np.nansum(res.resid[_nb:]**2),
            "mse": np.nanmean(res.resid[_nb:]**2),
            "cov_level": np.nanmean(res.level['smoothed_cov'][_nb:]),
            "cov_trend": np.nanmean(res.trend['smoothed_cov'][_nb:]),
            "cov_seas": np.nanmean(np.sum(seas_cov,axis=1)[_nb:]),
            "cov_ar": np.nanmean(ar_cov[_nb:]),
            "cv_amse": ex_score
        }
        
        return cls(name, timeseries,spec,dlm_fit_params,dlm_fit_rating,lvl, lvl_cov, trend, trend_cov, seas, seas_cov, ar, ar_cov, resid,_nb)
        
    @classmethod
    def load(cls,path):
        """Load TimeSeries object from .json

        :param path: path to .json file
        :type path: str
        :returns: DLMResult object
        :rtype: DLMResult
        """
        with open(path,'r') as f:
            s = f.read()
            argDict = json.loads(s, object_hook=JSONDecoder)
            fieldSet = {f.name for f in fields(cls) if f.init}
            filteredArgDict = {k : v for k, v in argDict.items() if k in fieldSet}
            
            filteredArgDict["timeseries"]= TimeSeries(**filteredArgDict["timeseries"])
            
            return cls(**filteredArgDict)
            
    def save(self,path,fname=None, verbose = 1):
        """Save DLMResult object as .json
        Filename will be 'DLMResult_fname.json'.
        If `fname` is not specified, the file is named using the
        `name` field of the object and an identifier generated
        from the DLM configuration.

        :param path: Path to .json file
        :type path: str
        :param fname: Filename, defaults to None
        :type fname: str, optional
        """
        if not fname:
            fname = f"DLMResult_{self.name}_{self.name_from_spec()}"
            
        s=json.dumps(asdict(self),cls=NpDecoder, indent=2)
        cpath = path+fname+".json"
        with open(cpath, 'w') as f:
            f.write(s)
            if verbose>=1:
                print("Saved data at:",cpath)
    
    def plot_summary(self, seas=True, fig=None, figsize=None):
        """
        Create a figure showing an overview of the dlm fit.
        
        :param seas: Whether the seasonal component is plotted
            in fit overview (if `False` only plot the level),
            defaults to True
        :type seas: bool, optional
        :param fig: Figure to use
        :type fig: matplotlib.figure, optional
        :param figsize: Tuple describing the figure size
        :type figsize: Tuple, optional
        :returns: Figure
        :rtype: matplotlib.figure
        """
        
        if self.timeseries.time64 is not None:
            time = self.timeseries.time64
        else:
            time = self.timeseries.time
            
        if fig is None:
            fig = plt.figure(figsize=figsize)
            
        gs = matplotlib.gridspec.GridSpec(4,5,figure=fig)
        
        ax1 = fig.add_subplot(gs[:,0:3])
        
        _, _ = self.plot(ax=ax1, seas=seas)
        
        ax2 = fig.add_subplot(gs[0:1,3:])
        ax2.set_title("Trend")
        ax2.plot(time, self.trend,color=CF)
        ax2.plot(time, self.trend+np.sqrt(self.trend_cov),color=CF,ls='--')
        ax2.plot(time, self.trend-np.sqrt(self.trend_cov),color=CF,ls='--')
        ax2.yaxis.set_ticks_position("right")
        ax2.yaxis.set_label_position("right")
        ax2.xaxis.set_tick_params(labelbottom=False)
        _ymax = np.nanpercentile(self.trend+np.sqrt(self.trend_cov),99)
        _ymin = np.nanpercentile(self.trend-np.sqrt(self.trend_cov),1)
        ax2.set_ylim(_ymin, _ymax)

        
        ax3 = fig.add_subplot(gs[1:2,3:],sharex=ax2)
        for i in range(0, self.seas.shape[1]):
            ax3.plot(time, self.seas[:,i][:],color=CF)
        ax3.set_title("Seasonal")
        ax3.yaxis.set_ticks_position("right")
        ax3.yaxis.set_label_position("right")
        ax3.xaxis.set_tick_params(labelbottom=False)
        
        ax4 = fig.add_subplot(gs[2:3,3:],sharex=ax2)
        ax4.set_title("Auto-regressive")
        ax4.plot(time, self.ar,color=CF)
        ax4.yaxis.set_ticks_position("right")
        ax4.yaxis.set_label_position("right")
        ax4.xaxis.set_tick_params(labelbottom=False)
       
        ax5 = fig.add_subplot(gs[3:4,3:],sharex=ax2)
        ax5.set_title("Residual")
        _resid = np.copy(self.resid)
        _resid[np.isnan(_resid)]=0
        ax5.plot(time, _resid,color=CF)
        ax5.yaxis.set_ticks_position("right")
        ax5.yaxis.set_label_position("right")
        _ymax = np.nanpercentile(np.abs(self.resid),99)
        ax5.set_ylim(-_ymax, _ymax)
        
        #TODO: fix and readd this
        #from scipy.stats import norm
        #dat = self.resid[~np.isnan(self.resid)][self._loglikeburn:]
        #mu, std = norm.fit(dat)
        #x=np.linspace(-_ymax,_ymax,20)
        #p = norm.pdf(x, mu, std)

        #axins = ax1.inset_axes([0.1,0.1,1,1])
        #axins.get_yaxis().set_visible(False)
        #axins.hist(dat,color=CH1,bins=x,density=True)
        #axins.plot(x,p,color=CF,lw=2)

        return fig
    
    def plot(self, ax=None,seas=True):
        """
        Plot the fit result. If `ax` is not specified, create a
        new figure. Returns the figure and axis.
        
        :param ax: (Optional) The axis the plot should be drawn on,
            defaults to None
        :type ax: matplotlib.axes, optional
        :param seas: Whether to draw the seasonal component,
            defaults to True
        :type seas: bool, optional
        :returns: The axis and figure
        :rtype: matplotlib.axes, matlotlib.figure
        """
        if self.timeseries.time64 is not None:
            time = self.timeseries.time64
        else:
            time = self.timeseries.time
        
        if ax is None: fig, ax = plt.subplots()
        
        ax.scatter(time, self.timeseries.data,label='data',color=C4, marker='.')
        
        if seas: ax.plot(time, self.level+np.sum(self.seas,axis=1),color=CF,label='level+seas')
        ax.plot(time, self.level, color=CF, ls='--',label='level')

        ax.text(0.05,0.9,f"{self.name_from_spec()}",transform=ax.transAxes)
        ax.tick_params(axis='y', labelrotation=45)
        ax.legend(loc='lower right')
        return ax.get_figure(), ax
        
        

    def summary(self):
        """Plots a summary of the object
        This is currently not completely implemented
        """
        header = f"Summary for {self.name}"
        border = "#" * len(header)
        print(f"\n{border}\n{header}\n{border}")

        for field_name, field_type in self.__annotations__.items():
            if field_type in [str, float]:
                field_value = getattr(self, field_name)
                print(f"{field_name}: {field_value}")
        
        if self.timeseries.time64 is not None:
            print(f"time range: {self.timeseries.time64[0]} --- {self.timeseries.time64[-1]}")    
        print(f"\n{border}")
        
        
    def name_from_spec(self) -> str:
        """ Creates a unique identifier describing the dlm
        configuration
        
        :returns: Identifier
        :rtype: str
        """
        spec = self.dlm_specification
        out = ''
        if spec['level']: out+='L'
        if spec['stochastic_level']: out+='s'
        if spec['trend']: out+='T'
        if spec['stochastic_trend']: out+='s'
        if spec['freq_seasonal']: 
            for i, s in enumerate(spec['freq_seasonal_periods']):
                out+='_S'
                if spec['stochastic_freq_seasonal'][i]: 
                    out+='s'
                out+='P'+str(spec['freq_seasonal_periods'][i])+'H'+str(spec['freq_seasonal_harmonics'][i])
        if spec['autoregressive']: out+='_A'+str(spec['ar_order'])
        if spec['irregular']: out+='_I'
        return out
    
    @classmethod
    def _name_from_spec(cls, spec) -> str:
        """Creates a unique identifier describing the dlm
        configuration

        :param spec: dictionary describing a dlm configuration
        :type spec: dict
        :returns: Identifier
        :rtype: str
        """
        out = ''
        if spec['level']: out+='L'
        if spec['stochastic_level']: out+='s'
        if spec['trend']: out+='T'
        if spec['stochastic_trend']: out+='s'
        if spec['freq_seasonal']: 
            for i, s in enumerate(spec['freq_seasonal_periods']):
                out+='_S'
                if spec['stochastic_freq_seasonal'][i]: 
                    out+='s'
                out+='P'+str(spec['freq_seasonal_periods'][i])+'H'+str(spec['freq_seasonal_harmonics'][i])
        if spec['autoregressive']: out+='_A'+str(spec['ar_order'])
        if spec['irregular']: out+='_I'
        return out
    
#convert np datatypes to python datatypes
class NpDecoder(json.JSONEncoder):
    """ """
    def default(self, obj):
        """"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj,datetime.datetime):
            return str(obj)
        if isinstance(obj, np.datetime64):
            return np.datetime_as_string(obj)
        return super(NpDecoder, self).default(obj)

#convert python objects to numpy dtypes when reading .json files
def JSONDecoder(dic):
    """"""
    for k in dic.keys():
        obj = dic[k]
        #if isinstance(obj, int):
        #    dic[k] = np.int64(obj)
        #if isinstance(obj, float):
        #    dic[k] = np.float64(obj)
        if isinstance(obj, list):
            dic[k] = np.asarray(obj)
    return dic



@dataclass
class DLMResultList:
    """This class is used for handling multiple dlm results
    created by :func:`dlmhelper.tools.dlm_ensemble`
    
    :params results: List of DLMResult objects
    :type results: List[DLMResult]
    """
    # define fields of the DLMResultList dataclass
    results: List[DLMResult]
    
    def __post_init__(self):
        # create a dictionary that maps each result name to its index in the results list
        self.name_to_index = {result.name_from_spec(): i for i, result in enumerate(self.results)}
    
    def __getitem__(self, name: str) -> DLMResult:
        # allow the DLMResultList to be accessed by result name
        index = self.name_to_index[name]
        return self.results[index]
    
    def save_archive(self, path: str):
        """Save the DLMResultList object as a .tar archive

        :param path: Path to file to save
        :type path: str 

        """
        # save the list of DLMResult as a tar archive of json files
        with tarfile.open(path, mode='w') as tar:
            for result in self.results:
                # create a unique filename for each result based on its name
                result_filename = f"{result.name_from_spec()}.json"
                # write the result to a json file
                with open(result_filename, 'w') as f:
                    json.dump(asdict(result), f,cls=NpDecoder)
                # add the json file to the tar archive
                tar.add(result_filename)
                # delete the temporary json file
                os.remove(result_filename)
                
    @classmethod
    def load_archive(cls, path: str):
        """Load DLMResultList object saved as a .tar archive

        :param path: Path to archive
        :type path: str
        :return: DLMResultList object
        :rtype: DLMResultList
        """
        # load a tar archive into the DLMResultList class
        results = []
        with tarfile.open(path, mode='r') as tar:
            for member in tar.getmembers():
                # read the json file into a dictionary
                with tar.extractfile(member) as f:
                    result_dict = json.load(f,object_hook=JSONDecoder)
                # create a DLMResult object from the dictionary
                result = DLMResult(**result_dict)
                # append the result to the list
                result.timeseries = TimeSeries(**result.timeseries)
                results.append(result)
        return cls(results=results)            
                

    def summary(self,converged=True,sort='aic'):
        """Print a list of all DLMResults

        :param converged: If `True` only show configurations
            for which the fit converged, defaults to `True`
        :type converged: bool, optional
        :param sort: Metric by which the results are sorted,
            defaults to `aic`
        :type sort: str
        """
        table = []
        header = ["Model",*list(self.results[0].dlm_fit_rating.keys())]

        for r in self.results:
            if converged & ~r.dlm_fit_rating['converged']: continue
            line = [r.name_from_spec(), *list(r.dlm_fit_rating.values())]
            table.append(line)
        ix = header.index(sort)
        table.sort(key=lambda l: l[ix])
        print(tabulate(table,headers=header,tablefmt='pipe'))
        
     
    def _dlm_specification_filter(self,result: DLMResult, dicts: List[dict]):
        """"""
        r = result
        if dicts is not None:
            _l= [[np.squeeze(r.dlm_specification[k]==f[k]).item() for k in f.keys()] for f in dicts]
            _ll = [np.all(_x) for _x in _l]
            if np.any(_ll):
                return True
        return False
    
    def _dlm_fit_params_filter(self, result: DLMResult, dicts: List[dict] = [{"sigma2.trend": [1e-9,1]}]):
        """"""
        r = result
        if dicts is not None:
            _l = []
            for f in dicts:
                _l1 = []
                for k in f.keys():
                    try:
                        _l1.append(~(r.dlm_fit_params[k]['param']>=f[k][0]) & (r.dlm_fit_params[k]['param']<=f[k][1]))
                    except Exception as e:
                        pass
                        _l1.append(True)
                _l.append(_l1)
            _ll = [np.all(_x) for _x in _l]
            if np.any(_ll):
                return True
        return False
    
    def get_best_result(self,converged: bool = True, sort: str ='aic', dlm_spec_filter: List[dict] = None, dlm_fit_params_filter: List[dict] = None,n: int = 0):
        """Get the best dlm fit result using the metric given by `sort`

        :param converged: If `True` only show configurations
            for which the fit converged, defaults to `True`
        :type converged: bool, optional
        :param sort: Metric by which the results are sorted,
            defaults to `aic`
        :type sort: str
        :param dlm_spec_filter: TODO
        :type dlm_spec_filter: List[dict], optional
        :param dlm_fit_params_filter: TODO 
        :type dlm_fit_params_filter: List[dict], optional
        :param n: Get the n-th best result, defaults to 0
        :type n: int, optional

        """
        llist=[]
        for r in self.results:
            if converged & ~r.dlm_fit_rating['converged']: 
                continue
            if self._dlm_specification_filter(r, dlm_spec_filter) | self._dlm_fit_params_filter(r, dlm_fit_params_filter):
                continue

            line = [r.name_from_spec(), r.dlm_fit_rating[sort]]
            llist.append(line)
        llist.sort(key=lambda l:l[1])
        if n< len(llist):
            return self.__getitem__(llist[n][0])
        else:
            return self.__getitem__(llist[-1][0])
    
    def plot_summary(self, num: Union[str, int] ='all', converged: bool = True, sort: str ='aic',seas: bool = True,dlm_spec_filter: List[dict] = None, dlm_fit_params_filter: List[dict] = None, figsize=(20,20)):
        """Plot a summary of the dlm results

       
        :param num: Number of dlm results to plot.
            Can be 'all' or integer
        :type num: str | int, optional
        :param converged: If `True` only show configurations
            for which the fit converged, defaults to `True`
        :type converged: bool, optional
        :param sort: Metric by which the results are sorted,
            defaults to `aic`
        :type sort: str, optional
        :param seas: If `True` plot level+seasonal term
        :type seas: bool, optional
        :param dlm_spec_filter: TODO 
        :type dlm_spec_filter: List[dict], optional
        :param dlm_fit_params_filter: TODO
        :type dlm_fit_params_filter: List[dict], optional
        :param figsize:  Figsize to be passed to matplotlib
        :type figsize: tuple, optional
        """
        i=0
        plist=[]
        for r in self.results: 
            if r.dlm_fit_rating['converged'] | (not converged):
                if self._dlm_specification_filter(r, dlm_spec_filter) | self._dlm_fit_params_filter(r, dlm_fit_params_filter):
                    continue
                i+=1
                plist.append(r)
        plist.sort(key=lambda x: x.dlm_fit_rating[sort])
        if num != 'all':
            if (type(num)==int) & (num<i):
                i=num
        if i<4:
            ncols=i
        else:
            ncols=4
        nrows= (i+3)//ncols
        #fig = plt.figure(figsize=figsize)
        fig, axs = plt.subplots(nrows,ncols,figsize=figsize, sharey='all', sharex='all')
        axs = axs.flatten()
        for j in range(0,i):
            #ax = fig.add_subplot(nrows,ncols,j+1,sharey='all')
            plist[j].plot(ax=axs[j],seas=seas)
                
    