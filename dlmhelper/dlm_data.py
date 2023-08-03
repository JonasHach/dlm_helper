# -*- coding: utf-8 -*-

"""dlm_data.py

This module provides two classes which are used to handle data from DLM results.

- DLMResult: Handles single results
- DLMResultList: Acts as a container for all DLMResults generated for a given time series

Version: 0.1.0
Latest changes: 26.07.2023
Author: Jonas Hachmeister
"""

import tarfile
import os
import json
import datetime as dt


from dataclasses import dataclass, fields, asdict, field
from typing import Union, List, Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
from tabulate import tabulate

import dlmhelper.spatio_temporal

from statsmodels.tsa.statespace.structural import UnobservedComponentsResults

MILLISECOND_ALIASES = ['millisecond', 'milliseconds', 'ms']
SECOND_ALIASES = ['second', 'seconds', 'sec', 'secs', 's']
MINUTE_ALIASES = ['minute','minutes','min','mins']
HOUR_ALIASES = ['hour', 'hours', 'h', 'H']
DAY_ALIASES = ['day','days','d','D']
WEEK_ALIASES =  ['week', 'weeks', 'w', 'W']
MONTH_ALIASES = ['month', 'months', 'm', 'M']
YEAR_ALIASES = ['year', 'years', 'yr', 'yrs', 'y', 'Y']

def _to_datetime64(time: ArrayLike, time_unit: str, reference_time: Union[np.datetime64,dt.datetime]):
    """
    Convert an array of time values given as fractionl values with unit [time_unit] to datetime64 values
    using the given reference_time
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

def grid_dim(lat_min: float, lat_max: float, lon_min: float, lon_max: float, 
             lat_step: float, lon_step: float) -> dict:
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
    if len(g.keys()) != 6:
        return False
    
    keys = grid_dim(0,0,0,0,0,0).keys()
    
    for k in keys:
        if k not in g.keys():
            return False
        
    return True

    
@dataclass
class TimeSeries:
    data: np.ndarray
    time: np.ndarray
    time_unit: str
    
    reference_time: np.datetime64 = None
    error: np.ndarray = None
    N: np.ndarray = None
    product_type: str = None
    grid_dim: dict = None
    
    _time64: np.array = None
    
    def __post_init__(self):
        
        if self.data is None or self.time is None or self.time_unit is None:
            raise ValueError("The data, time and time_unit field must be provided during initialization")
            
        self.data = np.asarray(self.data)
        self.time = np.asarray(self.time)
        
        if self.error is None:
            self.error = np.full(self.data.shape, np.nan)
        else:
            self.error = np.asarray(self.error)
        
        if self.N is None:
            self.N = np.full(self.data.shape, np.nan)
        else:
            self.N = np.asarray(self.N)
            
        if len(self.data.shape)>1 or len(self.error.shape)>1 or len(self.time.shape)>1 or len(self.N.shape)>1:
            raise ValueError("data, error, N and time field need to be one-dimensional!")
        
        if self.data.shape != self.error.shape or self.data.shape != self.time.shape or self.data.shape != self.N.shape:
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
        
        if self.reference_time is not None:
            self.reference_time = np.datetime64(self.reference_time)
            self._time64 = _to_datetime64(self.time, self.time_unit, self.reference_time)
        else:
            self._time64 = None
            
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for field_name, field_type in self.__annotations__.items():
            if field_type == np.ndarray:
                if not np.array_equal(getattr(self, field_name), getattr(other, field_name),equal_nan=True):
                    return False
            elif field_name == "_time64":
                if not np.all(np.equal(getattr(self, field_name), getattr(other, field_name))):
                    return False
            else:    
                if getattr(self, field_name) != getattr(other, field_name):
                    return False
        return True
    
    @classmethod
    def load(cls,path):
        with open(path,'r') as f:
            s = f.read()
            argDict = json.loads(s, object_hook=JSONDecoder)
            fieldSet = {f.name for f in fields(cls) if f.init}
            filteredArgDict = {k : v for k, v in argDict.items() if k in fieldSet}
            
            filteredArgDict["reference_time"] = np.datetime64(filteredArgDict["reference_time"])
            
            return cls(**filteredArgDict)
            
    def save(self,path,fname=None):
        if not fname:
            fname = f"{self.name}_{self.name_from_spec()}.json"
        s=json.dumps(asdict(self),cls=NpDecoder, indent=2)
        with open(path+fname, 'w') as f:
            f.write(s)
            print("Saved data at:",path+fname)

@dataclass
class Grid:
    data: np.ndarray
    error: np.ndarray
    
    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray
    time_unit: str
    
    reference_time: Union[np.datetime64, dt.datetime] = None
    grid_dim: dict = None
    N: ArrayLike = None
    product_type: str = None
    
    
    def __post_init__(self):
        
        if self.data is None or self.error is None or self.lat is None or self.lon is None or self.time is None or self.time_unit is None:
            raise ValueError("All fields (except product_type) must be provided during initialization")
        
        self.data = np.asarray(self.data)
        self.error = np.asarray(self.error)
        self.lat = np.asarray(self.lat)
        self.lon = np.asarray(self.lon)
        self.time = np.asarray(self.time)
        
        if self.reference_time is not None:
            self.reference_time = np.datetime64(self.reference_time)
            
        if self.N is not None:
            self.N = np.asarray(self.N)
        
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
        
        if self.error.shape != (self.lat.shape[0], self.lon.shape[0], self.time.shape[0]):
            raise ValueError(f"error array shape must be ({self.lat.shape[0]}, {self.lon.shape[0]}, {self.time.shape[0]}) but is {self.error.shape}")
            
        if self.N is not None:
            if self.N.shape != (self.lat.shape[0], self.lon.shape[0], self.time.shape[0]):
                raise ValueError(f"N array shape must be ({self.lat.shape[0]}, {self.lon.shape[0]}, {self.time.shape[0]}) but is {self.N.shape}")
            
            
        #TODO: add checks for dict composition
        
        #Sort gridded data and add missing days (filled with NaNs)    
        _t = self.time
        _d = self.data
        _e = self.error
        _N = self.N
    
        start = np.nanmin(_t)
        end = np.nanmax(_t)
        size = int(end-start+1)

        data = np.full((_d.shape[0],_d.shape[1],size), np.nan)
        error = np.full((_d.shape[0],_d.shape[1],size), np.nan)
        if _N is not None: N = np.zeros((_d.shape[0],_d.shape[1],size))

        time = np.zeros((size))

        for i in range(size):
            time[i]=np.nanmin(_t)+i
            if (_d[:,:,_t==start+i].shape[2]>0):
                data[:,:,i] = _d[:,:,_t==start+i][:,:,0]
                error[:,:,i] = _e[:,:,_t==start+i][:,:,0]
                if _N is not None: N[:,:,i] =_N[:,:,_t==start+i][:,:,0]

            else:
                data[:,:,i] = np.nan
                error[:,:,i] = np.nan
                if _N is not None: N[:,:,i] = 0
        
        self.time = time
        self.data = data
        self.error = error
        if _N is not None: self.N = N
        
    def inhomogeneity_spatial(self, scale_lat: float = None, scale_lon: float = None):
        if self.N is None:
            raise ValueError("Spatial inhomogeneity can only be calculated if number of data points per grid cell (N) is available!")
        return dlmhelper.spatio_temporal.inhomogeneity_spatial(self.lat, self.lon, self.N, scale_lat = scale_lat, scale_lon = scale_lon)

    def inhomogeneity_temporal(self):
        if self.N is None:
            raise ValueError("Temporal inhomogeneity can only be calculated if number of data points per grid cell (N) is available!")
        return dlmhelper.spatio_temporal.inhomogeneity_temporal(self.lat, self.lon, self.time, self.N)
    
    def filter_inhomogeneity_spatial(self, hs_lim: float = None, scale_lat: float = None, scale_lon: float = None):
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
        if self.N is None:
            raise ValueError("Temporal inhomogeneity can only be calculated if number of data points per grid cell (N) is available!")
        ht = dlmhelper.spatio_temporal.inhomogeneity_temporal(self.lat, self.lon, self.time, self.N)
        
        self.data[ht[...,0]>ht_lim,:]=np.nan
        self.error[ht[...,0]>ht_lim,:]=np.nan
        if self.N is not None:
            self.N[ht[...,0]>ht_lim,:]=0
            
    def to_timeseries(self, zonal_avg: bool = False):
        avg_data, avg_error = dlmhelper.spatio_temporal._area_weighted_average(self.data, self.error, self.lat, self.lon, zonal_avg = zonal_avg)
        if self.N is not None: 
            avg_N = np.zeros(self.time.size)
            avg_N = np.nansum(self.N, axis=(0,1),out=avg_N, where=(~np.isnan(np.nanmean(self.N, axis=(0,1)))))
        else:
            avg_N = None
    
        return TimeSeries(avg_data, self.time, self.time_unit, error =  avg_error, N = avg_N, product_type = self.product_type, reference_time = self.reference_time, grid_dim = self.grid_dim)
        
        

@dataclass
class DLMResult:
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
            
        
        resid = res.resid
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
        with open(path,'r') as f:
            s = f.read()
            argDict = json.loads(s, object_hook=JSONDecoder)
            fieldSet = {f.name for f in fields(cls) if f.init}
            filteredArgDict = {k : v for k, v in argDict.items() if k in fieldSet}
            
            filteredArgDict["timeseries"]= TimeSeries(**filteredArgDict["timeseries"])
            
            return cls(**filteredArgDict)
            
    def save(self,path,fname=None):
        if not fname:
            fname = f"{self.name}_{self.name_from_spec()}.json"
        s=json.dumps(asdict(self),cls=NpDecoder, indent=2)
        with open(path+fname, 'w') as f:
            f.write(s)
            print("Saved data at:",path+fname)
    
    def plot(self, figsize=(15,15)):
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(221)
        
        if self.timeseries._time64 is not None:
            time = self.timeseries._time64
        else:
            time = self.timeseries.time
            
        ax1.scatter(time, self.timeseries.data, marker='.', color='darkgray',label='input data')

        ax1.plot(time, self.level+np.sum(self.seas,axis=1)+self.ar,color='red',label='DLM fit')
        ax1.legend(); ax1.xaxis_date(); ax1.set_ylabel("XCH$_4$ [ppb]")
        ax2 = fig.add_subplot(222,sharey=ax1)
        ax2.plot(time, self.level, color='red', label='DLM: level')
        ax2.plot(time, self.level+np.sum(self.seas,axis=1), color='red', ls='--', label='DLM: level+seas')
        ax2.legend(); ax2.xaxis_date(); ax2.set_ylabel("XCH$_4$ [ppb]")
        ax3 = fig.add_subplot(223)
        ax3.plot(time, np.sum(self.seas,axis=1), color='red', label='DLM: seas')
        ax3.legend(); ax3.xaxis_date(); ax3.set_ylabel("XCH$_4$ [ppb]")
        ax4=fig.add_subplot(224)
        ax4.plot(time, self.ar, color='red', label='DLM: AR')
        ax4.legend(); ax4.xaxis_date(); ax4.set_ylabel("XCH$_4$ [ppb]")
        return fig
    
    def plot_summary(self, ax=None,ar=False, **kwargs):
        if self.timeseries._time64 is not None:
            time = self.timeseries._time64
        else:
            time = self.timeseries.time
        if ax is None: fig, ax = plt.subplots()
        
        ax.scatter(time, self.timeseries.data,label='data',color='gray',edgecolor='black',**kwargs)
        if not ar: ax.plot(time, self.level+np.sum(self.seas,axis=1),color='darkorange',label='level+seas')
        if ar: ax.plot(time, self.level+np.sum(self.seas,axis=1)+self.ar,color='darkorange',label='level+seas+ar')
        ax.plot(time, self.level, color='red', ls='--',label='level')
        ax.xaxis_date()
        ax.text(0.01,0.9,f"{self.name_from_spec()}",transform=ax.transAxes,backgroundcolor='wheat', alpha=0.8)
        ax.tick_params(axis='y', labelrotation=45)
        ax.legend(loc='lower right',fontsize='small')

    def summary(self):
        header = f"Summary for {self.name}"
        border = "#" * len(header)
        print(f"\n{border}\n{header}\n{border}")

        for field_name, field_type in self.__annotations__.items():
            if field_type in [str, float]:
                field_value = getattr(self, field_name)
                print(f"{field_name}: {field_value}")
        
        if self.timeseries._time64 is not None:
            print(f"time range: {self.timeseries._time64[0]} --- {self.timeseries._time64[-1]}")    
        print(f"\n{border}")
        
        
    def name_from_spec(self) -> str:
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
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj,dt.datetime):
            return str(obj)
        if isinstance(obj, np.datetime64):
            return np.datetime_as_string(obj)
        return super(NpDecoder, self).default(obj)

#convert python objects to numpy dtypes when reading .json files
def JSONDecoder(dic):
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
    # define fields of the DLMResultList dataclass
    results: List[DLMResult]
    
    def __post_init__(self):
        # create a dictionary that maps each result name to its index in the results list
        self.name_to_index = {result.name_from_spec(): i for i, result in enumerate(self.results)}
    
    def __getitem__(self, name: str) -> DLMResult:
        # allow the DLMResultList to be accessed by result name
        index = self.name_to_index[name]
        return self.results[index]
    
    
    def save_archive(self, filename: str):
        # save the list of DLMResult as a tar archive of json files
        with tarfile.open(filename, mode='w') as tar:
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
                

    def summary(self,converged=True,sort='aic'):
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
        r = result
        if dicts is not None:
            _l= [[np.squeeze(r.dlm_specification[k]==f[k]).item() for k in f.keys()] for f in dicts]
            _ll = [np.all(_x) for _x in _l]
            if np.any(_ll):
                return True
        return False
    
    def _dlm_fit_params_filter(self, result: DLMResult, dicts: List[dict] = [{"sigma2.trend": [1e-9,1]}]):
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
    
    def get_best_result(self,converged=True, sort='aic', dlm_spec_filter: List[dict] = None, dlm_fit_params_filter: List[dict] = None,n=0):
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
    
    def plot_summary(self,figsize=(20,20),num='all',sort='aic',converged=True,ar=False,dlm_spec_filter: List[dict] = None, dlm_fit_params_filter: List[dict] = None , **kwargs):
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
            plist[j].plot_summary(ax=axs[j],ar=ar,**kwargs)
                
    @classmethod
    def load_archive(cls, filename: str) -> 'DLMResultList':
        # load a tar archive into the DLMResultList class
        results = []
        with tarfile.open(filename, mode='r') as tar:
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