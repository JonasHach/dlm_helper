# -*- coding: utf-8 -*-
"""This module provides functions which are used in DLM notebooks

    Version: 0.1.0
    Latest changes: 26.07.2023
    Author: Jonas Hachmeister
"""

from typing import Union, List, Tuple, Optional

import warnings
import datetime
import itertools
import copy

import numpy as np
from numpy.typing import ArrayLike

import statsmodels.api as sm
import statsmodels.tools.sm_exceptions

from dlmhelper.dlm_data import DLMResult, DLMResultList, TimeSeries


def dlm_fit(timeseries: TimeSeries, name: str = None, level: bool = True,
            variable_level: bool = False, trend: bool = True,
            variable_trend: bool = True, seasonal: bool = True, 
            seasonal_period: List[float] = [365], 
            seasonal_harmonics: List[int] = [4], 
            variable_seasonal: List[bool] = [False],
            autoregressive: int = 1, irregular: bool = True
           ):
    """
    Performs a dynamic linear model fit on the given TimeSeries object and
    returns a DLMResult object.
    
    :param timeseries: TimeSeries:
    :param name: str:  (Default value = None)
    :param level: bool:  (Default value = True)
    :param variable_level: bool:  (Default value = False)
    :param trend: bool:  (Default value = True)
    :param variable_trend: bool:  (Default value = True)
    :param seasonal: bool:  (Default value = True)
    :param seasonal_period: List[float]:  (Default value = [365])
    :param seasonal_harmonics: List[int]:  (Default value = [4])
    :param variable_seasonal: List[bool]:  (Default value = [False])
    :param autoregressive: int:  (Default value = 1)
    :param irregular: bool:  (Default value = True)
    :param timeseries: TimeSeries: 
    :param name: str:  (Default value = None)
    :param level: bool:  (Default value = True)
    :param variable_level: bool:  (Default value = False)
    :param trend: bool:  (Default value = True)
    :param variable_trend: bool:  (Default value = True)
    :param seasonal: bool:  (Default value = True)
    :param seasonal_period: List[float]:  (Default value = [365])
    :param seasonal_harmonics: List[int]:  (Default value = [4])
    :param variable_seasonal: List[bool]:  (Default value = [False])
    :param autoregressive: int:  (Default value = 1)
    :param irregular: bool:  (Default value = True)

    """
    
    if seasonal:
        fs = []
        for i in range(len(seasonal_period)):
            fs.append({'period': seasonal_period[i], 
                       'harmonics': seasonal_harmonics[i]})
    else:
        fs = None
    model = sm.tsa.UnobservedComponents(
        timeseries.data,level=level, trend=trend,freq_seasonal=fs,
        autoregressive=autoregressive, stochastic_level=variable_level,
        stochastic_trend=variable_trend, 
        stochastic_freq_seasonal=variable_seasonal, irregular=irregular)
    
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=statsmodels.tools.sm_exceptions.ConvergenceWarning)
        
        result = model.fit(disp=0) 
        
    return DLMResult.create(name,timeseries, result)

def _create_folds(data,n=10):
    """
    
    :param data: param n:  (Default value = 10)
    :param n:  (Default value = 10)

    """
    idxs = np.floor(np.linspace(0, data.size-1,n+1,endpoint=True)
                   ).astype(np.int_)
    out = []
    for i in range(0,n):
        fold = np.copy(data)
        fold[0:idxs[i]] = np.nan
        fold[idxs[i+1]:] = np.nan
        rest = np.copy(data)
        rest[idxs[i]:idxs[i+1]] = np.nan
        out.append((fold,rest))
    return out

def cv_dlm_ensemble(
    timeseries: TimeSeries,
    name: str,
    level: List[bool] = [True],
    variable_level: List[bool] = [False],
    trend: List[bool] = [True],
    variable_trend: List[bool] = [True],
    seasonal: List[bool] = [True],
    seasonal_period: List[List[float]] = [[365]],
    seasonal_harmonics: List[List[List[int]]] = [[[1,2,3,4]]],
    variable_seasonal: List[List[List[bool]]] = [[[True, False]]],
    autoregressive: List[int] = [1],
    irregular: List[bool] = [False, True],
    scores: dict = None,
    folds: int = 5
    ) -> Tuple[DLMResultList, dict]:
    """

    :param timeseries: TimeSeries:
    :param name: str:
    :param level: List[bool]:  (Default value = [True])
    :param variable_level: List[bool]:  (Default value = [False])
    :param trend: List[bool]:  (Default value = [True])
    :param variable_trend: List[bool]:  (Default value = [True])
    :param seasonal: List[bool]:  (Default value = [True])
    :param seasonal_period: List[List[float]]:  (Default value = [[365]])
    :param seasonal_harmonics: List[List[List[int]]]:  (Default value = [[[1)
    :param 2: param 3:
    :param 4: param variable_seasonal: List[List[List[bool]]]:  (Default value = [[[True)
    :param False: param autoregressive: List[int]:  (Default value = [1])
    :param irregular: List[bool]:  (Default value = [False)
    :param True: param scores: dict:  (Default value = None)
    :param folds: int:  (Default value = 5)
    :param timeseries: TimeSeries: 
    :param name: str: 
    :param level: List[bool]:  (Default value = [True])
    :param variable_level: List[bool]:  (Default value = [False])
    :param trend: List[bool]:  (Default value = [True])
    :param variable_trend: List[bool]:  (Default value = [True])
    :param seasonal: List[bool]:  (Default value = [True])
    :param seasonal_period: List[List[float]]:  (Default value = [[365]])
    :param seasonal_harmonics: List[List[List[int]]]:  (Default value = [[[1)
    :param 3: 
    :param 4]]]: 
    :param variable_seasonal: List[List[List[bool]]]:  (Default value = [[[True)
    :param False]]]: 
    :param autoregressive: List[int]:  (Default value = [1])
    :param irregular: List[bool]:  (Default value = [False)
    :param True]: 
    :param scores: dict:  (Default value = None)
    :param folds: int:  (Default value = 5)

    """
    
    timeseries = copy.deepcopy(timeseries)
    
    data = _create_folds(timeseries.data, n = folds)
    
    ensembles = []
    for _fold, _train in data:
        timeseries.data = _train
        rlist = dlm_ensemble(timeseries, name, level, variable_level, trend,
                             variable_trend, seasonal, seasonal_period,
                             seasonal_harmonics, variable_seasonal,
                             autoregressive, irregular)
        ensembles.append(rlist)
    
    _scores = {}
    for i, _rlist in enumerate(ensembles):
        for _r in _rlist.results:
            _fold, _train = data[i]
            _fit = _r.level+_r.ar+np.sum(_r.seas,axis=1)
            _d = (_fit -_fold)
            _mse=(1/_d[~np.isnan(_d)].size)*np.nansum(_d**2)
            _name = _r.name_from_spec()

            if _name in _scores:
                _scores[_name].append(_mse)
            else:
                _scores[_name] = [_mse]
                
    scores = {}
    for key in _scores:
        scores[key] = np.mean(_scores[key])
        
    
    return rlist, scores

def dlm_ensemble(
    timeseries: TimeSeries,
    name: str,
    level: List[bool] = [True],
    variable_level: List[bool] = [False],
    trend: List[bool] = [True],
    variable_trend: List[bool] = [True],
    seasonal: List[bool] = [True],
    seasonal_period: List[List[float]] = [[365]],
    seasonal_harmonics: List[List[List[int]]] = [[[1,2,3,4]]],
    variable_seasonal: List[List[List[bool]]] = [[[True, False]]],
    autoregressive: List[int] = [1],
    irregular: List[bool] = [False, True],
    scores: dict = None
) -> DLMResultList:
    """
    Fits an ensemble of Dynamic Linear Models to a TimeSeries object.
    For all keyword arguments (except scores) a list or nested list is 
    used to determine the configurations used in the ensemble.
    
    For most parameters a boolean List is used. For example
    variable_level = [True, False] would include model configurations
    with and without a variable level in the ensemble. The possible values
    are therefore [True], [False], [True, False].
    
    If seasonal components are included in the ensemble they can be specified
    using nested lists. Each configuration can included multiple seasonal
    components::
    
    
    
    :param timeseries:
    :type timeseries: TimeSeries
    :param name: Identifier for the DLMResult object
    :type name: str
    :param level: 
    :type level: List[bool]
    :param variable_level: seasonal_harmonics (List[int]):
    :param trend: List[bool]:  (Default value = [True])
    :param variable_trend: List[bool]:  (Default value = [True])
    :param seasonal: List[bool]:  (Default value = [True])
    :param seasonal_period: List[List[float]]:  (Default value = [[365]])
    :param seasonal_harmonics: List[List[List[int]]]:  (Default value = [[[1)
    :param variable_seasonal: List[List[List[bool]]]:  (Default value = [[[True)
    :param False]]]: 
    :param autoregressive: List[int]:  (Default value = [1])
    :param irregular: List[bool]:  (Default value = [False)
    :param True]: 
    :param scores: dict:  (Default value = None)
    :returns: DLMResultList: An object containing multiple DLMResult objects

    """
    if len(seasonal_period)!=len(seasonal_harmonics):
        raise ValueError("""seasonal_period and seasonal_harmonics need
                        to have the same length!""")
        
    dicts = []
    dicts2 = []
    for i, periods in enumerate(seasonal_period):
        #print(periods)
        #print(sh[i],sv[i])

        temp1 = []
        for j in range(len(periods)):
            #print(config[j], sh[i][j])

            temp2 = []
            for k in seasonal_harmonics[i][j]:
                #print(config[j], k)
                temp2.append([periods[j],k])
            temp1.append(temp2)
        #print(configs1)
        for c in itertools.product(*temp1,*variable_seasonal[i]):
            #print(c)
            dict_list = []
            dict_list2 = []
            for idx in range(len(c)//2):
                #print(c[idx],c[idx+len(c)//2])
                dict_list.append(
                {'period': c[idx][0],
                'harmonics': c[idx][1]},
                )
                dict_list2.append(
                c[idx+len(c)//2]
                )
            dicts2.append(dict_list2)
            dicts.append(dict_list)
    
    out = []
    if True in seasonal:
        for idx, a, t, st, l, sl, i in itertools.product(
            range(len(dicts)), autoregressive, trend, variable_trend, level, variable_level, irregular):
            sc = dicts[idx]
            ss = dicts2[idx]
            
            model = sm.tsa.UnobservedComponents(
                timeseries.data,level=l, trend=t,
                freq_seasonal=sc, autoregressive=a, 
                stochastic_level=sl, stochastic_trend=st, 
                stochastic_freq_seasonal=ss,irregular=i)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore",
                    category=statsmodels.tools.sm_exceptions.ConvergenceWarning)
                result = model.fit(disp=0)    
            resobj=DLMResult.create(name,timeseries, result,score=scores)
            print(f"Processed: {resobj.name_from_spec()}")
            out.append(resobj)
    if False in seasonal:
        for idx, a, t, st, l, sl, i in itertools.product(
            range(len(dicts)), autoregressive, trend, variable_trend, level, variable_level, irregular):
           
            
            model = sm.tsa.UnobservedComponents(
                timeseries.data,level=l, trend=t,
                freq_seasonal=None, autoregressive=a, 
                stochastic_level=sl, stochastic_trend=st, 
                stochastic_freq_seasonal=None,irregular=i)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore",
                    category=statsmodels.tools.sm_exceptions.ConvergenceWarning)
                result = model.fit(disp=0)    
            resobj=DLMResult.create(name,timeseries, result,score=scores)
            print(f"Processed: {resobj.name_from_spec()}")
            out.append(resobj)

    return DLMResultList(out)












################ OLD
def model_selection_bias_AMI(results: DLMResultList, percentile: int = 25, 
                         years: ArrayLike = None):
    """Calculate the model selection bias for Dynamic Linear Models (DLM) results.
    
    This function computes the model selection bias for AMIs for the given DLMResultList. The bias is calculated by
    computing the weighted variance between the average fit AMI and each individual fit AMI for each year.
    The bias is calculated using all models whose aggregate metric is within the specified percentile.

    :param results: DLMResultList
    :param percentile: int
    :param Defaults: to 25
    :param years: ArrayLike
    :param If: None
    :param results: DLMResultList:
    :param percentile: int:  (Default value = 25)
    :param years: ArrayLike:  (Default value = None)
    :param results: DLMResultList: 
    :param percentile: int:  (Default value = 25)
    :param years: ArrayLike:  (Default value = None)
    :returns: np.ndarray: An array containing the model selection bias for each year specified in the
        'years' array.

    """
    
    if years == None:
        years = np.array([2018,2019,2020,2021,2022])

    agg_list = np.asarray([_r.dlm_fit_rating["agg"] for _r in results.results])
    agg_max = np.percentile(agg_list,percentile,method="nearest")

    ami_list = []
    ami_std_list = []
    for _r in results.results:
        if _r.dlm_fit_rating["agg"]>agg_max: continue

        ami = np.zeros_like(years,dtype=float)
        ami_std = np.zeros_like(years,dtype=float)
        for i, y in enumerate(years):
            ami[i], ami_std[i] = annual_vmr_increase(y, _r)

        ami_list.append(ami)
        ami_std_list.append(ami_std)

    ami_list = np.array(ami_list)
    ami_std_list = np.array(ami_std_list)

    ami = np.zeros_like(years,dtype=float)
    ami_std = np.zeros_like(years,dtype=float)
    for i, y in enumerate(years):
            ami[i], ami_std[i] = annual_vmr_increase(y, results.get_best_result(sort="agg"))
            
    ami_avg = np.average(ami_list, axis=0)
    ami_std_avg = np.std(ami_list, axis=0)
    return np.sqrt(np.average((ami_avg-ami_list)**2,weights=1/np.sqrt(ami_std_avg**2+ami_std_list**2),axis=0))


def model_selection_bias_trend(results: DLMResultList, percentile: int = 25):
    """Calculate the model selection bias for Dynamic Linear Models (DLM) results.
    
    This function computes the model selection bias for growth rates for the given DLMResultsList. The bias is calculated by
    computing the weighted variance between the average fit trend (growth rate) and each individual fit trend.
    The bias is calculated using all models whose aggregate metric is within the specified percentile.

    :param results: DLMResultList
    :param percentile: int
    :param Defaults: to 25
    :param results: DLMResultList:
    :param percentile: int:  (Default value = 25)
    :param results: DLMResultList: 
    :param percentile: int:  (Default value = 25)
    :returns: float: model selection bias

    """

    agg_list = np.asarray([_r.dlm_fit_rating["agg"] for _r in results.results])
    agg_max = np.percentile(agg_list,percentile,method="nearest")

    trend_list = []
    trend_cov_list = []
    for _r in results.results:
        if _r.dlm_fit_rating["agg"]>agg_max: continue


        trend_list.append(_r.trend)
        trend_cov_list.append(_r.trend_cov)

    trend_list = np.array(trend_list)
    trend_cov_list = np.array(trend_cov_list)

            
    trend_avg = np.average(trend_list, axis=0)
    trend_cov_avg = np.std(trend_list, axis=0)
    
    _r = results.get_best_result(sort="agg")
    
    return np.sqrt(np.average((trend_avg-_r.trend)**2,weights=1/np.sqrt(trend_cov_avg**2+_r.trend_cov**2),axis=0))

def mean_from_date(d1: int ,m1: int ,y1: int ,d2: int ,m2: int ,y2: int , X: ArrayLike ,d: ArrayLike ) -> float:
    """Calculate the mean of the values in X that fall within a given date range.

    :param d1: int
    :param m1: int
    :param y1: int
    :param d2: int
    :param m2: int
    :param y2: int
    :param X: ArrayLike
    :param d: ArrayLike
    :param corresponding: to the values in X
    :param d1: int:
    :param m1: int:
    :param y1: int:
    :param d2: int:
    :param m2: int:
    :param y2: int:
    :param X: ArrayLike:
    :param d: ArrayLike:
    :param d1: int: 
    :param m1: int: 
    :param y1: int: 
    :param d2: int: 
    :param m2: int: 
    :param y2: int: 
    :param X: ArrayLike: 
    :param d: ArrayLike: 
    :returns: float: Mean of the values in X that fall within the specified date range.

    """
    
    date_min=(datetime.datetime(y1,m1,d1) - datetime.datetime(1970,1,1)).days
    date_max=(datetime.datetime(y2,m2,d2) - datetime.datetime(1970,1,1)).days
    
    return np.nanmean(X[(d>=date_min) & (d<=date_max)])

def vmr_increase(d1: int, m1: int, y1: int, d2: int, m2: int, y2: int, L: ArrayLike, d: ArrayLike) -> float:
    """Calculate the increase in level between two days.

    :param d1: int
    :param m1: int
    :param y1: int
    :param d2: int
    :param m2: int
    :param y2: int
    :param L: ArrayLike
    :param d: ArrayLike
    :param d1: int:
    :param m1: int:
    :param y1: int:
    :param d2: int:
    :param m2: int:
    :param y2: int:
    :param L: ArrayLike:
    :param d: ArrayLike:
    :param d1: int: 
    :param m1: int: 
    :param y1: int: 
    :param d2: int: 
    :param m2: int: 
    :param y2: int: 
    :param L: ArrayLike: 
    :param d: ArrayLike: 
    :returns: float: Increase in level between the start and end dates.

    """
    L = np.asarray(L)
    d = np.asarray(d)
    
    date_min=(datetime.datetime(y1,m1,d1)-datetime.datetime(1970,1,1)).days
    date_max=(datetime.datetime(y2,m2,d2)-datetime.datetime(1970,1,1)).days
    
    
    try:
        out = (L[d==date_max] - L[d==date_min])[0]
    except Exception as e:
        print(e)
        out = 0
        
        
    return out


def vmr_std_increase(d1: int ,m1: int ,y1: int ,d2: int ,m2: int ,y2: int
                     , cL: ArrayLike, d: ArrayLike ) -> float:
    """"
    Calculate the standard deviation to the increase in level between two days.

    :param d1: int
    :param m1: int
    :param y1: int
    :param d2: int
    :param m2: int
    :param y2: int
    :param cL: ArrayLike
    :param d: ArrayLike
    :param d1: int:
    :param m1: int:
    :param y1: int:
    :param d2: int:
    :param m2: int:
    :param y2: int:
    :param cL: ArrayLike:
    :param d: ArrayLike:
    :param d1: int: 
    :param m1: int: 
    :param y1: int: 
    :param d2: int: 
    :param m2: int: 
    :param y2: int: 
    :param cL: ArrayLike: 
    :param d: ArrayLike: 
    :returns: float: Standard deviation corresponding to increase in level between the start and end dates.

    """
    cL = np.asarray(cL)
    d = np.asarray(d)
    
    date_min=(datetime.datetime(y1,m1,d1)-datetime.datetime(1970,1,1)).days
    date_max=(datetime.datetime(y2,m2,d2)-datetime.datetime(1970,1,1)).days
    
    try:
        out = (np.sqrt(cL[d==date_max] + cL[d==date_min]))[0] 
    except Exception as e:
        print(e)
        out=0
        
    return out  


def _get_idx_at_time(times,time, tolerance=np.timedelta64(1,'D')):
    delta = np.min(np.abs(times-time))
    idx = np.argmin(np.abs(times-time))
    if delta>tolerance:
        return None
    else:
        return idx
    
def annual_vmr_increase(year: int, data: DLMResult) -> Tuple[float, float]:
    """Calculate annual increase  and standard deviation in volume mixing ratio
        for a given DLMResult object.

    :param year: int
    :param data: DLMResult

    :returns: Tuple[float, float]: a tuple containing the annual increase and
        standard deviation.

    """
    inc = -999
    inc_std = -999
    
    t1 = np.datetime64(f"{year}-01-01")
    t2 = np.datetime64(f"{year}-12-31")
    
    idx1 = _get_idx_at_time(data.timeseries._time64, t1)
    idx2 = _get_idx_at_time(data.timeseries._time64, t2)
    
    if idx1 is not None and idx2 is not None:
        inc = data.level[idx2] - data.level[idx1]
        inc_std = np.sqrt(data.level_cov[idx2]+data.level_cov[idx1])
        return inc, inc_std
    else:
        #Todo raise error
        return None, None
    


def get_monthly_vmr(vmr: ArrayLike, date_min: Union[int, datetime.datetime], 
                    date_max: Union[int, datetime.datetime], 
                    year_range: tuple = (2018, 2023)) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate monthly volume mixing ration (vmr) from daily vmr data for a given time range

    :param vmr: array_like
    :param date_min: int
    :param or: datetime object for which to calculate vmr
    :param date_max: int
    :param or: datetime object for which to calculate vmr
    :param year_range: Tuple
    :param of: the range of years to calculate vmr for
    :param vmr: ArrayLike:
    :param date_min: Union[int:
    :param datetime: datetime]:
    :param date_max: Union[int:
    :param year_range: tuple:  (Default value = (2018)
    :param 2023: returns: Tuple[numpy.ndarray, numpy.ndarray]: a tuple containing arrays of dates and vmr values.
    :param vmr: ArrayLike: 
    :param date_min: Union[int: 
    :param datetime.datetime]: 
    :param date_max: Union[int: 
    :param year_range: tuple:  (Default value = (2018)
    :param 2023): 
    :returns: Tuple[numpy.ndarray, numpy.ndarray]: a tuple containing arrays of dates and vmr values.

    """
    
    vmr = np.asarray(vmr)
    if type(date_min)==datetime.datetime:
        date_min = (date_min - datetime.datetime(1970, 1, 1)).days
    if type(date_max)==datetime.datetime:
        date_max = (date_max - datetime.datetime(1970, 1, 1)).days
    
    vmr_month = []
    d = date_min + np.arange(0, vmr.shape[0])
    dm = []
    i=0
    for y in range(year_range[0], year_range[1] + 1):
        for m in range(1,13):
            dmin = (datetime.datetime(y, m, 1) - datetime.datetime(1970, 1, 1)).days
            if dmin < date_min:
                continue
            if dmin > date_max:
                break
            dm.append(dmin)
            if m==12: 
                dmax = (datetime.datetime(y + 1, 1, 1) - datetime.datetime(1970, 1, 1)).days
            else:
                dmax = (datetime.datetime(y, m + 1, 1)-datetime.datetime(1970, 1, 1)).days
    
            vmr_month.append(np.nanmean(vmr[(d >= dmin) & (d <= dmax)]))
            i += 1
            
    return np.array(dm), np.array(vmr_month)











