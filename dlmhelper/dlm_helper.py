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
    
    data = _create_folds(timeseries.data, n = folds)
    
    ensembles = []
    for _fold, _train in data:
        rlist = dlm_ensemble(timeseries, name, level, variable_level, trend,
                             variable_trend, seasonal, seasonal_period,
                             seasonal_harmonics, variable_seasonal,
                             autoregressive, irregular)
        ensembles.append(rlist)
    
    _scores = {}
    for i, _rlist in enumerate(_ensembles):
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
    Fits an ensemble of Dynamic Linear Models (DLMs) to a time series.

    :param level: level (List)
    :param variable_level: seasonal_harmonics (List[int]): 

    :param scores: (dict[float], optional) a dictionary in form 
    {name_string: score}, where name_string is is given by name_from_spec

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
            print(f"Fitting: {sc}, {ss}, {a}, {t}, {st}, {l}, {sl}, {i}")
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
            out.append(resobj)
    if False in seasonal:
        pass

    return DLMResultList(out)












################ OLD
def model_selection_bias_AMI(results: DLMResultList, percentile: int = 25, 
                         years: ArrayLike = None):
    """
    Calculate the model selection bias for Dynamic Linear Models (DLM) results.

    This function computes the model selection bias for AMIs for the given DLMResultList. The bias is calculated by
    computing the weighted variance between the average fit AMI and each individual fit AMI for each year. 
    The bias is calculated using all models whose aggregate metric is within the specified percentile. 

    Parameters:
        results (DLMResultList): A DLMResultList object containing a list of DLM results.
        percentile (int, optional): The percentile value used to select data points for computing the model selection bias.
                                   Defaults to 25.
        years (ArrayLike, optional): An array-like object containing the years for which to calculate the model selection bias.
                                     If None, the years will be set to [2018, 2019, 2020, 2021, 2022]. Defaults to None.

    Returns:
        np.ndarray: An array containing the model selection bias for each year specified in the 
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
    """
    Calculate the model selection bias for Dynamic Linear Models (DLM) results.

    This function computes the model selection bias for growth rates for the given DLMResultsList. The bias is calculated by
    computing the weighted variance between the average fit trend (growth rate) and each individual fit trend. 
    The bias is calculated using all models whose aggregate metric is within the specified percentile. 

    Parameters:
        results (DLMResultList): A DLMResultList object containing a list of DLM results.
        percentile (int, optional): The percentile value used to select data points for computing the model selection bias.
                                   Defaults to 25.

    Returns:
        float: model selection bias

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
    """
    Calculate the mean of the values in X that fall within a given date range.

    Parameters:
        d1 (int): Day of the month for the start date (1-31).
        m1 (int): Month of the start date (1-12).
        y1 (int): Year of the start date.
        d2 (int): Day of the month for the end date (1-31).
        m2 (int): Month of the end date (1-12).
        y2 (int): Year of the end date.
        X (ArrayLike): Array of values to calculate the mean from.
        d (ArrayLike): Array of date values (in days since 1970-01-01) 
                    corresponding to the values in X.

    Returns:
        float: Mean of the values in X that fall within the specified date range.

    """
    
    date_min=(datetime.datetime(y1,m1,d1) - datetime.datetime(1970,1,1)).days
    date_max=(datetime.datetime(y2,m2,d2) - datetime.datetime(1970,1,1)).days
    
    return np.nanmean(X[(d>=date_min) & (d<=date_max)])

def vmr_increase(d1: int, m1: int, y1: int, d2: int, m2: int, y2: int, L: ArrayLike, d: ArrayLike) -> float:
    """
    Calculate the increase in level between two days.

    Parameters:
        d1 (int): Day of the month for the start date (1-31).
        m1 (int): Month of the start date (1-12).
        y1 (int): Year of the start date.
        d2 (int): Day of the month for the end date (1-31).
        m2 (int): Month of the end date (1-12).
        y2 (int): Year of the end date.
        L (ArrayLike): Array of level values corresponding to the dates in 'd'.
        d (ArrayLike): Array of date values (in days since 1970-01-01) corresponding to the values in 'L'.

    Returns:
        float: Increase in level between the start and end dates.

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

     Parameters:
        d1 (int): Day of the month for the start date (1-31).
        m1 (int): Month of the start date (1-12).
        y1 (int): Year of the start date.
        d2 (int): Day of the month for the end date (1-31).
        m2 (int): Month of the end date (1-12).
        y2 (int): Year of the end date.
        cL (ArrayLike): Array of level covariance values corresponding to the dates in 'd'.
        d (ArrayLike): Array of date values (in days since 1970-01-01) corresponding to the values in 'L'.

    Returns:
        float: Standard deviation corresponding to increase in level between the start and end dates.

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


def annual_vmr_increase(year: int, data: DLMResult) -> Tuple[float, float]:
    """Calculate annual increase  and standard deviation in volume mixing ratio
        for a given DLMResult object.

    Parameters:
        year (int): the year for which the increase is calculated.
        data (DLMResult): a DLMResult object gained from read_dlm_results.

    Returns:
        Tuple[float, float]: a tuple containing the annual increase and 
        standard deviation.
    """
    inc = -999
    inc_std = -999
    
    if data.time_unit == "day":
       
        date_min = (datetime.datetime(year, 1, 1) - datetime.datetime(1970, 1, 1)).days
        date_max = (datetime.datetime(year, 12, 31) - datetime.datetime(1970, 1, 1)).days
        try:
            t = data.time
            l = data.level
            lc = data.level_cov
            a = data.ar
            ac = data.ar_cov
            
            #Sometimes the covariances are below zero, which is probably an issue with the machine precision and
            #covariances beeing close to zero. Thus we set values below zero to zero.
            lc[lc<0] = 0
            ac[ac<0] = 0
      
            inc = (l[t == date_max] - l[t == date_min])[0]
            inc_std = np.sqrt(lc[t == date_max] + lc[t == date_min])[0]
        except Exception as e:
            print(e)
    elif data.time_unit == "month":
        date_min = (datetime.datetime(year, 1, 1) - datetime.datetime(1970, 1, 1)).days
        date_max = (datetime.datetime(year + 1, 1, 1) - datetime.datetime(1970, 1, 1)).days
        try:
            t = data.time
            l = data.level
            lc = data.level_cov
            a = data.ar
            ac = data.ar_cov
            
            lc[lc<0] = 0
            ac[ac<0] = 0
            
            inc = (l[t == date_max] - l[t == date_min])[0]
            inc_std = np.sqrt(lc[t == date_max] + lc[t == date_min])[0]
        except Exception as e:
            print(e)
    return inc, inc_std


def get_monthly_vmr(vmr: ArrayLike, date_min: Union[int, datetime.datetime], 
                    date_max: Union[int, datetime.datetime], 
                    year_range: tuple = (2018, 2023)) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate monthly volume mixing ration (vmr) from daily vmr data for a given time range

    Parameters:
        vmr (array_like): an array of vmr values.
        date_min (int, datetime.datetime): the minimum date in days since 01.01.1970
            or datetime object for which to calculate vmr.
        date_max (int, datetime.datetime): the maximum date in days since 01.01.1970 
            or datetime object for which to calculate vmr.
        year_range (Tuple[int], optional): a list containing the start and end years
            of the range of years to calculate vmr for. Defaults to (2018, 2022).

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: a tuple containing arrays of dates and vmr values.
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











