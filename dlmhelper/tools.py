# -*- coding: utf-8 -*-
"""This module provides functions to perform dynamic linear model
fits and for the evaluation of the results.

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

from dlmhelper.data import DLMResult, DLMResultList, TimeSeries


def dlm_fit(timeseries: TimeSeries, name: str, level: bool = True,
            variable_level: bool = False, trend: bool = True,
            variable_trend: bool = True, seasonal: bool = True, 
            seasonal_period: List[float] = [365], 
            seasonal_harmonics: List[int] = [4], 
            variable_seasonal: List[bool] = [False],
            autoregressive: int = 1, irregular: bool = True,
            verbose: int = 0
           ) -> DLMResult:
    """
    Performs a dynamic linear model fit on the given TimeSeries object and
    returns a DLMResult object.

    :param timeseries: TimeSeries object do be fitted
    :type timeseries: TimeSeries
    :param name: Identifier for the DLMResult object
    :type name: str
    :param level: Whether to include a level component,
        defaults to True
    :type level: bool
    :param variable_level: Whether to allow the level component
        to vary, defaults to False
    :type variable_level: bool
    :param trend: Whether to include a trend (i.e. changing level),
        defaults to True
    :type trend: bool
    :param variable_trend: Whether to allow the trend component
        to vary, defaults to True
    :type variable_trend: bool
    :param seasonal: Whether to include seasonal components,
        defaults to True
    :type seasonal: bool
    :param seasonal_period: List of periods for the seasonal
        components, defaults to [365]
    :type seasonal_period: List[bool]
    :param seasonal_harmonics: Number of harmonics to use for the
        seasonal components, defaults to [4]
    :type seasonal_harmonics: List[int]
    :param variable_seasonal: Whether the seasonal componets are
        allowed to vary, defaults to [False]
    :type variable_seasonal: List[bool]
    :param autoregressive: Determines the order of the
        autoregressive component, use `None` to not include,
        defaults to 1
    :type autoregressive: int | None
    :param irregular: Whether to a Gaussian noise term,
        defaults to True
    :type irregular: bool
    :param verbose: Determines the amount of outpout, 0 means no output
        and 2 means maximum outout, defaults to 0
    :type verbose: int
    :returns: A DLMResult object
    :rtype: DLMResult
    """
    
    if seasonal:
        fs = []
        for i in range(len(seasonal_period)):
            fs.append({'period': seasonal_period[i], 
                       'harmonics': seasonal_harmonics[i]})
    else:
        fs = None
        variable_seasonal = None
        
    model = sm.tsa.UnobservedComponents(
        timeseries.data,level=level, trend=trend,freq_seasonal=fs,
        autoregressive=autoregressive, stochastic_level=variable_level,
        stochastic_trend=variable_trend, 
        stochastic_freq_seasonal=variable_seasonal, irregular=irregular)
    
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=statsmodels.tools.sm_exceptions.ConvergenceWarning)
        if verbose<2:
            result = model.fit(disp=0) 
        else:
            result = model.fit()
            
    return DLMResult.create(name,timeseries, result)

def _create_folds(data,n=10):
    """

    :param data: param n:  (Default value = 10)
    :param n: Default value = 10)

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
    folds: int = 5,
    verbose: int = 0
    ) -> dict:
    """
    Performs cross validation using the specified number of `folds` and
    calculates the average mean squated error (AMSE) for all configurations.
    
    Returns a dictionary with keys corresponding to the model configurations
    gained by :py:func:`dlmhelper.dlm_data.DLMResult.name_from_spec` and the
    AMSE as values.
    
    See :py:func:`dlmhelper.dlm_helper.dlm_ensemble` for information
    on the parameters.

    :param timeseries: TimeSeries object do be fitted
    :type timeseries: TimeSeries
    :param name: Identifier for the DLMResult object
    :type name: str
    :param level: Whether to include a level component, defaults to [True]
    :type level: List[bool]
    :param variable_level:  Wheter to allow the level component to vary,
        defaults to [False]
    :type variable_level: List[bool]
    :param trend: Whether to include a trend component, defaults to [True]
    :type trend: List[bool]
    :param variable_trend: Whether to allow the trend component to vary,
        defaults to [True]
    :type variable_trend: List[bool]
    :param seasonal: Whether to include a seasonal component,
        defaults to [True]
    :type seasonal: List[bool]
    :param seasonal_period: List of configurations of seasonal components.
        Each element is a list containing the periods of the seasonal 
        components, defaults to [[365]]
    :type seasonal_period: List[List[float]]
    :param seasonal_harmonics: List harmonics to try for the corresponding
        seasonal components. For each element of seasonal_period this 
        should include a list of harmonics to try, defaults to [[[1,2,3,4]]]
    :type seasonal_harmonics: List[List[List[int]]]
    :param variable_seasonal: Whether a seasonal component is allowed to vary.
        For each element of seasonal_period this should include a list of 
        options, defaults to [[[True, False]]]
    :type variable_seasonal: List[List[List[bool]]]
    :param autoregressive: List of autoregressive components to try, 
        the integer determines the order of the autoregressive component,
        defaults to [1]
    :type autoregressive: List[int]
    :param irregular: Whether to include an additional Gaussian noise,
        defaults to [True, False]
    :type irregular: List[bool]
    :param scores: A dictionary containing scores for different 
        configurations. Currently used to pass the results of 
        cross validation to the final ensemble fit, defaults to None
    :type scores: dict
    :param folds: Number of folds to use for cross validation,
        defaults to 5
    :type folds: int
    :param verbose: Determines the amount of outpout, 0 means no output
        and 2 means maximum outout, defaults to 0
    :type verbose: int
    :returns: A dictionary containing the AMSE for each model config
    :rtype: dict
    
    """
    
    timeseries = copy.deepcopy(timeseries)
    
    data = _create_folds(timeseries.data, n = folds)
    
    ensembles = []
    for _fold, _train in data:
        timeseries.data = _train
        rlist = dlm_ensemble(timeseries, "", level, variable_level, trend,
                             variable_trend, seasonal, seasonal_period,
                             seasonal_harmonics, variable_seasonal,
                             autoregressive, irregular, verbose=verbose)
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
        
    
    return scores

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
    irregular: List[bool] = [True,False],
    scores: dict = None,
    verbose: int = 0
) -> DLMResultList:
    """
    Fits an ensemble of Dynamic Linear Models to a TimeSeries object and
    returns a DLMResultList object.
    
    For all keyword arguments (except scores) a list or nested list is
    used to determine the configurations used in the ensemble.
    
    For most parameters a boolean List is used. For example
    variable_level = [True, False] would include model configurations
    with and without a variable level in the ensemble. The possible values
    are therefore [True], [False], [True, False].
    
    If seasonal components are included in the ensemble they can be specified
    using nested lists. Each configuration can included multiple seasonal
    components

    :param timeseries: TimeSeries object do be fitted
    :type timeseries: TimeSeries
    :param name: Identifier for the DLMResult object
    :type name: str
    :param level: Whether to include a level component, defaults to [True]
    :type level: List[bool]
    :param variable_level:  Wheter to allow the level component to vary,
        defaults to [False]
    :type variable_level: List[bool]
    :param trend: Whether to include a trend component, defaults to [True]
    :type trend: List[bool]
    :param variable_trend: Whether to allow the trend component to vary,
        defaults to [True]
    :type variable_trend: List[bool]
    :param seasonal: Whether to include a seasonal component,
        defaults to [True]
    :type seasonal: List[bool]
    :param seasonal_period: List of configurations of seasonal components.
        Each element is a list containing the periods of the seasonal 
        components, defaults to [[365]]
    :type seasonal_period: List[List[float]]
    :param seasonal_harmonics: List harmonics to try for the corresponding
        seasonal components. For each element of seasonal_period this 
        should include a list of harmonics to try, defaults to [[[1,2,3,4]]]
    :type seasonal_harmonics: List[List[List[int]]]
    :param variable_seasonal: Whether a seasonal component is allowed to vary.
        For each element of seasonal_period this should include a list of 
        options, defaults to [[[True, False]]]
    :type variable_seasonal: List[List[List[bool]]]
    :param autoregressive: List of autoregressive components to try, 
        the integer determines the order of the autoregressive component,
        defaults to [1]
    :type autoregressive: List[int]
    :param irregular: Whether to include an additional Gaussian noise,
        defaults to [True, False]
    :type irregular: List[bool]
    :param scores: A dictionary containing scores for different 
        configurations. Currently used to pass the results of 
        cross validation to the final ensemble fit, defaults to None
    :type scores: dict
    :param verbose: Determines the amount of outpout, 0 means no output
        and 2 means maximum outout, defaults to 0
    :type verbose: int
    :returns: An object containing multiple DLMResult objects
    :rtype:  DLMResultList
    """
    # TODO: more checks
    if len(seasonal_period)!=len(seasonal_harmonics):
        raise ValueError("""seasonal_period and seasonal_harmonics need
                        to have the same length!""")
        
    # This obscure code transforms the seasonal_period, seasonal_harmonics
    # and variable_seasonal parameters to two dicts which are later used to 
    # iterate over all possible ensemble configurations
    dicts = []
    dicts2 = []
    for i, periods in enumerate(seasonal_period):
        temp1 = []
        for j in range(len(periods)):
            temp2 = []
            for k in seasonal_harmonics[i][j]:
                temp2.append([periods[j],k])
            temp1.append(temp2)

        for c in itertools.product(*temp1,*variable_seasonal[i]):

            dict_list = []
            dict_list2 = []
            for idx in range(len(c)//2):
                dict_list.append(
                {'period': c[idx][0],
                'harmonics': c[idx][1]},
                )
                dict_list2.append(
                c[idx+len(c)//2]
                )
            dicts2.append(dict_list2)
            dicts.append(dict_list)
    ####
    
    # TODO: add progress bar, more possible output
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
            if verbose>=1: print(f"Processed: {resobj.name_from_spec()}")
            out.append(resobj)
    if False in seasonal:
        for a, t, st, l, sl, i in itertools.product(
            autoregressive, trend, variable_trend, level, variable_level, irregular):
           
            
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
            if verbose>=1: print(f"Processed: {resobj.name_from_spec()}")
            out.append(resobj)

    return DLMResultList(out)



def model_selection_bias_ALI(results: DLMResultList, years: ArrayLike,
                             percentile: float = 25, metric: str = "aic"):
    """
    Calculate the model selection bias for Dynamic Linear Models results.
    
    This function computes the model selection bias for ALIs for the given
    DLMResultList. The bias is calculated by computing the weighted variance
    between the average fit ALI and each individual fit ALI for each year.
    The bias is calculated using all models whose metric is within 
    the specified percentile.

    :param results: DLMResultList
    :type results: DLMResultList
    :param years: Array of years for which the bias is calculated
    :type years: ArrayLike
    :param percentile: Percentile of models to use for comparison
    :type percentile: float
    :param metric: Metric to use for comparison of models, defaults
        to 'aic'
    :type metric: str
    :returns: np.ndarray: An array containing the model selection bias
        for each year specified in the `years` array.

    """
    
    if years is not None:
        years = np.asarray(years)
        
    else:
        years = np.array([2018,2019,2020,2021,2022])
        
    score_list = np.asarray([_r.dlm_fit_rating[metric] for _r in results.results])
    score_max = np.percentile(score_list, percentile, method="nearest")

    ali_list = []
    ali_std_list = []
    for _r in results.results:
        if _r.dlm_fit_rating[metric]>score_max: continue

        ali = np.zeros_like(years,dtype=float)
        ali_std = np.zeros_like(years,dtype=float)
        for i, y in enumerate(years):
            ali[i], ali_std[i] = annual_level_increase(_r, y)

        ali_list.append(ali)
        ali_std_list.append(ali_std)

    ali_list = np.array(ali_list)
    ali_std_list = np.array(ali_std_list)

    ali = np.zeros_like(years,dtype=float)
    ali_std = np.zeros_like(years,dtype=float)
    for i, y in enumerate(years):
            ali[i], ali_std[i] = annual_level_increase(results.get_best_result(sort=metric), y)
            
    ali_avg = np.average(ali_list, axis=0)
    ali_std_avg = np.std(ali_list, axis=0)
    return np.sqrt(np.average((ali_avg-ali_list)**2,weights=1/np.sqrt(ali_std_avg**2+ali_std_list**2),axis=0))


def model_selection_bias_trend(results: DLMResultList,t1: np.datetime64 = None, t2: np.datetime64 = None,
                               percentile: float = 25, metric: str = "aic", tolerance: np.timedelta64 = None ):
    """Calculate the model selection bias for Dynamic Linear Models (DLM) results.
    
    This function computes the model selection bias for growth rates for the given DLMResultsList. The bias is calculated by
    computing the weighted variance between the average fit trend (growth rate) and each individual fit trend.
    The bias is calculated using all models whose metric is within the specified percentile. If `t1` and/or `t2`
    are specified the times will be used to determine the start and end date for the comparison.

    :param results: DLMResultList
    :type results: DLMResultList
    :param t1: Date
    :type t1: np.datetime64
    :param t2: Date
    :type t2: np.datetime64
    :param tolerance: Tolerance
    :type tolerance: np.timedelta64
    :param percentile: Percentile of models to use for comparison
    :type percentile: float
    :param metric: Metric to use for comparison of models, defaults
        to 'aic'
    :type metric: str
    :returns: np.ndarray: An array containing the model selection bias
        for each year specified in the `years` array.

    """
    _r = results.get_best_result(sort=metric)
    if t1 is not None:
        idx1 = _get_idx_at_time(_r.timeseries.time64, t1, tolerance=tolerance)
    else:
        idx1 = None
    if t2 is not None:
        idx2 = _get_idx_at_time(_r.timeseries.time64, t2, tolerance=tolerance)
    else:
        idx2 = None
    
    score_list = np.asarray([_r.dlm_fit_rating[metric] for _r in results.results])
    score_max = np.percentile(score_list,percentile,method="nearest")

    trend_list = []
    trend_cov_list = []
    for _r in results.results:
        if _r.dlm_fit_rating[metric]>score_max: continue


        trend_list.append(_r.trend[idx1:idx2])
        trend_cov_list.append(_r.trend_cov[idx1:idx2])

    trend_list = np.array(trend_list)
    trend_cov_list = np.array(trend_cov_list)

            
    trend_avg = np.average(trend_list, axis=0)
    trend_cov_avg = np.std(trend_list, axis=0)
    
    _r = results.get_best_result(sort=metric)
    
    return np.sqrt(np.average((trend_avg-_r.trend[idx1:idx2])**2,weights=1/np.sqrt(trend_cov_avg**2+_r.trend_cov[idx1:idx2]**2),axis=0))

def mean_level_from_dates(data: DLMResult, t1: np.datetime64, t2: np.datetime64, 
                          tolerance: np.timedelta64 = np.timedelta64(1,'D') ) -> float:
    """
    Returns the mean level between the two given dates. 
    Uses times from DLMResult object closest to 't1' and 't2' within 
    given tolerance for the calculation. The tolerance defaults to one day.
    Returns `None` if no times fall within tolerance.

    :param data: DLMResult object used for calculation
    :type data: DLMResult
    :param t1: Date
    :type t1: np.datetime64
    :param t2: Date
    :type t2: np.datetime64
    :param tolerance: Tolerance
    :type tolerance: np.timedelta64
    :returns: float: Mean of the values in X that fall within the specified date range.

    """
    
    idx1 = _get_idx_at_time(data.timeseries.time64, t1, tolerance=tolerance)
    idx2 = _get_idx_at_time(data.timeseries.time64, t2, tolerance=tolerance)
    
    if idx1 is not None and idx2 is not None:
        return np.mean(data.timeseries.level[idx1:idx2])
    else:
        return None
    

def _get_idx_at_time(times,time, tolerance=np.timedelta64(1,'D')):
    """

    :param times: 
    :param time: 
    :param tolerance:  (Default value = np.timedelta64(1)
    :param 'D'): 

    """
    delta = np.min(np.abs(times-time))
    idx = np.argmin(np.abs(times-time))
    if tolerance is None:
        return idx
    if delta>tolerance:
        return None
    else:
        return idx
    
def annual_level_increase(data: DLMResult, year: int, tolerance: np.timedelta64 = np.timedelta64(1,'D')) -> Tuple[float, float]:
    """
    Calculate annual increase in level increase between `year` and `year+1` 
    for a given DLMResult object. Returns increase and corresponding error.
    Uses times from DLMResult object closest to 'year-01-01' and 'year+1-12-31'
    within given tolerance for the calculation. The tolerance defaults to one day.
    Returns `(None, None)` if no times fall within tolerance.
    
    :param data: DLMResult object used for calculation
    :type data: DLMResult
    :param year: Year
    :type year: int
    :param tolerance: Tolerance
    :type tolerance: np.timedelta64

    :returns: Tuple[float, float]: a tuple containing the annual increase and
        standard deviation.

    """
    inc = -999
    inc_std = -999
    
    t1 = np.datetime64(f"{year}-01-01")
    t2 = np.datetime64(f"{year}-12-31")
    
    idx1 = _get_idx_at_time(data.timeseries.time64, t1, tolerance=tolerance)
    idx2 = _get_idx_at_time(data.timeseries.time64, t2, tolerance=tolerance)
    
    if idx1 is not None and idx2 is not None:
        inc = data.level[idx2] - data.level[idx1]
        inc_std = np.sqrt(data.level_cov[idx2]+data.level_cov[idx1])
        return inc, inc_std
    else:
        #Todo raise error
        return None, None
    











