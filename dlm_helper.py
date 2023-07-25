# -*- coding: utf-8 -*-
"""This module provides functions which are used in DLM notebooks

    Version: 0.1.0
    Latest changes: 25.07.2023
    Author: Jonas Hachmeister
"""

from typing import Union, List, Tuple, Optional

import datetime
import itertools

import numpy as np
from numpy.typing import ArrayLike
import xarray as xr
import statsmodels.api as sm

from dlm_helper.dlm_data import DLMResult, DLMResultList


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


def inhomogeneity_spatial(lat: ArrayLike, lon: ArrayLike, 
                        N: ArrayLike, scale_lat: float = 0.5, scale_lon: float = 0.5) -> np.ndarray:
    """
    Calculate the inhomogeneity for a spatial grid of data. 
    Based on Sofieva et al., 2014 (https://doi.org/10.5194/amt-7-1891-2014)

    Parameters:
        lat (array_like): 1D array of latitude values.
        lon (array_like): 1D array of longitude values.
        N (array_like): 2D array of data values, with shape (lat, lon) containing the Number of
            measurements per cell.
        scale_lat (float): Scaling factor for the latitude component of the homogeneity index.
            Default value is 0.5.
        scale_lon (float): Scaling factor for the longitude component of the homogeneity index.
            Default value is 0.5.

    Returns:
        numpy.ndarray: 2D array of homogeneity index values, with shape (N.shape(2), 3).
            Each row contains the inhomogeneity, asymmetry component, and entropy component
            for the corresponding time step in N.

    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    
    H_out = np.zeros((N.shape[2], 3))
    if lat.shape[0] > 1:
        lat_step = np.abs(lat[0] - lat[1])
        delta_lat = lat[-1] + lat_step - lat[0]
    else:
        lat_step = 0
        delta_lat = 1
        
    lon_step = np.abs(lon[0] - lon[1])
    delta_lon = lon[-1] + lon_step - lon[0]
    lat_mid = np.min(lat) + delta_lat / 2
    lon_mid = np.min(lon) + delta_lon / 2
    
    for c in range(0, N.shape[2]):
        data = N[:,:,c]
        mean_lat = mean_lon = 0
        
        for i, llat in enumerate(lat):
            for j, llon in enumerate(lon):
                    mean_lat += llat * data[i, j]
                    mean_lon += llon * data[i, j]
        mean_lat = mean_lat / np.nansum(data.flatten())
        mean_lon = mean_lon / np.nansum(data.flatten())
        A_lat = 2 * np.abs(mean_lat - lat_mid) / delta_lat
        A_lon = 2 * np.abs(mean_lon - lon_mid) / delta_lon
        
        A_total = scale_lat * A_lat + scale_lon * A_lon
        
        E = np.zeros(lat.shape)
        n0 = np.nansum(data[:,:])
        E=(-1 / np.log(lon.shape[0] * lat.shape[0])) * np.nansum((data[:,:] / n0) * np.log(data[:,:] / n0))
        H = 0.5 * (A_total + (1 - E))
        H_out[c, 0] = H
        H_out[c, 1] = A_total
        H_out[c, 2]= E
    return H_out

def inhomogeneity_temporal(lat: ArrayLike, lon: ArrayLike, time: ArrayLike, N: ArrayLike) -> np.ndarray:
    """
    Calculate the temporal inhomogeneity of data at each grid point.
    Based on Sofieva et al., 2014 (https://doi.org/10.5194/amt-7-1891-2014)

    Parameters:
        lat (array_like): 1D array of latitude values.
        lon (array_like): 1D array of longitude values.
        time (array_like): 1D array of datetime values corresponding to each time step in N.
        N (array_like): 3D array of data values with shape (lat.shape[0], lon.shape[0], days.shape[0]).

    Returns:
        ndarray: 3D array of temporal homogeneity values at each grid point,
        with shape (lat.shape[0], lon.shape[0], 3).
        The last dimension contains three values inhomogeneity, asymmetry component, and entropy component
    """
    
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    time = np.asarray(time)
    
    H_out = np.zeros((lat.shape[0], lon.shape[0], 3))
    l = N.shape[2]
    for i, llat in enumerate(lat):
        for j, llon in enumerate(lon):
            data = N[i,j,:]

            mean_t=0
            for z, d in enumerate(time):
                mean_t += d * data[z]
                            
            mean_t = mean_t/  np.nansum(data.flatten())
            A_t = 2 * np.abs(mean_t - np.nanmean(time)) / l
            A_total = A_t

            n0 = np.nansum(data[:])
            E=(-1 / np.log(l)) * np.nansum((data[:] / n0) * np.log(data[:] / n0))
            H = 0.5 * (A_total + (1 - E))
            H_out[i, j, 0] = H
            H_out[i, j, 1] = A_total
            H_out[i, j, 2] = E
    return H_out


def area_perc(theta1: float, theta2: float) -> float:
    """Calculate the percentage of surface area between two angles on a sphere.

    Parameters:
        theta1 (float): The first angle in degrees.
        theta2 (float): The second angle in degrees.

    Returns:
        float: The percentage of surface area between theta1 and theta2.
    """
    
    # Ensure theta2 is greater than theta1
    if theta2 <= theta1:
        _theta = theta1
        theta1 = theta2
        theta2 = _theta

    # Calculate surface area between pole and theta1
    A1 = 0.5 * (1 - np.cos((90 - theta1) * np.pi / 180))

    # Calculate surface area between pole and theta2
    A2 = 0.5 * (1 - np.cos((90 - theta2) * np.pi / 180))

    return A1 - A2


def weighted_vmr_average(vmr: ArrayLike, sdev: ArrayLike, lats: ArrayLike, lons: ArrayLike, zonal_avg: bool = False ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the area weighted average volume mixing ratio.

    Paramters:
    vmr (np.ndarray): A 3D array with the first two dimensions as the spatial grid and the third as the time grid
    sdev (np.ndarray): A 3D array with the first two dimensions as the spatial grid and the third as the time grid
    lats (np.ndarray): A 1D array of latitudes
    lons (np.ndarray): A 1D array of longitudes
    zonal_avg (bool, optional): If true average first over longitudes and second over latitudes. Else the array
                    is flattened and then averaged.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The weighted  mean and standard deviation
    """
    
    vmr = np.asarray(vmr)
    sdev = np.asarray(sdev)
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    
    if lats.shape[0] == 1:
        return np.nanmean(vmr, axis=(0, 1)), np.nanmean(sdev, axis=(0, 1))
    
    weights = np.zeros((vmr.shape))
    weights_zonal = np.zeros((vmr.shape[0],vmr.shape[2]))
    lat_step = np.abs(lats[0] - lats[1])
    nlons = vmr.shape[1]
    i = 0
    
    for x in np.arange(np.min(lats), np.max(lats) + lat_step, lat_step):
        w = area_perc(x, x + lat_step)
        weights[i, :, :] = w / nlons
        #weights[-i-1,:,:]=w/nlons
        weights_zonal[i] = w
        i = i + 1
    
    wweights = np.zeros((vmr.shape))
    wweights_zonal = np.zeros((weights_zonal.shape))
    for t in range(0,vmr.shape[2]):
        sumw = np.sum(weights[:, :, t][~np.isnan(vmr[:, :, t]) & ~np.isnan(sdev[:, :, t])])
        wweights[:, :, t] = 1 / sumw * weights[:, :, t]
        
        sumw2 = np.sum(weights_zonal[:, t][~np.isnan(np.nanmean(vmr[:, :, t],axis=1)) & ~np.isnan(np.nanmean(sdev[:, :, t],axis=1))])
        wweights_zonal[:,t] = 1/sumw2*weights_zonal[:,t]
        
    ma_weights = wweights
    ma_weights[np.isnan(vmr) | np.isnan(sdev)] = 0
    
    ma_weights_zonal = wweights_zonal
    ma_weights_zonal[np.isnan(np.nanmean(vmr,axis=1)) | np.isnan(np.nanmean(sdev,axis=1))] = 0
    
    if zonal_avg == True:
        vmr_zonal = np.nanmean(vmr, axis=1)
        sdev_zonal = np.nanmean(sdev, axis=1)
        vmr_mean2 = np.nansum(vmr_zonal*ma_weights_zonal,axis=0)
        sdev_mean2 = np.nansum(sdev_zonal*ma_weights_zonal,axis=0)
    else:
        vmr_mean2 = np.nansum(vmr * ma_weights, axis = (0,1))
        sdev_mean2 = np.nansum(sdev * ma_weights, axis = (0,1))
    
    vmr_mean2[vmr_mean2 == 0] = np.nan
    return vmr_mean2, sdev_mean2


def dlm_ensemble(
    name: str,
    product_type: str,
    time: ArrayLike,
    vmr: ArrayLike,
    time_unit: str,
    data_pre: Optional[ArrayLike] = None,
    data_post: Optional[ArrayLike] = None,
    grid: Optional[ArrayLike] = None,
    ht_lim: Optional[float] = None,
    hs_lim: Optional[float] = None,
    scale_lat: Optional[float] = None,
    scale_lon: Optional[float] = None,
    harmonics: ArrayLike = [1, 2, 3, 4],
    ar: ArrayLike = [1],
    trend: ArrayLike = [True],
    stochastic_trend: ArrayLike = [True],
    level: ArrayLike = [True],
    stochastic_level: ArrayLike = [False],
    seas: ArrayLike = [True],
    seas_period: ArrayLike = [365],
    stochastic_seas: ArrayLike = [False, True],
    irregular: ArrayLike = [False, True],
    scores: dict = None
) -> DLMResultList:
    """Fits an ensemble of Dynamic Linear Models (DLMs) to a time series.

    Parameters:
        name (str): The name of the DLM.
        product_type (str): The identifier of the data product.
        time (array-like): The time values of the data.
        vmr (array-like): The observed data.
        time_unit (str): The unit of time used.
        data_pre (array-like, optional): The data before pre-processing
        data_post (array-like, optional): Can be provided to include the fitted data to the
            DLMResult object, should be the same as vmr    
        grid (array-like, optional): The grid information formatted as
            [lat_low, lat_high, lon_low, lon_high, lat_step, lon_step]
        ht_lim (float, optional): Limit of temporal inhomogeneity
        hs_lim (float, optional): Limit of spatial inhomogeneity 
        scale_lat (float, optional): The scaling factor for latitude component of
            spatial inhomogeneity
        scale_lon (float, optional): The scaling factor for longitude component of
            spatial inhomogeneity    
        harmonics (ArrayLike[int], optional): The number of harmonics to use for seasonal components.
        ar (ArrayLike[int], optional): The order of the autoregressive component.
        trend (list of bool, optional): Whether or not to include a trend component.
        stochastic_trend (list of bool, optional): Whether or not the trend component is stochastic.
        level (list of bool, optional): Whether or not to include a level component.
        stochastic_level (list of bool, optional): Whether or not the level component is stochastic.
        seas (list of bool, optional): Whether or not to include a seasonal component.
        seas_period (list of int, optional): The period of the seasonal component in days.
        stochastic_seas (list of bool, optional): Whether or not the seasonal component is stochastic.
        irregular (list of bool, optional): Whether or not to include an irregular component.
        scores (dict[float], optional): dictionary in form {name_string: score}, where name_string is
            is given by name_from_spec

    Returns:
        DLMResultList: An object containing multiple DLMResult objects

    """
    time = np.asarray(time)
    vmr = np.asarray(vmr)
    if data_pre is not None:
        data_pre = np.asarray(data_pre)
    if data_post is not None:
        data_post = np.asarray(data_post)
    if grid is not None:
        grid = list(np.asarray(grid))
    harmonics = np.asarray(harmonics,dtype=int).tolist()
    
    ar = np.asarray(ar).tolist()
    trend = np.asarray(trend,dtype=bool).tolist()
    stochastic_trend = np.asarray(stochastic_trend,dtype=bool).tolist()
    level = np.asarray(level,dtype=bool).tolist()
    stochastic_level = np.asarray(stochastic_level,dtype=bool).tolist()
    seas = np.asarray(seas,dtype=bool).tolist()
    seas_period = np.asarray(seas_period,dtype=float).tolist()
    stochastic_seas = np.asarray(stochastic_seas,dtype=bool).tolist()
    irregular = np.asarray(irregular,dtype=bool).tolist()
    
    out = []
    for h, a, t, st, l, sl, s, sp, ss,i in itertools.product(harmonics,ar, trend, stochastic_trend, level, stochastic_level, seas, seas_period, stochastic_seas, irregular):
        
        model = sm.tsa.UnobservedComponents(vmr,level=l, trend=t,freq_seasonal=[{'period': sp,'harmonics':h}], autoregressive=a, stochastic_level=sl, stochastic_trend=st, stochastic_freq_seasonal=[ss],irregular=i)
        result = model.fit()    
        resobj=DLMResult.create(name,product_type, time, result,time_unit,data_pre=data_pre, data_post=data_post,grid=grid,ht_lim=ht_lim, hs_lim=hs_lim, scale_lat=scale_lat, scale_lon=scale_lon, score=scores)
        out.append(resobj)

    return DLMResultList(out)