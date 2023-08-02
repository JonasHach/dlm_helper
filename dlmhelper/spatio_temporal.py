# -*- coding: utf-8 -*-
"""This module handles operations on spatio temporal data


    Version: 0.1.0
    Latest changes: 27.07.2023
    Author: Jonas Hachmeister
"""

import warnings
from typing import Union, List, Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike


def _nansum(array, axis = None, out = None):
    "This alters the numpy.nansum to return nan if all values are nan"
    return np.where(np.all(np.isnan(array), axis=axis), np.nan, np.nansum(array,axis=axis, out=out))

def _area_weighted_average(data: ArrayLike, error: ArrayLike, lats: ArrayLike, lons: ArrayLike, zonal_avg: bool = False ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the area weighted average of spatio temporal data.

    Paramters:
    data (np.ndarray): A 3D array with the first two dimensions as the spatial grid and the third as the time grid
    error (np.ndarray): A 3D array with the first two dimensions as the spatial grid and the third as the time grid
    lats (np.ndarray): A 1D array of latitudes
    lons (np.ndarray): A 1D array of longitudes
    zonal_avg (bool, optional): If true average first over longitudes and second over latitudes. Else the array
                    is flattened and then averaged.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The weighted  mean and standard deviation
    """
    

    data = np.asarray(data)
    error = np.asarray(error)
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    
    if lats.shape[0] == 1:
        return np.nanmean(data, axis=(0, 1)), np.nanmean(error, axis=(0, 1))
    
    weights = np.zeros((data.shape))
    weights_zonal = np.zeros((data.shape[0],data.shape[2]))
    lat_step = np.abs(lats[0] - lats[1])
    nlons = data.shape[1]
    i = 0
    
    # Calculate weights for single grid cells / zonal bands
    for x in np.arange(np.min(lats), np.max(lats) + lat_step, lat_step):
        w = _area_perc(x, x + lat_step)
        weights[i, :, :] = w / nlons
        #weights[-i-1,:,:]=w/nlons
        weights_zonal[i] = w
        i = i + 1
    
    # Account for gaps in gridded data
    wweights = np.zeros((data.shape))
    wweights_zonal = np.zeros((weights_zonal.shape))
    for t in range(0,data.shape[2]):
        sumw = np.sum(weights[:, :, t][~np.isnan(data[:, :, t]) & ~np.isnan(error[:, :, t])])
        if sumw>0:
            wweights[:, :, t] = 1 / sumw * weights[:, :, t]
        else:
            continue
            
        #We expect mean of empty slice warnings that can be safely ignored
        with warnings.catch_warnings():    
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            
            sumw2 = np.sum(weights_zonal[:, t][~np.isnan(np.nanmean(data[:, :, t],axis=1)) & ~np.isnan(np.nanmean(error[:, :, t],axis=1))])
        if sumw2>0:
            wweights_zonal[:,t] = 1/sumw2*weights_zonal[:,t]
        else:
            wweights_zonal[:,t] = weights_zonal[:,t]
        
    ma_weights = wweights
    ma_weights[np.isnan(data) | np.isnan(error)] = 0
    
    ma_weights_zonal = wweights_zonal
    
    with warnings.catch_warnings():    
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        ma_weights_zonal[np.isnan(np.nanmean(data,axis=1)) | np.isnan(np.nanmean(error,axis=1))] = 0
    

    
    if zonal_avg == True:
        with warnings.catch_warnings():    
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            data_zonal = np.nanmean(data, axis=1)
            error_zonal = np.nanmean(error, axis=1)
            
            data_mean2 = _nansum(data_zonal*ma_weights_zonal,axis=0)
            error_mean2 = _nansum(error_zonal*ma_weights_zonal,axis=0)
    else:
        with warnings.catch_warnings():    
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            data_mean2 = _nansum(data * ma_weights, axis = (0,1))
            error_mean2 = _nansum(error * ma_weights, axis = (0,1))
    
    #TODO: possible problem with 0 in data?
    #data_mean2[data_mean2 == 0] = np.nan
    return data_mean2, error_mean2


def _area_perc(theta1: float, theta2: float) -> float:
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


def inhomogeneity_spatial(lat: ArrayLike, lon: ArrayLike, 
                        N: ArrayLike, scale_lat: float = None, scale_lon: float = None) -> np.ndarray:
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
    
    if len(lat.shape)>1:
        raise ValueError("Expected one dimensional array of latitude values!")
    if len(lon.shape)>1:
        raise ValueError("Expected one dimensional array of longitude values!")
    
    if len(lat)==1: 
        if len(lon)==1:
            raise ValueError("Spatial inhomogeneity not defined for one dimensional data!")
        else:
            _scale_lat = 0; _scale_lon=1
    else:
        if len(lon)==1:
            _scale_lat = 1; _scale_lon=0;
        else:
            _scale_lat = 0.5; _scale_lon=0.5
    
    if scale_lat is not None:
        _scale_lat = scale_lat
    if scale_lon is not None:
        _scale_lon = scale_lon
    
        
    
    H_out = np.zeros((N.shape[2], 3))
    if lat.shape[0] > 1:
        lat_step = np.abs(lat[0] - lat[1])
        delta_lat = lat[-1] + lat_step - lat[0]
    else:
        lat_step = 0
        delta_lat = 1
    if lon.shape[0] >1:  
        lon_step = np.abs(lon[0] - lon[1])
        delta_lon = lon[-1] + lon_step - lon[0]
    else:
        lon_step = 0
        delta_lon = 1
        
    lat_mid = np.min(lat) + delta_lat / 2
    lon_mid = np.min(lon) + delta_lon / 2
    
    for c in range(0, N.shape[2]):
        data = N[:,:,c]
        mean_lat = mean_lon = 0
        
        for i, llat in enumerate(lat):
            for j, llon in enumerate(lon):
                    mean_lat += llat * data[i, j]
                    mean_lon += llon * data[i, j]
        
        n0 = np.nansum(data)
        
        if n0>0:
            mean_lat = mean_lat / np.nansum(data.flatten())
            mean_lon = mean_lon / np.nansum(data.flatten())

            A_lat = 2 * np.abs(mean_lat - lat_mid) / delta_lat
            A_lon = 2 * np.abs(mean_lon - lon_mid) / delta_lon


            A_total = _scale_lat * A_lat + _scale_lon * A_lon

            E = np.zeros(lat.shape)
           
            E = (-1 / np.log(lon.shape[0] * lat.shape[0])) * np.nansum((data / n0) * np.log(data / n0,where=(data!=0)))
            #E = (-1 / np.log(lon.shape[0] * lat.shape[0])) * np.nansum((data / n0) * np.log(data / n0))
            H = 0.5 * (A_total + (1 - E))
            H_out[c, 0] = H
            H_out[c, 1] = A_total
            H_out[c, 2]= E
        else:
            H_out[c, 0] = np.nan
            H_out[c, 1] = np.nan
            H_out[c, 2]= np.nan
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
    
    if len(lat.shape)>1:
        raise ValueError("Expected one dimensional array of latitude values!")
    if len(lon.shape)>1:
        raise ValueError("Expected one dimensional array of longitude values!")
    
    H_out = np.zeros((lat.shape[0], lon.shape[0], 3))
    l = N.shape[2]
    for i, llat in enumerate(lat):
        for j, llon in enumerate(lon):
            data = N[i,j,:]
            
                
            n0 = np.nansum(data)
            
            if n0 > 0:
            
                mean_t=0
                for z, d in enumerate(time):
                    mean_t += d * data[z]

                mean_t = mean_t/  np.nansum(data.flatten())
                A_t = 2 * np.abs(mean_t - np.nanmean(time)) / l
                A_total = A_t

                mask = (data != 0)
                E=(-1 / np.log(l)) * np.nansum((data[mask] / n0) * np.log(data[mask] / n0))
                H = 0.5 * (A_total + (1 - E))
                H_out[i, j, 0] = H
                H_out[i, j, 1] = A_total
                H_out[i, j, 2] = E
            else:
                H_out[i, j, 0] = np.nan
                H_out[i, j, 1] = np.nan
                H_out[i, j, 2] = np.nan
                
            
    return H_out