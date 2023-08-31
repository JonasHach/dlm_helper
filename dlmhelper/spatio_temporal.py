# -*- coding: utf-8 -*-
"""This module handles operations on spatio temporal data

"""

import warnings
from typing import Tuple

import numpy as np



def _nansum(array: np.ndarray, axis: int = None, out: np.ndarray = None):
    """This alters the numpy.nansum to return nan if all values are nan

    :param array: Array
    :type array: np.ndarray
    :param axis: Axis to be  passed to numpy functions
    :type axis: int
    :param out: Out parameter to be passed to numpy functions
    :type out: np.ndarray
    :returns: The nansum of the input array, but if all values are
    Nan's then `np.nan` is returned
    :rtype: np.ndarray
    """
    return np.where(np.all(np.isnan(array), axis=axis), np.nan, np.nansum(array,axis=axis, out=out))

def _area_weighted_average(data: np.ndarray, error: np.ndarray, lats: np.ndarray, lons: np.ndarray, zonal_avg: bool = False ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the area weighted average of spatio temporal data.
    
    :param data: Data of the shape (lat, lon, time)
    :type data: np.ndarray
    :param error: Error corresponding to data of shape
        (lat, lon, time)
    :type error: np.ndarry
    :param lats: Lower latitude bounds of grid cells of shape (lat)
    :type lats: np.ndarray 
    :param lons: Lower longitude bounds of grid cells of shape (lon)
    :type lons: np.ndarray 
    :param zonal_avg: If `True` average over longitudes first, this
        can help minimizing sampling bias in certain cases
    :type zonal_avg: bool:
    :returns: timeseries and corresponding error of shape (time)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    
    
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
    ma_weights[np.isnan(data) | np.isnan(error)] = np.nan
    
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
    
   
    return data_mean2, error_mean2


def _area_perc(theta1: float, theta2: float) -> float:
    """Calculate the percentage of surface area between two angles on a sphere.

    :param theta1: Angle 1
    :param theta2: Angle 2
    :type theta1: float 
    :type theta2: float
    :returns: The percentage of surface area between theta1 and theta2.
    :rtype: float

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


def inhomogeneity_spatial(lat: np.ndarray, lon: np.ndarray, 
                        N: np.ndarray, scale_lat: float = None, scale_lon: float = None) -> np.ndarray:
    """Calculate the inhomogeneity for a spatial grid of data.
    Based on Sofieva et al., 2014 (https://doi.org/10.5194/amt-7-1891-2014)

    :param lat: Lower latitude bounds of grid cells of shape (lat)
    :type lat: np.ndarray
    :param lon: Lower longitude bounds of grid cells of shape (lon)
    :type lon: np.ndarray
    :param N: Number of measurements per cell of shape
        (lat, lon, time)
    :type N: np.ndarry
    :param scale_lat: Weight of the latitudinal part of the spatial
        inhomogeneity, if not specified lat and lon part will be
        equally weighted, defaults to `None`
    :type scale_lat: float
    :param scale_lon: Weight of the longitudinal part of the spatial
        inhomogeneity, if not specified lat and lon part will be
        equally weighted, defaults to `None`
    :type scale_lon: float
    :returns: Array of shape (time, 3), aach row contains the
        inhomogeneity, asymmetry component, and entropy component
        for the corresponding time step in N.
    :rtype: np.ndarray
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

def inhomogeneity_temporal(lat: np.ndarray, lon: np.ndarray, time: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Calculate the temporal inhomogeneity of data at each grid point.
    Based on Sofieva et al., 2014 (https://doi.org/10.5194/amt-7-1891-2014)

    :param lat: Lower latitude bounds of grid cells of shape (lat)
    :type lat: np.ndarray
    :param lon: Lower longitude bounds of grid cells of shape (lon)
    :type lon: np.ndarray
    :param time: Time values of shape (time)
    :type time: np.ndarray
    :param N: Number of measurements per cell of shape
        (lat, lon, time)
    :type N: np.ndarry
    
    :returns: Array of temporal homogeneity values at each grid point,
        with shape (lat, lon, 3).
        The last dimension contains the temporal inhomogeneity, 
        asymmetry component, and entropy component.
    :rtype: np.ndarray:
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