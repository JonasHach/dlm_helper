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

from dataclasses import dataclass, fields, asdict
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

@dataclass
class DLMResult:
    name: str 
    data_type: str
    grid: np.array
    ht_lim: float
    hs_lim: float
    scale_lat: float
    scale_lon: float
    
    time_unit: str
    time: np.array 
    
    data_pre: np.array #original data 
    data_post: np.array #data after processing (i.e. sampling bias treatment)
    
    dlm_specification: dict
    dlm_fit_params: dict
    dlm_fit_rating: dict
    
    level: np.array
    level_cov: np.array
    trend: np.array
    trend_cov: np.array
    seas: np.array
    seas_cov: np.array
    ar: np.array
    ar_cov: np.array
    resid: np.array
    
    _loglikeburn: int
    
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for field_name, field_type in self.__annotations__.items():
            if field_type == np.array:
                if not np.array_equal(getattr(self, field_name), getattr(other, field_name),equal_nan=True):
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
    def create(cls,name, data_type, time, results, time_unit="day",data_pre=None, data_post=None, grid=[-90,90,-180,180],ht_lim=-1, hs_lim=-1, scale_lat=-1, scale_lon=-1, score = None):
        res = results; 
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
            seas = np.empty([time.size, len(res.freq_seasonal)])
            seas_cov = np.empty([time.size, len(res.freq_seasonal)])
            for i in range(len(res.freq_seasonal)):
                seas[:,i] = res.freq_seasonal[i]['smoothed']
                seas_cov[:,i] = res.freq_seasonal[i]['smoothed_cov']
        else:
            seas_cov = seas = np.zeros(time.shape)

        if res.autoregressive is not None:
            ar = res.autoregressive['smoothed']
            ar_cov = res.autoregressive['smoothed_cov']
        else:
            ar_cov = ar = np.zeros(time.shape)
            
        
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
        
        return cls(name, data_type, np.asarray(grid),ht_lim, hs_lim, scale_lat, scale_lon,time_unit,time, data_pre,data_post
                   ,spec,dlm_fit_params,dlm_fit_rating,lvl, lvl_cov, trend, trend_cov, seas, seas_cov, ar, ar_cov, resid,_nb)
        
    @classmethod
    def load(cls,path):
        with open(path,'r') as f:
            s = f.read()
            argDict = json.loads(s, object_hook=JSONDecoder)
            fieldSet = {f.name for f in fields(cls) if f.init}
            filteredArgDict = {k : v for k, v in argDict.items() if k in fieldSet}
            return cls(**filteredArgDict)
            
    def save(self,path,fname=None):
        if not fname:
            fname = f"{self.name}_{self.data_type}_{self.time_unit}{''.join('_'+str(x) for x in self.grid)}.json"
        s=json.dumps(asdict(self),cls=NpDecoder, indent=2)
        with open(path+fname, 'w') as f:
            f.write(s)
            print("Saved data at:",path+fname)
    
    def plot(self, figsize=(15,15)):
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(221)

        if self.data_post is not None: ax1.scatter(self.time, self.data_post, marker='o', color='gray', edgecolor='black',label='input data')
        if self.data_pre is not None: ax1.scatter(self.time, self.data_pre, marker='.', color='darkgray',label='input data (pre-filtering)')
        ax1.plot(self.time, self.level+np.sum(self.seas,axis=1)+self.ar,color='red',label='DLM fit')
        ax1.legend(); ax1.xaxis_date(); ax1.set_ylabel("XCH$_4$ [ppb]")
        ax2 = fig.add_subplot(222,sharey=ax1)
        ax2.plot(self.time, self.level, color='red', label='DLM: level')
        ax2.plot(self.time, self.level+np.sum(self.seas,axis=1), color='red', ls='--', label='DLM: level+seas')
        ax2.legend(); ax2.xaxis_date(); ax2.set_ylabel("XCH$_4$ [ppb]")
        ax3 = fig.add_subplot(223)
        ax3.plot(self.time, np.sum(self.seas,axis=1), color='red', label='DLM: seas')
        ax3.legend(); ax3.xaxis_date(); ax3.set_ylabel("XCH$_4$ [ppb]")
        ax4=fig.add_subplot(224)
        ax4.plot(self.time, self.ar, color='red', label='DLM: AR')
        ax4.legend(); ax4.xaxis_date(); ax4.set_ylabel("XCH$_4$ [ppb]")
        return fig
    
    def plot_summary(self, ax=None,ar=False, **kwargs):
        if ax is None: fig, ax = plt.subplots()
        ax.scatter(self.time, self.data_post,label='data',color='gray',edgecolor='black',**kwargs)
        if not ar: ax.plot(self.time, self.level+np.sum(self.seas,axis=1),color='darkorange',label='level+seas')
        if ar: ax.plot(self.time, self.level+np.sum(self.seas,axis=1)+self.ar,color='darkorange',label='level+seas+ar')
        ax.plot(self.time, self.level, color='red', ls='--',label='level')
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
        print(f"grid: {self.grid}")
        print(f"time range: {dt.datetime(1970,1,1)+dt.timedelta(days=int(self.time[0])):%d.%m.%y} --- {dt.datetime(1970,1,1)+dt.timedelta(days=int(self.time[-1])):%d.%m.%y}")    
        print(f"\n{border}")
        
        
    def name_from_spec(self) -> str:
        spec = self.dlm_specification
        out = ''
        if spec['level']: out+='L'
        if spec['stochastic_level']: out+='s'
        if spec['trend']: out+='T'
        if spec['stochastic_trend']: out+='s'
        if spec['freq_seasonal']: out+='S'
        if spec['stochastic_freq_seasonal'][0]: out+='s'
        if spec['freq_seasonal']: out+='P'+str(spec['freq_seasonal_periods'][0])+'H'+str(spec['freq_seasonal_harmonics'][0])
        if spec['autoregressive']: out+='A'+str(spec['ar_order'])
        if spec['irregular']: out+='I'
        return out
    
    @classmethod
    def _name_from_spec(cls, spec) -> str:
        out = ''
        if spec['level']: out+='L'
        if spec['stochastic_level']: out+='s'
        if spec['trend']: out+='T'
        if spec['stochastic_trend']: out+='s'
        if spec['freq_seasonal']: out+='S'
        if spec['stochastic_freq_seasonal'][0]: out+='s'
        if spec['freq_seasonal']: out+='P'+str(spec['freq_seasonal_periods'][0])+'H'+str(spec['freq_seasonal_harmonics'][0])
        if spec['autoregressive']: out+='A'+str(spec['ar_order'])
        if spec['irregular']: out+='I'
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
                results.append(result)
        return cls(results=results)            