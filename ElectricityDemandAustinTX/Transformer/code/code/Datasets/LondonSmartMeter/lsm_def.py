# -*- coding: utf-8 -*-
import numpy as np

import os
import subprocess
import torch
from torch.utils.data import Dataset

import pickle
import pgzip
import copy

import datetime

class LondonSmartMeter(Dataset):
    '''
    Forecasting dataset of London electricity smart meter measurements.
    
    https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london?datasetId=4021
    
    __getitem__ returns: ( lookback sequence, forecast ground truth, lookback start time, forecast start time )
        lookback sequence and forecast ground truth have shape (length, 1) if weather is False,
        otherwise they have shape (length, 14) with additional 13 dims of weather data concatenated.
    
    Weather data arranged as follows:
        (visibility,
         wind bearing (cosine),
         wind bearing (sine),
         temperature,
         dew point,
         pressure,
         apparent temperature,
         wind speed,
         precip. type {rain:1, snow: 0},
         weather icon,
         relative humidity,
         cloud cover summary,
         wind level summary)
    '''
    def __init__(self, seq_len, pred_horz, path = '.', timestamps = False, weather = False):
        if 'lsm_dict.pkl.pgz' not in os.listdir(path):
            subprocess.check_call('python ./LondonSmartMeter_hhour.py ./LondonSmartMeter lsm_dict.pkl')
        if 'londonWeather.pkl.pgz' not in os.listdir(path):   
            raise ValueError
            
        with pgzip.open(os.path.join(path,'lsm_dict.pkl.pgz'),'rb') as f:
            s_dict = pickle.load(f)
        
        self.has_weather = weather
        self.return_timestamps = timestamps
        
        self.weather_dict = None
        if self.has_weather:
            with pgzip.open(os.path.join(path,'londonWeather.pkl.pgz'),'rb') as f2:
                weather_dict = pickle.load(f2)
            
            self.weather_dict = weather_dict
        
        #s_dict is dictionary as follows: { lclid: (start_timestamp,Tensor), ...}
        
        #List to store the dataset indices corresponding to each household
        self.household_idxs = [None]*len(s_dict)
        #List to hold the split series
        self.series = [None]*len(s_dict)
        #These two lists will be converted to torch tensors
        self.start_times = []
        self.pred_starttimes = []
        #These two lists stores the datetime format
        self.start_times__ = []
        #self.pred_starttimes__ = []
            
        index_count = 0

        for index, lclid in enumerate(s_dict.keys()):    
            start_time, s_tensor = s_dict[lclid]
            #start_time[0] = start_time[0]%400
            
            #pad_amt = (seq_len+pred_horz) - (len(s_tensor)%(seq_len+pred_horz))
            #s_tensor = torch.nn.functional.pad(s_tensor,pad = (pad_amt,0), value = torch.nan)
            s_tensors = s_tensor.split(seq_len+pred_horz)
            
            #Compute start timestamps for splits
            start_times = [None]*len(s_tensors)
            pred_starttimes = [None]*len(s_tensors)
            
            for i in range(len(start_times)):
                minutes_delta = 30*i*(seq_len + pred_horz)
                time_delta = datetime.timedelta(minutes=minutes_delta)
                new_start_time = start_time + time_delta
                start_times[i] = new_start_time
            
            minutes_delta = 30*seq_len
            time_delta = datetime.timedelta(minutes=minutes_delta)
            
            #Compute prediction start timestamps
            for i in range(len(pred_starttimes)):
                new_predtime = start_times[i] + time_delta
                pred_starttimes[i] = [new_predtime.year,
                                      new_predtime.month,
                                      new_predtime.day,
                                      new_predtime.hour,
                                      new_predtime.minute,
                                      new_predtime.second]
   
            start_times__ = copy.deepcopy(start_times)
            for i in range(len(start_times)):
                new_start_time = start_times[i]
                start_times[i] = [new_start_time.year,
                                  new_start_time.month,
                                  new_start_time.day,
                                  new_start_time.hour,
                                  new_start_time.minute,
                                  new_start_time.second]
            
            #Remove last if length less than the others
            if s_tensors[-1].shape[0] < seq_len+pred_horz:
                s_tensors = s_tensors[:-1]
                start_times = start_times[:-1]
                pred_starttimes = pred_starttimes[:-1]
                start_times__ = start_times__[:-1]
            
            if len(s_tensors) == 0:
                self.series[index] = torch.empty(0)                    
            else:
                s_tensors = torch.stack(s_tensors)
                #Remove invalid (more than 4/5 (80%) of series is 0 or nan)
                s_tensors[s_tensors==0] = torch.nan #0s are invalid too, replace with nan
                sel = (( (s_tensors==0) | s_tensors.isnan()).sum(dim=-1) < (4*(seq_len+pred_horz)//5))
                s_tensors = s_tensors[sel]
                start_times_ = [start_times[i] for i in range(len(start_times)) if sel[i]]
                pred_starttimes_ = [pred_starttimes[i] for i in range(len(pred_starttimes)) if sel[i]]
                _start_times__ = [ start_times__[i] for i in range(len(start_times__)) if sel[i]]
                
                self.series[index] = s_tensors
                self.start_times = self.start_times + start_times_
                self.pred_starttimes = self.pred_starttimes + pred_starttimes_
                self.start_times__ = self.start_times__ + _start_times__
             
            if len(s_tensors) == 0:
                self.household_idxs[index] = []
            else:
                self.household_idxs[index] = list(range(index_count,index_count+len(s_tensors)))
                index_count = index_count + len(s_tensors)
            
        self.series = torch.cat(self.series,dim=0).unsqueeze(-1)
        self.start_times = torch.tensor(self.start_times,dtype = torch.long)
        self.pred_starttimes = torch.tensor(self.pred_starttimes,dtype = torch.long)
        
        #self.series[:,:seq_len] = self.series[:,:seq_len].nan_to_num(nan=0.,posinf=0.,neginf=0.)

        #Series normalization
        smin = self.series.nan_to_num(nan=torch.finfo(self.series.dtype).max).amin(dim=-2,keepdim=True)
        smax = self.series.nan_to_num(nan=torch.finfo(self.series.dtype).min).amax(dim=-2,keepdim=True)
        self.series = (self.series - smin.broadcast_to(self.series.shape))/(smax-smin+1e-10).broadcast_to(self.series.shape)

        if self.has_weather:
            #Weather series normalization
            #Only dimensions 0, 3, 4, 5, 6, 7 needs normalization
            wdmin = self.weather_dict['tensor'].nan_to_num(nan=torch.finfo(self.weather_dict['tensor'].dtype).max).amin(dim=-2,keepdim=True)
            wdmax = self.weather_dict['tensor'].nan_to_num(nan=torch.finfo(self.weather_dict['tensor'].dtype).min).amax(dim=-2,keepdim=True)
            self.weather_dict['tensor'] = (self.weather_dict['tensor'] - wdmin.broadcast_to(self.weather_dict['tensor'].shape))\
                /(wdmax-wdmin + 1e-10).broadcast_to(self.weather_dict['tensor'].shape)
            
            self.weather_dict['tensor'] = self.weather_dict['tensor'].type(torch.float32)
            
        self.length = len(self.series)
        self.seq_len = seq_len
        self.pred_horz = pred_horz
        
    
    def get_household_indices(self):
        """Returns the list containing, for each household, a list of corresponding indices."""
        return copy.deepcopy(self.household_idxs)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        if self.has_weather and self.return_timestamps:
            #compute index
            weather_start = (self.start_times__[idx] - self.weather_dict['start_time'])//datetime.timedelta(minutes=30)
            wd = self.weather_dict['tensor']\
                [weather_start: weather_start + self.seq_len + self.pred_horz]
            return torch.cat((self.series[idx][:self.seq_len], wd[:self.seq_len]),1),\
                   torch.cat((self.series[idx][self.seq_len:], wd[self.seq_len:self.seq_len + self.pred_horz]),1),\
                   self.start_times[idx],\
                   self.pred_starttimes[idx]
        elif self.has_weather and (not self.return_timestamps):
            weather_start = (self.start_times__[idx] - self.weather_dict['start_time'])//datetime.timedelta(minutes=30)
            wd = self.weather_dict['tensor']\
                [weather_start: weather_start + self.seq_len + self.pred_horz]
            return torch.cat((self.series[idx][:self.seq_len], wd[:self.seq_len]),1),\
                self.series[idx][self.seq_len:]
                   #torch.cat((self.series[idx][self.seq_len:], wd[self.seq_len:self.seq_len + self.pred_horz]),1)
                   
        elif (not self.has_weather) and self.return_timestamps:
            return self.series[idx][:self.seq_len], self.series[idx][self.seq_len:], self.start_times[idx], self.pred_starttimes[idx]
        elif (not self.has_weather) and (not self.return_timestamps):
            return self.series[idx][:self.seq_len], self.series[idx][self.seq_len:]