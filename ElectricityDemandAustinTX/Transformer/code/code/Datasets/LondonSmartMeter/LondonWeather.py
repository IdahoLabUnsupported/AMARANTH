# -*- coding: utf-8 -*-
import datetime
import numpy as np
import scipy
import math

import copy

import torch
import pickle
import pgzip
import os

import sys

path = './LondonSmartMeter'
fname = 'weather_hourly_darksky.csv'

rows = []
with open(os.path.join(path,fname)) as ff:
    rows = ff.readlines()

header = rows[0]
rows = rows[1:]
for i in range(len(rows)):
    rows[i] = rows[i].split(',')
    tmp = [None]*14
    #visibility: float, windBearing: -(dir cosine, dir sine) +(r:0-360,r:-180,180)
    #This makes it so that at least 1 of the 2 angles is continuous
    tmp[0] = float(rows[i][0])
    tmp[1] = float(rows[i][1]) #math.cos(2*math.pi*(float(rows[i][1])/360))
    tmp[2] = (lambda x: x - (x>180)*360)(tmp[1]) #math.sin(2*math.pi*(float(rows[i][1])/360))
    
    #temperature: float
    tmp[3] = float(rows[i][2])
    
    #time
    tmp[4] = datetime.datetime.fromisoformat(rows[i][3])
    
    #dewPoint: float, pressure: float, apparentTemp: float, windSpeed: float
    tmp[5] = float(rows[i][4])
    try:
        tmp[6] = float(rows[i][5]) 
    except ValueError: 
        tmp[6] = float('nan')
    tmp[7] = float(rows[i][6])
    tmp[8] = float(rows[i][7])
    
    #precipType: {'rain': 1, 'snow': 0}
    tmp[9] = {'rain': 1, 'snow': 0}[rows[i][8]]
    
    #icon
    tmp[10] = {'wind': 0,
               'clear-night':0.1667,
               'partly-cloudy-night':0.3334,
               'fog':0.5001,
               'cloudy':0.6668,
               'partly-cloudy-day':0.8335,
               'clear-day':1}[rows[i][9]]
    #(relative) humidity: float
    tmp[11] = float(rows[i][10])
    
    #summary
    summ_str = rows[i][11]
    toks = summ_str.split(' and ')
    windlevel = 0
    cloudcover = 1

    for token in toks:
        try:
            windlevel = {'Breezy':0.5,'Windy':1}[token.strip()]
        except KeyError:
            pass
        
        try:
            cloudcover = {'Clear':1,
                          'Partly Cloudy':0.75,
                          'Mostly Cloudy':0.5,
                          'Overcast':0.25,
                          'Foggy':0}[token.strip()]
        except KeyError:
            pass
        
    tmp[12] = cloudcover; tmp[13] = windlevel
    rows[i] = tmp

rows.sort(key = lambda row: row[4])

# skips = []
# for i in range(len(rows)-1):
#     td = rows[i+1][4] - rows[i][4]
#     if td != datetime.timedelta(hours=1):
#         skips.append((i,rows[i],td))

start_time = rows[0][4]
end_time = rows[-1][4]

#Expand the dataset to include NaNs for missing timeslots
full_len = (end_time - start_time)//datetime.timedelta(hours=1) + 1
nr = [[float('nan')]*13 for i in range(full_len)]

#Strip the datetime field
for i in range(len(rows)):
    time = rows[i][4]
    j = (time - start_time)//datetime.timedelta(hours=1)
    
    nr[j] = rows[i][:4] + rows[i][5:]

nr = np.array(nr)

itrf = scipy.interpolate.interp1d(np.arange(len(nr)),nr,axis=0)
xr = itrf(np.arange(0,len(nr),0.5)[:-1])

for i in range(1,len(xr),2):
    ma = (xr[i-1][1] + xr[i][1] + xr[i+1][1])/3
    va = ((xr[i-1][1]-ma)**2 + (xr[i][1] - ma)**2 + (xr[i+1][1] - ma)**2)/3
    
    mb = (xr[i-1][2] + xr[i][2] + xr[i+1][2])/3
    vb = ((xr[i-1][2]-mb)**2 + (xr[i][2] - mb)**2 + (xr[i+1][2] - mb)**2)/3
    
    th = xr[i][1]
    if 1.33 * vb < va:
        th = xr[i][2]
    xr[i][1] = math.cos(2*math.pi*(th/360))
    xr[i][2] = math.sin(2*math.pi*(th/360))

for i in range(0,len(xr),2):
    th = xr[i][1]
    xr[i][1] = math.cos(2*math.pi*(th/360))
    xr[i][2] = math.sin(2*math.pi*(th/360))

weather = torch.tensor(xr)
dc = {'start_time':start_time,'tensor':weather}

with pgzip.open(os.path.join(path,"londonWeather.pkl.pgz"),'wb') as fw:
    pickle.dump(dc,fw)