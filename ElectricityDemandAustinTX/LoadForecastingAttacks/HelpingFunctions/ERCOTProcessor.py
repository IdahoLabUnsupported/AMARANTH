import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# --- helper functions for loading/processing ERCOT data ---

def convert_ts_02to16(r):
    """converts timestamp string to datetime index for the years 2002 to 2016"""
    t=str(r.year)+'-'+str(r.month)+'-'+str(r.day)+' '
    if r.minute > 30:
        t += str(r.hour)
    else:
        t += str(r.hour - 1)
    t += ':0'
    return pd.to_datetime(t)

def convert_ts_17(r):
    """converts timestamp string to datetime index for the year 2017"""
    h = int(str(r)[11:13])-1
    if h < 10:
        return pd.to_datetime(str(r)[:11]+'0'+str(h)+str(r)[13:])
    else:
        return pd.to_datetime(str(r)[:11]+str(h)+str(r)[13:])
    
def process_02to16(filename, load_col):
    """loads excel data into dataframe for the years 2002 to 2016"""
    df=pd.read_excel(filename, usecols='A,'+load_col, names=['time','load'])
    df['time'] = df['time'].apply(convert_ts_02to16)
    return df

def process_17(filename, load_col):
    """loads excel data into dataframe for the year 2017"""
    df=pd.read_excel(filename, usecols='A,'+load_col, names=['time','load'])
    df['time'] = df['time'].apply(convert_ts_17)
    return df


def fileprocessing(start_year, end_year):
    # setup list of filenames
    path = 'ElectricityDemandAustinTX/ercot_data/'
    suffix_02to14 = '_ercot_hourly_load_data.xls'
    fname_15 = 'native_Load_2015.xls'
    filenames = []
    if start_year <= 2002:
        for year in range(start_year, 2016):
            if year == 2015:
                filenames.append(path + 'native_Load_' + str(year) + '.xls')
            else:
                filenames.append(path + str(year) + '_ercot_hourly_load_data.xls')
            
    if start_year >= 2016:
        for year in range(start_year, end_year+1):
            filenames.append(path + 'native_Load_' + str(year) + '.xlsx')

    # load each file into a dataframe
    df_list = []
    year = start_year
    for filename in filenames:
        if year < 2017:
            df_list.append(process_02to16(filename, 'H'))
        else:
            df_list.append(process_17(filename, 'H'))
        year += 1
        
    # combine into one dataframe and set datetime index
    ercot_df = pd.concat(df_list)
    ercot_df.set_index('time', inplace=True)

    # fix missing/duplicate values
    ercot_df = ercot_df.groupby('time').mean()
    ercot_df = ercot_df.asfreq('H', method='pad')
    
    return ercot_df