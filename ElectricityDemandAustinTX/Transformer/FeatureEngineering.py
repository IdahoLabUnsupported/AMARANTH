###############################################################################
# Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
# Written by Bradley Marx 08/27/2025
#
# Functions and classes used to engineer features for transformer manipulation
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# helper functions for defining one hot encoding
def one_hot_encoding(weather_df):
    def tstorm(x):
        if x==14 or x==15 or x==7 or x==4:
            return 1
        return 0

    def clear(x):
        if x==0:
            return 1
        return 0

    def fog(x):
        if x==1 or x==2 or x==8:
            return 1
        return 0

    def rain(x):
        if x==3 or x==5 or x==12:
            return 1
        return 0

    def cloud(x):
        if x==9 or x==10 or x==11 or x==13:
            return 1
        return 0

    def snow(x):
        if x==6:
            return 1
        return 0

    # create one hot encoding
    weather_df['tstorm'] = weather_df['cond'].apply(tstorm)
    weather_df['clear'] = weather_df['cond'].apply(clear)
    weather_df['fog'] = weather_df['cond'].apply(fog)
    weather_df['rain'] = weather_df['cond'].apply(rain)
    weather_df['cloud'] = weather_df['cond'].apply(cloud)
    weather_df['snow'] = weather_df['cond'].apply(snow)

    # drop original column
    weather_df.drop('cond', axis=1, inplace=True)

    return weather_df

def merge_ercot(weather_df, ercot_df):
    # merge and set index
    df = ercot_df.merge(weather_df, how='left', left_index=True, right_on='time')
    df.set_index('time', inplace=True)

    # fix missing values
    df['tstorm'] = df['tstorm'].bfill()
    df['clear'] = df['clear'].bfill()
    df['fog'] = df['fog'].bfill()
    df['rain'] = df['rain'].bfill()
    df['cloud'] = df['cloud'].bfill()
    df['snow'] = df['snow'].bfill()
    df.interpolate(inplace=True)
    df.bfill(inplace=True)
    
    return df


def temporal_variables(df):
    df['weekday'] = (df.index.weekday < 5).astype(int)
    df['sin_hour'] = np.sin(2*np.pi*df.index.hour.values/24)
    df['cos_hour'] = np.cos(2*np.pi*df.index.hour.values/24)

    return df

def normalize(df):
    df_norm = df.copy()

    load_min, load_max = df['load'].min(), df['load'].max()
    temp_min, temp_max = df['temp'].min(), df['temp'].max()
    wnsp_min, wnsp_max = df['wnsp'].min(), df['wnsp'].max()

    df_norm['load'] = (df['load'] - load_min) / (load_max - load_min)
    df_norm['temp'] = (df['temp'] - temp_min) / (temp_max - temp_min)
    df_norm['wnsp'] = (df['wnsp'] - wnsp_min) / (wnsp_max - wnsp_min)

    return df_norm

def feature_engineering(weather_df, ercot_df):
    weather_df = one_hot_encoding(weather_df)
    df = merge_ercot(weather_df, ercot_df)
    df = temporal_variables(df)
    df_norm = normalize(df)

    return df_norm


def train_test_split(df, start_year, vyear1, vyear2, end_year):
    train = df[:str(start_year)]
    validate = df[str(vyear1):str(vyear2)]
    test = df[str(end_year):]
    
    return train, validate, test
    

    