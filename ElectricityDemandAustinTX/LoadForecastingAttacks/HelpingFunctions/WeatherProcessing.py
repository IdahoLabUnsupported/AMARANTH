import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getData(full_data, start_year, end_year):
    full_data['DATE'] = pd.to_datetime(full_data['DATE'])
    data = full_data.loc[(full_data.DATE.dt.year >= start_year) & (full_data.DATE.dt.year <= end_year)]
    data.to_csv("weather_data_"+str(start_year)+"to"+str(end_year)+".csv")
    return data

def weather_processing(df):
    df.loc[df['HourlyPresentWeatherType'].str.split(" ").str[0].str.contains('00') == True, 'Condition']= 0
    df.loc[df['HourlyPresentWeatherType'].str.split(" ").str[0].str.contains('SN|PL|FZ') == True, 'Condition'] = 6
    df.loc[df['HourlyPresentWeatherType'].str.split(" ").str[0].str.contains('RA|UP') == True, 'Condition'] = 3
    df.loc[df['HourlyPresentWeatherType'].str.split(" ").str[0].str.contains('BR|HZ|FG|FU|DU') == True, 'Condition'] = 1
    df.loc[df['HourlyPresentWeatherType'].str.split(" ").str[0].str.contains('TS') == True, 'Condition'] = 14
    df.loc[df['HourlyPresentWeatherType'].str.split(" ").str[0].str.contains('BL|41') == True, 'Condition'] = 9
    return df

# Get one day for weather
def one_day(data, dy, mnth, yr):
    day_weather = data.loc[(data.DATE.dt.year == yr) & (data.DATE.dt.month == mnth) & (data.DATE.dt.day == dy)]
    day_weather = day_weather[["DATE", "HourlyDryBulbTemperature", "HourlyPrecipitation", "HourlyPresentWeatherType", "HourlyRelativeHumidity", "HourlyWindSpeed"]]
    if len(day_weather) != 0:
        day_weather = weather_processing(day_weather)
        #print(day_weather['Condition'].unique().tolist())
        if len(day_weather['Condition']) > 1:
            mode_weather = day_weather['Condition'].mode()
            day_weather['Condition'] = day_weather['Condition'].mode()[0]
    day_weather["HourlyDryBulbTemperature"] = day_weather["HourlyDryBulbTemperature"].astype('float64', errors='ignore')
    day_weather["HourlyRelativeHumidity"] = day_weather["HourlyRelativeHumidity"].astype('float64', errors='ignore')
    day_weather["HourlyWindSpeed"] = day_weather["HourlyWindSpeed"].astype('float64', errors='ignore')
    day_weather.drop("HourlyPrecipitation",axis=1,inplace=True)
    day_weather.drop("HourlyPresentWeatherType",axis=1,inplace=True)
    day_weather = day_weather.rename(columns={"DATE" : "Time", "HourlyDryBulbTemperature" : "Temp(f)", "HourlyRelativeHumidity" : "Humidity", "HourlyWindSpeed" : "Wind Speed", "Condition" : "Condition"})
    weather1 = day_weather.copy()
    weather1 = weather1.set_index('Time')
    weather1.index = pd.to_datetime(weather1.index)
    weather1 = weather1.resample('h').mean(numeric_only=True)
    weather1 = weather1.reset_index()
    date = pd.Timestamp(str(yr) + '-' + str(mnth) + '-' + str(dy))
    nxt_date = date + pd.DateOffset(days=1)
    return day_weather, weather1, nxt_date
    

def all_days(data, day, month, year, no_of_days):
    """crawls through a given number of days to collect weather data"""
    list1 = []
    list2 = []
    result = one_day(data, day, month, year)
    list1.append(result[0])
    list2.append(result[1])
    
    for i in range(no_of_days -1):
        if result != None:
            result = one_day(data, result[2].day, result[2].month, result[2].year)
            list1.append(result[0])
            list2.append(result[1])
            
    weather1 = pd.concat(list1).reset_index(drop=True)
    weather2 = pd.concat(list2).reset_index(drop=True)
    return weather1, weather2


def t_or_e(x):
    """process irregular wind speed data"""
    try:
        return float(x)
    except:
        if x=='Calm':
            return 3.0
        else:
            return 4.5


def process(df):
    """general purpose weather data processing"""
    #df.dropna(inplace=True)
    y = df.Condition.unique().tolist()
    y = sorted(y)
    #df['Condition'] = df['Condition'].apply(lambda x: y.index(x))
    df['Wind Speed'] = df['Wind Speed'].apply(lambda x: t_or_e(x))
    df['Time'] = df['Time'].apply(lambda x: pd.to_datetime(x))
    df['Humidity'] = df['Humidity'].fillna(df['Humidity'].mean())
    df['Humidity'] = df['Humidity'].astype('float64')
    df['Humidity'] = df['Humidity'].apply(lambda x: x/100)
    df = df.set_index('Time', drop=True).resample('h').mean().reset_index()
    return df


def one_year(data, year):
    """processes an entire year of weather data"""
    all_months = list(range(1,13))
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    if year%4 == 0:
        days_in_month[1] = 29
    Month = list(zip(all_months,days_in_month))
    
    all_months_1 = []
    all_months_2 = []
    for i in Month:
        res = all_days(data, 1, i[0], year ,i[1])
        all_months_1.append(res[0])
        all_months_2.append(res[1])

    all_Months_1 = pd.concat(all_months_1).reset_index(drop=True)
    all_Months_2 = pd.concat(all_months_1).reset_index(drop=True)
    all_Months_1 = process(all_Months_1)
    return all_Months_1, all_Months_2


def all_years(data, list_of_years, filename):
    """for a given set of years, collect all weather data"""
    Weather_data = []
    for k in list_of_years:
        Weather_data.append(one_year(data, k)[0])
    Weather_data_all = pd.concat(Weather_data).reset_index(drop=True)
    Weather_data_all.dropna(inplace=True)
    #Weather_data_all.drop('Condition',axis=1, inplace=True)
    Weather_data_all.to_csv(filename)