from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams['figure.dpi'] = 125


# train SARIMA model and get validation set performance
def get_sarima_mae(y_tr, hp, y_te):
    try:
        model = SARIMAX(y_tr, order=(hp[0],hp[1],hp[2]), seasonal_order=(hp[3],hp[4],hp[5],12)).fit()
        y_hat = model.get_forecast(steps=len(y_te)).predicted_mean
        return np.mean(np.abs(y_hat - y_te))
    except:
        return None

def grid_search(train, validate):
    # define potential SARIMA hyerparameters
    print(train.head())
    p = d = q = P = D = Q = range(2)
    hp_list = list(product(p,d,q,P,D,Q))
    grid_search = pd.DataFrame(columns=['p','d','q','P','D','Q','mae'])
    # perform grid search
    for i, hp in enumerate(hp_list):
        mae = get_sarima_mae(train, hp, validate)
        if mae != None:
            params = {'p':hp[0], 'd':hp[1], 'q':hp[2], 'P':hp[3], 'D':hp[4], 'Q':hp[5], 'mae':mae}
            params_df = pd.DataFrame([params])
            grid_search = pd.concat([grid_search, params_df], ignore_index=True)

    # display best performing hyperparamters
    print(grid_search.sort_values('mae').head(1))
    return grid_search.sort_values('mae').head(1)

def sarima(train, validate, p,d,q, P, D, Q):
    # best hyperparameters from grid search
    best_monthly_order = (p,d,q)
    best_monthly_seas_order = (P,D,Q,12)

    # fit SARIMA model
    model_m = SARIMAX(train, order=best_monthly_order, seasonal_order=best_monthly_seas_order).fit()

    # get forecast and confidence interval for forecast
    forecast, pred, ci = sarima_forecast(model_m, validate)

    return model_m, forecast, pred, ci

def sarima_forecast(model_m, validate):
    # get forecast and confidence interval for forecast
    forecast = model_m.get_forecast(steps=len(validate))
    pred = pd.Series(forecast.predicted_mean, index=validate.index)
    ci = forecast.conf_int(alpha=0.05)

    return forecast, pred, ci
