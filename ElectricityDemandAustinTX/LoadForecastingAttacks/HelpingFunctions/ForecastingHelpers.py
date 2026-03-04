import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams['figure.dpi'] = 125
import os
os.chdir("../..")
from HelpingFunctions import FeatureEngineering
from mapie.regression import MapieRegressor
from statsmodels.tsa.api import ExponentialSmoothing
from HelpingFunctions import sarima_modeling
from scipy.stats import entropy
from scipy.special import kl_div

def compute_mae(y, yhat):
    """given predicted and observed values, computes mean absolute error"""
    return np.mean(np.abs(yhat - y))

def forecast(model, exog, y_init, alpha):
    """given a trained model, exogenous features, and initial AR term, makes forecasting predictions"""
    yhat = []
    y_ci = []
    Xi_te = np.hstack([y_init, exog[0]])[None,:]
    for i in range(len(exog)-1):
        yhat_i, y_ci_i = model.predict(Xi_te, alpha=alpha)
        yhat.append(yhat_i)
        y_ci.append((y_ci_i))
        #print(y_ci)
        Xi_te = np.hstack([yhat_i, exog[i+1]])[None,:]
    yhat_i, y_ci_i = model.predict(Xi_te, alpha=alpha)
    yhat.append(yhat_i)
    y_ci.append((y_ci_i))
    return np.array(yhat), np.array(y_ci)

def weekly_forecast(indexes, model, exog, y_init, alpha):
    """given a trained model exogenous features, and initial AR term, makes a series of 1-week-out forecasts"""
    yhat = []
    y_ci = []
    for i, yi in enumerate(y_init):
        exog_i = exog[168*i:168*(i+1),:]
        if exog_i.shape[0] < 1:
            break
        y_hat_i, y_ci_i = forecast(model, exog_i, yi, alpha=alpha)
        #print(str(y_hat_i.shape) + " , " + str(y_ci_i.shape))
        yhat.append(y_hat_i)
        y_ci.append((y_ci_i))
    mapie_hat = pd.DataFrame(np.vstack(yhat).reshape(-1))
    mapie_ci = pd.DataFrame(np.vstack(y_ci).reshape((np.vstack(yhat).shape[0],2)), index=indexes, columns=['Lower Bounds', 'Upper Bounds'])
    #mapie_ci.set_index(indexes, inplace=True)
    return mapie_hat.values.ravel(), mapie_ci


def plot_full(y, yhat, yhat_idx, ci):
    """plots observed and forecasted values for the full date range"""
    pred = pd.Series(yhat, index=yhat_idx)
    plt.figure()
    plt.plot(y, label='Observed')
    plt.plot(pred, label='Forecast', ls='--')
    plt.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1],
                 color='k', alpha=0.2, label='95% Conf Int')
    plt.ylabel('Normalized Hourly Residual Electricity Demand')
    plt.legend()
    
def plot_week(y, yhat, yhat_idx):
    """plots observed and forecasted values for one week span"""
    pred = pd.Series(yhat, index=yhat_idx)
    fig = plt.figure();
    plt.plot(y['2016-12-01':'2016-12-07'], label='Observed')
    plt.plot(pred['2016-12-01':'2016-12-07'], label='Forecast', ls='--')
    plt.ylabel('Normalized Hourly Residual Electricity Demand')
    plt.legend()
    fig.autofmt_xdate()

def hourlyresiduals(df_norm):
    hourly_res_norm = df_norm.copy()
    hourly_res_norm['load'] = df_norm['load'].groupby(pd.Grouper(freq='M')).transform(lambda x: x - x.mean())
    return hourly_res_norm
    
def prediction_intervals(model, X_train, y_train, X_test, alpha):
    #Use MAPIE to get prediction intervals for models who do not generate them
    mapie_regressor = MapieRegressor(model)
    mapie_regressor.fit(X_train, y_train)
    
    y_pred, y_ci = mapie_regressor.predict(X_test, alpha)
    
    ci = pd.DataFrame()
    ci['ll'] = y_ci[0]
    ci['ul'] = y_ci[1]
    return ci

def get_deviation(ci, preds):
    div = 0
    load_div = 0
    for i, index_label in enumerate(ci.index):
        lb = ci.iloc[i,0]
        ub = ci.iloc[i,1]
        if preds[i] > ub or preds[i] < lb:
            div = index_label
            load_div = preds[i]
            return div, load_div
            break
        else:
            return 0


def MTTF(df, div):
    return div - df.index[0] 

def calculateEntropy(norms, preds, end_year):
    base = 10
    pk = norms[preds.index[0]:str(end_year)]
    qk = preds[preds.index[0]:str(end_year)]

    H = entropy(pk, base=base)
    M = entropy(qk, base=base)
    D = entropy(pk, qk, base=base)

    return pk, qk, H,M,D

def calculateKLDivergence(pk, qk):
    diverge = kl_div(pk,qk)
    return diverge

def calculatePSI(pk,qk):
    psi = (pk-qk)*np.log(pk/qk)
    return psi

def testDivergence(diverge, test):
    diverge.replace([np.inf, -np.inf], np.nan, inplace=True)
    diverge.dropna(inplace=True)
    
   # if diverge.sum() <= 0.1/2:
        # No significant drift over the testing time detected
   #     return test.index[-1],test.index[-1]

    # determine where the cumulative sum exceeds 0.1 and 0.25 respectively for psi calculations.
    # if we assume D(P||Q) ~= D(Q||P), we can take the psi 0.1 and 0.25 and multiply it by two to figure out where the cummulative sum crosses the threshold of 0.1.
    try:
        med_drift = diverge.index[diverge.cumsum().searchsorted(0.1/2)]
    except:
        med_drift = np.inf
    try:
        high_drift = diverge.index[diverge.cumsum().searchsorted(0.25/2)]
    except:
        high_drift = np.inf

    return med_drift, high_drift

def testPSI(psi, test):
    psi.replace([np.inf, -np.inf], np.nan, inplace=True)
    psi.dropna(inplace=True)
    
    #if psi.cumsum() <= 0.1:
        # No significant drift over the testing time detected
    #    return test.index[-1],test.index[-1]

    # determine where the cumulative sum exceeds 0.1 and 0.25 respectively for psi calculations.
    # if we assume D(P||Q) ~= D(Q||P), we can take the psi 0.1 and 0.25 and divide it by two to figure out where the cummulative sum crosses the threshold of 0.1.
    med_drift = psi.index[psi.cumsum().searchsorted(0.1)]
    high_drift = psi.index[psi.cumsum().searchsorted(0.25)]

    return med_drift, high_drift

def plot_full_w_div(y, yhat, yhat_idx, ci, div):
    """plots observed and forecasted values for the full date range"""
    pred = pd.Series(yhat, index=yhat_idx)
    plt.figure()
    plt.plot(y, label='Observed')
    plt.plot(pred, label='Forecast', ls='--')
    plt.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1],
                 color='k', alpha=0.2, label='95% Conf Int')
    plt.axvline(x=div, color="black", alpha=0.5, linestyle="--")
    plt.ylabel('Normalized Hourly Residual Electricity Demand')
    plt.legend()

def predictKL(diverge):
    med_drift, high_drift = testDivergence(diverge, diverge)
    while med_drift == np.inf or high_drift == np.inf:
        # Iterate fit until predicted diverge = med_drift
        drift_diverge, diverge_forec, diverge_pred, diverge_ci = sarima_modeling.sarima(diverge, diverge, 1, 1, 1, 0, 1, 1)
        forecast = drift_diverge.get_forecast(steps=10)
        pred = pd.Series(forecast.predicted_mean, index=forecast.predicted_mean.index)
        diverge = pd.concat([diverge, pred])
        med_drift, high_drift = testDivergence(diverge, pred)

    return high_drift, med_drift, diverge
