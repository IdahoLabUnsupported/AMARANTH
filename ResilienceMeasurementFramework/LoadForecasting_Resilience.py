###############################################################################
# Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
# Written by Patience Yockey 06/12/2025
#
# Functions useful to calculating model drift within a load forecasting method
###############################################################################

import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor
from scipy.special import kl_div

def compute_mae(y, yhat):
    """ Given predicted and observed values, computes mean absolute error """
    return np.mean(np.abs(yhat - y))
    
def prediction_intervals(model, X_train, y_train, X_test, alpha):
    """ Use MAPIE to get prediction intervals for models who do not generate them """
    mapie_regressor = MapieRegressor(model)
    mapie_regressor.fit(X_train, y_train)
    
    y_pred, y_ci = mapie_regressor.predict(X_test, alpha)
    
    ci = pd.DataFrame()
    ci['ll'] = y_ci[0]
    ci['ul'] = y_ci[1]
    
    return ci

def get_deviation(ci, ci_te):
    """ Get the confidence interval width (c_x) right after training and validation and compare to the confidence interval width iterated to the point where the width of the interval exceeds the width of c_x, labeling it 'div' as the point at which a division happens. """
    for i, index_label in enumerate(ci_te.index):
        lb = ci_te.iloc[i,0]
        ub = ci_te.iloc[i,1]
        c_x = ci.iloc[-1,1]-ci.iloc[-1,0]
        if ub-lb > c_x:
            div = index_label
            break
        else:
            div = ci.index[-1]

    return div


def TTF(df, div):
    """ Get Time to Failure for a given point of deviation from KL-divergence, confidence interval estimation, or mean absolute error estimation """
    return div - df.index[0] 

def calculateKLDivergence(pk, qk):
    """ Calculate KL divergence using Scipy method. Current version assumes probability distributions pk and qk are given by the inputs of the model. This only works if the values input are normalized between 0 and 1. """
    # TODO: Add a method to calculate pk and qk regardless of scaling
    diverge = kl_div(pk,qk)
    
    return diverge

def calculatePSI(pk,qk):
    """ Calculates PSI assuming pk and qk are probability distributions between 0 and 1. """
    # TODO: Add a method to calculate pk and qk regardless of scaling
    psi = (pk-qk)*np.log(pk/qk)
    
    return psi

def testDivergence(diverge, test):
    """ Determine where KL divergence exceeds 0.05 and 0.125 to determine when medium drift occurs and high drift occurs """
    diverge.replace([np.inf, -np.inf], np.nan, inplace=True)
    diverge.dropna(inplace=True)

    """ Determine where the cumulative sum exceeds 0.1 and 0.25 respectively for psi calculations.
    If we assume D(P||Q) ~= D(Q||P), we can take the psi 0.1 and 0.25 and multiply it by two to figure out where the cummulative sum crosses the threshold of 0.1. """
    med_drift = diverge.index[diverge.cumsum().searchsorted(0.1/2)]
    high_drift = diverge.index[diverge.cumsum().searchsorted(0.25/2)]

    return med_drift, high_drift

def testPSI(psi, test):
    """ Determine where PSI exceeds 0.1 and 0.25 to determine when medium drift occurs and high drift occurs """
    psi.replace([np.inf, -np.inf], np.nan, inplace=True)
    psi.dropna(inplace=True)
    
     """ Determine where the cumulative sum exceeds 0.1 and 0.25 respectively for psi calculations. """
    med_drift = psi.index[psi.cumsum().searchsorted(0.1)]
    high_drift = psi.index[psi.cumsum().searchsorted(0.25)]

    return med_drift, high_drift

def testRMSE(sampled_pred, test_m, rmse_threshold):
    """ Determine where RMSE exceeds the threshold between the predicted values and original values """
    rmse_drift = np.sqrt(((sampled_pred - test_m).cumsum()**2)/len(test_m))
    rmse_point = rmse_drift.index[rmse_drift.searchsorted(rmse_threshold)]

    return rmse_drift, rmse_point
