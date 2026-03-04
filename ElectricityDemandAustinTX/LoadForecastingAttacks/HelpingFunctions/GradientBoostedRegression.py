from sklearn.ensemble import GradientBoostingRegressor
import ForecastingHelpers
import numpy as np
import pandas as pd

def forecast(model, exog, y_init):
    """given a trained model, exogenous features, and initial AR term, makes forecasting predictions"""
    yhat = []
    Xi_te = np.hstack([y_init, exog[0]])[None,:]
    for i in range(len(exog)-1):
        yhat_i = model.predict(Xi_te)[0]
        yhat.append(yhat_i)
        Xi_te = np.hstack([yhat_i, exog[i+1]])[None,:]
    yhat.append(model.predict(Xi_te)[0])
    return np.array(yhat)

def weekly_forecast(model, exog, y_init):
    """given a trained model exogenous features, and initial AR term, makes a series of 1-week-out forecasts"""
    yhat = []
    for i, yi in enumerate(y_init):
        exog_i = exog[168*i:168*(i+1),:]
        if exog_i.shape[0] < 1:
            break
        yhat.append(forecast(model, exog_i, yi))
    return np.hstack(yhat)

# train GBR model, and get validation set performance
def get_gbr_mae(lr, ne, md, X_tr, y_tr, exog_val, y_init_val, y_val):
    mod = GradientBoostingRegressor(learning_rate=lr, n_estimators=ne, max_depth=md)
    mod.fit(X_tr, y_tr)
    pred_val = weekly_forecast(mod, exog_val, y_init_val)
    return ForecastingHelpers.compute_mae(y_val, pred_val)

def grid_search(X_tr, y_tr, exog_val, y_init_val, y_val):
    # define potential sets of hyperparameters
    learning_rate = [0.01, 0.1, 1.]
    n_estimators = [100, 500, 1000]
    max_depth = [2, 3, 4]
    grid_search = pd.DataFrame(columns=['lr','ne','md','mae'])

    # perform grid search
    for lr in learning_rate:    
        for ne in n_estimators:        
            for md in max_depth:            
                mae = get_gbr_mae(lr, ne, md, X_tr, y_tr, exog_val, y_init_val, y_val)
                params = {'lr':lr, 'ne':ne, 'md':md, 'mae':mae} 
                params_df = pd.DataFrame([params])
                grid_search = pd.concat([grid_search, params_df], ignore_index=True)

    # display best hyperparameters based on grid search
    return grid_search.sort_values('mae').head(1)