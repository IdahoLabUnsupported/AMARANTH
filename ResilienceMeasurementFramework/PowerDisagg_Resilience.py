###############################################################################
# Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
# Written by Bradley Marx 08/27/2025
#
# Functions and classes used to calculate resilience metrics and simulate attacks for the seq2seq Power Disaggregation task.
###############################################################################

import numpy as np
from numba import njit, prange
from scipy.special import kl_div
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from matplotlib import pyplot as plt

from mapie.regression import SplitConformalRegressor


class KDE_Numba:

    """
    Credit goes entirely to the [KDE_Numba](https://github.com/ablancha/kde_numba/blob/master/README.md) repo
    
    Kernel Density Estimation with Numba
    Call KDE_Numba like you would stats.gaussian_kde"""

    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape

        if weights is not None:
            self._weights = np.atleast_1d(weights).astype(float)
            self._weights /= sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self._neff = 1/sum(self._weights**2)

        self.set_bandwidth(bw_method=bw_method)


    def evaluate(self, points):
        return self.kde_numba(points, np.squeeze(self.dataset), 
                              self.weights, self.bw)

    __call__ = evaluate


    @staticmethod
    @njit(parallel=True, cache=True)
    def kde_numba(x_eval, x_sampl, weights, bw):
        n = x_eval.shape[0]
        exps = np.zeros(n, dtype=np.float64)
        for i in prange(n):
            p = x_eval[i]
            d = (-(p-x_sampl)**2)/(2*bw**2)
            exps[i] += np.sum(np.exp(d)*weights)
        fac = np.sqrt(2*np.pi)*bw
        return exps/fac


    def scotts_factor(self):
        return np.power(self.neff, -1./(self.d+4))


    def silverman_factor(self):
        return np.power(self.neff*(self.d+2.0)/4.0, -1./(self.d+4))


    def set_bandwidth(self, bw_method=None):
        if bw_method is None:
            self.covariance_factor = self.scotts_factor
        elif bw_method == "scott":
            self.covariance_factor = self.scotts_factor
        elif bw_method == "silverman":
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method):
            self.covariance_factor = lambda: bw_method
        else:
            msg = "`bw_method` should be 'scott', 'silverman', or a scalar."
            raise ValueError(msg) 
        self.rescale_bandwidth()


    def rescale_bandwidth(self):
        cov = np.cov(self.dataset, 
                                           rowvar=1,
                                           bias=False,
                                           aweights=self.weights)
        self.bw = np.sqrt(cov)*self.covariance_factor()


    def pdf(self, x):
        return self.evaluate(x)


    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            self._weights = np.ones(self.n)/self.n
            return self._weights


    @property
    def neff(self):
        try:
            return self._neff
        except AttributeError:
            self._neff = 1/sum(self.weights**2)
            return self._neff


#### FUNCTIONS FOR CALCULATING AND PLOTTING MODEL DRIFT VIA KL DIVERGENCE ####


def kl_divergence_from_seqs(true_pred_df: pd.DataFrame, tgt_fields: list, train_cutoff, pred_fld_suffix: str = '_pred') -> pd.DataFrame:
    
    kl_by_day = []
    scaler = StandardScaler()
    # Generate predicted field names
    tgt_fields_pred = [tgt + pred_fld_suffix for tgt in tgt_fields]
    # Standardizing true and predicted values using the same scalar fit on true data over the training interval. 
    fit_set_true = scaler.fit_transform(true_pred_df.loc[:train_cutoff, tgt_fields].values)
    fit_set_pred = scaler.transform(true_pred_df.loc[:train_cutoff, tgt_fields_pred].values)
    
    # Transform all data with the fit scaler from before
    true_pred_df.loc[:, tgt_fields] = scaler.transform(true_pred_df.loc[:, tgt_fields].values)
    true_pred_df.loc[:, tgt_fields_pred] = scaler.transform(true_pred_df.loc[:, tgt_fields_pred].values)
    
    kdes = {}
    # Fit PDF for true and predicted daily values for each standardized channel
    for idx, ch in enumerate(tgt_fields):
        kdes[ch] = [KDE_Numba(fit_set_true[:, idx]), KDE_Numba(fit_set_pred[:, idx])]
    
    # Iterate over each day to convert sum Watt usage into probabilities and then probabilities into a 
    # KL Divergence scalar for that day
    for idx, vals in true_pred_df.groupby(true_pred_df.index.date):
        kl_df = pd.DataFrame()
        for idxs, chs in enumerate(tgt_fields):
            # calculate p(x) and q(x) using PDFs calculated earlier
            p_true = kdes[chs][0].evaluate(vals.loc[:, [chs]].values.flatten())
            p_pred = kdes[chs][1].evaluate(vals.loc[:, [chs + pred_fld_suffix]].values.flatten())
            # Calculate the DKL using the probabilities
            kl_df[f'{chs}_KL_divergence'] = kl_div(p_true, p_pred)
        kl_df = kl_df.replace(np.inf, 0.)
        kl_df.index = [idx]
        kl_by_day.append(kl_df.copy())
        
    kl_by_day = pd.concat(kl_by_day)

    return kl_by_day


def plot_drift(drift_df: pd.DataFrame, start_date, fld: str, fld_idx: int, medium_cutoff: float = 0.1/2, high_cutoff: float = 0.25/2) -> None:
    try:
        med_drift = drift_df.loc[:, [fld]].index[drift_df.iloc[:, fld_idx].cumsum().searchsorted(medium_cutoff)]
        plt.axvline(x=med_drift, color="green", alpha=0.5, linestyle="-.")
        plt.axvspan(start_date, med_drift, color='green', alpha=0.2, label="Low Drift")
        try:
            high_drift = drift_df.loc[:, [fld]].index[drift_df.iloc[:, fld_idx].cumsum().searchsorted(high_cutoff)]
            plt.axvspan(med_drift, high_drift, color='yellow', alpha=0.2, label="Moderate Drift")
            plt.axvline(x=high_drift, color="red", alpha=0.5, linestyle="-.")
            plt.axvspan(high_drift, drift_df.index[-1], color='red', alpha=0.2, label="High Drift")
        except:
            plt.axvspan(med_drift, drift_df.index[-1], color='yellow', alpha=0.2, label="Moderate Drift")
    except:
        plt.axvspan(start_date, drift_df.index[-1], color='green', alpha=0.2, label="Low Drift")


def plot_kl_drift(kl_df: pd.DataFrame, start_date, fld_names: list, display_names: list, medium_cutoff: float = 0.1/2, high_cutoff: float = 0.25/2) -> None:
    """
    Plots daily KL divergence from `start_date` onwards, along with drift intervals derived starting at that same time
    """
    # Iterate over each channel to render their respective drift visualizations
    for i, (fld, name) in enumerate(zip(fld_names, display_names)):
        kl_df.loc[start_date:, [fld]].plot.line(figsize=(20, 6), label=f'{name} KL Divergence')

        plot_drift(kl_df.loc[start_date:], start_date, fld, i, medium_cutoff, high_cutoff)
        # try:
        #     med_drift = kl_df.loc[start_date:, [fld]].index[kl_df.loc[start_date:].iloc[:, i].cumsum().searchsorted(medium_cutoff)]
        #     plt.axvline(x=med_drift, color="green", alpha=0.5, linestyle="-.")
        #     plt.axvspan(start_date, med_drift, color='green', alpha=0.2, label="Low Drift")
        #     try:
        #         high_drift = kl_df.loc[start_date:, [fld]].index[kl_df.loc[start_date:].iloc[:, i].cumsum().searchsorted(high_cutoff)]
        #         plt.axvspan(med_drift, high_drift, color='yellow', alpha=0.2, label="Moderate Drift")
        #         plt.axvline(x=high_drift, color="red", alpha=0.5, linestyle="-.")
        #         plt.axvspan(high_drift, kl_df.index[-1], color='red', alpha=0.2, label="High Drift")
        #     except:
        #         plt.axvspan(med_drift, kl_df.index[-1], color='yellow', alpha=0.2, label="Moderate Drift")
        # except:
        #     plt.axvspan(start_date, kl_df.index[-1], color='green', alpha=0.2, label="Low Drift")
    
        plt.title(f'Daily {name} KL Divergence')
        plt.ylabel('KL Divergence')
        plt.legend()
        plt.show()


#### FUNCTIONS AND CLASSES FOR MAPIE CONFIDENCE INTERVALS ####

# Custom estimator class to wrap around the Keras model, so that it can be integrated with Mapie
class TFWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, tf_model, channel_id: int):
        self.tf_model = tf_model
        self.channel_id = channel_id

    def fit(self, X, y):
        pass
        
    def predict(self, X):
        """ Predict output sequences for every channel, only select channel of interest, And then sum sequence into daily figures
        NOTE That this logic is made specifically for the power disaggregation model trained in this demo."""
        raw_out = self.tf_model.predict(X)['time_output_out']
        tgt_out = raw_out[:,:, self.channel_id]

        tgt_out[tgt_out < 0] = 0
        agg_out = np.sum(tgt_out, axis=1)
        return agg_out

    def __sklearn_is_fitted__(self):
        """
        Its already fitted.
        """
        return True


def calculate_mapie_cis_by_channel(model, full_data_raw: pd.DataFrame, train_cutoff_date, tgt_var_idx: int = -1, window_size: int = 1440) -> pd.DataFrame:
    
    mapie_cis = pd.DataFrame()
    
    for channel_id, name in enumerate(['Refrigerator', 'Washing Machine', 'Dishwasher', 'Solar Thermal Pump']):
        mapie_model = TFWrapper(model, channel_id)
        
        # sum minute readings up to day
        ch_train_data = full_data_raw.loc[:train_cutoff_date]
        conform_y = ch_train_data.iloc[:, channel_id].groupby(ch_train_data.index.date).agg('sum').values
    
        conform_x = ch_train_data.iloc[:, tgt_var_idx].values.reshape(-1, window_size)
    
        # consistency_check needs to be set to False, otherwise the system breaks. 
        mapie_regressor = SplitConformalRegressor(estimator=mapie_model, 
            confidence_level=0.95, prefit=True)
        mapie_regressor._conformity_score.consistency_check = False
        
        mapie_regressor = mapie_regressor.conformalize(conform_x, conform_y)
        
        all_y_df = full_data_raw.iloc[:, channel_id].groupby(full_data_raw.index.date).agg('sum')
        all_y = all_y_df.values
        all_x = full_data_raw.iloc[:, tgt_var_idx].values.reshape(tgt_var_idx, window_size)
        
        predicted_points, predicted_interval = mapie_regressor.predict_interval(all_x)
        
        scaler = StandardScaler()
        output_df = pd.DataFrame()
        # Fit standard scaler on channel signal over training period
        scaler.fit_transform(conform_y[:, np.newaxis])
        output_df['lower_bound'] = scaler.transform(predicted_interval[:, 0, :]).flatten()
        output_df['upper_bound'] = scaler.transform(predicted_interval[:, 1, :]).flatten()
        output_df.set_index(all_y_df.index, inplace=True)
        
        mapie_cis[f'{name} lower_bound'] = output_df['lower_bound']
        mapie_cis[f'{name} upper_bound'] = output_df['upper_bound']

    return mapie_cis


def plot_error_ci_drift(pred_true_df: pd.DataFrame, kl_df: pd.DataFrame, mapie_cis: pd.DataFrame, val_start_date, fld_names: list, display_names: list, medium_cutoff: float = 0.1/2, high_cutoff: float = 0.25/2, 
                        error_fld_suffix: str = '_deviation', pred_fld_suffix: str = '_pred', kl_fld_suffix: str = '_KL_divergence'):
    """
    Plots the standard-scaled true and predicted daily wattage for each requested channel, the corresponding confidence interval, and in the validation time interval, the low/medium/high drift boundaries.
    The incoming data should be at a day-level granularity, and the graphs show a 30-day moving average.
    """
    for i, (fld, name) in enumerate(zip(fld_names, display_names)):

        plt.figure(figsize=(20, 6))
        plt.plot(pred_true_df.loc[:, [fld+error_fld_suffix]].rolling(window=30).mean(), label='Error (Normalized)', color='black', ls='-')
        plt.plot(pred_true_df.loc[:, [fld]].rolling(window=30).mean(), label='True Daily Power Usage (Normalized)')
        plt.plot(pred_true_df.loc[:, [fld+pred_fld_suffix]].rolling(window=30).mean(), label='Predicted Daily Power Usage (Normalized)')
        plt.fill_between(mapie_cis.rolling(window=30).mean().index,
                         mapie_cis[f'{name} lower_bound'].rolling(window=30).mean(), 
                         mapie_cis[f'{name} upper_bound'].rolling(window=30).mean(), color='grey', alpha=0.6, label='95% Confidence Interval')
        plt.axhline(0, color='black')
    
        plot_drift(kl_df.loc[val_start_date:], val_start_date, fld+kl_fld_suffix, i)
    
        plt.title(f'Monthly {name} Pred/True Moving Average (Normalized)')
        plt.legend()
        plt.ylabel('Normalized Daily Watts')
        plt.show()


#### FUNCTIONS FOR WEIGHT MANIPULATION ATTACK SIMULATION ####

def perturb_weights(wgt_matrix: np.array, perturb_magnitude: float = 0.1, grad: np.array = None) -> np.array:
    '''
    Perturb input wgt_matrix by either randomly sampled values or from weight gradient.

    Perturbations represented as a vector of the same dimension as the input matrix (when flattened) that 
    has an L2 norm `perturb_magnitude` of the original matrix's L2 norm (again, when flattened into a vector)

    When this perturbation vector is substracted from the original weight vector, the resulting weight vector
    exists somewhere on the surface of an n-d 'ball' with radius `perturb_magnitude`*`weight vector L2 norm`, centered
    on the original weight vector.
    '''
    tgt_shape = wgt_matrix.shape
    start_norm = np.linalg.norm(wgt_matrix.flatten(), ord=2)
    if grad is None:
        rand_perturbs = np.random.randn(np.prod(tgt_shape))
    else:
        rand_perturbs = grad.flatten()
    # Normalize to have a norm of 1
    rand_perturbs /= np.linalg.norm(rand_perturbs, ord=2)
    # Scale to target size (perturb_magnitude of the norm of the original array
    rand_perturbs *= (start_norm * perturb_magnitude)
    # Reshape to original matrix 
    rand_perturbs = rand_perturbs.reshape(tgt_shape)

    # Subtract perturbations from original matrix
    new_weights = wgt_matrix - rand_perturbs

    return new_weights


def simulate_model_weight_attack(tgt_model, testing_data: pd.DataFrame, wgt_grads: list, perturb_schedule_lambda: float = 1.0, window_size: int = 1440, n_targets: int = 4) -> pd.DataFrame:

    # Collect metadata on target model
    orig_weights = tgt_model.get_weights()
    n_trainable_vars = len(tgt_model.trainable_variables)
    atk_intervals = []

    # Sample Poisson for number of weight matrices to change each day
    perturb_schedule = np.random.poisson(lam=perturb_schedule_lambda, size=testing_data.iloc[:, -1:].values.reshape(-1, window_size).shape[0])
    
    for n_perturbs, (tgt_day, vals) in zip(perturb_schedule, testing_data.groupby(testing_data.index.date)):
        # Ignore weight updates if no perturbations are scheduled for that day
        if n_perturbs > 0:
        
            new_wgt_list = []
            # Sample weight array indices to perturb
            wgts_to_change = np.random.choice(n_trainable_vars, size=n_perturbs, replace=False, p=None)
            # wgt_idx keeps track of the order of all weights, since non-trainable weights are also in the list being iterated.
            wgt_idx = 0
            for weight_array in tgt_model.weights:
                if not weight_array.trainable:
                    new_wgt_list.append(weight_array.numpy())
                elif wgt_idx in wgts_to_change:
                    
                    wgt_magnitude = np.random.uniform(0.005, 0.01)
                    new_wgt_list.append(perturb_weights(weight_array.numpy(), wgt_magnitude, -wgt_grads[wgt_idx].numpy()))
                    wgt_idx += 1
                else:
                    new_wgt_list.append(weight_array.numpy())
                    wgt_idx += 1
            # Weights are updated and used for every day following
            tgt_model.set_weights(new_wgt_list)
    
        # Make predictions with current state of attacked model for the current day.
        vals[[f'channel_{ch}_pred' for ch in [12, 5, 6, 3]]] = tgt_model.predict(vals.iloc[:, -1:].values.reshape(-1, window_size))['time_output_out'].reshape((-1, n_targets))
        atk_intervals.append(vals)

    # Reset model after simulation
    tgt_model.set_weights(orig_weights)
    atk_intervals = pd.concat(atk_intervals)
    return atk_intervals