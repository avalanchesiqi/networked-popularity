import numpy as np


def symmetric_mean_absolute_percentage_error(true, pred):
    # percentage error, zero if both true and pred are zero
    true = np.array(true)
    pred = np.array(pred)
    daily_smape_arr = 200 * np.nan_to_num(np.abs(true - pred) / (np.abs(true) + np.abs(pred)))
    return np.mean(daily_smape_arr), daily_smape_arr
