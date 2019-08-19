import numpy as np
import pandas as pd
from math import sqrt


def extract_trend_component(insample_data):
    """
    Calculates a & b parameters of LRL
    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b


def seasonality_test(original_ts, ppy):
    """
    Seasonality test
    :param original_ts: time series
    :param ppy: periods per year
    :return: boolean value: whether the TS is seasonal
    """
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))
    return (abs(acf(original_ts, ppy))) > limit


def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS

    :param ts_init: the original time series
    :param window: window length
    :return: moving averages ts
    """
    ts_init = pd.Series(ts_init)
    if len(ts_init) % 2 == 0:
        ts_ma = ts_init.rolling(window, center=True).mean()
        ts_ma = ts_ma.rolling(window=2, center=True).mean()
        ts_ma = np.roll(ts_ma, -1)
    else:
        ts_ma = ts_init.rolling(window, center=True).mean()
    return ts_ma


def acf(data, k):
    """
    Autocorrelation function
    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)
    return float(s1 / s2)


def extract_seasonal_component(original_ts, ppy):
    """
    Calculates and returns seasonal indices
    :param original_ts: original data
    :param ppy: periods per year
    :return:
    """
    """
    # === get in-sample data
    original_ts = original_ts[:-out_of_sample]
    """
    if seasonality_test(original_ts, ppy):
        # print("seasonal")
        # ==== get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        # ==== get seasonality indices
        le_ts = original_ts * 100 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100)
        si = si / norm
    else:
        # print("NOT seasonal")
        si = np.full(ppy, 100)
    return si


def deseasonalize(ts_data, freq=7):
    # extract seasonal component only from the training data
    ts_seasonality_in = extract_seasonal_component(ts_data, freq)
    desea_ts_data = np.zeros(len(ts_data))
    for i in range(len(ts_data)):
        desea_ts_data[i] = ts_data[i] * 100 / ts_seasonality_in[i % freq]
    return desea_ts_data, ts_seasonality_in


def reseasonalize(desea_ts_data, ts_seasonality_in, shift, freq=7):
    # when reseasonalize, the seasonality should have different start idx.
    ts_data = np.zeros(len(desea_ts_data))
    for i in range(len(desea_ts_data)):
        ts_data[i] = desea_ts_data[i] * ts_seasonality_in[(i + shift) % freq] / 100
    return ts_data


def normalize(ts_data, denom):
    return ts_data / denom


def denormalize(ts_data, denom):
    return ts_data * denom


def post_process_results(ts_data, denom, ts_seasonality_in, shift, freq=7):
    return reseasonalize(denormalize(ts_data, denom=denom), ts_seasonality_in, shift=shift, freq=freq).ravel()
