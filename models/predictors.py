""" Predictors to predict networked popularity.
Naive, Seasonal Naive, Autogressive, RNN, and ARNet.
"""

import sys, os
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.metrics import symmetric_mean_absolute_percentage_error as smape
from utils.tsa import *

from scipy import optimize
from autograd import grad
import autograd.numpy as np


class Naive:
    def __init__(self, ts_data, num_output):
        self.ts_data = ts_data
        self.num_output = num_output
        self.pred_test_output = [self.ts_data[-self.num_output - 1]] * self.num_output

    def evaluate(self):
        true = self.ts_data[-self.num_output:]
        pred = self.pred_test_output
        return smape(true, pred)[0]


class SeasonalNaive:
    def __init__(self, ts_data, num_output):
        self.ts_data = ts_data
        self.num_output = num_output
        self.pred_test_output = self.ts_data[-2 * num_output: -num_output]

    def evaluate(self):
        true = self.ts_data[-self.num_output:]
        pred = self.pred_test_output
        return smape(true, pred)[0]


class AutoRegression:
    def __init__(self, ts_data, num_output):
        self.ts_data = ts_data
        self.t = len(self.ts_data)
        self.num_output = num_output
        self.model = None
        self.fitted_params = None
        self.pred_test_output = None

    def train_ar(self, lag):
        self.model = AR(self.ts_data[: -self.num_output]).fit(maxlag=lag, trend='nc')
        self.fitted_params = self.model.params[::-1]
        # forecast out-of-sample data by rolling the predicted values
        self.pred_test_output = self.model.predict(start=self.t - self.num_output, end=self.t - 1)

    def evaluate(self):
        true = self.ts_data[-self.num_output:]
        pred = self.pred_test_output
        return smape(true, pred)[0]


def arnet_predict(params, model_input, mode='train'):
    num_features, num_step = model_input.shape
    ar_coef = params[: -num_features + 1]
    num_input = len(ar_coef)
    link_weights = params[-num_features + 1:]

    yhat = None
    latent_interest = None

    if mode == 'train':
        for t in range(num_input, num_step):
            # AR part
            res = sum(ar_coef * model_input[0, t - num_input: t])
            if latent_interest is None:
                latent_interest = np.array([res])
            else:
                latent_interest = np.hstack((latent_interest, res))

            # network part
            for i in range(1, num_features):
                res += link_weights[i - 1] * model_input[i, t]

            # write in this way because auto grad does not support vector assignment
            if yhat is None:
                yhat = np.array([res])
            else:
                yhat = np.hstack((yhat, res))
    elif mode == 'test':
        input_views = model_input[0, :]
        for t in range(num_input):
            # AR part
            res = sum(ar_coef * input_views[-num_input:])
            if latent_interest is None:
                latent_interest = np.array([res])
            else:
                latent_interest = np.hstack((latent_interest, res))

            # network part
            for i in range(1, num_features):
                res += link_weights[i - 1] * model_input[i, t]

            # write in this way because auto grad does not support vector assignment
            if yhat is None:
                yhat = np.array([res])
            else:
                yhat = np.hstack((yhat, res))
            input_views = np.hstack((input_views, res))
    return yhat, latent_interest


def arnet_cost_function(params, model_input, model_output):
    yhat = arnet_predict(params, model_input)[0]
    # minimize SMAPE
    return np.mean(200 * np.nan_to_num(np.abs(model_output - yhat) / (np.abs(model_output) + np.abs(yhat))))


class ARNet:
    def __init__(self, tar_ts_data, src_ts_data_mat, num_input, num_output, num_ensemble):
        self.tar_ts_data = np.array(tar_ts_data)
        self.src_ts_data_mat = np.array(src_ts_data_mat)
        self.num_src = self.src_ts_data_mat.shape[0]
        self.t = len(self.tar_ts_data)
        self.num_input = num_input
        self.num_output = num_output
        self.num_ensemble = num_ensemble

        self.true_train_output = self.tar_ts_data[self.num_input: -self.num_output]
        self.true_test_output = self.tar_ts_data[-self.num_output:]

        self.pred_train_output = None
        self.pred_test_output = None
        self.link_weights = None
        self.network_ratio = None

        self.train_input = np.vstack((self.tar_ts_data[: -self.num_output].reshape(1, -1), self.src_ts_data_mat[:, : -self.num_output]))
        self.test_input = np.vstack((self.tar_ts_data[-self.num_output - self.num_input: -self.num_output].reshape(1, -1), self.src_ts_data_mat[:, -self.num_input:]))

    def train_arnet(self, start_params):
        arnet_autograd_func = grad(arnet_cost_function)
        arnet_bounds = [(0, 1)] * len(start_params) + [(0, 1)] * self.num_src

        iter_cnt = 0
        pred_train_output_mat = np.empty((0, len(self.true_train_output)), np.float)
        pred_test_output_mat = np.empty((0, self.num_output), np.float)
        link_weights_list = np.empty((0, self.num_src), np.float)
        network_ratio_list = []
        while iter_cnt < self.num_ensemble:
            arnet_init_values = np.array(start_params + [np.random.random()] * self.num_src)
            arnet_optimizer = optimize.minimize(arnet_cost_function, arnet_init_values, jac=arnet_autograd_func,
                                                method='L-BFGS-B',
                                                args=(self.train_input, self.true_train_output),
                                                bounds=arnet_bounds,
                                                options={'maxiter': 100, 'disp': False})
            arnet_fitted_params = arnet_optimizer.x
            # arnet_ar_coef = arnet_fitted_params[: self.num_input]
            arnet_link_weights = arnet_fitted_params[self.num_input:]

            arnet_pred_train, arnet_latent_train = arnet_predict(arnet_fitted_params, self.train_input, mode='train')
            arnet_pred_test, arnet_latent_test = arnet_predict(arnet_fitted_params, self.test_input, mode='test')

            pred_train_output_mat = np.vstack((pred_train_output_mat, arnet_pred_train))
            pred_test_output_mat = np.vstack((pred_test_output_mat, arnet_pred_test))
            link_weights_list = np.vstack((link_weights_list, arnet_link_weights))
            network_ratio_list.append(1 - sum(arnet_latent_train) / sum(arnet_pred_train))
            iter_cnt += 1

        self.pred_train_output = np.nanmean(pred_train_output_mat, axis=0)
        self.pred_test_output = np.nanmean(pred_test_output_mat, axis=0)
        self.link_weights = np.nanmean(link_weights_list, axis=0)
        self.network_ratio = np.mean(network_ratio_list)

    def evaluate(self):
        true = self.true_test_output
        pred = self.pred_test_output
        return smape(true, pred)[0]


def smape_loss(y_true, y_pred):
    return K.mean(200 * K.abs(y_pred - y_true) / (K.abs(y_pred) + K.abs(y_true)))


class TemporalLSTM:
    def __init__(self, ts_data, num_input, num_output, num_features, num_neurons, freq, num_ensemble):
        self.ts_data = np.array(ts_data)
        self.t = len(self.ts_data)
        self.num_input = num_input
        self.num_output = num_output
        self.num_features = num_features
        self.num_neurons = num_neurons
        self.freq = freq
        self.num_ensemble = num_ensemble
        self.num_sequence = self.t - self.num_output - self.num_input - (self.num_output - 1)
        self.len_train_output = self.t - self.num_output - self.num_input

        self.train_data = self.ts_data[: -self.num_output]
        self.true_train_output = self.ts_data[self.num_input: -self.num_output]
        self.true_test_output = self.ts_data[-self.num_output:]
        self.pred_train_output = None
        self.pred_test_output = None

        self.model = None
        self.history = None

        self.ts_seasonality_in = None
        self.train_input = np.zeros(shape=(self.num_sequence, self.num_input + 7, self.num_features))
        self.train_output = np.zeros(shape=(self.num_sequence, self.num_output, 1))
        self.test_input = np.zeros(shape=(1, self.num_input + 7, self.num_features))
        self.train_denom_list = [None for _ in range(self.num_sequence)]
        self.test_denom = None

    def prepare_tensor(self):
        # extract seasonality cycle from all the training data
        # deseasonalize
        desea_ts_data, self.ts_seasonality_in = deseasonalize(self.train_data, freq=self.freq)
        desea_ts_data = np.array(desea_ts_data).reshape(-1, 1)

        for i in range(self.num_sequence):
            train_input_and_output = desea_ts_data[i: i + self.num_input + self.num_output]
            # use the last observation in train input to normalize the whole sequence
            train_denom = train_input_and_output[self.num_input - 1]
            self.train_denom_list[i] = train_denom
            train_input_and_output = normalize(train_input_and_output, train_denom)
            # feature: dow
            dow = np.zeros(shape=(7, 1))
            # first observation day in the first windows is Sat
            dow[(i + 5) % self.freq] = 1
            self.train_input[i] = np.vstack((train_input_and_output[: self.num_input], dow))
            self.train_output[i] = train_input_and_output[self.num_input:]

        test_input = desea_ts_data[self.len_train_output: self.t - self.num_output]
        # use the last observation in test input to normalize the whole sequence
        self.test_denom = test_input[self.num_input - 1]
        test_input = normalize(test_input, self.test_denom)
        # feature: dow
        dow = np.zeros(shape=(7, 1))
        # first observation day is Sat
        dow[5] = 1
        self.test_input = np.vstack((test_input, dow))[np.newaxis, :]

    def create_model(self):
        """ creates, compiles and returns a LSTM model
        """
        self.model = Sequential()
        self.model.add(LSTM(units=self.num_neurons, input_shape=(self.num_input + 7, self.num_features), return_sequences=False))
        self.model.add(Dropout(0.1))
        self.model.add(RepeatVector(self.num_output))
        self.model.add(LSTM(units=self.num_neurons, return_sequences=True))
        self.model.add(Dropout(0.1))
        self.model.add(TimeDistributed(Dense(1)))

        self.model.compile(loss=smape_loss, optimizer='adam')

    def train_lstm(self):
        num_epochs = 100
        # sanity check: check the shape of train input, train output, and test input
        # print('shape of train input: {0}, train output: {1}, test input: {2}'.format(self.train_input.shape, self.train_output.shape, self.test_input.shape))

        callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
        iter_cnt = 0
        pred_train_output_mat = np.empty((0, self.len_train_output), np.float)
        pred_test_output_mat = np.empty((0, self.num_output), np.float)
        while iter_cnt < self.num_ensemble:
            self.history = self.model.fit(self.train_input, self.train_output, validation_split=0.15, shuffle=False,
                                          batch_size=1, epochs=num_epochs, callbacks=callbacks, verbose=0)

            # get the predicted train output
            pred_train_output = self.model.predict(self.train_input)
            pred_train_output_mat = np.zeros(shape=(self.num_output, self.len_train_output), dtype=np.float)
            pred_train_output_mat.fill(np.nan)
            for i in range(self.num_sequence):
                seq_pred_train_output = post_process_results(pred_train_output[i],
                                                             denom=self.train_denom_list[i],
                                                             ts_seasonality_in=self.ts_seasonality_in,
                                                             shift=i,
                                                             freq=self.freq).ravel()
                pred_train_output_mat[i % self.num_output, i: i + self.num_output] = seq_pred_train_output
            iter_pred_train_output = np.nanmean(pred_train_output_mat, axis=0)
            iter_train_smape, _ = smape(self.true_train_output, iter_pred_train_output)
            if iter_train_smape < 150:
                # get the predicted test output
                iter_pred_test_output = self.model.predict(self.test_input).ravel()
                iter_pred_test_output = post_process_results(iter_pred_test_output,
                                                             denom=self.test_denom,
                                                             ts_seasonality_in=self.ts_seasonality_in,
                                                             shift=self.len_train_output,
                                                             freq=self.freq).ravel()

                pred_train_output_mat = np.vstack((pred_train_output_mat, iter_pred_train_output))
                pred_test_output_mat = np.vstack((pred_test_output_mat, iter_pred_test_output))
                iter_cnt += 1

        self.pred_train_output = np.nanmean(pred_train_output_mat, axis=0)
        self.pred_test_output = np.nanmean(pred_test_output_mat, axis=0)

    def evaluate(self):
        true = self.true_test_output
        pred = self.pred_test_output
        return smape(true, pred)[0]

    # def plot(self):
    #     fig, ax1 = plt.subplots(nrows=1, ncols=1)
    #     ax1.plot(np.arange(1, len(self.ts_data) + 1), self.ts_data, 'k--', label='true')
    #     train_smape, _ = smape(self.true_train_output, self.pred_train_output)
    #     ax1.plot(np.arange(1 + self.num_input, len(self.ts_data) + 1 - self.num_output), self.pred_train_output, 'b-o', label='train LSTM: {0:.2f}'.format(train_smape))
    #     test_smape, _ = smape(self.true_test_output, self.pred_test_output)
    #     ax1.plot(np.arange(len(self.ts_data) + 1 - self.num_output, len(self.ts_data) + 1), self.pred_test_output, 'r-o', label='test LSTM: {0:.2f}'.format(test_smape))
    #     for i in range(1, 9):
    #         ax1.axvline(x=7 * i + .5, color='g', linestyle='--', lw=1.5, zorder=30)
    #     ax1.set_xlabel('time index')
    #     ax1.set_ylabel('views')
    #     ax1.legend(frameon=False)
    #
    #     # ax2.plot(self.history.history['loss'], label='loss')
    #     # ax2.plot(self.history.history['val_loss'], label='val_loss')
    #     # ax2.set_xlabel('epoch')
    #     # ax2.set_ylabel('loss')
    #     # ax2.legend(frameon=False)
    #
    #     plt.show()
