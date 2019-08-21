#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Forecast view series in the last week (Sat, 2018-10-27 - Fri, 2018-11-02).
Training period: 2018-09-01 - 2018-10-26 (8 weeks, 56 days)

Usage: python forecast_next_week.py
Input data files: ../data/vevo_forecast_data_60k.csv, ../data/persistent_network.csv
Output data files: ./model_results/forecast_tracker_*.log
Time: ~1M
"""

import sys, os, json, gc
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.data_loader import DataLoader
from models.predictors import *


def main():
    # == == == == == == Part 1: Set up environment == == == == == == #
    data_prefix = '../data/'
    partition_dirname = './video_partition'
    result_dirname = './model_results'

    # == == == == == == Part 2: Load target videos set == == == == == == #
    tar_inlink_dict = defaultdict(list)
    with open(os.path.join(data_prefix, 'persistent_network.csv'), 'r') as fin:
        fin.readline()
        for line in fin:
            src_embed, tar_embed = map(int, line.rstrip().split(','))
            tar_inlink_dict[tar_embed].append(src_embed)
    tar_embed_list = list(sorted(tar_inlink_dict.keys()))

    # == == == == == == Part 3: Load target videos set == == == == == == #
    # partition tar embed into small files, each file contains 10 videos
    if not os.path.exists(partition_dirname):
        os.makedirs(partition_dirname)
        num_embed_in_partition = 50
        partition_idx = 1
        partition_fout = open(os.path.join(partition_dirname, 'partition_{0:03d}.txt'.format(partition_idx)), 'w')
        embed_cnt = 0
        for embed in tar_embed_list:
            if embed_cnt == num_embed_in_partition:
                partition_idx += 1
                partition_fout.close()
                partition_fout = open(os.path.join(partition_dirname, 'partition_{0:03d}.txt'.format(partition_idx)), 'w')
                embed_cnt = 0
            partition_fout.write('{0}\n'.format(embed))
            embed_cnt += 1
        partition_fout.close()

    if not os.path.exists(result_dirname):
        os.makedirs(result_dirname)

    load_partition = True
    visited_embed_set = set()
    if load_partition:
        partition_idx = int(sys.argv[1])
        result_filename = os.path.join(result_dirname, 'forecast_tracker_{0:03d}.json'.format(partition_idx))
        if os.path.exists(result_filename):
            with open(result_filename, 'r') as fin:
                for line in fin:
                    result_json = json.loads(line.rstrip())
                    visited_embed_set.add(result_json['embed'])

        partition_filename = os.path.join(partition_dirname, 'partition_{0:03d}.txt'.format(partition_idx))
        tar_embed_list = []
        with open(partition_filename, 'r') as fin:
            for line in fin:
                embed = int(line.rstrip())
                if embed not in visited_embed_set:
                    tar_embed_list.append(embed)
    else:
        result_filename = os.path.join(result_dirname, 'forecast_tracker_all.json')
    fout = open(result_filename, 'a')
    print('{0} videos to model'.format(len(tar_embed_list)))

    # == == == == == == Part 4: Load video views == == == == == == #
    data_loader = DataLoader()
    data_loader.load_video_views()
    embed_view_dict = data_loader.embed_view_dict

    # == == == == == == Part 5: Start prediction task == == == == == == #
    for item_cnt, tar_embed in enumerate(tar_embed_list):
        timer = Timer()
        timer.start()

        tar_ts_data = embed_view_dict[tar_embed]
        true_value = tar_ts_data[-NUM_OUTPUT:]

        # naive method
        naive_model = Naive(tar_ts_data, num_output=NUM_OUTPUT)
        naive_smape = naive_model.evaluate()
        naive_pred = naive_model.pred_test_output

        # seasonal naive method
        snaive_model = SeasonalNaive(tar_ts_data, num_output=NUM_OUTPUT)
        snaive_smape = snaive_model.evaluate()
        snaive_pred = snaive_model.pred_test_output

        # autoregressive method
        ar_model = AutoRegression(tar_ts_data, num_output=NUM_OUTPUT)
        ar_model.train_ar(lag=FREQ)
        ar_smape = ar_model.evaluate()
        ar_pred = list(map(int, ar_model.pred_test_output))

        # RNN with LSTM units
        rnn_model = TemporalLSTM(tar_ts_data,
                                 num_input=NUM_INPUT, num_output=NUM_OUTPUT,
                                 num_features=1, num_neurons=NUM_NEURONS, freq=FREQ,
                                 num_ensemble=NUM_ENSEMBLE)
        rnn_model.prepare_tensor()
        rnn_model.create_model()
        rnn_model.train_lstm()
        rnn_smape = rnn_model.evaluate()
        rnn_pred = list(map(int, rnn_model.pred_test_output))

        # autoregressive with network method
        # preset AR coefficient
        preset_ar_coef = list(ar_model.fitted_params)
        # network feature method
        src_ts_data_mat = np.empty((0, T), np.int)
        for src_embed in tar_inlink_dict[tar_embed]:
            src_ts_data = np.array(embed_view_dict[src_embed])
            src_ts_data_mat = np.vstack((src_ts_data_mat, src_ts_data))

        arnet_model = ARNet(tar_ts_data, src_ts_data_mat=src_ts_data_mat,
                            num_input=NUM_INPUT, num_output=NUM_OUTPUT,
                            num_ensemble=NUM_ENSEMBLE)
        arnet_model.train_arnet(start_params=preset_ar_coef)
        arnet_smape = arnet_model.evaluate()
        arnet_pred = list(map(int, arnet_model.pred_test_output))

        fout.write('{0}\n'.format(json.dumps({'embed': tar_embed,
                                              'true_value': true_value,
                                              'naive_pred': naive_pred,
                                              'snaive_pred': snaive_pred,
                                              'ar_pred': ar_pred,
                                              'rnn_pred': rnn_pred,
                                              'arnet_pred': arnet_pred,
                                              'net_ratio': arnet_model.network_ratio,
                                              'incoming_embeds': tar_inlink_dict[tar_embed],
                                              'link_weights': arnet_model.link_weights.tolist()})))

        print('embed: {0}, Naive: {1:.3f}, SeasonalNaive: {2:.3f}, AutoRegression: {3:.3f}, RNN: {4:.3f}, ARNet: {5:.3f}'.format(tar_embed, naive_smape, snaive_smape, ar_smape, rnn_smape, arnet_smape))

        naive_model = None
        del naive_model
        snaive_model = None
        del snaive_model
        ar_model = None
        del ar_model
        rnn_model = None
        del rnn_model
        arnet_model = None
        del arnet_model
        gc.collect()

        timer.stop()

    fout.close()


if __name__ == "__main__":
    T = 63

    FREQ = 7
    NUM_INPUT = 7
    NUM_OUTPUT = 7
    NUM_NEURONS = 25
    NUM_ENSEMBLE = 3

    main()
