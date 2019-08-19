#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Calculate the probability of forming a persistent link within width w in n consecutive days.
Each link is formed with probability p, and w=7.
Probability is computed via simulation.
Note: need run 'python extract_persistent_network.py' to generate ../data/persistent_network.csv

Usage: python compute_linkage_pearsonr.py
Input data files: ../data/vevo_forecast_data_60k.tsv, ../data/persistent_network.csv, ../data/network_pickle/
Output data files: ./reciprocal_pearsonr.log, ./persistent_pearsonr.log, ./ephemeral_pearsonr.log
Time: ~2H
"""

import sys, os, pickle
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, obj2str
from utils.data_loader import DataLoader
from utils.tsa import extract_seasonal_component, extract_trend_component


def detsn(ts_data, freq=7):
    """ deseasonalize, detrend, and normalize the time series data.
    """
    ts_data = np.array(ts_data)
    # deseasonalize
    ts_seasonality_in = extract_seasonal_component(ts_data, freq)
    for i in range(0, len(ts_data)):
        ts_data[i] = ts_data[i] * 100 / ts_seasonality_in[i % freq]
    # detrend
    a, b = extract_trend_component(ts_data)
    for i in range(0, len(ts_data)):
        ts_data[i] = ts_data[i] - ((a * i) + b)
    # normalize
    processed_ts_data = (ts_data - np.mean(ts_data)) / np.std(ts_data)
    return processed_ts_data


def main():
    # == == == == == == Part 1: Set up environment == == == == == == #
    timer = Timer()
    timer.start()

    data_prefix = '../data/'

    # == == == == == == Part 2: Load video views == == == == == == #
    data_loader = DataLoader()
    data_loader.load_video_views()
    embed_view_dict = data_loader.embed_view_dict
    embed_avg_view_dict = data_loader.embed_avg_view_dict

    # == == == == == == Part 3: Load persistent and non-persistent network == == == == == == #
    reciprocal_link_set = set()
    persistent_link_set = set()
    non_persistent_link_set = set()

    with open(os.path.join(data_prefix, 'persistent_network.csv'), 'r') as fin:
        fin.readline()
        for line in fin:
            src_embed, tar_embed = map(int, line.rstrip().split(','))

            link = '{0}-{1}'.format(src_embed, tar_embed)
            rec_link = '{1}-{0}'.format(src_embed, tar_embed)
            if rec_link in persistent_link_set:
                persistent_link_set.remove(rec_link)
                reciprocal_link_set.add(link)
            else:
                persistent_link_set.add(link)

    for t in range(T):
        target_date_str = obj2str(datetime(2018, 9, 1) + timedelta(days=t))
        filename = 'network_{0}.p'.format(target_date_str)
        network_dict = pickle.load(open(os.path.join(data_prefix, 'network_pickle', filename), 'rb'))
        for tar_embed in network_dict:
            src_embed_list = [x[0] for x in network_dict[tar_embed] if x[1] < NUM_REL]
            if len(src_embed_list) > 0:
                for src_embed in src_embed_list:
                    # filter: at least 100 daily views for target video,
                    # and the mean daily views of source video is at least 1% of the target video
                    if embed_avg_view_dict[tar_embed] >= 100 and embed_avg_view_dict[src_embed] >= 0.01 * embed_avg_view_dict[tar_embed]:
                        link = '{0}-{1}'.format(src_embed, tar_embed)
                        rec_link = '{1}-{0}'.format(src_embed, tar_embed)
                        if link not in persistent_link_set and rec_link not in persistent_link_set \
                                and link not in reciprocal_link_set and rec_link not in reciprocal_link_set \
                                and link not in non_persistent_link_set and rec_link not in non_persistent_link_set:
                            non_persistent_link_set.add(link)

    print('>>> Number of reciprocal links: {0}'.format(len(reciprocal_link_set)))
    print('>>> Number of persistent links (non-reciprocal): {0}'.format(len(persistent_link_set)))
    print('>>> Number of ephemeral links: {0}'.format(len(non_persistent_link_set)))

    for link_set, log_filename in zip([reciprocal_link_set, persistent_link_set, non_persistent_link_set],
                                      ['./reciprocal_pearsonr.log', './persistent_pearsonr.log',
                                       './ephemeral_pearsonr.log']):
        with open(log_filename, 'w') as log_file:
            for link in link_set:
                src_embed, tar_embed = map(int, link.split('-'))
                eff_size, pvalue = pearsonr(detsn(embed_view_dict[src_embed]), detsn(embed_view_dict[tar_embed]))
                log_file.write('{0},{1},{2},{3}\n'.format(src_embed, tar_embed, eff_size, pvalue))

    timer.stop()


if __name__ == '__main__':
    NUM_REL = 15
    T = 63

    main()
