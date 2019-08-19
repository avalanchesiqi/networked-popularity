#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract relevant_list files to pickle files for faster access.

Usage: python extract_network_pickle.py
Input data files: ../data/recsys/
Output data files: ../data/network_pickle/
Time: ~1H x number of files ~= 63H
"""

import sys, os, pickle, time, json
from datetime import datetime, timedelta
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, intify, obj2str


def main():
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    total_start_time = time.time()

    data_prefix = '../data/'
    forecast_filepath = 'vevo_forecast_data_60k.tsv'
    recsys_dirpath = 'recsys'
    snapshot_dirpath = 'network_pickle'

    if not os.path.exists(os.path.join(data_prefix, snapshot_dirpath)):
        os.mkdir(os.path.join(data_prefix, snapshot_dirpath))

    # == == == == == == Part 2: Load vevo en videos 61k dataset == == == == == == #
    vid_embed_dict = {}
    vid_view_dict = {}
    with open(os.path.join(data_prefix, forecast_filepath), 'r') as fin:
        for line in fin:
            embed, vid, ts_view, total_view = line.rstrip().split('\t')
            vid_embed_dict[vid] = int(embed)
            ts_view = np.array(intify(ts_view.split(',')))
            vid_view_dict[vid] = ts_view
    vevo_en_vid_list = list(sorted(vid_embed_dict.keys()))
    num_videos = len(vevo_en_vid_list)

    for t in range(T):
        timer = Timer()
        timer.start()

        target_date_str = obj2str(datetime(2018, 9, 1) + timedelta(days=t))
        recsys_filepath = 'recsys_{0}.json'.format(target_date_str)
        snapshot_filepath = 'network_{0}.p'.format(target_date_str)
        network_mat = {embed: [] for embed in range(num_videos)}

        with open(os.path.join(data_prefix, recsys_dirpath, recsys_filepath), 'r') as fin:
            for line in fin:
                network_json = json.loads(line.rstrip())
                source = network_json['vid']
                targets = network_json['relevant_list'][: MAX_POSITION]
                for position, target in enumerate(targets):
                    if target in vevo_en_vid_list:
                        # add embedding of incoming video and position of target video on source video
                        network_mat[vid_embed_dict[target]].append((vid_embed_dict[source], position, vid_view_dict[source][t]))

        with open(os.path.join(data_prefix, snapshot_dirpath, snapshot_filepath), 'wb') as fout:
            pickle.dump(network_mat, fout)

        print('>>> Finish dumping date {0}'.format(target_date_str))
        timer.stop()

    print('>>> Network structure has been dumped!')
    print('>>> Total elapsed time: {0}\n'.format(str(timedelta(seconds=time.time() - total_start_time))[:-3]))


if __name__ == '__main__':
    MAX_POSITION = 50
    T = 63

    main()
