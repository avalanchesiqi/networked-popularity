#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot how videos connect to one another, partitioned by avg views.

Usage: python plot_fig5_how_videos_connect.py
Input data files: ../data/vevo_forecast_data_60k.csv, ../data/network_pickle/
Time: ~2M
"""

import sys, os, itertools, pickle
from datetime import datetime, timedelta
import numpy as np
import igraph
from network2tikz import plot

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.data_loader import DataLoader
from utils.helper import Timer
from utils.plot import ColorPalette


def scaler(arr):
    max_value = 4
    scaling_factor = np.max(arr) / max_value
    return [x / scaling_factor for x in arr]


def main():
    # == == == == == == Part 1: Set up environment == == == == == == #
    timer = Timer()
    timer.start()

    data_prefix = '../data/'

    # == == == == == == Part 2: Load video views == == == == == == #
    data_loader = DataLoader()
    data_loader.load_video_views()
    embed_avg_view_dict = data_loader.embed_avg_view_dict
    num_videos = data_loader.num_videos

    # == == == == == == Part 3: Build views percentile partition == == == == == == #
    day_views = list(embed_avg_view_dict.values())
    median_value = np.median(day_views)
    # the top 1st quantile is 75th percentile and above
    first_quantile_value = np.percentile(day_views, 75)
    third_quantile_value = np.percentile(day_views, 25)

    embed_percentile_dict = {}
    for embed in np.arange(num_videos):
        if embed_avg_view_dict[embed] >= first_quantile_value:
            embed_percentile_dict[embed] = 0
        elif embed_avg_view_dict[embed] >= median_value:
            embed_percentile_dict[embed] = 1
        elif embed_avg_view_dict[embed] >= third_quantile_value:
            embed_percentile_dict[embed] = 2
        else:
            embed_percentile_dict[embed] = 3

    # == == == == == == Part 4: Load dynamic network snapshot == == == == == == #
    edge_weight_mat = np.zeros((4, 4), dtype=np.float32)
    for t in range(T):
        filename = 'network_{0}.p'.format((datetime(2018, 9, 1) + timedelta(days=t)).strftime('%Y-%m-%d'))
        with open(os.path.join(data_prefix, 'network_pickle', filename), 'rb') as fin:
            network_dict = pickle.load(fin)
            # embed_tar: [(embed_src, pos_src, view_src), ...]
            for embed_tar in range(num_videos):
                for embed_src, pos_src, _ in network_dict[embed_tar]:
                    if pos_src < NUM_REL:
                        edge_weight_mat[(embed_percentile_dict[embed_src], embed_percentile_dict[embed_tar])] += 1 / T
        print('>>> Finish loading day {0}...'.format(t + 1))
    edge_weight_mat = edge_weight_mat.astype(np.int)
    print('>>> Network structure has been loaded!')

    # == == == == == == Part 5: Plot graph by network2tikz == == == == == == #
    # Network
    # -------
    # every possible pair, including self loop
    network_structure = []
    num_partitions = 4
    for pair in itertools.product(np.arange(num_partitions), repeat=2):
        network_structure.append(pair)
    net = igraph.Graph(network_structure, directed=True)

    # Network attributes
    # ------------------
    # Network dicts
    # -------------
    layout = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}

    # Visual style dict
    # -----------------
    visual_style = {}

    # node styles
    # -----------
    visual_style['vertex_size'] = 0.9
    visual_style['vertex_color'] = ColorPalette.CCRGB4
    visual_style['vertex_opacity'] = 0.6
    visual_style['vertex_label'] = ['top 25\%', '(25\% 50\%', '(50\% 75\%', 'bottom 25\%']
    visual_style['vertex_label_distance'] = 0
    visual_style['vertex_label_size'] = [5, 4, 4, 4]

    # edge styles
    # -----------
    edge_width = list(np.ravel(edge_weight_mat))
    visual_style['edge_width'] = scaler(edge_width)
    visual_style['edge_curved'] = 0.7
    edge_label = ['{{{:,}}}'.format(x) for x in edge_width]
    visual_style['edge_label'] = edge_label
    visual_style['edge_label_size'] = 4.5
    visual_style['edge_loop_shape'] = 60
    visual_style['edge_loop_size'] = 1
    visual_style['edge_loop_position'] = [180, 0, 0, 0]
    visual_style['edge_arrow_size'] = 0.01
    visual_style['edge_arrow_width'] = [0.03, 0.01, 0.01, 0.01,
                                        0.02, 0.01, 0.01, 0.01,
                                        0.02, 0.01, 0.01, 0.01,
                                        0.02, 0.01, 0.01, 0.01]

    # general options
    # ---------------
    visual_style['layout'] = layout
    visual_style['canvas'] = (10, 3.5)
    visual_style['margin'] = 1.5

    # Create pdf figure of the network
    plot(net, '../images/measure_how_videos_connect.pdf', **visual_style)
    print('>>> Generated pdf file ../images/measure_how_videos_connect.pdf')

    timer.stop()


if __name__ == '__main__':
    NUM_REL = 15
    T = 63

    main()
