#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot spearman correlation in each year.

Usage: python plot_fig8_yearly_spearmanr.py
Time: ~2M
"""

import os, sys, platform, pickle
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import spearmanr

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.data_loader import DataLoader
from utils.helper import Timer, obj2str
from utils.plot import ColorPalette, hide_spines


def main():
    # == == == == == == Part 1: Set up environment == == == == == == #
    timer = Timer()
    timer.start()

    data_prefix = '../data/'
    year_labels = ["all years", "'09", "'10", "'11", "'12", "'13", "'14", "'15", "'16", "'17", "'18"]
    num_year = len(year_labels) - 1

    # == == == == == == Part 2: Load video views == == == == == == #
    data_loader = DataLoader()
    data_loader.load_video_views()
    data_loader.load_embed_content_dict()
    embed_avg_view_dict = data_loader.embed_avg_view_dict
    embed_uploadtime_dict = data_loader.embed_uploadtime_dict
    num_videos = data_loader.num_videos

    for embed in range(num_videos):
        upload_year = int(embed_uploadtime_dict[embed][:4])
        if upload_year >= 2009:
            year_idx = upload_year - 2009
        else:
            year_idx = 0
        embed_uploadtime_dict[embed] = year_idx

    views_by_years_list = [[] for _ in range(num_year)]
    indegrees_by_years_list = [[] for _ in range(num_year)]

    # == == == == == == Part 3: Load dynamic network snapshot == == == == == == #
    embed_indegree_dict_15 = {embed: np.zeros((T,)) for embed in np.arange(num_videos)}
    for t in range(T):
        filename = 'network_{0}.p'.format(obj2str(datetime(2018, 9, 1) + timedelta(days=t)))
        with open(os.path.join(data_prefix, 'network_pickle', filename), 'rb') as fin:
            network_dict = pickle.load(fin)
            # embed_tar: [(embed_src, pos_src, view_src)]
            for embed in range(num_videos):
                embed_indegree_dict_15[embed][t] = len([1 for x in network_dict[embed] if x[1] < NUM_REL_15])
        print('>>> Finish loading day {0}...'.format(t + 1))
    print('>>> Network structure has been loaded!')

    for embed in range(num_videos):
        views_by_years_list[embed_uploadtime_dict[embed]].append(embed_avg_view_dict[embed])
        indegrees_by_years_list[embed_uploadtime_dict[embed]].append(np.mean(embed_indegree_dict_15[embed]))

    spearman_traces = []
    all_views, all_indegrees = [], []
    for i in range(num_year):
        all_views.extend(views_by_years_list[i])
        all_indegrees.extend(indegrees_by_years_list[i])
    print('\n>>> {0}'.format(year_labels[0]), spearmanr(all_views, all_indegrees))
    spearman_traces.append(spearmanr(all_views, all_indegrees)[0])
    for i in range(num_year):
        spearman_traces.append(spearmanr(views_by_years_list[i], indegrees_by_years_list[i])[0])
        print('>>> {0} year'.format(year_labels[1 + i]), spearmanr(views_by_years_list[i], indegrees_by_years_list[i]))

    # == == == == == == Part 4: Plotting script == == == == == == #
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 2))
    tomato = ColorPalette.TOMATO
    blue = ColorPalette.BLUE

    bar1 = ax1.bar(range(num_year + 1), spearman_traces, edgecolor=['k'] * (num_year + 1),
                   color=[tomato] + [blue] * num_year, lw=1)
    for rect in bar1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '{0:.3f}'.format(height), ha='center', va='bottom')

    ax1.set_xticks(np.arange(11))
    ax1.set_xticklabels(year_labels)
    ax1.set_ylabel(r'spearman $\rho$')

    hide_spines(ax1)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/measure_spearmanr.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    NUM_REL_15 = 15
    T = 63

    main()
