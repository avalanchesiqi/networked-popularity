#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot the probability of videos in relevant list show in the recommended list,
and the probability of videos shown in recommended list originate from relevant list.

Usage: python plot_fig3_rel2rec.py
Input data files: ../data/recsys/
Time: ~10M
"""

import sys, os, platform, json
import numpy as np

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
mpl.rcParams['lines.linewidth'] = 1

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.plot import ColorPalette, hide_spines, stackedBarPlot


def switch(x):
    # categorize into the right bar
    if x == 0:
        return 0
    elif 1 <= x < 5:
        return 1
    elif 5 <= x < 10:
        return 2
    elif 10 <= x < 15:
        return 3
    elif 15 <= x < 30:
        return 4
    elif 30 <= x < 50:
        return 5


def main():
    # == == == == == == Part 1: Set up environment == == == == == == #
    timer = Timer()
    timer.start()

    data_prefix = '../data/recsys'

    num_relevant_by_rank = np.zeros((NUM_REL,))
    num_recommended_by_rank = np.zeros((NUM_REC,))

    # aggregate by rank1, rank2-5, rank6-10, rank11-15
    dense_relevant_in_recommended_mat = np.zeros((NUM_REL, 4))
    # aggregate by rank1, rank2-5, rank6-10, rank11-15, rank16-30, rank31-50
    dense_recommended_from_relevant_mat = np.zeros((NUM_REC, 6))

    relevant_in_recommended_arr = np.zeros((NUM_REL,))
    recommended_from_relevant_arr = np.zeros((NUM_REC,))

    # == == == == == == Part 2: Load both relevant list and recommended list == == == == == == #
    for subdir, _, files in os.walk(data_prefix):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                for line in fin:
                    network_json = json.loads(line.rstrip())
                    recommended_list = network_json['recommended_list'][: NUM_REC]
                    relevant_list = network_json['relevant_list'][: NUM_REL]

                    num_relevant_by_rank += np.pad(np.ones(len(relevant_list)), (0, NUM_REL - len(relevant_list)), 'constant')
                    num_recommended_by_rank += np.pad(np.ones(len(recommended_list)), (0, NUM_REC - len(recommended_list)), 'constant')

                    for rel_rank, vid in enumerate(relevant_list):
                        if vid in recommended_list:
                            relevant_in_recommended_arr[rel_rank] += 1
                            position_on_recommended = recommended_list.index(vid)
                            dense_relevant_in_recommended_mat[rel_rank, switch(position_on_recommended)] += 1

                    for rec_rank, vid in enumerate(recommended_list):
                        if vid in relevant_list:
                            recommended_from_relevant_arr[rec_rank] += 1
                            position_on_relevant = relevant_list.index(vid)
                            dense_recommended_from_relevant_mat[rec_rank, switch(position_on_relevant)] += 1

    # == == == == == == Part 3: Plot probabilities in each position == == == == == == #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = axes.ravel()
    color_cycle_6 = ColorPalette.CC6

    stackedBarPlot(ax=axes[0], data=dense_relevant_in_recommended_mat / num_relevant_by_rank.reshape(-1, 1),
                   cols=color_cycle_6,
                   edgeCols=['#000000'] * 4,
                   xlabel='position $x$ on relevant list',
                   ylabel='prob. of displaying on recommended list',
                   scale=False,
                   endGaps=True)

    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc=color_cycle_6[x], alpha=0.8, ec='k') for x in range(4)],
                   ['position 1', 'position 2-5', 'position 6-10', 'position 11-15'], fontsize=10,
                   frameon=False,
                   loc='upper right', fancybox=False, shadow=True, ncol=1)
    axes[0].set_title('(a)', fontsize=12)

    stackedBarPlot(ax=axes[1], data=dense_recommended_from_relevant_mat / num_recommended_by_rank.reshape(-1, 1),
                   cols=ColorPalette.CC6,
                   edgeCols=['#000000'] * 6,
                   xlabel='position $x$ on recommended list',
                   ylabel='prob. of originating from relevant list',
                   scale=False,
                   endGaps=True)

    axes[1].legend([plt.Rectangle((0, 0), 1, 1, fc=color_cycle_6[x], alpha=0.8, ec='k') for x in range(6)],
                   ['position 1', 'position 2-5', 'position 6-10', 'position 11-15', 'position 16-30', 'position 31-50'], fontsize=10,
                   frameon=False,
                   loc='upper right', fancybox=False, shadow=True, ncol=2)

    axes[1].set_title('(b)', fontsize=12)

    for ax in axes:
        ax.set_ylim(top=1)
        ax.set_ylim(bottom=0)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/data_rel2rec.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    NUM_REL = 50
    NUM_REC = 15

    main()
