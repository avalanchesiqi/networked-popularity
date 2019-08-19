#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot how indegree changes on next day and distribution of link frequency.
Filter: video with at least 10 indegree (top 11% in terms of indegree) on current day.

Usage: python plot_fig10_temporal_micro.py
Input data files: ../data/vevo_en_embeds_60k.txt, ../data/network_pickle/
Time: ~2M
"""

import sys, os, platform, pickle
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
mpl.rcParams['lines.linewidth'] = 1.5

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.data_loader import DataLoader
from utils.helper import Timer
from utils.plot import ColorPalette, concise_fmt, hide_spines


def smoothing(indegree_change_dict, target_x, percentile):
    if target_x - 1 in indegree_change_dict and target_x + 1 in indegree_change_dict:
        return 0.4 * np.percentile(indegree_change_dict[target_x], percentile) \
               + 0.1 * np.percentile(indegree_change_dict[target_x], percentile - 4) \
               + 0.1 * np.percentile(indegree_change_dict[target_x], percentile + 4) \
               + 0.1 * np.percentile(indegree_change_dict[target_x - 1], percentile) \
               + 0.1 * np.percentile(indegree_change_dict[target_x + 1], percentile) \
               + 0.05 * np.percentile(indegree_change_dict[target_x - 1], percentile - 4) \
               + 0.05 * np.percentile(indegree_change_dict[target_x + 1], percentile - 4) \
               + 0.05 * np.percentile(indegree_change_dict[target_x - 1], percentile + 4) \
               + 0.05 * np.percentile(indegree_change_dict[target_x + 1], percentile + 4)
    elif target_x - 1 in indegree_change_dict:
        return 0.5 * np.percentile(indegree_change_dict[target_x], percentile) \
               + 0.15 * np.percentile(indegree_change_dict[target_x], percentile - 4) \
               + 0.15 * np.percentile(indegree_change_dict[target_x], percentile + 4) \
               + 0.1 * np.percentile(indegree_change_dict[target_x - 1], percentile) \
               + 0.05 * np.percentile(indegree_change_dict[target_x - 1], percentile - 4) \
               + 0.05 * np.percentile(indegree_change_dict[target_x - 1], percentile + 4)
    elif target_x + 1 in indegree_change_dict:
        return 0.5 * np.percentile(indegree_change_dict[target_x], percentile) \
               + 0.15 * np.percentile(indegree_change_dict[target_x], percentile - 4) \
               + 0.15 * np.percentile(indegree_change_dict[target_x], percentile + 4) \
               + 0.1 * np.percentile(indegree_change_dict[target_x + 1], percentile) \
               + 0.05 * np.percentile(indegree_change_dict[target_x + 1], percentile - 4) \
               + 0.05 * np.percentile(indegree_change_dict[target_x + 1], percentile + 4)
    else:
        return 0.5 * np.percentile(indegree_change_dict[target_x], percentile) \
               + 0.25 * np.percentile(indegree_change_dict[target_x], percentile - 4) \
               + 0.25 * np.percentile(indegree_change_dict[target_x], percentile + 4)


def plot_contour(indegree_change_dict, target_x, ax, color='k', fsize=11):
    """ Plot contour for one target x value."""
    ax.plot([target_x, target_x],
            [np.percentile(indegree_change_dict[target_x], 0.5), np.percentile(indegree_change_dict[target_x], 99.5)],
            c=color, zorder=20)
    for percentile in [10, 25, 50, 75, 90]:
        ax.plot([target_x - 5, target_x + 5],
                [smoothing(indegree_change_dict, target_x, percentile), smoothing(indegree_change_dict, target_x, percentile)],
                c=color, zorder=20)
        ax.text(target_x - 20, smoothing(indegree_change_dict, target_x, percentile), '{0}%'.format(percentile),
                fontsize=fsize, verticalalignment='center', zorder=20)
    for percentile in [0.5, 99.5]:
        ax.plot([target_x - 5, target_x + 5],
                [np.percentile(indegree_change_dict[target_x], percentile),
                 np.percentile(indegree_change_dict[target_x], percentile)],
                c=color, zorder=20)

    ax.annotate('',
                xy=(target_x + 9, 0.6), xycoords='data',
                xytext=(target_x + 9, smoothing(indegree_change_dict, target_x, 75)), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.text(target_x + 12, 0.38, '25% videos gain {0:.0f}+ in-links'.format(
        abs(np.percentile(indegree_change_dict[target_x], 75)) * target_x),
            fontsize=fsize, verticalalignment='center', zorder=20)

    ax.annotate('',
                xy=(target_x + 9, -0.6), xycoords='data',
                xytext=(target_x + 9, smoothing(indegree_change_dict, target_x, 25)), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.text(target_x + 12, -0.4, '25% videos lose {0:.0f}+ in-links'.format(
        abs(np.percentile(indegree_change_dict[target_x], 25)) * target_x),
            fontsize=fsize, verticalalignment='center', zorder=20)


def main():
    # == == == == == == Part 1: Set up environment == == == == == == #
    timer = Timer()
    timer.start()

    data_prefix = '../data/'

    # == == == == == == Part 2: Load video views == == == == == == #
    data_loader = DataLoader()
    data_loader.load_video_views()
    num_videos = data_loader.num_videos

    # == == == == == == Part 3: Load dynamic network snapshot == == == == == == #
    embed_indegree_dict = {embed: np.zeros((T,)) for embed in np.arange(num_videos)}
    edge_frequency_dict = defaultdict(int)
    for t in range(T):
        filename = 'network_{0}.p'.format((datetime(2018, 9, 1) + timedelta(days=t)).strftime('%Y-%m-%d'))
        with open(os.path.join(data_prefix, 'network_pickle', filename), 'rb') as fin:
            network_dict = pickle.load(fin)
            # embed_tar: [(embed_src, pos_src, view_src), ...]
            for embed_tar in range(num_videos):
                inlinks = [x for x in network_dict[embed_tar] if x[1] < NUM_REL]
                if len(inlinks) > 0:
                    for embed_src, _, _, in inlinks:
                        edge_frequency_dict['{0}-{1}'.format(embed_src, embed_tar)] += 1
                embed_indegree_dict[embed_tar][t] = len(inlinks)
        print('>>> Finish loading day {0}...'.format(t + 1))
    print('>>> Network structure has been loaded!')

    link_frequency_counter = Counter(edge_frequency_dict.values())

    # == == == == == == Part 4: Plot how indegree changes == == == == == == #
    cornflower_blue = ColorPalette.BLUE
    tomato = ColorPalette.TOMATO

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
    ax1, ax2 = axes.ravel()
    indegree_change_dict = defaultdict(list)
    for embed in range(num_videos):
        for t in range(T-1):
            x0 = embed_indegree_dict[embed][t]
            x1 = embed_indegree_dict[embed][t+1]
            if x0 >= 10:
                indegree_change_dict[x0].append((x1-x0) / x0)

    x_axis = sorted([x for x in indegree_change_dict.keys() if len(indegree_change_dict[x]) >= 100])

    for i in np.arange(5, 50, 5):
        ax1.fill_between(x_axis, [smoothing(indegree_change_dict, x, 50 - i) for x in x_axis],
                         [smoothing(indegree_change_dict, x, 55 - i) for x in x_axis],
                         facecolor=cornflower_blue, alpha=(100 - 2 * i) / 100, lw=0)
        ax1.fill_between(x_axis, [smoothing(indegree_change_dict, x, 45 + i) for x in x_axis],
                         [smoothing(indegree_change_dict, x, 50 + i) for x in x_axis],
                         facecolor=cornflower_blue, alpha=(100 - 2 * i) / 100, lw=0)

    for i in [25, 75]:
        ax1.plot(x_axis, [smoothing(indegree_change_dict, x, i) for x in x_axis], color=cornflower_blue, alpha=0.8, zorder=15)
    ax1.plot(x_axis, [smoothing(indegree_change_dict, x, 50) for x in x_axis], color=cornflower_blue, alpha=1, zorder=15)

    ax1.set_ylim([-0.9, 0.9])
    ax1.set_xlabel('indegree', fontsize=12)
    ax1.set_ylabel('indegree change ratio the next day', fontsize=12)
    ax1.set_title('(a)', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    plot_contour(indegree_change_dict, target_x=100, ax=ax1)

    x_axis = range(1, 1 + T)
    y_axis = [link_frequency_counter[x] for x in x_axis]

    print('\nephemeral links of frequency 1, {0}, {1:.2f}%'.format(y_axis[0], y_axis[0] / sum(y_axis) * 100))
    print('persistent links of frequency 63, {0}, {1:.2f}%'.format(y_axis[-1], y_axis[-1] / sum(y_axis) * 100))

    ax2.plot(x_axis, y_axis, 'o-', c=tomato, mfc='none', mec=tomato, ms=4)
    ax2.set_xlabel('link frequency', fontsize=12)
    ax2.set_ylabel('num of video-to-video pairs', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title('(b)', fontsize=12)

    ax2.annotate('ephemeral links', fontsize=12,
                 xy=(3, 350000), xycoords='data',
                 xytext=(17, 350000), textcoords='data',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

    ax2.annotate('frequent links', fontsize=12,
                 xy=(61, 35000), xycoords='data',
                 xytext=(35, 55000), textcoords='data',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

    ax2.yaxis.set_major_formatter(FuncFormatter(concise_fmt))

    hide_spines(axes)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/measure_temporal_micro.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    NUM_REL = 15
    T = 63

    main()
