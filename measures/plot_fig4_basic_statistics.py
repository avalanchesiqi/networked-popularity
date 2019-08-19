#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot basic statistics of Vevo Music 60k dataset, e.g., indegree distribution, views distribution, uploading trend.
It also outputs the data for table 2.

Usage: python plot_fig4_basic_statistics.py
Input data files: ../data/vevo_forecast_data_60k.tsv, ../data/vevo_en_videos_60k.json, ../data/network_pickle/
Time: ~2M
"""

import sys, os, platform, pickle
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import spearmanr, percentileofscore
from collections import Counter, defaultdict
from powerlaw import Fit, plot_ccdf

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
mpl.rcParams['lines.linewidth'] = 1

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, str2obj, gini
from utils.data_loader import DataLoader
from utils.plot import ColorPalette, concise_fmt, hide_spines, stackedBarPlot


def main():
    # == == == == == == Part 1: Set up environment == == == == == == #
    timer = Timer()
    timer.start()

    data_prefix = '../data/'

    target_day_indices = [0, 15, 30, 45]
    color_cycle_4 = ColorPalette.CC4
    date_labels = ['Sep 01, 2018', 'Sep 16, 2018', 'Oct 01, 2018', 'Oct 16, 2018']

    # == == == == == == Part 2: Load video views == == == == == == #
    data_loader = DataLoader()
    data_loader.load_video_views()
    embed_view_dict = data_loader.embed_view_dict
    embed_avg_view_dict = data_loader.embed_avg_view_dict
    num_videos = data_loader.num_videos

    target_day_view_list = [[], [], [], []]
    for embed in range(num_videos):
        for target_idx, target_day in enumerate(target_day_indices):
            target_day_view_list[target_idx].append(embed_view_dict[embed][target_day])

    # == == == == == == Part 3: Load dynamic network snapshot == == == == == == #
    embed_indegree_dict = {embed: np.zeros((T,)) for embed in np.arange(num_videos)}  # daily indegree for each embed
    zero_indegree_list = []  # percentage of zero indegree for each day
    num_edges_list = []  # number of total edges for each day
    for t in range(T):
        filename = 'network_{0}.p'.format((datetime(2018, 9, 1) + timedelta(days=t)).strftime('%Y-%m-%d'))
        indegree_list = []
        with open(os.path.join(data_prefix, 'network_pickle', filename), 'rb') as fin:
            network_dict = pickle.load(fin)
            # embed_tar: [(embed_src, pos_src, view_src), ...]
            for tar_embed in range(num_videos):
                indegree_value = len([1 for x in network_dict[tar_embed] if x[1] < NUM_REL])
                embed_indegree_dict[tar_embed][t] = indegree_value
                indegree_list.append(indegree_value)
        indegree_counter = Counter(indegree_list)
        zero_indegree_list.append(indegree_counter[0] / num_videos)
        num_edges_list.append(sum(indegree_list))
        print('>>> Finish loading day {0}...'.format(t + 1))
    print('>>> Network structure has been loaded!')
    print('\n>>> Average number of edges: {0:.0f}, max: {1:.0f}, min: {2:.0f}'.format(sum(num_edges_list) / len(num_edges_list), max(num_edges_list), min(num_edges_list)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    ax1, ax2, ax3 = axes.ravel()

    # == == == == == == Part 4: Plot ax1 indegree CCDF == == == == == == #
    embed_avg_indegree_dict = defaultdict(float)
    for t in range(T):
        for embed in range(num_videos):
            embed_avg_indegree_dict[embed] += embed_indegree_dict[embed][t] / T

    indegree_ranked_embed_list = [x[0] for x in sorted(embed_avg_indegree_dict.items(), key=lambda kv: kv[1], reverse=True)]
    top_20_indegree_embeds = indegree_ranked_embed_list[:20]
    popular_ranked_embed_list = [x[0] for x in sorted(embed_avg_view_dict.items(), key=lambda kv: kv[1], reverse=True)]
    top_20_popular_embeds = popular_ranked_embed_list[:20]

    for target_idx, target_day in enumerate(target_day_indices):
        indegree_list = []
        for embed in range(num_videos):
            indegree_list.append(embed_indegree_dict[embed][target_day])

        print('video with 10 indegree has more in-links than {0:.2f}% videos on date {1}'.format(percentileofscore(indegree_list, 10), date_labels[target_idx]))
        print('video with 20 indegree has more in-links than {0:.2f}% videos on date {1}'.format(percentileofscore(indegree_list, 20), date_labels[target_idx]))

        plot_ccdf(indegree_list, ax=ax1, color=color_cycle_4[target_idx], label=date_labels[target_idx])

    # compute the powerlaw fit
    powerlaw_fit = Fit(list(embed_avg_indegree_dict.values()))
    infer_alpha = powerlaw_fit.power_law.alpha
    p = powerlaw_fit.power_law.ccdf()
    ins_x_axis = powerlaw_fit.power_law.__dict__['parent_Fit'].__dict__['data'][: int(0.9 * len(p))]
    ins_y_axis = 0.1 * p[: int(0.9 * len(p))]

    ax1.plot(ins_x_axis, ins_y_axis, 'k:')
    ax1.text(0.4, 0.6, r'$x^{{{0:.2f}}}$'.format(-infer_alpha + 1), size=12,
             ha='right', va='bottom', transform=ax1.transAxes)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('indegree', fontsize=11)
    ax1.set_ylabel('$P(X) \geq x$', fontsize=11)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_title('(a) indegree distribution', fontsize=12)

    ax1.legend(frameon=False, fontsize=11, ncol=1, fancybox=False, shadow=True)

    mean_zero_indegree = sum(zero_indegree_list) / len(zero_indegree_list)

    ax1.axhline(y=1 - mean_zero_indegree, color='k', linestyle='--', zorder=30)
    ax1.text(0.96, 0.9, '{0:.0f}% with 0 indegree'.format(mean_zero_indegree * 100), size=11,
             transform=ax1.transAxes, ha='right', va='top')

    # == == == == == == Part 5: Plot ax2 views distribution == == == == == == #
    for target_idx, views_list in enumerate(target_day_view_list):
        x_values = range(100)
        y_values = [np.percentile(views_list, x) for x in x_values]
        ax2.plot(x_values, y_values, color=color_cycle_4[target_idx], label=date_labels[target_idx])
    ax2.set_yscale('log')
    ax2.set_xlabel('views percentile', fontsize=11)
    ax2.set_ylabel('num of views', fontsize=11)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title('(b) daily views vs. its percentile', fontsize=12)

    avg_views_list = sorted(list(embed_avg_view_dict.values()), reverse=True)
    gini_coef = gini(avg_views_list)
    print('top 1% videos occupy {0:.2f}% views'.format(sum(avg_views_list[: int(0.01 * num_videos)]) / sum(avg_views_list) * 100))
    print('top 10% videos occupy {0:.2f}% views'.format(sum(avg_views_list[: int(0.1 * num_videos)]) / sum(avg_views_list) * 100))
    print('Gini coef: {0:.3f}'.format(gini_coef))

    spearman_degree = [embed_avg_indegree_dict[embed] for embed in range(num_videos)]
    spearman_views = [embed_avg_view_dict[embed] for embed in range(num_videos)]

    print('Spearman correlation between views and indegree: {0:.4f}, pvalue: {1:.2f}'.format(*spearmanr(spearman_views, spearman_degree)))

    median_views = np.median(avg_views_list)
    top_views_90th = np.percentile(avg_views_list, 90)
    top_views_99th = np.percentile(avg_views_list, 99)
    ax2_xmin = ax2.get_xlim()[0]
    ax2_ymin = ax2.get_ylim()[0]

    ax2.plot((50, 50), (ax2_ymin, median_views), color='k', linestyle='--', zorder=30)
    ax2.plot((ax2_xmin, 50), (median_views, median_views), color='k', linestyle='--', zorder=30)
    ax2.text(0.49, 0.45, 'median views {0:,.0f}'.format(median_views), size=11,
             transform=ax2.transAxes, ha='right', va='bottom')

    ax2.plot((90, 90), (ax2_ymin, top_views_90th), color='k', linestyle='--', zorder=30)
    ax2.plot((ax2_xmin, 90), (top_views_90th, top_views_90th), color='k', linestyle='--', zorder=30)
    ax2.text(0.88, 0.75, '90th views {0:,.0f}'.format(top_views_90th), size=11,
             transform=ax2.transAxes, ha='right', va='bottom')

    ax2.plot((99, 99), (ax2_ymin, top_views_99th), color='k', linestyle='--', zorder=30)
    ax2.plot((ax2_xmin, 99), (top_views_99th, top_views_99th), color='k', linestyle='--', zorder=30)
    ax2.text(0.91, 0.95, '99th views {0:,.0f}'.format(top_views_99th), size=11,
             transform=ax2.transAxes, ha='right', va='bottom')

    # == == == == == == Part 7: Plot ax3 video uploading trend == == == == == == #
    x_axis = range(2009, 2018)
    x_labels = ["'09", "'10", "'11", "'12", "'13", "'14", "'15", "'16", "'17"]
    upload_mat = np.zeros((len(x_axis), 8))

    target_topics = ['Pop_music', 'Rock_music', 'Hip_hop_music', 'Independent_music',
                     'Country_music', 'Electronic_music', 'Soul_music', 'Others']
    topic_labels = ['Pop', 'Rock', 'Hip hop', 'Independent',
                    'Country', 'Electronic', 'Soul', 'Others']

    color_cycle_8 = ColorPalette.CC8

    data_loader.load_embed_content_dict()
    embed_title_dict = data_loader.embed_title_dict
    embed_uploadtime_dict = data_loader.embed_uploadtime_dict
    embed_genre_dict = data_loader.embed_genre_dict

    for embed in range(num_videos):
        upload_year = int(embed_uploadtime_dict[embed][:4])
        if 2009 <= upload_year <= 2017:
            year_idx = upload_year - 2009

            genres = embed_genre_dict[embed]
            if len(genres) == 0:
                # add one to "Others" genre
                upload_mat[year_idx, 7] += 1
            else:
                for genre in genres:
                    upload_mat[year_idx, target_topics.index(genre)] += 1 / len(genres)

    print()
    print(['{0}: {1}'.format(topic, int(num)) for topic, num in zip(target_topics, np.sum(upload_mat, axis=0))])

    stackedBarPlot(ax=ax3, data=upload_mat,
                   cols=color_cycle_8,
                   edgeCols=['#000000'] * 8,
                   xlabel='uploaded year',
                   ylabel='num of videos',
                   scale=False,
                   endGaps=True)

    ax3.tick_params(axis='both', which='major', labelsize=9)
    ax3.set_xticks(np.arange(len(x_axis)))
    ax3.set_xticklabels(x_labels)
    ax3.yaxis.set_major_formatter(FuncFormatter(concise_fmt))
    ax3.legend([plt.Rectangle((0, 0), 1, 1, fc=c, ec='k', alpha=0.6) for c in color_cycle_8],
               topic_labels, fontsize=9,
               frameon=False, handletextpad=0.2, columnspacing=0.3, ncol=4,
               bbox_to_anchor=(1, -0.12), bbox_transform=ax3.transAxes,
               fancybox=False, shadow=True)
    ax3.set_title('(c) VEVO videos uploading trend', fontsize=12)

    union_top_set = set(top_20_indegree_embeds).union(top_20_popular_embeds)
    print('\n>>> Size of the union set at cutoff 15:', len(union_top_set))
    print('{0:>24} | {1:>17} | {2:>5} | {3:>8} | {4:>6} | {5:>10} | {6:>5}'.format('Video title', 'Artist', 'Age',
                                                                                   'Indegree', '-rank',
                                                                                   'Views', '-rank'))
    for embed in top_20_indegree_embeds:
        print('{0:>24} & {1:>17} & {2:>5} & {3:>8} & {4:>6} & {5:>10} & {6:>5} \\\\'
              .format(embed_title_dict[embed].split(' - ', 1)[1].split('(')[0].split('ft')[0].strip(),
                      embed_title_dict[embed].split(' - ', 1)[0].split('&')[0].split(',')[0].strip(),
                      '{0:,}'.format((datetime(2018, 11, 2) - str2obj(embed_uploadtime_dict[embed])).days),
                      '{0:,}'.format(int(embed_avg_indegree_dict[embed])),
                      '{0:,}'.format(top_20_indegree_embeds.index(embed) + 1),
                      '{0:,}'.format(int(embed_avg_view_dict[embed])),
                      '{0:,}'.format(popular_ranked_embed_list.index(embed) + 1)
                      ))

    print('\n{0:>24} | {1:>17} | {2:>5} | {3:>8} | {4:>6} | {5:>10} | {6:>5}'.format('Video title', 'Artist', 'Age',
                                                                                     'Indegree', '-rank',
                                                                                     'Views', '-rank'))
    for embed in top_20_popular_embeds:
        print('{0:>24} & {1:>17} & {2:>5} & {3:>8} & {4:>6} & {5:>10} & {6:>5} \\\\'
              .format(embed_title_dict[embed].split(' - ', 1)[1].split('(')[0].split('ft')[0].strip(),
                      embed_title_dict[embed].split(' - ', 1)[0].split('&')[0].split(',')[0].strip(),
                      '{0:,}'.format((datetime(2018, 11, 2) - str2obj(embed_uploadtime_dict[embed])).days),
                      '{0:,}'.format(int(embed_avg_indegree_dict[embed])),
                      '{0:,}'.format(indegree_ranked_embed_list.index(embed) + 1),
                      '{0:,}'.format(int(embed_avg_view_dict[embed])),
                      '{0:,}'.format(top_20_popular_embeds.index(embed) + 1)
                      ))

    hide_spines(axes)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/measure_basic_statistics.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    NUM_REL = 15
    T = 63

    main()
