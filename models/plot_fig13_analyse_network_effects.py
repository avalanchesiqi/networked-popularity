#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot prediction results.
(a) SMAPEs vs. estimated network contribution ratio
(b) percentile gained without network
(c) closer look on artists who gain massive popular percentile

Usage: python plot_fig13_analyse_network_effects.py
Input data files: ./embed_prediction.json, ../data/artist_details.json
Time: ~1M
"""

import sys, os, json, platform
import numpy as np
from collections import defaultdict
from scipy.stats import percentileofscore

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
mpl.rcParams['lines.linewidth'] = 1.5

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.data_loader import DataLoader
from utils.metrics import symmetric_mean_absolute_percentage_error as smape
from utils.helper import Timer, is_same_genre, gini
from utils.plot import ColorPalette, hide_spines


def main():
    timer = Timer()
    timer.start()

    cornflower_blue = ColorPalette.BLUE
    tomato = ColorPalette.TOMATO
    color_cycle_4 = ColorPalette.CC4
    label_fs = ColorPalette.LABELFS
    title_fs = ColorPalette.TITLEFS
    tick_style = ColorPalette.TICKSTYLE

    data_loader = DataLoader()
    data_loader.load_video_views()
    embed_view_dict = data_loader.embed_view_dict
    embed_avg_train_view_dict = {embed: np.mean(embed_view_dict[embed][:-NUM_OUTPUT]) for embed in embed_view_dict.keys()}
    data_loader.load_embed_content_dict()
    embed_cid_dict = data_loader.embed_cid_dict
    embed_genre_dict = data_loader.embed_genre_dict

    cid_artist_dict = {}
    cid_tag_dict = {}
    with open('../data/artist_details.json', 'r') as fin:
        for line in fin:
            artist_json = json.loads(line.rstrip())
            cid_artist_dict[artist_json['channel_id']] = artist_json['artist_name']
            cid_tag_dict[artist_json['channel_id']] = artist_json['tag-dict']

    cid_views_dict = defaultdict(int)
    cid_views_wo_network_dict = defaultdict(int)

    arnet_smape_list = []
    net_ratio_list = []
    same_artist_net_ratio_list = []
    same_genre_net_ratio_list = []
    total_views = 0
    network_explained_views = 0

    with open('./embed_prediction.json', 'r') as fin:
        for line in fin:
            result_json = json.loads(line.rstrip())
            tar_embed = result_json['embed']
            avg_train_views = embed_avg_train_view_dict[tar_embed]

            true_value = result_json['true_value']
            arnet_pred = result_json['arnet_pred']
            arnet_smape_list.append(smape(true_value, arnet_pred)[0])

            incoming_embeds = result_json['incoming_embeds']
            link_weights = result_json['link_weights']
            same_artist_contributed_views = 0
            same_genre_contributed_views = 0
            for edge_inx, src_embed in enumerate(incoming_embeds):
                if embed_cid_dict[tar_embed] == embed_cid_dict[src_embed]:
                    same_artist_contributed_views += link_weights[edge_inx] * embed_avg_train_view_dict[src_embed]
                if is_same_genre(embed_genre_dict[tar_embed], embed_genre_dict[src_embed]):
                    same_genre_contributed_views += link_weights[edge_inx] * embed_avg_train_view_dict[src_embed]

            # analyse network contribution
            arnet_net_ratio = result_json['net_ratio']
            net_ratio_list.append(arnet_net_ratio)
            # rounding issue can make the value slightly larger than 1
            same_artist_net_ratio_list.append(min(same_artist_contributed_views / avg_train_views, 1))
            same_genre_net_ratio_list.append(min(same_genre_contributed_views / avg_train_views, 1))

            cid_views_dict[embed_cid_dict[tar_embed]] += avg_train_views
            cid_views_wo_network_dict[embed_cid_dict[tar_embed]] += avg_train_views * (1 - arnet_net_ratio)

            total_views += avg_train_views
            network_explained_views += avg_train_views * arnet_net_ratio

    print('\nFor an average video in our dataset, we estimate {0:.1f}% of the views come from the network.'.format(100 * np.mean(net_ratio_list)))
    print('In particular, {0:.1f}% ({1:.1f}%) of the views come from the same artist.'.format(100 * np.mean(same_artist_net_ratio_list), 100 * np.mean(same_artist_net_ratio_list) / np.mean(net_ratio_list)))
    print('In total, our model estimates that the recommendation network contributes {0:.1f}% of popularity in the Vevo network.'.format(100 * network_explained_views / total_views))
    print('total views for 13K: {0:.1f}M'.format(total_views / 1000000))
    print('explained views for 13K: {0:.1f}M'.format(network_explained_views / 1000000))
    print('total views for 60K: {0:.1f}M'.format(np.sum(list(embed_avg_train_view_dict.values())) / 1000000))
    print('Gini coef with network: {0:.4f}'.format(gini(list(cid_views_dict.values()))))
    print('Gini coef without network: {0:.4f}\n'.format(gini(list(cid_views_wo_network_dict.values()))))

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 4.2))
    gs = axes[0, 0].get_gridspec()
    for ax in axes[:, 1]:
        ax.remove()
    ax_mid = fig.add_subplot(gs[:, 1])
    for ax in axes[:, 2]:
        ax.remove()
    ax_right = fig.add_subplot(gs[:, 2])
    axes = [axes[0, 0], axes[1, 0], ax_mid, ax_right]

    # == == == == == == Part 1: Plot SMAPE vs. traffic composition == == == == == == #
    num_bin = 10
    sorted_same_artist_tuple_list = sorted([(x, y) for x, y in zip(same_artist_net_ratio_list, arnet_smape_list)], key=lambda x: x[0])
    same_artist_split_values = [np.percentile(same_artist_net_ratio_list, x) for x in np.arange(10, 101, 10)]
    same_artist_bins = [[] for _ in range(num_bin)]
    for same_artist_net_ratio, arnet_smape in sorted_same_artist_tuple_list:
        slice_idx = int(np.floor(percentileofscore(same_artist_net_ratio_list, same_artist_net_ratio) / 10))
        if slice_idx >= num_bin:
            slice_idx = num_bin - 1
        same_artist_bins[slice_idx].append(arnet_smape)

    sorted_same_genre_tuple_list = sorted([(x, y) for x, y in zip(same_genre_net_ratio_list, arnet_smape_list)], key=lambda x: x[0])
    same_genre_split_values = [np.percentile(same_genre_net_ratio_list, x) for x in np.arange(10, 101, 10)]
    same_genre_bins = [[] for _ in range(num_bin)]
    for same_genre_net_ratio, arnet_smape in sorted_same_genre_tuple_list:
        slice_idx = int(np.floor(percentileofscore(same_genre_net_ratio_list, same_genre_net_ratio) / 10))
        if slice_idx >= num_bin:
            slice_idx = num_bin - 1
        same_genre_bins[slice_idx].append(arnet_smape)

    axes[0].plot(range(1, 11, 1), [np.mean(x) for x in same_artist_bins], color=cornflower_blue,
                 label='same artist', mfc='none', marker='o', markersize=4)
    axes[1].plot(range(1, 11, 1), [np.mean(x) for x in same_genre_bins], color=tomato,
                 label='same genre', mfc='none', marker='o', markersize=4)

    for ax in [axes[0], axes[1]]:
        ax.set_xlim([0.5, 10.5])
        ax.set_ylim([7, 10.5])
        ax.set_ylabel('SMAPE', fontsize=label_fs)
        ax.xaxis.set_ticks(np.arange(1, 10, 2))
        ax.tick_params(**tick_style)
        ax.legend(frameon=False)

    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: '({0:.3f})'.format(same_artist_split_values[int(x) - 1])))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{0:.0f}%\n({1:.3f})'.format(10 * x, same_genre_split_values[int(x) - 1])))

    # axes[0].xaxis.set_major_formatter(
    #     FuncFormatter(lambda x, _: '({0:.3f})'.format(10 * x)))
    # axes[1].xaxis.set_major_formatter(
    #     FuncFormatter(lambda x, _: '{0:.0f}%\n({1:.3f})'.format(10 * x, 10 * x)))

    axes[1].set_xlabel('$\eta_v$ percentile', fontsize=label_fs)
    axes[0].set_title('(a)', fontsize=title_fs)

    # == == == == == == Part 2: Plot who can utilize the network better? == == == == == == #
    artist_views_list = list(cid_views_dict.values())
    wo_network_artist_views_list = list(cid_views_wo_network_dict.values())
    cid_list = sorted(cid_views_dict.keys())
    artist_true_percentile = [percentileofscore(artist_views_list, cid_views_dict[cid]) for cid in cid_list]
    wo_network_artist_percentile = [percentileofscore(wo_network_artist_views_list, cid_views_wo_network_dict[cid]) for cid in cid_list]
    percentile_change = np.array([artist_true_percentile[i] - wo_network_artist_percentile[i] for i in range(len(cid_list))])

    num_popularity_loss = sum(percentile_change < 0)
    num_popularity_equal = sum(percentile_change == 0)
    num_popularity_gain = sum(percentile_change > 0)
    print('{0} ({1:.2f}%) artists lose popularity with network'.format(num_popularity_loss, num_popularity_loss / len(cid_list) * 100))
    print('{0} ({1:.2f}%) artists with no popularity change'.format(num_popularity_equal, num_popularity_equal / len(cid_list) * 100))
    print('{0} ({1:.2f}%) artists gain popularity with network\n'.format(num_popularity_gain, num_popularity_gain / len(cid_list) * 100))

    artist_percentile_mat = [[] for _ in range(10)]
    artist_cid_mat = [[] for _ in range(10)]
    for idx, percentile_value in enumerate(wo_network_artist_percentile):
        bin_idx = min(int(np.floor(percentile_value / 10)), 9)
        artist_percentile_mat[bin_idx].append(artist_true_percentile[idx] - percentile_value)
        artist_cid_mat[bin_idx].append(cid_list[idx])

    red_circle = dict(markerfacecolor=tomato, marker='o', markersize=4)
    axes[2].boxplot(artist_percentile_mat, showfliers=True, widths=0.5, flierprops=red_circle)
    axes[2].axhline(y=0, color=cornflower_blue, linestyle='--', lw=1, zorder=0)
    axes[2].set_xlabel('artist popularity percentile without network', fontsize=label_fs)
    axes[2].set_ylabel('percentile change with network', fontsize=label_fs)
    axes[2].tick_params(**tick_style)
    axes[2].set_xticks(axes[2].get_xticks()[::2])
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{0:.0f}%'.format(10 * x)))
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{0:.0f}%'.format(x)))
    axes[2].set_title('(b)', fontsize=12)

    # find outliers
    whis = 1.5
    top_outliers_list = []
    bottom_outliers_list = []
    for box_idx, box in enumerate(artist_percentile_mat):
        q1 = np.percentile(box, 25)
        q3 = np.percentile(box, 75)
        iq = q3 - q1
        hi_val = q3 + whis * iq
        lo_val = q1 - whis * iq
        for idx, val in enumerate(box):
            if val > hi_val:
                top_outliers_list.append((artist_cid_mat[box_idx][idx], val))
            elif val < lo_val:
                bottom_outliers_list.append((artist_cid_mat[box_idx][idx], val))

    sorted_top_outliers_list = sorted([(cid_artist_dict[x[0]], cid_tag_dict[x[0]], int(cid_views_dict[x[0]]), x[1]) for x in top_outliers_list], key=lambda t: t[2], reverse=True)
    for t in sorted_top_outliers_list:
        print(t)
    print('-------------------')
    sorted_bottom_outliers_list = sorted([(cid_artist_dict[x[0]], cid_tag_dict[x[0]], int(cid_views_dict[x[0]]), x[1]) for x in bottom_outliers_list], key=lambda t: t[2], reverse=True)
    for t in sorted_bottom_outliers_list:
        print(t)

    indie_xaxis, indie_yaxis = [], []
    rap_xaxis, rap_yaxis = [], []
    other_xaxis, other_yaxis = [], []
    lose_xaxis, lose_yaxis = [], []
    for top_outlier, _ in top_outliers_list:
        if 'indie' in ','.join(cid_tag_dict[top_outlier].keys()) or \
                'alternative' in ','.join(cid_tag_dict[top_outlier].keys()) or \
                'new wave' in ','.join(cid_tag_dict[top_outlier].keys()):
            indie_xaxis.append(cid_views_dict[top_outlier])
            indie_yaxis.append((cid_views_dict[top_outlier] - cid_views_wo_network_dict[top_outlier]) / cid_views_dict[top_outlier])
        elif 'rap' in ','.join(cid_tag_dict[top_outlier].keys()) or \
                'hip hop' in ','.join(cid_tag_dict[top_outlier].keys()) or \
                'rhythm and blues' in ','.join(cid_tag_dict[top_outlier].keys()) or \
                'reggae' in ','.join(cid_tag_dict[top_outlier].keys()) or \
                'punk' in ','.join(cid_tag_dict[top_outlier].keys()) or \
                'funk' in ','.join(cid_tag_dict[top_outlier].keys()) or \
                'r&b' in ','.join(cid_tag_dict[top_outlier].keys()):
            rap_xaxis.append(cid_views_dict[top_outlier])
            rap_yaxis.append((cid_views_dict[top_outlier] - cid_views_wo_network_dict[top_outlier]) / cid_views_dict[top_outlier])
        else:
            other_xaxis.append(cid_views_dict[top_outlier])
            other_yaxis.append((cid_views_dict[top_outlier] - cid_views_wo_network_dict[top_outlier]) / cid_views_dict[top_outlier])
    for bottom_outlier, _ in bottom_outliers_list:
        lose_xaxis.append(cid_views_dict[bottom_outlier])
        lose_yaxis.append((cid_views_dict[bottom_outlier] - cid_views_wo_network_dict[bottom_outlier]) / cid_views_dict[bottom_outlier])

    axes[3].scatter(indie_xaxis, indie_yaxis, marker='^', facecolors='none', edgecolors=color_cycle_4[0], s=20, label='Indie: {0}'.format(len(indie_xaxis)))
    axes[3].scatter(rap_xaxis, rap_yaxis, marker='o', facecolors='none', edgecolors=color_cycle_4[1], s=20, label='Hip hop: {0}'.format(len(rap_xaxis)))
    axes[3].scatter(other_xaxis, other_yaxis, marker='s', facecolors='none', edgecolors=color_cycle_4[2], s=20, label='Other: {0}'.format(len(other_xaxis)))
    # axes[3].scatter(lose_xaxis, lose_yaxis, marker='x', color=color_cycle_4[3], s=20, label='artists lose popularity: {0}'.format(len(bad_xaxis)))
    axes[3].set_ylim((-0.02, 1.02))
    axes[3].set_xscale('log')
    axes[3].set_xlabel('artist average daily views', fontsize=label_fs)
    axes[3].set_ylabel('network contribution ratio ' + '$\eta_v$', fontsize=label_fs)
    axes[3].tick_params(**tick_style)
    axes[3].legend(frameon=False, loc='lower left')
    axes[3].set_title('(c)', fontsize=title_fs)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout(w_pad=0.2)
    plt.savefig('../images/model_prediction_analysis.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    NUM_OUTPUT = 7

    main()
