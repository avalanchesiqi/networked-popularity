#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot prediction results.
(a) performance between different methods
(b) performance as forecast horizon extends
(c) link strength vs. view ratio from src to tar

Usage: python plot_fig12_prediction_results.py
Input data files: ./embed_prediction.json
Time: ~1M
"""

import sys, os, json, platform
import numpy as np

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
mpl.rcParams['lines.linewidth'] = 1.5

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.data_loader import DataLoader
from utils.metrics import symmetric_mean_absolute_percentage_error as smape
from utils.helper import Timer
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
    bar_text_style = ColorPalette.BARTEXTSTYLE

    data_loader = DataLoader()
    data_loader.load_video_views()
    embed_view_dict = data_loader.embed_view_dict
    embed_avg_train_view_dict = {embed: np.mean(embed_view_dict[embed][:-NUM_OUTPUT]) for embed in embed_view_dict.keys()}

    net_ratio_list = []

    src_to_tar_view_ratio = []
    link_weights_record = []

    naive_smape_list, snaive_smape_list, ar_smape_list, rnn_smape_list, arnet_smape_list = [[] for _ in range(5)]
    naive_daily_smape_mat, snaive_daily_smape_mat, ar_daily_smape_mat, rnn_daily_smape_mat, arnet_daily_smape_mat = [np.empty((0, NUM_OUTPUT), np.float) for _ in range(5)]

    with open('./embed_prediction.json', 'r') as fin:
        for line in fin:
            result_json = json.loads(line.rstrip())
            tar_embed = result_json['embed']

            true_value = result_json['true_value']
            naive_pred = result_json['naive_pred']
            snaive_pred = result_json['snaive_pred']
            ar_pred = result_json['ar_pred']
            rnn_pred = result_json['rnn_pred']
            arnet_pred = result_json['arnet_pred']

            naive_smape, naive_daily_smape_arr = smape(true_value, naive_pred)
            naive_smape_list.append(naive_smape)
            naive_daily_smape_mat = np.vstack((naive_daily_smape_mat, naive_daily_smape_arr))

            snaive_smape, snaive_daily_smape_arr = smape(true_value, snaive_pred)
            snaive_smape_list.append(snaive_smape)
            snaive_daily_smape_mat = np.vstack((snaive_daily_smape_mat, snaive_daily_smape_arr))

            ar_smape, ar_daily_smape_arr = smape(true_value, ar_pred)
            ar_smape_list.append(ar_smape)
            ar_daily_smape_mat = np.vstack((ar_daily_smape_mat, ar_daily_smape_arr))

            rnn_smape, rnn_daily_smape_arr = smape(true_value, rnn_pred)
            rnn_smape_list.append(rnn_smape)
            rnn_daily_smape_mat = np.vstack((rnn_daily_smape_mat, rnn_daily_smape_arr))

            arnet_smape, arnet_daily_smape_arr = smape(true_value, arnet_pred)
            arnet_smape_list.append(arnet_smape)
            arnet_daily_smape_mat = np.vstack((arnet_daily_smape_mat, arnet_daily_smape_arr))

            # analyse network contribution
            arnet_net_ratio = result_json['net_ratio']
            net_ratio_list.append(arnet_net_ratio)

            incoming_embeds = result_json['incoming_embeds']
            link_weights = result_json['link_weights']
            for edge_inx, src_embed in enumerate(incoming_embeds):
                view_ratio = np.log10(embed_avg_train_view_dict[src_embed] / embed_avg_train_view_dict[tar_embed])
                src_to_tar_view_ratio.append(view_ratio)
                link_weights_record.append(link_weights[edge_inx])

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12, 4))
    axes = axes.ravel()

    # == == == == == == Part 1: Plot performance comparison == == == == == == #
    smape_mat = [naive_smape_list, snaive_smape_list, ar_smape_list, rnn_smape_list, arnet_smape_list]
    axes[0].boxplot(smape_mat, showfliers=False, meanline=True, showmeans=True, widths=0.7)
    means = [np.mean(x) for x in smape_mat]
    pos = range(len(means))
    for tick, label in zip(pos, axes[1].get_xticklabels()):
        axes[0].text(pos[tick] + 1, means[tick] + 0.3, '{0:.3f}'.format(means[tick]), **bar_text_style)

    axes[0].set_xticklabels(['Naive', 'SN', 'AR', 'RNN', 'ARNet'], fontsize=label_fs)
    axes[0].set_ylabel('SMAPE', fontsize=label_fs)
    axes[0].tick_params(**tick_style)
    axes[0].set_title('(a)', fontsize=title_fs)

    # == == == == == == Part 2: Plot performance with forecast horizon extends == == == == == == #
    axes[1].plot(np.arange(1, 1 + NUM_OUTPUT), np.mean(naive_daily_smape_mat, axis=0), label='Naive', c='k', mfc='none', marker='D', markersize=4)
    axes[1].plot(np.arange(1, 1 + NUM_OUTPUT), np.mean(snaive_daily_smape_mat, axis=0), label='SN', c=color_cycle_4[0], mfc='none', marker='*', markersize=5)
    axes[1].plot(np.arange(1, 1 + NUM_OUTPUT), np.mean(ar_daily_smape_mat, axis=0), label='AR', c=color_cycle_4[1], mfc='none', marker='s', markersize=5)
    axes[1].plot(np.arange(1, 1 + NUM_OUTPUT), np.mean(rnn_daily_smape_mat, axis=0), label='RNN', c=color_cycle_4[2], mfc='none', marker='^', markersize=5)
    axes[1].plot(np.arange(1, 1 + NUM_OUTPUT), np.mean(arnet_daily_smape_mat, axis=0), label='ARNet', c=color_cycle_4[3], marker='o', markersize=5)

    axes[1].set_xlabel('forecast horizon', fontsize=label_fs)
    axes[1].set_ylabel('SMAPE', fontsize=label_fs)
    axes[1].set_ylim([6, 23])
    axes[1].tick_params(**tick_style)
    axes[1].legend(frameon=False)
    axes[1].set_title('(b)', fontsize=title_fs)

    # == == == == == == Part 3: Plot link strength vs. view ratio from src to tar == == == == == == #
    bin_axis = np.arange(-2, 1.9, 0.1)
    bin_records = [[] for _ in range(len(bin_axis))]
    for x, y in zip(src_to_tar_view_ratio, link_weights_record):
        if x >= -2:
            bin_records[int(np.floor((x + 2) * 10))].append(y)

    for t in np.arange(5, 50, 5):
        axes[2].fill_between(bin_axis, [np.percentile(x, 50 - t) for x in bin_records],
                         [np.percentile(x, 55 - t) for x in bin_records],
                         facecolor=cornflower_blue, alpha=(100 - 2 * t) / 100, lw=0)
        axes[2].fill_between(bin_axis, [np.percentile(x, 45 + t) for x in bin_records],
                         [np.percentile(x, 50 + t) for x in bin_records],
                         facecolor=cornflower_blue, alpha=(100 - 2 * t) / 100, lw=0)

    for t in [10, 30, 70, 90]:
        axes[2].plot(bin_axis, [np.percentile(x, t) for x in bin_records], color=cornflower_blue, alpha=(100-2*t)/100, lw=1, zorder=15)
    median_line = [np.percentile(x, 50) for x in bin_records]
    axes[2].plot(bin_axis, median_line, color='k', alpha=0.5, zorder=20, lw=1.5)
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: r'$10^{{{0:.0f}}}%$'.format(x)))

    peak1_idx = int(np.argmax(median_line))
    peak2_idx = 10 + int(np.argmax(median_line[10:]))
    peak1 = (bin_axis[peak1_idx], median_line[peak1_idx])
    peak2 = (bin_axis[peak2_idx], median_line[peak2_idx])
    axes[2].scatter(peak1[0], peak1[1], s=15, c=tomato, edgecolors='k', zorder=30)
    axes[2].text(peak1[0] + 0.08, peak1[1] + 0.01, '({0:.2f}, {1:.2f})'.format(10**peak1[0], peak1[1]), ha='left', va='center')
    axes[2].scatter(peak2[0], peak2[1], s=15, c=tomato, edgecolors='k', zorder=30)
    axes[2].text(peak2[0], peak2[1] + 0.02, '({0:.2f}, {1:.2f})'.format(10**peak2[0], peak2[1]), ha='center', va='bottom')

    axes[2].set_xlim((-2.05, 2.02))
    axes[2].set_ylim((-0.02, 1.01))
    axes[2].set_xlabel('views ratio from video ' + r'$u$' + ' to video ' + r'$v$', fontsize=label_fs)
    axes[2].set_ylabel('estimated link strength ' + r'$\beta_{u, v}$', fontsize=label_fs)
    axes[2].set_title('(c)', fontsize=title_fs)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/model_prediction_results.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    NUM_OUTPUT = 7

    main()
