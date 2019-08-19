#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot the profiling of persistent links.
(a) what's the probability of forming a persistent link?
(b) what's the percentage of links that are significant correlated?
(c) what's the percentage of significant correlated links that belong to the same artist or contain the same genre?
Note: need run 'python justify_persistent_link.py' to generate ./justify_persistent_link.log
and run 'python compute_linkage_pearsonr.py' to generate ./random_pearsonr.log, ./ephemeral_pearsonr.log, ./persistent_pearsonr.log, ./reciprocal_pearsonr.log

Usage: python plot_fig11_profile_persistent_network.py
Input data files: ./justify_persistent_link.log,
                  ./random_pearsonr.log, ./ephemeral_pearsonr.log, ./persistent_pearsonr.log, ./reciprocal_pearsonr.log
Time: ~1M
"""

import sys, os, platform, re
import numpy as np

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
mpl.rcParams['lines.linewidth'] = 1.5

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.data_loader import DataLoader
from utils.helper import Timer, is_same_genre
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
    data_loader.load_embed_content_dict()
    embed_cid_dict = data_loader.embed_cid_dict
    embed_genre_dict = data_loader.embed_genre_dict

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 4))
    gs = axes[0, 0].get_gridspec()
    for ax in axes[:, 0]:
        ax.remove()
    ax_left = fig.add_subplot(gs[:, 0])
    for ax in axes[:, 1]:
        ax.remove()
    ax_mid = fig.add_subplot(gs[:, 1])
    axes = [ax_left, ax_mid, axes[0, 2], axes[1, 2]]

    # == == == == == == Part 1: Plot the probability of forming a persistent link == == == == == == #
    p_form_list = []
    p_persistent_list = []
    with open('./justify_persistent_link.log', 'r') as fin:
        for line in fin:
            _, p_form, _, p_persistent = re.split(',|:', line)
            p_form = float(p_form.strip())
            p_persistent = float(p_persistent.strip())
            p_form_list.append(p_form)
            p_persistent_list.append(p_persistent)

    axes[0].plot(p_form_list, p_persistent_list, color=cornflower_blue)
    for p_form in [0.5, 0.7, 0.8, 0.9]:
        p_persistent = p_persistent_list[int(p_form * 100)]
        axes[0].scatter(p_form, p_persistent, s=15, c=tomato, edgecolors='k', zorder=30)
        axes[0].text(p_form - 0.01, p_persistent, '({0:.2f}, {1:.2f})'.format(p_form, p_persistent), ha='right', va='bottom')
    axes[0].set_xlabel('prob. of forming a link', fontsize=label_fs)
    axes[0].set_ylabel('prob. of being persistent link', fontsize=label_fs)
    axes[0].tick_params(**tick_style)
    axes[0].set_title('(a)', fontsize=title_fs)

    # == == == == == == Part 2: Plot the portion of persistent links that pass statistics test == == == == == == #
    log_files_list = ['./random_pearsonr.log', './ephemeral_pearsonr.log', './persistent_pearsonr.log', './reciprocal_pearsonr.log']
    link_cnt_list = []
    sign_ratio_list = []
    same_artist_list = []
    sign_ratio_same_artist_list = []
    same_genre_list = []
    sign_ratio_same_genre_list = []
    for log_file in log_files_list:
        cnt = 0
        same_artist_cnt = 0
        same_genre_cnt = 0

        sign_cnt = 0
        sign_cnt_same_artist = 0
        sign_cnt_same_genre = 0

        with open(log_file, 'r') as fin:
            for line in fin:
                src_embed, tar_embed, r, p = line.rstrip().split(',')
                src_embed = int(src_embed)
                tar_embed = int(tar_embed)
                r = float(r)
                p = float(p)
                if p < 0.05:
                    sign_cnt += 1

                cnt += 1
                if embed_cid_dict[src_embed] == embed_cid_dict[tar_embed]:
                    same_artist_cnt += 1
                    if p < 0.05:
                        sign_cnt_same_artist += 1
                if is_same_genre(embed_genre_dict[src_embed], embed_genre_dict[tar_embed]):
                    same_genre_cnt += 1
                    if p < 0.05:
                        sign_cnt_same_genre += 1

        sign_ratio_list.append(sign_cnt / cnt)

        same_artist_list.append(same_artist_cnt / cnt)
        sign_ratio_same_artist_list.append(sign_cnt_same_artist / cnt)

        same_genre_list.append(same_genre_cnt / cnt)
        sign_ratio_same_genre_list.append(sign_cnt_same_genre / cnt)

        link_cnt_list.append(cnt)
        print('#links: {0}, #sign links: {1}, #sign same artist: {2}, #sign same genre: {3}'.format(cnt, sign_cnt, sign_cnt_same_artist, sign_cnt_same_genre))

    ind = np.arange(len(log_files_list))
    axes[1].bar(ind, sign_ratio_list, 0.6, edgecolor=['k'] * 4, color=color_cycle_4, lw=1.5, alpha=0.6)
    axes[1].set_ylim([0, axes[0].get_ylim()[1]])
    axes[1].set_ylabel('percentage of links with p<0.05', fontsize=label_fs)
    axes[1].set_xticklabels(('',
                             'random' + r'$^{}$' + '\n({0:,})'.format(link_cnt_list[0]),
                             'ephemeral' + r'$^{}$' + '\n({0:,})'.format(link_cnt_list[1]),
                             'persistent' + r'$^{-}$' + '\n({0:,})'.format(link_cnt_list[2]),
                             'reciprocal' + r'$^{}$' + '\n({0:,})'.format(link_cnt_list[3])))
    for tick in ind:
        axes[1].text(tick, sign_ratio_list[tick] + 0.01, '{0:.3f}'.format(sign_ratio_list[tick]), **bar_text_style)
    axes[1].tick_params(**tick_style)
    axes[1].set_title('(b)', fontsize=title_fs)

    # == == == == == == Part 3: Plot the percentage of significant persistent links belong to the same artist or contain the same genre == == == == == == #
    axes[2].bar(ind, np.array(same_artist_list) - np.array(sign_ratio_same_artist_list), 0.6,
                bottom=sign_ratio_same_artist_list, edgecolor=color_cycle_4, color=['w'] * 4, hatch='//', lw=1.5, alpha=0.6)
    axes[2].bar(ind, sign_ratio_same_artist_list, 0.6, edgecolor=['k'] * 4, color=color_cycle_4, lw=1.5, alpha=0.6)
    axes[2].set_ylim([0, axes[0].get_ylim()[1]])
    axes[2].set_ylabel('same artist', fontsize=label_fs)
    axes[2].text(0, same_artist_list[0] + 0.01, '{0:.3f}'.format(same_artist_list[0]), **bar_text_style)
    for tick in ind[1:]:
        axes[2].text(tick, same_artist_list[tick] + 0.01, '{0:.3f}'.format(same_artist_list[tick]), **bar_text_style)
        axes[2].text(tick, sign_ratio_same_artist_list[tick] + 0.01, '{0:.3f}'.format(sign_ratio_same_artist_list[tick]), **bar_text_style)
    axes[2].tick_params(**tick_style)
    axes[2].get_xaxis().set_visible(False)
    axes[2].set_title('(c)', fontsize=title_fs)

    axes[3].bar(ind, np.array(same_genre_list) - np.array(sign_ratio_same_genre_list), 0.6,
                bottom=sign_ratio_same_genre_list, edgecolor=color_cycle_4, color=['w'] * 4, hatch='//', lw=1.5, alpha=0.6)
    axes[3].bar(ind, sign_ratio_same_genre_list, 0.6, edgecolor=['k'] * 4, color=color_cycle_4, lw=1.5, alpha=0.6)
    axes[3].set_ylim([0, axes[0].get_ylim()[1]])
    axes[3].set_ylabel('same genre', fontsize=label_fs)
    for tick in ind:
        axes[3].text(tick, same_genre_list[tick] + 0.01, '{0:.3f}'.format(same_genre_list[tick]), **bar_text_style)
        axes[3].text(tick, sign_ratio_same_genre_list[tick] + 0.01, '{0:.3f}'.format(sign_ratio_same_genre_list[tick]), **bar_text_style)
    axes[3].tick_params(**tick_style)
    axes[3].set_xticklabels(('',
                             'random' + r'$^{}$' + '\n({0:,})'.format(link_cnt_list[0]),
                             'ephemeral' + r'$^{}$' + '\n({0:,})'.format(link_cnt_list[1]),
                             'persistent' + r'$^{-}$' + '\n({0:,})'.format(link_cnt_list[2]),
                             'reciprocal' + r'$^{}$' + '\n({0:,})'.format(link_cnt_list[3])))

    hide_spines(axes)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/model_persistent_links.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
