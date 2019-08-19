#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot teaser figure that consists of 6 videos from Adele.
Fig.1(a) is generated by Gephi.

Usage: python plot_fig1_teaser.py
Input data files: ../data/teaser.json
Time: ~1M
"""

import sys, os, json, platform
from datetime import datetime, timedelta

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
mpl.rcParams['lines.linewidth'] = 1

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.plot import ColorPalette, concise_fmt, hide_spines


def main():
    timer = Timer()
    timer.start()

    data_prefix = '../data/'

    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(12, 4), sharex='col')
    gs = axes[0, 0].get_gridspec()
    for ax in axes[:, 0]:
        ax.remove()
    ax_left = fig.add_subplot(gs[:, 0])
    ax_left.set_axis_off()
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['bottom'].set_visible(False)
    ax_left.spines['left'].set_visible(False)

    ax_left.set_title('(a)', fontsize=12)
    axes = axes[:, 1:].ravel()

    video_title_list = ['Hello', 'Someone like you', 'Rolling in the deep',
                        'Skyfall', 'Set fire to the rain', 'Hometown glory']

    # == == == == == == Part 1: Load data == == == == == == #
    fig_idx = 0
    with open(os.path.join(data_prefix, 'teaser.json'), 'r') as fin:
        for line in fin:
            video_json = json.loads(line.rstrip())
            daily_view = video_json['insights']['dailyView']
            end_date = datetime.strptime(video_json['insights']['endDate'], '%Y-%m-%d')
            start_date = end_date - timedelta(days=len(daily_view))
            date_axis = [start_date + timedelta(days=t) for t in range(len(daily_view))]

            # plot daily view series
            axes[fig_idx].plot_date(date_axis, daily_view, 'k-')

            axes[fig_idx].axvline(x=datetime(2015, 10, 23), color=ColorPalette.TOMATO, linestyle='--', lw=1.5, zorder=30)
            axes[fig_idx].text(0.3, 0.95, video_title_list[fig_idx], size=10,
                               transform=axes[fig_idx].transAxes, ha='center', va='bottom')
            axes[fig_idx].tick_params(axis='both', which='major', labelsize=10)
            axes[fig_idx].yaxis.set_major_formatter(FuncFormatter(concise_fmt))
            axes[fig_idx].xaxis.set_major_formatter(mdates.DateFormatter("'%y"))

            fig_idx += 1

    axes[2].set_ylabel('daily views', fontsize=11)
    axes[0].set_title('(b)', fontsize=12)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/intro_teaser.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    T = 63

    main()

