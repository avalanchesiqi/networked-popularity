#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot how bow-tie structure changes with cutoff.
Note: need run 'python how_bowtie_changes_with_cutoff.py' to generate ./bowtie_cutoff.log

Usage: python plot_fig7_bowtie_changes_with_cutoff.py
Input data files: ./bowtie_cutoff.log
Time: ~1M
"""

import os, sys, platform

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.plot import ColorPalette


def extract_percentage(line):
    return float(line[line.find('(') + 1: line.find(')') - 1]) / 100


def main():
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.5), sharex='all', sharey='all')
    tomato = ColorPalette.TOMATO
    cornflower_blue = ColorPalette.BLUE

    x_axis = range(5, 51)
    lscc_structure, in_structure, out_structure, tendrils_structure, disc_structure = [], [], [], [], []
    structure_list = [lscc_structure, in_structure, out_structure, tendrils_structure, disc_structure]
    lscc_attention, in_attention, out_attention, tendrils_attention, disc_attention = [], [], [], [], []
    attention_list = [lscc_attention, in_attention, out_attention, tendrils_attention, disc_attention]
    label_list = ['LSCC', 'IN', 'OUT', 'Tendrils', 'Disconnected']

    with open('bowtie_cutoff.log', 'r') as fin:
        for line in fin:
            if 'nodes in the largest SCC' in line:
                lscc_structure.append(extract_percentage(line))
            elif 'views in the largest SCC' in line:
                lscc_attention.append(extract_percentage(line))
            elif 'nodes in the IN component' in line:
                in_structure.append(extract_percentage(line))
            elif 'views in the IN component' in line:
                in_attention.append(extract_percentage(line))
            elif 'nodes in the OUT component' in line:
                out_structure.append(extract_percentage(line))
            elif 'views in the OUT component' in line:
                out_attention.append(extract_percentage(line))
            elif 'nodes in the Tendrils' in line:
                tendrils_structure.append(extract_percentage(line))
            elif 'views in the Tendrils' in line:
                tendrils_attention.append(extract_percentage(line))
            elif 'nodes in the Disconnected' in line:
                disc_structure.append(extract_percentage(line))
            elif 'views in the Disconnected' in line:
                disc_attention.append(extract_percentage(line))

    for col_idx in range(len(structure_list)):
        axes[0, col_idx].plot(x_axis, structure_list[col_idx], color=cornflower_blue, lw=1.5)
        axes[1, col_idx].plot(x_axis, attention_list[col_idx], color=cornflower_blue, lw=1.5)

        axes[0, col_idx].scatter(x_axis[10], structure_list[col_idx][10], s=15, c=tomato, edgecolors='k', zorder=30)
        axes[1, col_idx].scatter(x_axis[10], attention_list[col_idx][10], s=15, c=tomato, edgecolors='k', zorder=30)
        axes[0, col_idx].text(x_axis[10], structure_list[col_idx][10], '{0:.4f}'.format(structure_list[col_idx][10]),
                              size=11, ha='left', va='bottom')
        axes[1, col_idx].text(x_axis[10], attention_list[col_idx][10], '{0:.4f}'.format(attention_list[col_idx][10]),
                              size=11, ha='left', va='bottom')

        axes[0, col_idx].set_title(label_list[col_idx], fontsize=13)

    axes[1, 2].set_xlabel('cutoff on relevant list', fontsize=13)
    axes[0, 0].set_ylabel('percent of nodes', fontsize=13)
    axes[1, 0].set_ylabel('percent of views', fontsize=13)
    axes[0, 0].set_ylim([-0.05, 1.05])

    for ax in axes.ravel():
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout(w_pad=0.2)
    plt.savefig('../images/measure_bowtie_cutoff.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
