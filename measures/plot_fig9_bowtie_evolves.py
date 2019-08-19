#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot how bow-tie structure evolves over time.
Note: need run 'python how_bowtie_evolves.py' to generate ./bowtie_evolves.log

Usage: python plot_fig9_bowtie_evolves.py
Input data files: ./bowtie_evolves.log
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

    x_axis = range(63)
    lscc_structure, in_structure, out_structure, tendrils_structure, disc_structure = [], [], [], [], []
    structure_list = [lscc_structure, in_structure, out_structure, tendrils_structure, disc_structure]
    lscc_attention, in_attention, out_attention, tendrils_attention, disc_attention = [], [], [], [], []
    attention_list = [lscc_attention, in_attention, out_attention, tendrils_attention, disc_attention]
    label_list = ['LSCC', 'IN', 'OUT', 'Tendrils', 'Disconnected']

    with open('bowtie_evolves.log', 'r') as fin:
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

        axes[0, col_idx].set_title(label_list[col_idx], fontsize=13)

    for col_idx in range(2):
        structure_min = min(structure_list[col_idx])
        structure_max = max(structure_list[col_idx])
        attention_min = min(attention_list[col_idx])
        attention_max = max(attention_list[col_idx])

        axes[0, col_idx].axhline(y=structure_min, color='grey', lw=1, linestyle='--')
        axes[0, col_idx].text(0.5, (structure_min + 0.02) / 1.1, 'min={0:.4f}'.format(structure_min), size=11,
                              transform=axes[0, col_idx].transAxes, ha='center', va='top')

        axes[0, col_idx].axhline(y=structure_max, color='grey', lw=1, linestyle='--')
        axes[0, col_idx].text(0.5, (structure_max + 0.06) / 1.1, 'max={0:.4f}'.format(structure_max), size=11,
                              transform=axes[0, col_idx].transAxes, ha='center', va='bottom')

        axes[1, col_idx].axhline(y=attention_min, color='grey', lw=1, linestyle='--')
        axes[1, col_idx].text(0.5, (attention_min + 0.02) / 1.1, 'min={0:.4f}'.format(attention_min), size=11,
                              transform=axes[1, col_idx].transAxes, ha='center', va='top')

        axes[1, col_idx].axhline(y=attention_max, color='grey', lw=1, linestyle='--')
        axes[1, col_idx].text(0.5, (attention_max + 0.06) / 1.1, 'max={0:.4f}'.format(attention_max), size=11,
                              transform=axes[1, col_idx].transAxes, ha='center', va='bottom')

    axes[1, 2].set_xlabel('day', fontsize=13)
    axes[0, 0].set_ylabel('percent of nodes', fontsize=13)
    axes[1, 0].set_ylabel('percent of views', fontsize=13)
    axes[0, 0].set_ylim([-0.05, 1.05])

    for ax in axes.ravel():
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout(w_pad=1.7)
    plt.savefig('../images/measure_bowtie_evolves.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
