#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract forecast data.
Windows of observation: 2018-09-01 - 2018-11-02 (9 weeks, 63 days)

Usage: python extract_forecast_tsv.py
Input data files: ../data/vevo_en_videos_60k.json
Output data files: ../data/vevo_forecast_data_60k.tsv, ../data/vevo_en_embeds_60k.txt
Time: ~1M
"""

import sys, os, json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, strify


def main():
    # == == == == == == Part 1: Set up experiment parameters == == == == == == #
    timer = Timer()
    timer.start()

    data_prefix = '../data/'
    vevo_en_videos_path = 'vevo_en_videos_60k.json'
    vevo_forecast_filepath = 'vevo_forecast_data_60k.tsv'
    vevo_embed_filepath = 'vevo_en_embeds_60k.txt'

    # == == == == == == Part 2: Load Vevo en forecast data == == == == == == #
    vevo_en_vid_list = []
    vid_title_dict = {}
    vid_forecast_view_dict = {}

    with open(os.path.join(data_prefix, vevo_en_videos_path), 'r') as fin:
        for line in fin:
            video_json = json.loads(line.rstrip())
            vid = video_json['id']
            vevo_en_vid_list.append(vid)

            title = (video_json['snippet']['title'].encode('ascii', 'ignore')).decode('utf-8')
            vid_title_dict[vid] = title

            daily_view = video_json['insights']['dailyView']
            forecast_view = daily_view[-T:]
            vid_forecast_view_dict[vid] = forecast_view

    vevo_en_vid_list = sorted(vevo_en_vid_list)
    num_videos = len(vevo_en_vid_list)

    with open(os.path.join(data_prefix, vevo_forecast_filepath), 'w') as fout:
        for embed in range(num_videos):
            vid = vevo_en_vid_list[embed]
            forecast_view = vid_forecast_view_dict[vid]
            fout.write('{0}\t{1}\t{2}\t{3}\n'.format(embed, vid, strify(forecast_view, delimiter=','), np.sum(forecast_view)))

    with open(os.path.join(data_prefix, vevo_embed_filepath), 'w', encoding='utf-8') as fout:
        for embed in range(num_videos):
            vid = vevo_en_vid_list[embed]
            fout.write('{0},{1},{2}\n'.format(embed, vid, vid_title_dict[vid]))

    timer.stop()


if __name__ == '__main__':
    T = 63

    main()
