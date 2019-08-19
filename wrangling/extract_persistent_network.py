#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract persistent network by removing ephemeral links and adding missing links.
Two filters:
1. at least 100 daily views for target video
2. the mean daily views of source video is at least 1% of the target video

Usage: python extract_persistent_network.py
Input data files: ../data/vevo_forecast_data_60k.tsv, ../data/network_pickle/
Output data files: ../data/persistent_network.csv
Time: ~7M
"""

import sys, os, pickle
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.data_loader import DataLoader
from utils.helper import Timer, obj2str, is_persistent_link, is_same_genre


def main():
    # == == == == == == Part 1: Set up environment == == == == == == #
    timer = Timer()
    timer.start()

    data_prefix = '../data/'

    # == == == == == == Part 2: Load video views == == == == == == #
    data_loader = DataLoader()
    data_loader.load_video_views()
    embed_avg_view_dict = data_loader.embed_avg_view_dict
    num_videos = data_loader.num_videos
    data_loader.load_embed_content_dict()
    embed_cid_dict = data_loader.embed_cid_dict
    embed_genre_dict = data_loader.embed_genre_dict

    # == == == == == == Part 3: Load dynamic network snapshot == == == == == == #
    network_dict_list = []
    for t in range(T):
        target_date_str = obj2str(datetime(2018, 9, 1) + timedelta(days=t))
        filename = 'network_{0}.p'.format(target_date_str)
        network_dict = pickle.load(open(os.path.join(data_prefix, 'network_pickle', filename), 'rb'))
        for embed in network_dict:
            network_dict[embed] = [x[0] for x in network_dict[embed] if x[1] < NUM_REL]
        network_dict_list.append(network_dict)

    persistent_src_embed_set = set()
    persistent_tar_embed_set = set()
    existing_edges = set()
    num_reciprocal_edges = 0
    num_same_artist = 0
    num_same_genre = 0

    with open(os.path.join(data_prefix, 'persistent_network.csv'), 'w') as fout:
        fout.write('Source,Target\n')

        for tar_embed in range(num_videos):
            src_union_set = set()
            for t in range(T):
                src_union_set.update(set(network_dict_list[t][tar_embed]))

            for src_embed in src_union_set:
                linkage_list = [0] * T
                for t in range(T):
                    if src_embed in network_dict_list[t][tar_embed]:
                        linkage_list[t] = 1
                if is_persistent_link(linkage_list):
                    # filter: at least 100 daily views for target video,
                    # and the mean daily views of source video is at least 1% of the target video
                    src_mean = embed_avg_view_dict[src_embed]
                    tar_mean = embed_avg_view_dict[tar_embed]

                    if tar_mean >= 100 and src_mean >= 0.01 * tar_mean:
                        fout.write('{0},{1}\n'.format(src_embed, tar_embed))
                        persistent_src_embed_set.add(src_embed)
                        persistent_tar_embed_set.add(tar_embed)
                        if '{1}-{0}'.format(src_embed, tar_embed) in existing_edges:
                            num_reciprocal_edges += 1
                        if embed_cid_dict[src_embed] == embed_cid_dict[tar_embed]:
                            num_same_artist += 1
                        if is_same_genre(embed_genre_dict[src_embed], embed_genre_dict[tar_embed]):
                            num_same_genre += 1
                        existing_edges.add('{0}-{1}'.format(src_embed, tar_embed))

    print('{0} edges in the persistent network'.format(len(existing_edges)))
    print('{0} source videos, {1} target videos, {2} videos appear in both set'.format(len(persistent_src_embed_set),
                                                                                       len(persistent_tar_embed_set),
                                                                                       len(persistent_src_embed_set.intersection(persistent_tar_embed_set))))
    print('{0} pairs of reciprocal edges'.format(num_reciprocal_edges))
    print('{0} ({1:.1f}%) edges belong to the same artist'.format(num_same_artist, 100 * num_same_artist / len(existing_edges)))
    print('{0} ({1:.1f}%) edges belong to the same genre'.format(num_same_genre, 100 * num_same_genre / len(existing_edges)))

    timer.stop()


if __name__ == '__main__':
    NUM_REL = 15
    T = 63

    main()
