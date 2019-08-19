#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract the logfile of bow-tie structure of Vevo Network evolves over time.

Usage: python how_bowtie_evolves.py
Input data files: ../data/vevo_forecast_data_60k.tsv, ../data/network_pickle/
Output data files: ./bowtie_evolves.log
Time: ~3H
"""

import os, sys, pickle, logging
from datetime import datetime, timedelta
from tarjan import tarjan
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, obj2str
from utils.data_loader import DataLoader
from utils.bowtie import is_in_component, is_out_component


def main():
    # == == == == == == Part 1: Set up environment == == == == == == #
    timer = Timer()
    timer.start()

    data_prefix = '../data'

    # == == == == == == Part 2: Load video views == == == == == == #
    data_loader = DataLoader()
    data_loader.load_video_views()
    embed_view_dict = data_loader.embed_view_dict
    num_videos = data_loader.num_videos
    total_views = []

    # == == == == == == Part 3: Load network snapshot as cutoff value changes == == == == == == #
    for t in range(T):
        total_views.append(sum([embed_view_dict[embed][t] for embed in range(num_videos)]))

        snapshot_date = obj2str(datetime(2018, 9, 1) + timedelta(days=t))
        snapshot_filename = 'network_{0}.p'.format(snapshot_date)
        nodes_set = set()
        num_edges = 0
        embedded_graph = defaultdict(list)
        with open(os.path.join(data_prefix, 'network_pickle', snapshot_filename), 'rb') as fin:
            network_dict = pickle.load(fin)
            # embed_tar: [(embed_src, pos_src, view_src)]
            for embed_tar in range(num_videos):
                for embed_src, pos_src, _ in network_dict[embed_tar]:
                    if pos_src < CUTOFF:
                        embedded_graph[embed_src].append(embed_tar)
                        nodes_set.add(embed_src)
                        nodes_set.add(embed_tar)
                        num_edges += 1

        logging.info('>>> Graph embedding @ date {0} has been loaded!'.format(snapshot_date))
        logging.info('>>> {0} nodes and {1} edges in the graph'.format(len(nodes_set), num_edges))
        logging.info('    {0} views throughout the graph'.format(total_views[t]))

        # == == == == == == Part 4: Extract bow-tie structure == == == == == == #
        scc_content = tarjan(embedded_graph)
        scc_content = sorted(scc_content, key=lambda x: len(x), reverse=True)

        # largest SCC
        largest_scc = scc_content.pop(0)
        logging.info('>>> {0} ({1:.2f}%) nodes in the largest SCC'.format(len(largest_scc), len(largest_scc) / num_videos * 100))
        largest_scc_views = sum([embed_view_dict[embed][t] for embed in largest_scc])
        logging.info('    {0} ({1:.2f}%) views in the largest SCC'.format(largest_scc_views, largest_scc_views / total_views[t] * 100))

        # find IN, OUT, Tendrils, Disconnected
        in_component = []
        num_scc_in = 0
        to_visit_scc = []
        for scc in scc_content:
            ret = is_in_component(scc, embedded_graph, largest_scc)
            if ret:
                in_component.extend(scc)
                num_scc_in += 1
            else:
                to_visit_scc.append(scc)
        logging.info('>>> {0} ({1:.2f}%) nodes in the IN component'.format(len(in_component), len(in_component) / num_videos * 100))
        logging.info('    {0} scc in the IN component'.format(num_scc_in))
        in_views = sum([embed_view_dict[embed][t] for embed in in_component])
        logging.info('    {0} ({1:.2f}%) views in the IN component'.format(in_views, in_views / total_views[t] * 100))

        out_component = []
        num_scc_out = 0
        to_visit_scc2 = []
        for scc in to_visit_scc:
            ret = is_out_component(scc, embedded_graph, largest_scc)
            if ret:
                out_component.extend(scc)
                num_scc_out += 1
            else:
                to_visit_scc2.append(scc)
        logging.info('>>> {0} ({1:.2f}%) nodes in the OUT component'.format(len(out_component), len(out_component) / num_videos * 100))
        logging.info('    {0} scc in the OUT component'.format(num_scc_out))
        out_views = sum([embed_view_dict[embed][t] for embed in out_component])
        logging.info('    {0} ({1:.2f}%) views in the OUT component'.format(out_views, out_views / total_views[t] * 100))

        tendrils = []
        num_scc_tendrils = 0
        disconnected = []
        num_disconnected = num_videos - len(nodes_set)
        num_scc_disconnected = 0
        for scc in to_visit_scc2:
            ret = is_out_component(scc, embedded_graph, in_component)
            if ret:
                tendrils.extend(scc)
                num_scc_tendrils += 1
            else:
                ret = is_in_component(scc, embedded_graph, out_component)
                if ret:
                    tendrils.extend(scc)
                    num_scc_tendrils += 1
                else:
                    disconnected.extend(scc)
                    num_scc_disconnected += 1
        logging.info('>>> {0} ({1:.2f}%) nodes in the Tendrils'.format(len(tendrils), len(tendrils) / num_videos * 100))
        logging.info('    {0} scc in the Tendrils'.format(num_scc_tendrils))
        tendrils_views = sum([embed_view_dict[embed][t] for embed in tendrils])
        logging.info('    {0} ({1:.2f}%) views in the Tendrils'.format(tendrils_views, tendrils_views / total_views[t] * 100))

        logging.info('>>> {0} ({1:.2f}%) nodes in the Disconnected'.format(num_disconnected + len(disconnected), (num_disconnected + len(disconnected)) / num_videos * 100))
        logging.info('    {0} scc in the Disconnected'.format(num_disconnected + num_scc_disconnected))
        disc_views = total_views[t] - largest_scc_views - in_views - out_views - tendrils_views
        logging.info('    {0} ({1:.2f}%) views in the Disconnected'.format(disc_views, disc_views / total_views[t] * 100))

        print('>>> Finish computing bowtie at day {0}...'.format(t + 1))

    timer.stop()


if __name__ == '__main__':
    T = 63
    CUTOFF = 15
    logging.basicConfig(filename='bowtie_evolves.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

    main()
