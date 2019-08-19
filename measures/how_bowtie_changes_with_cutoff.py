#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract the logfile of bow-tie structure of Vevo Network changes as the cutoff value changes.

Usage: python how_bowtie_changes_with_cutoff.py
Input data files: ../data/vevo_forecast_data_60k.tsv, ../data/network_pickle/network_2018-10-01.p
Output data files: ./bowtie_cutoff.log
Time: ~2H
"""

import os, sys, pickle, logging
from tarjan import tarjan
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.data_loader import DataLoader
from utils.bowtie import is_in_component, is_out_component


def main():
    # == == == == == == Part 1: Set up environment == == == == == == #
    timer = Timer()
    timer.start()

    data_prefix = '../data'
    snapshot_date = '2018-10-01'
    snapshot_filename = 'network_{0}.p'.format(snapshot_date)

    # == == == == == == Part 2: Load video views == == == == == == #
    data_loader = DataLoader()
    data_loader.load_video_views()
    embed_view_dict = data_loader.embed_view_dict
    # get the views on 2018-10-01
    embed_view_dict = {embed: ts_view[30] for embed, ts_view in embed_view_dict.items()}
    total_views = sum(embed_view_dict.values())
    num_videos = data_loader.num_videos

    # == == == == == == Part 3: Load network snapshot as cutoff value changes == == == == == == #
    for cutoff in range(5, 51):
        nodes_set = set()
        num_edges = 0
        embedded_graph = defaultdict(list)
        with open(os.path.join(data_prefix, 'network_pickle', snapshot_filename), 'rb') as fin:
            network_dict = pickle.load(fin)
            # embed_tar: [(embed_src, pos_src, view_src)]
            for embed_tar in range(num_videos):
                for embed_src, pos_src, _ in network_dict[embed_tar]:
                    if pos_src < cutoff:
                        embedded_graph[embed_src].append(embed_tar)
                        nodes_set.add(embed_src)
                        nodes_set.add(embed_tar)
                        num_edges += 1

        logging.info('>>> Graph embedding @ cutoff {0} has been loaded!'.format(cutoff))
        logging.info('>>> {0} nodes and {1} edges in the graph'.format(len(nodes_set), num_edges))
        logging.info('    {0} views throughout the graph'.format(total_views))

        # == == == == == == Part 4: Extract bow-tie structure == == == == == == #
        scc_content = tarjan(embedded_graph)
        scc_content = sorted(scc_content, key=lambda x: len(x), reverse=True)

        # largest SCC
        largest_scc = scc_content.pop(0)
        logging.info('>>> {0} ({1:.2f}%) nodes in the largest SCC'.format(len(largest_scc), len(largest_scc) / num_videos * 100))
        largest_scc_views = sum([embed_view_dict[embed] for embed in largest_scc])
        logging.info('    {0} ({1:.2f}%) views in the largest SCC'.format(largest_scc_views, largest_scc_views / total_views * 100))

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
        in_views = sum([embed_view_dict[embed] for embed in in_component])
        logging.info('    {0} ({1:.2f}%) views in the IN component'.format(in_views, in_views / total_views * 100))

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
        out_views = sum([embed_view_dict[embed] for embed in out_component])
        logging.info('    {0} ({1:.2f}%) views in the OUT component'.format(out_views, out_views / total_views * 100))

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
        tendrils_views = sum([embed_view_dict[embed] for embed in tendrils])
        logging.info('    {0} ({1:.2f}%) views in the Tendrils'.format(tendrils_views, tendrils_views / total_views * 100))

        logging.info('>>> {0} ({1:.2f}%) nodes in the Disconnected'.format(num_disconnected + len(disconnected), (num_disconnected + len(disconnected)) / num_videos * 100))
        logging.info('    {0} scc in the Disconnected'.format(num_disconnected + num_scc_disconnected))
        disc_views = total_views - largest_scc_views - in_views - out_views - tendrils_views
        logging.info('    {0} ({1:.2f}%) views in the Disconnected'.format(disc_views, disc_views / total_views * 100))

        print('>>> Finish computing bowtie at cutoff {0}...'.format(cutoff))

    timer.stop()


if __name__ == '__main__':
    logging.basicConfig(filename='bowtie_cutoff.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

    main()
