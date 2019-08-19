#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Calculate the probability of forming a persistent link within width w in n consecutive days.
Each link is formed with probability p, and w= 7.
Probability is computed via simulation.

Usage: python justify_persistent_link.py
Output data files: ./justify_persistent_link.log
Time: ~3H for 100,000 simulation
"""

import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, is_persistent_link


def simulate_linkage_list(n, p):
    q = 1 - p
    ret = []
    for _ in range(n):
        ret.append(np.random.choice(np.arange(2), p=[q, p]))
    return ret


def simulate_for_prob(p, n=63, num_sim=10000):
    num_persistent = 0
    num_nonpersistent = 0

    sim_cnt = 0
    while sim_cnt < num_sim:
        sim_arr = simulate_linkage_list(n, p)
        if is_persistent_link(sim_arr):
            num_persistent += 1
        else:
            num_nonpersistent += 1
        sim_cnt += 1

    return num_persistent / num_sim


def main():
    timer = Timer()
    timer.start()

    n = 63
    # change to 10,000 for faster computing
    num_sim = 10000

    with open('./justify_persistent_link.log', 'w') as fout:
        for p in np.arange(0, 1.01, 0.01):
            fout.write('p_form: {0:.2f}, p_persistent_link: {1:.4f}\n'.format(p, simulate_for_prob(p, n, num_sim)))
            print('>>> Finish simulating at prob {0:.2f}'.format(p))

    timer.stop()


if __name__ == '__main__':
    main()
