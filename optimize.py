"""   Interdiction events from optimization model   """

import os
import glob
import numpy as np


def optimize_interdiction_batch(ADJ):

    trgtfile = 'data/MTMCI_IntNodes.txt'

    Tintevent = np.loadtxt(trgtfile, dtype=int)
    intrdct_events = np.zeros(ADJ.shape)
    intrdct_nodes = Tintevent
    for j in np.arange(0, len(Tintevent)):
        iupstream = (ADJ[:, Tintevent[j]-1] == 1)
        intrdct_events[iupstream, Tintevent[j]-1] = 1

    return intrdct_events, intrdct_nodes
