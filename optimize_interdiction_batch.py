"""   Interdiction events from optimization model   """

import os
import glob
import numpy as np


def optimize_interdiction_batch(ADJ):

    trgtfile = 'data/MTMCI_IntNodes.txt'

    """
    readflag = 0
    print('Looking for ' + trgtfile)

    fileSearch = trgtfile.split('.')
    fnames = glob.glob(fileSearch[0]+"*"+fileSearch[1])
    for file in fnames:
        if os.path.getsize(trgtfile) > 0:
            readflag = 1

    print('Interdiction Input File Success, t= ' + t)
    """

    Tintevent = np.loadtxt(trgtfile, dtype=int)
    intrdct_events = np.zeros(ADJ.shape)
    intrdct_nodes = Tintevent
    for j in np.arange(0, len(Tintevent)):
        iupstream = (ADJ[:, Tintevent[j]-1] == 1)
        intrdct_events[iupstream, Tintevent[j]-1] = 1

    return intrdct_events, intrdct_nodes
