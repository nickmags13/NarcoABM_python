"""   Interdiction events from optimization model   """

import os
import glob
import numpy as np


def optimize_interdiction_batch(t, ADJ, testflag, erun, mrun, batchrun):

    readflag = 0
    trgtfile = 'MTMCI_IntNodes.txt'
    print('Looking for ' + trgtfile)

    fileSearch = trgtfile.split('.')
    fnames = glob.glob(fileSearch[0]+"*"+fileSearch[1])
    for file in fnames:
        if os.path.getsize(trgtfile) > 0:
            readflag = 1

    print('Interdiction Input File Success, t= ' + t)
    Tintevent = np.loadtxt(trgtfile)
    intrdct_events = np.zeros((ADJ.shape))
    intrdct_nodes = Tintevent
    for j in np.arange(0,len(Tintevent)):       # Check whether it needs len or shape
        iupstream = (ADJ[:, Tintevent[j]] == 1)
        intrdct_events[iupstream, Tintevent[j]] = 1

    return intrdct_events, intrdct_nodes
