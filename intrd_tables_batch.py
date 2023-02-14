# Interdiction Initialization #

import numpy as np
import pandas as pd


def intrd_tables_batch(FLOW, slsuccess, SLPROB, NodeTable, EdgeTable, t, testflag, erun, mrun, batchrun):
    Tflow = pd.DataFrame(columns=['End_Node', 'Start_Node', 'IntitFlow', 'DTO'], index=range(1, EdgeTable.shape[0]),
                         dtype=float)
    startFLOW = FLOW[:][:][t] + slsuccess[:][:][t]
    Tintrd = pd.DataFrame(columns=['End_Node', 'Start_Node', 'IntitProb'], index=range(1, EdgeTable.shape[0]),
                          dtype=float)

    if t == 1:
        startSLPROB = SLPROB[:][:][0]
    else:
        startSLPROB = SLPROB[:][:][t - 2]

    sumprob = np.sum(startSLPROB)

    return Tflow, Tintrd
