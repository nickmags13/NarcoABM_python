# Interdiction Initialization #

import numpy as np
import pandas as pd


def intrd_tables_batch(FLOW, slsuccess, SLPROB, NodeTable, EdgeTable, t, testflag, erun, mrun, batchrun):

    Tflow = pd.DataFrame(columns=['End_Node', 'Start_Node', 'IntitFlow', 'DTO'], index=range(1, EdgeTable))
    return Tflow, Tintrd
