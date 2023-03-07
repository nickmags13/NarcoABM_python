# Interdiction Initialization #

import numpy as np
import pandas as pd
import math


def intrd_tables_batch(FLOW, slsuccess, SLPROB, NodeTable, EdgeTable, t, testflag, erun, mrun, batchrun):
    Tflow = pd.DataFrame(columns=['End_Node', 'Start_Node', 'IntitFlow', 'DTO'], index=range(1, EdgeTable.shape[0]+1),
                         dtype=float)
    startFLOW = np.add(FLOW[:, :, t-1], slsuccess[:, :, t-1])
    Tintrd = pd.DataFrame(columns=['End_Node', 'Start_Node', 'IntitProb'], index=range(1, EdgeTable.shape[0]+1),
                          dtype=float)

    if t == 1:
        startSLPROB = SLPROB[:, :, 0]
    else:
        startSLPROB = SLPROB[:, :, t-2]

    sumprob = np.sum(startSLPROB)

    for i in range(EdgeTable.shape[0]):
        edge = EdgeTable.iloc[i]["EndNodes"]
        Tflow.iloc[i]["End_Node"] = edge[1]
        Tflow.iloc[i]["Start_Node"] = edge[0]
        Tflow.iloc[i]["IntitFlow"] = startFLOW[edge[0]][edge[1]]
        Tflow.iloc[i]["DTO"] = NodeTable.iloc[edge[1]]["DTO"]

        Tintrd.iloc[i]["End_Node"] = edge[1]
        Tintrd.iloc[i]["Start_Node"] = edge[0]
        Tintrd.iloc[i]["IntitProb"] = startSLPROB[edge[0]][edge[1]] / sumprob

    t1 = int(t >= 100)

    if t >= 100:
        t2 = math.floor((t - 100) / 10)
    else:
        t2 = math.floor(t / 10)

    mrun_t1 = math.floor(mrun / 10)
    mrun_t2 = mrun % 10
    erun_t1 = math.floor(erun/100)
    erun_t2 = math.floor(erun/10)
    erun_t3 = erun % 10

    Tflow.to_excel('../FunctionTesting/Tflow_python.txt')
    Tintrd.to_excel('../FunctionTesting/Tintrd_python.txt')
    breakpoint()

    return Tflow, Tintrd
