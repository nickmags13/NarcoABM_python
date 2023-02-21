"""
Function for executing NarcoLogic dynamics from Python
"""

import numpy as np
from optimize_interdiction_batch import optimize_interdiction_batch
from intrd_tables_batch import intrd_tables_batch


def NarcoLogic_dynamics_python_v1(t):
    # Check what variables NarcoLogic_wrksp.mat is loading and add code to load these variables

    intrdct_events, intrdct_nodes = optimize_interdiction_batch(t, ADJ, testflag, erun, mrun, batchrun)
    slevent[:][:][t] = intrdct_events
    
