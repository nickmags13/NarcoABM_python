# Initialize NarcoLogic for execution from Python #
# Notes for executing:


import numpy as np
import random
from load_expmntl_parms import load_expmntl_parms


def NarcoLogic_initialize_python_v1(mr):
    argout = 'Dummy output'

    batchrun = 9
    MRUNS = 30
    ERUNS = 11
    TSTART = 1
    TMAX = 180

    testflag = 1
    erun = 4
    mrun = mr

    # Start initialization, set random number generator state for repeatability

    random.seed(mrun)

    # load experimental parameters file
    sl_max,sl_min,baserisk,riskmltplr,startstock,sl_learn,rt_learn,losslim,prodgrow,targetseize,intcpctymodel,profitmodel,endstock,growthmdl,timewght,locthink,expandmax,empSLflag,optSLflag,suitflag,extnetflag,rtcap,basecap,p_sucintcpt = load_expmntl_parms(ERUNS)

    return argout
