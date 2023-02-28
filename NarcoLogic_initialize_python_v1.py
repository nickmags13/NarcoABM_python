# Initialize NarcoLogic for execution from Python #
# Notes for executing:


import numpy as np
import random
from load_expmntl_parms import load_expmntl_parms
import scipy


def NarcoLogic_initialize_python_v1(mr):
    argout = 'Dummy output'

    batchrun = 9
    MRUNS = 30
    ERUNS = 11
    TSTART = 1
    TMAX = 180  # 15 years at monthly time steps

    testflag = 1
    erun = 4
    mrun = mr

    # Start initialization, set random number generator state for repeatability

    random.seed(mrun)

    # load experimental parameters file
    sl_max, sl_min, baserisk, riskmltplr, startstock, sl_learn, rt_learn, losslim, prodgrow, targetseize, \
    intcpctymodel, profitmodel, endstock, growthmdl, timewght, locthink, expandmax, empSLflag, optSLflag, suitflag, \
    extnetflag, rtcap, basecap, p_sucintcpt = load_expmntl_parms(ERUNS)

    # Load landscape files
    scipy.io.loadmat('coast_dist')  # Check the file format and if it can be changed to non .mat file
    scipy.io.loadmat('landsuit_file_default')

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@ Agent Attributes @@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # Interdiction Agent #
    delta_sl = sl_learn(erun)  # reinforcement learning rate for S&L vents (i.e., weight on new information)

    # Network Agent #
    ndto = 2  # initial number of DTOs
    dtocutflag = np.zeros((ndto, 1))
    DTOBDGT = np.zeros((ndto, TMAX))
    losstol = losslim(erun)  # tolerance threshold for loss due to S&L, triggers route fragmentation
    stock_0 = startstock(erun)  # initial cocaine stock at producer node
    stock_max = endstock(erun)
    startvalue = 4500  # producer price, $385/kg: Zoe's numbers 4,500 in Panama
    deltavalue = 4.46  # added value for distance traveled $8/kilo/km: Zoe's numbers $4.46
    nodeloss = 0  # amount of cocaine that is normally lost (i.e., non-interdiction) at each node
    ctrans_inland = 371  # transportation costs (kg/km) over-ground (3.5), includes
    ctrans_coast = 160  # transportation costs (kg/km) via plane or boat (1.5)
    ctrans_air = 3486
    delta_rt = rt_learn(erun)   # reinforcement learning rate for network agent

    return argout
