"""
Function for initializing and executing NarcoLogic dynamics from Python
# NarcoLogic for execution from Python #
# Notes for executing:

"""

import numpy as np
import pandas as pd

from load_expmntl_parms import load_expmntl_parms
from optimize_interdiction_batch import optimize_interdiction_batch
from intrd_tables_batch import intrd_tables_batch
from lldistkm import lldistkm
import scipy
from ismember import ismember


def NarcoLogic_initialize_python_v1(mr):
    batchrun = 9
    MRUNS = 30
    ERUNS = 11
    TSTART = 1
    TMAX = 180  # 15 years at monthly time steps

    # thistate = scipy.io.loadmat('data/savedrngstate.mat')['thistate']
    thistate = 1  # Verify the correct file upload and remove it after testing

    testflag = 1
    erun = 3
    mrun = mr

    # Start initialization, set random number generator state for repeatability

    np.random.seed(mrun)

    # load experimental parameters file
    sl_max, sl_min, baserisk, riskmltplr, startstock, sl_learn, rt_learn, losslim, prodgrow, targetseize, \
    intcpctymodel, profitmodel, endstock, growthmdl, timewght, locthink, expandmax, empSLflag, optSLflag, suitflag, \
    rtcap, basecap, p_sucintcpt = load_expmntl_parms(ERUNS)

    # Load landscape files
    dcoast = scipy.io.loadmat('data/coast_dist')['dcoast']
    LANDSUIT = scipy.io.loadmat('data/landsuit_file_default')['LANDSUIT']

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@ Agent Attributes @@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # Interdiction Agent #
    delta_sl = sl_learn[0, erun]  # reinforcement learning rate for S&L vents (i.e., weight on new information)

    # Network Agent #
    ndto = 2  # initial number of DTOs
    dtocutflag = np.zeros((ndto, 1))
    DTOBDGT = np.zeros((ndto, TMAX))
    losstol = losslim[0, erun]  # tolerance threshold for loss due to S&L, triggers route fragmentation
    stock_0 = startstock[0, erun]  # initial cocaine stock at producer node
    stock_max = endstock[0, erun]
    startvalue = 4500  # producer price, $385/kg: Zoe's numbers 4,500 in Panama
    deltavalue = 4.46  # added value for distance traveled $8/kilo/km: Zoe's numbers $4.46
    nodeloss = 0  # amount of cocaine that is normally lost (i.e., non-interdiction) at each node
    ctrans_inland = 371  # transportation costs (kg/km) over-ground (3.5), includes
    ctrans_coast = 160  # transportation costs (kg/km) via plane or boat (1.5)
    ctrans_air = 3486
    delta_rt = rt_learn[0, erun]  # reinforcement learning rate for network agent

    # (i.e., weight on new information for successful routes)
    # perceived risk model
    alpharisk = 2
    betarisk = 0.5
    timewght_0 = timewght[0, erun]

    slprob_0 = 1 / (np.sum(np.power(timewght_0, np.array((np.arange(0, 13))))) + betarisk)
    bribepct = 0.3
    bribethresh = 12
    rentcap = 1 - bribepct
    edgechange = expandmax[0, erun] * np.ones((ndto, 1))

    savedState = np.random.get_state()  # CHECK
    np.random.seed(thistate)

    ###################################################################
    #   Build trafficking network - NodeTable and EdgeTable   #
    ###################################################################

    EdgeTable = pd.read_csv('data/EdgeTable.csv')
    EdgeTable['EndNodes'] = EdgeTable[['EndNodes_1', 'EndNodes_2']].values.tolist()
    EdgeTable = EdgeTable.drop(columns=['EndNodes_1', 'EndNodes_2'])
    NodeTable = pd.read_csv('data/NodeTable.csv')
    NodeTable['Row'] = NodeTable['Row'] - 1
    NodeTable['Col'] = NodeTable['Col'] - 1
    EdgeTable['Capacity'] = rtcap[0, erun] * np.ones(EdgeTable.shape[0])
    nnodes = NodeTable.shape[0]
    mexnode = nnodes
    endnodeset = mexnode
    # icoastdist = sub2ind(dcoast.shape, NodeTable['Row'], NodeTable['Col'])
    # coastdist = dcoast[icoastdist]  # convert to km
    coastdist = [dcoast[row][col] for row, col in zip(NodeTable['Row'], NodeTable['Col'])]
    NodeTable['CoastDist'] = coastdist
    NodeTable['CoastDist'][0] = 0
    NodeTable['CoastDist'][nnodes - 1] = 0
    breakpoint()
    # Assign nodes to initials DTOs
    # CHECK the variable assignments in the for loop ####
    for nn in range(1, nnodes - 2):
        westdir = NodeTable['Col'][nn] - np.where(np.isnan(dcoast[NodeTable['Row'][nn],
                                                                  np.arange(0, NodeTable['Col'][nn] - 2)]) == 1)[-1]
        eastdir = np.where(np.isnan(dcoast[NodeTable['Row'][nn], np.arange(NodeTable['Col'][nn] + 1,
                                                                           LANDSUIT.shape[1] + 1)]) == 1)[0]
        northdir = NodeTable['Row'][nn] - np.where(np.isnan(dcoast[np.arange(0, NodeTable['Row'][nn] - 2),
                                                                   NodeTable['Col'][nn]]) == 1)[-1]
        southdir = np.where(np.isnan(dcoast[np.arange(NodeTable['Row'][nn] + 1, LANDSUIT.shape[0] + 1),
                                            NodeTable['Col'][nn]]) == 1)[0]
        mindist, imindist = np.amin(np.array([westdir, eastdir, northdir, southdir]))
        if westdir < 2.5 * eastdir:
            NodeTable['DTO'][nn] = 1
        else:
            NodeTable['DTO'][nn] = 2

    dptcodes = scipy.io.loadmat('data/dptcodes.mat')['dptcodes']
    dptgrid = scipy.io.loadmat('data/dptgrid.mat')['dptgrid']
    Rdptgrid = scipy.io.loadmat('data/Rdptgrid.mat')['Rdptgrid']  # Geographic cells reference - check format in python

    ADJ = np.zeros((nnodes, nnodes))  # adjacency matrix for trafficking network
    TRRTY = np.zeros((nnodes, nnodes))  # control of nodes by each DTO
    DIST = np.zeros((nnodes, nnodes))  # geographic distance associated with edges
    ADDVAL = np.zeros((nnodes, nnodes))  # added value per edge in trafficking network
    WGHT = np.ones((nnodes, nnodes))  # dynamic weighting of edges
    FLOW = np.zeros((nnodes, nnodes, TMAX))  # dynamic flows of cocaine between nodes
    SLRISK = slprob_0 * np.ones((nnodes, nnodes))  # dynamic perceived risk of seisure and loss per edge by node agent
    INTRISK = np.zeros((nnodes, TMAX))  # dynamic perceived risk of interdiction at each node
    CPCTY = np.zeros((nnodes, nnodes))  # maximum flow possible between nodes
    CTRANS = np.zeros((nnodes, nnodes, TMAX))  # ransportation costs between nodes
    RMTFAC = np.zeros((nnodes, nnodes))  # landscape factor (remoteness) influencing S&L risk
    COASTFAC = np.zeros((nnodes, nnodes))  # landscape factor (distance to coast) influencing S&L risk
    LATFAC = np.zeros((nnodes, nnodes))  # decreased likelihood of S&L moving north to reflect greater DTO investment
    BRDRFAC = np.zeros((nnodes, nnodes))  # increased probability of S&L in department bordering an
    # international border

    SUITFAC = np.zeros((nnodes, nnodes))
    """Check which data structure to use instead of cell"""
    NEIHOOD = np.empty((nnodes, 2))
    STOCK = np.zeros((nnodes, TMAX))  # dynamic cocaine stock at each node
    PRICE = np.zeros((nnodes, TMAX))  # $/kilo at each node
    RISKPREM = np.ones((nnodes, nnodes, TMAX))
    INFLOW = np.zeros((nnodes, TMAX))  # dynamic stock of cocaine coming into at each node
    OUTFLOW = np.zeros((nnodes, TMAX))  # dynamic stock of cocaine leaving from each node
    TOTCPTL = np.zeros((nnodes, TMAX))  # total value of cocaine at each node
    ICPTL = np.zeros((nnodes, TMAX))  # dynamic illicit capital accumulated at each node
    LCPTL = np.zeros((nnodes, TMAX))  # dynamic legitimate capital accumulated at each node
    BRIBE = np.zeros((nnodes, TMAX))  # annual bribe payments made at each node to maintain control
    MARGIN = np.zeros((nnodes, TMAX))  # gross profit per node after purchasing, trafficking, and selling
    RENTCAP = np.zeros((nnodes, TMAX))  # portion of MARGIN retained at node as profit
    LEAK = np.zeros((nnodes, TMAX))  # dynamic amount of cocaine leaked at each node
    """Check which data structure to use instead of cell"""
    activeroute = np.empty((nnodes, TMAX))  # track active routes
    """Check which data structure to use instead of cell"""
    avgslrisk = np.empty((nnodes, TMAX))  # average S&L risk at each node given active routes
    totslrisk = np.zeros((1, TMAX))  # etwork-wide average S&L risk
    slcpcty = np.zeros((1, TMAX))

    np.random.seed(savedState)
    hitrngstate = np.random.rand(nnodes, 1)

    for k in range(0, nnodes):
        # Create adjacency matrix
        ADJ[k, EdgeTable['EndNodes'][EdgeTable['EndNodes'][:, 1] == k, 2]] = 1

    # Node Attributes
    remotefac = np.array([[0], [1 - NodeTable['PopSuit'][np.arange(1, nnodes + 1)]]])
    brdrfac = np.array([[0], [NodeTable['DistBorderSuit'][np.arange(1, nnodes + 1)]]])
    suitfac = np.array([[0], [NodeTable['LandSuit'][np.arange(1, nnodes + 1)]]])
    coastfac = np.array([[0], [NodeTable['CoastDist'][np.arange(1, nnodes + 1)] / np.amax(NodeTable.CoastDist)]])
    nwvec = np.sqrt(np.multiply(0.9, NodeTable['Lat'][np.arange(1, nnodes + 1)] ** 2) +
                    np.multiply(0.1, NodeTable['Lon'][np.arange(1, nnodes + 1)] ** 2))
    latfac = np.array([[0], [1 - nwvec / np.amax(nwvec)]])

    # Create adjacency matrix
    iendnode = NodeTable['ID'][NodeTable['DeptCode'] == 2]
    ADJ[EdgeTable['EndNodes'][EdgeTable['EndNodes'][:, 2] == iendnode, 1], iendnode] = 1
    iedge = np.where(ADJ == 1)
    subneihood = np.zeros((LANDSUIT.shape[0], LANDSUIT.shape[1]))

    for j in range(0, nnodes):
        # Create weight and capacity matrices
        WGHT[j, ADJ[j, :] == 1] = EdgeTable['Weight'][ADJ[j, :] == 1]
        CPCTY[j, ADJ[j, :] == 1] = EdgeTable['Capacity'][ADJ[j, :] == 1]
        # Create distance (in km) matrix
        latlon2 = np.array([NodeTable['Lat'][ADJ[j, :] == 1], NodeTable['Lon'][ADJ[j, :] == 1]])
        latlon1 = np.matlib.repmat(np.array([NodeTable['Lat'][j], NodeTable['Lon'][j]]), len(latlon2[:, 1]), 1)
        d1km, d2km = lldistkm(latlon1, latlon2)
        DIST[j, ADJ[j, :] == 1] = d1km

        # Create added value matrix (USD) and price per node
        ADDVAL[j, ADJ[j, :] == 1] = np.multiply(deltavalue, DIST[j, ADJ[j, :] == 1])
        if j == 1:
            PRICE[j, TSTART] = startvalue
        elif j in endnodeset:
            continue
        elif 157 <= j <= 160:
            isender = EdgeTable['EndNodes'][EdgeTable['EndNodes'][:, 1] == j, 0]
            inextleg = EdgeTable['EndNodes'][EdgeTable['EndNodes'][:, 0] == j, 1]
            PRICE[j, TSTART] = PRICE[isender, TSTART] + ADDVAL[isender, j] + PRICE[isender, TSTART] + np.mean(
                ADDVAL[j, inextleg])
            # even prices for long haul routes
            if j == 160:
                PRICE[np.array([157, 160]), TSTART] = np.amin(PRICE[np.array([157, 160]), TSTART])
                PRICE[np.array([158, 159]), TSTART] = np.amin(PRICE[np.array([158, 159]), TSTART])
        else:
            isender = EdgeTable['EndNodes'][EdgeTable['EndNodes'][:, 1] == j, 0]
            PRICE[j, TSTART] = np.mean(PRICE[isender, TSTART] + ADDVAL[isender, j])

        for en in range(0, len(endnodeset)):
            PRICE[endnodeset[en], TSTART] = np.amax(PRICE[ADJ[:, endnodeset[en]] == 1, TSTART])
        RMTFAC[j, ADJ[j, :] == 1] = remotefac[ADJ[j, :] == 1]
        COASTFAC[j, ADJ[j, :] == 1] = coastfac[ADJ[j, :] == 1]
        LATFAC[j, ADJ[j, :] == 1] = latfac[ADJ[j, :] == 1]
        BRDRFAC[j, ADJ[j, :] == 1] = brdrfac[ADJ[j, :] == 1]
        SUITFAC[j, ADJ[j, :] == 1] = suitfac[ADJ[j, :] == 1]

        # Transportation costs
        ireceiver = EdgeTable['EndNodes'][EdgeTable['EndNodes'][:, 1] == j, 2]
        idist_ground = np.logical_and(DIST[j, ireceiver] > 0, DIST[j, ireceiver] <= 500)
        idist_air = (DIST[j, ireceiver] > 500)
        CTRANS[j, ireceiver[idist_ground], TSTART] = np.multiply(ctrans_inland,
                                                                 DIST[j, ireceiver[idist_ground]]) / DIST[0, mexnode]
        CTRANS[j, ireceiver[idist_air], TSTART] = np.multiply(ctrans_air,
                                                              DIST[j, ireceiver[idist_air]]) / DIST[0, mexnode]
        if NodeTable['CoastDist'][j] < 20 or 157 <= j <= 159:
            ireceiver = EdgeTable['EndNodes'][EdgeTable['EndNodes'][:, 1] == j, 1]
            idist_coast = (NodeTable['CoastDist'][ireceiver] < 20)
            idist_inland = (NodeTable['CoastDist'][ireceiver] >= 20)
            CTRANS[j, ireceiver[idist_coast], TSTART] = np.multiply(ctrans_coast,
                                                                    DIST[j, ireceiver[idist_coast]]) / DIST[0, mexnode]
            if 157 <= j <= 159:
                CTRANS[j, ireceiver[idist_coast], TSTART] = 0
                CTRANS[0, j, TSTART] = CTRANS[0, j, TSTART] + np.mean(
                    np.multiply(ctrans_coast, DIST[j, ireceiver[idist_coast]]) / DIST[1, mexnode])

    # Initialize Interdiction agent
    # Create S&L probability layer
    routepref = np.zeros((nnodes, nnodes, TMAX))  # weighting by network agent of successful routes
    slevent = np.zeros((nnodes, nnodes, TMAX))  # occurrence of S&L event
    intrdctobs = np.zeros((nnodes, nnodes, TMAX))
    slnodes = np.empty((1, TMAX))
    slsuccess = np.zeros((nnodes, nnodes, TMAX))  # volume of cocaine seized in S&L events
    slvalue = np.zeros((nnodes, nnodes, TMAX))  # value of cocaine seized in S&L events
    slcount_edges = np.zeros((1, TMAX))
    slcount_vol = np.zeros((1, TMAX))
    INTRDPROB = np.zeros((nnodes, TMAX))
    SLPROB = np.zeros((nnodes, nnodes, TMAX))  # dynamic probability of S&L event per edge

    facmat = LATFAC
    facmat[:, :, 1] = COASTFAC
    facmat[:, :, 2] = RMTFAC
    facmat[:, :, 3] = DIST / np.amax(np.amax(DIST))
    facmat[:, :, 4] = BRDRFAC
    facmat[:, :, 5] = SUITFAC
    SLPROB[:, :, TSTART] = np.mean(facmat[:, :, range(0, 6)], 2)
    SLPROB[:, :, TSTART + 1] = SLPROB[:, :, TSTART]
    slmin = SLPROB[:, :, TSTART]
    INTRDPROB[:, TSTART + 1] = slprob_0 * np.ones((nnodes, 1))  # dynamic probability of interdiction at nodes

    # Initialize Node agents
    STOCK[:, TSTART] = NodeTable['Stock']
    TOTCPTL[:, TSTART] = NodeTable['Capital']
    PRICE[:, TSTART + 1] = PRICE[:, TSTART]
    slcpcty_0 = sl_min[0, erun]
    slcpcty_max = sl_max[0, erun]
    slcpcty[TSTART + 1] = slcpcty_0

    # subjective risk perception with time distortion
    twght = timewght_0 * np.ones((nnodes, 1))

    # Set-up trafficking netowrk benefit-cost logic  ############
    ltcoeff = locthink[0, erun] * np.ones((nnodes, 1))
    margval = np.zeros((nnodes, nnodes, TMAX))
    for q in range(0, nnodes):
        if len(np.where(ADJ[q, :] == 1)) > 0:
            continue
        margval[q, range(q, nnodes), TSTART] = PRICE[range(q, nnodes), TSTART] - PRICE[q, TSTART]

    for nd in range(0, ndto):
        idto = np.where(NodeTable['DTO'] == nd)
        margvalset = idto[not ismember(idto, endnodeset)]
        routepref[1, idto, TSTART + 1] = margval[1, idto] / np.amax(margval[1, margvalset])

    routepref[:, endnodeset, TSTART + 1] = 1
    totslrisk[TSTART + 1] = 1

    OWN = np.zeros((LANDSUIT.shape[0], LANDSUIT.shape[1]))  # node agent land ownership
    IOWN = np.empty((nnodes, TMAX))  # dynamic list of owned parcels
    CTRANS[:, :, TSTART + 1] = CTRANS[:, :, TSTART]

    # Set-up figure for trafficking movie
    MOV = np.zeros((nnodes, nnodes, TMAX))

    # Output tables for flows(t) and interdiction prob(t-1)
    t = TSTART + 1
    FLOW = scipy.io.loadmat('init_flow')['FLOW']

    rinit, cinit = ind2sub(np.array([nnodes, nnodes]), np.where(FLOW[:, :, 1] > 0))

    for w in range(0, len(rinit)):
        MOV[rinit[w], cinit[w], 1] = FLOW[rinit[w], cinit[w], 1]

    Tflow, Tintrd = intrd_tables_batch(FLOW, slsuccess, SLPROB, NodeTable, EdgeTable, t, testflag, erun, mrun, batchrun)


def sub2ind(sz, row, col):
    n_rows = sz[0]
    return [n_rows * (c - 1) + r for r, c in zip(row, col)]


def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1])
    return rows, cols
