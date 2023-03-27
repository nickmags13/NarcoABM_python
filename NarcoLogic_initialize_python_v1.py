"""
Function for initializing and executing NarcoLogic dynamics from Python
# NarcoLogic for execution from Python #
# Notes for executing:

"""

import numpy as np
import random
from load_expmntl_parms import load_expmntl_parms
from optimize_interdiction_batch import optimize_interdiction_batch
from intrd_tables_batch import intrd_tables_batch
from lldistkm import lldistkm
from extend_network import extend_network
import scipy
from ismember import ismember


def NarcoLogic_initialize_python_v1(mr):
    batchrun = 9
    MRUNS = 30
    ERUNS = 11
    TSTART = 1
    TMAX = 180  # 15 years at monthly time steps

    thistate = scipy.io.loadmat('data/savedrngstate.mat')['thistate']

    testflag = 1
    erun = 4
    mrun = mr

    # Start initialization, set random number generator state for repeatability

    np.random.seed(mrun)

    # load experimental parameters file
    sl_max, sl_min, baserisk, riskmltplr, startstock, sl_learn, rt_learn, losslim, prodgrow, targetseize, \
    intcpctymodel, profitmodel, endstock, growthmdl, timewght, locthink, expandmax, empSLflag, optSLflag, suitflag, \
    extnetflag, rtcap, basecap, p_sucintcpt = load_expmntl_parms(ERUNS)

    # Load landscape files
    dcoast = scipy.io.loadmat('data/coast_dist')['dcoast']
    LANDSUIT = scipy.io.loadmat('data/landsuit_file_default')['LANDSUIT']

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@ Agent Attributes @@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # Interdiction Agent #
    delta_sl = sl_learn[erun]  # reinforcement learning rate for S&L vents (i.e., weight on new information)

    # Network Agent #
    ndto = 2  # initial number of DTOs
    dtocutflag = np.zeros((ndto, 1))
    DTOBDGT = np.zeros((ndto, TMAX))
    losstol = losslim(erun)  # tolerance threshold for loss due to S&L, triggers route fragmentation
    stock_0 = startstock[erun]  # initial cocaine stock at producer node
    stock_max = endstock[erun]
    startvalue = 4500  # producer price, $385/kg: Zoe's numbers 4,500 in Panama
    deltavalue = 4.46  # added value for distance traveled $8/kilo/km: Zoe's numbers $4.46
    nodeloss = 0  # amount of cocaine that is normally lost (i.e., non-interdiction) at each node
    ctrans_inland = 371  # transportation costs (kg/km) over-ground (3.5), includes
    ctrans_coast = 160  # transportation costs (kg/km) via plane or boat (1.5)
    ctrans_air = 3486
    delta_rt = rt_learn[erun]  # reinforcement learning rate for network agent

    # (i.e., weight on new information for successful routes)
    # perceived risk model
    alpharisk = 2
    betarisk = 0.5
    timewght_0 = timewght[erun]

    slprob_0 = 1 / (sum(np.power(timewght_0, np.array((np.arange(0, 13)))) + betarisk))  # CHECK
    bribepct = 0.3
    bribethresh = 12
    rentcap = 1 - bribepct
    edgechange = expandmax[erun] * np.ones((ndto, 1))

    savedState = np.random.getstate()  # CHECK
    np.random.seed(thistate)

    ###################################################################
    #   Build trafficking network - NodeTable and EdgeTable   #
    ###################################################################

    EdgeTable = scipy.io.loadmat('data/EdgeTable.mat')['EdgeTable']
    NodeTable = scipy.io.loadmat('data/NodeTable.mat')['NodeTable']
    EdgeTable['Capacity'] = rtcap[erun] * np.ones(EdgeTable.shape[0], 1)
    nnodes = NodeTable.shape[0]
    mexnode = nnodes
    endnodeset = mexnode
    icoastdist = sub2ind(dcoast.shape, NodeTable.shape[0], NodeTable.shape[1])
    coastdist = dcoast[icoastdist]  # convert to km

    NodeTable['CoastDist'] = coastdist
    NodeTable['CoastDist'][0] = 0
    NodeTable['CoastDist'][nnodes - 1] = 0

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

    if extnetflag == 1:
        ext_NodeTable, ext_EdgeTable = extend_network(nnodes, NodeTable, EdgeTable, Rdptgrid)
        NodeTable = ext_NodeTable
        EdgeTable = ext_EdgeTable
        nnodes = NodeTable.shape[1]
        endnodeset = np.array([mexnode, np.arange(160, nnodes + 1)])
        EdgeTable['Capacity'] = basecap(erun) * np.ones((EdgeTable.shape[1], 1))

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
    NEIHOOD = cell(nnodes, 2)
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
    activeroute = cell(nnodes, TMAX)    # track active routes
    """Check which data structure to use instead of cell"""
    avgslrisk = cell(nnodes, TMAX)  # average S&L risk at each node given active routes
    totslrisk = np.zeros((1, TMAX)) # etwork-wide average S&L risk
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

        if extnetflag == 1:
            # add distance for extended network
            latlon1 = np.matlib.repmat(np.array([NodeTable['Lat'][0], NodeTable['Lon'][0]]), 4, 1)
            latlon2 = np.array([NodeTable['Lat'](np.array([[156], [161], [162], [163]])),
                                NodeTable['Lon'](np.array([[156], [161], [162], [163]]))])
            d1km, d2km = lldistkm(latlon1, latlon2)
            DIST[1, np.array[[[156], [161], [162], [163]]]] = d1km
            latlon1 = np.array([NodeTable['Lat'][157 * np.ones((len(np.where(ADJ[157, :] == 1)), 1))],
                                NodeTable['Lon'][157 * np.ones((len(np.where(ADJ[157, :] == 1)), 1))]])
            latlon2 = np.array([NodeTable['Lat'][np.transpose(np.where(ADJ[158, :] == 1))],
                                NodeTable['Lon'][np.transpose(np.where(ADJ[158, :] == 1))]])
            d1km, d2km = lldistkm(latlon1, latlon2)
            DIST[158, np.transpose[np.where[ADJ[158, :] == 1]]] = d1km
            latlon1 = np.array([NodeTable['Lat'][159 * np.ones((len(np.where(ADJ[159, :] == 1)), 1))],
                                NodeTable['Lon'][159 * np.ones((len(np.where(ADJ[159, :] == 1)), 1))]])
            latlon2 = np.array([NodeTable['Lat'][np.transpose(np.where(ADJ[159, :] == 1))],
                                NodeTable['Lon'][np.transpose(np.where(ADJ[159, :] == 1))]])
            d1km, d2km = lldistkm(latlon1, latlon2)
            DIST[159, np.transpose[np.where[ADJ[159, :] == 1]]] = d1km
            latlon1 = np.array([NodeTable['Lat'][160 * np.ones((len(np.where(ADJ[160, :] == 1)), 1))],
                                NodeTable['Lon'][160 * np.ones((len(np.where(ADJ[160, :] == 1)), 1))]])
            latlon2 = np.array([NodeTable['Lat'][np.transpose(np.where(ADJ[160, :] == 1))],
                                NodeTable['Lon'][np.transpose(np.where(ADJ[160, :] == 1))]])
            d1km, d2km = lldistkm(latlon1, latlon2)
            DIST[160, np.transpose[np.where[ADJ[160, :] == 1]]] = d1km

        # Create added value matrix (USD) and price per node
        ADDVAL[j, ADJ[j, :] == 1] = np.multiply(deltavalue, DIST[j, ADJ[j, :] == 1])
        if j == 1:
            PRICE[j, TSTART] = startvalue
        elif j in endnodeset:
            continue
        elif 157 <= j <= 160:
            isender = EdgeTable['EndNodes'][EdgeTable['EndNodes'][:, 1] == j, 0]
            inextleg = EdgeTable['EndNodes'][EdgeTable['EndNodes'][:, 0] == j, 1]
            PRICE[j, TSTART] = PRICE[isender, TSTART] + ADDVAL[isender, j] + PRICE[isender, TSTART] \
                               + np.mean(ADDVAL(j, inextleg))
            # even prices for long haul routes
            if j == 160:
                PRICE[np.array[[157, 160]], TSTART] = np.amin(PRICE[np.array([157, 160]), TSTART])
                PRICE[np.array[[158, 159]], TSTART] = np.amin(PRICE[np.array([158, 159]), TSTART])
        else:
            isender = EdgeTable['EndNodes'][EdgeTable['EndNodes'][:, 1] == j, 0]
            PRICE[j, TSTART] = np.mean(PRICE[isender, TSTART] + ADDVAL[isender, j])

        for en in range(0, len(endnodeset)):
            PRICE[endnodeset[en], TSTART] = np.amax(PRICE[ADJ[:, endnodeset[en]] == 1, TSTART])




def sub2ind(sz, row, col):
    n_rows = sz[0]
    return [n_rows * (c - 1) + r for r, c in zip(row, col)]
