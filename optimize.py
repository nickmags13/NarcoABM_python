"""   Interdiction events from optimization model   """

import numpy as np


def optimize_interdiction_batch(ADJ):
    trgtfile = 'data/MTMCI_IntNodes.txt'

    Tintevent = np.loadtxt(trgtfile, dtype=int)
    intrdct_events = np.zeros(ADJ.shape)
    intrdct_nodes = Tintevent
    for j in np.arange(0, len(Tintevent)):
        iupstream = (ADJ[:, Tintevent[j] - 1] == 1)
        intrdct_events[iupstream, Tintevent[j] - 1] = 1

    return intrdct_events, intrdct_nodes


# Top-down supply chain optimization ########
def optimizeroute_multidto(dtorefvec, subflow, supplyfit, expmax, subroutepref, dtoEdgeTable, dtoSLRISK, dtoADDVAL,
                           dtoCTRANS, losstolval, dtoslsuc):
    iactiveedges = np.concatenate((np.where(subflow > 0), np.where(dtoslsuc > 0)), axis=1)
    edgeparms = []
    for edge in range(len(iactiveedges[0])):
        edgeparms.append(np.array([subflow[iactiveedges[0][edge], iactiveedges[1][edge]],
                                   dtoSLRISK[iactiveedges[0][edge], iactiveedges[1][edge]], iactiveedges[0][edge],
                                   iactiveedges[1][edge]]))
    edgeparms = np.array(edgeparms)

    if supplyfit < losstolval:  # need to consolidate supply chain
        edgesort = edgeparms[edgeparms[:, 1].argsort()[::-1]]
        # primary movement
        iprimary = list(np.intersect1d(np.where(edgesort[:, 2] == 0)[0],
                                       np.where(edgesort[:, 3] != len(dtorefvec) - 1)[0]))
        upper_lim = min(round(len(iactiveedges[0]) * (supplyfit / (supplyfit + losstolval))), len(iactiveedges[0]) - 1)
        if upper_lim > 0:
            edgecut = np.arange(0, upper_lim)
        else:
            edgecut = []

        # Preserve at least one primary movement
        minrisk_primary = np.amin(edgesort[iprimary, 1])
        ikeep_primary = np.where(edgesort[iprimary, 1] == minrisk_primary)[0]
        if len(ikeep_primary) != 1:
            maxprofit_primary = max(edgesort[iprimary[ikeep_primary], 0])
            ikeep_primary = ikeep_primary[edgesort[iprimary[ikeep_primary], 0] == maxprofit_primary]
            if len(ikeep_primary) != 1:
                ikeep_primary = ikeep_primary[0]

        if len(edgecut) > 0:
            edgecut = np.delete(edgecut, np.intersect1d(edgecut, [iprimary[ikeep_primary[0]]] +
                                                        list(np.where(edgesort[edgecut, 2] ==
                                                                      edgesort[iprimary[ikeep_primary[0]], 3])[0])))
        breakpoint()
        # remove highest risk edges
        for j in range(0, len(edgecut)):
            icheckroute = np.where(subflow(edgesort[edgecut[j], 3],
                                           np.intersect1d(dtorefvec, dtoEdgeTable['EndNodes'].str[1]
                                           [np.where(dtoEdgeTable['EndNodes'].str[0] ==
                                                     dtorefvec[edgesort[edgecut[j], 3]])[0]]))
                                   > 0)
            actroutes = dtoEdgeTable['EndNodes'].str[1][np.where(dtoEdgeTable['EndNodes'].str[0] ==
                                                                 dtorefvec[edgesort[edgecut[j], 3]])[0]]
            checknoderoutes = (
                    len(actroutes[icheckroute]) == len(np.where(edgesort[edgecut, 3] == edgesort[edgecut[j], 3])))
            if checknoderoutes:
                cutsenders = np.where(dtorefvec[np.in1d(dtorefvec,
                                                        dtoEdgeTable['EndNodes'].str[0]
                                                        [np.where(dtoEdgeTable['EndNodes'].str[1] ==
                                                                  dtorefvec[edgesort[edgecut[j], 3]])[0]])]
                                      == 1)
                for i in range(len(cutsenders)):
                    subroutepref[cutsenders[i], edgesort[edgecut[j], 3]] = 0

            if len(icheckroute) == 1:
                subroutepref[edgesort[edgecut[j], 2]] = 0
                irmvsender = (edgesort[:, 4] == edgesort[edgecut[j], 3])
                subroutepref[edgesort[irmvsender, 2]] = 0
            else:
                subroutepref[edgesort[edgecut[j], 2]] = 0

    elif supplyfit >= losstolval:  # need to expand supply chain
        potnodes = np.delete(dtorefvec, dtorefvec[np.in1d(dtorefvec,
                                                          np.array([[1], [dtorefvec[np.unique(edgeparms[:, 3:5])]]]))])
        edgeadd = np.arange(0, np.amin(np.amax(np.ceil(supplyfit / losstolval), 1), np.amin(expmax, len(potnodes))))

        if len(np.where(potnodes)[0]) == 0:
            pass
        else:
            newedgeparms = []
            potsenders = np.unique(edgeparms[:, 3:5])  # dto node index
            potsenders = potsenders[potsenders != len(dtorefvec)]
            for k in range(0, len(potsenders)):
                ipotreceive = np.where(potnodes[np.in1d(potnodes,
                                                        dtoEdgeTable['EndNodes'].str[1]
                                                        [np.where(dtoEdgeTable['EndNodes'].str[0] ==
                                                                  dtorefvec[potsenders[k]])[0]])]
                                       == 1)
                if len(np.where(ipotreceive)[0]) == 0:
                    continue
                ipotedge_col = np.where(dtorefvec[np.in1d(dtorefvec, potnodes[ipotreceive])] == 1)[0]
                for i in range(len(ipotedge_col)):
                    newedgeparms.append([(dtoADDVAL[potsenders[k], ipotedge_col[i]] - dtoCTRANS[potsenders[k],
                                        ipotedge_col[i]]), dtoSLRISK[potsenders[k], ipotedge_col[i]], potsenders[k],
                                         ipotedge_col[i], ipotreceive[i]])
                newedgeparms = np.array(newedgeparms)
                edgesort = newedgeparms[newedgeparms[:, 1].argsort()[::-1]]
                subroutepref[edgesort[edgeadd, 2]] = 1
                ireceivers = dtoEdgeTable.loc[dtoEdgeTable['EndNodes'].str[0][np.in1d(dtoEdgeTable['EndNodes'].str[0],
                                                                                      dtorefvec[edgesort[edgeadd, 3]])]
                , 'EndNodes']
                send_row = []
                rec_col = []
                for jj in range(0, len(ireceivers[:, 0])):
                    send_row = np.array([[send_row], [np.where(dtorefvec[np.in1d(dtorefvec, ireceivers[jj, 0])] == 1)]])
                    rec_col = np.array([[rec_col], [np.where(dtorefvec[np.in1d(dtorefvec, ireceivers[jj, 1])] == 1)]])
                subroutepref[send_row, rec_col] = 1
                subroutepref[rec_col, len(dtorefvec)] = 1

    newroutepref = subroutepref

    return newroutepref


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]
