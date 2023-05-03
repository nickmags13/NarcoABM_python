# Top-down supply chain optimization ########

import numpy as np
from NarcoLogic import ismember


def optimizeroute_multidto(dtorefvec, subflow, supplyfit, expmax, subroutepref, dtoEdgeTable, dtoSLRISK, dtoADDVAL,
                           dtoCTRANS, losstolval, dtoslsuc):
    iactiveedges = np.logical_or(np.where(subflow > 0), np.where(dtoslsuc > 0))
    actrow = iactiveedges[0]
    actcol = iactiveedges[1]
    edgeparms = np.array([subflow[iactiveedges], dtoSLRISK[iactiveedges], iactiveedges, actrow, actcol])

    if supplyfit < losstolval:  # need to consolidate supply chain
        edgesort = edgeparms[edgeparms[:, 0].argsort()[::-1]]
        # primary movement
        iprimary = np.logical_and(np.where(edgesort[:, 3] == 1), np.where(edgesort[:, 4] != len(dtorefvec)))
        edgecut = np.arange(1, np.amin(np.round(len(iactiveedges) * (supplyfit / (supplyfit + losstolval))),
                                       len(iactiveedges) - 1))

        # Preserve at least one primary movement
        minrisk_primary = np.amin(edgesort[iprimary, 2])
        ikeep_primary = np.where(edgesort[iprimary, 2] == minrisk_primary)
        if len(ikeep_primary) == 1:
            edgecut = edgecut[not ismember(edgecut, np.array(
                [[iprimary[ikeep_primary]], [np.where(edgesort[edgecut, 3] == edgesort[iprimary[ikeep_primary], 4])]]))]
        else:
            maxprofit_primary = np.amax(edgesort[iprimary[ikeep_primary], 1])
            ikeep_primary = ikeep_primary[edgesort[iprimary[ikeep_primary], 1] == maxprofit_primary]
            if len(ikeep_primary) == 1:
                edgecut = edgecut[not ismember(edgecut, np.array([[iprimary[ikeep_primary]],
                                                                  [np.where(edgesort[edgecut, 3] == edgesort[
                                                                      iprimary[ikeep_primary], 5])]]))]
            else:
                ikeep_primary = ikeep_primary[1]
                edgecut = edgecut[not ismember(edgecut, np.array([[iprimary[ikeep_primary]],
                                                                  [np.where(edgesort[edgecut, 4] == edgesort[
                                                                      iprimary[ikeep_primary], 5])]]))]

    return newroutepref
