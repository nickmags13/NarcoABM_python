# Top-down supply chain optimization ########

import numpy as np

def optimizeroute_multidto(dtorefvec, subflow, supplyfit, expmax, subroutepref, dtoEdgeTable, dtoSLRISK, dtoADDVAL,
                           dtoCTRANS, losstolval, dtoslsuc):

    iactiveedges = np.logical_or(np.where(subflow > 0), np.where(dtoslsuc > 0))
    actrow = iactiveedges[0]
    actcol = iactiveedges[1]
    edgeparms = np.array([subflow[iactiveedges], dtoSLRISK[iactiveedges], iactiveedges, actrow, actcol])

    return newroutepref