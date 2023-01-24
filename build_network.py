"""
Method to Build trafficking network
"""

import numpy as np


def build_network(ca_adm0, Rcagrid_cntry, dptgrid, Rdptgrid, LANDSUIT, dptcodes, dptorder, savedState, stock_0):
    # Set-up producer and end supply nodes

    strow = ca_adm0.shape[0]
    stcol = ca_adm0.shape[1]
    edrow = 100
    edcol = 100
