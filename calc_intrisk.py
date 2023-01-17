import numpy as np


def calc_intrisk(sloccur, t_eff, alpharisk, betarisk, timeweight):
    slevnt = np.sum(np.multiply(sloccur, np.tile(np.transpose(np.power(timeweight, t_eff)), (1, len(sloccur[1, :])))),
                    0, keepdims=True)  # make sure to load t_eff as 2D array in order for this to work properly
    tmevnt = np.sum(np.power(timeweight, t_eff))
    sl_risk = (slevnt + alpharisk) / (tmevnt + alpharisk + betarisk)  # Check for ./ in python and
    # / equivalent to np.divide
    return sl_risk, slevnt, tmevnt
