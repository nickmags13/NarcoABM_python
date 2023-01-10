import numpy as np


def load_expmntl_parms(ERUNS):
    empSLflag = np.zeros((1, ERUNS))  # determines is empirical (1) or artificial (0) S&L schedule used
    optSLflag = np.ones((1, ERUNS))
    extnetflag = np.ones((1, ERUNS))
    suitflag = np.zeros((1, ERUNS))  # use RAT suitability (1) or build from covariates (0)

    sl_max = 125 * np.ones((1, ERUNS))  # baseline; maximum interdiction capacity

    sl_min = np.ceil(sl_max / 6)  # baseline; minimum interdiction capacity

    """
    CCDB year 2015 there were an average of 206 events (Carib and EP
    combined) per month, and a total volume of 1,592 MT. Assuming 25%
    underreporting in both events and volume, 258 events per month and a
    total volume of 1,990 MT or 166 MT per month. That would give an average
    capcaity of 644 kilos per movement per month.
    In 2015, 28%/72% of primary movements were in the Carib/East Pacific.
    Given the average monthly flow of 166 MT, 129,480 kg in EPac and 46,480 in
    Carib.
    With an average of 166MT per month and 156 over-land nodes, average
    capacity to handle off of that flow would be 1,071 kg
    """
    basecap = 32000 * np.ones((1, ERUNS))
    rtcap = np.array([[1, 1, 1, 1, 1, 1, 1.79, 2.55, 2.41, 2.01, 1.0, 1.03, 1.57, 2.06, 2.1, 3.07, 7.11, 6.44, 6.34],
                      [1, 1, 1, 1, 1, 1, 1.15, 1.4, 1.35, 1.2, 1.0, 1.31, 1.36, 1.46, 1.1, 1.13, 1.63, 1.49, 2.14]])
    low = np.linspace(0.1, 10, 6)
    high = np.linspace(10, 20, 6)
    p_sucintcpt = np.array([low, high(np.arange(1, 6))])

    baserisk = 0.43 * np.ones((1, ERUNS))
    riskmltplr = 2 * np.ones((1, ERUNS))


    return sl_max, sl_min, baserisk, riskmltplr, startstock, sl_learn, rt_learn, \
           losslim, prodgrow, targetseize, intcpctymodel, profitmodel, endstock, growthmdl, \
           timewght, locthink, expandmax, empSLflag, optSLflag, suitflag, extnetflag, rtcap, \
           basecap, p_sucintcpt
