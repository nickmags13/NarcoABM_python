import numpy as np


def calc_neival(c_trans, p_sl, y_node, q_node, lccf, rtpref, tslrisk, dtonei, profmdl, cutflag, totcpcty, totstock,
                edgechange):
    pay_noevent = np.zeros((c_trans.shape[1], 1))  # len in numpy defaults to 1st dimension and shape provides a tuple
    pay_event = np.zeros((c_trans.shape[1], 1))
    xpay_noevent = np.zeros((c_trans.shape[1], 1))
    ypay_noevent = np.zeros((c_trans.shape[1], 1))
    xpay_event = np.zeros((c_trans.shape[1], 1))
    ypay_event = np.zeros((c_trans.shape[1], 1))
    value_noevent = np.zeros((c_trans.shape[1], 1))
    value_event = np.zeros((c_trans.shape[1], 1))
    ival_noevent = np.zeros((c_trans.shape[1], 1))
    ival_event = np.zeros((c_trans.shape[1], 1))
    dwght_noevent = np.zeros((c_trans.shape[1], 1))
    dwght_event = np.zeros((c_trans.shape[1], 1))
    salwght_noevent = np.zeros((c_trans.shape[1], 1))
    salwght_event = np.zeros((c_trans.shape[1], 1))
    valuex = np.zeros((c_trans.shape[1], 1))
    valuey = np.zeros((c_trans.shape[1], 1))
    iset = np.arange(0, c_trans.shape[1])

    for i in np.arange(0, c_trans.shape[1]):
        pay_noevent[i, 0] = y_node[i, 0] * q_node[0, i] - c_trans[0, i] * q_node[0, i]  # payoff with no S&L event
        pay_event[i, 0] = y_node[i, 0] * q_node[0, i] - c_trans[0, i] * q_node[0, i] - y_node[i, 0] * q_node[
            0, i]  # payoff with S&L event
        xpay_noevent[i, 0] = pay_noevent[i, 0]  # payoff for route A with no S&L event
        xpay_event[i, 0] = pay_event[i, 0]  # payoff for route A with S&L event

    for i in np.arange(0, c_trans.shape[1]):
        inset = np.where(dtonei == dtonei[i, 0])[0]
        mask = np.not_equal(inset, i)
        ypay_noevent[i, 0] = np.mean([pay_noevent[j, 0] for j in range(len(mask)) if mask[j]])
        ypay_event[i, 0] = np.mean([pay_event[j, 0] for j in range(len(mask)) if mask[j]])
        value_noevent[i, 0] = np.abs(ypay_noevent[i, 0] - xpay_noevent[i, 0]) / (np.abs(ypay_noevent[i, 0]) +
                                                                                 np.abs(xpay_noevent[i, 0]) + 1)
        value_event[i, 0] = np.abs(ypay_event[i, 0] - xpay_event[i, 0]) / (np.abs(ypay_event[i, 0]) +
                                                                           np.abs(xpay_event[i, 0]) + 1)
        ipntlval = np.flip(np.argsort(np.array([value_noevent[i, 0], value_event[i, 0]])))  # CHECK
        ival_noevent[i, 0] = ipntlval[0]
        ival_event[i, 0] = ipntlval[1]
        dwght_noevent[i, 0] = (lccf ** ival_noevent[i, 0]) / ((lccf ** ival_noevent[i, 0]) * (1 - p_sl[0, i]) +
                                                              (lccf ** ival_event[i, 0]) * p_sl[0, i])
        dwght_event[i, 0] = (lccf ** ival_event[i, 0]) / ((lccf ** ival_noevent[i, 0]) * (1 - p_sl[0, i]) +
                                                          (lccf ** ival_event[i, 0]) * p_sl[0, i])
        salwght_noevent[i, 0] = (1 - p_sl[0, i]) * dwght_noevent[i, 0]
        salwght_event[i, 0] = p_sl[0, i] * dwght_event[i, 0]
        valuey[i, 0] = salwght_noevent[i, 0] * ypay_noevent[i, 0] + salwght_event[i, 0] * ypay_event[i, 0]
        valuex[i, 0] = salwght_noevent[i, 0] * xpay_noevent[i, 0] + salwght_event[i, 0] * xpay_event[i, 0]

    breakpoint()
    # Selection based on maximize profits while less than average S&L risk
    rankroute = np.sort(np.array(
        [np.multiply(np.transpose(rtpref), valuex), np.transpose(p_sl), np.transpose(q_node), np.transpose(iset),
         dtonei, np.transpose(totcpcty)]), axis=0)[::-1]  # CHECK

    dtos = np.unique(dtonei(dtonei != 0))

    icut = []  # moved the initialization out from if loop
    if len(dtos) > 1:
        for j in np.arange(0, len(dtos)):
            idto = np.where(rankroute[:, 5] == dtos[j])
            if profmdl == 1:
                if len(np.where(valuex(dtonei == dtos[j]) > 0, 1)) == 1:  # CHECK
                    subicut = np.transpose(
                        (np.arange(1, np.where(np.cumsum(rankroute[idto, 6]) >= totstock, edgechange[j],
                                               'first') + 1)))  # CHECK whether to start from 0 and should not have +1
                elif len(np.where(rankroute[idto, 1] > 0, 1)) == 1:
                    subicut = np.where(rankroute[idto, 1] >= 0, edgechange[j], 'first')
                elif len(np.where(np.cumsum(rankroute[idto, 6]) >= totstock, 1)) == 1:
                    subicut = np.where(rankroute[idto, 1] >= 0, edgechange[j], 'first')
                else:
                    subicut = np.where(rankroute[idto, 1] >= 0, edgechange[j], 'first')
                if cutflag(dtos[j]) == 1:
                    subicut = []
                icut = np.array([icut], [idto[subicut]])
            elif profmdl == 2:
                if len(np.where(valuex(dtonei == dtos[j]) > 0, 1)) == 1:
                    subicut = np.transpose(
                        (np.arange(1, np.where(np.cumsum(rankroute[idto, 6]) >= totstock, edgechange(j),
                                               'first') + 1)))
                elif len(np.where(np.cumsum(rankroute[idto, 1]) > 0, 1)) == 1:
                    subicut = np.where(rankroute[idto, 1] >= 0, edgechange[j], 'first')
                elif len(np.where(np.cumsum(rankroute[idto, 6]) >= totstock, 1)) == 1:
                    subicut = np.where(rankroute[idto, 1] >= 0, edgechange[j], 'first')
                else:
                    subicut = np.where(rankroute[idto, 1] >= 0, edgechange[j], 'first')  # CHECK why all else
                    # subicut are same
                if cutflag(dtos[j]) == 1:
                    subicut = []
                icut = np.array([icut], [idto[subicut]])
            if rankroute[rankroute[:, 5] == 0, 1] > 0:
                icut = np.array([[icut], [np.where(rankroute[:, 5] == 0)]])
    else:
        if profmdl == 1:
            if len(np.where(valuex > 0, 1)) == 1:
                if len(np.where(valuex > 0, 1)) == 1:
                    icut = np.transpose(
                        (np.arange(1, np.where(np.cumsum(rankroute[:, 6]) >= totstock, 1, 'first') + 1)))
                elif len(np.where(rankroute[:, 1] > 0, 1)) == 1:
                    volcut = np.transpose((np.arange(1, np.where(np.cumsum(rankroute[:, 6]) >= totstock, 1,
                                                                 'first') + 1)))
                    valcut = np.where(rankroute[:, 1] >= 0)
                    icut = np.isin(valcut, volcut)
                elif len(np.find(np.cumsum(rankroute[:, 6]) >= totstock, 1)) == 1:
                    icut = np.where(rankroute[:, 1] >= 0)
                else:
                    icut = np.where(rankroute[:, 1] >= 0)
        elif profmdl == 2:
            if len(np.where(valuex > 0, 1)) == 1:
                icut = np.transpose((np.arange(1, np.where(np.cumsum(rankroute[:, 6]) >= totstock, 1, 'first') + 1)))
            elif len(np.where(np.cumsum(rankroute[:, 1]) > 0, 1)) == 1:
                volcut = np.transpose((np.arange(1, np.where(np.cumsum(rankroute[:, 6]) >= totstock, 1, 'first') + 1)))
                valcut = np.where(np.cumsum(rankroute[:, 1]) >= 0)
                icut = np.isin(valcut, volcut)
            elif len(np.where(np.cumsum(rankroute[:, 6]) >= totstock, 1)) == 1:
                icut = np.where(rankroute[:, 1] >= 0)
            else:
                icut = np.where(rankroute[:, 1] >= 0)

    neipick = rankroute[icut, 4]
    neivalue = rankroute[icut, 1]

    return neipick, neivalue, valuex
