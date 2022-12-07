import numpy as np


def calc_neival(c_trans, p_sl, y_node, q_node, lccf, rtpref, tslrisk, dtonei, profmdl, cutflag, totcpcty, totstock,
                edgechange):
    pay_noevent = np.zeros(len(c_trans), 1)
    pay_event = np.zeros(len(c_trans), 1)
    xpay_noevent = np.zeros(len(c_trans), 1)
    ypay_noevent = np.zeros(len(c_trans), 1)
    xpay_event = np.zeros(len(c_trans), 1)
    ypay_event = np.zeros(len(c_trans), 1)
    value_noevent = np.zeros(len(c_trans), 1)
    value_event = np.zeros(len(c_trans), 1)
    ival_noevent = np.zeros(len(c_trans), 1)
    ival_event = np.zeros(len(c_trans), 1)
    dwght_noevent = np.zeros(len(c_trans), 1)
    dwght_event = np.zeros(len(c_trans), 1)
    salwght_noevent = np.zeros(len(c_trans), 1)
    salwght_event = np.zeros(len(c_trans), 1)
    valuex = np.zeros(len(c_trans), 1)
    valuey = np.zeros(len(c_trans), 1)
    iset = np.arange(1, len(c_trans) + 1)

    for i in np.arange(1, len(c_trans)+1):
        pay_noevent[i] = y_node(i) * q_node(i) - c_trans(i) * q_node(i)  # payoff with no S&L event
        pay_event[i] = y_node(i) * q_node(i) - c_trans(i) * q_node(i) - y_node(i) * q_node(i)  # payoff with S&L event
        xpay_noevent[i] = pay_noevent(i)  # payoff for route A with no S&L event
        xpay_event[i] = pay_event(i)   # payoff for route A with S&L event

    for i in np.arange(1, len(c_trans) + 1):
        inset = np.where(dtonei == dtonei(i))
        ypay_noevent[i] = np.mean(pay_noevent(inset != i))
        ypay_event[i] = np.mean(pay_event(inset != i))
        value_noevent[i] = np.abs(ypay_noevent(i) - xpay_noevent(i)) / (
                    np.abs(ypay_noevent(i)) + np.abs(xpay_noevent(i)) + 1)

    return neipick, neivalue, valuex