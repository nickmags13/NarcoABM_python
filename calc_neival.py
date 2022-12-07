import numpy as np


def calc_neival(c_trans, p_sl, y_node, q_node, lccf, rtpref, tslrisk, dtonei, profmdl, cutflag, totcpcty, totstock,
                edgechange):
    pay_noevent = np.zeros(len(c_trans))
    pay_event = np.zeros(len(c_trans))
    xpay_noevent = np.zeros(len(c_trans))
    ypay_noevent = np.zeros(len(c_trans))
    xpay_event = np.zeros(len(c_trans))
    ypay_event = np.zeros(len(c_trans))
    value_noevent = np.zeros(len(c_trans))
    value_event = np.zeros(len(c_trans))
    ival_noevent = np.zeros(len(c_trans))
    ival_event = np.zeros(len(c_trans))
    dwght_noevent = np.zeros(len(c_trans))
    dwght_event = np.zeros(len(c_trans))
    salwght_noevent = np.zeros(len(c_trans))
    salwght_event = np.zeros(len(c_trans))
    valuex = np.zeros(len(c_trans))
    valuey = np.zeros(len(c_trans))
    iset = np.arange(1, len(c_trans) + 1)

    for i in np.arange(1, len(c_trans)+1).reshape(-1):
        pay_noevent[i] = y_node(i) * q_node(i) - c_trans(i) * q_node(i)  # payoff with no S&L event
        pay_event[i] = y_node(i) * q_node(i) - c_trans(i) * q_node(i) - y_node(i) * q_node(i)  # payoff with S&L event
        xpay_noevent[i] = pay_noevent(i)  # payoff for route A with no S&L event
        xpay_event[i] = pay_event(i)   # payoff for route A with S&L event

    return neipick, neivalue, valuex