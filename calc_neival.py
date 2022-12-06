import numpy as np


def calc_neival(c_trans, p_sl, y_node, q_node, lccf, rtpref, tslrisk, dtonei, profmdl, cutflag, totcpcty, totstock,
                edgechange):
    pay_noevent = np.zeros((len(c_trans), 1))
    pay_event = np.zeros((len(c_trans), 1))
    xpay_noevent = np.zeros((len(c_trans), 1))
    ypay_noevent = np.zeros((len(c_trans), 1))
    xpay_event = np.zeros((len(c_trans), 1))
    ypay_event = np.zeros((len(c_trans), 1))
    value_noevent = np.zeros((len(c_trans), 1))
    value_event = np.zeros((len(c_trans), 1))
    ival_noevent = np.zeros((len(c_trans), 1))
    ival_event = np.zeros((len(c_trans), 1))
    dwght_noevent = np.zeros((len(c_trans), 1))
    dwght_event = np.zeros((len(c_trans), 1))
    salwght_noevent = np.zeros((len(c_trans), 1))
    salwght_event = np.zeros((len(c_trans), 1))
    valuex = np.zeros((len(c_trans), 1))
    valuey = np.zeros((len(c_trans), 1))
    iset = np.arange(1, len(c_trans) + 1)
