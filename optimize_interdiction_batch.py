"""   Interdiction events from optimization model   """

import os


def optimize_interdiction_batch(t, ADJ, testflag, erun, mrun, batchrun):

    if testflag == 1:
        os.chdir("C:\\Users\\pcbmi\\Box\\NSF_D-ISN\\Code\\NarcoLogic\\MTMCI_IntNodes")
    else:
        if batchrun == 1:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_1")
        elif batchrun == 2:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_2")
        elif batchrun == 3:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_3")
        elif batchrun == 4:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_4")
        elif batchrun == 5:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_5")
        elif batchrun == 6:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_6")
        elif batchrun == 7:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_7")
        elif batchrun == 8:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_8")
        elif batchrun == 9:
            os.chdir("C:\\Users\\pcbmi\\Box\\NSF_D-ISN\\Data\\IntegratedModels\\INT_Nodes")
        elif batchrun == 10:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_10")
        elif batchrun == 11:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_11")
        elif batchrun == 12:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_12")
        elif batchrun == 13:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_13")
        elif batchrun == 14:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_14")
        elif batchrun == 15:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_15")
        elif batchrun == 16:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_16")
        elif batchrun == 17:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_17")
        elif batchrun == 18:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_18")
        elif batchrun == 19:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_19")
        elif batchrun == 20:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_20")
        elif batchrun == 21:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_21")
        elif batchrun == 22:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_22")
        elif batchrun == 23:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_23")
        elif batchrun == 24:
            os.chdir("D:\\NSF_EAGER_Models\\INT_Nodes\\INT_Nodes_24")

    readflag = 0
    trgtfile = 'MTMCI_IntNodes.txt'
    print('Looking for ' + trgtfile)

    while readflag == 0:
        fnames = os.listdir()




    return intrdct_events, intrdct_nodes