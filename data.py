import os
import time
import pandas as pd


def data_sourcing():
    nodes_fp = r'C:\Users\pcbmi\Box\NSF_D-ISN\Code\NarcoLogic\TrialResults\Nodes_2.csv'
    if os.path.exists(nodes_fp):
        nodes = pd.read_csv(nodes_fp)
        nodes.drop(nodes.columns[[0]], axis=1, inplace=True)
        print('Nodes_2 found!')
    else:
        nodes = pd.DataFrame.spatial.from_featureclass(
            r'C:\Users\pcbmi\Box\NSF_D-ISN\Data\DISN.gdb\IllicitNetworks.gdb\NodeInfo_cln')
        nodes.drop(['OBJECTID'], axis=1, inplace=True)
        nodes['Timestep'] = 1
    return nodes



