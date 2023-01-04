import warnings
import numpy as np

warnings.filterwarnings("ignore")


def extend_network(nnodes, NodeTable, EdgeTable, Rdptgrid):
    ext_NodeTable = NodeTable
    ext_EdgeTable = EdgeTable

    # Add intermediate nodes
    # Caribbean
    inn1 = nnodes + 1
    ext_NodeTable[inn1, 1] = np.array([inn1])

    return ext_NodeTable, ext_EdgeTable
