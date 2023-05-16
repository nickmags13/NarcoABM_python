import os
import time
import pandas as pd
from gurobipy as gp


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


def data_processing(Tflow, i, mr):
    Nodes = data_sourcing()
    strt = time.time()
    # cast the returned TFlow table and timestep to py variables
    # input will be tflow for t = n x mr 1-n; need to group t by model run; doublecheck the tflow output
    tf2 = Tflow
    NLtimestep = i

    # convert object type column values in TFlow to float by first converting to string and omitting special characters
    tf2['End_Node'] = tf2['End_Node'].astype(str)
    tf2['End_Node'] = tf2['End_Node'].str.replace(r'[(),]', "")
    tf2['End_Node'] = tf2['End_Node'].astype(float)
    tf2['Start_Node'] = tf2['Start_Node'].astype(str)
    tf2['Start_Node'] = tf2['Start_Node'].str.replace(r'[(),]', "")
    tf2['Start_Node'] = tf2['Start_Node'].astype(float)
    tf2['IntitFlow'] = tf2['IntitFlow'].astype(str)
    tf2['IntitFlow'] = tf2['IntitFlow'].str.replace(r'[(),]', "")
    tf2['IntitFlow'] = tf2['IntitFlow'].astype(float).round(2)
    tf2['DTO'] = tf2['DTO'].astype(str)
    tf2['DTO'] = tf2['DTO'].str.replace(r'[(),]', "")
    tf2['DTO'] = tf2['DTO'].astype(float)  # can't be int bc there are 0 values
    tf2['MR'] = mr

    # combine Nodes with tf2 (tflow table) and write to new df
    Nodes_2 = Nodes.merge(tf2, on=['End_Node'], how='left')
    ActiveLinks = Nodes.merge(tf2, on=['End_Node'], how='left')

    # Get the Type value by node stored as a dict by ID e.g. {1: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # 2: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    ID = Nodes_2['ID']
    Types = Nodes_2.filter(regex='^Type', axis=1)
    ID_Type = Types.join(ID, how='right')
    ID_TypeDict = ID_Type.set_index('ID').T.to_dict('list')

    # Sum the positive type values by ID e.g. {2: 1.0, 3: 1.0, 4: 2.0,...} to divide the updated flow by per node.
    SumType = {k: sum(v) for k, v in ID_TypeDict.items()}

    # convert TypeSum dict to column
    Nodes_2['SumType'] = Nodes_2['ID'].map(SumType)

    # Read Interdicted Nodes text file to list; write to comp IntNodes
    with open(r'C:\Users\pcbmi\Box\NSF_D-ISN\Code\NarcoLogic\MTMCI_IntNodes\MTMCI_IntNodes.txt', "r") as f:
        IntList = [int(x) for x in f.read().split()]

    # Aggregate flows from NarcoLogic by End Node, count active number of trafficking nodes (node degree)
    # drop start nodes and individual link flow values--everything based on end node now, summed into tFlow
    nd = Nodes_2.groupby("ID")["IntitFlow"].agg(NodeDegree=lambda x: x[x != 0].count(), NLFlow="sum")
    Nodes_2 = pd.merge(Nodes_2, nd, how='inner', on='ID')
    Nodes_2.drop(['Start_Node'], axis=1, inplace=True)
    Nodes_2.drop_duplicates(subset=['ID'], keep='last', inplace=True)

    # Updated flow values need to be updated on Nodes_2 tble, but XNodes need to be on previous Nodes table.
    # Perform the 25% rule (.75 * expected flow value (PMNode) + .25 * actual flow (NLFlow) =  PMNode2
    # If the node is an interdicted node do: (.75 * expected flow value (PMNode) + .25 * actual flow (NLFlow) =  PMNode2
    # If not interdicted, reduce expected flow (25% rule);
    Nodes_2['PMNode2'] = np.where(Nodes_2['ID'].isin(IntList), ((Nodes_2['PMNode'] * .75) + (Nodes_2['NLFlow'] * .25)),
                                  (Nodes_2['PMNode'] * .75))

    # Update Flow column--calculate the updated flow value from PMNode2 to distribute to the positive ait columns
    # divide the updated flow value 'PMNode_2' by number of nodes w/ Type = 1 ('SumType')
    Nodes_2.loc[Nodes_2.SumType != 0, 'UpdateFlow'] = (Nodes_2['PMNode2'] / Nodes_2['SumType'])
    Nodes_2.loc[Nodes_2.SumType == 0, 'UpdateFlow'] = (Nodes_2['PMNode2'])

    # allocate the Update Flow to positive ait/type values
    TypeNums = [1, 2, 3, 4, 5, 6, 7, 8]
    for num in TypeNums:
        Nodes_2['ait{}'.format(num)] = Nodes_2['Type{}'.format(num)] * Nodes_2['UpdateFlow']

    # Update Timestep
    Nodes_2['Timestep'] = NLtimestep
    ActiveLinks['Timestep'] = NLtimestep

    # Overwrite PMNode with updated perceived interdiction flow vales (PMNode2),
    # Remove Unnecessary Columns, fill NaNs w 0
    Nodes_2['PMNode'] = Nodes_2['PMNode2']
    Nodes_2.drop(['IntitFlow', 'SumType', 'PMNode2', 'UpdateFlow', 'SHAPE'], axis=1, inplace=True)
    Nodes_2.fillna(0, inplace=True)

    # Drop unnecessary columns in Active Links, omit 0 Flow rows, reset index
    ActiveLinks.drop(ActiveLinks.columns[ActiveLinks.columns.str.contains('ait|Type|Country|PMNode|SHAPE')], axis=1,
                     inplace=True)
    ActiveLinks.drop(ActiveLinks.loc[ActiveLinks['IntitFlow'] == 0].index, inplace=True)
    ActiveLinks.dropna(inplace=True)
    ActiveLinks.reset_index(drop=True, inplace=True)

    # write active links to table; concat w/ each timestep
    CompActLinksFP = r'C:\Users\pcbmi\Box\NSF_D-ISN\Code\NarcoLogic\TrialResults\CompActLinks.csv'  # this filename will need a MR suffix
    if os.path.exists(CompActLinksFP):
        CompActLinks = pd.read_csv(CompActLinksFP)  # drop index
        CompActLinks.drop(CompActLinks.columns[0], axis=1, inplace=True)  # omit this line
        # write XNodes to CompActLinks
        ComprehensiveActLinks = pd.concat([CompActLinks, ActiveLinks])
        ComprehensiveActLinks.to_csv(CompActLinksFP)

    else:
        ActiveLinks.to_csv(CompActLinksFP)

    # Concatinate Nodes_2 table with initial Node table if TS 1->2, otherwise Concat CompDF with Nodes_2 and overwrite
    # CompDF csv

    CompFP = r'C:\Users\pcbmi\Box\NSF_D-ISN\Code\NarcoLogic\TrialResults\ComprehensiveNodesDF.csv'
    # this is t2 and up. This finds the existing CompDF table and concats the latest results (nodes2) to it.
    if os.path.exists(CompFP):
        Comp = pd.read_csv(CompFP)  # read this in w/o index
        Comp.drop(Comp.columns[0], axis=1, inplace=True)

        # IMPT! Add interdicted Nodes from txt file to PREVIOUS timestep node info df (*Comp df*)
        XNodes = pd.read_csv(r'C:\Users\pcbmi\Box\NSF_D-ISN\Code\NarcoLogic\MTMCI_IntNodes\MTMCI_IntNodes.txt',
                             header=None)
        XNodes.rename(columns={0: 'ID'}, inplace=True)
        XNodes['Values'] = '1'
        IntDict = dict(zip(XNodes['ID'], XNodes['Values']))
        # add filter for MR as well
        filt = Comp['Timestep'] == i
        Comp.loc[filt, ['XNodes']] = Comp['ID'].map(IntDict)
        Comp.fillna({'XNodes': 0}, inplace=True)

        # Then add Nodes2 (latest data) on to CompDf using concat
        ComprehensiveNodesDF = pd.concat([Comp, Nodes_2])
        ComprehensiveNodesDF.to_csv(CompFP)
    else:
        # this is t 1 to t 2 transition which writes the first CompDF table
        # Add interdicted Nodes from txt file to PREVIOUS timestep Nodes table (*Nodes df*)
        XNodes = pd.read_csv(r'C:\Users\pcbmi\Box\NSF_D-ISN\Code\NarcoLogic\MTMCI_IntNodes\MTMCI_IntNodes.txt',
                             header=None)
        XNodes.rename(columns={0: 'ID'}, inplace=True)
        XNodes['Values'] = '1'
        IntDict = dict(zip(XNodes['ID'], XNodes['Values']))
        filt = Nodes['Timestep'] == i
        Nodes.loc[filt, ['XNodes']] = Nodes['ID'].map(IntDict)
        Nodes.fillna({'XNodes': 0}, inplace=True)

        # Concat Nodes with Nodes2
        ComprehensiveNodesDF1 = pd.concat([Nodes, Nodes_2])
        ComprehensiveNodesDF1.to_csv(CompFP)

    # Make Nodes_2 like Nodes 1; Nodes_2 drop DTO NodeDegree Tflow Update Flow after its written to CompDF
    Nodes_2.drop(['DTO', 'NodeDegree', 'NLFlow', 'MR'], axis=1, inplace=True)

    # Write Nodes_2 to csv if NLtimestep 1 to 179; remove if t 180 so t 1, m:n can resume w Nodes feature
    Nodes_FP = r'C:\Users\pcbmi\Box\NSF_D-ISN\Code\NarcoLogic\TrialResults\Nodes_2.csv'
    # change this value to whatever the end of the NL range is e.g. 180
    if NLtimestep < 180:
        Nodes_2.to_csv(Nodes_FP)
    else:
        os.remove(Nodes_FP)

    #     NodesFP_i= r'C:\Users\pcbmi\Box\NSF_D-ISN\Code\NarcoLogic\TrialResults\Nodes_2Final_t{}.csv'.format
    #     (NLtimestep)
    #     Nodes_2.to_csv(NodesFP_i)

    print('Data Processing for timestep {} complete!'.format(NLtimestep))
    elapsed_time = time.time() - strt
    print('DP Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


def MTMCI_func(NodesDF, timestep, mr):
    """Writes Interdicted Nodes as text file."""
    MTMCLP = gp.Model()
    MTMCLP.setParam('OutputFlag', 0)
    x = {}
    y = {}

    I = NodesDF['ID'].tolist()
    J = NodesDF['ID'].tolist()
    T = range(1, 9)

    for j in J:
        for t in T:
            x[j, t] = MTMCLP.addVar(vtype=GRB.BINARY, name=f'Node{j, t}')
    for i in I:
        for t in T:
            y[i, t] = MTMCLP.addVar(vtype=GRB.BINARY, name=f'Demand{i, t}')

    # Create a[i,t] dictionary
    # make a dataframe with just the ait# and ID field
    Ait = NodesDF.filter(regex='^ait', axis=1)
    ID = NodesDF['ID']
    ID_Ait = Ait.join(ID, how='right')

    # Dataframe to MultiIndex TupleDict e.g. {(i,t):a, (i,t):a...}
    # stripping 'ait' string from list, converting to int
    IDs = ID_Ait.ID.tolist()
    others = list(ID_Ait.columns)
    others.remove("ID")
    others_t = list(map(lambda st: str.replace(st, 'ait', ''), others))
    t_int = list(map(lambda ele: int(ele), others_t))
    index_tuples = [(ID, other) for ID in IDs for other in t_int]
    multi_ix = pd.MultiIndex.from_tuples(index_tuples)
    df1 = pd.DataFrame(ID_Ait[others].values.ravel(), index=multi_ix, columns=["data"])
    a = df1.to_dict()["data"]

    # Create U[j] lists for each node, list the Type#s that have 0 as a value for that ID#
    # e.g. {5:[1,2,4,5,6,7,8]}
    Uj = NodesDF.filter(regex='^Type', axis=1)
    Uj_ID = Uj.join(ID, how='right')
    U = {k: [int(i[-1]) for i, v in d.items() if v == 0]
         for k, d in Uj_ID.set_index('ID').to_dict('index').items()}

    # KMC/formulation version
    for j in J:
        MTMCLP.addConstr(quicksum(x[j, t] for t in U[j]), GRB.EQUAL, 0)

    # #Cov Constraint w/o quicksum for j in J (3) #loc i can be covered >1 if by diff FP type (Const 3c)
    for i in I:
        for t in T:
            MTMCLP.addConstr(x[i, t], GRB.GREATER_EQUAL, y[i, t], f'Covering Constraint{i, t}')

    # Locate Number of Force Packages (4)
    P = {1: 3, 2: 2, 3: 4, 4: 2, 5: 3, 6: 4, 7: 1, 8: 2}

    for t in T:
        MTMCLP.addConstr(quicksum(x[j, t] for j in range(1, 164)), GRB.EQUAL, P[t], f'Type Force Packages{j, t}')

    ##Set the Objective Function (1)
    MTMCLP.setObjective(quicksum(a[i, t] * y[i, t] for i in I for t in T), GRB.MAXIMIZE)

    ##Update
    MTMCLP.update()
    MTMCLP.optimize()

    intSites = []
    for j, t in x:
        if x[j, t].X == 1:
            intSites.append(str(j))
            # print(f'Node {j} | Type {t}')

    obj = MTMCLP.getObjective()
    print("Objective Value: " + str(obj.getValue()))

    # write interdicted nodes to text file
    SolutionFP = r'C:\Users\pcbmi\Box\NSF_D-ISN\Code\NarcoLogic\MTMCI_IntNodes\MTMCI_IntNodes.txt'

    with open(SolutionFP, 'w') as q:
        for site in intSites:
            q.write('%s\n' % site)

    print("MTMCLP for timestep {}, model run {} is complete.".format(timestep, mr))