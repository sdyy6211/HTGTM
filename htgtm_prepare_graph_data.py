from scipy.sparse import bsr_matrix
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os

def get_cooccurance_matrix(traindata,testdata,user_id_remapping):

    data = pd.concat([traindata,testdata],axis=0).reset_index(drop=True)

    task_group = data.loc[:,['user_id','task_id']].groupby('task_id')

    # convert task_group to dict
    task_group_dict = dict(list(task_group['user_id']))

    task_group_dict_list = {k: v.tolist() for k, v in task_group_dict.items()}

    unique_user_number = data.user_id.unique().shape[0]

    trainuserids = traindata.user_id.unique()
    trainuserids.sort()
    testuserids = testdata.user_id.unique()
    testuserids.sort()

    # initialize the user_to_user_matrix with a scipy sparse matrix
    user_to_user_matrix = np.zeros((unique_user_number,unique_user_number))

    for task in task_group_dict_list:
        user_list = task_group_dict_list[task]
        for i in range(len(user_list)):
            for j in range(len(user_list)):
                if i != j:
                    user_to_user_matrix[user_id_remapping[user_list[i]],user_id_remapping[user_list[j]]] += 1
                    user_to_user_matrix[user_id_remapping[user_list[j]],user_id_remapping[user_list[i]]] += 1

    return user_to_user_matrix

def add_feature_to_graph(nodedf,G):
    for index, row in nodedf.iterrows():
        node = index  # assuming the first column is the node name
        features = row[:-1].values  # the rest of the columns are the features
        target = row[-1]
        G.nodes[node]['features'] = features
        G.nodes[node]['target'] = target
        G.nodes[node]['index'] = index
    return G

def build_graph(nodedf,user_matrix):
    G = nx.Graph()
    for i in nodedf.index:
        G.add_node(i)

    for i in nodedf.index:
        for j in nodedf.index:
            if user_matrix[i, j] > 0:
                G.add_edge(i, j)
                # add the weight to the edge
                G[i][j]['weight'] = user_matrix[i, j]

    G = add_feature_to_graph(nodedf,G)
    return G

if __name__ == '__main__':

    data_train = pd.read_csv(r'.\retention_train_feat.csv',index_col=0)
    data_test = pd.read_csv(r'.\retention_test_feat.csv',index_col=0)

    with open(r'.\user_id_remapping.pkl','rb') as f:
        user_id_remapping = pickle.load(f)

    user_matrix = get_cooccurance_matrix(data_train,data_test, user_id_remapping)
   
    G = nx.from_numpy_matrix(user_matrix)

    if not os.path.exists(r'.\temp'):
        os.mkdir(r'.\temp')

    # save the graph
    pickle.dump(G,open(r'.\temp\G.pkl','wb'))

    print('finished processing graph data')