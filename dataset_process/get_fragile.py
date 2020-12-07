import numpy as np
import numba
import scipy.sparse as sp
import scipy.linalg as spl

def isSubstring(s1, s2):
    M = len(s1)
    N = len(s2)

    # A loop to slide pat[] one by one
    for i in range(N - M + 1):

        # For current index i,
        # check for pattern match
        for j in range(M):
            if (s2[i + j] != s1[j]):
                break

        if j + 1 == M:
            return i

    return -1




def get_new_fragile(root,dataset,adj,global_budget):
    dataset1 ='cora'
    res = isSubstring(dataset1,dataset)
    if res!=-1:
        print("The current dataset is cora dataset")
        path = root
        prefix = dataset
        fragile_extra = np.load(path+prefix+'-fragile_set.npy')
        n = adj.shape[0]

        mst = sp.csgraph.minimum_spanning_tree(adj)
        mst = mst + mst.T
        fragile = np.column_stack((adj - mst).nonzero())
        fragile_set = set()
        for edge in fragile:
            fragile_set.add((edge[0], edge[1]))
        for edge in fragile_extra:
            fragile_set.add((edge[0], edge[1]))
        fragile =np.array([0,0])
        for edge in fragile_set:
            fragile= np.vstack((fragile,[edge[0],edge[1]]))
        fragile = np.delete(fragile, 0, axis=0)
        #fragile = np.concatenate((fragile,fragile_extra),axis=0)
        
        if fragile.shape[0]<global_budget:
            ###add some dummy edges to fulfill the need
            dummy_edges=np.array([0,0])
            num_edge_diff = global_budget - fragile.shape[0]
            while (len(dummy_edges)<=1.5*num_edge_diff):
                source = np.random.random_integers(low=0,high=adj.shape[0]-1)
                dest = np.random.random_integers(low=0, high=adj.shape[0]-1)
                dummy_edges =np.vstack((dummy_edges,np.array([source,dest])))
            dummy_edges = np.delete(dummy_edges,0,axis=0)
            fragile = np.concatenate((fragile,dummy_edges),axis=0)

        #get the transition matrix and prune to get at most B edges
        if fragile.shape[0]>global_budget:
            trans_mx = np.load(path+prefix+'-trans_mx.npy')
            cand_pairs =[]
            for i in range(fragile.shape[0]):
                edge = fragile[i]
                source = fragile[i][0]
                dest = fragile[i][1]
                tran_prob = trans_mx[source][dest]
                cand_pairs.append((i,tran_prob))
            #according to the tran_prob to rank the edge index
            cand_pairs = sorted(cand_pairs, key=lambda x: x[1], reverse=True)
            #only save the top-B edges
            select_idx=[]
            for j in range(global_budget):
                select_idx.append(cand_pairs[j][0])
            fragile = fragile[select_idx]
        return fragile
    
    dataset2 ='citeseer' 
    res = isSubstring(dataset2,dataset)
    if res!=-1:
        print("The current dataset is citeseer dataset")
        path = root
        prefix = dataset
        fragile_extra = np.load(path+prefix+'-fragile_set.npy')
        n = adj.shape[0]

        mst = sp.csgraph.minimum_spanning_tree(adj)
        mst = mst + mst.T
        fragile = np.column_stack((adj - mst).nonzero())
        fragile_set = set()
        for edge in fragile:
            fragile_set.add((edge[0], edge[1]))
        for edge in fragile_extra:
            fragile_set.add((edge[0], edge[1]))
        fragile =np.array([0,0])
        for edge in fragile_set:
            fragile= np.vstack((fragile,[edge[0],edge[1]]))
        fragile = np.delete(fragile, 0, axis=0)
        #fragile = np.concatenate((fragile,fragile_extra),axis=0)
        
        if fragile.shape[0]<global_budget:
            ###add some dummy edges to fulfill the need
            dummy_edges=np.array([0,0])
            num_edge_diff = global_budget - fragile.shape[0]
            while (len(dummy_edges)<=1.5*num_edge_diff):
                source = np.random.random_integers(low=0,high=adj.shape[0]-1)
                dest = np.random.random_integers(low=0, high=adj.shape[0]-1)
                dummy_edges =np.vstack((dummy_edges,np.array([source,dest])))
            dummy_edges = np.delete(dummy_edges,0,axis=0)
            fragile = np.concatenate((fragile,dummy_edges),axis=0)

        #get the transition matrix and prune to get at most B edges
        if fragile.shape[0]>global_budget:
            trans_mx = np.load(path+prefix+'-trans_mx.npy')
            cand_pairs =[]
            for i in range(fragile.shape[0]):
                edge = fragile[i]
                source = fragile[i][0]
                dest = fragile[i][1]
                tran_prob = trans_mx[source][dest]
                cand_pairs.append((i,tran_prob))
            #according to the tran_prob to rank the edge index
            cand_pairs = sorted(cand_pairs, key=lambda x: x[1], reverse=True)
            #only save the top-B edges
            select_idx=[]
            for j in range(global_budget):
                select_idx.append(cand_pairs[j][0])
            fragile = fragile[select_idx]
        return fragile
        

def get_fast_heu_fragile(root,dataset):
    dataset1 ='cora'
    res = isSubstring(dataset1,dataset)
    if res!=-1:
        print("The current dataset is cora dataset")
        path = root
        prefix = dataset
        fragile = np.load(path+prefix+'-fragile_set.npy')
        return fragile
        

        

