import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import *
from sklearn.metrics import *

comparison_metrics = [mutual_info_score, rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score, v_measure_score]

def compute_scores(pca, L):

    nc = len(np.unique(L))
    
    clus1 = AgglomerativeClustering(n_clusters=nc).fit(pca)
    labs1 = clus1.labels_

    clus2 = KMeans(n_clusters=nc).fit(pca)
    labs2 = clus2.labels_
    
    try:
        ui = np.triu_indices(len(pca), k=1)
        D = pairwise_distances(pca)[ui].flatten()
        eps = np.quantile(D,.05)
        gamma = np.quantile(D,.8)
    except:
        print('used defaults')
        eps = .1
        gamma = 1.

    clus3 = DBSCAN(eps=eps).fit(pca)
    labs3 = clus3.labels_

    try:
        clus4 = SpectralClustering(n_clusters=nc, gamma=gamma, affinity='rbf').fit(pca)
        labs4 = clus4.labels_
    except:
        print('used dummy labels for spectral embedding')
        labs4 = np.zeros(labs1.shape)

    labs = [labs1, labs2, labs3, labs4]

    return np.array([[met(L, l) for met in comparison_metrics] for l in labs])

def compute_mixings(pca, limit, list_kth):

    mixes = []
    for kth in list_kth:
        nbrs = NearestNeighbors(n_neighbors=kth, algorithm='ball_tree').fit(pca)
        _, indices = nbrs.kneighbors(pca)
        mix = []
        for c in range(len(indices)):
            c_indices = indices[c,1:]
            #print(indices[c,:])
            p1 = len(np.argwhere(c_indices<limit))/len(c_indices)
            p2 = len(np.argwhere(c_indices>=limit))/len(c_indices)
            if p1 == 0 or p2 == 0:
                mix.append(0)
            else:
                mix.append((-p1*np.log(p1)-p2*np.log(p2))/np.log(2))
        mixes.append(mix)

    return mixes
