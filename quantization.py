import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

def RandomChoiceQuantization(X=None, D=None, prms={}):
	n_centroids = prms['n_clusters']
	npts = len(X) if X is not None else len(D)
	indices = np.random.choice(npts, min(npts, n_centroids), replace=False)
	if D is not None:
		labels = np.argmin(D[:,indices], axis=1)
	else:
		nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X[indices,:])
		_, labels = nbrs.kneighbors(X)
	return indices, labels.ravel()

def HierarchicalQuantization(X=None, D=None, prms={}):
	n_centroids = prms['n_clusters']
	npts = len(X) if X is not None else len(D)
	if D is not None:
		clus = AgglomerativeClustering(n_clusters=n_centroids, affinity='precomputed', linkage='single')
		clus.fit(D)
	else:
		clus = AgglomerativeClustering(n_clusters=n_centroids, linkage='single')
		clus.fit(X)
	labels = clus.labels_.ravel()
	if X is None:
		indices = np.array([np.argwhere(labels==c).ravel()[0] for c in range(n_centroids)])
	else:
		centroids = np.vstack([  np.mean(X[np.argwhere(labels==c).ravel(),:],axis=0)[None,:] for c in range(n_centroids)  ])
		nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X)
		_, indices = nbrs.kneighbors(centroids)
	return indices.ravel(), labels

def KMeansQuantization(X=None, D=None, prms={}):
	pprms = {k:v for k,v in prms.items()}
	nclus = pprms.pop('n_clusters')
	new_nclus = min(len(X), nclus)
	clus = KMeans(n_clusters=new_nclus, **pprms).fit(X)
	cluster_centers = np.vstack([np.mean(X[np.argwhere(clus.labels_ == i).ravel(),:], axis=0)[None,:] for i in range(new_nclus)])
	nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X)
	_, indices = nbrs.kneighbors(cluster_centers)
	return indices.ravel(), clus.labels_

def QuantizationSizeMinParameters(n_clusters):
	def Q(X=None, D=None):
		if D is None:
			return {'n_clusters': min(len(X), n_clusters)}
		else:
			return {'n_clusters': min(len(D), n_clusters)}
	return Q

