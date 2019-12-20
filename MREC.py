import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.base import BaseEstimator, TransformerMixin

from ot.gromov import entropic_gromov_wasserstein
from ot.bregman import sinkhorn
from ot import unif

from scipy.optimize import minimize

from cvxopt.solvers import sdp
from cvxopt import matrix as cvxmat

try:    import matlab.engine
except ModuleNotFoundError: print("Warning: Matlab module not found, GWC_cplex and GWC_sdpnal not available")

####################
# Matching methods #
####################

def gwS(D1, D2, epsilon=5e-4, tol=1e-9, max_iter=1000):
	D1 /= D1.max()
	D2 /= D2.max()
	n1, n2 = D1.shape[0], D2.shape[0]
	d1, d2 = unif(n1), unif(n2)
	gw = entropic_gromov_wasserstein(D1, D2, d1, d2, "square_loss", epsilon=epsilon, max_iter=max_iter, tol=tol)
	mappings = [np.zeros(n1, dtype=np.int32), np.zeros(n2, dtype=np.int32)]
	for i in range(n1):	mappings[0][i] = np.argmax(gw[i,:])
	for i in range(n2):	mappings[1][i] = np.argmax(gw[:,i])
	return gw, mappings

def wS(X1, X2, metric="euclidean", epsilon=5e-4, tol=1e-09, max_iter=1000, pairwise=np.empty([0,0])):
	assert X1.shape[1] == X2.shape[1]
	n1, n2 = X1.shape[0], X2.shape[0]
	gamma = sinkhorn( a=(1/n1) * np.ones(n1), b=(1/n2) * np.ones(n2), M=pairwise, reg=epsilon, numItermax=max_iter, stopThr=tol) if metric=="precomputed" else sinkhorn( a=(1/n1) * np.ones(n1), b=(1/n2) * np.ones(n2), M=pairwise_distances(X1, X2, metric=metric), reg=epsilon )
	mappings = [np.zeros(n1, dtype=np.int32), np.zeros(n2, dtype=np.int32)]
	for i in range(n1):	mappings[0][i] = np.argmax(gamma[i,:])
	for i in range(n2):	mappings[1][i] = np.argmax(gamma[:,i])
	return gamma, mappings

def mixtureS(D1, D2, Z, alpha=.5, epsilon=5e-4, tol=1e-9, max_iter=1000):

	assert Z.shape[0] == D1.shape[0]
	assert Z.shape[1] == D2.shape[0]

	D1 /= D1.max()
	D1 -= np.mean(D1)
	D2 /= D2.max()
	D2 -= np.mean(D2)
	Z /= Z.max()
	
	n1, n2 = D1.shape[0], D2.shape[0]
	p, q = (1/n1) * np.ones(n1), (1/n2) * np.ones(n2)
	gamma = np.outer(p, q)
	cpt, err = 0, 1.

	if alpha == 1:	gamma = sinkhorn(p, q, Z, epsilon)
	else:        
		while (err > tol and cpt < max_iter):
            
			gamma_prev = gamma

			tens = -np.dot(D1, gamma).dot(D2.T)
			tens -= tens.min()
			tens_all = (1-alpha) * tens + alpha * Z
			gamma = sinkhorn(p, q, tens_all, epsilon)
        
			if cpt % 10 == 0:	err = np.linalg.norm(gamma - gamma_prev)
			# We can speed up the process by checking for the error only all the 10th iterations
				
	mappings = [np.zeros(n1, dtype=np.int32), np.zeros(n2, dtype=np.int32)]
	for i in range(n1):	mappings[0][i] = np.argmax(gamma[i,:])
	for i in range(n2):	mappings[1][i] = np.argmax(gamma[:,i])

	return gamma, mappings

def lagrangian(x, Gamma, A, b, sigma_m, lambda_m):
	U = np.matmul(A,x)-b
	return np.matmul(x.T,np.matmul(Gamma,x)) - np.matmul(lambda_m.T,U) + .5*sigma_m*np.matmul(U.T,U)

def lagrangian_grad(x, Gamma, A, b, sigma_m, lambda_m):
	U = np.matmul(A,x)-b
	grad = 2*np.matmul(Gamma,x) - np.matmul(A.T,lambda_m) + sigma_m*np.matmul(A.T,U)
	return grad

def lagrangian_hess(x, Gamma, A, b, sigma_m, lambda_m):
	H = 2*Gamma + sigma_m*np.matmul(A.T,A)
	return H

def gwNC(D1, D2, num_iter=15, sigma_m_0=5., mu=10., method="L-BFGS-B", map_init=None, verbose=False):

	n = D1.shape[0]
	assert D2.shape[0] == n

	D1 /= D1.max()
	D2 /= D2.max()

	# Compute distortions
	Gamma = np.abs(  np.repeat(np.repeat(D1,n,0),n,1) - np.tile(np.tile(D2,n).T,n).T  )

	# Initialize solution
	y = np.ones([n*n])/n if map_init is None else map_init

	# Initialize constraints
	lambda_m = np.ones([2*n])
	sigma_m = sigma_m_0
	A = np.concatenate([np.repeat(np.eye(n),n,1), np.tile(np.eye(n),n)], axis=0)
	b = np.ones([2*n])
	
	# Initialize feasability, objective and solution
	feasabilities = np.zeros([num_iter])
	feasabilities[0] = np.linalg.norm( np.matmul(A,y)-b )
	objectives = np.zeros([num_iter])
	objectives[0] = float(np.matmul(y.T, np.matmul(Gamma,y)))
	minimizers = np.zeros([n*n,num_iter])
	minimizers[:,0] = np.squeeze(y)

	for it in range(1, num_iter):

		if verbose:	print("iteration " + str(it))

		# Update
		y = minimize(fun=lagrangian, x0=y, args=(Gamma, A, b, sigma_m, lambda_m), jac=lagrangian_grad, hess=lagrangian_hess, bounds=[(0.,1.) for _ in range(len(y))], method=method)["x"]

		# Store values
		lambda_m = lambda_m - sigma_m * (np.matmul(A,y)-b)
		sigma_m = mu * sigma_m
		feasabilities[it] = np.linalg.norm( np.matmul(A,y)-b )
		objectives[it] = float(np.matmul(y.T, np.matmul(Gamma,y)))
		minimizers[:,it] = np.squeeze(y)

	mappings = [np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)]
	for i in range(n):	mappings[0][i] = np.argmax(minimizers[i*n:(i+1)*n, num_iter-1])
	for i in range(n):	mappings[1][i] = np.argmax(minimizers[i::n, num_iter-1])
	gam = np.reshape(minimizers[:,num_iter-1], [n,n])	

	return gam, mappings

def gwC_sdpnal(D1, D2, eng, use_birkhoff=False):

	n1, n2 = len(D1), len(D2)
	assert n1 == n2

	D1 /= D1.max()
	D2 /= D2.max()

	eng.workspace["D1"] = D1.tolist()
	eng.workspace["D2"] = D2.tolist()
	eng.run("gwCsdpnal.m", nargout=0)
	if use_birkhoff:
		M = np.array(eng.workspace["maps"]).flatten()
	else:
		Y = np.array(eng.workspace['X'])
		M = Y[:-1,-1]

	mappings = [np.zeros(n1, dtype=np.int32), np.zeros(n1, dtype=np.int32)]
	for i in range(n1):	mappings[0][i] = np.argmax(M[i*n1:(i+1)*n1])
	for i in range(n1):	mappings[1][i] = np.argmax(M[i::n1])
	gam = np.reshape(M, [n1, n1])
	return gam, mappings

def gwC_cplex(D1, D2, eng, maxtime=120):

	n1, n2 = len(D1), len(D2)
	assert n1 == n2

	D1 /= D1.max()
	D2 /= D2.max()

	eng.workspace["D1"] = D1.tolist()
	eng.workspace["D2"] = D2.tolist()
	eng.workspace["maxtime"] = maxtime
	eng.run("gwCcplex.m", nargout=0)
	Y = np.array(eng.workspace['X'])
	M = Y[:,-1]
	mappings = [np.zeros(n1, dtype=np.int32), np.zeros(n1, dtype=np.int32)]
	for i in range(n1):	mappings[0][i] = np.argmax(M[i*n1:(i+1)*n1])
	for i in range(n1):	mappings[1][i] = np.argmax(M[i::n1])
	gam = np.reshape(M, [n1, n1])
	return gam, mappings






########
# MREC #
########

class Quantization(BaseEstimator, TransformerMixin):
	"""
	This is a class that implements various quantization/clustering methods.
	"""
	def __init__(self, n_clusters=10, method="RandomChoice", metric="euclidean"):
		"""
		Constructor for the Quantization class.

		Parameters:
			n_clusters (int): number of clusters to use for quantization.
			method (str): "RandomChoice" or "KMeans". If None, no quantization is used. You can add your own method by giving it a name and implementing it in the fit method below.
			metric (str): distance to use to find clusters.
		"""
		self.n_clus_, self.metric_, self.method_ = n_clusters, metric, method

	def fit(self, X, D=None):
		"""
		Fit the Quantization class on a point cloud or distance matrix: compute the clusters and their centroids and store them in self.labels_, as well as a map assigning one of the centroids to each data point in self.indices_.
	
		Parameters:
			X (array of shape [num points, num dimensions]): point cloud
			D (array of shape [num points, num points]): distance matrix to use if self.metric_ == "precomputed".
		"""
		if self.method_ == "RandomChoice":
			self.indices_ = np.random.choice(X.shape[0], self.n_clus_, replace=False)
			if self.metric_ == "precomputed":
				assert D is not None
				self.labels_ = np.argmin(D[:, self.indices_], axis=1)
			else:	self.labels_ = np.argmin(pairwise_distances(X, X[self.indices_,:], metric=self.metric_), axis=1)

		if self.method_ is None:
			self.indices_ = np.arange(X.shape[0])
			self.labels_  = np.arange(X.shape[0])

		if self.method_ == "KMeans":
			clus = KMeans(n_clusters=self.n_clus_).fit(X)
			self.labels_ = clus.labels_
			cluster_centers = np.concatenate([np.mean(X[np.argwhere(clus.labels_ == i)[:,0],:], axis=0)[np.newaxis,:] for i in range(self.n_clus_)], axis=0)
			nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X)
			self.indices_ = np.squeeze(nbrs.kneighbors(cluster_centers)[1])

		return self


def distortion_score(X1, X2, gamma, metric="euclidean", mode=2):
	"""
	Compute the metric distortion associated to a matching between two datasets.

	Parameters:
		X1 (array of shape [num points 1, num dimensions]): first point cloud
		X2 (array of shape [num points 2, num dimensions]): second point cloud
		gamma (array of shape [num points 1, num points 2]): probabilistic matching between X1 and X2
		metric (str): metric to use in order to compute the distortion
		mode (int): 1, 2 or 3. Method to use to compute the distortion. 1 is slow but does not require a lot of RAM, 3 is fast but it requires a lot of RAM. 2 is intermediate
	"""
	if metric == "precomputed":	D1, D2 = X1, X2
	else:	D1, D2 = pairwise_distances(X1, metric=metric), pairwise_distances(X2, metric=metric)

	# Slowest, but does not need a lot of RAM
	if mode == 1:
		score = 0
		for i in range(len(X1)):
			for k in range(len(X2)):
				for j in range(len(X1)):
					for l in range(len(X2)):
						score += gamma[i,k] * gamma[j,l] * np.abs(D1[i,j] - D2[k,l])

	# Intermediate
	elif mode == 2:
		score = 0
		for i in range(len(X1)):
			for k in range(len(X2)):
				D = np.abs(np.expand_dims(D1[i,:], -1) - np.expand_dims(D2[k,:], 0))
				score += gamma[i,k] * np.sum(np.multiply(gamma,D))

	# Fastest, but needs a lot of RAM
	elif mode == 3:
		G = np.abs( np.expand_dims(np.expand_dims(D1,-1),-1) - np.expand_dims(np.expand_dims(D2,0),0) )
		score = np.sum(np.multiply(gamma, np.tensordot(G, gamma, axes=([1,3],[0,1]))))

	return score

def MREC(X1, X2, threshold=10, n_points=100, q_method="KMeans", metric="euclidean", backend_params={"mode": "GWNC", "num_iter": 3, "method": "L-BFGS-B"}, 
                sparse=False, return_gamma=False, idxs1=None, idxs2=None, mapping=None, D1_full=None, D2_full=None, Z_full=None, Gamma=None, mass=None):

	"""
	This function implements MREC. See the associated paper "MREC: a fast and versatile tool for aligning and matching genomic data". It uses a recursive approximation to compute matchings in a fast way.

	Parameters:
		X1 (array of shape [num points 1, num dimensions]): first point cloud
		X2 (array of shape [num points 2, num dimensions]): second point cloud
		threshold (int): datasets with less points than this parameter will be matched arbitrarily
		n_points (int): number of clusters and centroids used for the recursion
		q_method (str): name of the quantization method. It should be implemented in the Quantization class above
		metric (str): distance to use for computing and comparing the centroids. It is either a single string or a pair of strings if the two datasets have different metrics
		backend_params (dictionary): parameters used for the black box matching function. It has to contain a key called "mode" whose value is a string specifiying the method to use and the other required keys:
			"WS" (entropy Wasserstein):  "metric" (string, metric to use between datasets), "epsilon" (float, entropy regularization term), "tol" (float, tolerance threshold for stopping Sinkhorn iterations), "max_iter" (int, maximum number of Sinkhorn iterations)
			"GWS" (entropy Gromov-Wasserstein): "metric" ((pair of) string, metrics to use for each dataset), "epsilon" (float, entropy regularization term), "tol" (float, tolerance threshold for stopping Sinkhorn iterations), "max_iter" (int, maximum number of Sinkhorn iterations) 
			"MXS" (entropy weighted sum of Wasserstein and Gromov-Wasserstein): "metric" (pair of (pair of) string, metrics to use for each dataset, for Wasserstein and Gromov-Wasserstein costs), "alpha" (float (between 0. and 1.), weight of Wasserstein cost), "epsilon" (float, entropy regularization term), "tol" (float, tolerance threshold for stopping Sinkhorn iterations), "max_iter" (int, maximum number of Sinkhorn iterations)
			"GWNC" (non convex approximation of Gromov-Wasserstein): "num_iter" (int, number of non convex iterations), "sigma_m_0" (float, see https://arxiv.org/pdf/1610.05214.pdf), "mu" (float, see https://arxiv.org/pdf/1610.05214.pdf), "method" (string, optimization method, one of the methods proposed in scipy.optimize.minimize)
			"GWC_sdpnal" (convex approximation of Gromov-Wasserstein with sdpnal): "eng" (Matlab engine), "use_birkhoff" (bool, whether to use birkhoff projections)
			"GWC_cplex" (convex approximation of Gromov-Wasserstein with cplex)): "eng" (Matlab engine), "maxtime" (int, maximum number of seconds allowed)
		You can use your own method by giving it a name and implementing it in this file.
		D1_full (array of shape [num points 1, num points 1]): precomputed distance matrix of first dataset, used only if backend_params["metric"] == "precomputed" and backend_params["mode"] is not "WS"
		D2_full (array of shape [num points 2, num points 2]): precomputed distance matrix of second dataset, used only if backend_params["metric"] == "precomputed" and backend_params["mode"] is not "WS"
		Z_full (array of shape [num points 1, num points 2]): precomputed distance matrix between first and second datasets, used only if backend_params["metric"] == "precomputed" and backend_params["mode"] == "WS"
		sparse (bool): whether sparse matrices are used
		return_gamma (bool): whether to return the whole array representing the probabilistic matching, or just the mapping obtained by rounding it
		the other parameters are only used internally for the recursion

	Returns:
		Gamma (array of shape [num points 1, num points 2]): probabilistic matching between X1 and X2 (returned only if return_gamma is True) 
		matching (array of shape [num points 1]): matching for the points of X1 obtained by rounding Gamma
	"""
	n1, n2 = X1.shape[0], X2.shape[0]
	if idxs1 is None:	idxs1 = np.arange(n1)
	if idxs2 is None:	idxs2 = np.arange(n2)
	if mapping is None:	mapping = np.zeros(n1, dtype=np.int32)
	if return_gamma and Gamma is None:	Gamma = np.empty([n1,n2])
	if mass is None:	mass = 1 / n2

	if n1 <= threshold or n2 <= threshold:
		for i in range(len(idxs1)):	mapping[idxs1[i]] = idxs2[0]

	else:
		n_clus = min(min(n1, n2), n_points)
		if type(metric) == str:	quantize1, quantize2 = Quantization(n_clusters=n_clus, method=q_method, metric=metric), Quantization(n_clusters=n_clus, method=q_method, metric=metric)	
		else:	quantize1, quantize2 = Quantization(n_clusters=n_clus, method=q_method, metric=metric[0]), Quantization(n_clusters=n_clus, method=q_method, metric=metric[1])

		quantize1.fit(X1, D1_full)
		quantize2.fit(X2, D2_full)
		indices1, indices2 = quantize1.indices_, quantize2.indices_
		subX1, subX2 = X1[indices1, :], X2[indices2, :]
		if backend_params["mode"] is not "WS":
			metric = backend_params["metric"] if backend_params["mode"] is not "MXS" else backend_params["metric"][0]  
			if type(metric) == str:
				if metric == "precomputed":	D1, D2 = D1_full[:, indices1][indices1, :], D2_full[:, indices2][indices2, :]
				else:	D1, D2 = pairwise_distances(subX1, metric=metric), pairwise_distances(subX2, metric=metric)
			else:
				if metric[0] == "precomputed":	D1 = D1_full[:, indices1][indices1, :]
				else:	D1 = pairwise_distances(subX1, metric=metric[0])

				if metric[1] == "precomputed":	D2 = D2_full[:, indices2][indices2, :]
				else:	D2 = pairwise_distances(subX2, metric=metric[1])

		if backend_params["mode"] == "MXS":	Z = Z_full[indices1,:][:,indices2] if backend_params["metric"][1] == "precomputed" else pairwise_distances(subX1, subX2, metric=backend_params["metric"][1])
		if backend_params["mode"] == "WS":	Z = Z_full[indices1,:][:,indices2] if backend_params["metric"] == "precomputed" else pairwise_distances(subX1, subX2, metric=backend_params["metric"])

		backend_params_wo_mode = {k: v for (k,v) in iter(backend_params.items()) if k != "mode"}
		backend_params_wo_mode_wo_metric = {k: v for (k,v) in iter(backend_params_wo_mode.items()) if k != "metric"}

		if   backend_params["mode"] == "GWS":	gam, mapps = gwS(D1, D2, **backend_params_wo_mode_wo_metric)
		elif backend_params["mode"] == "WS":	gam, mapps = wS(subX1, subX2, pairwise=Z, **backend_params_wo_mode)
		elif backend_params["mode"] == "GWNC":	gam, mapps = gwNC(D1, D2, **backend_params_wo_mode_wo_metric)
		elif backend_params["mode"] == "GWC_sdpnal":	gam, mapps = gwC_sdpnal(D1, D2, **backend_params_wo_mode_wo_metric)
		elif backend_params["mode"] == "GWC_cplex":	gam, mapps = gwC_cplex(D1, D2, **backend_params_wo_mode_wo_metric)
		elif backend_params["mode"] == "MXS":	gam, mapps = mixtureS(D1, D2, Z, **backend_params_wo_mode_wo_metric)
		else:
			print("Warning: mode not specified")
			return 0

		m12 = mapps[0]
		nl1, nl2 = np.max(quantize1.labels_) + 1, np.max(quantize2.labels_) + 1
		pops1, pops2 = [np.argwhere(quantize1.labels_ == i)[:,0] for i in range(nl1)], [np.argwhere(quantize2.labels_ == i)[:,0] for i in range(nl2)]

		for i in range(nl1):

			I1 = pops1[i]
			if return_gamma:	tmp = Gamma[idxs1[I1], :]
			for j in range(nl2):
				I2 = pops2[j]
				if return_gamma:	tmp[:, idxs2[I2]] = mass * gam[i,j] / (len(I2) * np.sum(gam[i,:]))
			if return_gamma:	Gamma[idxs1[I1], :] = tmp

			I2                = pops2[m12[i]]
			popX1i, popX2i    = X1[I1, :], X2[I2, :]
			popI1i, popI2i    = idxs1[I1], idxs2[I2]
			Z_fulli           = Z_full[I1,:][:,I2]  if Z_full  is not None else Z_full
			D1_fulli          = D1_full[I1,:][:,I1] if D1_full is not None else D1_full
			D2_fulli          = D2_full[I2,:][:,I2] if D2_full is not None else D2_full
			massi             = mass * gam[i,m12[i]] / np.sum(gam[i,:])

			MREC(X1=popX1i, X2=popX2i, threshold=threshold, n_points=n_points, q_method=q_method, metric=metric, backend_params=backend_params, 
                                    idxs1=popI1i, idxs2=popI2i, mapping=mapping, D1_full=D1_fulli, D2_full=D2_fulli, Z_full=Z_fulli, Gamma=Gamma, mass=massi)

	return Gamma, mapping
