import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy
from ot import unif
numpy.set_printoptions(threshold=sys.maxsize)

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.base import BaseEstimator, TransformerMixin

from ot.gromov import entropic_gromov_wasserstein
from ot.bregman import sinkhorn
from ot.unbalanced import sinkhorn_unbalanced

from scipy.optimize import minimize

from cvxopt.solvers import sdp
from cvxopt import matrix as cvxmat

try:    import matlab.engine
except ModuleNotFoundError: print("Warning: Matlab module not found, GWC_cplex and GWC_sdpnal not available")

####################
# Matching methods #
####################

# Gromov-Wasserstein
def GWS(D1, D2, epsilon=5e-4, tol=1e-9, max_iter=1000):
	D1 /= D1.max()
	D2 /= D2.max()
	n1, n2 = D1.shape[0], D2.shape[0]
	d1, d2 = unif(n1), unif(n2)
	gamma = entropic_gromov_wasserstein(D1, D2, d1, d2, "square_loss", epsilon=epsilon, max_iter=max_iter, tol=tol)
	mappings = [np.zeros(n1, dtype=np.int32), np.zeros(n2, dtype=np.int32)]
	for i in range(n1):	mappings[0][i] = np.argmax(gamma[i,:])
	for i in range(n2):	mappings[1][i] = np.argmax(gamma[:,i])
	return gamma, mappings

# Wasserstein
def WS(X1, X2, metric="euclidean", epsilon=5e-4, tol=1e-09, max_iter=1000):
	n1, n2 = X1.shape[0], X2.shape[0]
	gamma = sinkhorn( a=(1/n1) * np.ones(n1), b=(1/n2) * np.ones(n2), M=pairwise_distances(X1, X2, metric=metric), reg=epsilon )
	mappings = [np.zeros(n1, dtype=np.int32), np.zeros(n2, dtype=np.int32)]
	for i in range(n1):	mappings[0][i] = np.argmax(gamma[i,:])
	for i in range(n2):	mappings[1][i] = np.argmax(gamma[:,i])
	return gamma, mappings

# Unbalanced Wasserstein
def UWS(X1, X2, metric="euclidean", epsilon=5e-4, delta=5e-4, tol=1e-09, max_iter=1000):
	n1, n2 = X1.shape[0], X2.shape[0]
	gamma = sinkhorn_unbalanced( a=(1/n1) * np.ones(n1), b=(1/n2) * np.ones(n2), M=pairwise_distances(X1, X2, metric=metric), reg=epsilon, reg_m=delta )
	mappings = [np.zeros(n1, dtype=np.int32), np.zeros(n2, dtype=np.int32)]
	for i in range(n1):	mappings[0][i] = np.argmax(gamma[i,:])
	for i in range(n2):	mappings[1][i] = np.argmax(gamma[:,i])
	return gamma, mappings

# Mixture of Wasserstein and Gromov-Wasserstein
def MXS(X1, X2, D1, D2, metric="euclidean", alpha=.5, epsilon=5e-4, tol=1e-9, max_iter=1000):

	Z = pairwise_distances(X1, X2, metric=metric)
	n1, n2 = D1.shape[0], D2.shape[0]
	p, q = (1/n1) * np.ones(n1), (1/n2) * np.ones(n2)
	gamma = np.outer(p, q)
	cpt, err = 0, 1.

	if alpha == 1:	gamma = sinkhorn(p, q, Z, epsilon)
	else:        
		while (err > tol and cpt < max_iter):
            
			gamma_prev = gamma

			tens = np.dot(D1, gamma).dot(D2.T)
			tens_all = (1-alpha) * tens + alpha * Z
			gamma = sinkhorn(p, q, tens_all, epsilon)
        
			if cpt % 10 == 0:	err = np.linalg.norm(gamma - gamma_prev)
				
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

# Non-convex approximation of Gromov-Wasserstein
def GWNC(D1, D2, num_iter=15, sigma_m_0=5., mu=10., method="L-BFGS-B", map_init=None, verbose=False):

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
	gamma = np.reshape(minimizers[:,num_iter-1], [n,n])	

	return gamma, mappings

# Convex approximation of Gromov-Wasserstein with SDPNAL solver
def GWC_sdpnal(D1, D2, eng, use_birkhoff=False):

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
	gamma = np.reshape(M, [n1, n1])
	return gamma, mappings

# Convex approximation of Gromov-Wasserstein with CPLEX solver
def GWC_cplex(D1, D2, eng, maxtime=120):

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
	gamma = np.reshape(M, [n1, n1])
	return gamma, mappings






########
# MREC #
########

class Quantization(BaseEstimator, TransformerMixin):
	"""
	This is a class that implements various quantization/clustering methods.
	"""
	def __init__(self, n_centroids=10, method="RandomChoice", metric="euclidean"):
		"""
		Constructor for the Quantization class.

		Parameters:
			n_clusters (int): number of clusters to use for quantization.
			method (str): "RandomChoice" or "KMeans". If None, no quantization is used. You can add your own method by giving it a name and implementing it in the fit method below.
			metric (str): distance to use to find clusters.
		"""
		self.n_clus_, self.metric_, self.method_ = n_centroids, metric, method

	def fit(self, X=None, D=None):
		"""
		Fit the Quantization class on a point cloud or distance matrix: compute the clusters and their centroids and store them in self.labels_, as well as a map assigning one of the centroids to each data point in self.indices_.
	
		Parameters:
			X (array of shape [num points, num dimensions]): point cloud
			D (array of shape [num points, num points]): distance matrix to use if self.metric_ == "precomputed".
		"""
		
		npts = X.shape[0] if self.metric_ is not "precomputed" else D.shape[0]
		if self.method_ == "RandomChoice":
			self.indices_ = np.random.choice(npts, min(npts, self.n_clus_), replace=False)
			if self.metric_ == "precomputed":
				assert D is not None
				self.labels_ = np.argmin(D[:, self.indices_], axis=1)
			else:
				self.labels_ = np.argmin(pairwise_distances(X, X[self.indices_,:], metric=self.metric_), axis=1)
				self.labels_[self.indices_] = np.arange(len(self.indices_))

		if self.method_ is None:
			self.indices_ = np.arange(npts)
			self.labels_  = np.arange(npts)

		if self.method_ == "KMeans":
			clus = KMeans(n_clusters=self.n_clus_).fit(X)
			self.labels_ = clus.labels_
			cluster_centers = np.concatenate([np.mean(X[np.argwhere(clus.labels_ == i)[:,0],:], axis=0)[np.newaxis,:] for i in range(self.n_clus_)], axis=0)
			nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X)
			self.indices_ = np.squeeze(nbrs.kneighbors(cluster_centers)[1])

		return self


def distortion_score(X1=None, X2=None, X12=None, D1=None, D2=None, 
                     gamma=None, computation="1", metric="euclidean"):
	"""
	Compute the metric distortion associated to a matching between two datasets. The function infers the distortion algorithm based on which input is None and which isn't. There are five possibilities: 1. X1/X2 not None, X12 None, D1/D2 None; 2. X1/X2 None, X12 not None, D1/D2 None; 3. X1/X2 not None, X12 None, D1/D2 not None; 4. X1/X2 None, X12 not None, D1/D2 not None; 5. X1/X2 None, X12 None, D1/D2 not None; 

	Parameters:
		X1 (array of shape [num points 1, num dimensions]): first point cloud
		X2 (array of shape [num points 2, num dimensions]): second point cloud
                X12 (array of shape [num points 1, num points 2]): pairwise distances between first and second point cloud
		D1 (array of shape [num points 1, num points 1]): pairwise distances of first point cloud
		D2 (array of shape [num points 2, num points 2]): pairwise distances of second point cloud
		gamma (array of shape [num points 1, num points 2]): probabilistic matching between the two data sets
		computation (int): 1, 2 or 3. Method to use to compute the distortion. 1 is slow but does not require a lot of RAM, 3 is fast but it requires a lot of RAM. 2 is intermediate
		metric (str): metric to use in order to compute the distortion (used only in cases 1. and 3.)
	"""
	if D1 is None:
		if X1 is not None:	
			mode = "1"                               # X1 <---> X2
			n1, n2 = X1.shape[0], X2.shape[0]
		elif X12 is not None:	
			mode = "2"                               # X12
			n1, n2 = X12.shape[0], X12.shape[1]
		else:
			print("Provide at least one input matrix")
			return 0
	else:
		if X1 is not None:
			mode = "3"                               # X1 <---> X2 + D1 <---> D2
			n1, n2 = X1.shape[0], X2.shape[0]
		elif X12 is not None:
			mode = "4"                               # X12 + D1 <---> D2
			n1, n2 = X12.shape[0], X12.shape[1]
		else:
			mode = "5"                               # D1 <---> D2
			n1, n2 = D1.shape[0], D2.shape[0]

	# Slowest, but does not need a lot of RAM
	if computation == "1":
		if mode == "1" or mode == "2":
			X12 = pairwise_distances(X1, X2, metric=metric) if mode == "1" else X12
			scoreX = 0
			for i in range(n1):
				for k in range(n2):
					scoreX += gamma[i,k] * X12[i,k]
			return scoreX

		elif mode == "3" or mode == "4" or mode == "5":
			scoreD = 0
			for i in range(n1):
				for k in range(n2):
					for j in range(n1):
						for l in range(n2):
							scoreD += gamma[i,k] * gamma[j,l] * np.abs(D1[i,j] - D2[k,l])

			if mode == "3" or mode == "4":
				X12 = pairwise_distances(X1, X2, metric=metric) if mode == "3" else X12
				scoreX = 0
				for i in range(n1):
					for k in range(n2):
						scoreX += gamma[i,k] * X12[i,k]
				return [scoreX, scoreD]
			if mode == "5":	return scoreD

	# Intermediate
	elif computation == "2":
		if mode == "1" or mode == "2":
			X12 = pairwise_distances(X1, X2, metric=metric) if mode == "1" else X12
			scoreX = np.sum(np.multiply(gamma, X12))
			return scoreX

		elif mode == "3" or mode == "4" or mode == "5":
			scoreD = 0
			for i in range(n1):
				for k in range(n2):
					D = np.abs(np.expand_dims(D1[i,:], -1) - np.expand_dims(D2[k,:], 0))
					scoreD += gamma[i,k] * np.sum(np.multiply(gamma,D))

			if mode == "3" or mode == "4":
				X12 = pairwise_distances(X1, X2, metric=metric) if mode == "3" else X12
				scoreX = np.sum(np.multiply(gamma, X12))
				return [scoreX, scoreD]
			if mode == "5":	return scoreD

	# Fastest, but needs a lot of RAM
	elif computation == "3":
		if mode == "1" or mode == "2":
			X12 = pairwise_distances(X1, X2, metric=metric) if mode == "1" else X12
			scoreX = np.sum(np.multiply(gamma, X12))
			return scoreX

		elif mode == "3" or mode == "4" or mode == "5":
			G = np.abs( np.expand_dims(np.expand_dims(D1,-1),-1) - np.expand_dims(np.expand_dims(D2,0),0) )
			scoreD = np.sum(np.multiply(gamma, np.tensordot(G, gamma, axes=([1,3],[0,1]))))
			if mode == "3" or mode == "4":
				X12 = pairwise_distances(X1, X2, metric=metric) if mode == "3" else X12
				scoreX = np.sum(np.multiply(gamma, X12))
				return [scoreX, scoreD]
			if mode == "5":	return scoreD


def MREC(X1=None, X2=None, X12=None, D1=None, D2=None, D1quant=None, D2quant=None, return_gamma=False,
         matching=MXS, matching_params={"metric": "euclidean", "alpha": .5, "epsilon": 5e-4, "tol": 1e-9, "max_iter": 1000}, 
         quant_params={"n_centroids": 100, "method": "RandomChoice", "metric": "euclidean"}, threshold=10,
         idxs1=None, idxs2=None, mapping=None, gamma=None, mass=None):
	"""
	This function implements MREC. See the associated paper "MREC: a fast and versatile tool for aligning and matching genomic data". It uses a recursive approximation to compute matchings in a fast way. The function infers the matching algorithm based on which input is None and which isn't. There are five possibilities: 1. X1/X2 not None, X12 None, D1/D2 None; 2. X1/X2 None, X12 not None, D1/D2 None; 3. X1/X2 not None, X12 None, D1/D2 not None; 4. X1/X2 None, X12 not None, D1/D2 not None; 5. X1/X2 None, X12 None, D1/D2 not None; 

	Parameters:
		X1 (array of shape [num points 1, num dimensions]): first point cloud
		X2 (array of shape [num points 2, num dimensions]): second point cloud
                X12 (array of shape [num points 1, num points 2]): precomputed pairwise distances/costs between first and second point cloud
		D1 (array of shape [num points 1, num points 1]): pairwise distances of first point cloud
		D2 (array of shape [num points 2, num points 2]): pairwise distances of second point cloud
		D1quant (array of shape [num points 1, num points 1]): pairwise distances used for quantizing of first point cloud
		D2quant (array of shape [num points 2, num points 2]): pairwise distances used for quantizing of second point cloud
		return_gamma (bool): do you want the full probabilistic matching matrix?
		matching (Python function): function to use for matching the centroids
		matching_params (dict): additional parameters of the matching function
		quant_params (dict): parameters of the quantization method
		threshold (int): number of points used for stopping recursion
		
		the other parameters are only used internally for the recursion

	Returns:
		Gamma (array of shape [num points 1, num points 2]): probabilistic matching between X1 and X2 (returned only if return_gamma is True) 
		matching (array of shape [num points 1]): matching for the points of X1 obtained by rounding Gamma
	"""

	if D1 is None:
		if X1 is not None:	
			mode = "1"                               # X1 <---> X2
			n1, n2 = X1.shape[0], X2.shape[0]
		elif X12 is not None:	
			mode = "2"                               # X12
			n1, n2 = X12.shape[0], X12.shape[1]
		else:
			print("Provide at least one input matrix")
			return 0
	else:
		if X1 is not None:
			mode = "3"                               # X1 <---> X2 + D1 <---> D2
			n1, n2 = X1.shape[0], X2.shape[0]
		elif X12 is not None:
			mode = "4"                               # X12 + D1 <---> D2
			n1, n2 = X12.shape[0], X12.shape[1]
		else:
			mode = "5"                               # D1 <---> D2
			n1, n2 = D1.shape[0], D2.shape[0]

	if idxs1 is None:	idxs1 = np.arange(n1)
	if idxs2 is None:	idxs2 = np.arange(n2)
	if mapping is None:	mapping = np.zeros(n1, dtype=np.int32)
	if return_gamma and gamma is None:	gamma = np.empty([n1,n2])
	if mass is None:	mass = 1/n2

	if n1 <= threshold or n2 <= threshold:
		for idx in idxs1:	mapping[idx] = idxs2[0]

	else:

		n_clus = min(min(n1, n2), quant_params["n_centroids"])
		if type(quant_params) == dict:
			quant_tmp = {k:v for k,v in quant_params.items()}
			quant_tmp["n_centroids"] = n_clus	
			quantize1, quantize2 = Quantization(**quant_tmp), Quantization(**quant_tmp)	
		elif type(quant_params) == list:
			quant_tmp = [{k:v for k,v in quant_params[0].items()}, {k:v for k,v in quant_params[1].items()}]
			quant_tmp[0]["n_centroids"], quant_tmp[1]["n_centroids"] = n_clus, n_clus
			quantize1, quantize2 = Quantization(**quant_tmp[0]), Quantization(**quant_tmp[1])

		quantize1.fit(X1, D1quant)
		quantize2.fit(X2, D2quant)
		indices1, indices2 = quantize1.indices_, quantize2.indices_

		if mode == "1":
			X1_sub, X2_sub = X1[indices1, :], X2[indices2, :]
			prms = matching_params(X1_sub, X2_sub) if type(matching_params) is not dict else matching_params
			gamma_sub, mappings_sub = matching(X1_sub, X2_sub, **prms)
		elif mode == "2":
			X12_sub = X12[indices1, :][:, indices2]
			prms = matching_params(X12_sub) if type(matching_params) is not dict else matching_params
			gamma_sub, mappings_sub = matching(X12_sub, **prms)
		elif mode == "3":
			X1_sub, X2_sub = X1[indices1, :], X2[indices2, :]
			D1_sub, D2_sub = D1[:, indices1][indices1, :], D2[:, indices2][indices2, :]
			prms = matching_params(X1_sub, X2_sub, D1_sub, D2_sub) if type(matching_params) is not dict else matching_params
			gamma_sub, mappings_sub = matching(X1_sub, X2_sub, D1_sub, D2_sub, **prms)
		elif mode == "4":
			X12_sub = X12[indices1, :][:, indices2]
			D1_sub, D2_sub = D1[:, indices1][indices1, :], D2[:, indices2][indices2, :]
			prms = matching_params(X12_sub, D1_sub, D2_sub) if type(matching_params) is not dict else matching_params
			gamma_sub, mappings_sub = matching(X12_sub, D1_sub, D2_sub, **prms)
		elif mode == "5":
			D1_sub, D2_sub = D1[:, indices1][indices1, :], D2[:, indices2][indices2, :]
			prms = matching_params(D1_sub, D2_sub) if type(matching_params) is not dict else matching_params
			gamma_sub, mappings_sub = matching(D1_sub, D2_sub, **prms)

		m12 = mappings_sub[0]
		l1, l2 = np.unique(quantize1.labels_), np.unique(quantize2.labels_) 
		pops1, pops2 = [np.argwhere(quantize1.labels_ == l)[:,0] for l in l1], [np.argwhere(quantize2.labels_ == l)[:,0] for l in l2]

		for i,_ in enumerate(l1):

			I1, I2 = pops1[i], pops2[m12[i]]
			I1i, I2i = idxs1[I1], idxs2[I2]
			massi = mass * gamma_sub[i,m12[i]] / np.sum(gamma_sub[i,:])

			if return_gamma:	tmp = gamma[I1i, :]
			for j,_ in enumerate(l2):
				I2_other = pops2[j]
				if return_gamma:	tmp[:, idxs2[I2_other]] = mass * gamma_sub[i,j] / (len(I2_other) * np.sum(gamma_sub[i,:]))
			if return_gamma:	gamma[I1i, :] = tmp

			if mode == "1":	X1i, X2i, X12i, D1i, D2i = X1[I1,:], X2[I2,:], X12, D1, D2
			if mode == "2":	X1i, X2i, X12i, D1i, D2i = X1, X2, X12[I1,:][:,I2], D1, D2
			if mode == "3":	X1i, X2i, X12i, D1i, D2i = X1[I1,:], X2[I2,:], X12, D1[I1,:][:,I1], D2[I2,:][:,I2]
			if mode == "4":	X1i, X2i, X12i, D1i, D2i = X1, X2, X12[I1,:][:,I2], D1[I1,:][:,I1], D2[I2,:][:,I2]
			if mode == "5":	X1i, X2i, X12i, D1i, D2i = X1, X2, X12, D1[I1,:][:,I1], D2[I2,:][:,I2]

			if (type(quant_params) == dict and quant_params["metric"] == "precomputed") or (type(quant_params) == list and "precomputed" in [qt["metric"] for qt in quant_params]):
				D1quanti, D2quanti = D1quant[I1,:][:,I1], D2quant[I2,:][:,I2]
			else:	D1quanti, D2quanti = D1quant, D2quant

			MREC(X1=X1i, X2=X2i, X12=X12i, D1=D1i, D2=D2i, D1quant=D1quanti, D2quant=D2quanti, 
                             matching=matching, matching_params=matching_params, quant_params=quant_params, threshold=threshold,
                             return_gamma=return_gamma, idxs1=I1i, idxs2=I2i, mapping=mapping, gamma=gamma, mass=massi)

	return gamma, mapping
