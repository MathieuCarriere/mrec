import numpy as np

from sklearn.metrics import pairwise_distances

from ot import unif
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

def SinkhornGromovWasserstein(D1=None, D2=None, prms={}): #epsilon=5e-4, tol=1e-9, max_iter=1000):
	n1, n2 = D1.shape[0], D2.shape[0]
	d1, d2 = unif(n1), unif(n2)
	gamma = entropic_gromov_wasserstein(D1, D2, d1, d2, **prms) #"square_loss", epsilon=epsilon, max_iter=max_iter, tol=tol)
	mappings = [np.argmax(gamma, axis=1), np.argmax(gamma, axis=0)]
	return mappings

def SinkhornWasserstein(X1=None, X2=None, X12=None, prms={}): #metric="euclidean", epsilon=5e-4, tol=1e-09, max_iter=1000):
	pprms = {k:v for k,v in prms.items()}
	if X12 is None:
		n1, n2 = X1.shape[0], X2.shape[0]
		metric = pprms.pop('metric')
	else:	n1, n2 = X12.shape[0], X12.shape[1]
	cost = X12 if X12 is not None else pairwise_distances(X1, X2, metric=metric)
	gamma = sinkhorn(a=(1/n1)*np.ones(n1), b=(1/n2)*np.ones(n2), M=cost, **pprms)
	mappings = [np.argmax(gamma, axis=1), np.argmax(gamma, axis=0)]
	return mappings

def SinkhornUnbalancedWasserstein(X1=None, X2=None, X12=None, prms={}): #metric="euclidean", epsilon=5e-4, delta=5e-4, tol=1e-09, max_iter=1000):
	pprms = {k:v for k,v in prms.items()}
	if X12 is None:
		n1, n2 = X1.shape[0], X2.shape[0]
		metric = pprms.pop('metric')
	else:	n1, n2 = X12.shape[0], X12.shape[1]
	cost = X12 if X12 is not None else pairwise_distances(X1, X2, metric=metric)
	gamma = sinkhorn_unbalanced(a=(1/n1)*np.ones(n1), b=(1/n2)*np.ones(n2), M=cost, **pprms)
	mappings = [np.argmax(gamma, axis=1), np.argmax(gamma, axis=0)]
	return mappings

def SinkhornWassersteinGromovWasserstein(X1=None, X2=None, X12=None, D1=None, D2=None, prms={}): #metric="euclidean", alpha=.5, epsilon=5e-4, tol=1e-9, max_iter=1000):
	pprms = {k:v for k,v in prms.items()}
	if X12 is None:	metric = pprms.pop('metric')
	alpha = pprms.pop('alpha')
	Z = X12 if X12 is not None else pairwise_distances(X1, X2, metric=metric)
	
	n1, n2 = D1.shape[0], D2.shape[0]
	p, q = (1/n1)*np.ones(n1), (1/n2)*np.ones(n2)
	gamma = np.outer(p, q)
	cpt, err = 0, 1.

	if alpha == 1:	gamma = sinkhorn(p, q, Z, **pprms)
	else:        
		while (err > tol and cpt < max_iter):
            
			gamma_prev = gamma

			tens = np.dot(D1, gamma).dot(D2.T)
			tens_all = (1-alpha) * tens + alpha * Z
			gamma = sinkhorn(p, q, tens_all, **pprms)
        
			if cpt % 10 == 0:	err = np.linalg.norm(gamma - gamma_prev)

	mappings = [np.argmax(gamma, axis=1), np.argmax(gamma, axis=0)]
	return mappings

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

def NonConvexGromovHausdorff(D1=None, D2=None, prms={}): #num_iter=15, sigma_m_0=5., mu=10., method="L-BFGS-B", map_init=None, verbose=False):

	num_iter, sigma_m_0, mu, method, map_init, verbose = prms['num_iter'], prms['sigma_m_0'], prms['mu'], prms['method'], prms['map_init'], prms['verbose']

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
	#gamma = np.reshape(minimizers[:,num_iter-1], [n,n])
	#return gamma, mappings
	return mappings

def SDPNALConvexGromovHausdorff(D1=None, D2=None, prms={}): #eng, use_birkhoff=False):

	eng, use_birkhoff = prms['eng'], prms['use_birkhoff']

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
	#gamma = np.reshape(M, [n1, n1])
	#return gamma, mappings
	return mappings

def CPLEXConvexGromovHausdorff(D1=None, D2=None, prms={}): #eng, maxtime=120):

	eng, maxtime = prms['eng'], prms['maxtime']

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
	#gamma = np.reshape(M, [n1, n1])
	#return gamma, mappings
	return mappings

#######################
# Matching parameters #
#######################

def SinkhornWassersteinMedianParameters(X1=None, X2=None, X12=None):
	if X12 is not None:
		reg = np.quantile(X12,.5)
		reg = reg if reg > 0 else 1.
		return {'metric': 'euclidean', 'reg': reg, 'max_iter': 1000, "tol": 1e-9}
	else:
		reg = np.quantile(pairwise_distances(X1, X2, metric='euclidean'),.5)
		reg = reg if reg > 0 else 1.
		return {'metric': 'euclidean', 'reg': reg, 'max_iter': 1000, "tol": 1e-9}

def SinkhornGromovWassersteinMedianParameters(D1=None, D2=None):
	epsilon = np.quantile(np.abs(D1[:40,:40,None,None]-D2[None,None,:40,:40]),.5)
	epsilon = epsilon if epsilon > 0 else 1.
	return {'epsilon': np.quantile(np.abs(D1[:40,:40,None,None]-D2[None,None,:40,:40]),.5), 'max_iter': 1000, "tol": 1e-9, "loss_fun": 'square_loss'}
