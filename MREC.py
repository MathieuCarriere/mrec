import numpy as np
from sklearn.metrics import pairwise_distances
from quantization import *
from matching import *

def distortion_score(X1=None, X2=None, X12=None, D1=None, D2=None, mapping=None, computation_mode="1", metric="euclidean"):
	"""
	Compute the metric distortion associated to a matching between two datasets. The function infers the distortion algorithm based on which input is None and which isn't. There are five possibilities: 
    ---1. X1/X2 not None, X12 None,     D1/D2 None;     Usual Wasserstein
    ---2. X1/X2 None,     X12 not None, D1/D2 None;     Wasserstein with precomputed metric
    ---3. X1/X2 not None, X12 None,     D1/D2 not None; Mixture of usual Wasserstein and usual Gromov-Wasserstein
    ---4. X1/X2 None,     X12 not None, D1/D2 not None; Mixture of Wasserstein with precomputed metric and usual Gromov-Wasserstein
    ---5. X1/X2 None,     X12 None,     D1/D2 not None; Usual Gromov-Wasserstein

	Parameters:
		X1 (array of shape [num points 1, num dimensions]): first point cloud
		X2 (array of shape [num points 2, num dimensions]): second point cloud
        X12 (array of shape [num points 1, num points 2]): pairwise distances between first and second point cloud
		D1 (array of shape [num points 1, num points 1]): pairwise distances of first point cloud
		D2 (array of shape [num points 2, num points 2]): pairwise distances of second point cloud
		computation_mode (int): 1, 2 or 3. Method to use to compute the distortion. 1 is slow but does not require a lot of RAM, 3 is fast but it requires a lot of RAM. 2 is intermediate
		metric (str): metric to use in order to compute the distortion (used only in cases 1. and 3.)
		gamma (array of shape [num points 1, num points 2]): probabilistic matching between the two data sets
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
	if computation_mode == "1":

		if mode == "1" or mode == "2":
			X12 = pairwise_distances(X1, X2, metric=metric) if mode == "1" else X12
			scoreX = 0
			for i in range(n1):
				scoreX += X12[i, mapping[i]]
			return scoreX

		elif mode == "3" or mode == "4" or mode == "5":
			scoreD = 0
			for i in range(n1):
				for j in range(n1):
					scoreD += np.abs(D1[i,j] - D2[mapping[i], mapping[j]])

			if mode == "3" or mode == "4":
				X12 = pairwise_distances(X1, X2, metric=metric) if mode == "3" else X12
				scoreX = 0
				for i in range(n1):
					scoreX += X12[i, mapping[i]]
				return [scoreX, scoreD]

			if mode == "5":	return scoreD

	# Intermediate
	elif computation_mode == "2":

		if mode == "1" or mode == "2":
			X12 = pairwise_distances(X1, X2, metric=metric) if mode == "1" else X12
			scoreX = np.sum(X12[:,mapping])
			return scoreX

		elif mode == "3" or mode == "4" or mode == "5":
			scoreD = 0
			for i in range(n1):
				scoreD += np.sum(np.abs(D1[i,:]-D2[mapping[i],mapping]))

			if mode == "3" or mode == "4":
				X12 = pairwise_distances(X1, X2, metric=metric) if mode == "3" else X12
				scoreX = np.sum(X12[:,mapping])
				return [scoreX, scoreD]

			if mode == "5":	return scoreD

	# Fastest, but needs a lot of RAM
	elif computation_mode == "3":

		if mode == "1" or mode == "2":
			X12 = pairwise_distances(X1, X2, metric=metric) if mode == "1" else X12
			scoreX = np.sum(X12[:,mapping])
			return scoreX

		elif mode == "3" or mode == "4" or mode == "5":
			scoreD = np.sum(D1 - D2[mapping,:][:,mapping])

			if mode == "3" or mode == "4":
				X12 = pairwise_distances(X1, X2, metric=metric) if mode == "3" else X12
				scoreX = np.sum(X12[:,mapping])
				return [scoreX, scoreD]

			if mode == "5":	return scoreD


def MREC(X1=None, X2=None, X12=None, D1=None, D2=None, D1quant=None, D2quant=None,
         matching=SinkhornWasserstein, matching_params=SinkhornWassersteinMedianParameters,
         quantization1=RandomChoiceQuantization, quantization_params1={"n_centroids": 100, "metric": "euclidean"}, 
         quantization2=RandomChoiceQuantization, quantization_params2={"n_centroids": 100, "metric": "euclidean"}, 
         threshold=10, last_matching='constant', impose_equal=False,
         mode=None, idxs1=None, idxs2=None,  mapping12=None):
	"""
	This function implements MREC. See the associated paper "MREC: a fast and versatile tool for aligning and matching genomic data". It uses a recursive approximation to compute matchings in a fast way. The function infers the matching algorithm based on which input is None and which isn't. There are five possibilities: 
    ---1. X1/X2 not None, X12 None,     D1/D2 None;     Usual Wasserstein
    ---2. X1/X2 None,     X12 not None, D1/D2 None;     Wasserstein with precomputed metric
    ---3. X1/X2 not None, X12 None,     D1/D2 not None; Mixture of usual Wasserstein and usual Gromov-Wasserstein
    ---4. X1/X2 None,     X12 not None, D1/D2 not None; Mixture of Wasserstein with precomputed metric and usual Gromov-Wasserstein
    ---5. X1/X2 None,     X12 None,     D1/D2 not None; Usual Gromov-Wasserstein

	Parameters:
		X1 (array of shape [num points 1, num dimensions]): first point cloud
		X2 (array of shape [num points 2, num dimensions]): second point cloud
        X12 (array of shape [num points 1, num points 2]): precomputed pairwise distances/costs between first and second point cloud
		D1 (array of shape [num points 1, num points 1]): pairwise distances of first point cloud
		D2 (array of shape [num points 2, num points 2]): pairwise distances of second point cloud
		D1quant (array of shape [num points 1, num points 1]): pairwise distances used for quantization of first point cloud
		D2quant (array of shape [num points 2, num points 2]): pairwise distances used for quantization of second point cloud
		matching (Python function): function to use for matching the centroids
		matching_params (dict): additional parameters of the matching function
        quantization1 (Python function): function to use for quantizing the first point cloud
		quantization_params1 (dict): parameters of the quantization method for the first point cloud
		quantization2 (Python function): function to use for quantizing the second point cloud
		quantization_params2 (dict): parameters of the quantization method for the second point cloud
		threshold (int): number of points used for stopping recursion
        last_matching (str): whether to use the matching black box ('match') or a constant matching ('constant') at the last recursion step.
		impose_equal (bool): whether to force preimages to have exactly the same cardinality
		mode (str): matching mode, either '1', '2', '3', '4' or '5'. See the five types of matching distances above 

		The other parameters are only used internally for the recursion.

	Returns:
		Gamma (array of shape [num points 1, num points 2]): probabilistic matching between X1 and X2 (returned only if return_gamma is True) 
		matching (array of shape [num points 1]): matching for the points of X1 obtained by rounding Gamma
	"""

	if mode is None:
		if D1 is None:
			if X1 is not None:	
				mode = "1"                               # X1 <---> X2
			elif X12 is not None:	
				mode = "2"                               # X12
			else:
				print("Provide at least one input matrix")
				return 0
		else:
			if X1 is not None:
				mode = "3"                               # X1 <---> X2 + D1 <---> D2
			elif X12 is not None:
				mode = "4"                               # X12 + D1 <---> D2
			else:
				mode = "5"                               # D1 <---> D2
		
	if X1 is not None:	
		n1, n2 = X1.shape[0], X2.shape[0]
	elif X12 is not None:	
		n1, n2 = X12.shape[0], X12.shape[1]
	else:
		n1, n2 = D1.shape[0], D2.shape[0]

	if idxs1 is None:	idxs1 = np.arange(n1)
	if idxs2 is None:	idxs2 = np.arange(n2)
	if mapping12 is None:	mapping12 = np.zeros(n1, dtype=np.int32)
	
	if n1 <= threshold or n2 <= threshold:

		if last_matching == 'constant': 
			mapping12[idxs1] = idxs2[0]

		elif last_matching == 'match':

			if mode == "1" or mode == "2":
				prms = matching_params(X1, X2, X12) if type(matching_params) is not dict else matching_params
				mappings_sub = matching(X1, X2, X12, prms)

			elif mode == "3" or mode == "4":
				prms = matching_params(X1, X2, X12, D1, D2) if type(matching_params) is not dict else matching_params
				mappings_sub = matching(X1, X2, X12, D1, D2, prms)

			elif mode == "5":
				prms = matching_params(D1, D2) if type(matching_params) is not dict else matching_params
				mappings_sub = matching(D1, D2, prms)

			m12 = mappings_sub[0]
			mapping12[idxs1] = idxs2[m12]

	else:

		prms1 = quantization_params1(X1, D1quant) if type(quantization_params1) is not dict else {k:v for (k,v) in quantization_params1.items()}
		prms2 = quantization_params2(X2, D2quant) if type(quantization_params2) is not dict else {k:v for (k,v) in quantization_params2.items()}

		if impose_equal:
			nclus = min(prms1['n_clusters'], prms2['n_clusters'])
			prms1['n_clusters'], prms2['n_clusters'] = nclus, nclus
					
		indices1, labels1 = quantization1(X1, D1quant, prms1)
		indices2, labels2 = quantization2(X2, D2quant, prms2)

		if mode == "1":
			X1_sub, X2_sub = X1[indices1, :], X2[indices2, :]
			prms = matching_params(X1_sub, X2_sub, X12) if type(matching_params) is not dict else matching_params
			mappings_sub = matching(X1_sub, X2_sub, X12, prms)

		elif mode == "2":
			X12_sub = X12[indices1, :][:, indices2]
			prms = matching_params(X1, X2, X12_sub) if type(matching_params) is not dict else matching_params
			mappings_sub = matching(X1, X2, X12_sub, prms)

		elif mode == "3":
			X1_sub, X2_sub = X1[indices1, :], X2[indices2, :]
			D1_sub, D2_sub = D1[:, indices1][indices1, :], D2[:, indices2][indices2, :]
			prms = matching_params(X1_sub, X2_sub, X12, D1_sub, D2_sub) if type(matching_params) is not dict else matching_params
			mappings_sub = matching(X1_sub, X2_sub, X12, D1_sub, D2_sub, prms)

		elif mode == "4":
			X12_sub = X12[indices1, :][:, indices2]
			D1_sub, D2_sub = D1[:, indices1][indices1, :], D2[:, indices2][indices2, :]
			prms = matching_params(X1, X2, X12_sub, D1_sub, D2_sub) if type(matching_params) is not dict else matching_params
			mappings_sub = matching(X1, X2, X12_sub, D1_sub, D2_sub, prms)

		elif mode == "5":
			D1_sub, D2_sub = D1[:, indices1][indices1, :], D2[:, indices2][indices2, :]
			prms = matching_params(D1_sub, D2_sub) if type(matching_params) is not dict else matching_params
			mappings_sub = matching(D1_sub, D2_sub, prms)

		m12 = mappings_sub[0]
		l1, l2 = np.unique(labels1), np.unique(labels2) 
		pops1, pops2 = [np.argwhere(labels1 == l)[:,0] for l in l1], [np.argwhere(labels2 == l)[:,0] for l in l2]

		for i,_ in enumerate(l1):

			I1, I2 = pops1[i], pops2[m12[i]]
			I1i, I2i = idxs1[I1], idxs2[I2]

			X1i = X1[I1,:] if X1 is not None else X1
			X2i = X2[I2,:] if X1 is not None else X2
			X12i = X12[I1,:][:,I2] if X12 is not None else X12
			D1i = D1[I1,:][:,I1] if D1 is not None else D1
			D2i = D2[I2,:][:,I2] if D2 is not None else D2

			if D1quant is not None and D2quant is not None:
				D1quanti, D2quanti = D1quant[I1,:][:,I1], D2quant[I2,:][:,I2]
			else:	D1quanti, D2quanti = D1quant, D2quant

			MREC(X1=X1i, X2=X2i, X12=X12i, D1=D1i, D2=D2i, D1quant=D1quanti, D2quant=D2quanti, 
                 matching=matching, matching_params=matching_params, 
                 quantization1=quantization1, quantization_params1=quantization_params1, 
                 quantization2=quantization2, quantization_params2=quantization_params2,
                 threshold=threshold, last_matching=last_matching, impose_equal=impose_equal,
                 idxs1=I1i, idxs2=I2i, mapping12=mapping12, mode=mode)

	return mapping12
