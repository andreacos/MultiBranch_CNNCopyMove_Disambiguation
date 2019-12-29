'''
Copyright is preserved to Quoc-Tin Phan (dimmoon2511[at]gmail.com)
'''

import sys
import numpy as np
from sklearn.cluster import KMeans
import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def laplacian(W):
	"""Computes the symetric normalized laplacian.
	L = D^{-1/2} W D{-1/2}
	"""
	D = np.zeros(W.shape)
	I = np.eye(W.shape[0])
	w = np.sum(W, axis=0)
	D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
	return I - D.dot(W).dot(D)

def spectral_clustering(S):
	L = laplacian(S)
	eig_values, eig_vecs = np.linalg.eig(L)
	idx = eig_values.argsort()[::-1]
	eig_values = eig_values[idx]
	eig_vecs = eig_vecs[:,idx]
	n = S.shape[0]
	# use eigengap to estimate number of clusters
	k = n - 1 - np.argmax(np.abs(eig_values[:-2] - np.abs(eig_values[1:-1])))
	features = eig_vecs[:,n-k:]
	features,_ = norm_normalize(features)
	kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
	return k,kmeans.labels_

def norm_normalize(X):
	'''
	X : x_samples x n_features
	'''
	norm = np.linalg.norm(X, axis=1, keepdims=True)
	return X / norm, norm

def std_normalize(X):
	'''
	X : x_samples x n_features
	'''
	std = X.std(axis=0, keepdims=True)
	return X / std, std

def mean_normalize(X):
	'''
	X: x_samples x n_features
	'''
	mean = X.mean(axis=0, keepdims=True)
	return X - mean, mean

def mean_std_normalize(X):
	'''
	X: x_samples x n_features
	'''
	mean = X.mean(axis=0, keepdims=True)
	std = X.std(axis=0, keepdims=True)
	return (X - mean)/std, mean, std

def is_pos_def(X):
    return np.all(np.linalg.eigvals(X) > 0)

def transform(coords, trans):
	'''
	coords: n_points x (2 or 3)
	T: 3 x 3
	'''
	if coords.shape[1]==2:
		coords_,mu = mean_normalize(coords) 
	else: 
		coords_,mu = mean_normalize(coords[:2,:]) 
	coords_h = np.column_stack((coords_,np.ones(coords_.shape[0])))
	coords_h_t = trans.dot(coords_h.T)
	return coords_h_t[:2,:].T + mu

def bb_mean(coords):
	'''
	coords: n_points x 2
	'''
	min_x,max_x = (coords[:,0].min(),coords[:,0].max())
	min_y,max_y = (coords[:,1].min(),coords[:,1].max())
	return (max_x + min_x)/2,(max_y + min_y)/2

def path_match(files, basename):
    '''
    We don't know the ext of image, so we have to find it by basename
    '''
    for f in files:
        b = os.path.basename(f)
        ext = b[b.find('.'):]
        if b.replace(ext,'') == basename:
            return f
    return None

def compute_affinity(X):
    N = X.shape[0]
    ans = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            ans[i][j] = squared_exponential(X[i], X[j])
    return ans

def squared_exponential(x, y, sig=0.8, sig2=1):
    norm = np.linalg.norm(x - y)
    dist = norm * norm
    return np.exp(- dist / (2 * sig * sig2))

def is_rigid(T):
    return np.sum(np.abs(T[:2,:2]) - np.eye(2)) == 0.

def sigmoid(x):
    return 1/(1+np.exp(-x))