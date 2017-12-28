import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import pinv, norm

import sampler.mcdvs as dvs
import sampler.utils as utils

# currently only support cpu mode
flag_gpu = False

data = np.loadtxt('data/ailerons.txt')[:5000]
X = data[:,:-1]
Y = data[:,-1]

k_group = [40,60,80,100,150]
error_unif = np.zeros(len(k_group))
error_dvs = np.zeros(len(k_group))

beta_ref = pinv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)

for run_id in xrange(5):
	for k_idx in xrange(len(k_group)):
		k = k_group[k_idx]
		# Uniform sampling
		unif_smpl = np.random.permutation(5000)[:k]

		X_hat = X[np.ix_(unif_smpl, range(X.shape[1]))]
		Y_hat = Y[np.ix_(unif_smpl)]
		beta_hat = pinv(X_hat.transpose().dot(X_hat)).dot(X_hat.transpose()).dot(Y_hat)
		error_unif[k_idx] += norm(beta_hat - beta_ref)

		# DVS sampling
		dvs_init = utils.kpp(X, k, flag_kernel=False)
		dvs_smpl  = dvs.sample(X.transpose(), 10000, k, init_rst=dvs_init)

		X_hat = X[np.ix_(dvs_smpl, range(X.shape[1]))]
		Y_hat = Y[np.ix_(dvs_smpl)]
		beta_hat = pinv(X_hat.transpose().dot(X_hat)).dot(X_hat.transpose()).dot(Y_hat)
		error_dvs[k_idx] += norm(beta_hat - beta_ref)


plt.figure(figsize=(4,4))
plt.title('Approximation Error')
plt.plot(k_group, error_unif / 5., label='unif', lw=2)
plt.plot(k_group, error_dvs / 5., label='dvs', lw=2)
plt.legend()

plt.savefig('fig/regression', bbox_inches='tight')



