import numpy as np
import scipy as sp

# MCDVS sampler
# input:
#   X: numpy 2d array, data matrix of size nxm
#   mix_step: number of mixing steps for Markov chain
#   k: size of sampled subset
#   beta: exponent in determinant: P(S) \propto det(X_S*X_S^T)^\beta
#   flag_gpu: use gpu acceleration

def sample(X, mix_step, k, beta=1, init_rst=None):
    n, m = X.shape
    rst = init_rst
    tic_len = mix_step // 5

    # k-dpp
    if rst is None:
        rst = rst = np.random.permutation(m)[:k]
    rst_bar = np.setdiff1d(range(m), rst)

    epseye = 0.001 * np.identity(n)

    C_mat = X[np.ix_(range(X.shape[0]), rst)]
    M = C_mat.dot(C_mat.transpose()) + epseye
    M_det = sp.linalg.det(M)**beta

    for i in xrange(mix_step):
        if (i+1) % tic_len == 0:
            print('{}-th iteration.'.format(i+1))
        rem_ind = np.random.randint(k)
        add_ind = np.random.randint(m-k)
        v = rst[rem_ind]
        u = rst_bar[add_ind]

        rst_new = np.copy(rst)
        rst_new[rem_ind] = u
        C_mat_new = X[np.ix_(range(X.shape[0]), rst_new)]
        update_det = sp.linalg.det(C_mat_new.dot(C_mat_new.transpose()) + epseye)**beta

        flag = np.random.uniform() < (update_det / M_det)

        if flag:
            M_det = update_det
            rst[rem_ind] = u
            rst_bar[add_ind] = v

    return rst




